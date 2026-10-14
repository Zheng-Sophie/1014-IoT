import io
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from itertools import combinations
from flask import Flask, render_template

from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)

DARK_BG  = "#0f1117"
CARD_BG  = "#1a1d2e"
TEXT     = "#e2e8f0"
GRID     = "#2a2d3e"
ACCENT   = "#7c6af7"
ACCENT2  = "#56cfb2"
PALETTE  = ["#7c6af7", "#56cfb2", "#f7936a", "#f7c56a", "#f76a8a"]


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def run_pipeline():
    # ── Data ──────────────────────────────────────────────────────
    df = pd.read_csv("50_Startups.csv")

    eda = {
        "shape": df.shape,
        "head": df.head().to_html(classes="data-table", border=0, index=True),
        "missing": df.isnull().sum().to_frame("Missing").to_html(classes="data-table", border=0),
        "duplicates": int(df.duplicated().sum()),
        "describe": df.describe(include="all").round(2).to_html(classes="data-table", border=0),
        "state_dist": df["State"].value_counts().to_dict(),
    }

    # ── Prep ──────────────────────────────────────────────────────
    df = df.drop_duplicates().dropna(subset=["Profit"])
    FEATURES = ["R&D Spend", "Administration", "Marketing Spend", "State"]
    NUMERIC  = FEATURES[:3]

    X, y = df[FEATURES], df["Profit"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    split_info = {"train": len(X_train), "test": len(X_test)}

    # ── Helpers ───────────────────────────────────────────────────
    def build_preprocessor(sel):
        numeric = [f for f in sel if f != "State"]
        transformers = []
        if numeric:
            transformers.append(("num", Pipeline([
                ("imp", SimpleImputer(strategy="mean")),
                ("sc",  StandardScaler())]), numeric))
        if "State" in sel:
            transformers.append(("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("enc", OneHotEncoder(
                    categories=[["California", "Florida", "New York"]],
                    drop="first", handle_unknown="ignore", sparse_output=False))
            ]), ["State"]))
        return ColumnTransformer(transformers)

    def build_model(sel):
        return Pipeline([("pre", build_preprocessor(sel)),
                         ("reg", LinearRegression())])

    def eval_subset(sel):
        """Returns (PMSE, R2) via cross_val_predict on training data."""
        model = build_model(sel)
        pred  = cross_val_predict(model, X_train[sel], y_train, cv=cv)
        pmse  = float(mean_squared_error(y_train, pred))
        r2    = float(r2_score(y_train, pred))
        return pmse, r2

    # ── Per-k selection functions ──────────────────────────────────

    def best_exhaustive(k):
        best_score, best_sub = float("inf"), None
        for sub in combinations(FEATURES, k):
            sub = list(sub)
            pmse, _ = eval_subset(sub)
            if pmse < best_score:
                best_score, best_sub = pmse, sub
        return best_sub

    def best_rfe(k):
        cands = []
        for inc_state in [False, True]:
            n_num = k - int(inc_state)
            if n_num == 0 and inc_state:
                cands.append(eval_subset(["State"]) + (["State"],))
                continue
            if not (1 <= n_num <= 3):
                continue
            num_pipe = Pipeline([
                ("imp", SimpleImputer(strategy="mean")),
                ("sc",  StandardScaler())])
            X_num = num_pipe.fit_transform(X_train[NUMERIC])
            rfe = RFE(LinearRegression(), n_features_to_select=n_num)
            rfe.fit(X_num, y_train)
            sel = [f for f, keep in zip(NUMERIC, rfe.support_) if keep]
            if inc_state:
                sel = sel + ["State"]
            cands.append(eval_subset(sel) + (sel,))
        best = min(cands, key=lambda x: x[0])
        return best[2]

    def best_sfs(k):
        selected, remaining = [], FEATURES.copy()
        while len(selected) < k:
            cands = []
            for feat in remaining:
                sub = selected + [feat]
                pmse, _ = eval_subset(sub)
                cands.append((pmse, feat))
            _, best_feat = min(cands, key=lambda x: x[0])
            selected.append(best_feat)
            remaining.remove(best_feat)
        return selected

    def best_lasso(k):
        pre = build_preprocessor(FEATURES)
        X_proc = pre.fit_transform(X_train)
        lasso = LassoCV(cv=cv, random_state=42, max_iter=100000)
        lasso.fit(X_proc, y_train)
        names = pre.get_feature_names_out()
        importance = {f: 0.0 for f in FEATURES}
        for name, coef in zip(names, lasso.coef_):
            orig = name.replace("num__", "") if name.startswith("num__") else "State"
            importance[orig] = max(importance[orig], abs(coef))
        ranked = sorted(FEATURES, key=lambda f: importance[f], reverse=True)
        return ranked[:k]

    def best_sbs(k):
        selected = FEATURES.copy()
        while len(selected) > k:
            cands = []
            for feat in selected:
                sub = [f for f in selected if f != feat]
                pmse, _ = eval_subset(sub)
                cands.append((pmse, sub))
            selected = min(cands, key=lambda x: x[0])[1]
        return selected

    SCHEMES = {
        "Exhaustive": best_exhaustive,
        "RFE":        best_rfe,
        "SFS":        best_sfs,
        "Lasso":      best_lasso,
        "SBS":        best_sbs,
    }

    # ── Compute per-k results for all schemes ──────────────────────
    per_k_results = {name: {"k": [], "PMSE": [], "R2": [], "features": []}
                     for name in SCHEMES}

    for name, selector in SCHEMES.items():
        print(f"  Running {name}…")
        for k in range(1, len(FEATURES) + 1):
            sel    = selector(k)
            pmse, r2 = eval_subset(sel)
            per_k_results[name]["k"].append(k)
            per_k_results[name]["PMSE"].append(pmse)
            per_k_results[name]["R2"].append(r2)
            per_k_results[name]["features"].append(sel)

    # ── Best overall (minimum PMSE) ────────────────────────────────
    best_scheme = None
    best_pmse   = float("inf")
    best_features = None
    for name, data in per_k_results.items():
        idx = int(np.argmin(data["PMSE"]))
        if data["PMSE"][idx] < best_pmse:
            best_pmse   = data["PMSE"][idx]
            best_scheme = name
            best_features = data["features"][idx]

    # Build comparison table (best subset per scheme)
    rows = []
    for name, data in per_k_results.items():
        idx = int(np.argmin(data["PMSE"]))
        rows.append({
            "Scheme":   name,
            "Features": ", ".join(data["features"][idx]),
            "CV_PMSE":  round(data["PMSE"][idx], 2),
            "CV_R2":    round(data["R2"][idx], 4),
        })
    comp_df = pd.DataFrame(rows).sort_values("CV_PMSE").reset_index(drop=True)

    # ── Final model ────────────────────────────────────────────────
    final_model = build_model(best_features)
    final_model.fit(X_train[best_features], y_train)
    y_pred = final_model.predict(X_test[best_features])

    metrics = {
        "mae":  round(float(mean_absolute_error(y_test, y_pred)), 2),
        "mse":  round(float(mean_squared_error(y_test, y_pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2),
        "r2":   round(float(r2_score(y_test, y_pred)), 4),
    }

    pred_df = pd.DataFrame({
        "Actual Profit":    y_test.values.round(2),
        "Predicted Profit": y_pred.round(2),
        "Residual":         (y_test.values - y_pred).round(2),
    })

    new_startup = pd.DataFrame({
        "R&D Spend": [150000], "Administration": [120000],
        "Marketing Spend": [250000], "State": ["California"]
    })
    sample_pred = float(final_model.predict(new_startup[best_features])[0])

    # ── Chart helpers ──────────────────────────────────────────────
    def styled_ax(ax, title):
        ax.set_facecolor(CARD_BG)
        ax.set_title(title, color=TEXT, pad=12, fontsize=12, fontweight="600")
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.spines[:].set_visible(False)
        ax.grid(True, color=GRID, alpha=0.6, zorder=0)
        ax.set_axisbelow(True)

    # ── Line chart 1: Num Features vs PMSE ────────────────────────
    fig1, ax1 = plt.subplots(figsize=(9, 4.5))
    fig1.patch.set_facecolor(DARK_BG)
    styled_ax(ax1, "Number of Features vs PMSE")
    for i, (name, data) in enumerate(per_k_results.items()):
        ax1.plot(data["k"], data["PMSE"],
                 marker="o", linewidth=2.2, markersize=7,
                 color=PALETTE[i], label=name, zorder=3)
    ax1.set_xlabel("Number of Selected Features", color=TEXT, labelpad=8)
    ax1.set_ylabel("PMSE", color=TEXT, labelpad=8)
    ax1.set_xticks(range(1, len(FEATURES) + 1))
    ax1.legend(facecolor=CARD_BG, labelcolor=TEXT, edgecolor=GRID,
               fontsize=8, loc="upper right")
    chart_pmse = fig_to_b64(fig1)

    # ── Line chart 2: Num Features vs R² ──────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
    fig2.patch.set_facecolor(DARK_BG)
    styled_ax(ax2, "Number of Features vs R-squared")
    for i, (name, data) in enumerate(per_k_results.items()):
        ax2.plot(data["k"], data["R2"],
                 marker="o", linewidth=2.2, markersize=7,
                 color=PALETTE[i], label=name, zorder=3)
    ax2.set_xlabel("Number of Selected Features", color=TEXT, labelpad=8)
    ax2.set_ylabel("R-squared", color=TEXT, labelpad=8)
    ax2.set_xticks(range(1, len(FEATURES) + 1))
    ax2.legend(facecolor=CARD_BG, labelcolor=TEXT, edgecolor=GRID,
               fontsize=8, loc="lower right")
    chart_r2 = fig_to_b64(fig2)

    # ── Scatter: actual vs predicted ───────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(6, 5.5))
    fig3.patch.set_facecolor(DARK_BG)
    styled_ax(ax3, f"Actual vs Predicted ({best_scheme})")
    ax3.scatter(y_test, y_pred, color=ACCENT, alpha=0.85,
                edgecolors=ACCENT2, linewidths=0.8, s=80, zorder=3)
    lo = min(float(y_test.min()), float(y_pred.min())) * 0.97
    hi = max(float(y_test.max()), float(y_pred.max())) * 1.03
    ax3.plot([lo, hi], [lo, hi], "--", color=ACCENT2, linewidth=1.5,
             label="Perfect Prediction")
    ax3.set_xlabel("Actual Profit ($)", color=TEXT, labelpad=8)
    ax3.set_ylabel("Predicted Profit ($)", color=TEXT, labelpad=8)
    ax3.legend(facecolor=CARD_BG, labelcolor=TEXT, edgecolor=GRID)
    chart_scatter = fig_to_b64(fig3)

    # ── Scheme bar chart ───────────────────────────────────────────
    fig4, ax4 = plt.subplots(figsize=(9, 4))
    fig4.patch.set_facecolor(DARK_BG)
    styled_ax(ax4, "Best CV RMSE by Feature Selection Scheme")
    rmse_vals = comp_df["CV_PMSE"].apply(np.sqrt).values
    bars = ax4.bar(comp_df["Scheme"], rmse_vals,
                   color=PALETTE[:len(comp_df)], edgecolor="none", zorder=3)
    for bar in bars:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2, h + 30,
                 f"{h:,.0f}", ha="center", va="bottom",
                 color=TEXT, fontsize=9, fontweight="bold")
    ax4.set_xlabel("Scheme", color=TEXT, labelpad=8)
    ax4.set_ylabel("CV RMSE", color=TEXT, labelpad=8)
    ax4.yaxis.grid(True, color=GRID, zorder=0)
    chart_scheme = fig_to_b64(fig4)

    return {
        "eda": eda,
        "split": split_info,
        "comp_table": comp_df.to_html(classes="data-table", border=0, index=False),
        "best_scheme": best_scheme,
        "best_features": best_features,
        "metrics": metrics,
        "pred_table": pred_df.to_html(classes="data-table", border=0, index=False),
        "sample_pred": sample_pred,
        "chart_pmse":    chart_pmse,
        "chart_r2":      chart_r2,
        "chart_scatter": chart_scatter,
        "chart_scheme":  chart_scheme,
    }


print("Running ML pipeline…")
DATA = run_pipeline()
print("Pipeline complete. Starting Flask…")


@app.route("/")
def index():
    return render_template("index.html", **DATA)


if __name__ == "__main__":
    app.run(debug=False)

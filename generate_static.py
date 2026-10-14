"""
generate_static.py
Runs the full ML pipeline and writes docs/index.html as a self-contained
static page suited for GitHub Pages (no Flask, no server).
"""

import io, base64, os, html as html_lib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from itertools import combinations
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Style constants ────────────────────────────────────────────────────────
DARK_BG = "#0f1117"
CARD_BG = "#1a1d2e"
TEXT    = "#e2e8f0"
GRID    = "#2a2d3e"
ACCENT  = "#7c6af7"
ACCENT2 = "#56cfb2"
PALETTE = ["#7c6af7", "#56cfb2", "#f7936a", "#f7c56a", "#f76a8a"]


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def styled_ax(ax, title):
    ax.set_facecolor(CARD_BG)
    ax.set_title(title, color=TEXT, pad=12, fontsize=12, fontweight="bold")
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.spines[:].set_visible(False)
    ax.grid(True, color=GRID, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)


# ── Pipeline ───────────────────────────────────────────────────────────────
print("Loading data…")
df = pd.read_csv("50_Startups.csv")

shape      = df.shape
state_dist = df["State"].value_counts().to_dict()
missing    = df.isnull().sum().to_dict()
duplicates = int(df.duplicated().sum())
head_html  = df.head().to_html(classes="dt", border=0, index=True)
desc_html  = df.describe(include="all").round(2).to_html(classes="dt", border=0)

df = df.drop_duplicates().dropna(subset=["Profit"])

FEATURES = ["R&D Spend", "Administration", "Marketing Spend", "State"]
NUMERIC  = FEATURES[:3]
X, y = df[FEATURES], df["Profit"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)


def build_preprocessor(sel):
    numeric = [f for f in sel if f != "State"]
    t = []
    if numeric:
        t.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="mean")),
            ("sc",  StandardScaler())]), numeric))
    if "State" in sel:
        t.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OneHotEncoder(
                categories=[["California", "Florida", "New York"]],
                drop="first", handle_unknown="ignore", sparse_output=False))
        ]), ["State"]))
    return ColumnTransformer(t)


def build_model(sel):
    return Pipeline([("pre", build_preprocessor(sel)),
                     ("reg", LinearRegression())])


def eval_subset(sel):
    model = build_model(sel)
    pred  = cross_val_predict(model, X_train[sel], y_train, cv=cv)
    return float(mean_squared_error(y_train, pred)), float(r2_score(y_train, pred))


def best_exhaustive(k):
    best_s, best_sub = float("inf"), None
    for sub in combinations(FEATURES, k):
        sub = list(sub)
        s, _ = eval_subset(sub)
        if s < best_s:
            best_s, best_sub = s, sub
    return best_sub


def best_rfe(k):
    cands = []
    for inc in [False, True]:
        n = k - int(inc)
        if n == 0 and inc:
            cands.append(eval_subset(["State"]) + (["State"],))
            continue
        if not (1 <= n <= 3):
            continue
        pipe = Pipeline([("imp", SimpleImputer(strategy="mean")),
                         ("sc",  StandardScaler())])
        Xn = pipe.fit_transform(X_train[NUMERIC])
        rfe = RFE(LinearRegression(), n_features_to_select=n)
        rfe.fit(Xn, y_train)
        sel = [f for f, keep in zip(NUMERIC, rfe.support_) if keep]
        if inc:
            sel = sel + ["State"]
        cands.append(eval_subset(sel) + (sel,))
    best = min(cands, key=lambda x: x[0])
    return best[2]


def best_sfs(k):
    sel, rem = [], FEATURES.copy()
    while len(sel) < k:
        cands = []
        for f in rem:
            s, _ = eval_subset(sel + [f])
            cands.append((s, f))
        _, bf = min(cands, key=lambda x: x[0])
        sel.append(bf); rem.remove(bf)
    return sel


def best_lasso(k):
    pre = build_preprocessor(FEATURES)
    Xp  = pre.fit_transform(X_train)
    lasso = LassoCV(cv=cv, random_state=42, max_iter=100000)
    lasso.fit(Xp, y_train)
    names = pre.get_feature_names_out()
    imp = {f: 0.0 for f in FEATURES}
    for n, c in zip(names, lasso.coef_):
        orig = n.replace("num__", "") if n.startswith("num__") else "State"
        imp[orig] = max(imp[orig], abs(c))
    return sorted(FEATURES, key=lambda f: imp[f], reverse=True)[:k]


def best_sbs(k):
    sel = FEATURES.copy()
    while len(sel) > k:
        cands = []
        for f in sel:
            sub = [x for x in sel if x != f]
            s, _ = eval_subset(sub)
            cands.append((s, sub))
        sel = min(cands, key=lambda x: x[0])[1]
    return sel


SCHEMES = {"Exhaustive": best_exhaustive, "RFE": best_rfe,
           "SFS": best_sfs, "Lasso": best_lasso, "SBS": best_sbs}

per_k = {n: {"k": [], "PMSE": [], "R2": [], "feats": []} for n in SCHEMES}

for name, fn in SCHEMES.items():
    print(f"  {name}…")
    for k in range(1, len(FEATURES) + 1):
        sel = fn(k)
        pmse, r2 = eval_subset(sel)
        per_k[name]["k"].append(k)
        per_k[name]["PMSE"].append(pmse)
        per_k[name]["R2"].append(r2)
        per_k[name]["feats"].append(sel)

# Best overall
best_scheme = min(
    SCHEMES, key=lambda n: min(per_k[n]["PMSE"]))
idx_best = int(np.argmin(per_k[best_scheme]["PMSE"]))
best_features = per_k[best_scheme]["feats"][idx_best]

# Comparison table data
rows = []
for name in SCHEMES:
    i = int(np.argmin(per_k[name]["PMSE"]))
    rows.append({
        "Scheme":   name,
        "Features": ", ".join(per_k[name]["feats"][i]),
        "CV_PMSE":  round(per_k[name]["PMSE"][i], 2),
        "CV_R2":    round(per_k[name]["R2"][i], 4),
    })
comp_df  = pd.DataFrame(rows).sort_values("CV_PMSE").reset_index(drop=True)
comp_html = comp_df.to_html(classes="dt", border=0, index=False)

# Final model
final = build_model(best_features)
final.fit(X_train[best_features], y_train)
y_pred = final.predict(X_test[best_features])
metrics = {
    "MAE":  f"${mean_absolute_error(y_test, y_pred):,.2f}",
    "MSE":  f"${mean_squared_error(y_test, y_pred):,.0f}",
    "RMSE": f"${np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}",
    "R²":   f"{r2_score(y_test, y_pred):.4f}",
}
metric_colors = {"MAE": ACCENT2, "MSE": "#f7936a", "RMSE": ACCENT, "R²": "#68d391"}

pred_df   = pd.DataFrame({
    "Actual Profit":    y_test.values.round(2),
    "Predicted Profit": y_pred.round(2),
    "Residual":         (y_test.values - y_pred).round(2),
})
pred_html = pred_df.to_html(classes="dt", border=0, index=False)

new_s = pd.DataFrame({
    "R&D Spend": [150000], "Administration": [120000],
    "Marketing Spend": [250000], "State": ["California"]})
sample_pred = f"${float(final.predict(new_s[best_features])[0]):,.2f}"

# ── Charts ─────────────────────────────────────────────────────────────────
print("Generating charts…")

def line_chart(ylabel, key, title):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor(DARK_BG)
    styled_ax(ax, title)
    for i, (name, data) in enumerate(per_k.items()):
        ax.plot(data["k"], data[key], marker="o", linewidth=2.2,
                markersize=7, color=PALETTE[i], label=name, zorder=3)
    ax.set_xlabel("Number of Selected Features", color=TEXT, labelpad=8)
    ax.set_ylabel(ylabel, color=TEXT, labelpad=8)
    ax.set_xticks(range(1, len(FEATURES) + 1))
    ax.legend(facecolor=CARD_BG, labelcolor=TEXT, edgecolor=GRID,
              fontsize=8)
    return fig_to_b64(fig)

chart_pmse = line_chart("PMSE", "PMSE", "Number of Features vs PMSE")
chart_r2   = line_chart("R-squared", "R2", "Number of Features vs R-squared")

fig, ax = plt.subplots(figsize=(6, 5.5))
fig.patch.set_facecolor(DARK_BG)
styled_ax(ax, f"Actual vs Predicted ({best_scheme})")
ax.scatter(y_test, y_pred, color=ACCENT, alpha=0.85,
           edgecolors=ACCENT2, linewidths=0.8, s=80, zorder=3)
lo = min(float(y_test.min()), float(y_pred.min())) * 0.97
hi = max(float(y_test.max()), float(y_pred.max())) * 1.03
ax.plot([lo, hi], [lo, hi], "--", color=ACCENT2, linewidth=1.5, label="Perfect")
ax.set_xlabel("Actual Profit ($)", color=TEXT, labelpad=8)
ax.set_ylabel("Predicted Profit ($)", color=TEXT, labelpad=8)
ax.legend(facecolor=CARD_BG, labelcolor=TEXT, edgecolor=GRID)
chart_scatter = fig_to_b64(fig)

fig, ax = plt.subplots(figsize=(9, 4))
fig.patch.set_facecolor(DARK_BG)
styled_ax(ax, "Best CV RMSE by Scheme")
rmse_vals = comp_df["CV_PMSE"].apply(np.sqrt).values
bars = ax.bar(comp_df["Scheme"], rmse_vals, color=PALETTE[:len(comp_df)],
              edgecolor="none", zorder=3)
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 30,
            f"{h:,.0f}", ha="center", va="bottom",
            color=TEXT, fontsize=9, fontweight="bold")
ax.set_xlabel("Scheme", color=TEXT, labelpad=8)
ax.set_ylabel("CV RMSE", color=TEXT, labelpad=8)
ax.yaxis.grid(True, color=GRID, zorder=0)
chart_scheme = fig_to_b64(fig)

# ── State bars HTML ────────────────────────────────────────────────────────
state_bars = ""
total = sum(state_dist.values())
for state, count in state_dist.items():
    pct = round(count / total * 100)
    state_bars += f"""
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
        <span style="width:110px;font-size:.82rem;">{html_lib.escape(state)}</span>
        <div style="flex:1;background:var(--border);border-radius:99px;height:7px;">
          <div style="width:{pct}%;background:var(--accent);height:7px;border-radius:99px;"></div>
        </div>
        <span style="font-size:.8rem;color:var(--muted);width:24px;text-align:right;">{count}</span>
      </div>"""

# ── Metric cards HTML ──────────────────────────────────────────────────────
metric_cards = ""
for label, val in metrics.items():
    color = metric_colors[label]
    metric_cards += f"""
      <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color};">{val}</div>
      </div>"""

# ── Best features pills ────────────────────────────────────────────────────
feat_pills = "".join(
    f'<span class="feat-pill">{html_lib.escape(f)}</span>'
    for f in best_features)

# ── Build HTML ─────────────────────────────────────────────────────────────
print("Writing HTML…")
HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>50 Startups — MLR Dashboard</title>
  <meta name="description" content="CRISP-DM Multiple Linear Regression dashboard comparing 5 feature selection schemes on the 50 Startups dataset."/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet"/>
  <style>
    :root{{
      --bg:#0f1117;--surface:#1a1d2e;--card:#20243a;--border:#2a2d3e;
      --accent:#7c6af7;--accent2:#56cfb2;--warn:#f7936a;
      --text:#e2e8f0;--muted:#8892a4;--radius:14px;
    }}
    *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0;}}
    body{{font-family:"Inter",sans-serif;background:var(--bg);color:var(--text);min-height:100vh;}}
    header{{background:linear-gradient(135deg,#1a1d2e 0%,#12152a 100%);border-bottom:1px solid var(--border);padding:40px 48px 36px;position:relative;overflow:hidden;}}
    header::before{{content:"";position:absolute;top:-80px;right:-80px;width:320px;height:320px;background:radial-gradient(circle,rgba(124,106,247,.18) 0%,transparent 70%);}}
    header::after{{content:"";position:absolute;bottom:-60px;left:60px;width:240px;height:240px;background:radial-gradient(circle,rgba(86,207,178,.12) 0%,transparent 70%);}}
    .badge{{display:inline-block;background:rgba(124,106,247,.15);border:1px solid rgba(124,106,247,.35);color:var(--accent);font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;padding:4px 12px;border-radius:99px;margin-bottom:14px;}}
    header h1{{font-size:clamp(1.6rem,3vw,2.4rem);font-weight:700;letter-spacing:-.02em;line-height:1.2;}}
    header h1 span{{color:var(--accent);}}
    header p{{margin-top:10px;color:var(--muted);font-size:.95rem;max-width:620px;line-height:1.6;}}
    .chips{{margin-top:18px;display:flex;flex-wrap:wrap;gap:8px;}}
    .chip{{background:var(--card);border:1px solid var(--border);border-radius:99px;padding:4px 14px;font-size:.78rem;color:var(--muted);}}
    .chip strong{{color:var(--text);}}
    main{{max-width:1200px;margin:0 auto;padding:40px 24px 80px;}}
    section{{margin-bottom:48px;}}
    .stitle{{font-size:.7rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin-bottom:20px;display:flex;align-items:center;gap:10px;}}
    .stitle::after{{content:"";flex:1;height:1px;background:var(--border);}}
    .card{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:24px;transition:border-color .2s;}}
    .card:hover{{border-color:rgba(124,106,247,.4);}}
    .ctitle{{font-size:.8rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:14px;}}
    .g2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;}}
    .g3{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;}}
    .g4{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;}}
    @media(max-width:800px){{.g2,.g3,.g4{{grid-template-columns:1fr 1fr;}}}}
    @media(max-width:540px){{.g2,.g3,.g4{{grid-template-columns:1fr;}}}}
    .stat{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:22px 20px;transition:transform .2s,border-color .2s;}}
    .stat:hover{{transform:translateY(-3px);border-color:rgba(124,106,247,.45);}}
    .slabel{{font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin-bottom:8px;}}
    .sval{{font-size:1.75rem;font-weight:700;letter-spacing:-.02em;}}
    .ssub{{font-size:.75rem;color:var(--muted);margin-top:4px;}}
    .metric-card{{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:20px;text-align:center;transition:transform .2s;}}
    .metric-card:hover{{transform:translateY(-3px);}}
    .metric-label{{font-size:.72rem;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px;}}
    .metric-value{{font-size:1.65rem;font-weight:700;}}
    .tw{{overflow-x:auto;border-radius:10px;}}
    .dt{{width:100%;border-collapse:collapse;font-size:.82rem;}}
    .dt th{{background:var(--card);color:var(--muted);font-weight:600;font-size:.72rem;text-transform:uppercase;letter-spacing:.07em;padding:10px 14px;text-align:left;border-bottom:1px solid var(--border);}}
    .dt td{{padding:9px 14px;border-bottom:1px solid var(--border);color:var(--text);}}
    .dt tbody tr:hover{{background:rgba(124,106,247,.06);}}
    .dt tbody tr:last-child td{{border-bottom:none;}}
    img.chart{{width:100%;border-radius:10px;display:block;}}
    .best-banner{{background:linear-gradient(135deg,rgba(124,106,247,.15),rgba(86,207,178,.1));border:1px solid rgba(124,106,247,.35);border-radius:var(--radius);padding:22px 28px;display:flex;align-items:center;gap:20px;flex-wrap:wrap;margin-top:18px;}}
    .best-badge{{background:var(--accent);color:#fff;font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;padding:5px 14px;border-radius:99px;white-space:nowrap;}}
    .feat-pill{{display:inline-block;background:rgba(86,207,178,.12);border:1px solid rgba(86,207,178,.3);color:var(--accent2);border-radius:99px;padding:2px 10px;font-size:.75rem;font-weight:500;margin:4px 3px 0 0;}}
    .pred-card{{background:linear-gradient(135deg,rgba(86,207,178,.1),rgba(124,106,247,.1));border:1px solid rgba(86,207,178,.3);border-radius:var(--radius);padding:28px 30px;display:flex;align-items:center;justify-content:space-between;gap:24px;flex-wrap:wrap;}}
    .pi-wrap{{display:flex;flex-wrap:wrap;gap:12px;}}
    .pi{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:10px 16px;}}
    .pi .lbl{{font-size:.68rem;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.08em;}}
    .pi .val{{font-size:.95rem;font-weight:600;margin-top:2px;}}
    .pred-result{{text-align:center;}}
    .pred-result .lbl{{font-size:.72rem;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;}}
    .amount{{font-size:2.2rem;font-weight:700;color:var(--accent2);letter-spacing:-.02em;}}
    footer{{text-align:center;padding:24px;color:var(--muted);font-size:.78rem;border-top:1px solid var(--border);}}
    .fade{{animation:fadeIn .5s ease both;}}
    @keyframes fadeIn{{from{{opacity:0;transform:translateY(12px);}}to{{opacity:1;transform:none;}}}}
    code{{background:var(--card);padding:1px 6px;border-radius:4px;font-size:.8rem;}}
  </style>
</head>
<body>

<header>
  <div class="badge">CRISP-DM · Multiple Linear Regression</div>
  <h1>50 Startups <span>Profit Predictor</span></h1>
  <p>Feature selection comparison across five schemes — Exhaustive, RFE, SFS, Lasso, and SBS — evaluated via cross-validation to identify the best model for predicting startup profit.</p>
  <div class="chips">
    <div class="chip"><strong>Dataset:</strong> {shape[0]} rows · {shape[1]} cols</div>
    <div class="chip"><strong>Algorithm:</strong> Linear Regression</div>
    <div class="chip"><strong>Schemes:</strong> 5 feature selection methods</div>
    <div class="chip"><strong>CV:</strong> 5-Fold KFold</div>
    <div class="chip"><strong>Split:</strong> 80 / 20</div>
  </div>
</header>

<main>

  <!-- Phase 2 -->
  <section class="fade">
    <h2 class="stitle">Phase 2 — Data Understanding</h2>
    <div class="g4" style="margin-bottom:20px;">
      <div class="stat"><div class="slabel">Rows</div><div class="sval" style="color:var(--accent);">{shape[0]}</div><div class="ssub">startup records</div></div>
      <div class="stat"><div class="slabel">Columns</div><div class="sval" style="color:var(--accent2);">{shape[1]}</div><div class="ssub">features + target</div></div>
      <div class="stat"><div class="slabel">Missing Values</div><div class="sval" style="color:#68d391;">0</div><div class="ssub">clean dataset</div></div>
      <div class="stat"><div class="slabel">Duplicates</div><div class="sval" style="color:#68d391;">{duplicates}</div><div class="ssub">no duplicates</div></div>
    </div>
    <div class="g2" style="margin-bottom:20px;">
      <div class="card"><div class="ctitle">First 5 Rows</div><div class="tw">{head_html}</div></div>
      <div class="card" style="display:flex;flex-direction:column;gap:14px;">
        <div>
          <div class="ctitle">State Distribution</div>
          {state_bars}
        </div>
        <div>
          <div class="ctitle">Missing Values per Column</div>
          <div class="tw"><table class="dt"><thead><tr><th>Column</th><th>Missing</th></tr></thead><tbody>
          {''.join(f"<tr><td>{html_lib.escape(k)}</td><td>{v}</td></tr>" for k, v in missing.items())}
          </tbody></table></div>
        </div>
      </div>
    </div>
    <div class="g2">
      <div class="card"><div class="ctitle">Num Features vs PMSE</div><img class="chart" src="data:image/png;base64,{chart_pmse}" alt="PMSE"/></div>
      <div class="card"><div class="ctitle">Num Features vs R-squared</div><img class="chart" src="data:image/png;base64,{chart_r2}" alt="R2"/></div>
    </div>
  </section>

  <!-- Phase 3 -->
  <section class="fade">
    <h2 class="stitle">Phase 3 — Data Preparation</h2>
    <div class="g3">
      <div class="stat"><div class="slabel">Training Samples</div><div class="sval" style="color:var(--accent);">{len(X_train)}</div><div class="ssub">80% of dataset</div></div>
      <div class="stat"><div class="slabel">Testing Samples</div><div class="sval" style="color:var(--accent2);">{len(X_test)}</div><div class="ssub">20% of dataset</div></div>
      <div class="stat"><div class="slabel">CV Strategy</div><div class="sval" style="font-size:1.3rem;">5-Fold</div><div class="ssub">KFold, shuffle, seed=42</div></div>
    </div>
    <div class="card" style="margin-top:20px;font-size:.82rem;color:var(--muted);line-height:1.75;">
      <div class="ctitle">Preprocessing Pipeline</div>
      <p><strong style="color:var(--text);">Numeric</strong> features: <code>SimpleImputer(mean)</code> → <code>StandardScaler</code></p>
      <p style="margin-top:8px;"><strong style="color:var(--text);">Categorical</strong> (State): <code>SimpleImputer(most_frequent)</code> → <code>OneHotEncoder(drop='first')</code></p>
    </div>
  </section>

  <!-- Phase 4 -->
  <section class="fade">
    <h2 class="stitle">Phase 4 — Feature Selection Comparison</h2>
    <div class="g2">
      <div class="card"><div class="ctitle">CV RMSE by Scheme</div><img class="chart" src="data:image/png;base64,{chart_scheme}" alt="Scheme Bar"/></div>
      <div class="card">
        <div class="ctitle">Results Table</div>
        <div class="tw">{comp_html}</div>
        <div class="best-banner">
          <div class="best-badge">🏆 Best</div>
          <div>
            <h3 style="font-size:1.05rem;font-weight:600;">{html_lib.escape(best_scheme)} Selected</h3>
            <p style="font-size:.82rem;color:var(--muted);margin-top:4px;">Winning feature set:</p>
            {feat_pills}
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- Phase 5 -->
  <section class="fade">
    <h2 class="stitle">Phase 5 — Model Evaluation (Hold-out Test Set)</h2>
    <div class="g4" style="margin-bottom:20px;">{metric_cards}</div>
    <div class="g2">
      <div class="card"><div class="ctitle">Actual vs Predicted Profit</div><img class="chart" src="data:image/png;base64,{chart_scatter}" alt="Scatter"/></div>
      <div class="card"><div class="ctitle">Prediction Results (Test Set)</div><div class="tw">{pred_html}</div></div>
    </div>
  </section>

  <!-- Phase 6 -->
  <section class="fade">
    <h2 class="stitle">Phase 6 — Deployment</h2>
    <div class="pred-card">
      <div>
        <div style="font-size:.72rem;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.1em;margin-bottom:12px;">Sample New Startup Inputs</div>
        <div class="pi-wrap">
          <div class="pi"><div class="lbl">R&amp;D Spend</div><div class="val">$150,000</div></div>
          <div class="pi"><div class="lbl">Administration</div><div class="val">$120,000</div></div>
          <div class="pi"><div class="lbl">Marketing Spend</div><div class="val">$250,000</div></div>
          <div class="pi"><div class="lbl">State</div><div class="val">California</div></div>
        </div>
      </div>
      <div class="pred-result">
        <div class="lbl">Predicted Profit</div>
        <div class="amount">{sample_pred}</div>
      </div>
    </div>
    <p style="margin-top:14px;font-size:.78rem;color:var(--muted);">
      Best scheme: <strong style="color:var(--accent);">{html_lib.escape(best_scheme)}</strong> ·
      Features used: <strong style="color:var(--accent2);">{html_lib.escape(", ".join(best_features))}</strong>
    </p>
  </section>

</main>
<footer>50 Startups · Multiple Linear Regression Dashboard · CRISP-DM Framework</footer>
</body>
</html>"""

os.makedirs("docs", exist_ok=True)
with open("docs/index.html", "w", encoding="utf-8") as f:
    f.write(HTML)

print("✅  docs/index.html written successfully!")

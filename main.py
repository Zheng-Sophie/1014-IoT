
# ============================================================
# CRISP-DM: 50 Startups Multiple Linear Regression
# Five Feature Selection Methods
#
# 1. Exhaustive Feature Selection
# 2. Recursive Feature Elimination (RFE)
# 3. Sequential Forward Selection (SFS)
# 4. Lasso Feature Selection
# 5. Sequential Backward Selection (SBS)
#
# X-axis: Number of Selected Features
# Y-axis: PMSE / R-squared
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from itertools import combinations

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_predict,
    cross_val_score
)
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import (
    LinearRegression,
    LassoCV
)
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

RANDOM_STATE = 42

FEATURES = [
    "R&D Spend",
    "Administration",
    "Marketing Spend",
    "State"
]

NUMERIC_FEATURES = FEATURES[:3]


# ============================================================
# Phase 1: Business Understanding
# ============================================================

print("=" * 60)
print("Phase 1: Business Understanding")
print("=" * 60)

print("Objective: Predict startup Profit")
print("Algorithm: Multiple Linear Regression")
print("Dataset: Kaggle 50 Startups")
print("Feature Selection: 5 Methods")


# ============================================================
# Phase 2: Data Understanding
# ============================================================

print("\n" + "=" * 60)
print("Phase 2: Data Understanding")
print("=" * 60)

df = pd.read_csv("50_Startups.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe(include="all"))

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nState Distribution:")
print(df["State"].value_counts())

print("\nCorrelation Matrix:")
print(df.select_dtypes(include=np.number).corr())


# ============================================================
# Phase 3: Data Preparation
# ============================================================

print("\n" + "=" * 60)
print("Phase 3: Data Preparation")
print("=" * 60)

# Remove duplicates and rows with missing target
df = df.drop_duplicates()
df = df.dropna(subset=["Profit"])

X = df[FEATURES]
y = df["Profit"]

# Hold-out test set (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# Outer CV: evaluation of feature-selection procedures
outer_cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


def make_cv(n_samples):
    """Create inner CV for feature selection."""
    return KFold(
        n_splits=min(5, n_samples),
        shuffle=True,
        random_state=RANDOM_STATE
    )


def build_preprocessor(selected_features):
    """Preprocess numeric and categorical features."""

    numeric = [
        f for f in selected_features
        if f != "State"
    ]

    transformers = []

    if numeric:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        transformers.append((
            "num",
            numeric_pipeline,
            numeric
        ))

    if "State" in selected_features:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(
                strategy="most_frequent"
            )),
            ("onehot", OneHotEncoder(
                categories=[[
                    "California",
                    "Florida",
                    "New York"
                ]],
                drop="first",
                handle_unknown="ignore",
                sparse_output=False
            ))
        ])

        transformers.append((
            "cat",
            categorical_pipeline,
            ["State"]
        ))

    return ColumnTransformer(transformers)


def build_model(selected_features):
    """Create multiple linear regression pipeline."""

    return Pipeline([
        ("preprocessor",
         build_preprocessor(selected_features)),
        ("regressor", LinearRegression())
    ])


def cv_pmse(X_data, y_data, subset):
    """Inner cross-validation prediction MSE."""

    model = build_model(subset)

    pred = cross_val_predict(
        model,
        X_data[subset],
        y_data,
        cv=make_cv(len(y_data))
    )

    return mean_squared_error(y_data, pred)


# ============================================================
# Phase 4: Modeling and Feature Selection
# ============================================================

print("\n" + "=" * 60)
print("Phase 4: Modeling and Feature Selection")
print("=" * 60)


# ------------------------------------------------------------
# Scheme 1: Exhaustive Feature Selection
# ------------------------------------------------------------

def select_exhaustive(X_data, y_data, k):

    best_score = float("inf")
    best_subset = None

    for subset in combinations(FEATURES, k):
        subset = list(subset)

        score = cv_pmse(X_data, y_data, subset)

        if score < best_score:
            best_score = score
            best_subset = subset

    return best_subset


# ------------------------------------------------------------
# Scheme 2: Recursive Feature Elimination (RFE)
# ------------------------------------------------------------

def select_rfe(X_data, y_data, k):

    candidates = []

    for include_state in [False, True]:

        n_numeric = k - int(include_state)

        # Handle State-only model
        if n_numeric == 0 and include_state:
            subset = ["State"]
            score = cv_pmse(X_data, y_data, subset)
            candidates.append((score, subset))
            continue

        if not 1 <= n_numeric <= 3:
            continue

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        X_numeric = numeric_pipeline.fit_transform(
            X_data[NUMERIC_FEATURES]
        )

        rfe = RFE(
            estimator=LinearRegression(),
            n_features_to_select=n_numeric
        )

        rfe.fit(X_numeric, y_data)

        subset = [
            f for f, keep in zip(
                NUMERIC_FEATURES,
                rfe.support_
            )
            if keep
        ]

        if include_state:
            subset.append("State")

        score = cv_pmse(X_data, y_data, subset)
        candidates.append((score, subset))

    return min(candidates, key=lambda x: x[0])[1]


# ------------------------------------------------------------
# Scheme 3: Sequential Forward Selection (SFS)
# ------------------------------------------------------------

def select_sfs(X_data, y_data, k):

    selected = []
    remaining = FEATURES.copy()

    while len(selected) < k:

        candidates = []

        for feature in remaining:
            subset = selected + [feature]
            score = cv_pmse(X_data, y_data, subset)

            candidates.append((score, feature))

        _, best_feature = min(
            candidates,
            key=lambda x: x[0]
        )

        selected.append(best_feature)
        remaining.remove(best_feature)

    return selected


# ------------------------------------------------------------
# Scheme 4: Lasso Feature Selection
# ------------------------------------------------------------

def select_lasso(X_data, y_data, k):

    preprocessor = build_preprocessor(FEATURES)

    X_processed = preprocessor.fit_transform(X_data)

    lasso = LassoCV(
        cv=make_cv(len(y_data)),
        random_state=RANDOM_STATE,
        max_iter=100000
    )

    lasso.fit(X_processed, y_data)

    names = preprocessor.get_feature_names_out()

    importance = {f: 0.0 for f in FEATURES}

    for name, coef in zip(names, lasso.coef_):

        if name.startswith("num__"):
            original = name.replace("num__", "")
        else:
            original = "State"

        importance[original] = max(
            importance[original],
            abs(coef)
        )

    ranked = sorted(
        FEATURES,
        key=lambda f: importance[f],
        reverse=True
    )

    return ranked[:k]


# ------------------------------------------------------------
# Scheme 5: Sequential Backward Selection (SBS)
# ------------------------------------------------------------

def select_sbs(X_data, y_data, k):

    selected = FEATURES.copy()

    while len(selected) > k:

        candidates = []

        for feature in selected:
            subset = [
                f for f in selected
                if f != feature
            ]

            score = cv_pmse(X_data, y_data, subset)
            candidates.append((score, subset))

        selected = min(
            candidates,
            key=lambda x: x[0]
        )[1]

    return selected


# ------------------------------------------------------------
# All Feature Selection Schemes
# ------------------------------------------------------------

SCHEMES = {
    "Exhaustive": select_exhaustive,
    "RFE": select_rfe,
    "SFS": select_sfs,
    "Lasso": select_lasso,
    "SBS": select_sbs
}


# ============================================================
# Phase 5: Evaluation
# ============================================================

print("\n" + "=" * 60)
print("Phase 5: Evaluation")
print("=" * 60)

results = []

# Nested cross-validation:
# Selection is repeated within each outer training fold.

for method_name, selector in SCHEMES.items():

    print(f"\nEvaluating: {method_name}")

    for k in range(1, len(FEATURES) + 1):

        all_true = []
        all_pred = []

        for train_idx, valid_idx in outer_cv.split(X_train):

            X_inner_train = X_train.iloc[train_idx]
            X_inner_valid = X_train.iloc[valid_idx]

            y_inner_train = y_train.iloc[train_idx]
            y_inner_valid = y_train.iloc[valid_idx]

            # Feature selection on inner training only
            subset = selector(
                X_inner_train,
                y_inner_train,
                k
            )

            # Train regression model
            model = build_model(subset)

            model.fit(
                X_inner_train[subset],
                y_inner_train
            )

            predictions = model.predict(
                X_inner_valid[subset]
            )

            all_true.extend(y_inner_valid.to_numpy())
            all_pred.extend(predictions)

        # Out-of-fold evaluation
        pmse = mean_squared_error(all_true, all_pred)
        r2 = r2_score(all_true, all_pred)

        # Select features using all training data
        # for final model construction only
        final_subset = selector(
            X_train,
            y_train,
            k
        )

        results.append({
            "Method": method_name,
            "Num_Features": k,
            "Features": final_subset,
            "PMSE": pmse,
            "R2": r2
        })

        print(
            f"k={k}, PMSE={pmse:.2f}, "
            f"R2={r2:.4f}, "
            f"Features={final_subset}"
        )


comparison_df = pd.DataFrame(results)

print("\nComplete Feature Selection Results:")
print(
    comparison_df.sort_values(
        ["Method", "Num_Features"]
    ).to_string(index=False)
)

comparison_df.to_csv(
    "feature_selection_results.csv",
    index=False
)


# ============================================================
# Figure 1: Number of Features vs PMSE
# ============================================================

plt.figure(figsize=(10, 6))

for method, group in comparison_df.groupby("Method"):

    group = group.sort_values("Num_Features")

    plt.plot(
        group["Num_Features"],
        group["PMSE"],
        marker="o",
        linewidth=2,
        label=method
    )

plt.xlabel("Number of Selected Features")
plt.ylabel("PMSE")
plt.title("Feature Selection Comparison - PMSE")
plt.xticks(range(1, len(FEATURES) + 1))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "feature_selection_pmse.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ============================================================
# Figure 2: Number of Features vs R-squared
# ============================================================

plt.figure(figsize=(10, 6))

for method, group in comparison_df.groupby("Method"):

    group = group.sort_values("Num_Features")

    plt.plot(
        group["Num_Features"],
        group["R2"],
        marker="o",
        linewidth=2,
        label=method
    )

plt.xlabel("Number of Selected Features")
plt.ylabel("R-squared")
plt.title("Feature Selection Comparison - R-squared")
plt.xticks(range(1, len(FEATURES) + 1))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "feature_selection_r2.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ============================================================
# Find Best Scheme and Number of Features
# ============================================================

best_result = comparison_df.loc[
    comparison_df["PMSE"].idxmin()
]

best_scheme = best_result["Method"]
best_features = best_result["Features"]
best_k = best_result["Num_Features"]

print("\n" + "=" * 60)
print("Best Feature Selection Result")
print("=" * 60)

print("Best Scheme:", best_scheme)
print("Number of Features:", best_k)
print("Selected Features:", best_features)
print("Cross-Validation PMSE:", best_result["PMSE"])
print("Cross-Validation R2:", best_result["R2"])


# ============================================================
# Train Final Multiple Linear Regression Model
# ============================================================

final_model = build_model(best_features)

final_model.fit(
    X_train[best_features],
    y_train
)

y_pred = final_model.predict(
    X_test[best_features]
)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nFinal Test Evaluation:")
print(f"MAE:  {mae:.2f}")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2:   {r2:.4f}")


# ============================================================
# Actual vs Predicted Profit
# ============================================================

prediction_df = pd.DataFrame({
    "Actual Profit": y_test.to_numpy(),
    "Predicted Profit": y_pred,
    "Residual": y_test.to_numpy() - y_pred
})

print("\nPrediction Results:")
print(prediction_df.round(2).to_string(index=False))

prediction_df.to_csv(
    "startup_predictions.csv",
    index=False
)

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

low = min(y_test.min(), y_pred.min())
high = max(y_test.max(), y_pred.max())

plt.plot(
    [low, high],
    [low, high],
    "r--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title("Actual vs Predicted Startup Profit")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ============================================================
# Phase 6: Deployment
# ============================================================

print("\n" + "=" * 60)
print("Phase 6: Deployment")
print("=" * 60)

joblib.dump(
    {
        "model": final_model,
        "features": best_features,
        "scheme": best_scheme
    },
    "startup_profit_model.pkl"
)

print("Model saved successfully!")

# Example: Predict profit for a new startup
new_startup = pd.DataFrame({
    "R&D Spend": [150000],
    "Administration": [120000],
    "Marketing Spend": [250000],
    "State": ["California"]
})

predicted_profit = final_model.predict(
    new_startup[best_features]
)

print("\nNew Startup Prediction:")
print(f"Predicted Profit: ${predicted_profit[0]:,.2f}")

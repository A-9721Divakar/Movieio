"""
MovieIQ — Predictive Analytics on Film Success
Full analysis pipeline: data prep -> EDA -> statistical testing -> modeling
Run this once to regenerate assets/ charts and model/ artifacts used by the Streamlit app.
"""

import ast
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

warnings.filterwarnings("ignore")
sns.set_theme(style="darkgrid", palette="viridis")
plt.rcParams["figure.dpi"] = 120

ASSETS = "assets"
MODEL_DIR = "model"

# ---------------------------------------------------------------------------
# STAGE 1 — DATA PREPARATION
# ---------------------------------------------------------------------------
print("=" * 70)
print("STAGE 1: DATA PREPARATION")
print("=" * 70)

df = pd.read_csv("movies.csv")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("\nSummary statistics (numeric fields):")
print(df[["budget", "revenue", "popularity", "runtime", "vote_average"]].describe())

# Missing values / zero checks
missing = df.isna().sum()
zero_budget = (df["budget"] <= 0).sum()
zero_revenue = (df["revenue"] <= 0).sum()
print(f"\nMissing values per column:\n{missing}")
print(f"\nRows with budget <= 0: {zero_budget}")
print(f"Rows with revenue <= 0: {zero_revenue}")
# A budget or revenue of 0 is almost always a data-entry placeholder rather than
# a true value (a real theatrical release does not cost or earn literally $0),
# so a success ratio computed from it would be meaningless. We drop such rows.
before = len(df)
df = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()
print(f"Dropped {before - len(df)} rows with zero/invalid budget or revenue.")

# Parse the genres column (stored as a stringified list of TMDB-style dicts,
# e.g. "[{'id': 18, 'name': 'Drama'}]"; some rows contain the placeholder "N A").
def parse_genres(raw):
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [d["name"] for d in parsed if isinstance(d, dict) and "name" in d]
    except (ValueError, SyntaxError):
        pass
    return []


df["genre_list"] = df["genres"].apply(parse_genres)
df["genre_primary"] = df["genre_list"].apply(lambda g: g[0] if g else "Unknown")
n_no_genre = (df["genre_primary"] == "Unknown").sum()
print(f"\nRows with unparseable / missing genre tag: {n_no_genre}")

# Target column
df["success"] = (df["revenue"] > df["budget"]).astype(int)
success_rate = df["success"].mean()
print(f"\nSuccess rate: {success_rate:.2%}  "
      f"({df['success'].sum()} successful / {len(df) - df['success'].sum()} not)")
print("Dataset balance: "
      + ("reasonably balanced" if 0.35 < success_rate < 0.65 else "imbalanced")
      + f" ({success_rate:.1%} positive class).")

df.to_csv("movies_clean.csv", index=False)

# ---------------------------------------------------------------------------
# STAGE 2 — EXPLORATORY DATA ANALYSIS (10 charts)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STAGE 2: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# EDA 1 — Budget vs Revenue scatter
plt.figure(figsize=(8, 5.5))
sns.scatterplot(
    data=df, x="budget", y="revenue", hue="success",
    palette={0: "#e15759", 1: "#59a14f"}, alpha=0.6, s=35,
)
lims = [df["budget"].min(), df["budget"].max()]
plt.plot(lims, lims, "--", color="gray", linewidth=1, label="Break-even (revenue = budget)")
plt.title("Budget vs. Revenue")
plt.xlabel("Budget ($)")
plt.ylabel("Revenue ($)")
plt.legend(title="Success")
plt.tight_layout()
plt.savefig(f"{ASSETS}/budget_vs_revenue.png")
plt.close()
budget_revenue_corr = df["budget"].corr(df["revenue"])
print(f"EDA 1 saved: budget_vs_revenue.png  (correlation = {budget_revenue_corr:.3f})")

# EDA 2 — Genre distribution (most common genres)
plt.figure(figsize=(8, 5))
genre_counts = df["genre_primary"].value_counts()
sns.barplot(x=genre_counts.values, y=genre_counts.index, palette="viridis")
plt.title("Movie Count by Genre")
plt.xlabel("Number of Movies")
plt.ylabel("Genre")
plt.tight_layout()
plt.savefig(f"{ASSETS}/genre_distribution.png")
plt.close()
print("EDA 2 saved: genre_distribution.png")

# EDA 3 — Genre success rate
plt.figure(figsize=(8, 5))
genre_success = df.groupby("genre_primary")["success"].mean().sort_values(ascending=False)
sns.barplot(x=genre_success.values, y=genre_success.index, palette="mako")
plt.title("Success Rate by Genre")
plt.xlabel("Success Rate")
plt.ylabel("Genre")
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig(f"{ASSETS}/genre_success_rate.png")
plt.close()
print("EDA 3 saved: genre_success_rate.png")

# EDA 4 — Popularity vs success (boxplot)
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="success", y="popularity", hue="success", palette={0: "#e15759", 1: "#59a14f"}, legend=False)
plt.xticks([0, 1], ["Not Successful", "Successful"])
plt.title("Popularity by Success Outcome")
plt.tight_layout()
plt.savefig(f"{ASSETS}/popularity_vs_success.png")
plt.close()
print("EDA 4 saved: popularity_vs_success.png")

# EDA 5 — Runtime vs success (boxplot)
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="success", y="runtime", hue="success", palette={0: "#e15759", 1: "#59a14f"}, legend=False)
plt.xticks([0, 1], ["Not Successful", "Successful"])
plt.title("Runtime by Success Outcome")
plt.tight_layout()
plt.savefig(f"{ASSETS}/runtime_vs_success.png")
plt.close()
print("EDA 5 saved: runtime_vs_success.png")

# EDA 6 — Vote average vs success (boxplot)
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="success", y="vote_average", hue="success", palette={0: "#e15759", 1: "#59a14f"}, legend=False)
plt.xticks([0, 1], ["Not Successful", "Successful"])
plt.title("Average Vote by Success Outcome")
plt.tight_layout()
plt.savefig(f"{ASSETS}/vote_average_vs_success.png")
plt.close()
print("EDA 6 saved: vote_average_vs_success.png")

# EDA 7 — Correlation heatmap
plt.figure(figsize=(7, 6))
num_cols = ["budget", "revenue", "popularity", "runtime", "vote_average", "success"]
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.savefig(f"{ASSETS}/correlation_heatmap.png")
plt.close()
print("EDA 7 saved: correlation_heatmap.png")

# EDA 8 — Distribution of vote_average
plt.figure(figsize=(7, 5))
sns.histplot(df["vote_average"], bins=25, kde=True, color="#4e79a7")
plt.title("Distribution of Average Vote")
plt.xlabel("vote_average")
plt.tight_layout()
plt.savefig(f"{ASSETS}/vote_average_distribution.png")
plt.close()
print("EDA 8 saved: vote_average_distribution.png")

# EDA 9 — Popularity distribution split by success (KDE)
plt.figure(figsize=(7, 5))
sns.kdeplot(data=df, x="popularity", hue="success", fill=True,
            palette={0: "#e15759", 1: "#59a14f"}, common_norm=False, alpha=0.4)
plt.title("Popularity Distribution: Success vs. Not")
plt.tight_layout()
plt.savefig(f"{ASSETS}/popularity_distribution_by_success.png")
plt.close()
print("EDA 9 saved: popularity_distribution_by_success.png")

# EDA 10 — Average revenue-to-budget ratio by genre
df["roi"] = df["revenue"] / df["budget"]
genre_roi = df.groupby("genre_primary")["roi"].mean().sort_values(ascending=False)
plt.figure(figsize=(8, 5))
sns.barplot(x=genre_roi.values, y=genre_roi.index, palette="crest")
plt.axvline(1.0, color="gray", linestyle="--", linewidth=1)
plt.title("Average Revenue-to-Budget Ratio by Genre")
plt.xlabel("Avg. Revenue / Budget")
plt.tight_layout()
plt.savefig(f"{ASSETS}/genre_roi.png")
plt.close()
print("EDA 10 saved: genre_roi.png")

# ---------------------------------------------------------------------------
# STAGE 3 — STATISTICAL TESTING
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STAGE 3: STATISTICAL TESTING")
print("=" * 70)

# T-Test: popularity between successful vs unsuccessful movies
pop_success = df.loc[df["success"] == 1, "popularity"]
pop_fail = df.loc[df["success"] == 0, "popularity"]
t_stat, t_pvalue = stats.ttest_ind(pop_success, pop_fail, equal_var=False)
print("T-Test — H0: mean popularity is equal for successful and unsuccessful movies.")
print(f"  t-statistic = {t_stat:.4f}, p-value = {t_pvalue:.6f}")
print(f"  Conclusion: {'Reject H0 — significant difference' if t_pvalue < 0.05 else 'Fail to reject H0 — no significant difference'} at alpha = 0.05.")

# Chi-Square: genre vs success
contingency = pd.crosstab(df["genre_primary"], df["success"])
chi2, chi_pvalue, dof, expected = stats.chi2_contingency(contingency)
print("\nChi-Square — H0: genre and success are independent.")
print(f"  chi2 = {chi2:.4f}, dof = {dof}, p-value = {chi_pvalue:.6f}")
print(f"  Conclusion: {'Reject H0 — genre is associated with success' if chi_pvalue < 0.05 else 'Fail to reject H0 — no significant association'} at alpha = 0.05.")

stats_results = {
    "t_test": {"statistic": float(t_stat), "p_value": float(t_pvalue),
               "feature": "popularity", "alpha": 0.05,
               "significant": bool(t_pvalue < 0.05)},
    "chi_square": {"statistic": float(chi2), "p_value": float(chi_pvalue),
                   "dof": int(dof), "feature": "genre_primary", "alpha": 0.05,
                   "significant": bool(chi_pvalue < 0.05)},
}
with open(f"{MODEL_DIR}/stats_results.json", "w") as f:
    json.dump(stats_results, f, indent=2)
print("\nSaved stats_results.json")

# ---------------------------------------------------------------------------
# STAGE 4 — PREDICTIVE MODELING (RANDOM FOREST)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STAGE 4: PREDICTIVE MODELING")
print("=" * 70)

# Features: budget, popularity, runtime, vote_average, genre (one-hot).
# revenue is excluded because it is used to derive the target itself (data leakage);
# title is excluded because it is a unique identifier with no predictive signal.
feature_cols_numeric = ["budget", "popularity", "runtime", "vote_average"]
genre_dummies = pd.get_dummies(df["genre_primary"], prefix="genre")
X = pd.concat([df[feature_cols_numeric], genre_dummies], axis=1)
y = df["success"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)} (80/20 split, stratified on target).")

rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-score:  {f1:.3f}")
print(f"Confusion matrix:\n{cm}")

# Confusion matrix chart
plt.figure(figsize=(5.5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Successful", "Successful"],
            yticklabels=["Not Successful", "Successful"])
plt.title("Confusion Matrix — Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{ASSETS}/confusion_matrix.png")
plt.close()
print("Saved confusion_matrix.png")

# Feature importance chart
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8, 6))
sns.barplot(x=importances.values, y=importances.index, palette="flare")
plt.title("Feature Importance — Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{ASSETS}/feature_importance.png")
plt.close()
print("Saved feature_importance.png")
print(f"\nTop features:\n{importances.head(6)}")

model_metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "confusion_matrix": cm.tolist(),
    "feature_importance": importances.to_dict(),
    "feature_columns": list(X.columns),
    "genre_options": sorted(df["genre_primary"].unique().tolist()),
}
with open(f"{MODEL_DIR}/model_metrics.json", "w") as f:
    json.dump(model_metrics, f, indent=2)

# Persist model + the exact feature column order for inference
joblib.dump(rf, f"{MODEL_DIR}/movie_success_model.pkl")
joblib.dump(list(X.columns), f"{MODEL_DIR}/feature_columns.pkl")
joblib.dump(sorted(df["genre_primary"].unique().tolist()), f"{MODEL_DIR}/genre_encoder.pkl")

print("\nSaved model/movie_success_model.pkl, feature_columns.pkl, genre_encoder.pkl")
print("\nPipeline complete.")

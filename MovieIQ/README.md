# 🎬 MovieIQ — Predictive Analytics on Film Success

MovieIQ is an interactive dashboard that analyzes and predicts whether a movie
will succeed, using budget, revenue, popularity, runtime, and average votes.

**A movie is labeled successful when `revenue > budget`.**

## Project Structure

```
MovieIQ/
├── MovieIQ.ipynb            # Full analysis notebook (Stages 0–4)
├── MovieIQ.py                # Streamlit dashboard (Stage 5)
├── analysis_pipeline.py      # Standalone script version of the notebook
├── movies.csv                 # Raw dataset
├── movies_clean.csv           # Cleaned dataset (generated)
├── requirements.txt
├── README.md
├── model/
│   ├── movie_success_model.pkl
│   ├── feature_columns.pkl
│   ├── genre_encoder.pkl
│   ├── model_metrics.json
│   └── stats_results.json
├── assets/                    # 10 EDA charts + confusion matrix + feature importance
└── images/
    └── dashboard.png
```

## Stage 0 — Problem Statement

- **Success rule:** `success = 1` if `revenue > budget`, else `0`.
- **Why it matters:** Studios use success predictions to decide whether to
  greenlight a project; investors/distributors use them to size marketing
  and acquisition budgets.
- **Objective:** Build a reproducible pipeline that cleans the data, explains
  what drives success through EDA and statistical tests, trains a classifier,
  and ships it as an interactive app.
- **Problem type:** Binary classification. Target variable: `success` (0/1).

## Stage 1 — Data Preparation

- 2,000 rows, 7 columns: `budget, revenue, popularity, runtime, vote_average, title, genres`.
- No missing values or zero budget/revenue rows in this dataset (checked and
  handled defensively in code regardless, since a $0 budget/revenue is not a
  real value and would corrupt the success ratio).
- `genres` is a stringified list of TMDB-style dicts (e.g. `[{'id': 18, 'name': 'Drama'}]`);
  181 rows contain the placeholder `"N A"` and are parsed to `Unknown`.
- Overall success rate: **80.7%** — the dataset is **imbalanced** toward
  successful movies, which is why the model uses `class_weight="balanced"`
  and why precision/recall (not just accuracy) matter for evaluation.

## Stage 2 — Exploratory Data Analysis (10 charts, in `assets/`)

1. `budget_vs_revenue.png` — Budget vs. Revenue scatter (correlation ≈ 0.76: higher budgets trend toward higher revenue, but with wide spread).
2. `genre_distribution.png` — Movie count by genre.
3. `genre_success_rate.png` — Success rate by genre.
4. `popularity_vs_success.png` — Popularity by success outcome.
5. `runtime_vs_success.png` — Runtime by success outcome.
6. `vote_average_vs_success.png` — Average vote by success outcome.
7. `correlation_heatmap.png` — Correlation heatmap of numeric features.
8. `vote_average_distribution.png` — Distribution of average vote.
9. `popularity_distribution_by_success.png` — Popularity density split by outcome.
10. `genre_roi.png` — Average revenue-to-budget ratio by genre.

## Stage 3 — Statistical Testing

- **T-Test** (popularity, successful vs. unsuccessful): t = 2.06, **p = 0.0397** →
  significant difference at α = 0.05; successful movies tend to be more popular.
- **Chi-Square** (genre vs. success): χ² = 1.77, dof = 9, **p = 0.995** → fail to
  reject H0; in this dataset genre alone is not significantly associated with success.
- A p-value is the probability of seeing a difference this extreme if there were
  truly no effect; we use the standard **α = 0.05** threshold.

## Stage 4 — Predictive Modeling (Random Forest)

- **Features:** `budget, popularity, runtime, vote_average` + one-hot encoded genre.
  `revenue` is excluded (it defines the target — using it would leak the answer);
  `title` is excluded (unique identifier, no signal).
- **Split:** 80/20 train/test, stratified on the target, so the class balance is
  preserved in both sets.
- **Model:** `RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced")`.
  A random forest trains many decision trees on bootstrapped samples/feature
  subsets and averages their votes, which reduces overfitting versus a single tree.
- **Test performance:** Accuracy 79.8%, Precision 81.5%, Recall 96.9%, F1 88.5%.
- **Confusion matrix:** the model catches most true successes (high recall) but
  still misclassifies a chunk of unsuccessful movies as successful — expected,
  given the class imbalance.
- **Top features:** popularity, budget, and vote_average are the strongest
  predictors, roughly matching the EDA/T-test finding that popularity separates
  the two classes; genre importance is low individually, consistent with the
  chi-square result.

## Stage 5 — Streamlit Dashboard

Run locally:

```bash
pip install -r requirements.txt
python analysis_pipeline.py     # generates movies_clean.csv, assets/, model/
streamlit run MovieIQ.py
```

The app has four pages (sidebar navigation): **Overview**, **Exploratory
Analysis**, **Statistical Tests**, and **Predict a Movie**, plus sidebar
filters for genre and minimum average vote that update the Overview and EDA
pages live.

### Deployment (Streamlit Community Cloud)

1. Push this folder to a public GitHub repo.
2. On [share.streamlit.io](https://share.streamlit.io), point to `MovieIQ.py`
   as the main file.
3. Make sure `movies.csv`, `analysis_pipeline.py` output (`movies_clean.csv`,
   `model/`, `assets/`) are committed, or run the pipeline as a one-off build
   step, since Streamlit Cloud's filesystem doesn't persist between deploys.
4. Add `requirements.txt` at the repo root (already included).

**Live link:** _add your deployed URL here after publishing._

## Reflection

MovieIQ gives a reasonable first-pass signal (≈80% accuracy, strong recall),
but I would trust it as a supporting data point, not a verdict, if a studio
asked "will this film succeed?" Two limitations: (1) the success label ignores
marketing spend, release timing, and franchise/IP effects, which are known to
heavily influence real box-office outcomes; (2) the class imbalance (81%
successful) means the model can look accurate while still misreading many of
the harder, unsuccessful cases. With more time and a larger real-world
dataset, I'd add engineered features (studio, release month, franchise flag,
cast/crew popularity) and try a gradient-boosted model with proper
cross-validation and threshold tuning.

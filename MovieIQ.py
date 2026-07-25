"""
MovieIQ — Predictive Analytics on Film Success
Interactive Streamlit dashboard: EDA, statistical tests, and a live Random Forest
success predictor, filterable by genre and minimum vote average.

Run locally with:  streamlit run MovieIQ.py
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MovieIQ — Predictive Analytics on Film Success",
    page_icon="🎬",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "movies_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "movie_success_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model", "feature_columns.pkl")
GENRES_PATH = os.path.join(BASE_DIR, "model", "genre_encoder.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "model_metrics.json")
STATS_PATH = os.path.join(BASE_DIR, "model", "stats_results.json")

sns.set_theme(style="darkgrid")


# ---------------------------------------------------------------------------
# DATA / MODEL LOADING (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(
            "movies_clean.csv not found. Run `python analysis_pipeline.py` once "
            "to generate cleaned data, charts, and the trained model."
        )
        st.stop()
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model_artifacts():
    missing = [p for p in [MODEL_PATH, FEATURES_PATH, GENRES_PATH] if not os.path.exists(p)]
    if missing:
        st.error(
            "Model artifacts are missing. Run `python analysis_pipeline.py` once "
            "before launching the app to train the model and save it to model/."
        )
        st.stop()
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    genre_options = joblib.load(GENRES_PATH)
    return model, feature_cols, genre_options


@st.cache_data
def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


df = load_data()
model, feature_cols, genre_options = load_model_artifacts()
model_metrics = load_json(METRICS_PATH)
stats_results = load_json(STATS_PATH)

# ---------------------------------------------------------------------------
# SIDEBAR — FILTERS
# ---------------------------------------------------------------------------
st.sidebar.title("🎬 MovieIQ")
st.sidebar.caption("Predictive Analytics on Film Success")
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

all_genres = sorted(df["genre_primary"].unique().tolist())
selected_genres = st.sidebar.multiselect(
    "Genre", options=all_genres, default=all_genres,
    help="Filter the dashboard by one or more genres."
)
min_vote = st.sidebar.slider(
    "Minimum average vote", min_value=float(df["vote_average"].min()),
    max_value=float(df["vote_average"].max()), value=float(df["vote_average"].min()),
    step=0.1,
)

filtered = df[df["genre_primary"].isin(selected_genres) & (df["vote_average"] >= min_vote)]

st.sidebar.markdown("---")
st.sidebar.metric("Movies matching filters", len(filtered))
st.sidebar.metric("Success rate (filtered)", f"{filtered['success'].mean():.1%}" if len(filtered) else "—")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Exploratory Analysis", "Statistical Tests", "Predict a Movie"],
)

# ---------------------------------------------------------------------------
# PAGE: OVERVIEW
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("🎬 MovieIQ — Predictive Analytics on Film Success")
    st.markdown(
        "A movie is labeled **successful** when its **revenue exceeds its budget**. "
        "This dashboard explores what drives success and predicts it for new movies "
        "using a trained Random Forest classifier."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Movies", len(filtered))
    c2.metric("Success Rate", f"{filtered['success'].mean():.1%}" if len(filtered) else "—")
    c3.metric("Avg. Budget", f"${filtered['budget'].mean():,.0f}" if len(filtered) else "—")
    c4.metric("Avg. Revenue", f"${filtered['revenue'].mean():,.0f}" if len(filtered) else "—")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Success by Genre (filtered)")
        genre_success = filtered.groupby("genre_primary")["success"].mean().sort_values(ascending=False)
        st.bar_chart(genre_success)
    with col2:
        st.subheader("Model Performance Snapshot")
        if model_metrics:
            m1, m2 = st.columns(2)
            m1.metric("Accuracy", f"{model_metrics.get('accuracy', 0):.1%}")
            m2.metric("Precision", f"{model_metrics.get('precision', 0):.1%}")
            m3, m4 = st.columns(2)
            m3.metric("Recall", f"{model_metrics.get('recall', 0):.1%}")
            m4.metric("F1-score", f"{model_metrics.get('f1', 0):.1%}")

    st.markdown("---")
    st.subheader("Filtered Data Sample")
    st.dataframe(
        filtered[["title", "genre_primary", "budget", "revenue", "popularity",
                  "runtime", "vote_average", "success"]].head(20),
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# PAGE: EXPLORATORY ANALYSIS
# ---------------------------------------------------------------------------
elif page == "Exploratory Analysis":
    st.title("📊 Exploratory Data Analysis")
    st.caption("Charts reflect the current sidebar filters where applicable.")

    tab1, tab2, tab3 = st.tabs(["Budget & Revenue", "Genre Trends", "Feature Relationships"])

    with tab1:
        st.subheader("Budget vs. Revenue")
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = filtered["success"].map({0: "#e15759", 1: "#59a14f"})
        ax.scatter(filtered["budget"], filtered["revenue"], c=colors, alpha=0.6, s=25)
        lims = [df["budget"].min(), df["budget"].max()]
        ax.plot(lims, lims, "--", color="gray", linewidth=1, label="Break-even line")
        ax.set_xlabel("Budget ($)")
        ax.set_ylabel("Revenue ($)")
        ax.legend()
        st.pyplot(fig)
        corr = filtered["budget"].corr(filtered["revenue"]) if len(filtered) > 1 else np.nan
        st.info(f"Correlation between budget and revenue (filtered): **{corr:.3f}**. "
                "Higher budgets tend to correlate with higher revenue, though the "
                "relationship is far from perfect.")

        st.subheader("Correlation Heatmap")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        num_cols = ["budget", "revenue", "popularity", "runtime", "vote_average", "success"]
        sns.heatmap(filtered[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, ax=ax2)
        st.pyplot(fig2)

    with tab2:
        st.subheader("Movie Count by Genre")
        genre_counts = filtered["genre_primary"].value_counts()
        st.bar_chart(genre_counts)

        st.subheader("Success Rate by Genre")
        genre_success = filtered.groupby("genre_primary")["success"].mean().sort_values(ascending=False)
        st.bar_chart(genre_success)

        st.subheader("Average Revenue-to-Budget Ratio by Genre")
        filtered_roi = filtered.copy()
        filtered_roi["roi"] = filtered_roi["revenue"] / filtered_roi["budget"]
        genre_roi = filtered_roi.groupby("genre_primary")["roi"].mean().sort_values(ascending=False)
        st.bar_chart(genre_roi)

    with tab3:
        st.subheader("Popularity, Runtime & Vote Average vs. Success")
        feat = st.selectbox("Choose a feature", ["popularity", "runtime", "vote_average"])
        fig3, ax3 = plt.subplots(figsize=(7, 4.5))
        sns.boxplot(data=filtered, x="success", y=feat, hue="success",
                    palette={0: "#e15759", 1: "#59a14f"}, legend=False, ax=ax3)
        ax3.set_xticks([0, 1])
        ax3.set_xticklabels(["Not Successful", "Successful"])
        st.pyplot(fig3)

        means = filtered.groupby("success")[feat].mean()
        st.caption(
            f"Mean {feat} — Not Successful: {means.get(0, float('nan')):.2f} | "
            f"Successful: {means.get(1, float('nan')):.2f}"
        )

        st.subheader("Distribution of Average Vote")
        fig4, ax4 = plt.subplots(figsize=(7, 4))
        sns.histplot(filtered["vote_average"], bins=25, kde=True, color="#4e79a7", ax=ax4)
        st.pyplot(fig4)

# ---------------------------------------------------------------------------
# PAGE: STATISTICAL TESTS
# ---------------------------------------------------------------------------
elif page == "Statistical Tests":
    st.title("🧪 Statistical Testing")
    st.markdown(
        "Results below are computed once on the full cleaned dataset "
        "(precomputed during the pipeline run) and shown alongside a live "
        "re-run on your currently filtered data."
    )

    st.subheader("1. T-Test — Popularity by Success")
    if stats_results.get("t_test"):
        tt = stats_results["t_test"]
        st.write(f"**H0:** mean popularity is equal for successful and unsuccessful movies.")
        st.write(f"**t-statistic:** {tt['statistic']:.4f}  |  **p-value:** {tt['p_value']:.6f}")
        verdict = "significant difference — reject H0" if tt["significant"] else "no significant difference — fail to reject H0"
        st.success(f"Conclusion: {verdict} at α = {tt['alpha']}.") if tt["significant"] else st.warning(f"Conclusion: {verdict} at α = {tt['alpha']}.")

    st.markdown("Live re-run on filtered data:")
    if filtered["success"].nunique() == 2:
        pop_s = filtered.loc[filtered["success"] == 1, "popularity"]
        pop_f = filtered.loc[filtered["success"] == 0, "popularity"]
        t_stat, t_p = stats.ttest_ind(pop_s, pop_f, equal_var=False)
        st.write(f"t = {t_stat:.4f}, p = {t_p:.6f} "
                 f"({'significant' if t_p < 0.05 else 'not significant'} at α = 0.05)")
    else:
        st.caption("Need both success and failure cases in the filtered data to run this test.")

    st.markdown("---")
    st.subheader("2. Chi-Square Test — Genre vs. Success")
    if stats_results.get("chi_square"):
        cs = stats_results["chi_square"]
        st.write(f"**H0:** genre and success are statistically independent.")
        st.write(f"**chi2:** {cs['statistic']:.4f}  |  **dof:** {cs['dof']}  |  **p-value:** {cs['p_value']:.6f}")
        verdict = "genre is associated with success — reject H0" if cs["significant"] else "no significant association — fail to reject H0"
        st.success(f"Conclusion: {verdict} at α = {cs['alpha']}.") if cs["significant"] else st.warning(f"Conclusion: {verdict} at α = {cs['alpha']}.")

    st.markdown("---")
    st.subheader("What does a p-value mean?")
    st.markdown(
        "A p-value is the probability of observing a difference at least as extreme as the "
        "one in the data, **assuming the null hypothesis is true**. A small p-value "
        "(conventionally **below 0.05**) suggests the observed pattern is unlikely to be "
        "due to chance alone, so we reject the null hypothesis in favor of a real effect. "
        "We use the standard 0.05 threshold here as a common convention balancing false "
        "positives and false negatives."
    )

# ---------------------------------------------------------------------------
# PAGE: PREDICT A MOVIE
# ---------------------------------------------------------------------------
elif page == "Predict a Movie":
    st.title("🔮 Predict a Movie's Success")
    st.markdown("Enter a hypothetical movie's details to get a live prediction from the trained Random Forest model.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            budget = st.number_input("Budget ($)", min_value=100_000, max_value=500_000_000,
                                      value=100_000_000, step=1_000_000)
            popularity = st.slider("Popularity", 0.0, 100.0, 50.0)
            runtime = st.slider("Runtime (minutes)", 60, 220, 120)
        with col2:
            vote_average = st.slider("Expected average vote", 0.0, 10.0, 6.0)
            genre = st.selectbox("Primary genre", genre_options)

        submitted = st.form_submit_button("Predict Success")

    if submitted:
        row = pd.DataFrame([{
            "budget": budget, "popularity": popularity,
            "runtime": runtime, "vote_average": vote_average,
        }])
        for g in genre_options:
            row[f"genre_{g}"] = 1 if g == genre else 0
        row = row.reindex(columns=feature_cols, fill_value=0)

        pred = model.predict(row)[0]
        proba = model.predict_proba(row)[0][1]

        if pred == 1:
            st.success(f"✅ Predicted: **Successful** (confidence: {proba:.1%})")
        else:
            st.error(f"❌ Predicted: **Not Successful** (confidence: {1 - proba:.1%})")

        st.progress(min(max(proba, 0.0), 1.0))
        st.caption(
            "Confidence reflects the share of decision trees in the forest that "
            "voted for the 'successful' class."
        )

st.sidebar.markdown("---")
st.sidebar.caption("MovieIQ · Built with Streamlit, scikit-learn & seaborn")

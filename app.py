"""
Bankruptcy Risk Lab - Streamlit app.

What this app does:
  1. The user uploads the test CSV (features + 'Bankrupt?' column).
  2. All 5 pre-trained models (saved as .pkl files) predict on that data.
     NOTHING is retrained here.
  3. We show metrics, confusion matrix, ROC curve and a model comparison.
"""
import io
import os

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from evaluation_utils import TARGET_COLUMN_NAME, calculate_evaluation_metrics, clean_column_names

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
APP_FOLDER = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_FOLDER = os.path.join(APP_FOLDER, "model")
CSS_FILE_PATH = os.path.join(APP_FOLDER, "styles.css")

# Dropdown name  ->  file saved by train_models.py
SAVED_MODEL_FILE_NAMES = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "Decision Tree": "Decision_Tree.pkl",
    "K-Nearest Neighbors": "K-Nearest_Neighbors.pkl",
    "Naive Bayes (Gaussian)": "Naive_Bayes_Gaussian.pkl",
    "Support Vector Machine": "Support_Vector_Machine.pkl",
}

# Short plain-English explanation shown under each metric card
METRIC_EXPLANATIONS = {
    "Accuracy": "Correct predictions overall",
    "AUC Score": "Ranking power (1 = perfect)",
    "Precision": "Flagged firms truly bankrupt",
    "Recall": "Bankrupt firms caught",
    "F1 Score": "Precision / recall balance",
    "MCC Score": "Fair score for imbalance",
}

CHART_COLORS_PER_MODEL = ["#8B5CF6", "#22D3EE", "#F5C26B", "#F472B6", "#34D399"]

st.set_page_config(page_title="Bankruptcy Risk Lab", page_icon="📉", layout="wide")


def load_custom_css() -> None:
    """Read styles.css and apply it to the page."""
    with open(CSS_FILE_PATH) as css_file:
        st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)


load_custom_css()


# ----------------------------------------------------------------------------
# Loading saved models and scoring the uploaded data
# ----------------------------------------------------------------------------
@st.cache_resource
def load_saved_model(model_name: str):
    """Load one trained model (scaler + classifier) from its .pkl file."""
    model_file_path = os.path.join(SAVED_MODELS_FOLDER, SAVED_MODEL_FILE_NAMES[model_name])
    return joblib.load(model_file_path)


@st.cache_resource
def load_required_feature_columns() -> list:
    """The exact list of 94 feature columns the models were trained on."""
    return joblib.load(os.path.join(SAVED_MODELS_FOLDER, "feature_columns.pkl"))


@st.cache_data(show_spinner="Scoring all models on your test data...")
def score_uploaded_data(uploaded_file_bytes: bytes) -> dict:
    """
    Run every saved model on the uploaded CSV and calculate its metrics.
    Result is cached, so switching models in the dropdown is instant.
    """
    test_dataframe = clean_column_names(pd.read_csv(io.BytesIO(uploaded_file_bytes)))

    # Make sure the uploaded file has every column the models need
    required_feature_columns = load_required_feature_columns()
    required_columns = required_feature_columns + [TARGET_COLUMN_NAME]
    missing_columns = [column for column in required_columns if column not in test_dataframe.columns]
    if missing_columns:
        return {"missing_columns": missing_columns}

    test_features = test_dataframe[required_feature_columns]
    true_labels = test_dataframe[TARGET_COLUMN_NAME].astype(int).to_numpy()

    results_by_model = {}
    for model_name in SAVED_MODEL_FILE_NAMES:
        model = load_saved_model(model_name)
        predicted_labels = model.predict(test_features)                       # 0 or 1
        predicted_probabilities = model.predict_proba(test_features)[:, 1]    # chance of bankruptcy
        results_by_model[model_name] = {
            "predicted_labels": predicted_labels,
            "predicted_probabilities": predicted_probabilities,
            "metrics": calculate_evaluation_metrics(true_labels, predicted_labels, predicted_probabilities),
        }

    return {
        "missing_columns": [],
        "true_labels": true_labels,
        "results_by_model": results_by_model,
        "first_ten_rows": test_dataframe.head(10),
        "number_of_rows": len(test_dataframe),
    }


# ----------------------------------------------------------------------------
# Small helpers that build the HTML for the custom-looking parts of the page
# ----------------------------------------------------------------------------
def build_metric_cards_html(metrics: dict) -> str:
    """One card per metric: name, big number, progress bar, short explanation."""
    all_cards_html = ""
    for metric_name, metric_value in metrics.items():
        metric_is_available = not np.isnan(metric_value)
        text_to_show = f"{metric_value:.3f}" if metric_is_available else "N/A"
        bar_width_percent = max(0, min(1, metric_value)) * 100 if metric_is_available else 0

        all_cards_html += (
            f'<div class="kpi">'
            f'<div class="kpi-label">{metric_name}</div>'
            f'<div class="kpi-value">{text_to_show}</div>'
            f'<div class="kpi-bar"><span style="width:{bar_width_percent:.0f}%"></span></div>'
            f'<div class="kpi-help">{METRIC_EXPLANATIONS[metric_name]}</div>'
            f'</div>'
        )
    return f'<div class="kpi-grid">{all_cards_html}</div>'


def build_confusion_cell_html(count: int, total_in_row: int, color_class: str, caption: str) -> str:
    """One box of the confusion matrix, e.g. '36 · Caught · 81.8%'."""
    share_of_row = count / total_in_row
    return (f'<div class="cm-cell {color_class}"><b>{count}</b>'
            f'<small>{caption} · {share_of_row:.1%}</small></div>')


def build_confusion_matrix_html(true_labels, predicted_labels) -> str:
    """2x2 grid: green boxes are correct predictions, pink boxes are mistakes."""
    (true_negatives, false_positives), (false_negatives, true_positives) = confusion_matrix(
        true_labels, predicted_labels, labels=[0, 1]
    )
    total_healthy = max(true_negatives + false_positives, 1)      # max(...,1) avoids dividing by zero
    total_bankrupt = max(false_negatives + true_positives, 1)

    return (
        '<div class="cm-grid">'
        '<div></div><div class="cm-head">Predicted<br>Healthy</div><div class="cm-head">Predicted<br>Bankrupt</div>'
        '<div class="cm-head">Actually<br>Healthy</div>'
        + build_confusion_cell_html(true_negatives, total_healthy, "cm-good", "Correct")
        + build_confusion_cell_html(false_positives, total_healthy, "cm-bad", "False alarm")
        + '<div class="cm-head">Actually<br>Bankrupt</div>'
        + build_confusion_cell_html(false_negatives, total_bankrupt, "cm-bad", "Missed")
        + build_confusion_cell_html(true_positives, total_bankrupt, "cm-good", "Caught")
        + '</div>'
    )


def build_comparison_table_html(metrics_of_all_models: pd.DataFrame) -> str:
    """Leaderboard table: models ranked by MCC, best value in each column highlighted."""
    ranked_models = metrics_of_all_models.sort_values("MCC Score", ascending=False)
    best_value_per_metric = ranked_models.max()

    header_html = "<tr><th>Model</th>" + "".join(f"<th>{metric_name}</th>" for metric_name in ranked_models.columns) + "</tr>"

    body_html = ""
    for rank_position, (model_name, model_metrics) in enumerate(ranked_models.iterrows()):
        best_badge_html = '<span class="badge">Best MCC</span>' if rank_position == 0 else ""
        metric_cells_html = ""
        for metric_name in ranked_models.columns:
            is_best_in_column = model_metrics[metric_name] == best_value_per_metric[metric_name]
            css_class = "best" if is_best_in_column else ""
            metric_cells_html += f'<td class="{css_class}">{model_metrics[metric_name]:.3f}</td>'
        body_html += f"<tr><td>{model_name}{best_badge_html}</td>{metric_cells_html}</tr>"

    return f'<table class="cmp-table">{header_html}{body_html}</table>'


def apply_dark_chart_style(chart: alt.Chart) -> alt.Chart:
    """Make Altair charts match the dark theme (transparent background, soft grid)."""
    return (chart.configure_view(strokeWidth=0)
            .configure(background="transparent")
            .configure_axis(labelColor="#9AA4C7", titleColor="#9AA4C7", gridColor="#1E2748", domainColor="#2A3560")
            .configure_legend(labelColor="#C9D0EE", titleColor="#9AA4C7"))


def show_section_title(title_text: str) -> None:
    st.markdown(f'<div class="section-title">{title_text}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Sidebar: upload + model choice
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Control panel")
    uploaded_file = st.file_uploader(
        "Upload test data (CSV)", type=["csv"],
        help="Use test_data.csv: all 94 features plus the 'Bankrupt?' column.",
    )
    selected_model_name = st.selectbox("Select a model", list(SAVED_MODEL_FILE_NAMES.keys()))
    st.divider()
    st.caption("Models are pre-trained and loaded from disk. Your upload is only scored, never used for training.")
    st.caption("Dataset: Taiwanese Bankruptcy Prediction (UCI) · 6,819 companies · 3.2% bankrupt")

# ----------------------------------------------------------------------------
# Page header
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>Bankruptcy Risk Lab</h1>
  <p>Compare five machine-learning classifiers on real company financials and see which one spots bankruptcy risk best.</p>
  <span class="pill">5 models</span><span class="pill">94 financial ratios</span>
  <span class="pill">Imbalanced data (3.2%)</span><span class="pill">6 evaluation metrics</span>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Before upload: show instructions and stop
# ----------------------------------------------------------------------------
if uploaded_file is None:
    step_one_column, step_two_column, step_three_column = st.columns(3)
    step_one_column.markdown('<div class="step"><div class="num">1</div><h4>Upload test data</h4><p>Drop <b>test_data.csv</b> in the sidebar. It must include all features and the Bankrupt? column.</p></div>', unsafe_allow_html=True)
    step_two_column.markdown('<div class="step"><div class="num">2</div><h4>Pick a model</h4><p>Choose Logistic Regression, Decision Tree, KNN, Naive Bayes or SVM.</p></div>', unsafe_allow_html=True)
    step_three_column.markdown('<div class="step"><div class="num">3</div><h4>Explore results</h4><p>See metrics, confusion matrix, ROC curve and a ranked comparison of all models.</p></div>', unsafe_allow_html=True)
    st.stop()

# ----------------------------------------------------------------------------
# After upload: score the data and validate it
# ----------------------------------------------------------------------------
scoring_output = score_uploaded_data(uploaded_file.getvalue())

if scoring_output["missing_columns"]:
    missing_columns = scoring_output["missing_columns"]
    st.error(f"The uploaded file is missing {len(missing_columns)} required column(s), for example: {missing_columns[:5]}")
    st.stop()

true_labels = scoring_output["true_labels"]
results_by_model = scoring_output["results_by_model"]
selected_model_result = results_by_model[selected_model_name]

number_of_companies = scoring_output["number_of_rows"]
number_of_bankrupt_companies = int(true_labels.sum())
number_of_healthy_companies = number_of_companies - number_of_bankrupt_companies
st.success(f"Loaded {number_of_companies:,} companies · {number_of_bankrupt_companies} bankrupt · {number_of_healthy_companies:,} healthy")

# One table with a row per model and a column per metric
metrics_of_all_models = pd.DataFrame(
    {model_name: model_result["metrics"] for model_name, model_result in results_by_model.items()}
).transpose()

# One-click comparison panel, visible above all tabs
with st.expander("📊  Compare all 5 models at a glance (click to open)"):
    st.markdown(build_comparison_table_html(metrics_of_all_models), unsafe_allow_html=True)
    st.caption("Ranked by MCC. The highlighted value in each column is the best of the five models.")

performance_tab, comparison_tab, data_tab = st.tabs(["Model performance", "Compare all models", "Data & notes"])

# ----------------------------------------------------------------------------
# Tab 1: results for the selected model
# ----------------------------------------------------------------------------
with performance_tab:
    show_section_title(f"{selected_model_name}: evaluation metrics")
    st.markdown(build_metric_cards_html(selected_model_result["metrics"]), unsafe_allow_html=True)

    (_, false_positives), (false_negatives, true_positives) = confusion_matrix(
        true_labels, selected_model_result["predicted_labels"], labels=[0, 1]
    )
    st.markdown(
        f'<div class="insight">💡 <b>{selected_model_name}</b> caught <b>{true_positives} of {number_of_bankrupt_companies}</b> '
        f'bankrupt companies and missed {false_negatives}, while raising <b>{false_positives}</b> false alarms.</div>',
        unsafe_allow_html=True,
    )

    confusion_matrix_column, roc_curve_column = st.columns(2, gap="large")

    with confusion_matrix_column:
        show_section_title("Confusion matrix")
        st.markdown(build_confusion_matrix_html(true_labels, selected_model_result["predicted_labels"]), unsafe_allow_html=True)

    with roc_curve_column:
        show_section_title("ROC curve")
        false_positive_rates, true_positive_rates, _ = roc_curve(true_labels, selected_model_result["predicted_probabilities"])
        roc_points = pd.DataFrame({"False positive rate": false_positive_rates, "True positive rate": true_positive_rates})
        random_guess_line = pd.DataFrame({"False positive rate": [0, 1], "True positive rate": [0, 1]})

        roc_curve_chart = (
            alt.Chart(roc_points).mark_area(line={"color": "#22D3EE", "strokeWidth": 3}, color="#22D3EE", opacity=0.15)
            .encode(x=alt.X("False positive rate:Q", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("True positive rate:Q", scale=alt.Scale(domain=[0, 1])))
            + alt.Chart(random_guess_line).mark_line(strokeDash=[6, 6], color="#6B7699")
            .encode(x="False positive rate:Q", y="True positive rate:Q")
        ).properties(height=300)
        st.altair_chart(apply_dark_chart_style(roc_curve_chart), width="stretch")

    show_section_title("Classification report")
    report_table = pd.DataFrame(classification_report(
        true_labels, selected_model_result["predicted_labels"],
        target_names=["Healthy", "Bankrupt"], output_dict=True, zero_division=0,
    )).transpose().round(3)
    st.dataframe(report_table, width="stretch")

# ----------------------------------------------------------------------------
# Tab 2: all models side by side
# ----------------------------------------------------------------------------
with comparison_tab:
    show_section_title("Leaderboard (ranked by MCC, best value per metric highlighted)")
    st.markdown(build_comparison_table_html(metrics_of_all_models), unsafe_allow_html=True)

    show_section_title("Metric-by-metric comparison")
    # Bar charts need "long" data: one row per (model, metric) pair
    metrics_for_bar_chart = (metrics_of_all_models.reset_index()
                             .rename(columns={"index": "Model"})
                             .melt(id_vars="Model", var_name="Metric", value_name="Score"))
    metric_comparison_chart = (
        alt.Chart(metrics_for_bar_chart).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(x=alt.X("Metric:N", axis=alt.Axis(labelAngle=0), title=None),
                xOffset="Model:N",
                y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Model:N", scale=alt.Scale(range=CHART_COLORS_PER_MODEL)),
                tooltip=["Model", "Metric", alt.Tooltip("Score:Q", format=".3f")])
        .properties(height=380)
    )
    st.altair_chart(apply_dark_chart_style(metric_comparison_chart), width="stretch")

    st.markdown('<div class="insight">📌 Only about 3% of companies are bankrupt, so <b>accuracy alone is misleading</b>: '
                'a model that always says "healthy" would score ~97%. Use <b>MCC, AUC and Recall</b> to judge models fairly.</div>',
                unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Tab 3: data preview and notes
# ----------------------------------------------------------------------------
with data_tab:
    show_section_title("Preview of uploaded data (first 10 rows)")
    st.dataframe(scoring_output["first_ten_rows"], width="stretch")

    show_section_title("About the data")
    st.markdown("- **Source:** Taiwan Economic Journal, 1999-2009, via the UCI Machine Learning Repository.\n"
                "- **Features:** 94 financial ratios (profitability, leverage, liquidity, cash flow).\n"
                "- **Target:** `Bankrupt?` where 1 means the company went bankrupt.\n"
                "- **Pipeline:** every model is a StandardScaler plus classifier saved with joblib; no retraining happens in this app.")

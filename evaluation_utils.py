"""Shared evaluation helpers used by both train_models.py and app.py."""
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

TARGET_COLUMN_NAME = "Bankrupt?"


def clean_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """The UCI file has leading/trailing spaces in column names; remove them."""
    cleaned_dataframe = dataframe.copy()
    cleaned_dataframe.columns = [column_name.strip() for column_name in cleaned_dataframe.columns]
    return cleaned_dataframe


def calculate_evaluation_metrics(true_labels, predicted_labels, predicted_probabilities) -> dict:
    """Return the 6 required metrics for a binary classifier."""
    try:
        auc_score = roc_auc_score(true_labels, predicted_probabilities)
    except ValueError:
        # Happens if the uploaded test data contains only one class
        auc_score = float("nan")

    return {
        "Accuracy": accuracy_score(true_labels, predicted_labels),
        "AUC Score": auc_score,
        "Precision": precision_score(true_labels, predicted_labels, zero_division=0),
        "Recall": recall_score(true_labels, predicted_labels, zero_division=0),
        "F1 Score": f1_score(true_labels, predicted_labels, zero_division=0),
        "MCC Score": matthews_corrcoef(true_labels, predicted_labels),
    }

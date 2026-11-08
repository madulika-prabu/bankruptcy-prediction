"""
Run this ONCE on your own computer:  python train_models.py

It downloads the Taiwanese Bankruptcy dataset, trains 5 classifiers,
evaluates them, and saves everything the Streamlit app needs.
"""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from evaluation_utils import (
    TARGET_COLUMN_NAME,
    clean_column_names,
    calculate_evaluation_metrics,
)

RANDOM_SEED = 42
TEST_SIZE_FRACTION = 0.20
MODEL_FOLDER = "model"
LOCAL_DATA_FILE = "data.csv"


def load_bankruptcy_dataset() -> pd.DataFrame:
    """Use a local data.csv if present, otherwise download from UCI."""
    if os.path.exists(LOCAL_DATA_FILE):
        full_dataframe = pd.read_csv(LOCAL_DATA_FILE)
    else:
        from ucimlrepo import fetch_ucirepo

        taiwanese_bankruptcy_prediction = fetch_ucirepo(id=572)
        features_dataframe = taiwanese_bankruptcy_prediction.data.features
        target_dataframe = taiwanese_bankruptcy_prediction.data.targets
        full_dataframe = pd.concat([features_dataframe, target_dataframe], axis=1)

    full_dataframe = clean_column_names(full_dataframe)

    # Drop columns that never change (e.g. "Net Income Flag") - they carry no information
    constant_columns = [
        column for column in full_dataframe.columns
        if column != TARGET_COLUMN_NAME and full_dataframe[column].nunique() <= 1
    ]
    return full_dataframe.drop(columns=constant_columns)


def build_models() -> dict:
    """Each model is a Pipeline (scaler + classifier) so scaling is saved with the model."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)),
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=RANDOM_SEED)),
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Naive Bayes (Gaussian)": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", GaussianNB()),
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_SEED)),
        ]),
    }


def main():
    os.makedirs(MODEL_FOLDER, exist_ok=True)

    bankruptcy_dataframe = load_bankruptcy_dataset()
    print(f"Dataset shape: {bankruptcy_dataframe.shape}")
    print(bankruptcy_dataframe[TARGET_COLUMN_NAME].value_counts())

    feature_columns = [column_name for column_name in bankruptcy_dataframe.columns if column_name != TARGET_COLUMN_NAME]
    features_all = bankruptcy_dataframe[feature_columns]
    target_all = bankruptcy_dataframe[TARGET_COLUMN_NAME]

    # stratify keeps the rare "bankrupt" class present in both train and test
    features_train, features_test, target_train, target_test = train_test_split(
        features_all, target_all,
        test_size=TEST_SIZE_FRACTION, stratify=target_all, random_state=RANDOM_SEED,
    )

    # Save the test data: this is the file you upload in the Streamlit app
    test_dataframe = features_test.copy()
    test_dataframe[TARGET_COLUMN_NAME] = target_test
    test_dataframe.to_csv("test_data.csv", index=False)

    joblib.dump(feature_columns, os.path.join(MODEL_FOLDER, "feature_columns.pkl"))

    evaluation_rows = []
    for model_name, model_pipeline in build_models().items():
        print(f"Training {model_name} ...")
        model_pipeline.fit(features_train, target_train)

        predicted_labels = model_pipeline.predict(features_test)
        predicted_probabilities = model_pipeline.predict_proba(features_test)[:, 1]

        metrics = calculate_evaluation_metrics(target_test, predicted_labels, predicted_probabilities)
        evaluation_rows.append({"Model": model_name, **metrics})

        safe_file_name = model_name.replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"
        joblib.dump(model_pipeline, os.path.join(MODEL_FOLDER, safe_file_name))

    model_evaluation_results = pd.DataFrame(evaluation_rows).round(4)
    model_evaluation_results.to_csv("metrics_summary.csv", index=False)
    print("\n", model_evaluation_results.to_string(index=False))


if __name__ == "__main__":
    main()

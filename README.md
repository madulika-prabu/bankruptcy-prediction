# Bankruptcy Risk Lab: Taiwanese Company Bankruptcy Prediction

An end-to-end machine learning project: 5 classification models trained on real company financial data, evaluated with 6 metrics, and demonstrated in an interactive Streamlit web app.

**Live app:** `https://bankruptcy-prediction-taiwan.streamlit.app`
**GitHub repository:** `https://github.com/madulika-prabu/bankruptcy-prediction`

---

## 1. Problem Statement
Predict whether a company will go **bankrupt (1)** or stay **healthy (0)** using its financial ratios. This is a **binary classification** problem with a **heavily imbalanced target**: only 3.2% of companies are bankrupt.

## 2. Dataset
| Item | Details |
|---|---|
| Name | Taiwanese Bankruptcy Prediction |
| Source | UCI Machine Learning Repository (id 572), https://doi.org/10.24432/C5004D |
| Collected by | Taiwan Economic Journal, 1999-2009 |
| Instances | 6,819 companies (220 bankrupt, 6,599 healthy) |
| Features | 94 numeric financial ratios (95 originally; the constant `Net Income Flag` column was dropped) |
| Target | `Bankrupt?` (1 = bankrupt, 0 = healthy) |
| Missing values | None |

**Split:** 80% training (5,455 companies) / 20% test (1,364 companies, 44 bankrupt), stratified so both parts keep the same bankruptcy rate. `random_state=42`.

## 3. Models Used
All five models are scikit-learn pipelines (`StandardScaler` + classifier) trained on the same data:

1. Logistic Regression (`class_weight="balanced"`)
2. Decision Tree (`max_depth=5`, `class_weight="balanced"`)
3. K-Nearest Neighbors (`n_neighbors=5`)
4. Gaussian Naive Bayes
5. Support Vector Machine (RBF kernel, `class_weight="balanced"`)

### Evaluation results (on the 1,364-company test set)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8783 | 0.9171 | 0.1856 | 0.8182 | 0.3025 | 0.3533 |
| Decision Tree | 0.9025 | 0.8331 | 0.2129 | 0.7500 | 0.3317 | 0.3661 |
| KNN | 0.9707 | 0.7401 | 0.6429 | 0.2045 | 0.3103 | 0.3519 |
| Naive Bayes | 0.2903 | 0.7481 | 0.0380 | 0.8636 | 0.0728 | 0.0539 |
| Support Vector Machine | 0.8996 | 0.9174 | 0.1879 | 0.6364 | 0.2902 | 0.3085 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Caught 36 of 44 bankrupt companies (recall 0.82) with an AUC of 0.917, among the best. The cost is 158 false alarms, so precision is low (0.19). Strong, stable, and the most useful when missing a bankruptcy is costly. |
| Decision Tree | Highest MCC (0.366) and F1 (0.332). Caught 33 of 44 with 122 false alarms. Its AUC (0.833) is clearly lower than Logistic Regression and SVM, so its ranking of risk is less reliable. |
| KNN | Highest accuracy (0.971) but misleading: it caught only 9 of 44 bankrupt companies (recall 0.20). Its precision is the best (0.64, only 5 false alarms), but it misses most bankruptcies. Without class weighting it favours the majority class. |
| Naive Bayes | Worst overall (MCC 0.054, accuracy 0.29). It flagged 962 healthy companies as bankrupt. Its independence assumption fails because the 94 financial ratios are highly correlated. High recall (0.86) here is just over-predicting bankruptcy. |
| Support Vector Machine | Best AUC (0.917, tied with Logistic Regression) but lower recall (0.64, 28 of 44 caught) at the 0.5 threshold. It ranks companies well, but its default cut-off misses more bankruptcies than Logistic Regression. |
| **Overall winner for this dataset** | **Logistic Regression.** It has top-tier AUC (0.917), the highest recall among the usable models (0.82), and an MCC (0.353) within 0.013 of the best. It is also simple and interpretable. Decision Tree is the runner-up. Note: accuracy is misleading here (an "always healthy" model scores about 97%), so MCC, AUC and recall were used to decide. With only 44 bankrupt companies in the test set, each missed company moves recall by about 2.3 points, so small differences between models should be read with caution. |

## 4. Streamlit App Features
- **Dataset upload (CSV):** upload the test data only (`test_data.csv`)
- **Model selection dropdown:** choose any of the 5 models
- **Evaluation metrics:** Accuracy, AUC, Precision, Recall, F1, MCC
- **Confusion matrix, ROC curve and classification report**
- **Comparison of all models:** one-click leaderboard on the test data

The app **loads the saved models and does not retrain**. The uploaded CSV must contain all 94 feature columns and the `Bankrupt?` column.

## 5. Repository Structure
```
├── app.py                  # Streamlit app
├── styles.css              # App styling
├── evaluation_utils.py     # Metric calculation shared by training and app
├── train_models.py         # Trains and saves the 5 models, creates test_data.csv
├── requirements.txt
├── README.md
├── test_data.csv           # Test data for the app (1,364 rows)
├── metrics_summary.csv     # Metrics recorded at training time
├── .streamlit/config.toml  # App theme
└── model/                  # Saved trained models
    ├── Logistic_Regression.pkl
    ├── Decision_Tree.pkl
    ├── K-Nearest_Neighbors.pkl
    ├── Naive_Bayes_Gaussian.pkl
    ├── Support_Vector_Machine.pkl
    └── feature_columns.pkl # Feature list used to validate uploads
```

## 6. Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then upload `test_data.csv` in the sidebar.

**To retrain** (optional): place the original `data.csv` next to `train_models.py` and run `python train_models.py`. It recreates `model/`, `test_data.csv` and `metrics_summary.csv`.

## 7. Deployment
Deployed on Streamlit Community Cloud from this repository (branch `main`, main file `app.py`).

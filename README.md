# Credit Card Default Risk Model Validation

## Project Overview

This project investigates the ability of machine learning models to predict credit card payment defaults using customer demographics, credit limits, repayment history, billing statements, and previous payment behavior.

The objective was not simply to maximize accuracy, but to evaluate how different models perform, whether they generalize to unseen data, and how prediction errors affect potential credit-risk decisions.

The project focuses on model comparison, overfitting, cross-validation, classification thresholds, and the trade-off between precision and recall.

## Dataset

**Source:** [UCI Machine Learning Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

- **Observations:** 30,000 credit card clients
- **Predictive features:** 23
- **Target:** Default payment (1 = default, 0 = no default)
- **Class distribution:** Approximately 78% non-default and 22% default

The dataset contains customer financial and repayment information.

Because the target distribution is imbalanced, accuracy alone is not sufficient to evaluate model performance.

## Tools and Technologies

- Python
- pandas and NumPy
- scikit-learn
- Matplotlib
- Jupyter Notebook
- Git and GitHub
- Visual Studio Code

## Methodology

### 1. Data Preparation

The dataset was inspected for missing values, target distribution, feature types, and categorical inconsistencies.

Customer identifiers were excluded from predictive features.

The dataset was divided using stratified sampling:

| Dataset | Percentage | Customers |
|---|---:|---:|
| Training | 70% | 21,000 |
| Validation | 15% | 4,500 |
| Test | 15% | 4,500 |

Categorical variables were encoded, numerical features were standardized, and repayment-status features were retained as ordered values.

Preprocessing was fitted on training data to avoid leakage.

### 2. Baseline and Model Training

A majority-class baseline was established by predicting no default for every customer.

Three classification approaches were investigated:

- Logistic Regression
- Decision Tree
- Random Forest

An unrestricted Decision Tree achieved nearly perfect training accuracy but substantially worse validation performance, demonstrating severe overfitting.

Restricting the tree's maximum depth improved generalization.

### 3. Validation Model Comparison

| Model | Accuracy | Precision | Recall |
|---|---:|---:|---:|
| Logistic Regression | 80.89% | 69.45% | 24.22% |
| Decision Tree (depth 5) | 81.82% | 67.32% | 34.57% |
| Random Forest | 81.27% | 63.38% | 36.18% |

The Random Forest achieved the highest recall among the three models at their default prediction rules, but at the expense of precision.

This comparison demonstrated that choosing a model requires evaluating the consequences of different types of errors, rather than relying on accuracy alone.

### 4. Cross-Validation and Stability

Stratified five-fold cross-validation was performed on the training dataset, with preprocessing included inside each fold's pipeline to prevent data leakage.

| Model | Mean Recall | Recall Standard Deviation |
|---|---:|---:|
| Decision Tree (depth 5) | 36.58% | 1.64 percentage points |
| Random Forest | 37.70% | 2.19 percentage points |

Both models showed relatively consistent recall across folds, although their default-detection performance remained limited.

### 5. Threshold Analysis

Random Forest decision thresholds of 0.50, 0.40, and 0.30 were evaluated on the validation dataset.

| Threshold | Precision | Recall |
|---|---:|---:|
| 0.50 | 63.40% | 37.09% |
| 0.40 | 56.52% | 45.73% |
| 0.30 | 49.20% | 55.78% |

Lowering the threshold increased the number of actual defaults identified, but also increased false positives.

A hypothetical cost analysis and a manual-review capacity scenario were used to explore how business requirements could affect threshold selection.

These assumptions were illustrative and were not derived from real banking operations.

## Final Test Results

A Random Forest using its default prediction rule was selected for final evaluation on the previously untouched test set.

| Metric | Final Test Result |
|---|---:|
| Accuracy | 81.36% |
| Precision | 63.51% |
| Recall | 37.05% |
| True Negatives | 3,292 |
| False Positives | 212 |
| False Negatives | 627 |
| True Positives | 369 |

Validation and test performance were similar, providing evidence of reasonably consistent generalization within this dataset.

However, the model missed 627 of the 996 customers who actually defaulted.

This highlights a central finding: **A model can achieve relatively high accuracy while failing to identify a substantial proportion of the minority class.**

## Key Findings

1. **Accuracy can be misleading.** The majority-class baseline achieved approximately 78% accuracy while identifying no defaults.
2. **Model complexity affects generalization.** The unrestricted Decision Tree overfit severely, while limiting tree depth reduced the training-validation performance gap.
3. **Cross-validation provides additional stability evidence.** Both evaluated models showed relatively consistent recall across training folds.
4. **Classification thresholds affect business trade-offs.** Lower thresholds increased recall but generated more false positives and review workload.
5. **Generalization does not guarantee deployment readiness.** The selected Random Forest performed similarly on validation and test data, but its recall remained limited.

## Limitations and Future Work

The models are not presented as suitable for real-world credit approval decisions.

Important limitations include:

- Low recall at the selected operating point
- Historical dataset and potential changes in customer behavior over time
- Hypothetical rather than measured business costs
- No completed probability calibration or fairness assessment
- No external or out-of-time validation

Potential future improvements include threshold optimization under realistic operating constraints, probability calibration, additional model tuning, and evaluation on newer or independent datasets.

## Project Status

**Completed — Initial Model Validation Study**

The project includes exploratory data analysis, preprocessing, model comparison, cross-validation, threshold analysis, and independent test evaluation.

The main analysis is available in `notebooks/01_eda.ipynb`.

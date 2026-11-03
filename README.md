# IT Customer Churn Prediction - ML Classification & Streamlit Deployment

## a. Problem Statement

Customer churn (customers leaving the service) is one of the costliest problems for IT / telecom
service providers: acquiring a new customer is far more expensive than retaining an existing one.
The goal of this project is to build and compare **five machine-learning classification models**
that predict, from a customer's profile, services and billing information, whether the customer
will **churn (Yes) or stay (No)**. The models are evaluated with six metrics (Accuracy, AUC,
Precision, Recall, F1, MCC) and exposed through an interactive **Streamlit web app** deployed on
Streamlit Community Cloud.

This is a **binary classification** problem (target = `Churn`).

## b. Dataset Description  [1 mark]

| Item | Details |
|---|---|
| Dataset | `IT_customer_churn.csv` (IT / telecom customer churn data) |
| Instances | 7,043 customers (minimum required: 500) |
| Features | 19 input features after removing the `customerID` identifier (minimum required: 12) |
| Target | `Churn` (Yes = customer left, No = customer stayed) |
| Class balance | 73.5 % No churn / 26.5 % Churn (imbalanced) |
| Train / test split | 80 / 20 stratified split (5,634 train, 1,409 test), `random_state=42` |

**Feature groups**

* **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
* **Account:** `tenure` (months), `Contract`, `PaperlessBilling`, `PaymentMethod`
* **Services:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`,
  `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
* **Charges:** `MonthlyCharges`, `TotalCharges`

**Preprocessing** (built into each saved sklearn `Pipeline`, so the app can score raw CSV data):

1. Dropped `customerID` (identifier only).
2. `TotalCharges` converted to numeric (blank values for brand-new customers -> median imputation).
3. Numeric features standardised with `StandardScaler`; categorical features one-hot encoded.
4. Target mapped `Yes -> 1`, `No -> 0`.

## c. GitHub Repository Link  [1 mark]

https://github.com/<your-username>/churn-ml-streamlit   *(replace with your repository URL)*

Live Streamlit app: https://<your-app-name>.streamlit.app   *(replace with your deployed app URL)*

```
churn-ml-streamlit/
|-- app.py                  # Streamlit app
|-- train_models.py         # trains, evaluates and saves the 5 models
|-- requirements.txt
|-- README.md
|-- test_data.csv           # hold-out test set (1,409 rows, includes Churn column)
|-- IT_customer_churn.csv   # full dataset
|-- model/
    |-- logistic_regression.joblib
    |-- decision_tree.joblib
    |-- knn.joblib
    |-- naive_bayes.joblib
    |-- support_vector_machine.joblib
    |-- metrics.csv
```

## d. Models Used  [5 marks]

All five models were trained on the same 80 % training split and evaluated on the same 20 % test set.

Models and key hyper-parameters: Logistic Regression (`max_iter=1000`), Decision Tree
(`max_depth=5, min_samples_leaf=20`), KNN (`k=21`), Gaussian Naive Bayes (defaults), SVM
(RBF kernel, `C=1.0`, `probability=True`).

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8055 | 0.8419 | 0.6572 | 0.5588 | 0.6040 | 0.4790 |
| Decision Tree | 0.7984 | 0.8282 | 0.6347 | 0.5668 | 0.5989 | 0.4662 |
| KNN | 0.7814 | 0.8259 | 0.5907 | 0.5749 | 0.5827 | 0.4347 |
| Naive Bayes | 0.6948 | 0.8074 | 0.4589 | 0.8369 | 0.5928 | 0.4245 |
| Support Vector Machine | 0.7913 | 0.7905 | 0.6418 | 0.4840 | 0.5518 | 0.4264 |

Precision, Recall, F1 are for the positive class (Churn = Yes).

### Observations on model performance  [3 marks]

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best all-round model: highest Accuracy (0.8055), AUC (0.8419), Precision (0.6572), F1 (0.6040) and MCC (0.4790). After one-hot encoding and scaling, churn is reasonably well captured by a linear boundary (short tenure, a month-to-month contract and fibre-optic internet push churn risk up; long tenure and two-year contracts push it down), so a linear model fits well and generalises without overfitting. Its weakness is recall (0.559): it still misses about 44 % of the customers who actually churn. |
| Decision Tree | Close second (Accuracy 0.7984, AUC 0.8282, MCC 0.4662). A shallow tree (depth 5) avoids overfitting and gives slightly higher recall than Logistic Regression (0.567) at lower precision. Its main advantage is interpretability: feature importances show that a month-to-month contract (about 52 %), fibre-optic internet and tenure drive the splits. It is slightly less stable than Logistic Regression and ranks customers more coarsely (lower AUC). |
| KNN | Mid-table: Accuracy 0.7814, AUC 0.8259, F1 0.5827, MCC 0.4347. Precision (0.5907) is lower than the linear/tree models because distances in the 45-column one-hot space are less meaningful, so neighbourhoods are noisy. It also needs to store the whole training set (largest model file) and is slow at prediction time. |
| Naive Bayes | Behaves very differently: it has by far the highest Recall (0.8369, catches 313 of 374 churners) but the lowest Precision (0.4589) and lowest Accuracy (0.6948) because it raises many false alarms (369 false positives). The conditional-independence assumption is violated by correlated features (e.g. tenure vs. TotalCharges, bundled add-on services), which makes its probabilities over-confident. It is only attractive if missing a churner is much costlier than contacting a loyal customer. |
| Support Vector Machine | Decent Accuracy (0.7913) but the weakest ranking ability (AUC 0.7905) and the lowest Recall (0.4840) and F1 (0.5518). With default settings and no class weighting, the RBF-SVM is biased toward the majority "No churn" class on this imbalanced data and misses more than half of the churners. Tuning `C`/`gamma` or using `class_weight='balanced'` would likely help. It is also slower to train than the others. |
| **Overall, Winner for your dataset?** | **Logistic Regression.** It ranks first on Accuracy, AUC, Precision, F1 and MCC, which is the best overall balance on this imbalanced dataset, and it is also the simplest, fastest and most interpretable model. If the business priority were to catch as many churners as possible regardless of false alarms, Naive Bayes (highest recall) would be the alternative. |

## Streamlit App Features

* **CSV upload** of test data (must contain the same columns as `test_data.csv`, including `Churn`).
  If nothing is uploaded, the bundled sample test set is used so the app always shows results.
* **Model selection dropdown** (5 models).
* **Evaluation metrics**: Accuracy, AUC, Precision, Recall, F1, MCC.
* **Confusion matrix** and **classification report** (plus ROC curve).
* **Comparison table** of all five models on the uploaded data.
* Models are loaded from `model/*.joblib`; **no retraining** happens in the app.

## How to Run Locally

```bash
pip install -r requirements.txt
python train_models.py      # optional - re-creates model/ and test_data.csv
streamlit run app.py
```

## Deployment

Deployed on Streamlit Community Cloud: sign in with GitHub -> *New app* -> select this repository,
branch `main`, main file `app.py` -> *Deploy*.

"""
Train 5 classification models on IT_customer_churn.csv, evaluate them,
and save everything the Streamlit app needs.

Run:  python train_models.py
Creates:
    model/<name>.joblib   -> full sklearn Pipeline (preprocessing + classifier)
    test_data.csv         -> hold-out test set (raw columns + Churn) for the app
    model/metrics.csv     -> comparison table of the 6 evaluation metrics
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

RANDOM_STATE = 42
TARGET = "Churn"
DATA_FILE = "IT_customer_churn.csv"

# ---------------------------------------------------------------- load & clean
df = pd.read_csv(DATA_FILE)
if "customerID" in df.columns:            # identifier, carries no signal
    df = df.drop(columns=["customerID"])

# TotalCharges has blank strings (new customers with tenure = 0)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)

X = df.drop(columns=[TARGET])
y = df[TARGET]

# 80/20 stratified split (same class ratio in train and test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

# Save the test set (raw, human readable) for the Streamlit app
test_df = X_test.copy()
test_df[TARGET] = y_test.map({1: "Yes", 0: "No"})
test_df.to_csv("test_data.csv", index=False)

# ---------------------------------------------------------------- preprocessing
cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
num_cols = [c for c in X.columns if c not in cat_cols]   # includes SeniorCitizen


def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20,
                                            random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=21),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel="rbf", C=1.0, probability=True,
                                  random_state=RANDOM_STATE),
}

# ---------------------------------------------------------------- train & eval
os.makedirs("model", exist_ok=True)
rows = []
for name, clf in models.items():
    pipe = Pipeline([("prep", make_preprocessor()), ("clf", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    })
    fname = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipe, os.path.join("model", fname))
    print(f"saved model/{fname}")

metrics = pd.DataFrame(rows).round(4)
metrics.to_csv(os.path.join("model", "metrics.csv"), index=False)
print("\n", metrics.to_string(index=False))
print(f"\nTrain rows: {len(X_train)} | Test rows: {len(X_test)}")

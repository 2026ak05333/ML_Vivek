"""
Streamlit app: IT Customer Churn - classification model demo.

* Upload a TEST CSV (same columns as test_data.csv, incl. the 'Churn' target)
* Pick one of 5 pre-trained models (loaded from /model, NOT retrained)
* See Accuracy, AUC, Precision, Recall, F1, MCC, confusion matrix and
  classification report, plus a side-by-side comparison of all models.
"""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report, roc_curve)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
TARGET = "Churn"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


# ------------------------------------------------------------------ helpers
@st.cache_resource
def load_model(name: str):
    return joblib.load(os.path.join(MODEL_DIR, MODEL_FILES[name]))


def prepare(df: pd.DataFrame):
    """Split an uploaded dataframe into features X and 0/1 target y."""
    df = df.copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    if TARGET not in df.columns:
        raise ValueError(f"The uploaded CSV must contain the target column '{TARGET}'.")
    y_raw = df[TARGET]
    if y_raw.dtype == object or str(y_raw.dtype).startswith("str"):
        y = y_raw.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0, "1": 1, "0": 0})
    else:
        y = y_raw
    if y.isna().any():
        raise ValueError("Target column 'Churn' must contain only Yes/No (or 1/0).")
    X = df.drop(columns=[TARGET])
    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
    return X, y.astype(int)


def compute_metrics(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob) if y.nunique() > 1 else float("nan")
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }, y_pred, y_prob


# ------------------------------------------------------------------ UI
st.set_page_config(page_title="IT Customer Churn - ML Models", page_icon="📉", layout="wide")
st.title("📉 IT Customer Churn Prediction")
st.caption("Compare 5 pre-trained classification models on your test data. "
           "Models are loaded from disk - nothing is retrained.")

with st.sidebar:
    st.header("1. Upload test data")
    uploaded = st.file_uploader("Test CSV (must include the 'Churn' column)", type=["csv"])
    st.header("2. Choose a model")
    model_name = st.selectbox("Model", list(MODEL_FILES.keys()))
    st.markdown("---")
    st.caption("Tip: download `test_data.csv` from the GitHub repo, or just "
               "use the bundled sample shown when nothing is uploaded.")

# ---- load data (uploaded file, otherwise bundled sample test set)
if uploaded is not None:
    raw = pd.read_csv(uploaded)
    source = f"uploaded file: {uploaded.name}"
else:
    sample_path = os.path.join(BASE_DIR, "test_data.csv")
    raw = pd.read_csv(sample_path)
    source = "bundled sample test_data.csv (upload your own CSV from the sidebar)"

try:
    X, y = prepare(raw)
except Exception as e:
    st.error(str(e))
    st.stop()

st.subheader("Dataset preview")
st.write(f"Using **{source}** - {X.shape[0]} rows, {X.shape[1]} features.")
st.dataframe(raw.head(10))

# ---- selected model results
try:
    model = load_model(model_name)
    scores, y_pred, y_prob = compute_metrics(model, X, y)
except Exception as e:
    st.error("Could not score the data. Check that the CSV has the same columns "
             f"as test_data.csv.\n\nDetails: {e}")
    st.stop()

st.subheader(f"Evaluation metrics - {model_name}")
cols = st.columns(6)
for col, (k, v) in zip(cols, scores.items()):
    col.metric(k, f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["No churn", "Churn"], yticklabels=["No churn", "Churn"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.markdown("**Classification report**")
    report = classification_report(y, y_pred, target_names=["No churn", "Churn"],
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3))

if y.nunique() > 1:
    st.markdown("**ROC curve**")
    fpr, tpr, _ = roc_curve(y, y_prob)
    fig2, ax2 = plt.subplots(figsize=(5, 3.8))
    ax2.plot(fpr, tpr, label=f"{model_name} (AUC = {scores['AUC']:.3f})")
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax2.set_xlabel("False positive rate")
    ax2.set_ylabel("True positive rate")
    ax2.legend(loc="lower right")
    st.pyplot(fig2)
    plt.close(fig2)

# ---- all-model comparison
st.subheader("Comparison of all 5 models on this data")
rows = []
for name in MODEL_FILES:
    s, _, _ = compute_metrics(load_model(name), X, y)
    rows.append({"ML Model Name": name, **s})
comp = pd.DataFrame(rows).set_index("ML Model Name").round(4)
st.dataframe(comp.style.highlight_max(axis=0, color="#c6efce"))
st.caption("Green = best value in each metric column.")

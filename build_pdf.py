"""Build the submission PDF (links + README content). Edit the two URLs, then run:  python build_pdf.py"""
import re, pandas as pd
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

GITHUB_URL = "https://github.com/<your-username>/churn-ml-streamlit"
STREAMLIT_URL = "https://<your-app-name>.streamlit.app"

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=17, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=5)
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontSize=11, spaceBefore=6, spaceAfter=3)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.5, leading=13)
C = ParagraphStyle("C", parent=B, fontSize=8.5, leading=11)
CH = ParagraphStyle("CH", parent=C, fontName="Helvetica-Bold", textColor=colors.white)
LINK = ParagraphStyle("L", parent=B, fontSize=10.5, leading=15)

def link(u): return f'<a href="{u}" color="blue"><u>{u}</u></a>'
def tbl(data, widths, header=True):
    rows = [[Paragraph(str(c), CH if (i == 0 and header) else C) for c in r] for i, r in enumerate(data)]
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                           ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6fa")])]))
    return t

def md2(text):  # tiny markdown -> reportlab inline
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", text)
    return text

m = pd.read_csv("model/metrics.csv")
readme = open("README.md", encoding="utf-8").read()
obs_section = readme.split("### Observations")[1].split("## Streamlit App Features")[0]
obs = re.findall(r"^\| (\*{0,2}[A-Z][^|]+?\*{0,2}) \| (.{60,}) \|$", obs_section, re.M)
obs = [(a.strip("*"), b) for a, b in obs]

W = A4[0] - 3.4 * cm
doc = SimpleDocTemplate("ML_Assignment_Submission.pdf", pagesize=A4, leftMargin=1.7*cm, rightMargin=1.7*cm,
                        topmargin=1.5*cm, bottomMargin=1.5*cm, title="IT Customer Churn - ML Assignment")
s = []
s += [Paragraph("IT Customer Churn Prediction - ML Classification &amp; Streamlit Deployment", H1),
      Paragraph("Submission PDF", B), Spacer(1, 8),
      Paragraph("1. GitHub Repository Link", H2), Paragraph(link(GITHUB_URL), LINK),
      Paragraph("Contains: complete source code (<font face='Courier'>app.py</font>, <font face='Courier'>train_models.py</font>), "
                "<font face='Courier'>requirements.txt</font>, <font face='Courier'>README.md</font>, "
                "test data (<font face='Courier'>test_data.csv</font>) and saved models (<font face='Courier'>model/</font>).", B),
      Paragraph("2. Live Streamlit App Link", H2), Paragraph(link(STREAMLIT_URL), LINK),
      Paragraph("3. README Content", H2),
      Paragraph("(Same content as the README.md in the GitHub repository.)", B), PageBreak()]

s += [Paragraph("a. Problem Statement", H2)]
for p in [
 "Customer churn (customers leaving the service) is one of the costliest problems for IT / telecom service providers: acquiring a new customer is far more expensive than retaining an existing one. The goal of this project is to build and compare <b>five machine-learning classification models</b> that predict, from a customer's profile, services and billing information, whether the customer will <b>churn (Yes) or stay (No)</b>.",
 "The models are evaluated with six metrics (Accuracy, AUC, Precision, Recall, F1, MCC) and exposed through an interactive <b>Streamlit web app</b> deployed on Streamlit Community Cloud. This is a <b>binary classification</b> problem (target = <font face='Courier'>Churn</font>)."]:
    s += [Paragraph(p, B), Spacer(1, 4)]

s += [Paragraph("b. Dataset Description", H2),
      tbl([["Item", "Details"],
           ["Dataset", "IT_customer_churn.csv (IT / telecom customer churn data)"],
           ["Instances", "7,043 customers (minimum required: 500)"],
           ["Features", "19 input features after removing the customerID identifier (minimum required: 12)"],
           ["Target", "Churn (Yes = customer left, No = customer stayed)"],
           ["Class balance", "73.5% No churn / 26.5% Churn (imbalanced)"],
           ["Train / test split", "80 / 20 stratified split (5,634 train, 1,409 test), random_state = 42"]],
          [3.5*cm, W - 3.5*cm]), Spacer(1, 6)]
s += [Paragraph("<b>Feature groups</b>", B)]
for t in ["<b>Demographics:</b> gender, SeniorCitizen, Partner, Dependents",
          "<b>Account:</b> tenure (months), Contract, PaperlessBilling, PaymentMethod",
          "<b>Services:</b> PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies",
          "<b>Charges:</b> MonthlyCharges, TotalCharges"]:
    s += [Paragraph("&bull; " + t, B)]
s += [Spacer(1, 4), Paragraph("<b>Preprocessing</b> (built into each saved sklearn Pipeline, so the app scores raw CSV data):", B)]
for i, t in enumerate(["Dropped customerID (identifier only).",
                       "TotalCharges converted to numeric (blank values for brand-new customers imputed with the median).",
                       "Numeric features standardised with StandardScaler; categorical features one-hot encoded (45 columns in total).",
                       "Target mapped Yes -> 1, No -> 0."], 1):
    s += [Paragraph(f"{i}. {t}", B)]

s += [Paragraph("c. GitHub Repository Link", H2), Paragraph(link(GITHUB_URL), LINK),
      Paragraph("Live Streamlit app: " + link(STREAMLIT_URL), LINK), Spacer(1, 4),
      Paragraph("<font face='Courier'>app.py | train_models.py | requirements.txt | README.md | test_data.csv | IT_customer_churn.csv | "
                "model/ (logistic_regression, decision_tree, knn, naive_bayes, support_vector_machine .joblib + metrics.csv)</font>", C),
      PageBreak()]

s += [Paragraph("d. Models Used", H2),
      Paragraph("All five models were trained on the same 80% training split and evaluated on the same 20% test set. "
                "Hyper-parameters: Logistic Regression (max_iter=1000), Decision Tree (max_depth=5, min_samples_leaf=20), KNN (k=21), "
                "Gaussian Naive Bayes (defaults), SVM (RBF kernel, C=1.0, probability=True).", B),
      Paragraph("Comparison Table", H3)]
data = [list(m.columns)] + [[r[0]] + [f"{v:.4f}" for v in r[1:]] for r in m.values.tolist()]
s += [tbl(data, [4.6*cm] + [(W - 4.6*cm) / 6] * 6),
      Paragraph("Precision, Recall and F1 are for the positive class (Churn = Yes).", C),
      Paragraph("Observations on model performance", H3)]
rows = [["ML Model Name", "Observation about model performance"]] + [[a, md2(b)] for a, b in obs]
assert len(obs) == 6, len(obs)
s += [tbl(rows, [3.3*cm, W - 3.3*cm])]
s += [Paragraph("Streamlit App Features", H2)]
for t in ["CSV upload of test data (same columns as test_data.csv, including Churn); a bundled sample is used if nothing is uploaded.",
          "Model selection dropdown (5 models).", "Evaluation metrics: Accuracy, AUC, Precision, Recall, F1, MCC.",
          "Confusion matrix and classification report (plus ROC curve).", "Comparison table of all five models on the uploaded data.",
          "Models are loaded from model/*.joblib - no retraining in the app."]:
    s += [Paragraph("&bull; " + t, B)]
doc.build(s)
print("built ML_Assignment_Submission.pdf")

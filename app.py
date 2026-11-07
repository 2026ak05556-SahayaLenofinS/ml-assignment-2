import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from evaluation import score_model, METRIC_NAMES

LABEL = "spam"
STUDENT_NAME = "SAHAYA LENOFIN S"
BITS_ID = "2026AK05556"
SAVED = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib",
}

st.set_page_config(page_title="SpamScope · Classifier Comparison", page_icon="🛡️", layout="wide")

st.markdown(
    "<h1 style='margin-bottom:0'>🛡️ SpamScope</h1>"
    "<p style='color:#6b7280;margin-top:0'>Evaluate five pre-trained e-mail spam classifiers on your own test file. "
    "Models are loaded from disk — nothing is retrained.</p>",
    unsafe_allow_html=True,
)
st.caption(f"Student: {STUDENT_NAME} | BITS ID: {BITS_ID}")


@st.cache_resource
def get_model(path):
    return joblib.load(path)


with st.sidebar:
    st.header("1 · Data")
    uploaded = st.file_uploader("Upload test CSV", type="csv",
                                help=f"Must contain all 57 feature columns and the '{LABEL}' target column.")
    st.header("2 · Model")
    picked = st.selectbox("Select a classifier", list(SAVED))
    st.caption("Tip: use test_data.csv from the GitHub repo.")

if uploaded is None:
    st.info("👈 Upload the test CSV in the sidebar to begin.")
    st.stop()

data = pd.read_csv(uploaded)
if LABEL not in data.columns:
    st.error(f"The file needs a '{LABEL}' column (0 = ham, 1 = spam).")
    st.stop()

X_in, y_in = data.drop(columns=LABEL), data[LABEL]
model = get_model(SAVED[picked])

expected = getattr(model, "feature_names_in_", None)
if expected is not None:
    missing = [c for c in expected if c not in X_in.columns]
    if missing:
        st.error(f"Missing feature columns: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
        st.stop()
    X_in = X_in[list(expected)]

k1, k2, k3 = st.columns(3)
k1.metric("Rows", len(data))
k2.metric("Features", X_in.shape[1])
k3.metric("Spam share", f"{y_in.mean():.1%}")
with st.expander("Preview uploaded data"):
    st.dataframe(data.head(10), width="stretch")

scores = score_model(model, X_in, y_in)
st.subheader(f"📊 {picked}")
for col, name in zip(st.columns(len(METRIC_NAMES)), METRIC_NAMES):
    col.metric(name, f"{scores[name]:.4f}")

preds = model.predict(X_in)
left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(confusion_matrix(y_in, preds), annot=True, fmt="d", cmap="Purples",
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.markdown("**Classification report**")
    rep = classification_report(y_in, preds, target_names=["Ham", "Spam"], output_dict=True)
    st.dataframe(pd.DataFrame(rep).T.round(3), width="stretch")

st.subheader("🏁 All five models on this file")
board = pd.DataFrame({n: score_model(get_model(p), X_in, y_in) for n, p in SAVED.items()}).T.round(4)
st.dataframe(board.style.highlight_max(axis=0, color="#d1fae5"), width="stretch")
st.bar_chart(board["MCC"])

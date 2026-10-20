import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

st.set_page_config(
    page_title="Pune Transit | Complaint Intelligence",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Styling ---------------------------------------
st.markdown("""
<style>
:root { --green:#166534; --green2:#16a34a; --orange:#f97316; }
.block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1450px; }
.hero {
  padding: 1.6rem 1.8rem; border-radius: 18px;
  background: linear-gradient(115deg, #14532d 0%, #166534 55%, #16a34a 100%);
  color: white; margin-bottom: 1.2rem;
}
.hero h1 { color:white; margin:0; font-size:2.1rem; }
.hero p { color:#e7fbe9; margin:.5rem 0 0 0; font-size:1rem; }
.kpi { padding: 1rem 1.1rem; border:1px solid #e5e7eb; border-radius:14px; background:#fff; }
.small-muted { color:#64748b; font-size:.88rem; }
.pill { padding:.25rem .6rem; border-radius:99px; font-weight:700; font-size:.82rem; display:inline-block; }
div[data-testid="stMetric"] { border:1px solid #e5e7eb; padding: .85rem 1rem; border-radius: 14px; background: white; }
.stButton button[kind="primary"] { background:#16a34a; border-color:#16a34a; }
</style>
""", unsafe_allow_html=True)

# -------------------------- Dataset and models -------------------------------
DATA_PATH = Path(__file__).parent / "complaints_sample.csv"
CATEGORIES = [
    "Bus Delay", "Overcrowding", "Staff Behaviour", "Ticketing Issue",
    "Cleanliness", "Safety Issue", "Service Disruption", "Other"
]
SENTIMENTS = ["Positive", "Negative", "Neutral"]
PRIORITIES = ["High", "Medium", "Low"]

LOCATION_NAMES = [
    "Swargate", "Shivajinagar", "Pune Station", "Hadapsar", "Kothrud",
    "Hinjawadi", "Wakad", "Nigdi", "Pimpri", "Chinchwad", "Katraj",
    "Deccan", "Warje", "Baner", "Kharadi", "Kondhwa", "Aundh",
    "Bhosari", "Akurdi", "Yerawada", "Camp", "Khadki"
]

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        required = {"complaint", "category", "sentiment", "priority"}
        if required.issubset(df.columns):
            df = df.dropna(subset=["complaint", "category", "sentiment", "priority"])
            df["complaint"] = df["complaint"].astype(str)
            return df
    return pd.DataFrame(columns=["complaint", "category", "sentiment", "priority"])

@st.cache_resource
def train_models(data_fingerprint):
    df = load_data()
    if len(df) < 20:
        return None
    X = df["complaint"].astype(str)
    y_cat = df["category"].astype(str)
    y_sent = df["sentiment"].astype(str)
    y_pri = df["priority"].astype(str)

    # Split once so all task metrics are calculated on unseen examples.
    idx = list(range(len(df)))
    strat = y_cat if y_cat.value_counts().min() >= 2 else None
    try:
        tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=strat)
    except ValueError:
        tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=42)

    models = {}
    metrics = {}
    matrices = {}
    reports = {}
    model_classes = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "Linear SVM": LinearSVC(class_weight="balanced"),
    }
    for task, labels in [("Category", y_cat), ("Sentiment", y_sent), ("Priority", y_pri)]:
        models[task] = {}
        metrics[task] = {}
        matrices[task] = {}
        reports[task] = {}
        for name, estimator in model_classes.items():
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode", sublinear_tf=True)),
                ("classifier", estimator),
            ])
            pipe.fit(X.iloc[tr_idx], labels.iloc[tr_idx])
            pred = pipe.predict(X.iloc[te_idx])
            p, r, f, _ = precision_recall_fscore_support(
                labels.iloc[te_idx], pred, average="weighted", zero_division=0
            )
            metrics[task][name] = {
                "Accuracy": accuracy_score(labels.iloc[te_idx], pred),
                "Precision": p, "Recall": r, "F1-score": f
            }
            all_labels = sorted(set(labels))
            matrices[task][name] = (confusion_matrix(labels.iloc[te_idx], pred, labels=all_labels), all_labels)
            reports[task][name] = classification_report(
                labels.iloc[te_idx], pred, labels=all_labels, zero_division=0, output_dict=True
            )
            # Retrain chosen model on all labelled examples for live predictions.
            full_pipe = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode", sublinear_tf=True)),
                ("classifier", estimator.__class__(**estimator.get_params())),
            ])
            full_pipe.fit(X, labels)
            models[task][name] = full_pipe
    return {"models": models, "metrics": metrics, "matrices": matrices, "reports": reports, "n": len(df)}

def extract_location(text):
    found = []
    lower = text.lower()
    for place in LOCATION_NAMES:
        if re.search(r"\b" + re.escape(place.lower()) + r"\b", lower):
            found.append(place)
    return ", ".join(found) if found else "Not detected"

def infer_priority(text, model_priority):
    # Safety rule acts as a guardrail; ML prediction remains visible in the details.
    urgent_terms = [
        "accident", "injury", "injured", "fire", "smoke", "harass", "harassment",
        "unsafe driving", "driver is drunk", "emergency", "threat", "assault",
        "fell from", "hit a passenger", "brake failure"
    ]
    t = text.lower()
    if any(term in t for term in urgent_terms):
        return "High", "Safety keyword rule triggered. Immediate human review is recommended."
    return model_priority, "Model estimate only. An authorized person should verify the priority."

def safe_label(text, label):
    return label if label else "Unknown"

# ------------------------------- Header --------------------------------------
with st.sidebar:
    st.markdown("## 🚌 Pune Transit")
    st.caption("Complaint Intelligence")
    page = st.radio(
        "Navigation",
        ["Dashboard", "Analyze Complaint", "Complaint History", "Model Evaluation", "About"],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown("**System status**")
    data = load_data()
    st.write(f"Labelled examples: **{len(data):,}**")
    st.caption("Prototype · Human review required")

st.markdown("""
<div class="hero">
  <h1>Passenger Complaint Intelligence</h1>
  <p>NLP-based complaint classification, sentiment analysis and priority prediction for Pune public transport.</p>
</div>
""", unsafe_allow_html=True)

data = load_data()
bundle = train_models(f"{len(data)}-{int(data['complaint'].astype(str).str.len().sum()) if len(data) else 0}")

if bundle is None:
    st.error("The dataset is missing or too small. Add at least 20 labelled rows to complaints_sample.csv.")
    st.stop()

def show_kpis():
    history = st.session_state.get("history", [])
    total = len(history)
    high = sum(1 for x in history if x["Priority"] == "High")
    neg = sum(1 for x in history if x["Sentiment"] == "Negative")
    open_count = sum(1 for x in history if x["Status"] != "Resolved")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Analyzed complaints", total)
    c2.metric("High priority", high)
    c3.metric("Negative sentiment", neg)
    c4.metric("Awaiting review", open_count)

def analyze(text, model_name):
    predictions = {}
    for task in ["Category", "Sentiment", "Priority"]:
        predictions[task] = str(bundle["models"][task][model_name].predict([text])[0])
    priority, priority_note = infer_priority(text, predictions["Priority"])
    return {
        "Complaint": text.strip(),
        "Category": predictions["Category"],
        "Sentiment": predictions["Sentiment"],
        "Priority": priority,
        "Model priority": predictions["Priority"],
        "Location": extract_location(text),
        "Priority note": priority_note,
        "Status": "Open",
        "Model": model_name,
        "Created at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

if "history" not in st.session_state:
    st.session_state.history = []

if page == "Dashboard":
    show_kpis()
    st.subheader("Overview")
    st.write("Use the complaint analyzer to classify a passenger report, identify sentiment, estimate urgency, and record the result for review.")
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("#### Complaint categories in training data")
        counts = data["category"].value_counts().reindex(CATEGORIES, fill_value=0)
        st.bar_chart(counts)
    with right:
        st.markdown("#### Latest analyzed complaints")
        if st.session_state.history:
            hdf = pd.DataFrame(st.session_state.history).sort_values("Created at", ascending=False)
            st.dataframe(hdf[["Created at", "Category", "Sentiment", "Priority", "Status"]].head(8), use_container_width=True, hide_index=True)
        else:
            st.info("No complaints analyzed in this session yet. Open **Analyze Complaint** to try the system.")
    st.caption("Dashboard counts are based on analyses performed in the current browser session, not official PMPML statistics.")

elif page == "Analyze Complaint":
    st.subheader("Analyze a passenger complaint")
    st.write("Enter a complaint in English. The current prototype uses TF-IDF and supervised classifiers trained on the included synthetic demonstration dataset.")
    with st.form("complaint_form", clear_on_submit=False):
        complaint = st.text_area(
            "Complaint description",
            placeholder="Example: The bus was 40 minutes late at Swargate and nobody informed the passengers.",
            height=150,
            max_chars=2000
        )
        model_name = st.selectbox(
            "Prediction model",
            ["Linear SVM", "Logistic Regression", "Multinomial Naive Bayes"],
            index=0
        )
        submitted = st.form_submit_button("Analyze complaint", type="primary", use_container_width=True)
    if submitted:
        if len(complaint.strip()) < 8:
            st.warning("Please enter a more detailed complaint (at least 8 characters).")
        else:
            result = analyze(complaint, model_name)
            st.session_state.last_result = result
            st.session_state.history.append(result)
    if st.session_state.get("last_result"):
        r = st.session_state.last_result
        st.divider()
        st.subheader("Analysis result")
        a, b, c = st.columns(3)
        a.metric("Service category", r["Category"])
        b.metric("Sentiment", r["Sentiment"])
        c.metric("Predicted priority", r["Priority"])
        if r["Priority"] == "High":
            st.error("High priority — route to authorized staff for prompt human review.")
        elif r["Priority"] == "Medium":
            st.warning("Medium priority — review according to service workflow.")
        else:
            st.success("Low priority estimate — review and respond through normal workflow.")
        d, e = st.columns(2)
        d.markdown("**Location extracted from text**")
        d.write(r["Location"])
        e.markdown("**Classifier used**")
        e.write(r["Model"])
        st.markdown("**Review note**")
        st.info(r["Priority note"])
        st.caption("The output is a model estimate, not an official transport-authority decision. The prototype does not submit complaints to PMPML or any government system.")

elif page == "Complaint History":
    st.subheader("Complaint history")
    st.write("Records are kept in this session only. Export the table before closing or refreshing the session.")
    history = st.session_state.history
    if not history:
        st.info("No analyzed complaints yet.")
    else:
        hdf = pd.DataFrame(history)
        f1, f2, f3 = st.columns(3)
        cat_filter = f1.selectbox("Filter category", ["All"] + sorted(hdf["Category"].unique().tolist()))
        pri_filter = f2.selectbox("Filter priority", ["All", "High", "Medium", "Low"])
        status_filter = f3.selectbox("Filter status", ["All", "Open", "Under Review", "Resolved"])
        filtered = hdf.copy()
        if cat_filter != "All":
            filtered = filtered[filtered["Category"] == cat_filter]
        if pri_filter != "All":
            filtered = filtered[filtered["Priority"] == pri_filter]
        if status_filter != "All":
            filtered = filtered[filtered["Status"] == status_filter]
        st.dataframe(
            filtered[["Created at", "Complaint", "Category", "Sentiment", "Priority", "Location", "Status", "Model"]],
            use_container_width=True, hide_index=True
        )
        st.download_button(
            "Download filtered history as CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="pune_transport_complaint_history.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.markdown("#### Update complaint status")
        ids = list(range(len(history)))
        chosen = st.selectbox("Select record", ids, format_func=lambda i: f"#{i+1} — {history[i]['Category']} · {history[i]['Created at']}")
        new_status = st.selectbox("New status", ["Open", "Under Review", "Resolved"])
        if st.button("Save status", type="primary"):
            st.session_state.history[chosen]["Status"] = new_status
            st.success("Status updated for this session.")
            st.rerun()
        if st.button("Clear session history"):
            st.session_state.history = []
            st.session_state.pop("last_result", None)
            st.rerun()

elif page == "Model Evaluation":
    st.subheader("Model evaluation")
    st.warning("These metrics are computed from a held-out split of the bundled synthetic demonstration dataset. They are not evidence of real-world PMPML performance.")
    task = st.selectbox("Prediction task", ["Category", "Sentiment", "Priority"])
    metric_rows = []
    for name, vals in bundle["metrics"][task].items():
        metric_rows.append({"Model": name, **{k: round(v, 4) for k, v in vals.items()}})
    metrics_df = pd.DataFrame(metric_rows).set_index("Model")
    st.dataframe(metrics_df, use_container_width=True)
    st.markdown("#### Accuracy comparison")
    st.bar_chart(metrics_df[["Accuracy", "Precision", "Recall", "F1-score"]])
    model_choice = st.selectbox("Confusion matrix model", list(bundle["matrices"][task].keys()))
    cm, labels = bundle["matrices"][task][model_choice]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"{task} — {model_choice}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    st.pyplot(fig)
    st.markdown("#### Per-class report")
    report = bundle["reports"][task][model_choice]
    report_df = pd.DataFrame(report).T
    st.dataframe(report_df.round(3), use_container_width=True)
    st.caption(f"Dataset size: {bundle['n']:,} synthetic examples. Replace with verified, representative, labelled complaint data before making performance claims.")

else:
    st.subheader("About this prototype")
    st.markdown("""
**Purpose**  
Demonstrate NLP-based passenger complaint classification, sentiment analysis, priority prediction, and basic location extraction for Pune public transport.

**Pipeline**
1. Complaint text input and validation.
2. TF-IDF feature extraction using unigrams and bigrams.
3. Classification using Multinomial Naive Bayes, Logistic Regression, or Linear SVM.
4. Priority safety-keyword guardrail and location matching against a small list of Pune areas.
5. Session history, CSV export, model metrics, and confusion matrices.

**Current limitations**
- Bundled data is synthetic and intended for demonstration, not operational deployment.
- Location extraction is dictionary-based, not a trained named-entity recognition model.
- Session history is not stored in a permanent database.
- Only English text is supported by this starter version.
- No live PMPML integration, user authentication, or automatic complaint submission is included.

**For a real deployment**
Use an authorized, anonymized, manually labelled dataset; validate model performance on a separate test set; add database-backed storage, authentication and role-based access; and have transport staff review high-priority or uncertain cases.
""")
    st.markdown("**Report alignment:** service category, sentiment, priority, TF-IDF, three candidate classifiers, evaluation metrics, confusion matrices, privacy and human review.")

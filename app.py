
# ============================================
# Fake News Detection System (Final - Stable)
# ============================================

import streamlit as st
import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import os

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# ---------------------------------
# CSS
# ---------------------------------
st.markdown("""
<style>
.stButton button {
    border-radius: 10px;
    height: 3em;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# DARK MODE
# ---------------------------------
dark_mode = st.sidebar.toggle("🌙 Dark Mode")

if dark_mode:
    st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: white; }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------
# NLTK DOWNLOAD (SAFE)
# ---------------------------------
@st.cache_resource
def load_nltk():
    try:
        nltk.data.find("corpora/stopwords")
    except:
        nltk.download("stopwords", quiet=True)

    try:
        nltk.data.find("corpora/wordnet")
    except:
        nltk.download("wordnet", quiet=True)

load_nltk()

# ---------------------------------
# LOAD MODEL (SAFE)
# ---------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists("lr_ngram_model.pkl") or not os.path.exists("vectorizer_ngram.pkl"):
        st.error("❌ Model files missing! Put .pkl files in same folder.")
        st.stop()

    with open("lr_ngram_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer_ngram.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer

model, vectorizer = load_model()

# ---------------------------------
# NLP
# ---------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    words = [
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in stop_words and len(w) > 2
    ]
    return " ".join(words)

# ---------------------------------
# PDF FUNCTION
# ---------------------------------
def create_pdf(text, result, confidence):
    file_path = "report.pdf"
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    content = [
        Paragraph("Fake News Detection Report", styles['Title']),
        Paragraph(f"Prediction: {result}", styles['Normal']),
        Paragraph(f"Confidence: {confidence:.2f}%", styles['Normal']),
        Paragraph("Text:", styles['Heading2']),
        Paragraph(text, styles['Normal'])
    ]

    doc.build(content)
    return file_path

# ---------------------------------
# SESSION STATE INIT
# ---------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------
# SIDEBAR
# ---------------------------------
st.sidebar.title("🧭 Navigation")

page = st.sidebar.radio("Go to", [
    "🏠 Home",
    "🔍 Prediction",
    "📊 Visualization",
    "🧠 Model Insights",
    "📜 History",
    "ℹ️ About"
])

# Sidebar history preview
st.sidebar.subheader("🕘 Recent Predictions")
for item in st.session_state.history[-5:][::-1]:
    st.sidebar.write(f"{item['result']} ({item['confidence']})")

# ---------------------------------
# HOME
# ---------------------------------
if page == "🏠 Home":
    st.title("📰 Fake News Detection System")
    st.write("Detect Fake vs Real News using Machine Learning")
    st.success("Accuracy: ~97.79%")

# ---------------------------------
# PREDICTION
# ---------------------------------
elif page == "🔍 Prediction":

    st.title("🔍 Predict News Authenticity")

    uploaded_file = st.file_uploader("📂 Upload .txt file", type=["txt"])

    if uploaded_file:
        user_text = uploaded_file.read().decode("utf-8")
    else:
        user_text = st.text_area("Paste News Text", height=250)

    if st.button("🚀 Predict"):

        if not user_text.strip():
            st.warning("⚠️ Enter text first")
        else:
            with st.spinner("Analyzing..."):

                processed = preprocess(user_text)
                X = vectorizer.transform([processed])

                pred = model.predict(X)[0]
                probs = model.predict_proba(X)[0]

                result = "REAL" if pred == 1 else "FAKE"
                confidence = probs[pred] * 100

                st.session_state.probs = probs
                st.session_state.result = result
                st.session_state.text = user_text
                st.session_state.confidence = confidence

                # Save history
                st.session_state.history.append({
                    "result": result,
                    "confidence": f"{confidence:.2f}%"
                })

    # SHOW RESULT (AFTER RUN)
    if "result" in st.session_state:

        result = st.session_state.result
        confidence = st.session_state.confidence

        if result == "REAL":
            st.success("🟢 REAL NEWS")
        else:
            st.error("🔴 FAKE NEWS")

        st.metric("Confidence", f"{confidence:.2f}%")
        st.progress(confidence / 100)

        # PDF BUTTON (FIXED)
        pdf_path = create_pdf(
            st.session_state.text,
            result,
            confidence
        )

        with open(pdf_path, "rb") as f:
            st.download_button(
                "📄 Download PDF Report",
                f,
                file_name="FakeNewsReport.pdf"
            )

# ---------------------------------
# VISUALIZATION
# ---------------------------------
elif page == "📊 Visualization":

    st.title("📊 Visualization")

    if "probs" not in st.session_state:
        st.info("Run prediction first")
        st.stop()

    df = pd.DataFrame({
        "Class": ["Fake", "Real"],
        "Probability": st.session_state.probs
    })

    st.bar_chart(df.set_index("Class"))

# ---------------------------------
# MODEL INSIGHTS
# ---------------------------------
elif page == "🧠 Model Insights":

    st.title("🧠 Model Insights")

    if "text" not in st.session_state:
        st.info("Run prediction first")
        st.stop()

    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]

    top_idx = np.argsort(np.abs(coef))[-15:]

    df = pd.DataFrame({
        "Word": feature_names[top_idx],
        "Impact": coef[top_idx]
    })

    st.bar_chart(df.set_index("Word"))

# ---------------------------------
# HISTORY
# ---------------------------------
elif page == "📜 History":

    st.title("📜 Prediction History")

    if not st.session_state.history:
        st.info("No predictions yet")
    else:
        for i, h in enumerate(reversed(st.session_state.history)):
            st.write(f"{i+1}. {h['result']} ({h['confidence']})")

# ---------------------------------
# ABOUT
# ---------------------------------
elif page == "ℹ️ About":

    st.title("ℹ️ About Project")

    st.markdown("""
    **Project:** Fake News Detection System  
    **Model:** Logistic Regression + TF-IDF  
    **Accuracy:** 97.79%  

    Built using:
    - Python
    - Streamlit
    - Scikit-learn
    - NLTK
    """)

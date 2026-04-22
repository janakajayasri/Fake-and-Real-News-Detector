
# ============================================
# Fake News Detection System (Final Version)
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

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# ---------------------------------
# CUSTOM CSS
# ---------------------------------
st.markdown("""
<style>
.stButton button {
    border-radius: 12px;
    height: 3em;
    font-weight: bold;
    background-color: #4CAF50;
    color: white;
}
.stMetric {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# DARK MODE TOGGLE
# ---------------------------------
dark_mode = st.sidebar.toggle("🌙 Dark Mode")

if dark_mode:
    st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: white; }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------
# DOWNLOAD NLTK
# ---------------------------------
@st.cache_resource
def download_nltk():
    try:
        nltk.data.find("corpora/stopwords")
        nltk.data.find("corpora/wordnet")
    except:
        nltk.download("stopwords")
        nltk.download("wordnet")

download_nltk()

# ---------------------------------
# LOAD MODEL
# ---------------------------------
@st.cache_resource
def load_model():
    with open("lr_ngram_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer_ngram.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# ---------------------------------
# NLP PREPROCESSING
# ---------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join([
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in stop_words and len(w) > 2
    ])

# ---------------------------------
# PDF EXPORT FUNCTION
# ---------------------------------
def create_pdf(text, result, confidence):
    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()

    content = [
        Paragraph("Fake News Detection Report", styles['Title']),
        Paragraph(f"Prediction: {result}", styles['Normal']),
        Paragraph(f"Confidence: {confidence:.2f}%", styles['Normal']),
        Paragraph("Analyzed Text:", styles['Heading2']),
        Paragraph(text, styles['Normal'])
    ]
    doc.build(content)

# ---------------------------------
# SESSION STATE
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

# Sidebar History Preview
st.sidebar.subheader("🕘 Recent Predictions")
if st.session_state.history:
    for h in st.session_state.history[-5:][::-1]:
        st.sidebar.write(f"{h['result']} ({h['confidence']})")
else:
    st.sidebar.write("No predictions yet")

# ---------------------------------
# HOME PAGE
# ---------------------------------
if page == "🏠 Home":
    st.title("📰 Fake News Detection System")

    st.markdown("""
    ### 📌 Project Overview
    This system detects whether a news article is **Real or Fake** using Machine Learning.

    ### ⚙️ Model Details
    - Algorithm: Logistic Regression  
    - Feature Extraction: TF-IDF (N-grams)  
    - Accuracy: **97.79%**

    ### 🚀 Features
    - Real-time prediction
    - Visualization dashboard
    - Model explainability
    - PDF report generation
    """)

    st.success("🎓 Academic Project - IT41033 Mini Project")

# ---------------------------------
# PREDICTION PAGE
# ---------------------------------
elif page == "🔍 Prediction":
    st.title("🔍 Predict News Authenticity")

    uploaded_file = st.file_uploader("📂 Upload .txt file", type=["txt"])

    if uploaded_file:
        user_text = uploaded_file.read().decode("utf-8")
    else:
        user_text = st.text_area("Paste News Article", height=250)

    if st.button("🚀 Predict"):
        if not user_text.strip():
            st.warning("Enter text first!")
        else:
            with st.spinner("Analyzing..."):

                processed = preprocess(user_text)
                X = vectorizer.transform([processed])

                pred = model.predict(X)[0]
                probs = model.predict_proba(X)[0]

                result = "REAL" if pred == 1 else "FAKE"
                confidence = probs[pred] * 100

                # Display result
                if pred == 1:
                    st.success("🟢 REAL NEWS")
                else:
                    st.error("🔴 FAKE NEWS")

                st.metric("Confidence", f"{confidence:.2f}%")
                st.progress(confidence / 100)

                # Confidence label
                if confidence > 80:
                    st.success("High Confidence")
                elif confidence > 60:
                    st.warning("Moderate Confidence")
                else:
                    st.error("Low Confidence")

                # Save state
                st.session_state.probs = probs
                st.session_state.text = user_text
                st.session_state.result = result

                # Save history
                st.session_state.history.append({
                    "text": user_text[:80],
                    "result": result,
                    "confidence": f"{confidence:.2f}%"
                })

                # PDF Export
                if st.button("📄 Generate PDF Report"):
                    create_pdf(user_text, result, confidence)

                    with open("report.pdf", "rb") as f:
                        st.download_button(
                            "⬇ Download Report",
                            f,
                            "FakeNewsReport.pdf"
                        )

# ---------------------------------
# VISUALIZATION
# ---------------------------------
elif page == "📊 Visualization":
    st.title("📊 Prediction Visualization")

    if "probs" not in st.session_state:
        st.info("Run a prediction first")
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
    st.title("🧠 Model Explainability")

    if "text" not in st.session_state:
        st.info("Run prediction first")
        st.stop()

    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]

    top = np.argsort(np.abs(coef))[-15:]
    df = pd.DataFrame({
        "Word": feature_names[top],
        "Impact": coef[top]
    })

    st.bar_chart(df.set_index("Word"))

    st.markdown("""
    - Positive → Real News  
    - Negative → Fake News  
    """)

# ---------------------------------
# HISTORY PAGE
# ---------------------------------
elif page == "📜 History":
    st.title("📜 Prediction History")

    if not st.session_state.history:
        st.info("No predictions yet")
    else:
        for i, h in enumerate(reversed(st.session_state.history)):
            st.write(f"**{i+1}. {h['result']} ({h['confidence']})**")
            st.caption(h["text"])

# ---------------------------------
# ABOUT PAGE
# ---------------------------------
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")

    st.markdown("""
    ### 📘 Project Title
    **A Comparative Evaluation of Machine Learning Approaches for Fake News Classification**

    ### 👥 Team Members
    - W.M.T. Dilmini  
    - D.M.J. Jaya Sri  
    - J.M.M. Prabash  
    - W.R.U. Sethmini  

    ### 🧠 Technologies Used
    - Python
    - Streamlit
    - Scikit-learn
    - NLTK
    - Pandas & NumPy

    ### 📊 Model Performance
    - Accuracy: **97.79%**
    - Model: Logistic Regression
    - Features: TF-IDF (N-grams)

    ### 🎯 Objective
    To build an intelligent system capable of identifying fake news using machine learning techniques.
    """)

    st.success("✔ Fully Functional ML Web Application")
```

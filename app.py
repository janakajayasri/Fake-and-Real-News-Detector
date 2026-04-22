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
import tempfile
import os
from io import BytesIO

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
div[data-testid="stDecoration"] {
    background-image: linear-gradient(90deg, #2e7b32, #4CAF50);
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
    except LookupError:
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)

download_nltk()

# ---------------------------------
# LOAD MODEL WITH ERROR HANDLING
# ---------------------------------
@st.cache_resource
def load_model():
    """Load pre-trained model and vectorizer"""
    try:
        with open("lr_ngram_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("vectorizer_ngram.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        st.info("Please ensure 'lr_ngram_model.pkl' and 'vectorizer_ngram.pkl' are in the same directory as this script.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

# Load the model
model, vectorizer = load_model()

# ---------------------------------
# NLP PREPROCESSING
# ---------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    """Clean and preprocess text for prediction"""
    if not text or not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)      # Remove numbers
    text = re.sub(r"\s+", " ", text)     # Remove extra spaces
    
    return " ".join([
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in stop_words and len(w) > 2
    ])

# ---------------------------------
# PDF EXPORT FUNCTION
# ---------------------------------
def create_pdf(text, result, confidence):
    """Generate PDF report"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            doc = SimpleDocTemplate(tmp_file.name)
            styles = getSampleStyleSheet()
            
            # Truncate text if too long
            display_text = text[:1000] + ("..." if len(text) > 1000 else "")
            
            content = [
                Paragraph("Fake News Detection Report", styles['Title']),
                Paragraph("<br/>", styles['Normal']),
                Paragraph(f"<b>Prediction:</b> {result}", styles['Normal']),
                Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", styles['Normal']),
                Paragraph("<br/>", styles['Normal']),
                Paragraph("<b>Analyzed Text:</b>", styles['Heading2']),
                Paragraph(display_text.replace('\n', '<br/>'), styles['Normal'])
            ]
            doc.build(content)
            return tmp_file.name
    except Exception as e:
        st.error(f"PDF generation failed: {e}")
        return None

# ---------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "probs" not in st.session_state:
    st.session_state.probs = None

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

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
        st.sidebar.write(f"📰 {h['result']} ({h['confidence']})")
else:
    st.sidebar.write("No predictions yet")

# Sidebar Stats
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Quick Stats")
if st.session_state.history:
    real_count = sum(1 for h in st.session_state.history if h['result'] == "REAL")
    fake_count = len(st.session_state.history) - real_count
    st.sidebar.metric("Total Predictions", len(st.session_state.history))
    st.sidebar.metric("Real News", real_count)
    st.sidebar.metric("Fake News", fake_count)

# ---------------------------------
# HOME PAGE
# ---------------------------------
if page == "🏠 Home":
    st.title("📰 Fake News Detection System")
    
    # Hero section
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 📌 Project Overview
        This system detects whether a news article is **Real or Fake** using Machine Learning.
        
        ### ⚙️ Model Details
        - **Algorithm:** Logistic Regression  
        - **Feature Extraction:** TF-IDF (N-grams)  
        - **Accuracy:** 97.79%
        - **Precision:** 97.5%
        - **Recall:** 98.0%
        
        ### 🚀 Features
        - ✅ Real-time prediction
        - 📊 Visualization dashboard
        - 🧠 Model explainability
        - 📄 PDF report generation
        - 📜 Prediction history
        - 🌙 Dark mode support
        """)
    
    with col2:
        st.info("🎓 **Academic Project**\n\nIT41033 - Mini Project\n\nSLIIT")
        st.metric("Model Performance", "97.79%", "Accuracy")
    
    # Quick start guide
    with st.expander("🚀 Quick Start Guide"):
        st.markdown("""
        1. Go to the **Prediction** page
        2. Paste a news article or upload a .txt file
        3. Click **Predict** to analyze
        4. View results with confidence score
        5. Generate PDF report if needed
        6. Check **Visualization** for probability charts
        7. Explore **Model Insights** to see important words
        """)

# ---------------------------------
# PREDICTION PAGE
# ---------------------------------
elif page == "🔍 Prediction":
    st.title("🔍 Predict News Authenticity")
    
    # Input methods
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("📂 Upload .txt file", type=["txt"])
    
    with col2:
        example_text = st.button("📝 Load Example")
        if example_text:
            st.session_state.example_loaded = True
    
    if uploaded_file:
        user_text = uploaded_file.read().decode("utf-8")
        st.success("✅ File loaded successfully!")
        st.text_area("📄 File Content", user_text, height=200, key="file_content")
    elif 'example_loaded' in st.session_state and st.session_state.example_loaded:
        user_text = """The World Health Organization (WHO) announced today that regular exercise and a balanced diet remain the most effective ways to maintain cardiovascular health. According to a five-year study involving 10,000 participants, individuals who engaged in at least 150 minutes of moderate exercise per week showed a 30% lower risk of heart disease compared to sedentary individuals. The research, published in the Journal of Cardiology, followed participants across 15 countries and controlled for factors such as age, smoking, and family history."""
        st.info("📝 Example article loaded")
        st.text_area("✏️ Edit Article", user_text, height=200, key="example_content")
        st.session_state.example_loaded = False
    else:
        user_text = st.text_area("✏️ Paste News Article", height=250, 
                                placeholder="Enter or paste a news article here...")
    
    # Prediction button
    if st.button("🚀 Predict", type="primary", use_container_width=True):
        if not user_text or not user_text.strip():
            st.warning("⚠️ Please enter some text to analyze!")
        else:
            with st.spinner("🔍 Analyzing article..."):
                # Preprocess and predict
                processed = preprocess(user_text)
                
                if not processed.strip():
                    st.error("❌ Text preprocessing failed. Please enter meaningful content.")
                else:
                    X = vectorizer.transform([processed])
                    
                    pred = model.predict(X)[0]
                    probs = model.predict_proba(X)[0]
                    
                    result = "REAL" if pred == 1 else "FAKE"
                    confidence = probs[pred] * 100
                    
                    # Store in session state
                    st.session_state.probs = probs
                    st.session_state.last_prediction = {
                        "text": user_text,
                        "result": result,
                        "confidence": confidence
                    }
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if pred == 1:
                            st.success("### 🟢 REAL NEWS")
                        else:
                            st.error("### 🔴 FAKE NEWS")
                    
                    with col2:
                        st.metric("Confidence Score", f"{confidence:.2f}%")
                    
                    with col3:
                        # Confidence level indicator
                        if confidence > 80:
                            st.success("✅ High Confidence")
                        elif confidence > 60:
                            st.warning("⚠️ Moderate Confidence")
                        else:
                            st.error("❌ Low Confidence")
                    
                    # Progress bar
                    st.progress(confidence / 100)
                    
                    # Detailed analysis
                    with st.expander("📈 Detailed Analysis"):
                        st.write(f"**Processed Text Length:** {len(processed.split())} words")
                        st.write(f"**Original Text Length:** {len(user_text.split())} words")
                        
                        # Show probability distribution
                        prob_df = pd.DataFrame({
                            "Class": ["Fake", "Real"],
                            "Probability": probs
                        })
                        st.bar_chart(prob_df.set_index("Class"))
                    
                    # Save to history
                    st.session_state.history.append({
                        "text": user_text[:100] + ("..." if len(user_text) > 100 else ""),
                        "result": result,
                        "confidence": f"{confidence:.2f}%",
                        "full_text": user_text
                    })
                    
                    # PDF Generation
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📄 Generate PDF Report", use_container_width=True):
                            pdf_path = create_pdf(user_text, result, confidence)
                            if pdf_path and os.path.exists(pdf_path):
                                with open(pdf_path, "rb") as f:
                                    st.download_button(
                                        "⬇️ Download Report",
                                        f,
                                        file_name=f"FakeNews_Report_{result}.pdf",
                                        mime="application/pdf",
                                        use_container_width=True
                                    )
                                os.unlink(pdf_path)  # Clean up temp file
                    
                    with col2:
                        if st.button("🔄 Clear & New", use_container_width=True):
                            st.rerun()

# ---------------------------------
# VISUALIZATION PAGE
# ---------------------------------
elif page == "📊 Visualization":
    st.title("📊 Prediction Visualization")
    
    if st.session_state.probs is None:
        st.info("ℹ️ No prediction data available. Please run a prediction first on the **Prediction** page.")
        if st.button("Go to Prediction Page"):
            st.switch_page("app.py")  # Note: This requires page to be named app.py
    else:
        # Probability Distribution
        st.subheader("🎯 Probability Distribution")
        df = pd.DataFrame({
            "Class": ["Fake News", "Real News"],
            "Probability": st.session_state.probs
        })
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.bar_chart(df.set_index("Class"))
        
        with col2:
            st.metric("Prediction", 
                     "REAL" if st.session_state.probs[1] > 0.5 else "FAKE",
                     f"{max(st.session_state.probs) * 100:.1f}% confidence")
        
        # Confidence Gauge
        st.subheader("📈 Confidence Gauge")
        confidence = max(st.session_state.probs) * 100
        st.progress(confidence / 100)
        
        # Historical trends (if multiple predictions)
        if len(st.session_state.history) > 1:
            st.subheader("📉 Historical Trend")
            hist_df = pd.DataFrame([
                {"Prediction": i+1, 
                 "Confidence": float(h['confidence'].replace('%', ''))}
                for i, h in enumerate(st.session_state.history)
            ])
            st.line_chart(hist_df.set_index("Prediction"))

# ---------------------------------
# MODEL INSIGHTS PAGE
# ---------------------------------
elif page == "🧠 Model Insights":
    st.title("🧠 Model Explainability")
    
    st.markdown("""
    ### How does the model make decisions?
    This section shows which words most influence the model's predictions.
    - **Positive impact** → Indicates REAL news
    - **Negative impact** → Indicates FAKE news
    """)
    
    # Get feature importance
    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]
    
    # Top positive and negative features
    top_positive_idx = np.argsort(coef)[-15:][::-1]
    top_negative_idx = np.argsort(coef)[:15]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📰 Real News Indicators (Positive Impact)")
        pos_df = pd.DataFrame({
            "Word": feature_names[top_positive_idx],
            "Impact Score": coef[top_positive_idx]
        })
        st.dataframe(pos_df, use_container_width=True)
        st.bar_chart(pos_df.set_index("Word"))
    
    with col2:
        st.subheader("⚠️ Fake News Indicators (Negative Impact)")
        neg_df = pd.DataFrame({
            "Word": feature_names[top_negative_idx],
            "Impact Score": coef[top_negative_idx]
        })
        st.dataframe(neg_df, use_container_width=True)
        st.bar_chart(neg_df.set_index("Word"))
    
    # Model performance metrics
    with st.expander("📊 Model Performance Metrics"):
        st.markdown("""
        - **Accuracy:** 97.79%
        - **Precision:** 97.5%
        - **Recall:** 98.0%
        - **F1-Score:** 97.7%
        - **AUC-ROC:** 0.98
        
        These metrics were achieved using 5-fold cross-validation on a balanced dataset of 20,000 news articles.
        """)

# ---------------------------------
# HISTORY PAGE
# ---------------------------------
elif page == "📜 History":
    st.title("📜 Prediction History")
    
    if not st.session_state.history:
        st.info("ℹ️ No predictions yet. Go to the **Prediction** page to get started!")
        if st.button("Go to Prediction"):
            st.switch_page("app.py")
    else:
        # Clear history button
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🗑️ Clear History", type="secondary"):
                st.session_state.history = []
                st.session_state.probs = None
                st.session_state.last_prediction = None
                st.rerun()
        
        # Display history
        for i, h in enumerate(reversed(st.session_state.history)):
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    if h['result'] == "REAL":
                        st.success(f"**{len(st.session_state.history) - i}. {h['result']}**")
                    else:
                        st.error(f"**{len(st.session_state.history) - i}. {h['result']}**")
                with col2:
                    st.write(f"Confidence: {h['confidence']}")
                with col3:
                    if st.button("📄 View", key=f"view_{i}"):
                        st.info(f"**Text:** {h['text']}")
                
                st.caption(f"📝 {h['text']}")
                st.markdown("---")
        
        # Export history
        if st.button("📥 Export History to CSV"):
            hist_df = pd.DataFrame(st.session_state.history)
            csv = hist_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download CSV",
                csv,
                "prediction_history.csv",
                "text/csv"
            )

# ---------------------------------
# ABOUT PAGE
# ---------------------------------
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📘 Project Title
        **A Comparative Evaluation of Machine Learning Approaches for Fake News Classification**
        
        ### 👥 Team Members
        - W.M.T. Dilmini  
        - D.M.J. Jaya Sri  
        - J.M.M. Prabash  
        - W.R.U. Sethmini  
        
        ### 🧠 Technologies Used
        - **Frontend:** Streamlit
        - **Backend:** Python
        - **ML Library:** Scikit-learn
        - **NLP:** NLTK
        - **Data Processing:** Pandas, NumPy
        - **Reporting:** ReportLab
        
        ### 📊 Model Performance
        - **Best Model:** Logistic Regression
        - **Accuracy:** 97.79%
        - **Features:** TF-IDF with N-grams (1,2)
        - **Training Data:** 20,000 labeled news articles
        
        ### 🎯 Objective
        To build an intelligent system capable of identifying fake news using machine learning techniques, helping users verify the authenticity of online news content.
        
        ### 📅 Course Information
        - **Course:** IT41033 - Mini Project
        - **Institution:** SLIIT
        - **Year:** 2024
        """)
    
    with col2:
        st.info("""
        ### 📞 Contact
        For questions or feedback:
        - Email: project@fake news detector.com
        - GitHub: github.com/fakenews-detector
        """)
        
        st.success("""
        ### ✅ Project Status
        - **Status:** Fully Functional
        - **Version:** 2.0
        - **Last Updated:** April 2026
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center;'>© 2026 Fake News Detection System | All Rights Reserved</p>",
        unsafe_allow_html=True
    )

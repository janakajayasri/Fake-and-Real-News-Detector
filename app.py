import streamlit as st
import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from datetime import datetime
import base64
from fpdf import FPDF
import io

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------
# Download NLTK Data
# ---------------------------------
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)

download_nltk_data()

# ---------------------------------
# Load Model & Vectorizer
# ---------------------------------
@st.cache_resource
def load_model():
    try:
        with open("lr_ngram_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("vectorizer_ngram.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        st.error("❌ Model files not found! Please ensure `lr_ngram_model.pkl` and `vectorizer_ngram.pkl` are in the app directory.")
        st.stop()

model, vectorizer = load_model()

# ---------------------------------
# Preprocessing
# ---------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words and len(word) > 2
    ]
    return " ".join(tokens)

# ---------------------------------
# PDF Report Generator
# ---------------------------------
def generate_pdf_report(text, prediction, confidence, processed_text, probs):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Fake News Detection Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Prediction Result", ln=True)
    pdf.set_font("Arial", '', 12)
    result_text = "REAL NEWS" if prediction == "REAL" else "FAKE NEWS"
    color = "Green" if prediction == "REAL" else "Red"
    pdf.cell(0, 10, f"Result: {result_text}", ln=True)
    pdf.cell(0, 10, f"Confidence: {confidence:.2f}%", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Original Text Preview:", ln=True)
    pdf.set_font("Arial", '', 10)
    preview = text[:500] + "..." if len(text) > 500 else text
    pdf.multi_cell(0, 5, preview)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Probability Distribution:", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Fake: {probs[0]*100:.2f}%", ln=True)
    pdf.cell(0, 10, f"Real: {probs[1]*100:.2f}%", ln=True)
    
    # Save to bytes
    pdf_output = io.BytesIO()
    pdf.output(pdf_output, 'S')
    pdf_output.seek(0)
    return pdf_output

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------
# Sidebar
# ---------------------------------
st.sidebar.title("🧭 Navigation")

# Theme Toggle
theme = st.sidebar.toggle("🌙 Dark Mode", value=True)
if not theme:
    st.markdown("""
        <style>
        .stApp { background-color: white; color: black; }
        </style>
    """, unsafe_allow_html=True)

page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🔍 Prediction", "📊 Visualization", "🧠 Model Insights", "📜 History", "ℹ️ About"]
)

# ---------------------------------
# HOME PAGE
# ---------------------------------
if page == "🏠 Home":
    st.title("📰 Fake News Detection System")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Welcome to the Intelligent Fake News Detector!")
        st.markdown("""
        This application uses **Logistic Regression** with **TF-IDF N-grams** to accurately classify news articles as:
        
        - 🟢 **Real News**
        - 🔴 **Fake News**
        
        **Best Accuracy Achieved:** **97.79%**
        """)
        st.info("👉 Use the sidebar to navigate through different sections.")
    
    with col2:
        st.image("https://img.icons8.com/fluency/240/fake-news.png", width=180)
    
    st.success("✅ Academic Mini Project - IT41033")

# ---------------------------------
# PREDICTION PAGE
# ---------------------------------
elif page == "🔍 Prediction":
    st.title("🔍 Predict News Authenticity")
    
    # Input method tabs
    input_method = st.radio("Choose input method:", ["✍️ Paste Text", "📁 Upload .txt File"], horizontal=True)
    
    if input_method == "✍️ Paste Text":
        user_input = st.text_area(
            "Paste the News Article Here:",
            height=350,
            placeholder="Enter or paste the full news text..."
        )
    else:
        uploaded_file = st.file_uploader("Upload a .txt file containing the news article", type=["txt"])
        if uploaded_file is not None:
            user_input = uploaded_file.read().decode("utf-8")
            st.text_area("Uploaded Content:", value=user_input[:1000] + "..." if len(user_input) > 1000 else user_input, height=200, disabled=True)
        else:
            user_input = ""

    col1, col2 = st.columns([1, 4])
    with col1:
        predict_btn = st.button("🚀 Predict", type="primary", use_container_width=True)

    if predict_btn:
        if not user_input or user_input.strip() == "":
            st.warning("⚠️ Please enter or upload some news text to analyze.")
        else:
            with st.spinner("Analyzing news article..."):
                processed_text = preprocess_text(user_input)
                X = vectorizer.transform([processed_text])
                
                prediction = model.predict(X)[0]
                probabilities = model.predict_proba(X)[0]
                
                result = "REAL" if prediction == 1 else "FAKE"
                confidence = probabilities[prediction] * 100
                
                # Display Result
                if prediction == 1:
                    st.success(f"🟢 **REAL NEWS**", icon="✅")
                    st.balloons()
                else:
                    st.error(f"🔴 **FAKE NEWS**", icon="🚨")
                
                st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
                st.progress(float(confidence) / 100)
                
                # Save to session state
                st.session_state.probs = probabilities
                st.session_state.text = user_input
                st.session_state.processed = processed_text
                st.session_state.prediction = result
                st.session_state.confidence = confidence
                
                # Add to history
                st.session_state.history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "prediction": result,
                    "confidence": round(confidence, 2),
                    "text_preview": user_input[:150] + "..." if len(user_input) > 150 else user_input
                })
                
                # PDF Download Button
                pdf_file = generate_pdf_report(user_input, result, confidence, processed_text, probabilities)
                st.download_button(
                    label="📄 Download Prediction Report (PDF)",
                    data=pdf_file.getvalue(),
                    file_name=f"fake_news_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

# ---------------------------------
# VISUALIZATION PAGE
# ---------------------------------
elif page == "📊 Visualization":
    st.title("📊 Prediction Visualizations")
    if "probs" not in st.session_state:
        st.info("Please make a prediction first on the **Prediction** page.")
        st.stop()
    
    probs = st.session_state.probs
    text = st.session_state.text
    
    tab1, tab2 = st.tabs(["📈 Probability Distribution", "📝 Text Statistics"])
    
    with tab1:
        df_prob = pd.DataFrame({
            "Class": ["Fake", "Real"],
            "Probability": probs
        })
        st.bar_chart(df_prob.set_index("Class"), use_container_width=True, color="#1f77b4")
    
    with tab2:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Word Count", len(text.split()))
        with col2:
            st.metric("Character Count", len(text))
        with col3:
            st.metric("Predicted As", st.session_state.get("prediction", "—"))

# ---------------------------------
# MODEL INSIGHTS PAGE
# ---------------------------------
elif page == "🧠 Model Insights":
    st.title("🧠 Model Explainability")
    if "text" not in st.session_state:
        st.info("Run a prediction first to see influential words.")
        st.stop()
    
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    
    top_indices = np.argsort(np.abs(coefficients))[-15:]
    words = feature_names[top_indices]
    scores = coefficients[top_indices]
    
    df_insight = pd.DataFrame({
        "Word": words,
        "Impact Score": scores
    }).sort_values(by="Impact Score", ascending=False)
    
    st.subheader("Top 15 Most Influential Words")
    st.bar_chart(df_insight.set_index("Word")["Impact Score"], use_container_width=True)
    
    st.markdown("""
    **How to interpret:**
    - **Positive Impact** → pushes the model toward **Real News**
    - **Negative Impact** → pushes the model toward **Fake News**
    """)

# ---------------------------------
# HISTORY PAGE
# ---------------------------------
elif page == "📜 History":
    st.title("📜 Prediction History")
    
    if not st.session_state.history:
        st.info("No predictions made yet. Go to the **Prediction** page and analyze some news!")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

# ---------------------------------
# ABOUT PAGE
# ---------------------------------
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    ### Fake News Detection System
    **Project Title:** A Comparative Evaluation of Machine Learning Approaches for Fake News Classification
    
    **Team Members:**
    - W.M.T. Dilmini (ITBIN-2211-0111)
    - D.M.J. Jaya Sri (ITBIN-2211-0125)
    - J.M.M. Prabash (ITBIN-2211-0331)
    - W.R.U. Sethmini (ITBIN-2211-0101)
    
    **Best Model:** Logistic Regression with TF-IDF N-grams  
    **Accuracy:** **97.79%**
    
    ### Technologies Used
    - Python + Streamlit
    - Scikit-learn
    - NLTK
    - Pandas & NumPy
    - FPDF (for report generation)
    """)
    st.success("✔ Fully functional Streamlit web application for academic submission")

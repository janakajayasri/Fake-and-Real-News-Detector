import streamlit as st
import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ---------------------------------
# Download NLTK Data (First Run)
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
        st.error("❌ Model files not found! Please make sure `lr_ngram_model.pkl` and `vectorizer_ngram.pkl` are in the same folder as `app.py`.")
        st.stop()
model, vectorizer = load_model()
# ---------------------------------
# Preprocessing Function
# ---------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text) # Remove punctuation
    tokens = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words and len(word) > 2
    ]
    return " ".join(tokens)
# ---------------------------------
# Sidebar Navigation
# ---------------------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🔍 Prediction", "📊 Visualization", "🧠 Model Insights", "ℹ️ About"]
)
# ---------------------------------
# HOME PAGE
# ---------------------------------
if page == "🏠 Home":
    st.title("📰 Fake News Detection System")
    st.markdown("### Welcome to the Intelligent Fake News Detector!")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        This application uses **Logistic Regression** with **TF-IDF N-grams** to classify news articles as:
       
        - 🟢 **Real News**
        - 🔴 **Fake News**
       
        **Accuracy:** ~97.79%
        """)
       
        st.info("👉 Use the sidebar to navigate through different sections.")
    with col2:
        st.image("1742547741365.png", width=150)
    st.success("✅ Academic Project - IT41033 Mini Project")
# ---------------------------------
# PREDICTION PAGE
# ---------------------------------
elif page == "🔍 Prediction":
    st.title("🔍 Predict News Authenticity")
    user_input = st.text_area(
        "Paste the News Article Here:",
        height=300,
        placeholder="Enter or paste the full news text..."
    )
    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        predict_btn = st.button("🚀 Predict", type="primary", use_container_width=True)
    if predict_btn:
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some news text to analyze.")
        else:
            with st.spinner("Analyzing news..."):
                # Preprocess
                processed_text = preprocess_text(user_input)
               
                # Vectorize
                X = vectorizer.transform([processed_text])
               
                # Predict
                prediction = model.predict(X)[0]
                probabilities = model.predict_proba(X)[0]
               
                # Results
                result = "REAL" if prediction == 1 else "FAKE"
                confidence = probabilities[prediction] * 100
               
                # Display Result
                if prediction == 1:
                    st.success(f"🟢 **REAL NEWS**")
                    st.balloons()
                else:
                    st.error(f"🔴 **FAKE NEWS**")
               
                st.metric(label="Confidence", value=f"{confidence:.2f}%")
                st.progress(float(confidence)/100)
                # Save to session state for Visualization & Insights
                st.session_state["probs"] = probabilities
                st.session_state["text"] = user_input
                st.session_state["processed"] = processed_text
                st.session_state["prediction"] = result
# ---------------------------------
# VISUALIZATION PAGE
# ---------------------------------
elif page == "📊 Visualization":
    st.title("📊 Prediction Visualizations")
    if "probs" not in st.session_state:
        st.info("Please make a prediction first on the **Prediction** page.")
        st.stop()
    probs = st.session_state["probs"]
    text = st.session_state["text"]
    tab1, tab2 = st.tabs(["📈 Probability Distribution", "📝 Text Statistics"])
    with tab1:
        df_prob = pd.DataFrame({
            "Class": ["Fake", "Real"],
            "Probability": probs
        })
        st.bar_chart(df_prob.set_index("Class"), use_container_width=True)
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
    # Top 15 influential words
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
    **How to read this chart:**
    - Positive scores → push prediction toward **Real News**
    - Negative scores → push prediction toward **Fake News**
    """)
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
    - NLTK (Lemmatization + Stopwords)
    - Pandas & NumPy
    """)
    st.success("✔ Fully functional Streamlit web application for academic submission")

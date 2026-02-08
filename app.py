import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# ---------------------------------
# Load Model
# ---------------------------------
try:
    with open("lr_ngram_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer_ngram.pkl", "rb") as f:
        vectorizer = pickle.load(f)
except:
    st.error("Model or vectorizer not found!")
    st.stop()

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
        if word not in stop_words
    ]
    return " ".join(tokens)

# ---------------------------------
# Sidebar Navigation (SLIDES)
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

    st.markdown("""
    ### Welcome!
    This application uses **Machine Learning** to classify news as:

    - 🟢 **Real News**
    - 🔴 **Fake News**

    ### Features
    - Logistic Regression model  
    - TF-IDF N-gram text features  
    - High accuracy (97.79%)  
    - Interactive visualizations  

    👉 Use the **navigation menu** to move between slides.
    """)

# ---------------------------------
# PREDICTION PAGE
# ---------------------------------
elif page == "🔍 Prediction":
    st.title("🔍 News Prediction")

    user_input = st.text_area(
        "Enter News Text",
        height=220,
        placeholder="Paste the news article here..."
    )

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter news text")
        else:
            processed = preprocess_text(user_input)
            X = vectorizer.transform([processed])

            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]

            result = "REAL" if prediction == 1 else "FAKE"
            confidence = probabilities[prediction]

            if prediction == 1:
                st.success(f"🟢 REAL NEWS")
            else:
                st.error(f"🔴 FAKE NEWS")

            st.write(f"**Confidence:** {confidence*100:.2f}%")
            st.progress(float(confidence))

            st.session_state["probs"] = probabilities
            st.session_state["text"] = user_input
            st.session_state["X"] = X

# ---------------------------------
# VISUALIZATION PAGE
# ---------------------------------
elif page == "📊 Visualization":
    st.title("📊 Prediction Visualizations")

    if "probs" not in st.session_state:
        st.info("Please run a prediction first.")
    else:
        probs = st.session_state["probs"]

        tab1, tab2 = st.tabs(["📈 Probability Chart", "📌 Text Stats"])

        with tab1:
            df = pd.DataFrame({
                "Class": ["Fake", "Real"],
                "Probability": probs
            })
            st.bar_chart(df.set_index("Class"))

        with tab2:
            text = st.session_state["text"]

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Word Count", len(text.split()))
            with col2:
                st.metric("Character Count", len(text))

# ---------------------------------
# MODEL INSIGHTS PAGE
# ---------------------------------
elif page == "🧠 Model Insights":
    st.title("🧠 Model Explainability")

    if "X" not in st.session_state:
        st.info("Run a prediction to view insights.")
    else:
        feature_names = vectorizer.get_feature_names_out()
        coefficients = model.coef_[0]

        top_indices = np.argsort(np.abs(coefficients))[-10:]
        words = feature_names[top_indices]
        scores = coefficients[top_indices]

        df = pd.DataFrame({
            "Word": words,
            "Impact Score": scores
        }).sort_values(by="Impact Score", ascending=False)

        st.subheader("Top Influential Words")
        st.bar_chart(df.set_index("Word"))

        st.markdown("""
        **Explanation:**  
        These words had the strongest influence on the model’s decision.
        """)

# ---------------------------------
# ABOUT PAGE
# ---------------------------------
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")

    st.markdown("""
    **Project Title:** Fake News Detection System  

    **Model:** Logistic Regression  
    **Feature Extraction:** TF-IDF with N-grams  
    **Accuracy:** 97.79%  

    **Developed for:** IT41033 Mini Project  

    ### Technologies Used
    - Python  
    - Streamlit  
    - Scikit-learn  
    - NLTK  

    ### Objective
    To detect misleading and fake news articles using machine learning.
    """)

    st.success("✔ Academic-ready application with navigation slides")

import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

# Load the trained model and vectorizer
try:
    with open('lr_ngram_model.pkl', 'rb') as f:
        lr_ngram = pickle.load(f)
    with open('vectorizer_ngram.pkl', 'rb') as f:
        vectorizer_ngram = pickle.load(f)
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'lr_ngram_model.pkl' and 'vectorizer_ngram.pkl' are in the directory.")
    st.stop()

# Preprocessing function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# Streamlit app
st.title("Fake News Detector")
st.write("Enter a news article text to classify it as Fake or Real. Deployed on August 16, 2025, 08:50 PM +0530.")

# User input
user_input = st.text_area("News Text", "Type or paste your news article here...")

if st.button("Predict"):
    if user_input:
        # Preprocess input
        processed_text = preprocess_text(user_input)
        # Transform to TF-IDF
        X_input = vectorizer_ngram.transform([processed_text])
        # Predict
        prediction = lr_ngram.predict(X_input)[0]
        probability = lr_ngram.predict_proba(X_input)[0][prediction]
        # Display result
        result = "Real" if prediction == 1 else "Fake"
        st.success(f"Prediction: {result} (Confidence: {probability:.2f})")
    else:
        st.error("Please enter some text to predict.")

# Additional information
st.write("Model: Logistic Regression with N-grams, Test Accuracy: 0.9779")
st.write("Model Saved: August 16, 2025, 08:18 PM +0530")
st.write("Note: This is a simulated deployment for the IT41033 Mini-Project by xAI.")
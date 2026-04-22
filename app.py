import streamlit as st
import pickle
from preprocess import preprocess_text

# Load model
model = pickle.load(open("lr_ngram_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer_ngram.pkl", "rb"))

st.title("📰 Fake News Detection App")

text = st.text_area("Enter News Article")

if st.button("Predict"):
    if text.strip() != "":
        processed = preprocess_text(text)
        vector = vectorizer.transform([processed])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.success("✅ Real News")
        else:
            st.error("❌ Fake News")
    else:
        st.warning("Please enter text")

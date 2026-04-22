# ============================================
# Fake News Detection System (Final Version with Animations)
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
import time
from streamlit.components.v1 import html

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------
# ANIMATION FUNCTIONS
# ---------------------------------

def add_loading_animation():
    """Adds a loading spinner animation"""
    loading_html = """
    <div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">
        <div class="loader"></div>
        <style>
            .loader {
                border: 4px solid #f3f3f3;
                border-radius: 50%;
                border-top: 4px solid #4CAF50;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
        <p>Analyzing news article...</p>
    </div>
    """
    return loading_html

def add_pulse_animation():
    """Adds pulse animation for prediction results"""
    pulse_html = """
    <style>
        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.05); opacity: 0.8; }
            100% { transform: scale(1); opacity: 1; }
        }
        .pulse-animation {
            animation: pulse 0.5s ease-in-out;
        }
    </style>
    """
    return pulse_html

def add_fade_in_animation():
    """Adds fade-in animation for content"""
    fade_html = """
    <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .fade-in {
            animation: fadeIn 0.8s ease-out;
        }
    </style>
    """
    return fade_html

def add_typing_animation():
    """Adds typing animation for text"""
    typing_html = """
    <style>
        @keyframes typing {
            from { width: 0; }
            to { width: 100%; }
        }
        @keyframes blink {
            50% { border-color: transparent; }
        }
        .typing-animation {
            overflow: hidden;
            white-space: nowrap;
            border-right: 2px solid #4CAF50;
            animation: typing 2s steps(40, end), blink 0.75s step-end infinite;
        }
    </style>
    """
    return typing_html

def add_floating_animation():
    """Adds floating animation for cards"""
    floating_html = """
    <style>
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }
        .float-animation {
            animation: float 3s ease-in-out infinite;
        }
    </style>
    """
    return floating_html

def add_shimmer_effect():
    """Adds shimmer effect for loading states"""
    shimmer_html = """
    <style>
        @keyframes shimmer {
            0% { background-position: -1000px 0; }
            100% { background-position: 1000px 0; }
        }
        .shimmer {
            background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
            background-size: 1000px 100%;
            animation: shimmer 2s infinite;
        }
    </style>
    """
    return shimmer_html

def add_confetti_animation():
    """Adds confetti animation for real news"""
    confetti_js = """
    <script>
    function startConfetti() {
        var canvas = document.createElement('canvas');
        canvas.style.position = 'fixed';
        canvas.style.top = '0';
        canvas.style.left = '0';
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        canvas.style.pointerEvents = 'none';
        canvas.style.zIndex = '1000';
        document.body.appendChild(canvas);
        
        var ctx = canvas.getContext('2d');
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        
        var particles = [];
        for(var i = 0; i < 150; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height - canvas.height,
                size: Math.random() * 5 + 2,
                speedY: Math.random() * 5 + 2,
                speedX: Math.random() * 2 - 1,
                color: 'hsl(' + Math.random() * 360 + ', 100%, 50%)'
            });
        }
        
        function animate() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            for(var i = 0; i < particles.length; i++) {
                var p = particles[i];
                ctx.fillStyle = p.color;
                ctx.fillRect(p.x, p.y, p.size, p.size);
                p.y += p.speedY;
                p.x += p.speedX;
                if(p.y > canvas.height) {
                    p.y = -p.size;
                    p.x = Math.random() * canvas.width;
                }
            }
            requestAnimationFrame(animate);
        }
        animate();
        
        setTimeout(function() {
            canvas.remove();
        }, 3000);
    }
    </script>
    """
    return confetti_js

def add_sad_animation():
    """Adds sad animation for fake news"""
    sad_html = """
    <style>
        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
            20%, 40%, 60%, 80% { transform: translateX(5px); }
        }
        .shake-animation {
            animation: shake 0.5s ease-in-out;
        }
    </style>
    """
    return sad_html

# ---------------------------------
# CUSTOM CSS WITH ANIMATIONS
# ---------------------------------
st.markdown("""
<style>
/* Smooth transitions for all elements */
* {
    transition: all 0.3s ease;
}

/* Button hover animations */
.stButton button {
    transition: all 0.3s ease;
    transform: scale(1);
}

.stButton button:hover {
    transform: scale(1.05);
    transition: all 0.3s ease;
}

/* Card hover animations */
[data-testid="stMetric"] {
    transition: all 0.3s ease;
    cursor: pointer;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-5px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
}

/* Progress bar animation */
.stProgress > div > div > div > div {
    transition: width 0.5s ease;
}

/* Sidebar animation */
[data-testid="stSidebar"] {
    transition: all 0.3s ease;
}

/* Text area focus animation */
.stTextArea textarea:focus, .stTextInput input:focus {
    transform: scale(1.02);
    transition: all 0.3s ease;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# DARK MODE TOGGLE WITH ANIMATION
# ---------------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)

if dark_mode != st.session_state.dark_mode:
    st.session_state.dark_mode = dark_mode
    st.balloons()  # Fun animation on theme switch
    time.sleep(0.5)
    st.rerun()

if st.session_state.dark_mode:
    st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117 !important;
        animation: fadeIn 0.5s ease;
    }
    
    .stApp * {
        color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #1E1E1E !important;
    }
    
    .stTextArea textarea, .stTextInput input {
        background-color: #2E2E2E !important;
        border: 1px solid #4A4A4A !important;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------
# INJECT ALL ANIMATIONS
# ---------------------------------
st.markdown(add_pulse_animation(), unsafe_allow_html=True)
st.markdown(add_fade_in_animation(), unsafe_allow_html=True)
st.markdown(add_typing_animation(), unsafe_allow_html=True)
st.markdown(add_floating_animation(), unsafe_allow_html=True)
st.markdown(add_shimmer_effect(), unsafe_allow_html=True)

# ---------------------------------
# DOWNLOAD NLTK
# ---------------------------------
@st.cache_resource
def download_nltk():
    try:
        nltk.data.find("corpora/stopwords")
        nltk.data.find("corpora/wordnet")
    except LookupError:
        with st.spinner("📥 Downloading NLTK data..."):
            nltk.download("stopwords", quiet=True)
            nltk.download("wordnet", quiet=True)

download_nltk()

# ---------------------------------
# LOAD MODEL WITH ANIMATION
# ---------------------------------
@st.cache_resource
def load_model():
    with st.spinner("🚀 Loading AI Model..."):
        time.sleep(0.5)  # Simulate loading for animation
        try:
            with open("lr_ngram_model.pkl", "rb") as f:
                model = pickle.load(f)
            with open("vectorizer_ngram.pkl", "rb") as f:
                vectorizer = pickle.load(f)
            return model, vectorizer
        except FileNotFoundError as e:
            st.error(f"❌ Model file not found: {e}")
            st.stop()

model, vectorizer = load_model()

# ---------------------------------
# NLP PREPROCESSING
# ---------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    if not text or not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text)
    
    return " ".join([
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in stop_words and len(w) > 2
    ])

# ---------------------------------
# PDF EXPORT FUNCTION
# ---------------------------------
def create_pdf(text, result, confidence):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            doc = SimpleDocTemplate(tmp_file.name)
            styles = getSampleStyleSheet()
            
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
# SIDEBAR WITH ANIMATIONS
# ---------------------------------
st.sidebar.markdown('<div class="fade-in">', unsafe_allow_html=True)
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
        if h['result'] == "REAL":
            st.sidebar.success(f"📰 {h['result']} ({h['confidence']})")
        else:
            st.sidebar.error(f"📰 {h['result']} ({h['confidence']})")
else:
    st.sidebar.info("No predictions yet")

# Sidebar Stats
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Quick Stats")
if st.session_state.history:
    real_count = sum(1 for h in st.session_state.history if h['result'] == "REAL")
    fake_count = len(st.session_state.history) - real_count
    st.sidebar.metric("Total Predictions", len(st.session_state.history))
    st.sidebar.metric("Real News", real_count)
    st.sidebar.metric("Fake News", fake_count)

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------
# HOME PAGE WITH ANIMATIONS
# ---------------------------------
if page == "🏠 Home":
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    
    # Animated title
    st.markdown('<h1 class="typing-animation">📰 Fake News Detection System</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="float-animation">', unsafe_allow_html=True)
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
        - ✅ Real-time prediction with animations
        - 📊 Visualization dashboard
        - 🧠 Model explainability
        - 📄 PDF report generation
        - 📜 Prediction history
        - 🌙 Dark mode with animations
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="float-animation">', unsafe_allow_html=True)
        st.info("🎓 **Academic Project**\n\nIT41033 - Mini Project\n\nSLIIT")
        st.metric("Model Performance", "97.79%", "Accuracy")
        
        # Animated welcome message
        if st.button("✨ Click for Welcome Animation ✨"):
            st.balloons()
            st.snow()
            st.success("Welcome to Fake News Detection System!")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------
# PREDICTION PAGE WITH ANIMATIONS
# ---------------------------------
elif page == "🔍 Prediction":
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.title("🔍 Predict News Authenticity")
    
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("📂 Upload .txt file", type=["txt"])
    
    with col2:
        example_text = st.button("📝 Load Example", use_container_width=True)
        if example_text:
            st.session_state.example_loaded = True
            st.toast("📰 Example article loaded!", icon="✅")
    
    if uploaded_file:
        user_text = uploaded_file.read().decode("utf-8")
        st.success("✅ File loaded successfully!")
        st.text_area("📄 File Content", user_text, height=200, key="file_content")
    elif 'example_loaded' in st.session_state and st.session_state.example_loaded:
        user_text = """The World Health Organization (WHO) announced today that regular exercise and a balanced diet remain the most effective ways to maintain cardiovascular health. According to a five-year study involving 10,000 participants, individuals who engaged in at least 150 minutes of moderate exercise per week showed a 30% lower risk of heart disease compared to sedentary individuals."""
        st.info("📝 Example article loaded")
        st.text_area("✏️ Edit Article", user_text, height=200, key="example_content")
        st.session_state.example_loaded = False
    else:
        user_text = st.text_area("✏️ Paste News Article", height=250, 
                                placeholder="Enter or paste a news article here...")
    
    if st.button("🚀 Predict", type="primary", use_container_width=True):
        if not user_text or not user_text.strip():
            st.warning("⚠️ Please enter some text to analyze!")
        else:
            # Show loading animation
            with st.spinner("🔍 Analyzing article with AI..."):
                time.sleep(0.5)  # Brief pause for animation
                
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
                    
                    # Show result with animations
                    st.markdown("---")
                    st.subheader("📊 Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if pred == 1:
                            st.markdown('<div class="pulse-animation">', unsafe_allow_html=True)
                            st.success("### 🟢 REAL NEWS")
                            # Trigger confetti for real news
                            st.balloons()
                            st.toast("🎉 Great! This appears to be real news!", icon="✅")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="shake-animation">', unsafe_allow_html=True)
                            st.error("### 🔴 FAKE NEWS")
                            st.toast("⚠️ Warning! This appears to be fake news!", icon="⚠️")
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("Confidence Score", f"{confidence:.2f}%")
                    
                    with col3:
                        if confidence > 80:
                            st.success("✅ High Confidence")
                            st.progress(confidence / 100)
                        elif confidence > 60:
                            st.warning("⚠️ Moderate Confidence")
                            st.progress(confidence / 100)
                        else:
                            st.error("❌ Low Confidence")
                            st.progress(confidence / 100)
                    
                    # Detailed analysis with animation
                    with st.expander("📈 Detailed Analysis"):
                        st.write(f"**Processed Text Length:** {len(processed.split())} words")
                        st.write(f"**Original Text Length:** {len(user_text.split())} words")
                        
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
                            with st.spinner("📝 Generating PDF..."):
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
                                    os.unlink(pdf_path)
                                    st.success("PDF generated successfully!")
                    
                    with col2:
                        if st.button("🔄 Clear & New", use_container_width=True):
                            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------
# VISUALIZATION PAGE WITH ANIMATIONS
# ---------------------------------
elif page == "📊 Visualization":
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.title("📊 Prediction Visualization")
    
    if st.session_state.probs is None:
        st.info("ℹ️ No prediction data available. Please run a prediction first on the **Prediction** page.")
        if st.button("Go to Prediction Page", use_container_width=True):
            st.switch_page("app.py")
    else:
        st.subheader("🎯 Probability Distribution")
        df = pd.DataFrame({
            "Class": ["Fake News", "Real News"],
            "Probability": st.session_state.probs
        })
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown('<div class="float-animation">', unsafe_allow_html=True)
            st.bar_chart(df.set_index("Class"))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            prediction = "REAL" if st.session_state.probs[1] > 0.5 else "FAKE"
            st.metric("Prediction", prediction, f"{max(st.session_state.probs) * 100:.1f}% confidence")
        
        st.subheader("📈 Confidence Gauge")
        confidence = max(st.session_state.probs) * 100
        st.progress(confidence / 100)
        
        if len(st.session_state.history) > 1:
            st.subheader("📉 Historical Trend")
            hist_df = pd.DataFrame([
                {"Prediction": i+1, 
                 "Confidence": float(h['confidence'].replace('%', ''))}
                for i, h in enumerate(st.session_state.history)
            ])
            st.line_chart(hist_df.set_index("Prediction"))
    
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------
# MODEL INSIGHTS PAGE
# ---------------------------------
elif page == "🧠 Model Insights":
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.title("🧠 Model Explainability")
    
    st.markdown("""
    ### How does the model make decisions?
    This section shows which words most influence the model's predictions.
    - **Positive impact** → Indicates REAL news
    - **Negative impact** → Indicates FAKE news
    """)
    
    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]
    
    top_positive_idx = np.argsort(coef)[-15:][::-1]
    top_negative_idx = np.argsort(coef)[:15]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="float-animation">', unsafe_allow_html=True)
        st.subheader("📰 Real News Indicators (Positive Impact)")
        pos_df = pd.DataFrame({
            "Word": feature_names[top_positive_idx],
            "Impact Score": coef[top_positive_idx]
        })
        st.dataframe(pos_df, use_container_width=True)
        st.bar_chart(pos_df.set_index("Word"))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="float-animation">', unsafe_allow_html=True)
        st.subheader("⚠️ Fake News Indicators (Negative Impact)")
        neg_df = pd.DataFrame({
            "Word": feature_names[top_negative_idx],
            "Impact Score": coef[top_negative_idx]
        })
        st.dataframe(neg_df, use_container_width=True)
        st.bar_chart(neg_df.set_index("Word"))
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------
# HISTORY PAGE WITH ANIMATIONS
# ---------------------------------
elif page == "📜 History":
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.title("📜 Prediction History")
    
    if not st.session_state.history:
        st.info("ℹ️ No predictions yet. Go to the **Prediction** page to get started!")
        if st.button("Go to Prediction", use_container_width=True):
            st.switch_page("app.py")
    else:
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
                st.session_state.history = []
                st.session_state.probs = None
                st.session_state.last_prediction = None
                st.rerun()
        
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
        
        if st.button("📥 Export History to CSV", use_container_width=True):
            hist_df = pd.DataFrame(st.session_state.history)
            csv = hist_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download CSV",
                csv,
                "prediction_history.csv",
                "text/csv",
                use_container_width=True
            )
    
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------
# ABOUT PAGE WITH ANIMATIONS
# ---------------------------------
elif page == "ℹ️ About":
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.title("ℹ️ About This Project")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="float-animation">', unsafe_allow_html=True)
        st.markdown("""
        ### 📘 Project Title
        **A Comparative Evaluation of Machine Learning Approaches for Fake News Classification**
        
        ### 👥 Team Members
        - W.M.T. Dilmini  
        - D.M.J. Jaya Sri  
        - J.M.M. Prabash  
        - W.R.U. Sethmini  
        
        ### 🧠 Technologies Used
        - **Frontend:** Streamlit with CSS Animations
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
        To build an intelligent system capable of identifying fake news using machine learning techniques.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="float-animation">', unsafe_allow_html=True)
        st.info("""
        ### 📞 Contact
        For questions or feedback:
        - Email: project@fakenewsdetector.com
        - GitHub: github.com/fakenews-detector
        """)
        
        st.success("""
        ### ✅ Project Status
        - **Status:** Fully Functional with Animations
        - **Version:** 3.0 (Animated)
        - **Last Updated:** April 2026
        """)
        
        # Fun animation button
        if st.button("🎉 Celebrate with Animation 🎉"):
            st.balloons()
            st.snow()
            st.toast("Thanks for using Fake News Detection System!", icon="🎊")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; animation: pulse 2s infinite;'>© 2026 Fake News Detection System | All Rights Reserved</p>",
        unsafe_allow_html=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

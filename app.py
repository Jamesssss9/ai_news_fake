import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
import os

# Page Configuration
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
        padding: 2rem;
    }
    .stTitle {
        color: #2c3e50;
        font-size: 2.5rem !important;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .real-news {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .fake-news {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and vectorizer
@st.cache_resource
def load_models():
    vectorizer = joblib.load("vectorizer.jb")
    model = joblib.load("lr_model.jb")
    return vectorizer, model

vectorizer, model = load_models()

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Header
st.markdown("# 📰 Fake News Detector")
st.markdown("**AI-Powered News Authenticity Checker**")
st.divider()

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence required to make a prediction"
    )
    
    st.divider()
    st.markdown("### 📊 Model Information")
    st.info(f"""
    **Model Type:** Logistic Regression
    **Vectorizer:** TF-IDF
    **Max Features:** 5000
    **Status:** ✅ Loaded Successfully
    """)
    
    st.divider()
    if st.button("🗑️ Clear History", key="clear_history"):
        st.session_state.history = []
        st.success("History cleared!")

# Main Content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Enter News Article")
    inputn = st.text_area(
        "Paste your news article here:",
        height=200,
        placeholder="Enter the news article text you want to analyze..."
    )

with col2:
    st.markdown("### 📈 Quick Stats")
    st.metric("Articles Analyzed", len(st.session_state.history))
    if st.session_state.history:
        fake_count = sum(1 for p in st.session_state.history if p['prediction'] == 0)
        real_count = len(st.session_state.history) - fake_count
        st.metric("Fake Detected", fake_count)
        st.metric("Real Detected", real_count)

st.divider()

# Prediction Section
if st.button("🔍 Check News", use_container_width=True, type="primary"):
    if inputn.strip():
        # Vectorize and predict
        transform_input = vectorizer.transform([inputn])
        prediction = model.predict(transform_input)[0]
        confidence = model.predict_proba(transform_input)[0]
        
        # Determine prediction label and confidence
        is_real = prediction == 1
        confidence_score = confidence[1] if is_real else confidence[0]
        
        # Store in history
        history_item = {
            'text': inputn[:100] + "..." if len(inputn) > 100 else inputn,
            'prediction': prediction,
            'confidence': confidence_score,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state.history.insert(0, history_item)
        
        # Display Results
        st.divider()
        st.markdown("### 🎯 Prediction Result")
        
        if is_real:
            st.markdown("""
            <div class="prediction-box real-news">
                <h3>✅ This News is REAL</h3>
                <p>The article appears to be authentic news based on our analysis.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="prediction-box fake-news">
                <h3>⚠️ This News is FAKE</h3>
                <p>The article shows signs of being false or misleading.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Confidence Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Overall Confidence",
                f"{confidence_score:.2%}",
                delta="High" if confidence_score > 0.8 else "Medium" if confidence_score > 0.6 else "Low"
            )
        with col2:
            st.metric("Probability: Real", f"{confidence[1]:.2%}")
        with col3:
            st.metric("Probability: Fake", f"{confidence[0]:.2%}")
        
        # Confidence Bar
        st.markdown("**Confidence Distribution:**")
        confidence_df = pd.DataFrame({
            'Category': ['Fake', 'Real'],
            'Confidence': [confidence[0], confidence[1]]
        })
        st.bar_chart(confidence_df.set_index('Category'))
        
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# History Section
if st.session_state.history:
    st.divider()
    st.markdown("### 📋 Recent Predictions")
    
    history_df = pd.DataFrame(st.session_state.history)
    history_df['Prediction'] = history_df['prediction'].apply(lambda x: '✅ Real' if x == 1 else '⚠️ Fake')
    history_df['Confidence'] = history_df['confidence'].apply(lambda x: f"{x:.2%}")
    
    display_df = history_df[['text', 'Prediction', 'Confidence', 'timestamp']].rename(columns={
        'text': 'Article Preview',
        'prediction': 'Result',
        'timestamp': 'Time'
    })
    
    st.dataframe(display_df, use_container_width=True, hide_index=True) 
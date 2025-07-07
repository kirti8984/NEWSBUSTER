import streamlit as st
import joblib
import numpy as np
import base64
import time
import pandas as pd
import re
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier



# ----------------- Load model and vectorizer -----------------
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ----------------- Set background image -----------------
def set_bg_image(image_path):
    with open(image_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    bg_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-attachment: fixed;
        font-family: 'Poppins', sans-serif;
        animation: fadein 2s ease-in;
    }}
    @keyframes fadein {{
        from {{opacity: 0;}}
        to {{opacity: 1;}}
    }}
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)

set_bg_image("imagepro.jpg")

# ----------------- Page Configuration -----------------
st.set_page_config(page_title="NewsBuster", layout="centered")

# ----------------- Fonts and Styles -----------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;600;800&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap" rel="stylesheet">
<style>
    @keyframes soft-glow {
        0% {
            text-shadow: 0 0 10px lightblue, 0 0 20px aqua, 0 0 30px lightcyan;
        }
        50% {
            text-shadow: 0 0 15px skyblue, 0 0 30px cyan, 0 0 40px lightblue;
        }
        100% {
            text-shadow: 0 0 10px lightblue, 0 0 20px aqua, 0 0 30px lightcyan;
        }
    }

    @keyframes label-glow {
        0% { text-shadow: 0 0 5px lightskyblue; }
        50% { text-shadow: 0 0 10px deepskyblue; }
        100% { text-shadow: 0 0 5px lightskyblue; }
    }

    .stTextInput > div > div > input {
        background-color: #ffffffcc;
        border: 2px solid #0047AB;
        padding: 10px;
        border-radius: 8px;
        font-size: 16px;
        color: #000;
    }

    .stButton button {
        background-color: #0047AB;
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 8px;
        font-weight: bold;
        transition: 0.3s ease-in-out;
    }

    .stButton button:hover {
        background-color: #002f6c;
        box-shadow: 0 0 15px #0047AB;
    }

    .custom-label {
        color: brown;
        font-size: 22px;
        font-weight: bold;
        display: block;
        text-align: center;
        animation: label-glow 2s infinite alternate;
    }

    .result-text {
        font-size: 26px;
        font-weight: bold;
        color: #000000;
        margin-bottom: 10px;
    }

    .confidence-text {
        font-size: 18px;
        color: #111;
        margin: 6px 0;
    }

    .feedback-style {
        background: rgba(255,255,255,0.85);
        padding: 1rem;
        border-radius: 10px;
        color: #000000;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Header -----------------
st.markdown("""
    <div style="text-align: center; background: rgba(255,255,255,0.75); padding: 2rem; border-radius: 12px; margin-bottom: 30px;">
        <h1 style="font-size: 5rem; font-family: 'Bebas Neue', cursive; color: #0047AB; margin-bottom: 0;
                   animation: soft-glow 2s infinite alternate;">
            NewsBuster
        </h1>
        <p style="font-size: 1.4rem; color: #333333;">Turn confusion into clarity — verify what you read</p>
    </div>
""", unsafe_allow_html=True)

# ----------------- Input Section -----------------
st.markdown("""<label class='custom-label'>📝 Please enter a news headline to check its authenticity...</label>""", unsafe_allow_html=True)
user_input = st.text_input("", max_chars=300)

if user_input.strip() != "":
    with st.spinner("Analyzing the news headline..."):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)
        vectorized_input = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_input)[0]
        probabilities = model.predict_proba(vectorized_input)[0]

    fake_prob = probabilities[0] * 100
    real_prob = probabilities[1] * 100
    confidence = max(probabilities)
    result = "✅ REAL" if prediction == 1 else "❌ FAKE"

    # Confidence interpretation
    if confidence >= 0.8:
        interpretation = "High Confidence"
    elif confidence >= 0.6:
        interpretation = "Moderate Confidence"
    else:
        interpretation = "Low Confidence"

    # ----------------- Display Result -----------------
    st.markdown(f"""
        <div style="text-align:center; background: rgba(255,255,255,0.9); padding: 1.5rem; border-radius: 12px;">
            <div class="result-text">{result}</div>
            <div class="confidence-text">🔴 Fake Probability: {fake_prob:.2f}%</div>
            <div class="confidence-text">🟢 Real Probability: {real_prob:.2f}%</div>
            <div class="confidence-text"><b>Model Confidence:</b> {interpretation}</div>
        </div>
    """, unsafe_allow_html=True)

# ----------------- Feedback Section -----------------
st.markdown("""<label style='color: black; font-size: 16px; font-weight: bold;'>Your Feedback</label>""", unsafe_allow_html=True)
feedback = st.text_area("", placeholder="Write your suggestions or comments here...", height=100)

if st.button("Submit Feedback"):
    st.success("✅ Thank you for your feedback!")
    st.markdown(f"""
        <div class="feedback-style">
            <b>Your Feedback:</b><br>
            {feedback}
        </div>
    """, unsafe_allow_html=True)

# ----------------- About Us -----------------
st.markdown("""
    <div style="margin-top: 5rem; padding: 2rem; background-color: rgba(0,0,0,0.65); color: white; border-radius: 12px; font-family: 'Comic Sans MS', cursive;">
        <h4 style="margin-bottom: 0.5rem;">About Us</h4>
        <p>Kirti Laxmi<br>
        MCO22388<br>
        B.E (Computer Science Engineering)<br>
        Chandigarh College of Engineering and Technology</p>
        <b>Chandigarh, SEC-26</b>
        <div style="margin-top: 1rem;">Follow us:
            <a href="#" style="color:white; text-decoration:none; margin-right: 10px;">📘</a>
            <a href="#" style="color:white; text-decoration:none;">📸</a>
        </div>
    </div>
""", unsafe_allow_html=True)

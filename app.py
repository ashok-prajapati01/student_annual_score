import streamlit as st
import pickle
import numpy as np

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="ML Model Prediction",
    page_icon="🤖",
    layout="centered"
)

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#0f172a,#1e293b,#334155);
}

.main-card{
    background:white;
    padding:35px;
    border-radius:20px;
    box-shadow:0px 8px 30px rgba(0,0,0,0.35);
}

.title{
    text-align:center;
    color:#2563eb;
    font-size:40px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    margin-bottom:20px;
}

.stButton>button{
    width:100%;
    background:#2563eb;
    color:white;
    font-size:18px;
    border-radius:10px;
    padding:10px;
    border:none;
}

.stButton>button:hover{
    background:#1d4ed8;
}

.prediction{
    padding:15px;
    background:#d1fae5;
    color:#065f46;
    border-radius:12px;
    font-size:22px;
    text-align:center;
    font-weight:bold;
    margin-top:20px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# Load Model
# -------------------------
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# -------------------------
# UI
# -------------------------
st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.markdown('<div class="title">Machine Learning Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter the values below to make a prediction.</div>', unsafe_allow_html=True)

feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)

if st.button("Predict"):

    input_data = np.array([[feature1, feature2, feature3]])

    prediction = model.predict(input_data)[0]

    st.markdown(
        f'<div class="prediction">Prediction : {prediction:.4f}</div>',
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)

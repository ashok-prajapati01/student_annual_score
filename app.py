import streamlit as st
import pickle
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="🎓",
    layout="centered"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#1d3557,#457b9d,#a8dadc);
}

.main-card{
    background:white;
    padding:35px;
    border-radius:20px;
    box-shadow:0px 8px 25px rgba(0,0,0,0.35);
}

.title{
    text-align:center;
    color:#1d3557;
    font-size:40px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    margin-bottom:20px;
}

div.stButton > button{
    width:100%;
    background:#1d3557;
    color:white;
    font-size:18px;
    border-radius:12px;
    height:50px;
    border:none;
}

div.stButton > button:hover{
    background:#457b9d;
    color:white;
}

.result{
    background:#e9f7ef;
    color:#145a32;
    padding:18px;
    border-radius:12px;
    text-align:center;
    font-size:24px;
    font-weight:bold;
    box-shadow:0px 5px 15px rgba(0,0,0,0.2);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# -----------------------------
# UI
# -----------------------------
st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.markdown('<p class="title">🎓 Student Score Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict Final Score using KNN Regression</p>', unsafe_allow_html=True)

exam1 = st.number_input(
    "Exam 01 Score",
    min_value=0.0,
    max_value=100.0,
    step=1.0
)

exam2 = st.number_input(
    "Exam 02 Score",
    min_value=0.0,
    max_value=100.0,
    step=1.0
)

exam3 = st.number_input(
    "Exam 03 Score",
    min_value=0.0,
    max_value=100.0,
    step=1.0
)

if st.button("Predict Score"):

    features = np.array([[exam1, exam2, exam3]])

    prediction = model.predict(features)[0]

    st.markdown(
        f"""
        <div class="result">
        Predicted Final Score<br><br>
        {prediction:.2f}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

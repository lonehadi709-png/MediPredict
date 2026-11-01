
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ------------------------------------------------------
# Stylish Background + CSS
# ------------------------------------------------------

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1b2735, #3a506b, #5bc0be);
        color: white;
    }
    .title {
        font-size: 38px;
        font-weight: bold;
        text-align: center;
        color: #ffffff;
        padding: 10px;
    }
    .subtitle {
        font-size: 20px;
        text-align: center;
        color: #d8e3e7;
        margin-bottom: 25px;
    }
    .card {
        background: rgba(255,255,255,0.12);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    /* Input fields styling */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    color: #111827 !important;
    background-color: #ffffff !important;
}

/* Input labels */
.stTextInput label,
.stNumberInput label,
.stTextArea label {
    color: #ffffff !important;
}

/* Text area */
.stTextArea textarea {
    caret-color: #111827 !important;
}
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# Title
# ------------------------------------------------------

st.markdown(
    "<div class='title'>🩺 MediPredict — Smart Health Analysis</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>AI-powered Early Diabetes Risk Estimation</div>",
    unsafe_allow_html=True
)

st.write("---")

# ------------------------------------------------------
# Load Model
# ------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "diabetes_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# ------------------------------------------------------
# Input Form
# ------------------------------------------------------

with st.form("patient_form"):

    st.markdown(
        "<div class='card'><h3>🧍 Patient Details</h3>",
        unsafe_allow_html=True
    )

    name = st.text_input("Full Name")
    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    phone = st.text_input("Phone Number")
    address = st.text_area("Address")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='card'><h3>📊 Medical Measurements</h3>",
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 0)
        glucose = st.number_input("Glucose", 0, 300, 120)

    with col2:
        bp = st.number_input("Blood Pressure", 0, 200, 70)
        skin = st.number_input("Skin Thickness", 0, 100, 20)

    with col3:
        insulin = st.number_input("Insulin", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)

    with col4:
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 5.0, 0.5)
        age2 = st.number_input("Confirm Age", 1, 120, int(age))

    st.markdown("</div>", unsafe_allow_html=True)

    submit = st.form_submit_button("🔍 Predict")

# ------------------------------------------------------
# Prediction
# ------------------------------------------------------

if submit:

    features = [[
        pregnancies, glucose, bp, skin,
        insulin, bmi, dpf, age2
    ]]

    df = pd.DataFrame(features, columns=[
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ])

    try:
        scaled = scaler.transform(df)

        pred = model.predict(scaled)[0]

        proba = model.predict_proba(scaled)[0][1] * 100

        st.write("---")

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🧾 Patient Summary")
        st.write(f"**Name:** {name}")
        st.write(f"**Age:** {age}")
        st.write(f"**Phone:** {phone}")
        st.write(f"**Address:** {address}")
        st.markdown("</div>", unsafe_allow_html=True)

        if pred == 1:

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.error("🔴 HIGH RISK of Diabetes Detected!")
            st.write(f"**Model probability:** {proba:.2f}%")
            st.write("- Consult a qualified doctor")
            st.write("- Maintain a healthy lifestyle")
            st.markdown("</div>", unsafe_allow_html=True)

        else:

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.success("🟢 LOW RISK — No Diabetes Detected")
            st.write(f"**Model probability:** {proba:.2f}%")
            st.write("- Continue a healthy lifestyle")
            st.markdown("</div>", unsafe_allow_html=True)

        st.caption(
            "⚕ Model: Logistic Regression | Educational purpose only. "
            "This prediction is not a medical diagnosis."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")

import streamlit as st
import pandas as pd
import joblib
import os

# ==============================
# Custom CSS
# ==============================

st.markdown("""
<style>

.main {
    background-color: #f5f9fc;
}

.title {
    text-align: center;
    color: #0b5ed7;
    font-size: 40px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #555;
    font-size: 18px;
    margin-bottom: 30px;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    background-color: #ffffff;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)


# ==============================
# Project Path
# ==============================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# ==============================
# Model Path
# ==============================

model_path = os.path.join(
    BASE_DIR,
    "models",
    "best_model.pkl"
)

scaler_path = os.path.join(
    BASE_DIR,
    "models",
    "scaler.pkl"
)


# ==============================
# Load Model
# ==============================

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)


# ==============================
# Page Configuration
# ==============================

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)


# ==============================
# Title
# ==============================

st.markdown(
    '<div class="title">🩺 Healthcare AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Diabetes Risk Prediction System</div>',
    unsafe_allow_html=True
)

st.info(
    "Enter patient health information below to "
    "estimate diabetes risk using Machine Learning."
)

# ==============================
# Input Fields
# ==============================

col1, col2 = st.columns(2)


with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0.0,
        max_value=300.0,
        value=120.0
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0.0,
        max_value=100.0,
        value=20.0
    )


with col2:

    insulin = st.number_input(
        "Insulin",
        min_value=0.0,
        max_value=900.0,
        value=80.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )


# ==============================
# Prediction Button
# ==============================

if st.button("🔍 Predict Diabetes Risk"):

    input_data = pd.DataFrame({

        "Pregnancies": [pregnancies],

        "Glucose": [glucose],

        "BloodPressure": [blood_pressure],

        "SkinThickness": [skin_thickness],

        "Insulin": [insulin],

        "BMI": [bmi],

        "DiabetesPedigreeFunction": [
            diabetes_pedigree
        ],

        "Age": [age]
    })


    # ==============================
    # Scale Input
    # ==============================

    input_scaled = scaler.transform(input_data)


    # ==============================
    # Prediction
    # ==============================

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(
        input_scaled
    )[0][1]


    # ==============================
    # Result
    # ==============================

    st.markdown(
        '<div class="result-box">',
        unsafe_allow_html=True
    )

    if prediction == 1:

        st.error(
            "⚠️ Higher Diabetes Risk Detected"
        )

    else:

        st.success(
            "✅ Lower Diabetes Risk Detected"
        )

    st.metric(
        "Diabetes Probability",
        f"{probability * 100:.2f}%"
    )

    st.progress(float(probability))

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )
    
# ==============================
# Model Performance
# ==============================

st.divider()

st.header("📊 Model Performance")

comparison_path = os.path.join(
    BASE_DIR,
    "models",
    "model_comparison.csv"
)

if os.path.exists(comparison_path):

    results = pd.read_csv(comparison_path)

    st.dataframe(
        results,
        use_container_width=True
    )

    st.subheader("📈 Model Comparison")

    chart_data = results.set_index("Model")[
        ["Accuracy", "Precision", "Recall", "F1 Score"]
    ]

    st.bar_chart(chart_data)

else:

    st.warning(
        "Model comparison data not found."
    )
    
# ==============================
# About Project Section
# ==============================
st.divider()

st.header("ℹ️ About This Project")

st.write("""
This Healthcare Disease Prediction system uses
Machine Learning to estimate diabetes risk based
on patient health parameters.

Models evaluated:

• Logistic Regression
• Random Forest
• XGBoost

The best-performing model is used for the final
prediction system.
""")

st.warning(
    "⚠️ This application is developed for educational "
    "and demonstration purposes only. It is not a "
    "substitute for professional medical diagnosis."
)

# ==============================
# Evaluation Charts
# ==============================

st.divider()

st.header("📈 Model Evaluation")

col1, col2 = st.columns(2)

with col1:

    confusion_path = os.path.join(
        BASE_DIR,
        "models",
        "confusion_matrix.png"
    )

    if os.path.exists(confusion_path):

        st.subheader("Confusion Matrix")

        st.image(
            confusion_path,
            use_container_width=True
        )


with col2:

    roc_path = os.path.join(
        BASE_DIR,
        "models",
        "roc_curve.png"
    )

    if os.path.exists(roc_path):

        st.subheader("ROC Curve")

        st.image(
            roc_path,
            use_container_width=True
        )
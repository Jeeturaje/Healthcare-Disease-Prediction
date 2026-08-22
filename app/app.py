import streamlit as st
import pandas as pd
import joblib
import os


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)


# =========================================================
# CUSTOM CSS
# =========================================================

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


# =========================================================
# PROJECT PATH
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# =========================================================
# MODEL PATH
# =========================================================

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


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)


# =========================================================
# TITLE
# =========================================================

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


# =========================================================
# INPUT FIELDS
# =========================================================

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


# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button(
    "🔍 Predict Diabetes Risk",
    use_container_width=True
):

    # =====================================================
    # CREATE INPUT DATA
    # =====================================================

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


    # =====================================================
    # SCALE INPUT
    # =====================================================

    input_scaled = scaler.transform(
        input_data
    )


    # =====================================================
    # PREDICTION
    # =====================================================

    prediction = model.predict(
        input_scaled
    )[0]

    probability = model.predict_proba(
        input_scaled
    )[0][1]


    # =====================================================
    # RESULT
    # =====================================================

    st.divider()

    st.subheader("🔮 Prediction Result")

    st.markdown(
        '<div class="result-box">',
        unsafe_allow_html=True
    )


    if prediction == 1:

        st.error(
            "⚠️ Higher Diabetes Risk Detected"
        )

        risk_result = "Higher Diabetes Risk Detected"

    else:

        st.success(
            "✅ Lower Diabetes Risk Detected"
        )

        risk_result = "Lower Diabetes Risk Detected"


    # =====================================================
    # PROBABILITY
    # =====================================================

    st.metric(
        "Diabetes Probability",
        f"{probability * 100:.2f}%"
    )

    st.progress(
        float(probability)
    )


    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # DOWNLOAD REPORT
    # =====================================================

    st.divider()

    st.subheader(
        "📥 Download Prediction Report"
    )


    report = f"""
==================================================
       HEALTHCARE DISEASE PREDICTION REPORT
==================================================

Patient Information
--------------------------------------------------

Pregnancies                 : {pregnancies}
Glucose                     : {glucose}
Blood Pressure              : {blood_pressure}
Skin Thickness              : {skin_thickness}
Insulin                     : {insulin}
BMI                         : {bmi}
Diabetes Pedigree Function  : {diabetes_pedigree}
Age                         : {age}


Prediction Result
--------------------------------------------------

Result                      : {risk_result}
Diabetes Probability        : {probability * 100:.2f}%


Machine Learning Model
--------------------------------------------------

Model used: Best Performing Model


Medical Disclaimer
--------------------------------------------------

This application is developed for educational
and demonstration purposes only.

The prediction generated by this application
is NOT a medical diagnosis.

Please consult a qualified healthcare professional
for medical advice.


==================================================
              END OF REPORT
==================================================
"""


    st.download_button(
        label="📥 Download Prediction Report",
        data=report,
        file_name="diabetes_prediction_report.txt",
        mime="text/plain",
        use_container_width=True
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.divider()

st.header(
    "📊 Model Performance"
)


comparison_path = os.path.join(
    BASE_DIR,
    "models",
    "model_comparison.csv"
)


if os.path.exists(comparison_path):

    results = pd.read_csv(
        comparison_path
    )


    st.dataframe(
        results,
        use_container_width=True
    )


    st.subheader(
        "📈 Model Comparison"
    )


    chart_data = results.set_index(
        "Model"
    )[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]
    ]


    st.bar_chart(
        chart_data
    )

else:

    st.warning(
        "Model comparison data not found."
    )


# =========================================================
# ABOUT PROJECT
# =========================================================

st.divider()

st.header(
    "ℹ️ About This Project"
)


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

The application is developed using Python,
Scikit-learn, XGBoost, Pandas and Streamlit.
""")


# =========================================================
# MEDICAL DISCLAIMER
# =========================================================

st.warning(
    "⚠️ Medical Disclaimer: This application is "
    "developed for educational and demonstration "
    "purposes only. It is not a substitute for "
    "professional medical diagnosis or treatment."
)


# =========================================================
# MODEL EVALUATION
# =========================================================

st.divider()

st.header(
    "📈 Model Evaluation"
)


col1, col2 = st.columns(2)


# =========================================================
# CONFUSION MATRIX
# =========================================================

with col1:

    confusion_path = os.path.join(
        BASE_DIR,
        "models",
        "confusion_matrix.png"
    )


    if os.path.exists(
        confusion_path
    ):

        st.subheader(
            "Confusion Matrix"
        )


        st.image(
            confusion_path,
            use_container_width=True
        )

    else:

        st.warning(
            "Confusion matrix image not found."
        )


# =========================================================
# ROC CURVE
# =========================================================

with col2:

    roc_path = os.path.join(
        BASE_DIR,
        "models",
        "roc_curve.png"
    )


    if os.path.exists(
        roc_path
    ):

        st.subheader(
            "ROC Curve"
        )


        st.image(
            roc_path,
            use_container_width=True
        )

    else:

        st.warning(
            "ROC curve image not found."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🩺 Healthcare Disease Prediction | "
    "Machine Learning Project | "
    "Developed using Python & Streamlit"
)
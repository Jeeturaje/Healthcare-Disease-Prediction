import streamlit as st
import pandas as pd
import joblib
import os
import mysql.connector

from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)


# =========================================================
# PROJECT PATH
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# =========================================================
# MYSQL DATABASE CONNECTION
# =========================================================

def get_db_connection():

    connection = mysql.connector.connect(
        host=st.secrets["mysql"]["host"],
        user=st.secrets["mysql"]["user"],
        password=st.secrets["mysql"]["password"],
        database=st.secrets["mysql"]["database"],
        port=st.secrets["mysql"]["port"]
    )

    return connection


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
# LOAD MODEL
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

try:

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

except Exception as e:

    st.error(
        f"Model loading error: {e}"
    )

    st.stop()


# =========================================================
# PDF REPORT FUNCTION
# =========================================================

def create_pdf_report(
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree,
    age,
    prediction,
    probability
):

    buffer = BytesIO()

    pdf = canvas.Canvas(
        buffer,
        pagesize=A4
    )

    width, height = A4


    # =====================================================
    # TITLE
    # =====================================================

    pdf.setFillColor(
        colors.HexColor("#0b5ed7")
    )

    pdf.setFont(
        "Helvetica-Bold",
        20
    )

    pdf.drawCentredString(
        width / 2,
        height - 60,
        "Healthcare Disease Prediction"
    )

    pdf.setFillColor(
        colors.black
    )

    pdf.setFont(
        "Helvetica",
        12
    )

    pdf.drawCentredString(
        width / 2,
        height - 85,
        "Diabetes Risk Prediction Report"
    )


    # =====================================================
    # PATIENT INFORMATION
    # =====================================================

    y = height - 130

    pdf.setFont(
        "Helvetica-Bold",
        14
    )

    pdf.drawString(
        50,
        y,
        "Patient Information"
    )

    y -= 30

    pdf.setFont(
        "Helvetica",
        11
    )

    information = [

        f"Pregnancies: {pregnancies}",

        f"Glucose: {glucose}",

        f"Blood Pressure: {blood_pressure}",

        f"Skin Thickness: {skin_thickness}",

        f"Insulin: {insulin}",

        f"BMI: {bmi}",

        f"Diabetes Pedigree Function: {diabetes_pedigree}",

        f"Age: {age}"
    ]


    for item in information:

        pdf.drawString(
            70,
            y,
            item
        )

        y -= 22


    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    y -= 15

    pdf.setFont(
        "Helvetica-Bold",
        14
    )

    pdf.drawString(
        50,
        y,
        "Prediction Result"
    )

    y -= 30

    pdf.setFont(
        "Helvetica",
        11
    )


    if prediction == 1:

        result = "Higher Diabetes Risk Detected"

    else:

        result = "Lower Diabetes Risk Detected"


    pdf.drawString(
        70,
        y,
        f"Result: {result}"
    )

    y -= 25

    pdf.drawString(
        70,
        y,
        f"Diabetes Probability: {probability * 100:.2f}%"
    )


    # =====================================================
    # MODEL INFORMATION
    # =====================================================

    y -= 50

    pdf.setFont(
        "Helvetica-Bold",
        14
    )

    pdf.drawString(
        50,
        y,
        "Machine Learning Model"
    )

    y -= 25

    pdf.setFont(
        "Helvetica",
        11
    )

    pdf.drawString(
        70,
        y,
        "Best Performing Model"
    )


    # =====================================================
    # DISCLAIMER
    # =====================================================

    y -= 55

    pdf.setFont(
        "Helvetica-Bold",
        12
    )

    pdf.drawString(
        50,
        y,
        "Medical Disclaimer"
    )

    y -= 25

    pdf.setFont(
        "Helvetica",
        9
    )

    disclaimer = [

        "This application is developed for educational purposes only.",

        "It is not a substitute for professional medical diagnosis or treatment.",

        "Please consult a qualified healthcare professional for medical advice."
    ]


    for line in disclaimer:

        pdf.drawString(
            50,
            y,
            line
        )

        y -= 15


    # =====================================================
    # FOOTER
    # =====================================================

    pdf.setFont(
        "Helvetica",
        8
    )

    pdf.drawCentredString(
        width / 2,
        30,
        "Healthcare Disease Prediction | Machine Learning Project"
    )


    pdf.save()

    buffer.seek(0)

    return buffer.getvalue()


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
    "Enter patient health information below "
    "to estimate diabetes risk using Machine Learning."
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

    try:

        input_scaled = scaler.transform(
            input_data
        )

    except Exception as e:

        st.error(
            f"Input processing error: {e}"
        )

        st.stop()


    # =====================================================
    # PREDICTION
    # =====================================================

    try:

        prediction = model.predict(
            input_scaled
        )[0]

        probability = model.predict_proba(
            input_scaled
        )[0][1]

    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )

        st.stop()


    # =====================================================
    # SAVE PREDICTION TO MYSQL
    # =====================================================

    try:

        connection = get_db_connection()

        cursor = connection.cursor()


        query = """
        INSERT INTO prediction_history
        (
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age,
            prediction,
            probability
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """


        values = (

            pregnancies,

            glucose,

            blood_pressure,

            skin_thickness,

            insulin,

            bmi,

            diabetes_pedigree,

            age,

            "Higher Risk"
            if prediction == 1
            else "Lower Risk",

            float(probability)
        )


        cursor.execute(
            query,
            values
        )

        connection.commit()

        cursor.close()

        connection.close()


        st.success(
            "✅ Prediction saved to MySQL database!"
        )


    except Exception as e:

        st.warning(
            f"⚠️ Database Error: {e}"
        )


    # =====================================================
    # RESULT
    # =====================================================

    st.divider()

    st.subheader(
        "🔮 Prediction Result"
    )


    if prediction == 1:

        st.error(
            "⚠️ Higher Diabetes Risk Detected"
        )

        risk_result = (
            "Higher Diabetes Risk Detected"
        )

    else:

        st.success(
            "✅ Lower Diabetes Risk Detected"
        )

        risk_result = (
            "Lower Diabetes Risk Detected"
        )


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


    # =====================================================
    # PDF REPORT
    # =====================================================

    st.divider()

    st.subheader(
        "📥 Download Prediction Report"
    )


    try:

        pdf_data = create_pdf_report(

            pregnancies,

            glucose,

            blood_pressure,

            skin_thickness,

            insulin,

            bmi,

            diabetes_pedigree,

            age,

            prediction,

            probability
        )


        st.download_button(

            label="📄 Download PDF Report",

            data=pdf_data,

            file_name="diabetes_prediction_report.pdf",

            mime="application/pdf",

            use_container_width=True
        )


    except Exception as e:

        st.error(
            f"PDF generation error: {e}"
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


if os.path.exists(
    comparison_path
):

    try:

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


        required_columns = [

            "Model",

            "Accuracy",

            "Precision",

            "Recall",

            "F1 Score"
        ]


        if all(
            col in results.columns
            for col in required_columns
        ):

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
                "Required model performance "
                "columns are missing."
            )


    except Exception as e:

        st.error(
            f"Could not read model comparison: {e}"
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

Technologies used:

• Python
• Pandas
• NumPy
• Scikit-learn
• XGBoost
• Streamlit
• MySQL
• ReportLab
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
    "Developed using Python, Streamlit & MySQL"
)

# =========================================================
# PREDICTION HISTORY
# =========================================================

st.divider()

st.header("📜 Prediction History")

if st.button("🔄 Load Prediction History"):

    try:

        connection = get_db_connection()

        query = """
        SELECT
            id,
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age,
            prediction,
            probability,
            created_at
        FROM prediction_history
        ORDER BY id DESC
        """

        history = pd.read_sql(
            query,
            connection
        )

        connection.close()

        if len(history) > 0:

            history["probability"] = (
                history["probability"] * 100
            ).round(2)

            history = history.rename(
                columns={
                    "id": "ID",
                    "pregnancies": "Pregnancies",
                    "glucose": "Glucose",
                    "blood_pressure": "Blood Pressure",
                    "skin_thickness": "Skin Thickness",
                    "insulin": "Insulin",
                    "bmi": "BMI",
                    "diabetes_pedigree": "Diabetes Pedigree",
                    "age": "Age",
                    "prediction": "Prediction",
                    "probability": "Probability %",
                    "created_at": "Date"
                }
            )

            st.dataframe(
                history,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No prediction history found."
            )

    except Exception as e:

        st.error(
            f"Database Error: {e}"
        )
        
# =========================================================
# DATABASE DASHBOARD
# =========================================================

st.divider()

st.header("📊 Prediction Dashboard")

try:

    connection = get_db_connection()

    dashboard_query = """
    SELECT
        prediction,
        probability
    FROM prediction_history
    """

    dashboard_data = pd.read_sql(
        dashboard_query,
        connection
    )

    connection.close()

    if not dashboard_data.empty:

        total_predictions = len(dashboard_data)

        higher_risk = len(
            dashboard_data[
                dashboard_data["prediction"] == "Higher Risk"
            ]
        )

        lower_risk = len(
            dashboard_data[
                dashboard_data["prediction"] == "Lower Risk"
            ]
        )

        average_probability = (
            dashboard_data["probability"].mean() * 100
        )

        # -------------------------------------------------
        # DASHBOARD CARDS
        # -------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "📊 Total Predictions",
                total_predictions
            )

        with col2:
            st.metric(
                "🔴 Higher Risk",
                higher_risk
            )

        with col3:
            st.metric(
                "🟢 Lower Risk",
                lower_risk
            )

        with col4:
            st.metric(
                "📈 Avg Probability",
                f"{average_probability:.2f}%"
            )

        # -------------------------------------------------
        # RISK DISTRIBUTION
        # -------------------------------------------------

        st.subheader("📊 Risk Distribution")

        risk_data = pd.DataFrame({
            "Risk": [
                "Higher Risk",
                "Lower Risk"
            ],
            "Count": [
                higher_risk,
                lower_risk
            ]
        })

        st.bar_chart(
            risk_data.set_index("Risk")
        )

    else:

        st.info(
            "No prediction data available yet."
        )

except Exception as e:

    st.error(
        f"Dashboard Database Error: {e}"
    )        

# =========================================================
# DATABASE HISTORY TOOLS
# =========================================================

st.divider()

st.header("🗂️ Database History")

try:

    connection = get_db_connection()

    history_query = """
    SELECT
        id,
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age,
        prediction,
        probability
    FROM prediction_history
    ORDER BY id DESC
    """

    history_data = pd.read_sql(
        history_query,
        connection
    )

    connection.close()

    if not history_data.empty:

        # Convert probability to percentage
        history_data["probability"] = (
            history_data["probability"] * 100
        ).round(2)

        history_data = history_data.rename(
            columns={
                "id": "ID",
                "pregnancies": "Pregnancies",
                "glucose": "Glucose",
                "blood_pressure": "Blood Pressure",
                "skin_thickness": "Skin Thickness",
                "insulin": "Insulin",
                "bmi": "BMI",
                "diabetes_pedigree": "Diabetes Pedigree",
                "age": "Age",
                "prediction": "Prediction",
                "probability": "Probability %"
            }
        )

        # -------------------------------------------------
        # SHOW HISTORY
        # -------------------------------------------------

        st.dataframe(
            history_data,
            use_container_width=True,
            hide_index=True
        )

        # -------------------------------------------------
        # CSV DOWNLOAD
        # -------------------------------------------------

        csv_data = history_data.to_csv(
            index=False
        )

        st.download_button(
            label="📥 Download Prediction History",
            data=csv_data,
            file_name="prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )

    else:

        st.info(
            "No prediction history available."
        )

except Exception as e:

    st.error(
        f"History Error: {e}"
    )    
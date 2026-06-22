import streamlit as st
import joblib
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# Load Model and Scaler
model = joblib.load("logistic_model.joblib")
scaler = joblib.load("scaler.joblib")

# Title
st.title("❤️ Coronary Heart Disease Prediction")
st.write("Predict the risk of Coronary Heart Disease (CHD) in the next 10 years.")

# User Inputs
gender = st.selectbox("Gender", ["Female", "Male"])
male = 1 if gender == "Male" else 0

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=40
)

smoker = st.selectbox("Current Smoker", ["No", "Yes"])
currentSmoker = 1 if smoker == "Yes" else 0

cigsPerDay = st.number_input(
    "Cigarettes Per Day",
    min_value=0.0,
    value=0.0
)

bp_meds = st.selectbox("Taking BP Medicines", ["No", "Yes"])
BPMeds = 1 if bp_meds == "Yes" else 0

stroke = st.selectbox("Previous Stroke", ["No", "Yes"])
prevalentStroke = 1 if stroke == "Yes" else 0

hypertension = st.selectbox("Hypertension", ["No", "Yes"])
prevalentHyp = 1 if hypertension == "Yes" else 0

diabetes_choice = st.selectbox("Diabetes", ["No", "Yes"])
diabetes = 1 if diabetes_choice == "Yes" else 0

totChol = st.number_input(
    "Total Cholesterol",
    min_value=0.0,
    value=200.0
)

sysBP = st.number_input(
    "Systolic Blood Pressure",
    min_value=0.0,
    value=120.0
)

diaBP = st.number_input(
    "Diastolic Blood Pressure",
    min_value=0.0,
    value=80.0
)

BMI = st.number_input(
    "BMI",
    min_value=0.0,
    value=25.0
)

heartRate = st.number_input(
    "Heart Rate",
    min_value=0.0,
    value=75.0
)

glucose = st.number_input(
    "Glucose Level",
    min_value=0.0,
    value=80.0
)

# Prediction Button
if st.button("Predict Risk"):

    input_data = np.array([[
        male,
        age,
        currentSmoker,
        cigsPerDay,
        BPMeds,
        prevalentStroke,
        prevalentHyp,
        diabetes,
        totChol,
        sysBP,
        diaBP,
        BMI,
        heartRate,
        glucose
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Coronary Heart Disease in the Next 10 Years")
    else:
        st.success("✅ Low Risk of Coronary Heart Disease in the Next 10 Years")

# Footer
st.markdown("---")

st.markdown("""
### 👨‍💻 Developed By
**Bhupinder Sandhu**

### 🛠️ Technologies Used
- Python
- Streamlit
- NumPy
- Scikit-Learn
- Logistic Regression
- StandardScaler
- Joblib
- Machine Learning
""")
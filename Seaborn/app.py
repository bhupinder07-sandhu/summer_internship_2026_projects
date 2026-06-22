import streamlit as st
import joblib
import numpy as np

# Load model, scaler and encoder
model = joblib.load("linear_regression_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# Page Config
st.set_page_config(
    page_title="Vehicle Fuel Efficiency Predictor",
    page_icon="🚗",
    layout="centered"
)

# Custom Styling
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.stButton>button {
    width: 100%;
    background-color: #0E1117;
    color: white;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #e8f5e9;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    color: #1b5e20;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("🚗 Vehicle Fuel Efficiency Predictor")
st.write(
    "Predict the fuel efficiency (MPG - Miles Per Gallon) of a vehicle "
    "using Machine Learning."
)

st.divider()

# Input Section
st.subheader("Vehicle Specifications")

col1, col2 = st.columns(2)

with col1:
    cylinders = st.number_input(
        "Cylinders",
        min_value=3,
        max_value=12,
        value=4
    )

    horsepower = st.number_input(
        "Horsepower",
        min_value=40.0,
        max_value=250.0,
        value=100.0
    )

    acceleration = st.number_input(
        "Acceleration",
        min_value=5.0,
        max_value=30.0,
        value=15.0
    )

    origin = st.selectbox(
        "Origin",
        ["usa", "europe", "japan"]
    )

with col2:
    displacement = st.number_input(
        "Displacement",
        min_value=50.0,
        max_value=500.0,
        value=150.0
    )

    weight = st.number_input(
        "Weight",
        min_value=1000.0,
        max_value=6000.0,
        value=2500.0
    )

    model_year = st.slider(
        "Model Year",
        min_value=70,
        max_value=82,
        value=76
    )

st.divider()

# Prediction
if st.button("🔍 Predict Fuel Efficiency"):

    origin_encoded = le.transform([origin])[0]

    features = np.array([[
        cylinders,
        displacement,
        horsepower,
        weight,
        acceleration,
        model_year,
        origin_encoded
    ]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]

    st.markdown(
        f"""
        <div class='result-box'>
            Predicted MPG<br>
            {prediction:.2f}
        </div>
        """,
        unsafe_allow_html=True
    )

    if prediction >= 30:
        st.success("Excellent fuel efficiency 🚀")
    elif prediction >= 20:
        st.info("Average fuel efficiency 👍")
    else:
        st.warning("Low fuel efficiency ⛽")

st.divider()
st.divider()

st.markdown("""
### 🛠️ Tools & Technologies
✅ Python  
✅ Pandas  
✅ NumPy  
✅ Seaborn  
✅ Matplotlib  
✅ Scikit-Learn  
✅ Streamlit  
✅ Joblib
""")

st.markdown(
    """
    <div style='text-align:center; padding:15px;'>
        <h4>🚗 Vehicle Fuel Efficiency Predictor</h4>
        <p>Made by <b>Bhupinder Sandhu</b></p>
    </div>
    """,
    unsafe_allow_html=True
)
st.caption(
    "Built with Streamlit | Machine Learning Regression Project"
)
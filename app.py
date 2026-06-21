import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("car_price_model.pkl")
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)
st.markdown("""
<h1 style='text-align:center;
color:white;
font-size:50px;
font-weight:bold;'>
Car Price Predictor
</h1>

<p style='text-align:center;
color:#d1d5db;
font-size:18px;'>
Get an AI-powered estimate of your car's market value
</p>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(
        135deg,
        #87CEEB,
        #4FC3F7,
        #29B6F6,
        #0288D1
    );
}

/* Glassmorphism Inputs */
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    background: rgba(255,255,255,0.15) !important;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.3) !important;
    border-radius: 15px !important;
    color: white !important;
}

/* Button */

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg,#ff6a00,#ee0979);
    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 15px;
    height: 55px;
    border: none;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}

.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 8px 20px rgba(0,0,0,0.4);
}

</style>
""", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    name = st.text_input(
        " Car Name",
        "Hyundai Santro Xing XO ERLX Euro III"
    )

    company = st.text_input(
        " Company",
        "Hyundai"
    )

with col2:
    year = st.number_input(
        "Year",
        min_value=1990,
        max_value=2030,
        value=2015
    )

    kms_driven = st.number_input(
        " Kilometers Driven",
        min_value=0,
        value=45000
    )

fuel_type = st.selectbox(
    " Fuel Type",
    ["Petrol", "Diesel"]
)

if st.button("Predict Price", key="predict_btn"):

    new_car = pd.DataFrame({
        "name": [name],
        "company": [company],
        "year": [year],
        "kms_driven": [kms_driven],
        "fuel_type": [fuel_type]
    })

    try:
        prediction = model.predict(new_car)

        st.markdown(f"""
        <div style="
            background: rgba(255,255,255,0.15);
            backdrop-filter: blur(15px);
            border: 1px solid rgba(255,255,255,0.25);
            border-radius: 20px;
            padding: 30px;
            text-align: center;
            margin-top: 25px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        ">

        <h3 style="color:white;">
         Estimated Market Value
        </h3>

        <h1 style="
            color:white;
            font-size:50px;
            font-weight:bold;">
            ₹ {prediction[0]:,.0f}
        </h1>

        <p style="color:white;">
        
        </p>

        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(e)

st.markdown("""
<hr style="margin-top:40px;">

<p style="
text-align:center;
color:#d1d5db;
font-size:16px;">
Built with using Machine Learning & Streamlit
</p>
""", unsafe_allow_html=True)
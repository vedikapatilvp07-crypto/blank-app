import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="⚙️",
    layout="centered"
)

# Load model
model = joblib.load("predictive_maintenance_model.pkl")

# Title
st.title("⚙️ AI Predictive Maintenance")
st.write("Predict machine failure using machine operating parameters.")

st.divider()

# Input parameters
st.subheader("Enter Machine Parameters")

air_temp = st.number_input(
    "Air Temperature [K]",
    min_value=290.0,
    max_value=320.0,
    value=298.0
)

process_temp = st.number_input(
    "Process Temperature [K]",
    min_value=300.0,
    max_value=330.0,
    value=308.0
)

rot_speed = st.number_input(
    "Rotational Speed [rpm]",
    min_value=1000,
    max_value=3000,
    value=1500
)

torque = st.number_input(
    "Torque [Nm]",
    min_value=0.0,
    max_value=80.0,
    value=40.0
)

tool_wear = st.number_input(
    "Tool Wear [min]",
    min_value=0,
    max_value=300,
    value=100
)

machine_type = st.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)

# Convert machine type into the same features used during training
type_L = 1 if machine_type == "L" else 0
type_M = 1 if machine_type == "M" else 0

# Create input dataframe
input_data = pd.DataFrame({
    "Air temperature [K]": [air_temp],
    "Process temperature [K]": [process_temp],
    "Rotational speed [rpm]": [rot_speed],
    "Torque [Nm]": [torque],
    "Tool wear [min]": [tool_wear],
    "Type_L": [type_L],
    "Type_M": [type_M]
})

st.divider()

# Prediction
if st.button("🔍 Predict Machine Failure", use_container_width=True):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    st.metric(
        "Failure Probability",
        f"{probability * 100:.2f}%"
    )

    if prediction == 1:
        st.error("⚠️ Machine Failure Predicted")
        st.warning("Maintenance is recommended.")
    else:
        st.success("✅ No Machine Failure Predicted")
        st.info("Machine is operating normally.")

    st.subheader("Input Parameters")
    st.dataframe(input_data)

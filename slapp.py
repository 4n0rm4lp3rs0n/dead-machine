import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the saved model and scaler
model = joblib.load('rf_combined_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('labels.pkl')

# Define the expected feature names (MUST match training order)
feature_names = [
    'Air temperature [K]', 'Process temperature [K]',
    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
    'Type'
]

st.title("Machine Failure Prediction")

# Create input fields for each feature
def get_user_input():
    air_temp = st.number_input("Air Temperature (K)", value=300.0)
    process_temp = st.number_input("Process Temperature (K)", value=310.0)
    rpm = st.number_input("Rotational Speed (rpm)", value=1500.0)
    torque = st.number_input("Torque (Nm)", value=40.0)
    tool_wear = st.number_input("Tool Wear (min)", value=10.0)
    
    machine_type = st.selectbox("Machine Type", le.classes_)
    encoded_type = le.transform([machine_type])[0]

    data = np.array([[air_temp, process_temp, rpm, torque, tool_wear, encoded_type]])
    return data

input_data = get_user_input()

# Scale the input
scaled_input = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)
    st.write(f"### Predicted: `{prediction}`")
    
    # Show class probabilities
    prob_df = pd.DataFrame(probabilities, columns=model.classes_)
    st.write("#### Class Probabilities:")
    st.dataframe(prob_df.T)


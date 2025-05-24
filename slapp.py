import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('rf_combined_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define expected columns
FEATURE_COLUMNS = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
    'Type_H',
    'Type_L',
]

st.title("📊 Machine Failure Prediction (Batch Upload)")

# File uploader
uploaded_file = st.file_uploader("Upload your machine data (CSV)", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Preview of Uploaded Data")
    st.dataframe(df.head())

    # Check if required columns exist
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
    else:
        # Scale the input
        X_scaled = scaler.transform(df[FEATURE_COLUMNS])
        
        # Predict
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)

        # Append predictions to dataframe
        df['Predicted Class'] = predictions

        # Add prediction probabilities (optional)
        for i, class_label in enumerate(model.classes_):
            df[f'Prob_{class_label}'] = probabilities[:, i]

        st.subheader("✅ Prediction Results")
        st.dataframe(df)

        # Optionally download result
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions as CSV", csv, "predictions.csv", "text/csv")

import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = joblib.load("rf_combined_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🛠️ Machine Failure Predictor")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file with machine data", type=["csv"])

if uploaded_file is not None:
    # Read uploaded data
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Uploaded Data Preview")
    st.dataframe(df)

    # One-hot encode the Type column
    if 'Type' in df.columns:
        df = pd.get_dummies(df, columns=['Type'], drop_first=True)
    else:
        st.error("Missing 'Type' column in the uploaded file.")
        st.stop()

    # Ensure Type_H and Type_L columns exist
    for col in ['Type_H', 'Type_L']:
        if col not in df.columns:
            df[col] = 0  # Add missing column with 0s

    # Ensure consistent column order
    feature_columns = ['Air temperature [K]', 'Process temperature [K]',
                       'Rotational speed [rpm]', 'Torque [Nm]',
                       'Tool wear [min]', 'Type_H', 'Type_L']
    
    # Check if all required features are in the uploaded data
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()

    # Select features and scale them
    X = df[feature_columns]
    X_scaled = scaler.transform(X)

    # Predict
    predictions = model.predict(X_scaled)

    # Show predictions
    df['Prediction'] = predictions
    st.subheader("📊 Predictions")
    st.dataframe(df)

    # Optionally allow download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download predictions as CSV", data=csv, file_name='predictions.csv', mime='text/csv')
else:
    st.info("Please upload a CSV file to start.")

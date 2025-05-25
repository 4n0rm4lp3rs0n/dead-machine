import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and encoder
model = joblib.load('rf_combined_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('labels.pkl')

st.title("🛠️ Machine Failure Prediction")

uploaded_file = st.file_uploader("📂 Upload a CSV file with machine data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Uploaded Data Preview")
    st.dataframe(df)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Required columns in the exact order used during training
    required_cols = [
        'Type',
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]

    # Check for missing columns
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    # Validate 'Type' values
    if not set(df['Type']).issubset(set(le.classes_)):
        st.error(f"The 'Type' column contains invalid values. Expected: {list(le.classes_)}")
        st.stop()

    # Encode 'Type'
    df['Type'] = le.transform(df['Type'])

    # Select features in correct order
    X = df[required_cols]

    # Handle missing or invalid values
    if X.isnull().any().any():
        st.warning("Filling missing values with column means.")
        X = X.fillna(X.mean(numeric_only=True))

    # Scale and predict
    try:
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        df['Prediction'] = predictions
    except ValueError as e:
        st.error(f"Error during prediction: {e}")
        st.stop()

    st.subheader("📊 Predictions")
    st.dataframe(df)

    # Download predictions
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download predictions as CSV",
        data=csv,
        file_name='predictions.csv',
        mime='text/csv'
    )
else:
    st.info("Please upload a CSV file to start.")
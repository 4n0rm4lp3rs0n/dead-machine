import streamlit as st
import pandas as pd
import joblib

# Load the saved model, scaler, and label encoder
model = joblib.load('rf_combined_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('labels.pkl')

st.title("🛠️ Machine Failure Prediction")

uploaded_file = st.file_uploader("📂 Upload a CSV file with machine data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Uploaded Data Preview")
    st.dataframe(df)

    # Check required columns
    required_cols = ['Type' ,'Air temperature [K]', 'Process temperature [K]',
                     'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()

    # Encode 'Type' column
    try:
        df['Type'] = le.transform(df['Type'])
    except ValueError as e:
        st.error(f"Encoding error: {e}")
        st.stop()

    # Select features and scale
    X = df[required_cols]
    X_scaled = scaler.transform(X)

    # Predict
    predictions = model.predict(X_scaled)
    df['Prediction'] = predictions

    st.subheader("📊 Predictions")
    st.dataframe(df)

    # Download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download predictions as CSV", data=csv,
                       file_name='predictions.csv', mime='text/csv')

else:
    st.info("Please upload a CSV file to start.")

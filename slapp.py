REQUIRED_COLUMNS = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
    'Type'
]

DUMMY_COLUMNS = ['Type_H', 'Type_L']  # The columns used in training

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Preview of Uploaded Data")
    st.dataframe(df.head())

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
    else:
        # One-hot encode 'Type' like during training (drop_first=True → no Type_M)
        df_encoded = pd.get_dummies(df, columns=['Type'], drop_first=True)

        # Ensure all dummy columns exist
        for col in DUMMY_COLUMNS:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        # Final feature list for prediction
        final_features = [
            'Air temperature [K]',
            'Process temperature [K]',
            'Rotational speed [rpm]',
            'Torque [Nm]',
            'Tool wear [min]',
            'Type_H',
            'Type_L'
        ]

        # Reorder columns to match training
        df_encoded = df_encoded[final_features]

        # Scale and predict
        X_scaled = scaler.transform(df_encoded)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)

        # Append predictions
        df['Predicted Class'] = predictions
        for i, class_label in enumerate(model.classes_):
            df[f'Prob_{class_label}'] = probabilities[:, i]

        st.subheader("✅ Prediction Results")
        st.dataframe(df)

        # Allow CSV download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")

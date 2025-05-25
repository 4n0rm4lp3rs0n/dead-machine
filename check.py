# import pandas as pd
# import joblib
# from tabulate import tabulate

# # Tải mô hình và scaler
# rf_model = joblib.load('rf_combined_model.pkl')
# scaler = joblib.load('scaler.pkl')
# le = joblib.load('labels.pkl')

# # Hàm dự đoán
# def predict_maintenance(air_temp, process_temp, rpm, torque, tool_wear, type_product):
#     t_e = le.transform([type_product])[0]
    
#     # Tạo DataFrame cho dữ liệu mới
#     input_data = pd.DataFrame({
#         'Air temperature [K]': [air_temp],
#         'Process temperature [K]': [process_temp],
#         'Rotational speed [rpm]': [rpm],
#         'Torque [Nm]': [torque],
#         'Tool wear [min]': [tool_wear],
#         'Type': [t_e]
#     })
    
#     # Chuẩn hóa dữ liệu mới
#     input_scaled = scaler.transform(input_data)
    
#     # Dự đoán
#     prediction = rf_model.predict(input_scaled)
#     probabilities = rf_model.predict_proba(input_scaled)
    
#     # Tạo kết quả
#     machine_failure = 'No Failure' if prediction[0] == 'No Failure' else 'Failure'
#     prob_dict = dict(zip(rf_model.classes_, probabilities[0]))
    
#     # Tạo bảng kết quả
#     result_table = [
#         ['Prediction', prediction[0]],
#         ['Machine Failure', machine_failure]
#     ]
#     prob_table = [[key, f"{value:.4f}"] for key, value in prob_dict.items()]
    
#     print("\n=== Prediction Result ===")
#     print(tabulate(result_table, headers=['Metric', 'Value'], tablefmt='fancy_grid'))
#     print("\n=== Probabilities ===")
#     print(tabulate(prob_table, headers=['Class', 'Probability'], tablefmt='fancy_grid', floatfmt='.4f'))
    
#     return {
#         'Prediction': prediction[0],
#         'Machine Failure': machine_failure,
#         'Probabilities': prob_dict
#     }

# # Ví dụ dự đoán trên nhiều mẫu
# test_samples = [
#     {
#         'air_temp': 298.1,
#         'process_temp': 308.6,
#         'rpm': 1551,
#         'torque': 42.8,
#         'tool_wear': 0,
#         'type_product': 'M'
#     },
#     {
#         'air_temp': 300.0,
#         'process_temp': 310.0,
#         'rpm': 1400,
#         'torque': 50.0,
#         'tool_wear': 200,
#         'type_product': 'L'
#     }
# ]

# print("\n=== Testing Multiple Samples ===")
# for i, sample in enumerate(test_samples, 1):
#     print(f"\nSample {i}:")
#     result = predict_maintenance(
#         air_temp=sample['air_temp'],
#         process_temp=sample['process_temp'],
#         rpm=sample['rpm'],
#         torque=sample['torque'],
#         tool_wear=sample['tool_wear'],
#         type_product=sample['type_product']
#     )

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
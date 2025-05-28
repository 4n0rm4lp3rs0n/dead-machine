# Link: https://check-dead-machine.streamlit.app/
import streamlit as st
import pandas as pd
import joblib
import heapq
import numpy as np
import ht

# Load model, scaler, and encoder
model = joblib.load('rf_combined_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('labels.pkl')

st.title("🛠️ Machine Failure Prediction")

# File uploader
uploaded_file = st.file_uploader("📂 Upload a CSV file with machine data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Uploaded Data Preview")
    st.dataframe(df)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Required columns for prediction
    required_cols = [
        'Type',
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]

    # Check for missing required columns
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"Missing required columns for prediction: {missing}")
        st.stop()

    # Validate 'Type' values
    if not set(df['Type']).issubset(set(le.classes_)):
        st.error(f"The 'Type' column contains invalid values. Expected: {list(le.classes_)}")
        st.stop()

    # Encode 'Type'
    df['Type'] = le.transform(df['Type'])

    # Select features for prediction
    X = df[required_cols]

    # Handle missing or invalid values
    if X.isnull().any().any():
        st.warning("Filling missing values with column means.")
        X = X.fillna(X.mean(numeric_only=True))

    # Scale and predict
    try:
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        df['Prediction'] = predictions
    except ValueError as e:
        st.error(f"Error during prediction: {e}")
        st.stop()

    try:
        df['Type'] = le.inverse_transform(df['Type'])
    except Exception:
        pass

    # Get failure types
    failure_types = []
    if 'Failure Type' in df.columns:
        failure_types = df['Failure Type'].unique().tolist()
        st.info("Using failure types from 'Failure Type' column in the uploaded CSV.")
    else:
        failure_types = np.unique(predictions).tolist()
        st.info("No 'Failure Type' column found. Using predicted failure types.")

    # Validate failure types
    if not failure_types:
        st.error("No valid failure types found in the data.")
        st.stop()

    # Section for user to input priority order
    st.subheader("⚙️ Set Priority Order for Failure Types")
    st.write("Select failure types in order of priority (top = highest priority).")
    priority_order = st.multiselect(
        "Order failure types by priority",
        options=failure_types,
        default=failure_types,  # Default to all types in their natural order
        key="priority_order"
    )

    # Validate priority order
    if not priority_order:
        st.warning("No priority order selected. Displaying unsorted predictions.")
        
        # Add probability columns
        prob_cols = [f'Prob_{cls}' for cls in model.classes_]
        prob_df = pd.DataFrame(probabilities, columns=prob_cols)
        df = pd.concat([df, prob_df], axis=1)

        st.subheader("📊 Predictions (Unsorted)")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download predictions as CSV",
            data=csv,
            file_name='predictions_unsorted.csv',
            mime='text/csv'
        )
        st.stop()

    invalid_types = [t for t in priority_order if t not in failure_types]
    if invalid_types:
        st.error(f"Invalid failure types in priority order: {invalid_types}")
        st.stop()

    # Assign priority scores (lower index = higher priority)
    priority_scores = {failure: idx for idx, failure in enumerate(priority_order)}
    # Assign high priority (large index) to unlisted failures
    max_priority = len(priority_order)
    for failure in failure_types:
        if failure not in priority_scores:
            priority_scores[failure] = max_priority
            max_priority += 1

    # Create priority queue (min-heap) based on user-defined priority
    heap = []
    failure_classes = [cls for cls in model.classes_ if cls != 'No Failure']
    failure_prob_indices = [i for i, cls in enumerate(model.classes_) if cls != 'No Failure']
    for idx, (pred, probs) in enumerate(zip(predictions, probabilities)):
        priority = priority_scores.get(pred, max_priority)  # Get priority score
        # Use max failure probability as tie-breaker
        max_failure_prob = np.max(probs[failure_prob_indices]) if failure_prob_indices else 0
        # Use priority directly (lower = higher priority) with negative prob for tie-breaker
        heapq.heappush(heap, (priority, -max_failure_prob, idx, pred, probs))

    # Create sorted DataFrame based on priority
    sorted_indices = []
    while heap:
        _, _, idx, _, _ = heapq.heappop(heap)
        sorted_indices.append(idx)

    if not sorted_indices:
        st.error("❌ Heap sorting failed — no records prioritized.")
        st.stop()

    # 🔧 Build a hashtable from sorted_df using UDI or Product ID as key
    sorted_df = df.iloc[sorted_indices].reset_index(drop=True)
    hashtable = ht.HashTable()
    key_column = None

    if 'UDI' in sorted_df.columns:
        key_column = 'UDI'
    elif 'Product ID' in sorted_df.columns:
        key_column = 'Product ID'

    seen_keys = set()

    if key_column:
       for _, row in sorted_df.iterrows():
            for col in ['UDI', 'Product ID']:
                if col in row:
                    key = str(row[col]).strip().upper()
                    if key not in seen_keys:
                        hashtable.insert(key, row.to_dict())
                        seen_keys.add(key)
    else:
        st.warning("No 'UDI' or 'Product ID' column found to build hashtable.")

    # Add probability columns for display
    prob_cols = [f'Prob_{cls}' for cls in model.classes_]
    prob_df = pd.DataFrame(probabilities, columns=prob_cols)
    sorted_df = pd.concat([sorted_df, prob_df.iloc[sorted_indices].reset_index(drop=True)], axis=1)

    # Set index on UDI or Product ID for faster search
    search_cols = [col for col in ['UDI', 'Product ID'] if col in sorted_df.columns]
    if search_cols:
        # Create a copy to avoid modifying the original DataFrame
        search_df = sorted_df.set_index(search_cols, drop=False)
    else:
        search_df = sorted_df  # Fallback to original if no search columns

    # Display original sorted predictions
    st.subheader("📊 Predictions (Sorted by User-Defined Priority)")
    st.write(f"Samples are sorted by your priority order: {priority_order}, with max failure probability as tie-breaker.")
    st.dataframe(sorted_df)

    # Download original sorted predictions
    csv = sorted_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download sorted predictions as CSV",
        data=csv,
        file_name='sorted_predictions.csv',
        mime='text/csv'
    )

    # Search bar for UDI or Product ID
    st.subheader("🔎 Search by UDI or Product ID")
    search_term = st.text_input("Enter UDI or Product ID to filter results:", "")

    # Filter DataFrame based on search term
    if search_term:
        if key_column is None:
            st.error("No 'UDI' or 'Product ID' column found in the data.")
        else:
            search_key = str(search_term).strip().upper()
            result = hashtable.get(search_key)
            if result:
                st.subheader(f"📊 Search Result for '{search_key}' (From Hashtable)")
                st.dataframe(pd.DataFrame([result]))

                csv = pd.DataFrame([result]).to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"⬇️ Download search result as CSV",
                    data=csv,
                    file_name=f'search_result_{search_key}.csv',
                    mime='text/csv'
                )
            else:
                st.warning(f"No result found for '{search_term}' in column {key_column}.")
else:
    st.info("Please upload a CSV file to start.")

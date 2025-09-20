
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load artifacts
try:
    model = joblib.load('fraud_model.joblib')
    scaler = joblib.load('scaler.joblib')
    model_columns = joblib.load('model_columns.joblib')
except FileNotFoundError:
    st.error("Model artifacts not found. Please ensure 'fraud_model.joblib', 'scaler.joblib', and 'model_columns.joblib' are in the same directory.")
    st.stop()

# Streamlit interface
st.title("Credit Card Fraud Detection App")
st.write(
    "This app uses a trained XGBoost model to predict the likelihood of a credit card transaction being fraudulent. "
    "Please input the transaction details below, focusing on the most important features."
)

st.header("Input Transaction Features")
col1, col2, col3 = st.columns(3)

with col1:
    v14 = st.slider('V14', -20.0, 10.0, 0.0, step=0.1)
    v12 = st.slider('V12', -20.0, 10.0, 0.0, step=0.1)
    v10 = st.slider('V10', -25.0, 25.0, 0.0, step=0.1)

with col2:
    v17 = st.slider('V17', -25.0, 10.0, 0.0, step=0.1)
    v11 = st.slider('V11', -5.0, 15.0, 0.0, step=0.1)
    v4 = st.slider('V4', -5.0, 17.0, 0.0, step=0.1)

with col3:
    hour = st.slider('Hour of Day (0-23)', 0, 23, 12)
    amount = st.number_input('Transaction Amount (€)', min_value=0.0, value=50.0, step=10.0)

if st.button('Predict Fraud Likelihood'):
    # Preprocessing and prediction
    
    # Create input DataFrame with all required columns
    input_data = pd.DataFrame(columns=model_columns)
    input_data.loc[0] = 0 # Initialize all features to zero or neutral
    
    input_data['V14'] = v14
    input_data['V12'] = v12
    input_data['V10'] = v10
    input_data['V17'] = v17
    input_data['V11'] = v11
    input_data['V4'] = v4
    input_data['Hour'] = hour
    input_data['Amount'] = amount
    
    # Scale amount feature
    input_data['Amount'] = scaler.transform(input_data[['Amount']])
    
    # Predict
    prediction_proba = model.predict_proba(input_data[model_columns])[0][1]
    
    # Show results
    st.header("Prediction Result")
    business_threshold = 0.12
    
    if prediction_proba >= business_threshold:
        st.error(f"ALERT: High Likelihood of Fraud!")
    else:
        st.success(f"Transaction Appears Legitimate")
        
    st.metric(label="Fraud Probability", value=f"{prediction_proba:.2%}")
    st.info(f"Note: This prediction is based on a business - optimized threshold of {business_threshold:.2%}.")
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Load model and scaler
model = load_model(r"C:\Users\pvjay\Desktop\projects\done_by_me\Data_Analyst\Predicting_Stock_Prices_using_Time_Series_analysis\apple_lstm_model.h5")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# App title
st.title("Apple Stock Price Prediction")

# Upload CSV
uploaded_file = st.file_uploader("Upload Apple stock CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={'Close/Last': 'Close'})
    
    # Clean numeric columns
    for col in ['Open','High','Low','Close','Volume']:
        df[col] = df[col].astype(str).str.replace('$','').str.replace(',','')
        df[col] = df[col].astype(float)

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    
    # Scale input data
    scaled_data = scaler.transform(df[['Open','High','Low','Close','Volume']])
    
    # Prepare data for prediction (use last 60 days)
    time_step = 60
    X_input = []
    X_input.append(scaled_data[-time_step:])
    X_input = np.array(X_input)
    
    # Predict next day
    pred_scaled = model.predict(X_input)
    
    # Reverse scale prediction
    close_scaler = scaler
    close_pred = pred_scaled * (close_scaler.data_range_[3]) + close_scaler.data_min_[3]
    
    st.write(f"Predicted Next Day Closing Price: ${close_pred[0][0]:.2f}")

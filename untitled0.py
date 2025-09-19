# Step 1: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Step 2: Load dataset
df = pd.read_csv(r"C:\Users\pvjay\Desktop\projects\done_by_me\Data_Analyst\Predicting_Stock_Prices_using_Time_Series_analysis\apple_stocks.csv")  # Replace with your CSV path

# Step 3: Clean column names
df.columns = df.columns.str.strip()
df = df.rename(columns={'Close/Last': 'Close'})

# Step 4: Clean numeric columns
for col in ['Open','High','Low','Close','Volume']:
    df[col] = df[col].astype(str)
    df[col] = df[col].str.replace('$','').str.replace(',','')
    df[col] = df[col].astype(float)

# Step 5: Convert Date to datetime and set as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.sort_index()

# Step 6: Select relevant columns
df = df[['Open','High','Low','Close','Volume']]

# Step 7: Visualize closing price
plt.figure(figsize=(14,5))
plt.plot(df['Close'], label='Closing Price')
plt.title('Apple Stock Closing Price')
plt.xlabel('Date')
plt.ylabel('Price USD')
plt.legend()
plt.show()

# Step 8: Scale data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df)

# Step 9: Create training and test sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size-60:]  # include 60 previous days for sliding window

def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i-time_step:i])  # all features
        Y.append(dataset[i, 3])           # 'Close' column (index 3)
    return np.array(X), np.array(Y)

X_train, Y_train = create_dataset(train_data, 60)
X_test, Y_test = create_dataset(test_data, 60)

# Step 10: Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))  # Predict 'Close'

model.compile(optimizer='adam', loss='mean_squared_error')

# Step 11: Train the model
history = model.fit(X_train, Y_train, batch_size=32, epochs=50, validation_split=0.1)

# Step 12: Make predictions
predictions = model.predict(X_test)

# Step 13: Scale back predictions and Y_test
close_scaler = MinMaxScaler()
close_scaler.min_, close_scaler.scale_ = scaler.min_[3], scaler.scale_[3]  # Only for 'Close'
predictions = close_scaler.inverse_transform(predictions.reshape(-1,1))
Y_test_actual = close_scaler.inverse_transform(Y_test.reshape(-1,1))

# Step 14: Prepare x-axis matching predictions
test_index = df.index[train_size:]      # all test rows
test_index = test_index[:len(Y_test)]   # match length of predictions/Y_test

# Step 15: Visualize predictions
plt.figure(figsize=(14,5))
plt.plot(test_index, Y_test_actual, label='Actual Price')
plt.plot(test_index, predictions, label='Predicted Price')
plt.title('Apple Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price USD')
plt.legend()
plt.show()

# Step 16: Evaluate model
rmse = np.sqrt(mean_squared_error(Y_test_actual, predictions))
mae = mean_absolute_error(Y_test_actual, predictions)
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")



# Step 1: Save LSTM model
model.save("apple_lstm_model.h5")
print("Model saved as apple_lstm_model.h5")

import pickle

# Save the scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Scaler saved as scaler.pkl")


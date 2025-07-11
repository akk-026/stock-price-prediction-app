import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import math
import os

# 1. Download AAPL data
ticker = "AAPL"
df = yf.download(ticker, start="2010-01-01", progress=False)
df1 = df['Close'].values.reshape(-1, 1)

# 2. Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
df1_scaled = scaler.fit_transform(df1)

# 3. Train/test split
training_size = int(len(df1_scaled) * 0.65)
test_size = len(df1_scaled) - training_size
train_data = df1_scaled[0:training_size, :]
test_data = df1_scaled[training_size:, :]

# 4. Create dataset for LSTM
def create_dataset(dataset, time_step=100):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape for LSTM [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# 5. Build conservative LSTM model
model = Sequential([
    LSTM(32, return_sequences=True, input_shape=(time_step, 1)),
    Dropout(0.3),
    LSTM(32, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1)
])
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

# 6. Add early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# 7. Train model with early stopping
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# 8. Predict and inverse transform
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# 9. Calculate RMSE
train_rmse = math.sqrt(mean_squared_error(y_train_inv, train_predict))
test_rmse = math.sqrt(mean_squared_error(y_test_inv, test_predict))
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")

# 10. Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 11. Plot results
plt.subplot(1, 2, 2)
plt.plot(scaler.inverse_transform(df1_scaled), label='Actual Price', alpha=0.7)
# Shift train predictions for plotting
trainPredictPlot = np.empty_like(df1_scaled)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[time_step:len(train_predict) + time_step, :] = train_predict
# Shift test predictions for plotting
testPredictPlot = np.empty_like(df1_scaled)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict) + (time_step * 2) + 1:len(df1_scaled) - 1, :] = test_predict
plt.plot(trainPredictPlot, label='Train Prediction', alpha=0.8)
plt.plot(testPredictPlot, label='Test Prediction', alpha=0.8)
plt.legend()
plt.title("AAPL Stock Price Prediction (Conservative LSTM)")
plt.tight_layout()
plt.show()

# 12. Test future predictions with conservative approach
print("\nTesting future predictions...")
# Get the last 100 values for prediction
last_100 = df1_scaled[-100:].reshape(1, -1)
temp_input = last_100[0].tolist()
lst_output = []

# Generate predictions for next 30 days with conservative limits
current_price = float(df['Close'].iloc[-1])
max_change_percent = 0.15  # Maximum 15% change in 30 days

for i in range(30):
    x_input = np.array(temp_input[-100:]).reshape(1, 100, 1)
    yhat = model.predict(x_input, verbose=0)
    
    # Apply conservative limits
    predicted_price = scaler.inverse_transform(yhat)[0][0]
    max_allowed = current_price * (1 + max_change_percent)
    min_allowed = current_price * (1 - max_change_percent)
    
    # Clamp the prediction to reasonable bounds
    if predicted_price > max_allowed:
        predicted_price = max_allowed
    elif predicted_price < min_allowed:
        predicted_price = min_allowed
    
    # Convert back to scaled value
    yhat_scaled = scaler.transform([[predicted_price]])[0][0]
    
    temp_input.append(yhat_scaled)
    lst_output.append(yhat_scaled)

# Inverse transform predictions
pred_values = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))

print(f"Current AAPL price: ${current_price:.2f}")
print(f"Predicted price after 30 days: ${pred_values[-1][0]:.2f}")
print(f"Predicted change: {((pred_values[-1][0] - current_price) / current_price * 100):.2f}%")

# Plot future predictions
plt.figure(figsize=(12, 6))
plt.plot(range(len(df)), df['Close'], label='Historical Price', color='blue')
plt.plot(range(len(df), len(df) + 30), pred_values.flatten(), 
         label='Future Prediction', color='red', linestyle='--')
plt.title("AAPL Stock Price: Historical + 30-Day Prediction (Conservative)")
plt.xlabel("Days")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 13. Save model and scaler
os.makedirs("model", exist_ok=True)
model.save("model/lstm_model.h5")
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Conservative model and scaler saved in 'model/' directory.") 
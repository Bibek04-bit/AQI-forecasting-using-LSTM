# Import required libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
from sklearn.model_selection import TimeSeriesSplit

# Load Dataset
file_path = "your_dataset.csv"  # Change this to your file
data = pd.read_csv(file_path, parse_dates=["date_and_time"], index_col="date_and_time")

# Selecting Features and Target
features = ["PM2.5", "O3", "TEMPERATURE", "PRESSURE", "DEWPOINT", "RAIN"]
target = "PM2.5"  # Predicting PM2.5 levels

# Scaling Data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features])

# Splitting into Train & Test Sets (80-20)
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# Function to Create Sequences
def create_sequences(data, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])
        y.append(data[i + time_steps, 0])  # Target is PM2.5
    return np.array(X), np.array(y)

time_steps = 10
X_train, y_train = create_sequences(train_data, time_steps)
X_test, y_test = create_sequences(test_data, time_steps)

# Define LSTM Model for Hyperparameter Tuning
def build_lstm_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=128, step=32), 
                   return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(LSTM(units=hp.Int('units_2', min_value=32, max_value=128, step=32)))
    model.add(Dense(1))  # Output Layer
    model.compile(loss="mse", optimizer=Adam(learning_rate=hp.Choice("learning_rate", [0.001, 0.0005, 0.0001])))
    return model

# Hyperparameter Tuning
tuner = kt.RandomSearch(
    build_lstm_model, objective="val_loss", max_trials=5, executions_per_trial=2, directory="lstm_tuning"
)

tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=1)

# Get Best Hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Train Best Model
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Predictions
y_pred = best_model.predict(X_test)

# Rescale Predictions & Actual Values
y_pred_rescaled = scaler.inverse_transform(np.concatenate((y_pred, np.zeros((y_pred.shape[0], len(features) - 1))), axis=1))[:, 0]
y_test_rescaled = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], len(features) - 1))), axis=1))[:, 0]

# Model Evaluation
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
r2 = r2_score(y_test_rescaled, y_pred_rescaled)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R² Score: {r2}")


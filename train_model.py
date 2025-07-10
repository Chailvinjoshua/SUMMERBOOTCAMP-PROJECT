import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# --- Configuration ---
# Update DATASET_PATH to point to sales_data.csv
DATASET_PATH = "sales_data.csv"
# New model save path to distinguish it from the age prediction model
MODEL_SAVE_PATH = "sales_forecast_model.pkl"

# --- Step 1: Load dataset ---
try:
    df = pd.read_csv(DATASET_PATH)
    print(f"✅ Dataset loaded successfully from {DATASET_PATH}")
except FileNotFoundError:
    print(f"Error: Dataset file '{DATASET_PATH}' not found.")
    print("Please ensure 'sales_data.csv' is in the same directory as this script.")
    exit()

# Convert 'Date' column to datetime objects
df['Date'] = pd.to_datetime(df['Date'])

# Sort data by date to ensure correct time series order
df = df.sort_values(by='Date')

# Create a numerical feature from the date (e.g., days since the first date)
# This is crucial for using linear regression on time series data
df['Days'] = (df['Date'] - df['Date'].min()).dt.days

# --- Step 2: Define features and label ---
# Features will be 'Days' for time-based forecasting
X = df[['Days']]
# Target will be 'Sales'
y = df['Sales']

# --- Step 3: Split into training and testing sets ---
# test_size=0.2 means 20% of data for testing, 80% for training
# random_state ensures reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Dataset split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")

# --- Step 4: Train the model ---
# Using Linear Regression as specified (suitable for linear trends over time)
model = LinearRegression()
model.fit(X_train, y_train)
print("✅ Linear Regression model trained for sales forecasting.")

# --- Step 5: Predict on test set ---
y_pred = model.predict(X_test)
print("Predictions made on the test set.")

# --- Step 6: Evaluate the model ---
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse) # RMSE is more interpretable as it's in the same units as the target
r2 = r2_score(y_test, y_pred) # R-squared measures how well the model fits the data

print("\n✅ Model Evaluation Results:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score (Accuracy): {r2:.2f}")
print("RMSE indicates the typical error in sales predictions (e.g., predicted sales are off by this amount).")
print("R² score indicates the proportion of variance in sales that is predictable from the 'Days' feature.")

# --- Step 7: Save the trained model ---
# The model will be saved as 'sales_forecast_model.pkl' in the same directory as this script.
# You will need to update your Flask app (app.py) to load this new model file.
joblib.dump(model, MODEL_SAVE_PATH)
print(f"✅ Model saved as {MODEL_SAVE_PATH}")

print("\nTraining complete. Remember to update your Flask application (`app.py`) to load 'sales_forecast_model.pkl'.")

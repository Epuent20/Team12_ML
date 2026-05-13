import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# file paths

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH        = os.path.join(SCRIPT_DIR, 'data/raw/cold_source_control_dataset.csv')
PROCESSED_DATA_DIR   = os.path.join(SCRIPT_DIR, 'data/processed')
PROCESSED_FILE_PATH  = os.path.join(PROCESSED_DATA_DIR, 'cleaned_telemetry.csv')
MODEL_SAVE_PATH      = os.path.join(SCRIPT_DIR, 'random_forest_model.pkl')
OUTPUT_DIR           = os.path.join(SCRIPT_DIR, 'output')
PERF_GRAPH_PATH      = os.path.join(OUTPUT_DIR, 'model_performance_graph.png')

TARGET_COL = 'Output'
ACTION_COL = 'Cooling_Strategy_Action'


# data cleaning

def clean_data():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    print("Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH)

    print("Dropping broken sensor readings...")
    df = df.dropna()

    # Isolate the exact features we need, but keep the numbers RAW
    features = ['Server_Workload(%)', 'Inlet_Temperature(°C)', 'Cooling_Unit_Power_Consumption(kW)']
    df_clean = df[features]

    df_clean.to_csv(PROCESSED_FILE_PATH, index=False)
    print(f"Success! Cleaned raw data saved to: {PROCESSED_FILE_PATH}")

# train model

def train_and_visualize():
    print("Loading processed data...")
    df = pd.read_csv(PROCESSED_FILE_PATH)

    # Define Features and Target
    X = df[['Server_Workload(%)', 'Inlet_Temperature(°C)']]
    y = df['Cooling_Unit_Power_Consumption(kW)']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Make Predictions for the Test Set
    y_pred = rf_model.predict(X_test)

    # Calculate Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print(f"\nModel Performance:\nRMSE: {rmse:.4f} kW\nR2: {r2:.4f}")

    # Create the Regression Graph
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='#2ca02c', label='Predictions')
    line_coords = [y_test.min(), y_test.max()]
    plt.plot(line_coords, line_coords, color='red', linestyle='--', lw=2, label='Perfect Prediction')
    plt.title('Model Evaluation: Actual vs. Predicted Cooling Power')
    plt.xlabel('Actual Cooling Power (kW)')
    plt.ylabel('AI Predicted Cooling Power (kW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(PERF_GRAPH_PATH)
    print(f"\nSuccess! Performance graph saved as '{PERF_GRAPH_PATH}'")
    plt.show()

    # Save the model
    joblib.dump(rf_model, MODEL_SAVE_PATH)

# predictions

def run_predictions():
    try:
        rf_model = joblib.load(MODEL_SAVE_PATH)
        print("Success: Random Forest model loaded into memory.\n")
    except FileNotFoundError:
        print("Error: model not found. Run train_and_visualize() first.")
        return

    # Simulate a sudden spike in workload
    live_telemetry = pd.DataFrame({
        'Server_Workload(%)':    [95.0, 40.0],   # Heavy load vs light load
        'Inlet_Temperature(°C)': [26.5, 19.0]
    })

    print("--- Incoming Live Sensor Data ---")
    print(live_telemetry)

    predictions = rf_model.predict(live_telemetry)

    print("\n--- Model Cooling Predictions ---")
    print(f"Rack 1 (Heavy Load): Requires {predictions[0]:.2f} kW of cooling power.")
    print(f"Rack 2 (Light Load):  Requires {predictions[1]:.2f} kW of cooling power.")


# main pipeline

def run_pipeline():
    print("=== STEP 1: Cleaning and Preparing Local Kaggle Data ===")
    clean_data()

    print("\n=== STEP 2: Training the Random Forest Brain ===")
    train_and_visualize()
    

    print("\n=== STEP 3: Running a Live Test Case ===")
    run_predictions()

    print("\n=== Pipeline Complete: Local Model is Ready ===")


if __name__ == "__main__":
    run_pipeline()

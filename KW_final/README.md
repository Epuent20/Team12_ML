# Cooling Power Prediction — Random Forest Regressor

A machine-learning pipeline that predicts **cooling unit power consumption (kW)** for data-center racks using a Random Forest regression model trained on real sensor telemetry.

## Overview

Data centers must continuously adjust cooling output in response to fluctuating server workloads and inlet temperatures. This project builds a predictive model that:

1. **Cleans** raw sensor data from a Kaggle cold-source-control dataset.
2. **Trains** a Random Forest regressor on workload and temperature features.
3. **Evaluates** model accuracy (RMSE & R²) and generates an Actual vs. Predicted scatter plot.
4. **Predicts** cooling power requirements for live telemetry readings.

## Project Structure

```
KW_final/
├── data/
│   ├── raw/                         # Original Kaggle dataset
│   │   └── cold_source_control_dataset.csv
│   └── processed/                   # Cleaned feature subset (generated)
│       └── cleaned_telemetry.csv
├── output/
│   └── model_performance_graph.png  # Actual vs. Predicted plot (generated)
├── kW.py                            # Main pipeline script
├── random_forest_model.pkl          # Saved model artifact (generated)
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/KW_final.git
cd KW_final

# 2. Create a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline
python kW.py
```

## Pipeline Steps

| Step | Function               | What It Does                                                        |
|------|------------------------|---------------------------------------------------------------------|
| 1    | `clean_data()`         | Drops NaN rows, isolates key features, saves `cleaned_telemetry.csv`|
| 2    | `train_and_visualize()`| Trains 100-tree RF regressor, prints RMSE/R², saves model & plot    |
| 3    | `run_predictions()`    | Loads the saved model and predicts kW for two sample scenarios      |

## Features Used

| Feature                  | Description                            |
|--------------------------|----------------------------------------|
| `Server_Workload(%)`     | Current server utilization percentage  |
| `Inlet_Temperature(°C)`  | Air temperature entering the rack      |

**Target:** `Cooling_Unit_Power_Consumption(kW)`

## Results

The trained model achieves strong predictive accuracy on the held-out test set. Running the pipeline prints RMSE and R² to the console and saves a scatter plot to `output/model_performance_graph.png`.

## Dataset

The raw data (`cold_source_control_dataset.csv`) originates from a Kaggle cold-source-control dataset containing operational sensor readings such as workload, inlet/outlet temperatures, and cooling power consumption.

## License

This project is provided for educational and research purposes.

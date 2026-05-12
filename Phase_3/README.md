# Data Center Cooling Strategy Optimizer

A Random Forest classifier that predicts the optimal cooling strategy for a data center based on historical sensor data and cost-optimal analysis.

## Problem

Given 9 sensor features from a data center's cooling system, predict which of 5 cooling strategies minimizes energy cost while maintaining safe temperatures:

| Code | Strategy | When to use |
|------|----------|-------------|
| 0 | Increase Chiller | High workload + warm inlet temps |
| 1 | Reduce AHU | Low ambient temps, AHU is overworking |
| 2 | Maintain | Stable mid-range conditions |
| 3 | Boost All | High deviation, needs aggressive cooling |
| 4 | Eco Mode | Low workload + cool conditions |

## Features

| Feature | Description |
|---------|-------------|
| Server_Workload(%) | Current server utilization |
| Inlet_Temperature(°C) | Cold aisle temperature |
| Outlet_Temperature(°C) | Hot aisle temperature |
| Ambient_Temperature(°C) | Outside/room temperature |
| Cooling_Unit_Power_Consumption(kW) | Current cooling energy draw |
| Chiller_Usage(%) | Chiller capacity utilization |
| AHU_Usage(%) | Air handling unit utilization |
| Total_Energy_Cost($) | Current hourly energy cost |
| Temperature_Deviation(°C) | Deviation from target temperature |

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Train and evaluate
python cooling_strategy_optimizer.py
```

## Project structure

```
├── cooling_strategy_optimizer.py    # Main pipeline (label engineering, training, evaluation)
├── plot_labels.py                   # Visualizations for optimized (new) labels
├── plot_old_labels.py               # Visualizations for original (old) CSV labels
├── data/
│   └── cold_source_control_dataset.csv  # Dataset (3,498 hourly readings)
├── output/                          # Generated after running
│   ├── random_forest_model.pkl      # Trained model
│   ├── labeled_dataset.csv          # Dataset with engineered labels
│   ├── 01_correlation_matrix.png    # Feature correlations
│   ├── 02_class_distribution.png    # Class balance chart
│   ├── 03_confusion_matrix.png      # Prediction errors
│   ├── 04_feature_importance.png    # MDI feature importance
│   ├── new_labels_scatter_grid.png  # Scatter grid (optimized labels)
│   ├── new_labels_pca_projection.png
│   ├── new_labels_tsne_projection.png
│   ├── old_labels_scatter_grid.png  # Scatter grid (original labels)
│   ├── old_labels_pca_projection.png
│   └── old_labels_tsne_projection.png
├── requirements.txt
└── README.md
```

## Note on labels

The original dataset labels (`Cooling_Strategy_Action`) show near-identical feature distributions across all 5 strategies, making them effectively random. This project redefines the target by computing the *cost-optimal* strategy for each operating condition — the strategy that historically minimized `energy_cost + 0.01 × temperature_deviation` within binned workload, inlet temperature, and ambient temperature ranges.

# Data Center Cooling Strategy Optimization

Predicting the optimal cooling strategy for a data center using sensor data — workload, temperatures, and energy metrics.

## Dataset

The raw data (`cold_source_control_dataset.csv`) originates from a Kaggle cold-source-control dataset containing operational sensor readings such as workload, inlet/outlet temperatures, and cooling power consumption.

3,498 hourly readings from a data center cooling system (Jan–May 2025). 9 sensor features, 5 cooling strategies: Increase Chiller, Reduce AHU, Maintain, Boost All, Eco Mode.

## Project phases

### Phase 1 — Classification on original labels

Trained Random Forest, SVM, and KNN on the dataset's original strategy labels. All three scored ~20% accuracy — random chance for 5 classes. The labels have no correlation with any sensor feature.

**Script:** `classifiers_analysis.py`

### Phase 2 — Regression pivot

Switched to predicting `Cooling_Unit_Power_Consumption(kW)` from workload and inlet temperature to verify the features themselves are meaningful. Random Forest Regressor achieved R² = 0.855, confirming the data contains strong signal — the problem was the labels, not the features.

**Script:** `kW.py`

### Phase 3 — Offline policy learning

Defined a cost-minimization objective (`energy_cost + 0.01 × temp_deviation`), binned operating conditions into 36 groups (4 workload × 3 inlet temp × 3 ambient temp), identified the best-performing strategy per group, and relabeled the dataset. Trained a Random Forest classifier on the engineered labels.

**Result:** 98.86% test accuracy (692/700 correct).

**Scripts:** `cooling_strategy_optimizer.py`, `plot_labels.py`, `plot_old_labels.py`

## Quick start


# Phase 1 — classifier comparison
python classifiers_analysis.py

# Phase 2 — regression model
python kW.py

# Phase 3 — final model
python cooling_strategy_optimizer.py
python plot_labels.py
python plot_old_labels.py


## Output

All results save to `output/`. Key artifacts:

- `random_forest_model.pkl` — trained Phase 3 model
- `*.png` — confusion matrices, feature importance, PCA/t-SNE projections, old vs new label comparisons

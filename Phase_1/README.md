# 🌡️ Cold Source Control — Multi-Classifier Comparison

A machine-learning pipeline that trains and evaluates **three classifiers** on a data-center cooling-strategy dataset, then generates publication-ready visualizations comparing their performance.

| Model | Highlights |
|-------|-----------|
| **Random Forest** | 200 estimators, feature-importance analysis |
| **SVM (RBF)** | Standardized features, γ = scale |
| **K-Nearest Neighbors** | k = 7, uniform weighting |

---

## 📂 Project Structure

```
tree_classifier/
├── classifiers_analysis.py   # Main pipeline (train → evaluate → visualize)
├── data/
│   └── cold_source_control_dataset.csv
├── output/                   # Generated after running the script
│   ├── confusion_matrices.png
│   ├── accuracy_f1_comparison.png
│   ├── per_class_f1_heatmap.png
│   ├── rf_feature_importance.png
│   └── class_distribution.png
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
python classifiers_analysis.py
```

The script will:

1. Load `data/cold_source_control_dataset.csv`
2. Encode the `Cooling_Strategy_Action` label and split into 75 / 25 train-test sets
3. Train **Random Forest**, **SVM**, and **KNN** models
4. Print a full classification report with accuracy, weighted F1, and 5-fold cross-validation scores
5. Save five visualizations to the `output/` directory

---

## 📊 Visualizations

| File | Description |
|------|-------------|
| `confusion_matrices.png` | Side-by-side confusion matrices for all three models |
| `accuracy_f1_comparison.png` | Grouped bar chart of test accuracy vs weighted F1-score |
| `per_class_f1_heatmap.png` | Heatmap of per-class F1-scores across models |
| `rf_feature_importance.png` | Horizontal bar chart of Random Forest feature importances |
| `class_distribution.png` | Dataset class distribution bar chart |

---

## 🗂️ Dataset

The dataset (`cold_source_control_dataset.csv`) contains operational sensor readings from a data-center cooling system with the following features:

- `Server_Workload(%)` — Current server workload
- `Inlet_Temperature(°C)` — Cold-aisle inlet temperature
- `Outlet_Temperature(°C)` — Hot-aisle outlet temperature
- `Ambient_Temperature(°C)` — Outside ambient temperature
- `Cooling_Unit_Power_Consumption(kW)` — Cooling unit energy draw
- `Chiller_Usage(%)` — Chiller utilization
- `AHU_Usage(%)` — Air-handling unit utilization
- `Total_Energy_Cost($)` — Total energy cost
- `Temperature_Deviation(°C)` — Deviation from setpoint

**Target variable:** `Output` — encoded cooling strategy class derived from `Cooling_Strategy_Action`.

---



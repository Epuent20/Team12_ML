"""
Cold Source Control Dataset — Multi-Classifier Comparison
=========================================================
Three classifiers trained on the cooling-strategy dataset:
  1. Random Forest Classifier
  2. Support Vector Machine (SVM)
  3. K-Nearest Neighbors (KNN)

Each model is evaluated with accuracy, classification report,
confusion matrix, and additional comparison visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
import os
import warnings

warnings.filterwarnings("ignore")

# output directory 
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# styling
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "font.family": "monospace",
    "font.size": 10,
})

PALETTE = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff"]
MODEL_COLORS = {"Random Forest": "#58a6ff", "SVM": "#3fb950", "KNN": "#d29922"}

# 1. load & prepare dataset
df = pd.read_csv(os.path.join("data", "cold_source_control_dataset.csv"))

# Encode the strategy label
le = LabelEncoder()
df["Strategy_Encoded"] = le.fit_transform(df["Cooling_Strategy_Action"])

# Features: all numeric columns except Output and the encoded strategy
feature_cols = [
    "Server_Workload(%)",
    "Inlet_Temperature(°C)",
    "Outlet_Temperature(°C)",
    "Ambient_Temperature(°C)",
    "Cooling_Unit_Power_Consumption(kW)",
    "Chiller_Usage(%)",
    "AHU_Usage(%)",
    "Total_Energy_Cost($)",
    "Temperature_Deviation(°C)",
]

X = df[feature_cols].values
y = df["Output"].values

class_names = le.classes_  # original strategy names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Scale features (important for SVM & KNN)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# 2. define & train models
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
    ),
    "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7, weights="uniform", n_jobs=-1),
}

results = {}
predictions = {}

print("=" * 70)
print("       COLD SOURCE CONTROL — CLASSIFIER COMPARISON REPORT")
print("=" * 70)
print(f"\nDataset size  : {len(df)} samples")
print(f"Train / Test  : {len(X_train)} / {len(X_test)}")
print(f"Features used : {len(feature_cols)}")
print(f"Classes       : {list(range(5))}  ->  {list(class_names)}")
print()

for name, model in models.items():
    # SVM and KNN use scaled data
    Xtr = X_train_sc if name in ("SVM", "KNN") else X_train
    Xte = X_test_sc if name in ("SVM", "KNN") else X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    cv = cross_val_score(model, Xtr, y_train, cv=5, scoring="accuracy", n_jobs=-1)

    results[name] = {
        "accuracy": acc,
        "f1_weighted": f1,
        "cv_mean": cv.mean(),
        "cv_std": cv.std(),
        "cv_scores": cv,
    }
    predictions[name] = y_pred

    print("-" * 70)
    print(f"  MODEL: {name}")
    print("-" * 70)
    print(f"  Test Accuracy       : {acc:.4f}")
    print(f"  Weighted F1-Score   : {f1:.4f}")
    print(f"  5-Fold CV Accuracy  : {cv.mean():.4f} ± {cv.std():.4f}")
    print(f"  CV per fold         : {np.round(cv, 4)}")
    print()
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4, zero_division=0))
    print()

# 3. summary table
print("=" * 70)
print("  QUICK COMPARISON")
print("=" * 70)
summary = pd.DataFrame(
    {
        "Model": list(results.keys()),
        "Accuracy": [r["accuracy"] for r in results.values()],
        "F1 (weighted)": [r["f1_weighted"] for r in results.values()],
        "CV Mean": [r["cv_mean"] for r in results.values()],
        "CV Std": [r["cv_std"] for r in results.values()],
    }
)
print(summary.to_string(index=False))
print()

# 4. visualizations

# 4-A confusion matrices (one per model)
fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
fig.suptitle("Confusion Matrices", fontsize=16, fontweight="bold", y=1.02)

for ax, (name, y_pred) in zip(axes, predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        linecolor="#30363d",
        cbar_kws={"shrink": 0.75},
    )
    ax.set_title(name, fontsize=13, fontweight="bold", color=MODEL_COLORS[name])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices.png"), dpi=180, bbox_inches="tight")
plt.close()

# 4-B accuracy & F1 bar chart
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(results))
width = 0.3

acc_vals = [r["accuracy"] for r in results.values()]
f1_vals = [r["f1_weighted"] for r in results.values()]

bars1 = ax.bar(x - width / 2, acc_vals, width, label="Accuracy", color="#58a6ff", edgecolor="#0f1117")
bars2 = ax.bar(x + width / 2, f1_vals, width, label="F1 (weighted)", color="#3fb950", edgecolor="#0f1117")

for b in bars1:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
            f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9, color="#c9d1d9")
for b in bars2:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
            f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9, color="#c9d1d9")

ax.set_xticks(x)
ax.set_xticklabels(results.keys(), fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score")
ax.set_title("Test Accuracy vs Weighted F1-Score", fontsize=14, fontweight="bold")
ax.legend(loc="upper right", framealpha=0.6)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_f1_comparison.png"), dpi=180, bbox_inches="tight")
plt.close()


# 4-D per-class F1-score heatmap
per_class_f1 = {}
for name, y_pred in predictions.items():
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    per_class_f1[name] = [report[c]["f1-score"] for c in class_names]

f1_df = pd.DataFrame(per_class_f1, index=class_names)

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(
    f1_df,
    annot=True,
    fmt=".3f",
    cmap="YlGnBu",
    linewidths=0.6,
    linecolor="#30363d",
    ax=ax,
    vmin=0,
    vmax=1,
    cbar_kws={"label": "F1-Score"},
)
ax.set_title("Per-Class F1-Score by Model", fontsize=14, fontweight="bold")
ax.set_ylabel("Cooling Strategy")
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "per_class_f1_heatmap.png"), dpi=180, bbox_inches="tight")
plt.close()

# 4-E feature importance (Random Forest only)
rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
idx = np.argsort(importances)

fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.barh(
    np.array(feature_cols)[idx],
    importances[idx],
    color="#58a6ff",
    edgecolor="#0f1117",
    height=0.6,
)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{w:.3f}", va="center", fontsize=9, color="#c9d1d9")

ax.set_xlabel("Importance")
ax.set_title("Random Forest — Feature Importances", fontsize=14, fontweight="bold")
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rf_feature_importance.png"), dpi=180, bbox_inches="tight")
plt.close()

# 4-F class distribution in dataset
fig, ax = plt.subplots(figsize=(7, 4.5))
counts = df["Output"].value_counts().sort_index()
bars = ax.bar(
    [class_names[i] for i in counts.index],
    counts.values,
    color=PALETTE,
    edgecolor="#0f1117",
    width=0.55,
)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
            str(val), ha="center", fontsize=10, color="#c9d1d9", fontweight="bold")

ax.set_ylabel("Count")
ax.set_title("Dataset Class Distribution", fontsize=14, fontweight="bold")
ax.tick_params(axis="x", rotation=25)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=180, bbox_inches="tight")
plt.close()

print(f"[Success] All visualizations saved to '{OUTPUT_DIR}/'")
print(f"   • {OUTPUT_DIR}/confusion_matrices.png")
print(f"   • {OUTPUT_DIR}/accuracy_f1_comparison.png")
print(f"   • {OUTPUT_DIR}/per_class_f1_heatmap.png")
print(f"   • {OUTPUT_DIR}/rf_feature_importance.png")
print(f"   • {OUTPUT_DIR}/class_distribution.png")

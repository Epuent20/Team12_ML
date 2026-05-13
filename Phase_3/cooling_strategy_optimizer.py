"""
Data Center Cooling Strategy Optimizer — Random Forest
Predicts the optimal cooling strategy for a data center based on
the historical data and the cost-optimal cooling strategy
"""

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# setting paths and variables

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "data", "cold_source_control_dataset.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
MODEL_PATH = os.path.join(OUTPUT_DIR, "random_forest_model.pkl")
LABELED_DATASET_PATH = os.path.join(OUTPUT_DIR, "labeled_dataset.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURE_COLS = [
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

LABEL_MAP = {
    0: "Increase Chiller",
    1: "Reduce AHU",
    2: "Maintain",
    3: "Boost All",
    4: "Eco Mode",
}

STRATEGY_ENCODE = {v: k for k, v in LABEL_MAP.items()}

PALETTE = {
    "Increase Chiller": "#E05C2A",
    "Reduce AHU":       "#3D8EDE",
    "Maintain":         "#27AE60",
    "Boost All":        "#9B59B6",
    "Eco Mode":         "#F1C40F",
}


# 0. Label engineering, cost function 

def compute_labels(df):
    """Bin operating conditions and assign the cost-optimal strategy per bin."""
    df = df.copy()
    df["Workload_Bin"] = pd.qcut(
        df["Server_Workload(%)"], q=4, labels=False, duplicates="drop"
    )
    df["Temp_Bin"] = pd.qcut(
        df["Inlet_Temperature(°C)"], q=3, labels=False, duplicates="drop"
    )
    df["Ambient_Bin"] = pd.qcut(
        df["Ambient_Temperature(°C)"], q=3, labels=False, duplicates="drop"
    )
    df["Cost_Score"] = (
        df["Total_Energy_Cost($)"] + 0.01 * df["Temperature_Deviation(°C)"]
    )

    optimal = (
        df.groupby(
            ["Workload_Bin", "Temp_Bin", "Ambient_Bin", "Cooling_Strategy_Action"]
        )["Cost_Score"]
        .mean()
        .reset_index()
    )
    best_idx = (
        optimal.groupby(["Workload_Bin", "Temp_Bin", "Ambient_Bin"])["Cost_Score"]
        .idxmin()
        .dropna()
    )
    best = optimal.loc[best_idx]
    best_map = best.set_index(["Workload_Bin", "Temp_Bin", "Ambient_Bin"])[
        "Cooling_Strategy_Action"
    ].to_dict()

    df["Optimal_Strategy"] = df.apply(
        lambda r: best_map.get(
            (r["Workload_Bin"], r["Temp_Bin"], r["Ambient_Bin"]), "Eco Mode"
        ),
        axis=1,
    )
    df["Target"] = df["Optimal_Strategy"].map(STRATEGY_ENCODE)
    return df


# 1. Data loading and preparation

def load_and_prepare(path):
    """Load dataset and create optimized cooling strategy labels."""
    print("─" * 50)
    print("STEP 1 — Loading and preparing data")
    print("─" * 50)

    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} rows × {len(df.columns)} columns")

    df = compute_labels(df)

    print(f"\n  Label distribution:")
    for strategy, count in df["Optimal_Strategy"].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {strategy:20s} — {count:5d} ({pct:.1f}%)")

    return df


# 2. Exploratory data analysis

def run_eda(df):
    """Generate and save a correlation heatmap and class distribution chart."""
    print("\n" + "─" * 50)
    print("STEP 2 — Exploratory data analysis")
    print("─" * 50)

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[FEATURE_COLS].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, square=True, linewidths=0.5, ax=ax,
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "01_correlation_matrix.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 01_correlation_matrix.png")

    # Class distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df["Optimal_Strategy"].value_counts()
    colors = ["#1D9E75", "#378ADD", "#D85A30", "#7F77DD", "#D4537E"]
    ax.bar(range(len(counts)), counts.values, color=colors[:len(counts)])
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=20, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Strategy Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "02_class_distribution.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 02_class_distribution.png")


# 3. Model training

def train_model(X_train, y_train):
    """Train a Random Forest classifier with fixed hyperparameters."""
    print("\n" + "─" * 50)
    print("STEP 3 — Training Random Forest")
    print("─" * 50)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("  Training complete.")

    return model



# 4. Evaluation

def evaluate_model(model, X_test, y_test):
    """Evaluate model and generate confusion matrix and feature importance plots."""
    print("\n" + "─" * 50)
    print("STEP 4 — Evaluation")
    print("─" * 50)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n  Test accuracy: {acc:.4f}")
    print(f"\n  Classification report:")
    print(classification_report(y_test, y_pred, target_names=list(LABEL_MAP.values())))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens",
        xticklabels=list(LABEL_MAP.values()),
        yticklabels=list(LABEL_MAP.values()),
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "03_confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 03_confusion_matrix.png")

    # Feature importance (MDI)
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)
    short_names = [c.split("(")[0].strip() for c in FEATURE_COLS]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(importances)), importances[sorted_idx], color="#1D9E75")
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels([short_names[i] for i in sorted_idx])
    ax.set_xlabel("Importance (MDI)")
    ax.set_title("Feature Importance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "04_feature_importance.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 04_feature_importance.png")

    results = {
        "test_accuracy": round(acc, 4),
        "feature_importance": {
            FEATURE_COLS[i]: round(float(importances[i]), 4)
            for i in range(len(FEATURE_COLS))
        },
        "confusion_matrix": cm.tolist(),
    }
    return results



# 5. Save model and results

def save_artifacts(model, results):
    """Persist the trained model."""
    print("\n" + "─" * 50)
    print("STEP 5 — Saving artifacts")
    print("─" * 50)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"  Model → {MODEL_PATH}")


# main function
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1 — Load data and engineer optimized labels
    df_labeled = load_and_prepare(DATASET_PATH)

    # Step 2 — EDA
    run_eda(df_labeled)

    # Step 3 — Split and train
    X = df_labeled[FEATURE_COLS].values
    y = df_labeled["Target"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    print(f"\n  Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    model = train_model(X_train, y_train)

    # Step 4 — Evaluate
    results = evaluate_model(model, X_test, y_test)

    # Step 5 — Save model, results, and labeled dataset
    save_artifacts(model, results)
    df_labeled.to_csv(LABELED_DATASET_PATH, index=False)
    print(f"  Dataset → {LABELED_DATASET_PATH}")

    # Step 6 — Label visualizations
    #   Both scripts receive df_labeled, which contains:
    #  "Optimal_Strategy" — new engineered labels  (used by plot_labels)
    #  "Cooling_Strategy_Action" — original CSV labels (used by plot_old_labels)

    print("\n" + "─" * 50)
    print("STEP 6 — Generating label visualizations")
    print("─" * 50)

    from plot_labels import plot_scatter_grid, plot_pca, plot_tsne
    from plot_old_labels import plot_scatter_grid as plot_old_scatter, plot_pca as plot_old_pca, plot_tsne as plot_old_tsne

    print("  New (optimized) labels:")
    plot_scatter_grid(df_labeled)
    plot_pca(df_labeled)
    plot_tsne(df_labeled)
    print("  Old (raw CSV) labels:")
    plot_old_scatter(df_labeled)
    plot_old_pca(df_labeled)
    plot_old_tsne(df_labeled)

    print("\n" + "=" * 50)
    print(f"  COMPLETE — Accuracy: {results['test_accuracy']*100:.1f}%")
    print(f"  Outputs saved to: {OUTPUT_DIR}/")
    print("=" * 50)


if __name__ == "__main__":
    main()

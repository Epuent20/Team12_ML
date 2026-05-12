"""
Old Label Datapoint Visualizer
===============================
Plots the ORIGINAL Cooling_Strategy_Action labels that came directly from the
CSV — no engineering, no re-assignment.  Compare these charts against the ones
produced by plot_labels.py (new / optimal labels) to see how much the label
engineering improved class separability.

All shared constants come from cooling_strategy_optimizer.py — this file only
contains plot functions.

Outputs (saved to output/):
    old_labels_scatter_grid.png    — 2-D scatter grid across key feature pairs
    old_labels_pca_projection.png  — PCA 2-D projection of all features
    old_labels_tsne_projection.png — t-SNE 2-D projection (slower, ~20-40 s)

Called by cooling_strategy_optimizer.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# ─── Shared config — imported from the main pipeline ─────────────────────────
from cooling_strategy_optimizer import (
    FEATURE_COLS, PALETTE, OUTPUT_DIR,
)
LABEL_COL = "Cooling_Strategy_Action"   # raw column — no recomputation

# ─── Helpers ──────────────────────────────────────────────────────────────────

def legend_patches(palette):
    return [mpatches.Patch(color=c, label=lbl) for lbl, c in palette.items()]

def get_colors(series, palette):
    return series.map(palette).fillna("#AAAAAA")

def dark_axes(ax):
    ax.set_facecolor("#1A1D27")
    ax.tick_params(colors="#AAAAAA", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")

# ─── Plot 1 — Scatter grid ────────────────────────────────────────────────────

def plot_scatter_grid(df):
    pairs = [
        ("Server_Workload(%)",              "Outlet_Temperature(°C)"),
        ("Inlet_Temperature(°C)",           "Cooling_Unit_Power_Consumption(kW)"),
        ("Chiller_Usage(%)",                "AHU_Usage(%)"),
        ("Total_Energy_Cost($)",            "Temperature_Deviation(°C)"),
        ("Ambient_Temperature(°C)",         "Total_Energy_Cost($)"),
        ("Server_Workload(%)",              "Chiller_Usage(%)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor("#0F1117")
    axes = axes.flatten()

    sample = df.sample(min(4000, len(df)), random_state=42)
    colors = get_colors(sample[LABEL_COL], PALETTE)

    for ax, (xcol, ycol) in zip(axes, pairs):
        dark_axes(ax)
        ax.scatter(
            sample[xcol], sample[ycol],
            c=colors, s=14, alpha=0.65, linewidths=0,
        )
        ax.set_xlabel(xcol.split("(")[0].strip(), color="#CCCCCC", fontsize=9)
        ax.set_ylabel(ycol.split("(")[0].strip(), color="#CCCCCC", fontsize=9)

    fig.legend(
        handles=legend_patches(PALETTE),
        loc="lower center", ncol=5,
        framealpha=0.15, labelcolor="white",
        fontsize=10, bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        "Original (Old) Cooling Strategy Labels — Feature-Space Scatter Grid",
        color="white", fontsize=15, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    out = os.path.join(OUTPUT_DIR, "old_labels_scatter_grid.png")
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out}")


# ─── Plot 2 — PCA ─────────────────────────────────────────────────────────────

def plot_pca(df):
    X      = df[FEATURE_COLS].dropna()
    labels = df.loc[X.index, LABEL_COL]

    Xs  = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    Z   = pca.fit_transform(Xs)
    ev  = pca.explained_variance_ratio_

    colors = get_colors(labels, PALETTE)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0F1117")
    dark_axes(ax)

    ax.scatter(Z[:, 0], Z[:, 1], c=colors, s=16, alpha=0.6, linewidths=0)
    ax.set_xlabel(f"PC 1 ({ev[0]*100:.1f}% var)", color="#CCCCCC", fontsize=10)
    ax.set_ylabel(f"PC 2 ({ev[1]*100:.1f}% var)", color="#CCCCCC", fontsize=10)
    ax.legend(handles=legend_patches(PALETTE), framealpha=0.2, labelcolor="white", fontsize=9)
    ax.set_title(
        "PCA Projection — Original (Old) Cooling Strategy Labels",
        color="white", fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "old_labels_pca_projection.png")
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out}")


# ─── Plot 3 — t-SNE ───────────────────────────────────────────────────────────

def plot_tsne(df, n_sample=3000):
    idx    = df[FEATURE_COLS].dropna().index
    sample = df.loc[idx].sample(min(n_sample, len(idx)), random_state=42)
    X      = sample[FEATURE_COLS].values
    labels = sample[LABEL_COL]

    Xs = StandardScaler().fit_transform(X)

    print("  Running t-SNE (may take ~20-40 s)…")
    Z      = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000).fit_transform(Xs)
    colors = get_colors(labels, PALETTE)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0F1117")
    dark_axes(ax)

    ax.scatter(Z[:, 0], Z[:, 1], c=colors, s=16, alpha=0.65, linewidths=0)
    ax.set_xlabel("t-SNE 1", color="#CCCCCC", fontsize=10)
    ax.set_ylabel("t-SNE 2", color="#CCCCCC", fontsize=10)
    ax.legend(handles=legend_patches(PALETTE), framealpha=0.2, labelcolor="white", fontsize=9)
    ax.set_title(
        "t-SNE Projection — Original (Old) Cooling Strategy Labels",
        color="white", fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "old_labels_tsne_projection.png")
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out}")




"""
Publication-Grade Plot Generator
==================================
Generates 4 comparison charts proving quantum advantage under fair conditions:
  1. Clean accuracy comparison (all 4 models)
  2. Adversarial robustness under PGD attack
  3. Robustness Retention Ratio = PGD / Clean  ← THE KEY QUANTUM ADVANTAGE CHART
  4. Parameter efficiency (accuracy per 1k params)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os

# Short display labels for the 4 models
LABEL_MAP = {
    "ClassicalCNN (Unconstrained)":            "ClassicalCNN\n(Unconstrained)",
    "FairCNN (Parameter-Matched to QCNN)":     "FairCNN\n(Param-Matched)",
    "HybridQNN":                               "HybridQNN\n(Hybrid)",
    "MultiClassQCNN (Pure Quantum)":           "QCNN\n(Pure Quantum)",
}

# Color palette — quantum models get warm highlight colors
COLORS = {
    "ClassicalCNN\n(Unconstrained)":  "#5B8DB8",
    "FairCNN\n(Param-Matched)":       "#7EB5D6",
    "HybridQNN\n(Hybrid)":            "#E8924A",
    "QCNN\n(Pure Quantum)":           "#E84A4A",
}


def annotate_bars(ax, fmt=".3f", offset=9):
    for p in ax.patches:
        h = p.get_height()
        if h > 0.0001:
            ax.annotate(format(h, fmt),
                        (p.get_x() + p.get_width() / 2., h),
                        ha="center", va="bottom",
                        xytext=(0, offset), textcoords="offset points",
                        fontsize=10, fontweight="bold")


def generate_plots(csv_path="results_comparison.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run train.py first.")
        return

    df = pd.read_csv(csv_path)

    # Apply short labels
    df["Model"] = df["Model"].map(lambda x: LABEL_MAP.get(x, x))

    clean_df = df[df["Condition"] == "Clean"].copy()
    pgd_df   = df[df["Condition"] == "PGD_Adversarial"].copy()

    # Merge to compute Robustness Retention Ratio per model
    merged = clean_df[["Model", "Accuracy", "NumParams", "AccuracyPer1000Params"]].merge(
        pgd_df[["Model", "Accuracy"]].rename(columns={"Accuracy": "PGD_Acc"}),
        on="Model"
    )
    merged["RetentionRatio"] = merged["PGD_Acc"] / merged["Accuracy"].replace(0, np.nan)

    model_order = list(LABEL_MAP.values())
    palette     = [COLORS.get(m, "#999999") for m in model_order]

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Fair Quantum Advantage Benchmark — HMBD-v1 Arabic OCR (115 Classes)\n"
        "All models: identical AdamW optimizer · same LR schedule · same data",
        fontsize=14, fontweight="bold", y=1.01
    )

    # ── Plot 1: Clean Accuracy ────────────────────────────────────────────────
    ax1 = axes[0, 0]
    sns.barplot(data=clean_df, x="Model", y="Accuracy",
                order=model_order, palette=palette, ax=ax1)
    ax1.set_title("① Clean Validation Accuracy", fontweight="bold")
    ax1.set_ylabel("Accuracy"); ax1.set_xlabel("")
    ax1.set_ylim(0, 1.1)
    annotate_bars(ax1)

    # ── Plot 2: PGD Adversarial Accuracy ─────────────────────────────────────
    ax2 = axes[0, 1]
    sns.barplot(data=pgd_df, x="Model", y="Accuracy",
                order=model_order, palette=palette, ax=ax2)
    ax2.set_title("② Accuracy Under PGD Adversarial Attack", fontweight="bold")
    ax2.set_ylabel("Accuracy Under Attack"); ax2.set_xlabel("")
    ax2.set_ylim(0, 1.1)
    annotate_bars(ax2)

    # ── Plot 3: Robustness Retention Ratio ← KEY CHART ───────────────────────
    ax3 = axes[1, 0]
    merged_ordered = merged.set_index("Model").reindex(model_order).reset_index()
    bar_colors = [COLORS.get(m, "#999999") for m in merged_ordered["Model"]]
    bars = ax3.bar(merged_ordered["Model"], merged_ordered["RetentionRatio"],
                   color=bar_colors, edgecolor="white", linewidth=0.8)
    ax3.set_title("③ Robustness Retention Ratio  (PGD Acc / Clean Acc)\n"
                  "← KEY QUANTUM ADVANTAGE METRIC: Higher = More Robust",
                  fontweight="bold")
    ax3.set_ylabel("Retention Ratio  [0 → 1]"); ax3.set_xlabel("")
    ax3.set_ylim(0, 1.15)
    ax3.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% retention")
    ax3.legend(fontsize=9)
    for bar, val in zip(bars, merged_ordered["RetentionRatio"]):
        if not np.isnan(val):
            ax3.text(bar.get_x() + bar.get_width()/2., val + 0.02,
                     f"{val:.3f}", ha="center", va="bottom",
                     fontsize=11, fontweight="bold")

    # ── Plot 4: Parameter Efficiency ─────────────────────────────────────────
    ax4 = axes[1, 1]
    sns.barplot(data=clean_df, x="Model", y="AccuracyPer1000Params",
                order=model_order, palette=palette, ax=ax4)
    ax4.set_title("④ Parameter Efficiency  (Clean Acc / 1k Params)", fontweight="bold")
    ax4.set_ylabel("Efficiency Score"); ax4.set_xlabel("")
    annotate_bars(ax4, fmt=".4f")

    plt.tight_layout()
    plt.savefig("quantum_advantage_benchmark.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Print summary table
    print("\n" + "="*70)
    print("  BENCHMARK RESULTS SUMMARY")
    print("="*70)
    print(f"  {'Model':<35} {'Clean':>7} {'PGD':>7} {'Retention':>10} {'Params':>10}")
    print("  " + "-"*68)
    for _, row in merged_ordered.iterrows():
        print(f"  {row['Model'].replace(chr(10),' '):<35} "
              f"{row['Accuracy']:>7.3f} "
              f"{row['PGD_Acc']:>7.3f} "
              f"{row['RetentionRatio']:>10.3f} "
              f"{int(row['NumParams']):>10,}")
    print("="*70)
    print("\n  Saved: quantum_advantage_benchmark.png")


if __name__ == "__main__":
    generate_plots()

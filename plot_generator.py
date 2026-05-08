"""
Publication-Grade Plot Generator
==================================
Handles two experiment types:
  1. Four-model fair benchmark  → quantum_advantage_benchmark.png
  2. Class-sweep  (Fig 5 style) → class_sweep_plot.png

Both are generated from their respective CSV files.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


# ── Label & colour mappings ───────────────────────────────────────────────────
LABEL_MAP = {
    "ClassicalCNN (Unconstrained)":        "ClassicalCNN\n(Unconstrained)",
    "FairCNN (Parameter-Matched to QCNN)": "FairCNN\n(Param-Matched)",
    "HybridQNN":                           "HybridQNN\n(Hybrid)",
    "MultiClassQCNN (Pure Quantum)":       "QCNN\n(Pure Quantum)",
}
COLORS = {
    "ClassicalCNN\n(Unconstrained)":  "#5B8DB8",
    "FairCNN\n(Param-Matched)":       "#7EB5D6",
    "HybridQNN\n(Hybrid)":            "#E8924A",
    "QCNN\n(Pure Quantum)":           "#E84A4A",
}


def _annotate_bars(ax, fmt=".3f", offset=9):
    for p in ax.patches:
        h = p.get_height()
        if h > 0.0001:
            ax.annotate(format(h, fmt),
                        (p.get_x() + p.get_width() / 2., h),
                        ha="center", va="bottom",
                        xytext=(0, offset), textcoords="offset points",
                        fontsize=9, fontweight="bold")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Four-model fair benchmark
# ─────────────────────────────────────────────────────────────────────────────
def generate_benchmark_plots(csv_path="results_comparison.csv"):
    if not os.path.exists(csv_path):
        print(f"[benchmark] {csv_path} not found – skipping."); return

    df = pd.read_csv(csv_path)
    df["Model"] = df["Model"].map(lambda x: LABEL_MAP.get(x, x))

    clean = df[df["Condition"] == "Clean"].copy()
    pgd   = df[df["Condition"] == "PGD_Adversarial"].copy()

    merged = clean[["Model", "Accuracy", "NumParams", "AccuracyPer1000Params"]].merge(
        pgd[["Model", "Accuracy"]].rename(columns={"Accuracy": "PGD_Acc"}), on="Model")
    merged["RetentionRatio"] = merged["PGD_Acc"] / merged["Accuracy"].replace(0, np.nan)

    order   = list(LABEL_MAP.values())
    palette = [COLORS.get(m, "#999") for m in order]

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Fair Quantum Advantage Benchmark — HMBD-v1 Arabic OCR (115 Classes)\n"
        "All models: identical AdamW · same LR schedule · same data",
        fontsize=14, fontweight="bold", y=1.01)

    # Panel 1 – Clean accuracy
    sns.barplot(data=clean, x="Model", y="Accuracy", order=order,
                palette=palette, ax=axes[0, 0])
    axes[0, 0].set_title("① Clean Validation Accuracy", fontweight="bold")
    axes[0, 0].set_ylim(0, 1.15); axes[0, 0].set_xlabel("")
    _annotate_bars(axes[0, 0])

    # Panel 2 – PGD adversarial accuracy
    sns.barplot(data=pgd, x="Model", y="Accuracy", order=order,
                palette=palette, ax=axes[0, 1])
    axes[0, 1].set_title("② Accuracy Under PGD Adversarial Attack", fontweight="bold")
    axes[0, 1].set_ylim(0, 1.15); axes[0, 1].set_xlabel("")
    _annotate_bars(axes[0, 1])

    # Panel 3 – Robustness retention ratio  ← KEY CHART
    mo = merged.set_index("Model").reindex(order).reset_index()
    bc = [COLORS.get(m, "#999") for m in mo["Model"]]
    bars = axes[1, 0].bar(mo["Model"], mo["RetentionRatio"],
                           color=bc, edgecolor="white")
    axes[1, 0].set_title(
        "③ Robustness Retention Ratio  (PGD / Clean)\n← KEY QUANTUM ADVANTAGE METRIC",
        fontweight="bold")
    axes[1, 0].set_ylim(0, 1.15); axes[1, 0].set_xlabel("")
    axes[1, 0].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    for bar, val in zip(bars, mo["RetentionRatio"]):
        if not np.isnan(val):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., val + 0.02,
                             f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")

    # Panel 4 – Parameter efficiency
    sns.barplot(data=clean, x="Model", y="AccuracyPer1000Params", order=order,
                palette=palette, ax=axes[1, 1])
    axes[1, 1].set_title("④ Parameter Efficiency (Clean Acc / 1k Params)", fontweight="bold")
    axes[1, 1].set_xlabel("")
    _annotate_bars(axes[1, 1], fmt=".4f")

    plt.tight_layout()
    out = "quantum_advantage_benchmark.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    # Terminal summary table
    print("\n" + "="*72)
    print("  BENCHMARK SUMMARY")
    print("="*72)
    print(f"  {'Model':<35} {'Clean':>7} {'PGD':>7} {'Retention':>10} {'Params':>10}")
    print("  " + "-"*70)
    for _, row in mo.iterrows():
        print(f"  {row['Model'].replace(chr(10),' '):<35} "
              f"{row['Accuracy']:>7.3f} {row['PGD_Acc']:>7.3f} "
              f"{row['RetentionRatio']:>10.3f} {int(row['NumParams']):>10,}")
    print("="*72)
    print(f"\n  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Class-sweep  (Fig 5 style from Mordacci et al.)
# ─────────────────────────────────────────────────────────────────────────────
def generate_sweep_plot(csv_path="class_sweep_results.csv"):
    if not os.path.exists(csv_path):
        print(f"[sweep] {csv_path} not found – skipping."); return

    df = pd.read_csv(csv_path)
    palette = {"FairCNN (Classical)": "#5B8DB8", "QCNN (Quantum)": "#E84A4A"}

    sns.set_theme(style="whitegrid", font_scale=1.2)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "QCNN vs Classical CNN — Accuracy Scaling with Number of Classes\n"
        "Arabic Handwritten Character Recognition  (HMBD-v1)\n"
        "Extends Mordacci et al. (2024) [arXiv:2404.12741] to 115 classes",
        fontsize=13, fontweight="bold")

    cnn_df  = df[df["Model"] == "FairCNN (Classical)"].sort_values("Classes")
    qcnn_df = df[df["Model"] == "QCNN (Quantum)"].sort_values("Classes")
    classes = cnn_df["Classes"].values

    # Panel 1 – Accuracy vs class count (line chart)
    ax1 = axes[0]
    for name, color in palette.items():
        sub = df[df["Model"] == name].sort_values("Classes")
        ax1.plot(sub["Classes"], sub["Accuracy"] * 100,
                 marker="o", linewidth=2.5, markersize=8, label=name, color=color)
        for _, row in sub.iterrows():
            ax1.annotate(f"{row['Accuracy']*100:.1f}%",
                         (row["Classes"], row["Accuracy"] * 100),
                         textcoords="offset points", xytext=(0, 10),
                         ha="center", fontsize=9, color=color, fontweight="bold")
    ax1.fill_between(classes, cnn_df["Accuracy"].values * 100, qcnn_df["Accuracy"].values * 100,
                     where=(qcnn_df["Accuracy"].values > cnn_df["Accuracy"].values),
                     alpha=0.12, color="#E84A4A")
    ax1.fill_between(classes, cnn_df["Accuracy"].values * 100, qcnn_df["Accuracy"].values * 100,
                     where=(cnn_df["Accuracy"].values >= qcnn_df["Accuracy"].values),
                     alpha=0.12, color="#5B8DB8")
    ax1.set_xlabel("Number of Classes"); ax1.set_ylabel("Validation Accuracy (%)")
    ax1.set_title("Accuracy vs Number of Classes\n(higher is better)", fontweight="bold")
    ax1.legend(); ax1.set_xticks(classes)

    # Panel 2 – Advantage gap bar chart
    ax2 = axes[1]
    gap = (qcnn_df["Accuracy"].values - cnn_df["Accuracy"].values) * 100
    bar_colors = ["#E84A4A" if g > 0 else "#5B8DB8" for g in gap]
    bars = ax2.bar([str(c) for c in classes], gap,
                   color=bar_colors, edgecolor="white")
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_xlabel("Number of Classes")
    ax2.set_ylabel("QCNN Advantage (percentage points)")
    ax2.set_title("Quantum Advantage Gap\n(+ve = QCNN wins, −ve = CNN wins)",
                  fontweight="bold")
    for bar, g in zip(bars, gap):
        ax2.text(bar.get_x() + bar.get_width()/2.,
                 g + (0.3 if g >= 0 else -0.8),
                 f"{g:+.1f}pp", ha="center", fontsize=10, fontweight="bold")

    if len(gap) > 0 and max(gap) > 0:
        mi = int(np.argmax(gap))
        ax2.annotate(f"Peak: {gap[mi]:+.1f}pp\nat {classes[mi]} classes",
                     xy=(str(classes[mi]), gap[mi]),
                     xytext=(0.6, 0.85), textcoords="axes fraction",
                     arrowprops=dict(arrowstyle="->"),
                     fontsize=10, color="#E84A4A", fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    plt.tight_layout()
    out = "class_sweep_plot.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Class sweep plot saved → {out}")


if __name__ == "__main__":
    generate_benchmark_plots()
    generate_sweep_plot()

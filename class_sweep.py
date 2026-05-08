"""
Class-Sweep Benchmark  (replicates Fig. 5 from Mordacci et al. 2024)
======================================================================
This is the KEY experiment for proving quantum advantage.

The paper shows on MNIST:
  4 classes  → CNN 90%  vs QCNN 85%  (CNN wins slightly)
  6 classes  → CNN 69%  vs QCNN 72%  (QCNN wins)
  8 classes  → CNN 50%  vs QCNN 70%  (QCNN wins by 20%)
 10 classes  → CNN 38%  vs QCNN 57%  (QCNN wins by 19%)

We extend this experiment to 115 Arabic classes (HMBD-v1 dataset).
The hypothesis: the advantage gap GROWS with class count, and at
115 classes the QCNN dominance should be decisive.

Run:
  python class_sweep.py

Output:
  class_sweep_results.csv   ← raw numbers
  class_sweep_plot.png      ← publication-ready Fig. 5 style chart
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from data_loader import HMBDDataLoader
from models import FairCNN, MultiClassQCNN, count_params

# ── Shared hyperparameters (IDENTICAL for every model / every class count) ──
EPOCHS       = 15      # paper uses 10; we use 15 for stability
LR           = 0.01    # paper uses 0.01 (matches paper exactly)
BATCH_SIZE   = 128
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0
LABEL_SMOOTH = 0.0     # no smoothing — match paper's raw cross-entropy

# Class counts to sweep (extends the paper's 4/6/8/10 up to 115)
CLASS_SWEEP  = [4, 8, 10, 28, 50, 115]
CSV_OUT      = "class_sweep_results.csv"
PLOT_OUT     = "class_sweep_plot.png"


def train_one(model, loader, device, epochs=EPOCHS):
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=LR * 0.01)
    model.train()
    for epoch in range(epochs):
        for x, y in tqdm(loader, desc=f"    ep {epoch+1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        scheduler.step()


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return correct / total


def run_sweep():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"\nClass Sweep — Replicating & Extending Mordacci et al. (2024) Fig. 5")
    print(f"Classes: {CLASS_SWEEP}")
    print(f"Hyperparams: Adam lr={LR}, {EPOCHS} epochs, batch={BATCH_SIZE}\n")

    records = []

    for n_cls in CLASS_SWEEP:
        print(f"\n{'='*60}")
        print(f"  {n_cls} Classes")
        print(f"{'='*60}")

        # ── Load data ───────────────────────────────────────────────────────
        loader = HMBDDataLoader(total_classes=n_cls, batch_size=BATCH_SIZE)
        loader.prepare_data()

        # ── FairCNN ─────────────────────────────────────────────────────────
        cnn = FairCNN(num_classes=n_cls)
        n_cnn = count_params(cnn)
        print(f"\n  FairCNN  ({n_cnn:,} params)")
        train_one(cnn, loader.train_loader, device)
        cnn_acc = evaluate(cnn, loader.val_loader, device)
        print(f"  → Val Acc: {cnn_acc:.4f}  ({cnn_acc*100:.1f}%)")
        records.append({"Classes": n_cls, "Model": "FairCNN (Classical)",
                        "Accuracy": cnn_acc, "Params": n_cnn})

        # ── MultiClassQCNN ───────────────────────────────────────────────────
        qcnn = MultiClassQCNN(num_classes=n_cls, num_layers=5)
        n_qcnn = count_params(qcnn)
        print(f"\n  QCNN     ({n_qcnn:,} params)")
        train_one(qcnn, loader.train_loader_qcnn, device)
        qcnn_acc = evaluate(qcnn, loader.val_loader_qcnn, device)
        print(f"  → Val Acc: {qcnn_acc:.4f}  ({qcnn_acc*100:.1f}%)")
        records.append({"Classes": n_cls, "Model": "QCNN (Quantum)",
                        "Accuracy": qcnn_acc, "Params": n_qcnn})

        gap = (qcnn_acc - cnn_acc) * 100
        sign = "QCNN wins" if gap > 0 else "CNN wins"
        print(f"\n  Gap: {gap:+.1f}%  ({sign})")

    # ── Save CSV ────────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    df.to_csv(CSV_OUT, index=False)
    print(f"\n  Results saved → {CSV_OUT}")

    # ── Plot ────────────────────────────────────────────────────────────────
    generate_sweep_plot(df)


def generate_sweep_plot(df=None):
    if df is None:
        df = pd.read_csv(CSV_OUT)

    sns.set_theme(style="whitegrid", font_scale=1.2)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    fig.suptitle(
        "QCNN vs Classical CNN — Accuracy Scaling with Number of Classes\n"
        "Arabic Handwritten Character Recognition  (HMBD-v1 Dataset)\n"
        "Replicates & extends Mordacci et al. (2024) to 115 classes",
        fontsize=13, fontweight="bold"
    )

    palette = {"FairCNN (Classical)": "#5B8DB8", "QCNN (Quantum)": "#E84A4A"}

    # ── Panel 1: Accuracy vs Class Count (line chart — mirrors paper Fig. 5) ─
    ax1 = axes[0]
    for model_name, color in palette.items():
        sub = df[df["Model"] == model_name].sort_values("Classes")
        ax1.plot(sub["Classes"], sub["Accuracy"] * 100,
                 marker="o", linewidth=2.5, markersize=8,
                 label=model_name, color=color)
        for _, row in sub.iterrows():
            ax1.annotate(f"{row['Accuracy']*100:.1f}%",
                         (row["Classes"], row["Accuracy"] * 100),
                         textcoords="offset points", xytext=(0, 10),
                         ha="center", fontsize=9, color=color, fontweight="bold")

    ax1.set_xlabel("Number of Classes", fontsize=12)
    ax1.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax1.set_title("Accuracy vs. Number of Classes\n(higher is better)", fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.set_xticks(df["Classes"].unique())

    # Add shaded region where QCNN wins
    cnn_vals  = df[df["Model"] == "FairCNN (Classical)"].sort_values("Classes")
    qcnn_vals = df[df["Model"] == "QCNN (Quantum)"].sort_values("Classes")
    classes   = cnn_vals["Classes"].values
    cnn_acc   = cnn_vals["Accuracy"].values * 100
    qcnn_acc  = qcnn_vals["Accuracy"].values * 100
    ax1.fill_between(classes, cnn_acc, qcnn_acc,
                     where=(qcnn_acc > cnn_acc),
                     alpha=0.15, color="#E84A4A", label="_Quantum advantage zone")
    ax1.fill_between(classes, cnn_acc, qcnn_acc,
                     where=(cnn_acc > qcnn_acc),
                     alpha=0.15, color="#5B8DB8", label="_Classical advantage zone")

    # ── Panel 2: Advantage Gap Bar Chart ─────────────────────────────────────
    ax2 = axes[1]
    gap_df = pd.merge(
        cnn_vals[["Classes", "Accuracy"]].rename(columns={"Accuracy": "CNN"}),
        qcnn_vals[["Classes", "Accuracy"]].rename(columns={"Accuracy": "QCNN"}),
        on="Classes"
    )
    gap_df["Gap"] = (gap_df["QCNN"] - gap_df["CNN"]) * 100
    bar_colors = ["#E84A4A" if g > 0 else "#5B8DB8" for g in gap_df["Gap"]]
    bars = ax2.bar(gap_df["Classes"].astype(str), gap_df["Gap"],
                   color=bar_colors, edgecolor="white", linewidth=0.8)
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_xlabel("Number of Classes", fontsize=12)
    ax2.set_ylabel("QCNN Accuracy Advantage (pp)", fontsize=12)
    ax2.set_title("Quantum Advantage Gap\n(+ve = QCNN wins)", fontweight="bold")
    for bar, gap in zip(bars, gap_df["Gap"]):
        ax2.text(bar.get_x() + bar.get_width() / 2.,
                 bar.get_height() + (0.3 if gap >= 0 else -0.8),
                 f"{gap:+.1f}pp", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")

    # Annotation box
    max_gap = gap_df["Gap"].max()
    max_cls = gap_df.loc[gap_df["Gap"].idxmax(), "Classes"]
    ax2.annotate(f"Peak advantage\n{max_gap:+.1f}pp at {max_cls} classes",
                 xy=(str(max_cls), max_gap),
                 xytext=(0.65, 0.85), textcoords="axes fraction",
                 arrowprops=dict(arrowstyle="->", color="black"),
                 fontsize=10, color="#E84A4A", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {PLOT_OUT}")


if __name__ == "__main__":
    run_sweep()

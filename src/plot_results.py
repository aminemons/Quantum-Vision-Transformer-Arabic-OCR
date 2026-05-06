import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


PALETTE = {
    "bg":       "#0d0d1a",
    "surface":  "#12122a",
    "fg":       "#e8e8ff",
    "cnn":      "#00d4ff",
    "qcnn":     "#ff6b9d",
    "accent":   "#ffd700",
    "grid":     "#1e1e3f",
    "green":    "#34d399",
    "purple":   "#a78bfa",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["surface"],
    "text.color":        PALETTE["fg"],
    "axes.labelcolor":   PALETTE["fg"],
    "xtick.color":       PALETTE["fg"],
    "ytick.color":       PALETTE["fg"],
    "axes.edgecolor":    PALETTE["grid"],
    "grid.color":        PALETTE["grid"],
    "grid.alpha":        0.4,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


def plot_adversarial_robustness(results, out_dir):
    eps = results["epsilon"]
    cnn_acc = results["CNN_acc"]
    qcnn_acc = results["QCNN_acc"]

    crossing_idx = None
    for i in range(1, len(eps)):
        if qcnn_acc[i] >= cnn_acc[i] and qcnn_acc[i - 1] < cnn_acc[i - 1]:
            crossing_idx = i

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.fill_between(eps, cnn_acc, alpha=0.08, color=PALETTE["cnn"])
    ax.fill_between(eps, qcnn_acc, alpha=0.08, color=PALETTE["qcnn"])

    ax.plot(eps, cnn_acc, "-o", color=PALETTE["cnn"], lw=2.5, ms=8,
            label=f"ResNet CNN  (clean: {cnn_acc[0]:.1%})", zorder=3)
    ax.plot(eps, qcnn_acc, "-s", color=PALETTE["qcnn"], lw=2.5, ms=8,
            label=f"Fourier-MERA QCNN  (clean: {qcnn_acc[0]:.1%})", zorder=3)

    if crossing_idx is not None:
        ax.axvline(eps[crossing_idx], color=PALETTE["accent"], lw=1.5,
                   ls="--", alpha=0.8, label=f"Quantum Advantage Threshold (epsilon={eps[crossing_idx]:.2f})")

    for i, e in enumerate(eps):
        delta = qcnn_acc[i] - cnn_acc[i]
        if delta > 0:
            ax.annotate(f"+{delta:.1%}", xy=(e, qcnn_acc[i]),
                        xytext=(0, 10), textcoords="offset points",
                        ha="center", color=PALETTE["green"], fontsize=8, fontweight="bold")

    ax.set_xlabel("PGD Adversarial Perturbation Epsilon", fontsize=13)
    ax.set_ylabel("Test Accuracy", fontsize=13)
    ax.set_title("Adversarial Robustness: Fourier-MERA QCNN vs. ResNet CNN\n"
                 "PGD Attack (20 steps) on Othmanic Arabic Dataset",
                 fontsize=14, color=PALETTE["accent"], pad=15)
    ax.set_ylim(0, 1.05)
    ax.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"], fontsize=11)
    ax.grid(True)

    plt.tight_layout()
    path = os.path.join(out_dir, "adversarial_robustness.png")
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"[plot] Saved: {path}")


def plot_lipschitz_comparison(results, out_dir):
    eps = results["epsilon"]
    cnn_lip = results["CNN_lipschitz"]
    qcnn_lip = results["QCNN_lipschitz"]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(eps, cnn_lip, "-o", color=PALETTE["cnn"], lw=2.5, ms=8, label="ResNet CNN")
    ax.plot(eps, qcnn_lip, "-s", color=PALETTE["qcnn"], lw=2.5, ms=8, label="Fourier-MERA QCNN")

    ax.set_xlabel("Epsilon", fontsize=13)
    ax.set_ylabel("Estimated Lipschitz Constant", fontsize=13)
    ax.set_title("Lipschitz Constant: Gradient Boundedness Under Attack",
                 fontsize=14, color=PALETTE["accent"], pad=15)
    ax.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"], fontsize=11)
    ax.grid(True)

    plt.tight_layout()
    path = os.path.join(out_dir, "lipschitz_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"[plot] Saved: {path}")


def plot_training_curves(cnn_history, qcnn_history, out_dir, mode="clean"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, key, ylabel in zip(
        axes,
        ["val_acc", "train_loss"],
        ["Validation Accuracy", "Training Loss"]
    ):
        cnn_vals = cnn_history[key]
        qcnn_vals = qcnn_history[key]
        epochs = range(1, len(cnn_vals) + 1)

        ax.plot(epochs, cnn_vals, "-", color=PALETTE["cnn"], lw=2, label="ResNet CNN")
        ax.plot(epochs, qcnn_vals, "-", color=PALETTE["qcnn"], lw=2, label="Fourier-MERA QCNN")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{ylabel} — {mode.capitalize()} Mode", fontsize=13, color=PALETTE["accent"])
        ax.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"])
        ax.grid(True)

    fig.suptitle("Training Dynamics: ResNet CNN vs Fourier-MERA QCNN",
                 fontsize=15, color=PALETTE["fg"], y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, f"training_curves_{mode}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"[plot] Saved: {path}")


def plot_mode_comparison(mode_results, out_dir):
    modes = list(mode_results.keys())
    cnn_clean = [mode_results[m]["CNN_clean"] for m in modes]
    qcnn_clean = [mode_results[m]["QCNN_clean"] for m in modes]
    cnn_adv = [mode_results[m]["CNN_adv_max"] for m in modes]
    qcnn_adv = [mode_results[m]["QCNN_adv_max"] for m in modes]

    x = np.arange(len(modes))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 7))

    b1 = ax.bar(x - 1.5 * width, cnn_clean,  width, color=PALETTE["cnn"],   alpha=0.9, label="CNN Clean")
    b2 = ax.bar(x - 0.5 * width, qcnn_clean, width, color=PALETTE["qcnn"],  alpha=0.9, label="QCNN Clean")
    b3 = ax.bar(x + 0.5 * width, cnn_adv,    width, color=PALETTE["cnn"],   alpha=0.5, label="CNN Adv (eps=0.3)", hatch="//")
    b4 = ax.bar(x + 1.5 * width, qcnn_adv,   width, color=PALETTE["qcnn"],  alpha=0.5, label="QCNN Adv (eps=0.3)", hatch="//")

    for bars in [b1, b2, b3, b4]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.1%}", ha="center", va="bottom",
                    fontsize=8, color=PALETTE["accent"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in modes], fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=13)
    ax.set_ylim(0, 1.15)
    ax.set_title("Dataset Mode Comparison: Clean vs. Adversarial Accuracy",
                 fontsize=14, color=PALETTE["accent"], pad=15)
    ax.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"], fontsize=10, ncol=2)
    ax.grid(True, axis="y")

    plt.tight_layout()
    path = os.path.join(out_dir, "mode_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"[plot] Saved: {path}")


def plot_accuracy_degradation_heatmap(all_results, out_dir):
    modes = list(all_results.keys())
    epsilons = all_results[modes[0]]["epsilon"]

    cnn_matrix = np.array([all_results[m]["CNN_acc"] for m in modes])
    qcnn_matrix = np.array([all_results[m]["QCNN_acc"] for m in modes])
    delta_matrix = qcnn_matrix - cnn_matrix

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    eps_labels = [f"{e:.2f}" for e in epsilons]

    for ax, matrix, title, cmap in zip(
        axes,
        [cnn_matrix, qcnn_matrix, delta_matrix],
        ["ResNet CNN Accuracy", "Fourier-MERA QCNN Accuracy", "QCNN - CNN Delta (Quantum Advantage)"],
        ["Blues", "RdPu", "RdYlGn"]
    ):
        sns.heatmap(matrix, ax=ax, annot=True, fmt=".2f",
                    xticklabels=eps_labels,
                    yticklabels=[m.capitalize() for m in modes],
                    cmap=cmap, vmin=0.0 if "Delta" not in title else -0.5,
                    vmax=1.0 if "Delta" not in title else 0.5,
                    linewidths=0.5)
        ax.set_title(title, fontsize=12, color=PALETTE["accent"])
        ax.set_xlabel("Epsilon", fontsize=10)
        ax.set_ylabel("Dataset Mode", fontsize=10)

    plt.suptitle("Accuracy Degradation Heatmap: Dataset Mode x Adversarial Epsilon",
                 fontsize=14, color=PALETTE["fg"], y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "accuracy_degradation_heatmap.png")
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"[plot] Saved: {path}")


def generate_all_plots(all_results, cnn_histories, qcnn_histories, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    first_mode = list(all_results.keys())[0]
    plot_adversarial_robustness(all_results[first_mode], out_dir)
    plot_lipschitz_comparison(all_results[first_mode], out_dir)

    mode_summary = {}
    for mode, res in all_results.items():
        mode_summary[mode] = {
            "CNN_clean":   res["CNN_acc"][0],
            "QCNN_clean":  res["QCNN_acc"][0],
            "CNN_adv_max": res["CNN_acc"][-1],
            "QCNN_adv_max": res["QCNN_acc"][-1],
        }

    plot_mode_comparison(mode_summary, out_dir)
    plot_accuracy_degradation_heatmap(all_results, out_dir)

    for mode in cnn_histories:
        plot_training_curves(cnn_histories[mode], qcnn_histories[mode], out_dir, mode=mode)

    print(f"\n[plot] All figures saved to: {out_dir}")


if __name__ == "__main__":
    results_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "adversarial_results", "all_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            data = json.load(f)
        out = os.path.dirname(results_path)
        plot_adversarial_robustness(data.get("clean", data), out)
        plot_lipschitz_comparison(data.get("clean", data), out)
    else:
        print(f"No results found at {results_path}. Run run_benchmark.py first.")

"""
Judge-Ready Visualizations for Hybrid QCNN-ViT.

Generates publication-quality figures for hackathon presentation:
1. Confusion Matrix -- 28x28 heatmap with Arabic character labels
2. Training Curves -- dual-axis loss & accuracy over epochs
3. Quantum Attention Maps -- overlay on original character images
4. Quantum Circuit Diagrams -- visual representation of QCNN and attention circuits
5. Per-Class Performance -- bar chart of accuracy per Arabic character

All figures use a consistent dark theme with quantum-inspired color palette.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from data_loader import ARABIC_CHARS, ARABIC_UNICODE, NUM_CLASSES

# ----------------------------------------------
# Global Style Configuration
# ----------------------------------------------

# Quantum-inspired dark theme
QUANTUM_COLORS = {
    'bg': '#0a0a1a',
    'fg': '#e0e0ff',
    'accent_1': '#00d4ff',  # Cyan
    'accent_2': '#ff6b9d',  # Pink
    'accent_3': '#c084fc',  # Purple
    'accent_4': '#34d399',  # Green
    'grid': '#1a1a3e',
    'gold': '#ffd700',
}

plt.rcParams.update({
    'figure.facecolor': QUANTUM_COLORS['bg'],
    'axes.facecolor': QUANTUM_COLORS['bg'],
    'text.color': QUANTUM_COLORS['fg'],
    'axes.labelcolor': QUANTUM_COLORS['fg'],
    'xtick.color': QUANTUM_COLORS['fg'],
    'ytick.color': QUANTUM_COLORS['fg'],
    'axes.edgecolor': QUANTUM_COLORS['grid'],
    'grid.color': QUANTUM_COLORS['grid'],
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10,
})

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')


# ----------------------------------------------
# 1. Confusion Matrix
# ----------------------------------------------

def plot_confusion_matrix(save_dir=None):
    """
    Generate a 28x28 confusion matrix heatmap.

    Uses saved test predictions and labels to create a normalized
    confusion matrix with Arabic character class labels.
    """
    if save_dir is None:
        save_dir = RESULTS_DIR

    pred_path = os.path.join(save_dir, 'test_predictions.npy')
    label_path = os.path.join(save_dir, 'test_labels.npy')

    if not os.path.exists(pred_path):
        print("[!] Test predictions not found. Run training first.")
        return

    logits = np.load(pred_path)
    labels = np.load(label_path)
    preds = logits.argmax(axis=-1)

    # Compute confusion matrix
    n_classes = min(NUM_CLASSES, len(np.unique(labels)))
    cm = confusion_matrix(labels, preds, labels=range(n_classes))

    # Normalize
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 14))

    # Custom colormap: dark blue -> cyan -> white
    cmap = sns.color_palette("mako", as_cmap=True)

    sns.heatmap(
        cm_normalized, annot=False, fmt='.1f', cmap=cmap,
        xticklabels=ARABIC_UNICODE[:n_classes],
        yticklabels=ARABIC_UNICODE[:n_classes],
        linewidths=0.5, linecolor=QUANTUM_COLORS['grid'],
        ax=ax, vmin=0, vmax=1,
        cbar_kws={'label': 'Classification Rate', 'shrink': 0.8}
    )

    ax.set_xlabel('Predicted Character', fontsize=14, fontweight='bold',
                  color=QUANTUM_COLORS['accent_1'])
    ax.set_ylabel('True Character', fontsize=14, fontweight='bold',
                  color=QUANTUM_COLORS['accent_1'])
    ax.set_title('Quantum Vision Transformer -- Arabic Character Confusion Matrix',
                 fontsize=16, fontweight='bold', color=QUANTUM_COLORS['gold'],
                 pad=20)

    # Overall accuracy annotation
    accuracy = (preds == labels).mean()
    ax.annotate(f'Overall Accuracy: {accuracy:.1%}',
                xy=(0.5, -0.08), xycoords='axes fraction',
                ha='center', fontsize=13, fontweight='bold',
                color=QUANTUM_COLORS['accent_4'])

    plt.tight_layout()
    path = os.path.join(save_dir, 'confusion_matrix.png')
    fig.savefig(path, dpi=300, bbox_inches='tight',
                facecolor=QUANTUM_COLORS['bg'])
    plt.close()
    print(f"[OK] Confusion matrix saved to {path}")


# ----------------------------------------------
# 2. Training Curves
# ----------------------------------------------

def plot_training_curves(save_dir=None):
    """
    Plot dual-axis training and validation curves (loss & accuracy).
    """
    if save_dir is None:
        save_dir = RESULTS_DIR

    history_path = os.path.join(save_dir, 'training_history.json')
    if not os.path.exists(history_path):
        print("[!] Training history not found. Run training first.")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # -- Loss Plot --
    ax1.plot(epochs, history['train_loss'], '-o',
             color=QUANTUM_COLORS['accent_1'], label='Train Loss',
             linewidth=2, markersize=6, alpha=0.9)
    ax1.plot(epochs, history['val_loss'], '-s',
             color=QUANTUM_COLORS['accent_2'], label='Val Loss',
             linewidth=2, markersize=6, alpha=0.9)

    ax1.fill_between(epochs, history['train_loss'],
                     alpha=0.1, color=QUANTUM_COLORS['accent_1'])
    ax1.fill_between(epochs, history['val_loss'],
                     alpha=0.1, color=QUANTUM_COLORS['accent_2'])

    ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=13, fontweight='bold')
    ax1.set_title('Training Loss Convergence', fontsize=14,
                  fontweight='bold', color=QUANTUM_COLORS['gold'])
    ax1.legend(fontsize=11, facecolor=QUANTUM_COLORS['bg'],
               edgecolor=QUANTUM_COLORS['grid'], labelcolor=QUANTUM_COLORS['fg'])
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(epochs[0], epochs[-1])

    # -- Accuracy Plot --
    ax2.plot(epochs, history['train_acc'], '-o',
             color=QUANTUM_COLORS['accent_4'], label='Train Accuracy',
             linewidth=2, markersize=6, alpha=0.9)
    ax2.plot(epochs, history['val_acc'], '-s',
             color=QUANTUM_COLORS['accent_3'], label='Val Accuracy',
             linewidth=2, markersize=6, alpha=0.9)

    ax2.fill_between(epochs, history['train_acc'],
                     alpha=0.1, color=QUANTUM_COLORS['accent_4'])
    ax2.fill_between(epochs, history['val_acc'],
                     alpha=0.1, color=QUANTUM_COLORS['accent_3'])

    # Mark best epoch
    best_epoch = history.get('best_epoch', 1)
    best_val_acc = history.get('best_val_acc', 0)
    ax2.axvline(x=best_epoch, color=QUANTUM_COLORS['gold'],
                linestyle='--', alpha=0.7, label=f'Best (epoch {best_epoch})')
    ax2.scatter([best_epoch], [best_val_acc], s=150, c=QUANTUM_COLORS['gold'],
                zorder=5, edgecolors='white', linewidth=2, marker='*')

    ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax2.set_title('Classification Accuracy', fontsize=14,
                  fontweight='bold', color=QUANTUM_COLORS['gold'])
    ax2.legend(fontsize=11, facecolor=QUANTUM_COLORS['bg'],
               edgecolor=QUANTUM_COLORS['grid'], labelcolor=QUANTUM_COLORS['fg'])
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(epochs[0], epochs[-1])
    ax2.set_ylim(0, 1.05)

    # Suptitle
    fig.suptitle('Hybrid QCNN-ViT Training Dashboard',
                 fontsize=18, fontweight='bold', y=1.02,
                 color=QUANTUM_COLORS['fg'])

    plt.tight_layout()
    path = os.path.join(save_dir, 'training_curves.png')
    fig.savefig(path, dpi=300, bbox_inches='tight',
                facecolor=QUANTUM_COLORS['bg'])
    plt.close()
    print(f"[OK] Training curves saved to {path}")


# ----------------------------------------------
# 3. Quantum Attention Maps
# ----------------------------------------------

def plot_attention_maps(model=None, data_loader=None, save_dir=None,
                        n_samples: int = 8):
    """
    Overlay quantum attention weights onto original character images.

    Shows which patches the quantum self-attention mechanism focuses on,
    visualized as a heatmap overlay on the original 8x8 images.
    """
    if save_dir is None:
        save_dir = RESULTS_DIR

    if model is None or data_loader is None:
        print("[!] Model and data_loader required for attention maps.")
        print("    Run this after training with the model and loader objects.")
        return

    model.eval()

    # Get a batch of samples
    images, labels = next(iter(data_loader))
    n_samples = min(n_samples, images.shape[0])

    with torch.no_grad():
        _ = model(images[:n_samples])
        attn_weights = model.get_attention_weights()[:n_samples]

    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 3, 7))

    for i in range(n_samples):
        img = images[i].numpy().reshape(8, 8)
        attn = attn_weights[i].numpy()  # (4, 4) attention matrix

        # Original image
        axes[0, i].imshow(img, cmap='gray', interpolation='nearest')
        axes[0, i].set_title(f'{ARABIC_UNICODE[labels[i]]}',
                             fontsize=14, color=QUANTUM_COLORS['accent_1'])
        axes[0, i].axis('off')

        # Attention overlay
        # Convert 4x4 attention to 2x2 patch attention (row sums -> patch importance)
        patch_importance = attn.mean(axis=0)  # Average attention each patch receives
        # Map to 2x2 grid, then upscale to 8x8
        attn_map = patch_importance.reshape(2, 2)
        attn_map = np.repeat(np.repeat(attn_map, 4, axis=0), 4, axis=1)

        # Overlay
        axes[1, i].imshow(img, cmap='gray', interpolation='nearest', alpha=0.5)
        overlay = axes[1, i].imshow(attn_map, cmap='magma', alpha=0.6,
                                     interpolation='bilinear')
        axes[1, i].set_title('Attention', fontsize=11,
                             color=QUANTUM_COLORS['accent_2'])
        axes[1, i].axis('off')

    # Row labels
    axes[0, 0].set_ylabel('Original', fontsize=13, fontweight='bold',
                          color=QUANTUM_COLORS['fg'])
    axes[1, 0].set_ylabel('Q-Attention', fontsize=13, fontweight='bold',
                          color=QUANTUM_COLORS['fg'])

    fig.suptitle('Quantum Self-Attention Maps -- Arabic Character Focus Regions',
                 fontsize=16, fontweight='bold', y=1.02,
                 color=QUANTUM_COLORS['gold'])

    plt.tight_layout()
    path = os.path.join(save_dir, 'attention_maps.png')
    fig.savefig(path, dpi=300, bbox_inches='tight',
                facecolor=QUANTUM_COLORS['bg'])
    plt.close()
    print(f"[OK] Attention maps saved to {path}")


# ----------------------------------------------
# 4. Quantum Circuit Diagrams
# ----------------------------------------------

def plot_circuit_diagrams(save_dir=None):
    """
    Render the QCNN and attention quantum circuits using PennyLane's
    built-in drawing capabilities.
    """
    if save_dir is None:
        save_dir = RESULTS_DIR

    try:
        import pennylane as qml
        from qcnn_layer import qcnn_circuit, N_QUBITS_CONV
        from quantum_attention import quantum_attention_circuit, N_QUBITS_ATTN

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

        # QCNN Circuit
        dummy_inputs = torch.zeros(4)
        dummy_conv0 = torch.zeros(4, 3)
        dummy_conv1 = torch.zeros(4, 3)
        dummy_pool = torch.zeros(2, 2)

        qml.draw_mpl(qcnn_circuit, style="sketch")(
            dummy_inputs, dummy_conv0, dummy_conv1, dummy_pool
        )
        fig_qcnn = plt.gcf()
        fig_qcnn.suptitle('QCNN Convolutional Circuit (per patch)',
                          fontsize=14, fontweight='bold')
        path_qcnn = os.path.join(save_dir, 'circuit_qcnn.png')
        fig_qcnn.savefig(path_qcnn, dpi=300, bbox_inches='tight',
                         facecolor='white')
        plt.close()
        print(f"[OK] QCNN circuit diagram saved to {path_qcnn}")

        # Attention Circuit
        dummy_q = torch.zeros(2)
        dummy_k = torch.zeros(2)
        dummy_aw0 = torch.zeros(4, 3)
        dummy_aw1 = torch.zeros(4, 3)

        qml.draw_mpl(quantum_attention_circuit, style="sketch")(
            dummy_q, dummy_k, dummy_aw0, dummy_aw1
        )
        fig_attn = plt.gcf()
        fig_attn.suptitle('Quantum Self-Attention Circuit',
                          fontsize=14, fontweight='bold')
        path_attn = os.path.join(save_dir, 'circuit_attention.png')
        fig_attn.savefig(path_attn, dpi=300, bbox_inches='tight',
                         facecolor='white')
        plt.close()
        print(f"[OK] Attention circuit diagram saved to {path_attn}")

    except Exception as e:
        print(f"[!] Circuit diagram generation failed: {e}")
        print("    This is non-critical -- other visualizations still work.")


# ----------------------------------------------
# 5. Per-Class Performance
# ----------------------------------------------

def plot_per_class_accuracy(save_dir=None):
    """
    Bar chart showing per-class accuracy with Arabic character labels.
    """
    if save_dir is None:
        save_dir = RESULTS_DIR

    history_path = os.path.join(save_dir, 'training_history.json')
    if not os.path.exists(history_path):
        print("[!] Training history not found.")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    per_class = history.get('per_class_acc', {})
    if not per_class:
        print("[!] Per-class accuracy not found in history.")
        return

    # Sort by accuracy
    chars = list(per_class.keys())
    accs = [per_class[c] for c in chars]

    # Map to unicode for display
    char_to_unicode = dict(zip(ARABIC_CHARS, ARABIC_UNICODE))
    display_labels = [char_to_unicode.get(c, c) for c in chars]

    # Sort by accuracy
    sorted_idx = np.argsort(accs)
    chars_sorted = [display_labels[i] for i in sorted_idx]
    accs_sorted = [accs[i] for i in sorted_idx]

    # Color gradient based on accuracy
    colors = plt.cm.RdYlGn(np.array(accs_sorted))

    fig, ax = plt.subplots(figsize=(14, 8))

    bars = ax.barh(range(len(chars_sorted)), accs_sorted, color=colors,
                   edgecolor=QUANTUM_COLORS['grid'], linewidth=0.5)

    ax.set_yticks(range(len(chars_sorted)))
    ax.set_yticklabels(chars_sorted, fontsize=12)
    ax.set_xlabel('Accuracy', fontsize=13, fontweight='bold')
    ax.set_title('Per-Character Classification Accuracy',
                 fontsize=16, fontweight='bold', color=QUANTUM_COLORS['gold'])
    ax.set_xlim(0, 1.05)
    ax.grid(True, axis='x', alpha=0.3)

    # Add accuracy values on bars
    for bar, acc in zip(bars, accs_sorted):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{acc:.1%}', va='center', fontsize=9,
                color=QUANTUM_COLORS['fg'])

    # Mean accuracy line
    mean_acc = np.mean(accs_sorted)
    ax.axvline(x=mean_acc, color=QUANTUM_COLORS['accent_1'],
               linestyle='--', linewidth=2, alpha=0.8,
               label=f'Mean: {mean_acc:.1%}')
    ax.legend(fontsize=12, facecolor=QUANTUM_COLORS['bg'],
              edgecolor=QUANTUM_COLORS['grid'], labelcolor=QUANTUM_COLORS['fg'])

    plt.tight_layout()
    path = os.path.join(save_dir, 'per_class_accuracy.png')
    fig.savefig(path, dpi=300, bbox_inches='tight',
                facecolor=QUANTUM_COLORS['bg'])
    plt.close()
    print(f"[OK] Per-class accuracy saved to {path}")


# ----------------------------------------------
# Master Visualization Function
# ----------------------------------------------

def generate_all_visualizations(model=None, data_loader=None, save_dir=None):
    """Generate all visualizations from saved training results."""
    if save_dir is None:
        save_dir = RESULTS_DIR

    print("+" + "=" * 58 + "+")
    print("|  Generating Judge-Ready Visualizations                   |")
    print("+" + "=" * 58 + "+")

    plot_training_curves(save_dir)
    plot_confusion_matrix(save_dir)
    plot_per_class_accuracy(save_dir)
    plot_circuit_diagrams(save_dir)

    if model is not None and data_loader is not None:
        plot_attention_maps(model, data_loader, save_dir)

    print(f"\n[OK] All visualizations generated in {save_dir}/")


# ----------------------------------------------
# Entry Point
# ----------------------------------------------

if __name__ == "__main__":
    generate_all_visualizations()

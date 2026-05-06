"""
Exhaustive Benchmarking Suite for Quantum vs Classical OCR
Runs a grid search over:
- Models: Classical CNN (Heavy), Classical CNN (Light), Hybrid QViT
- Few-Shot Regimes: 1.0 (Full), 0.1 (10%), 0.05 (5%), 0.01 (1%)
- Noise Levels: 0.0, 0.1, 0.2, 0.3

Saves results to benchmark_results.json and generates plots.
"""

import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from data_loader import load_data

# Import existing models from train_workstation
from train_workstation import ClassicalCNN as ClassicalCNN_Heavy
from train_workstation import HybridCNNQViT, train_model

# ─── Config ───────────────────────────────────────────────
IMG_SIZE    = 32
BATCH_SIZE  = 256
EPOCHS      = 30
LR          = 0.002
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR    = os.path.join(os.path.dirname(__file__), "benchmark_results")
os.makedirs(SAVE_DIR, exist_ok=True)

# ─── Lightweight CNN (Parameter-Matched to QViT) ───────────
class ClassicalCNN_Light(nn.Module):
    """
    A lighter CNN designed to have roughly the same parameter count 
    (~490k) as the Hybrid QViT, to ensure a fair comparison.
    """
    def __init__(self, img_size=32, n_classes=28):
        super().__init__()
        self.img_size = img_size
        self.features = nn.Sequential(
            nn.Conv2d(1, 32,  3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),  # 16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),  # 8
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),  # 4
        )
        flat = 64 * (img_size // 8) ** 2
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128),  nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )
    def forward(self, x):
        return self.head(self.features(x.view(-1, 1, self.img_size, self.img_size)))

# ─── Orchestrator ──────────────────────────────────────────
def run_benchmark():
    models_to_test = {
        'CNN_Heavy': ClassicalCNN_Heavy,
        'CNN_Light': ClassicalCNN_Light,
        'Hybrid_QViT': HybridCNNQViT
    }
    
    few_shot_levels = [1.0, 0.1, 0.05, 0.01]
    noise_levels = [0.0, 0.1, 0.2, 0.3]
    
    # Store results: results[few_shot][noise][model] = test_acc
    results = {}
    
    total_runs = len(few_shot_levels) * len(noise_levels) * len(models_to_test)
    current_run = 0
    
    for fs in few_shot_levels:
        results[str(fs)] = {}
        for nl in noise_levels:
            results[str(fs)][str(nl)] = {}
            
            print(f"\n{'#'*60}")
            print(f" EXPERIMENT REGIME: Few-Shot = {fs*100}% | Noise = {nl}")
            print(f"{'#'*60}")
            
            # 1. Load Data for this specific regime
            train_loader, val_loader, test_loader, class_names = load_data(
                batch_size=BATCH_SIZE, img_size=IMG_SIZE,
                few_shot_ratio=fs, noise_level=nl, use_tashkeel=False
            )
            n_classes = len(class_names)
            
            # Helper to temporarily override globals in train_workstation since it uses them
            import train_workstation
            train_workstation.train_loader = train_loader
            train_workstation.val_loader = val_loader
            train_workstation.test_loader = test_loader
            
            # 2. Train each model
            for model_name, ModelClass in models_to_test.items():
                current_run += 1
                print(f"\n>>> [Run {current_run}/{total_runs}] Training {model_name}...")
                
                model = ModelClass(img_size=IMG_SIZE, n_classes=n_classes)
                # We train it
                h = train_model(model, f"{model_name}_fs{fs}_nl{nl}", epochs=EPOCHS, lr=LR)
                
                # Evaluate on test set
                model.load_state_dict(torch.load(f"{train_workstation.SAVE_DIR}/best_{model_name}_fs{fs}_nl{nl}.pt", weights_only=True))
                model.eval().to(DEVICE)
                tot_c = tot_n = 0
                with torch.no_grad():
                    for imgs, lbls in test_loader:
                        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                        tot_c += (model(imgs).argmax(1)==lbls).sum().item()
                        tot_n += len(imgs)
                test_acc = tot_c / tot_n
                print(f"  --> {model_name} Final Test Acc: {test_acc:.4f}")
                
                results[str(fs)][str(nl)][model_name] = test_acc
                
                # Save partial results in case of crash
                with open(os.path.join(SAVE_DIR, 'partial_benchmark.json'), 'w') as f:
                    json.dump(results, f, indent=2)

    # Save final
    with open(os.path.join(SAVE_DIR, 'final_benchmark.json'), 'w') as f:
        json.dump(results, f, indent=2)
        
    print("\n[OK] Benchmarking Complete! Generating Matrix Plot...")
    generate_matrix_plot(results, few_shot_levels, noise_levels)

def generate_matrix_plot(results, few_shot_levels, noise_levels):
    C = {'bg':'#0a0a1a','fg':'#e0e0ff','cnn_h':'#00d4ff','cnn_l':'#a78bfa','qvit':'#ff6b9d'}
    plt.rcParams.update({'figure.facecolor':C['bg'],'axes.facecolor':C['bg'],'text.color':C['fg'],
        'axes.labelcolor':C['fg'],'xtick.color':C['fg'],'ytick.color':C['fg']})

    fig, axes = plt.subplots(len(few_shot_levels), 1, figsize=(10, 4*len(few_shot_levels)))
    if len(few_shot_levels) == 1: axes = [axes]
    
    for idx, fs in enumerate(few_shot_levels):
        ax = axes[idx]
        nl_str = [str(nl) for nl in noise_levels]
        
        acc_cnn_h = [results[str(fs)][nl]['CNN_Heavy'] for nl in nl_str]
        acc_cnn_l = [results[str(fs)][nl]['CNN_Light'] for nl in nl_str]
        acc_qvit  = [results[str(fs)][nl]['Hybrid_QViT'] for nl in nl_str]
        
        ax.plot(nl_str, acc_cnn_h, '-o', color=C['cnn_h'], lw=2, label='CNN Heavy (~650k)')
        ax.plot(nl_str, acc_cnn_l, '-^', color=C['cnn_l'], lw=2, label='CNN Light (~490k)')
        ax.plot(nl_str, acc_qvit,  '-s', color=C['qvit'],  lw=2, label='Hybrid QViT (~490k)')
        
        ax.set_title(f"Data Fraction: {fs*100}%", fontsize=14, color='#ffd700')
        ax.set_xlabel("Noise Level")
        ax.set_ylabel("Test Accuracy")
        ax.grid(alpha=0.2)
        if idx == 0:
            ax.legend(facecolor=C['bg'], edgecolor='#1a1a3e')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'benchmark_matrix.png'), dpi=300, bbox_inches='tight')
    print(f"Saved plot to {SAVE_DIR}/benchmark_matrix.png")

if __name__ == "__main__":
    run_benchmark()

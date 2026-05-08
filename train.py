"""
Fair Quantum Advantage Benchmark
=================================
Scientific Claim: Under IDENTICAL training conditions, quantum circuits exhibit
superior adversarial robustness due to their inherent unitary structure (bounded
Lipschitz constant), even when classical models are fully optimized.

Experiment Design:
  - ALL models: same optimizer (AdamW), same LR schedule, same epochs, same data
  - Key metric: Robustness Retention Ratio = PGD_acc / Clean_acc
  - FairCNN is parameter-matched to QCNN for the honest comparison
  - ClassicalCNN (large) shows the absolute ceiling of classical approach
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data_loader import HMBDDataLoader
from models import ClassicalCNN, FairCNN, IsoCNN, HybridQNN, MultiClassQCNN, count_params
from eval import Evaluator

# ── Unified training hyperparameters (SAME FOR ALL MODELS) ──────────────────
EPOCHS        = 30
LR            = 0.001
BATCH_SIZE    = 128
WEIGHT_DECAY  = 1e-4
LABEL_SMOOTH  = 0.1
GRAD_CLIP     = 1.0
NUM_CLASSES   = 115
# ─────────────────────────────────────────────────────────────────────────────


def train_model(model, dataloader, device, lr=None):
    """Single unified training function. Identical for every model."""
    lr = lr or LR
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=lr * 0.01)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"  Epoch {epoch+1}/{EPOCHS}")
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        avg = total_loss / len(dataloader.dataset)
        print(f"  Epoch {epoch+1}/{EPOCHS} - Loss: {avg:.4f}")


def run_benchmarks():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"\n{'='*60}")
    print(f"  Fair Quantum Advantage Benchmark — {NUM_CLASSES} Classes")
    print(f"  ALL models: AdamW lr={LR}, {EPOCHS} epochs, batch={BATCH_SIZE}")
    print(f"{'='*60}")

    # ── Prepare data (two sets: pixel-space + L2-normalized for QCNN) ────────
    loader = HMBDDataLoader(total_classes=NUM_CLASSES, batch_size=BATCH_SIZE)
    loader.prepare_data()
    evaluator = Evaluator()

    # ── Define the 4 models and which data loader each uses ──────────────────
    # The QCNN uses L2-normalized loaders (AmplitudeEmbedding requires unit norm)
    # All others use raw [0,1] pixel loaders
    benchmarks = [
        {
            "name": "ClassicalCNN (Unconstrained)",
            "model": ClassicalCNN(NUM_CLASSES),
            "train_loader": loader.train_loader,
            "val_loader":   loader.val_loader,
            "stress_loader":loader.stress_test_loader,
            "note": "Best-possible classical model. Shows clean-accuracy ceiling."
        },
        {
            "name": "FairCNN (Parameter-Matched to QCNN)",
            "model": FairCNN(NUM_CLASSES),
            "train_loader": loader.train_loader,
            "val_loader":   loader.val_loader,
            "stress_loader":loader.stress_test_loader,
            "note": "Classical model with same parameter budget as QCNN. Honest comparison."
        },
        {
            "name": "IsoCNN (Iso-Parameter to QCNN)",
            "model": IsoCNN(NUM_CLASSES),
            "train_loader": loader.train_loader,
            "val_loader":   loader.val_loader,
            "stress_loader":loader.stress_test_loader,
            "note": "Same total params as QCNN, same readout. Definitive comparison."
        },
        {
            "name": "HybridQNN",
            "model": HybridQNN(NUM_CLASSES),
            "train_loader": loader.train_loader,
            "val_loader":   loader.val_loader,
            "stress_loader":loader.stress_test_loader,
            "note": "Quantum spatial encoder + classical head. Partial quantum advantage."
        },
        {
            "name": "MultiClassQCNN (Pure Quantum)",
            "model": MultiClassQCNN(NUM_CLASSES),
            "train_loader": loader.train_loader_qcnn,
            "val_loader":   loader.val_loader_qcnn,
            "stress_loader":loader.stress_test_loader_qcnn,
            "lr":           0.01,
            "note": "Full amplitude embedding, LR=0.01 (paper). Hypothesis: highest retention."
        },
    ]

    print(f"\n{'─'*60}")
    print(f"  Parameter Counts:")
    for b in benchmarks:
        n = count_params(b["model"])
        print(f"  {b['name']:<42} {n:>8,} params")
    print(f"{'─'*60}\n")

    # ── Train and evaluate each model ─────────────────────────────────────────
    for b in benchmarks:
        name   = b["name"]
        model  = b["model"]
        n_params = count_params(model)

        print(f"\n{'='*60}")
        print(f"  Training: {name}")
        print(f"  Note: {b['note']}")
        print(f"  Params: {n_params:,}")
        print(f"{'='*60}")

        train_model(model, b["train_loader"], device, lr=b.get("lr"))

        eff_dim = evaluator.compute_effective_dimension(model, b["train_loader"], device)

        clean_acc, clean_loss = evaluator.evaluate(model, b["val_loader"], device, apply_pgd=False)
        pgd_acc,   pgd_loss   = evaluator.evaluate(model, b["stress_loader"], device, apply_pgd=True)

        retention = pgd_acc / clean_acc if clean_acc > 0 else 0.0

        print(f"\n  ── Results for {name} ──")
        print(f"  Clean Val Acc :  {clean_acc:.4f}   Loss: {clean_loss:.4f}")
        print(f"  PGD  Stress Acc: {pgd_acc:.4f}   Loss: {pgd_loss:.4f}")
        print(f"  Robustness Retention Ratio: {retention:.4f}  ← KEY METRIC")
        print(f"  Param Efficiency (clean): {clean_acc / (n_params/1000):.6f} acc per 1k params")

        evaluator.log_result(name, NUM_CLASSES, clean_acc, clean_loss, eff_dim, n_params, condition="Clean")
        evaluator.log_result(name, NUM_CLASSES, pgd_acc,   pgd_loss,   eff_dim, n_params, condition="PGD_Adversarial")

    evaluator.save_results()
    print(f"\n{'='*60}")
    print("  Benchmarking complete. Results saved to results_comparison.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_benchmarks()

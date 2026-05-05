"""
Training Pipeline for Hybrid QCNN-ViT Arabic Character Recognition.

Implements a robust training loop with:
- Adam optimizer with weight decay
- CrossEntropyLoss for multi-class classification
- Learning rate scheduling (cosine annealing)
- Early stopping with patience
- Best model checkpointing
- Comprehensive metric logging (loss, accuracy, per-class metrics)
- JSON-serializable training history for visualization

Usage:
    python src/train.py                    # Full training (10 epochs)
    python src/train.py --epochs 5         # Custom epochs
    python src/train.py --max-samples 500  # Fast test run
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from data_loader import load_data, ARABIC_CHARS, NUM_CLASSES
from hybrid_qvit import create_model


# ----------------------------------------------
# Training Configuration
# ----------------------------------------------

DEFAULT_CONFIG = {
    'epochs': 10,
    'batch_size': 16,
    'learning_rate': 0.005,
    'weight_decay': 1e-4,
    'max_samples': 2000,       # Subsample for NISQ simulation speed
    'img_size': 8,
    'val_split': 0.15,
    'patience': 5,             # Early stopping patience
    'lr_scheduler': 'cosine',  # 'cosine' or 'step'
    'seed': 42,
    'device': 'cpu',           # Quantum simulation is CPU-based
    'save_dir': os.path.join(PROJECT_ROOT, 'results'),
}


# ----------------------------------------------
# Metrics
# ----------------------------------------------

def compute_accuracy(logits, targets):
    """Compute top-1 accuracy."""
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def compute_per_class_accuracy(logits, targets, n_classes):
    """Compute per-class accuracy."""
    preds = logits.argmax(dim=-1)
    per_class = {}
    for c in range(n_classes):
        mask = targets == c
        if mask.sum() > 0:
            per_class[c] = (preds[mask] == targets[mask]).float().mean().item()
    return per_class


# ----------------------------------------------
# Training Loop
# ----------------------------------------------

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    n_batches = len(train_loader)

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping for stability with quantum gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Metrics
        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_samples += images.size(0)

        # Progress
        if (batch_idx + 1) % max(1, n_batches // 5) == 0 or batch_idx == n_batches - 1:
            running_loss = total_loss / total_samples
            running_acc = total_correct / total_samples
            print(f"    Batch [{batch_idx+1}/{n_batches}] "
                  f"Loss: {running_loss:.4f} | Acc: {running_acc:.4f}")

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_logits = []
    all_labels = []

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_samples += images.size(0)

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return epoch_loss, epoch_acc, all_logits, all_labels


# ----------------------------------------------
# Main Training Function
# ----------------------------------------------

def train(config=None):
    """
    Full training pipeline for QCNN-ViT.

    Args:
        config: dict of training hyperparameters (uses DEFAULT_CONFIG if None)

    Returns:
        model: trained model
        history: dict of training metrics
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    # -- Setup --
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    device = torch.device(config['device'])
    os.makedirs(config['save_dir'], exist_ok=True)

    print("+" + "=" * 58 + "+")
    print("|  Hybrid QCNN-ViT Training Pipeline                      |")
    print("|  Arabic Handwritten Character Recognition                |")
    print("+" + "=" * 58 + "+")
    print(f"\n[CONFIG]")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # -- Data --
    print(f"\n{'-' * 60}")
    print("[STEP 1] Loading and preprocessing data...")
    train_loader, val_loader, test_loader, class_names = load_data(
        max_samples=config['max_samples'],
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        val_split=config['val_split'],
        seed=config['seed'],
    )

    # -- Model --
    print(f"\n{'-' * 60}")
    print("[STEP 2] Building Hybrid QCNN-ViT model...")
    model = create_model(
        n_classes=NUM_CLASSES,
        img_size=config['img_size'],
        device=config['device'],
    )

    params = model.count_parameters()
    print(f"  Total parameters:   {params['total']}")
    print(f"  Quantum parameters: {params['quantum']}")
    print(f"  Classical params:   {params['classical']}")

    # -- Optimizer & Scheduler --
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    if config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=1e-5
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.5
        )

    # -- Training History --
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rates': [],
        'epoch_times': [],
        'best_val_acc': 0.0,
        'best_epoch': 0,
        'config': config,
    }

    best_val_acc = 0.0
    patience_counter = 0

    # -- Training Loop --
    print(f"\n{'-' * 60}")
    print(f"[STEP 3] Training for {config['epochs']} epochs...")
    print(f"{'-' * 60}")

    total_start = time.time()

    for epoch in range(config['epochs']):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\n+- Epoch {epoch+1}/{config['epochs']} "
              f"(lr={current_lr:.6f}) {'-' * 30}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Log metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)
        history['epoch_times'].append(epoch_time)

        # Display
        print(f"|")
        print(f"|  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.4f}")
        print(f"|  Val   Loss: {val_loss:.4f}  |  Val   Acc: {val_acc:.4f}")
        print(f"|  Time: {epoch_time:.1f}s")

        # Best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history['best_val_acc'] = best_val_acc
            history['best_epoch'] = epoch + 1
            patience_counter = 0

            checkpoint_path = os.path.join(config['save_dir'], 'best_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'val_loss': val_loss,
                'config': config,
            }, checkpoint_path)
            print(f"|  * New best model saved! (val_acc={best_val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"|  Patience: {patience_counter}/{config['patience']}")

        print(f"+{'-' * 55}")

        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\n[!] Early stopping triggered at epoch {epoch+1}")
            break

    total_time = time.time() - total_start

    # -- Final Evaluation --
    print(f"\n{'-' * 60}")
    print("[STEP 4] Final evaluation on test set...")

    # Load best model
    checkpoint = torch.load(
        os.path.join(config['save_dir'], 'best_model.pt'),
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_logits, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    # Per-class accuracy
    per_class_acc = compute_per_class_accuracy(test_logits, test_labels, NUM_CLASSES)

    print(f"\n[TEST RESULTS]")
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Best Val Acc:  {best_val_acc:.4f} (epoch {history['best_epoch']})")
    print(f"  Total Time:    {total_time:.1f}s")

    print(f"\n[PER-CLASS ACCURACY]")
    for cls_idx, acc in sorted(per_class_acc.items()):
        char_name = class_names[cls_idx] if cls_idx < len(class_names) else f"Class {cls_idx}"
        print(f"  {char_name:8s}: {acc:.4f}")

    # -- Save Results --
    history['test_loss'] = test_loss
    history['test_acc'] = test_acc
    history['per_class_acc'] = {class_names[k]: v for k, v in per_class_acc.items()
                                 if k < len(class_names)}
    history['total_time'] = total_time
    history['timestamp'] = datetime.now().isoformat()

    # Save training history
    history_path = os.path.join(config['save_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2, default=str)
    print(f"\n[OK] Training history saved to {history_path}")

    # Save predictions for visualization
    np.save(os.path.join(config['save_dir'], 'test_predictions.npy'),
            test_logits.numpy())
    np.save(os.path.join(config['save_dir'], 'test_labels.npy'),
            test_labels.numpy())
    print(f"[OK] Test predictions saved for visualization")

    print(f"\n{'=' * 60}")
    print(f"  Training Complete!")
    print(f"  Best Model: epoch {history['best_epoch']}, "
          f"val_acc={best_val_acc:.4f}, test_acc={test_acc:.4f}")
    print(f"{'=' * 60}")

    return model, history


# ----------------------------------------------
# CLI Entry Point
# ----------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train Hybrid QCNN-ViT for Arabic Character Recognition'
    )
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.005,
                       help='Learning rate')
    parser.add_argument('--max-samples', type=int, default=2000,
                       help='Max training samples (for fast quantum sim)')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config.update({
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'max_samples': args.max_samples,
        'patience': args.patience,
        'seed': args.seed,
    })

    train(config)

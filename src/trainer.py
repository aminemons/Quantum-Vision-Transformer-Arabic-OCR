import os
import sys
import time
import json
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, loader, optimizer, criterion, scaler=None, show_progress=False):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    it = tqdm(loader, leave=False, desc="  batch") if show_progress else loader
    for imgs, labels in it:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * len(imgs)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(imgs)

    return total_loss / total, correct / total


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(imgs)

    return correct / total


def train_model(model, train_loader, val_loader, save_path,
                epochs=30, lr=1e-3, use_amp=True, model_name="model",
                show_batch_progress=False):
    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[train] {model_name} | {n_params:,} params | device={DEVICE}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs
    )
    scaler = torch.amp.GradScaler("cuda") if (use_amp and DEVICE == "cuda") else None

    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    best_val = 0.0

    for epoch in range(epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            show_progress=show_batch_progress
        )
        scheduler.step()
        val_acc = evaluate(model, val_loader)
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        marker = ""
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), save_path)
            marker = " << BEST"

        print(f"  Ep {epoch+1:3d}/{epochs} | loss={tr_loss:.4f} | "
              f"tr={tr_acc:.4f} | val={val_acc:.4f} | {elapsed:.1f}s{marker}")

    model.load_state_dict(torch.load(save_path, weights_only=True))
    return model, history

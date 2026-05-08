import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from tqdm import tqdm

from data_loader import HMBDDataLoader
from models import ClassicalCNN, HybridQNN, MultiClassQCNN
from eval import Evaluator

def train_model(model, dataloader, epochs=30, lr=0.001, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

def run_benchmarks():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = 115
    print(f"\n{'='*40}\nRunning Benchmark for {num_classes} Classes\n{'='*40}")

    loader = HMBDDataLoader(total_classes=num_classes, batch_size=128)
    loader.prepare_data()

    evaluator = Evaluator()

    # ----------------------------------------------------------------
    # Model A: ClassicalCNN -- uses raw [0,1] pixel loaders
    # ----------------------------------------------------------------
    name = "ClassicalCNN"
    model = ClassicalCNN(num_classes=num_classes)
    print(f"\nTraining {name}...")
    print(f"Parameters: {evaluator.count_parameters(model)}")
    train_model(model, loader.train_loader, epochs=30, lr=0.001, device=device)

    eff_dim = evaluator.compute_effective_dimension(model, loader.train_loader, device)
    clean_acc, clean_loss = evaluator.evaluate(model, loader.val_loader, device, apply_pgd=False)
    print(f"Clean Val - Acc: {clean_acc:.4f}, Loss: {clean_loss:.4f}")
    evaluator.log_result(name, num_classes, clean_acc, clean_loss, eff_dim, evaluator.count_parameters(model), condition="Clean")

    noisy_acc, noisy_loss = evaluator.evaluate(model, loader.stress_test_loader, device, apply_pgd=True)
    print(f"PGD Stress Test - Acc: {noisy_acc:.4f}, Loss: {noisy_loss:.4f}")
    evaluator.log_result(name, num_classes, noisy_acc, noisy_loss, eff_dim, evaluator.count_parameters(model), condition="PGD_Adversarial")

    # ----------------------------------------------------------------
    # Model B: HybridQNN -- uses raw [0,1] pixel loaders
    # ----------------------------------------------------------------
    name = "HybridQNN"
    model = HybridQNN(num_classes=num_classes)
    print(f"\nTraining {name}...")
    print(f"Parameters: {evaluator.count_parameters(model)}")
    train_model(model, loader.train_loader, epochs=30, lr=0.001, device=device)

    eff_dim = evaluator.compute_effective_dimension(model, loader.train_loader, device)
    clean_acc, clean_loss = evaluator.evaluate(model, loader.val_loader, device, apply_pgd=False)
    print(f"Clean Val - Acc: {clean_acc:.4f}, Loss: {clean_loss:.4f}")
    evaluator.log_result(name, num_classes, clean_acc, clean_loss, eff_dim, evaluator.count_parameters(model), condition="Clean")

    noisy_acc, noisy_loss = evaluator.evaluate(model, loader.stress_test_loader, device, apply_pgd=True)
    print(f"PGD Stress Test - Acc: {noisy_acc:.4f}, Loss: {noisy_loss:.4f}")
    evaluator.log_result(name, num_classes, noisy_acc, noisy_loss, eff_dim, evaluator.count_parameters(model), condition="PGD_Adversarial")

    # ----------------------------------------------------------------
    # Model C: MultiClassQCNN -- uses L2-unit-normalized loaders ONLY
    # ----------------------------------------------------------------
    name = "MultiClassQCNN"
    model = MultiClassQCNN(num_classes=num_classes)
    print(f"\nTraining {name}...")
    print(f"Parameters: {evaluator.count_parameters(model)}")
    train_model(model, loader.train_loader_qcnn, epochs=30, lr=0.005, device=device)

    eff_dim = evaluator.compute_effective_dimension(model, loader.train_loader_qcnn, device)
    clean_acc, clean_loss = evaluator.evaluate(model, loader.val_loader_qcnn, device, apply_pgd=False)
    print(f"Clean Val - Acc: {clean_acc:.4f}, Loss: {clean_loss:.4f}")
    evaluator.log_result(name, num_classes, clean_acc, clean_loss, eff_dim, evaluator.count_parameters(model), condition="Clean")

    noisy_acc, noisy_loss = evaluator.evaluate(model, loader.stress_test_loader_qcnn, device, apply_pgd=True)
    print(f"PGD Stress Test - Acc: {noisy_acc:.4f}, Loss: {noisy_loss:.4f}")
    evaluator.log_result(name, num_classes, noisy_acc, noisy_loss, eff_dim, evaluator.count_parameters(model), condition="PGD_Adversarial")

    evaluator.save_results()
    print("\nBenchmarking complete. Results saved to results_comparison.csv")

if __name__ == "__main__":
    run_benchmarks()

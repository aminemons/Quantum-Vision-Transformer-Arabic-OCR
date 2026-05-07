import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from tqdm import tqdm

from data_loader import HMBDDataLoader
from models import ClassicalCNN, HybridQNN, MultiClassQCNN
from eval import Evaluator

def train_model(model, dataloader, epochs=10, lr=0.002, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        scheduler.step()
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

def run_benchmarks():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    class_counts = [28, 115]
    evaluator = Evaluator()
    
    for c in class_counts:
        print(f"\n{'='*40}\nRunning Benchmark for {c} Classes\n{'='*40}")
        loader = HMBDDataLoader(total_classes=c, batch_size=16)
        loader.prepare_data()
        
        models = {
            "ClassicalCNN": ClassicalCNN(num_classes=c),
            "HybridQNN": HybridQNN(num_classes=c),
            "MultiClassQCNN": MultiClassQCNN(num_classes=c)
        }
        
        for name, model in models.items():
            print(f"\nTraining {name} on {c} classes...")
            
            num_params = evaluator.count_parameters(model)
            print(f"Parameters: {num_params}")
            
            train_model(model, loader.train_loader, epochs=10, lr=0.002, device=device)
            
            eff_dim = evaluator.compute_effective_dimension(model, loader.train_loader, device)
            
            clean_acc, clean_loss = evaluator.evaluate(model, loader.val_loader, device, apply_pgd=False)
            print(f"Clean Val - Acc: {clean_acc:.4f}, Loss: {clean_loss:.4f}")
            evaluator.log_result(name, c, clean_acc, clean_loss, eff_dim, num_params, condition="Clean")
            
            noisy_acc, noisy_loss = evaluator.evaluate(model, loader.stress_test_loader, device, apply_pgd=True)
            print(f"PGD Stress Test - Acc: {noisy_acc:.4f}, Loss: {noisy_loss:.4f}")
            evaluator.log_result(name, c, noisy_acc, noisy_loss, eff_dim, num_params, condition="PGD_Adversarial")
            
    evaluator.save_results()
    print("\nBenchmarking complete. Results saved to results_comparison.csv")

if __name__ == "__main__":
    run_benchmarks()

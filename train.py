import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os

from data_loader import HMBDDataLoader
from models import ClassicalCNN, HybridQNN, MultiClassQCNN
from eval import Evaluator

def train_model(model, dataloader, epochs=5, lr=0.01, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            
        scheduler.step()
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

def run_benchmarks():
    device = torch.device("cpu")
    class_counts = [28, 115]
    evaluator = Evaluator()
    
    for c in class_counts:
        print(f"\n{'='*40}\nRunning Benchmark for {c} Classes\n{'='*40}")
        loader = HMBDDataLoader(data_dir="./data", total_classes=c, batch_size=16)
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
            
            train_model(model, loader.train_loader, epochs=2, lr=0.01, device=device)
            
            eff_dim = evaluator.compute_effective_dimension(model, loader.train_loader, device)
            
            clean_acc, clean_loss = evaluator.evaluate(model, loader.val_loader, device, apply_noise=False)
            print(f"Clean Val - Acc: {clean_acc:.4f}, Loss: {clean_loss:.4f}")
            evaluator.log_result(name, c, clean_acc, clean_loss, eff_dim, num_params, condition="Clean")
            
            noisy_acc, noisy_loss = evaluator.evaluate(model, loader.stress_test_loader, device, apply_noise=True)
            print(f"Noisy Stress Test - Acc: {noisy_acc:.4f}, Loss: {noisy_loss:.4f}")
            evaluator.log_result(name, c, noisy_acc, noisy_loss, eff_dim, num_params, condition="Noisy")
            
    evaluator.save_results()
    print("\nBenchmarking complete. Results saved to results_comparison.csv")

if __name__ == "__main__":
    run_benchmarks()

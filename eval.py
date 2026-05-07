import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

class Evaluator:
    def __init__(self, csv_path="results_comparison.csv"):
        self.csv_path = csv_path
        self.results = []
        self.criterion = nn.CrossEntropyLoss()

    def add_noise(self, x, noise_level=0.1):
        noise = torch.randn_like(x) * noise_level
        noisy_x = x + noise
        noisy_x = torch.clamp(noisy_x, -1.0, 1.0)
        
        norms = torch.norm(noisy_x, dim=1, keepdim=True)
        norms[norms == 0] = 1.0
        return noisy_x / norms

    def compute_effective_dimension(self, model, dataloader, device, num_samples=100):
        model.eval()
        gradients = []
        
        for idx, (x, y) in enumerate(dataloader):
            if idx * x.size(0) >= num_samples:
                break
                
            x, y = x.to(device), y.to(device)
            
            for i in range(x.size(0)):
                model.zero_grad()
                out = model(x[i:i+1])
                loss = self.criterion(out, y[i:i+1])
                loss.backward()
                
                grad_vec = []
                for param in model.parameters():
                    if param.grad is not None:
                        grad_vec.append(param.grad.view(-1))
                
                if grad_vec:
                    gradients.append(torch.cat(grad_vec).cpu().numpy())
                    
        if not gradients:
            return 0.0
            
        grad_matrix = np.stack(gradients)
        
        try:
            _, s, _ = np.linalg.svd(grad_matrix, full_matrices=False)
            eff_dim = np.sum(s > 1e-5)
        except Exception:
            eff_dim = len(gradients[0])
            
        return eff_dim

    def evaluate(self, model, dataloader, device, apply_noise=False):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                
                if apply_noise:
                    x = self.add_noise(x)
                    
                outputs = model(x)
                loss = self.criterion(outputs, y)
                
                total_loss += loss.item() * x.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                
        avg_loss = total_loss / total
        accuracy = correct / total
        return accuracy, avg_loss

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def log_result(self, model_name, num_classes, accuracy, loss, eff_dim, num_params, condition="Clean"):
        acc_per_1000 = (accuracy / num_params) * 1000 if num_params > 0 else 0
        
        self.results.append({
            "Model": model_name,
            "Classes": num_classes,
            "Condition": condition,
            "Accuracy": accuracy,
            "CrossEntropyLoss": loss,
            "EffectiveDimension": eff_dim,
            "Parameters": num_params,
            "AccuracyPer1000Params": acc_per_1000
        })

    def save_results(self):
        df = pd.DataFrame(self.results)
        df.to_csv(self.csv_path, index=False)
        return df

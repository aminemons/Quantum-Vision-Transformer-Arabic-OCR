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

    def pgd_attack(self, model, x, y, epsilon=0.1, alpha=0.02, iters=10):
        x_adv = x.clone().detach().requires_grad_(True)
        
        for _ in range(iters):
            model.zero_grad()
            outputs = model(x_adv)
            loss = self.criterion(outputs, y)
            loss.backward()
            
            with torch.no_grad():
                adv_data = x_adv + alpha * x_adv.grad.sign()
                eta = torch.clamp(adv_data - x, min=-epsilon, max=epsilon)
                x_adv = torch.clamp(x + eta, min=-1.0, max=1.0)
                
                norms = torch.norm(x_adv, dim=1, keepdim=True)
                norms[norms == 0] = 1.0
                x_adv = x_adv / norms
                
            x_adv.requires_grad_(True)
            
        return x_adv.detach()

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

    def evaluate(self, model, dataloader, device, apply_pgd=False):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            if apply_pgd:
                # Need to temporarily set model to train mode for gradients if needed, 
                # but eval mode is fine as long as requires_grad is on inputs.
                x = self.pgd_attack(model, x, y)
                
            with torch.no_grad():
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

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset, DataLoader

class SyntheticHMBDDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.long)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

class HMBDDataLoader:
    def __init__(self, data_dir, total_classes=115, stress_test_samples_per_class=250, train_val_samples_per_class=100, pca_components=256, batch_size=16):
        self.data_dir = data_dir
        self.total_classes = total_classes
        self.stress_test_samples_per_class = stress_test_samples_per_class
        self.train_val_samples_per_class = train_val_samples_per_class
        self.pca_components = pca_components
        self.batch_size = batch_size
        self.pca_model = PCA(n_components=self.pca_components)
        
        self.train_loader = None
        self.val_loader = None
        self.stress_test_loader = None

    def generate_synthetic_data(self):
        total_samples_per_class = self.train_val_samples_per_class + self.stress_test_samples_per_class
        total_samples = self.total_classes * total_samples_per_class
        
        x_raw = np.zeros((total_samples, 32 * 32), dtype=np.float32)
        y_raw = np.repeat(np.arange(self.total_classes), total_samples_per_class)
        
        base_patterns = np.random.rand(self.total_classes, 32 * 32).astype(np.float32)
        for i in range(total_samples):
            class_idx = y_raw[i]
            x_raw[i] = base_patterns[class_idx] + np.random.randn(32 * 32).astype(np.float32) * 0.2
        
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        
        x_raw = x_raw[indices]
        y_raw = y_raw[indices]
        
        return x_raw, y_raw

    def prepare_data(self):
        x_raw, y_raw = self.generate_synthetic_data()
        
        x_train_val = []
        y_train_val = []
        x_stress = []
        y_stress = []
        
        for class_idx in range(self.total_classes):
            class_mask = y_raw == class_idx
            x_class = x_raw[class_mask]
            y_class = y_raw[class_mask]
            
            x_stress.append(x_class[:self.stress_test_samples_per_class])
            y_stress.append(y_class[:self.stress_test_samples_per_class])
            
            x_train_val.append(x_class[self.stress_test_samples_per_class:])
            y_train_val.append(y_class[self.stress_test_samples_per_class:])
            
        x_stress = np.concatenate(x_stress, axis=0)
        y_stress = np.concatenate(y_stress, axis=0)
        x_train_val = np.concatenate(x_train_val, axis=0)
        y_train_val = np.concatenate(y_train_val, axis=0)
        
        self.pca_model.fit(x_train_val)
        
        x_train_val_pca = self.pca_model.transform(x_train_val)
        x_stress_pca = self.pca_model.transform(x_stress)
        
        norms_train_val = np.linalg.norm(x_train_val_pca, axis=1, keepdims=True)
        norms_train_val[norms_train_val == 0] = 1.0
        x_train_val_normalized = x_train_val_pca / norms_train_val
        
        norms_stress = np.linalg.norm(x_stress_pca, axis=1, keepdims=True)
        norms_stress[norms_stress == 0] = 1.0
        x_stress_normalized = x_stress_pca / norms_stress
        
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val_normalized, 
            y_train_val, 
            test_size=0.2, 
            stratify=y_train_val
        )
        
        train_dataset = SyntheticHMBDDataset(x_train, y_train)
        val_dataset = SyntheticHMBDDataset(x_val, y_val)
        stress_dataset = SyntheticHMBDDataset(x_stress_normalized, y_stress)
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.stress_test_loader = DataLoader(stress_dataset, batch_size=self.batch_size, shuffle=False)

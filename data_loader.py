import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from PIL import Image

class HMBDDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.long)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

class HMBDDataLoader:
    def __init__(self, data_dir="./data/hmbd-v1", total_classes=115, stress_test_samples_per_class=250, batch_size=16):
        self.data_dir = data_dir
        self.total_classes = total_classes
        self.stress_test_samples_per_class = stress_test_samples_per_class
        self.batch_size = batch_size
        
        self.train_loader = None
        self.val_loader = None
        self.stress_test_loader = None

    def load_raw_images(self):
        actual_dir = os.path.join(self.data_dir, "Dataset") if os.path.exists(os.path.join(self.data_dir, "Dataset")) else self.data_dir
        print(f"Loading real HMBD-v1 dataset from {actual_dir}...")
        
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((16, 16)),
            transforms.ToTensor()
        ])
        
        def is_valid(path):
            try:
                img = Image.open(path)
                img.verify()
                return True
            except Exception:
                return False
        
        dataset = datasets.ImageFolder(root=actual_dir, transform=transform, is_valid_file=is_valid)
        
        if len(dataset.classes) != self.total_classes:
            print(f"Warning: Expected {self.total_classes} classes, but found {len(dataset.classes)} in {actual_dir}")
        
        # Load all into memory for PCA (54k images of 32x32 is ~55MB, easily fits in memory)
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
        
        x_all = []
        y_all = []
        
        for batch_x, batch_y in dataloader:
            # Flatten to 1024
            batch_x = batch_x.view(batch_x.size(0), -1).numpy()
            x_all.append(batch_x)
            y_all.append(batch_y.numpy())
            
        x_all = np.concatenate(x_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)
        
        # Filter down to the requested number of classes
        mask = y_all < self.total_classes
        x_all = x_all[mask]
        y_all = y_all[mask]
        
        return x_all, y_all

    def prepare_data(self):
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Dataset not found at {self.data_dir}. Please download the hossammbalaha/hmbd-v1 dataset from Kaggle and extract it there.")
            
        x_raw, y_raw = self.load_raw_images()
        
        x_train_val = []
        y_train_val = []
        x_stress = []
        y_stress = []
        
        # Stratified Split ensuring exactly stress_test_samples_per_class per class
        for class_idx in np.unique(y_raw):
            class_mask = y_raw == class_idx
            x_class = x_raw[class_mask]
            y_class = y_raw[class_mask]
            
            # Shuffle class samples
            indices = np.arange(len(x_class))
            np.random.shuffle(indices)
            x_class = x_class[indices]
            y_class = y_class[indices]
            
            x_stress.append(x_class[:self.stress_test_samples_per_class])
            y_stress.append(y_class[:self.stress_test_samples_per_class])
            
            x_train_val.append(x_class[self.stress_test_samples_per_class:])
            y_train_val.append(y_class[self.stress_test_samples_per_class:])
            
        x_stress = np.concatenate(x_stress, axis=0)
        y_stress = np.concatenate(y_stress, axis=0)
        x_train_val = np.concatenate(x_train_val, axis=0)
        y_train_val = np.concatenate(y_train_val, axis=0)
        
        print("Normalizing features via Amplitude Encoding (L2-norm = 1)...")
        
        # Amplitude Encoding Normalization (L2 norm = 1)
        norms_train_val = np.linalg.norm(x_train_val, axis=1, keepdims=True)
        norms_train_val[norms_train_val == 0] = 1.0
        x_train_val_normalized = x_train_val / norms_train_val
        
        norms_stress = np.linalg.norm(x_stress, axis=1, keepdims=True)
        norms_stress[norms_stress == 0] = 1.0
        x_stress_normalized = x_stress / norms_stress
        
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val_normalized, 
            y_train_val, 
            test_size=0.2, 
            stratify=y_train_val,
            random_state=42
        )
        
        print(f"Data prepared! Train: {len(x_train)}, Val: {len(x_val)}, Stress: {len(x_stress_normalized)}")
        
        train_dataset = HMBDDataset(x_train, y_train)
        val_dataset = HMBDDataset(x_val, y_val)
        stress_dataset = HMBDDataset(x_stress_normalized, y_stress)
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.stress_test_loader = DataLoader(stress_dataset, batch_size=self.batch_size, shuffle=False)

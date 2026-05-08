import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
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
    def __init__(self, data_dir="./data/hmbd-v1", total_classes=115, stress_test_samples_per_class=250, batch_size=128):
        self.data_dir = data_dir
        self.total_classes = total_classes
        self.stress_test_samples_per_class = stress_test_samples_per_class
        self.batch_size = batch_size

        # Standard pixel-space loaders (for ClassicalCNN + HybridQNN)
        self.train_loader = None
        self.val_loader = None
        self.stress_test_loader = None

        # L2-unit-normalized loaders (for MultiClassQCNN AmplitudeEmbedding only)
        self.train_loader_qcnn = None
        self.val_loader_qcnn = None
        self.stress_test_loader_qcnn = None

    def load_raw_images(self):
        actual_dir = os.path.join(self.data_dir, "Dataset") if os.path.exists(os.path.join(self.data_dir, "Dataset")) else self.data_dir
        print(f"Loading HMBD-v1 from {actual_dir}...")

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

        n_found = len(dataset.classes)
        if n_found < self.total_classes:
            print(f"Warning: Found only {n_found} classes (expected {self.total_classes}).")

        loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

        x_all, y_all = [], []
        for batch_x, batch_y in loader:
            x_all.append(batch_x.view(batch_x.size(0), -1).numpy())
            y_all.append(batch_y.numpy())

        x_all = np.concatenate(x_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)

        mask = y_all < self.total_classes
        return x_all[mask], y_all[mask]

    def _stratified_split(self, x_raw, y_raw):
        x_train_val, x_stress = [], []
        y_train_val, y_stress = [], []

        for class_idx in np.unique(y_raw):
            mask = y_raw == class_idx
            x_cls, y_cls = x_raw[mask], y_raw[mask]
            idx = np.random.permutation(len(x_cls))
            x_cls, y_cls = x_cls[idx], y_cls[idx]

            n_stress = min(self.stress_test_samples_per_class, len(x_cls) // 3)
            x_stress.append(x_cls[:n_stress])
            y_stress.append(y_cls[:n_stress])
            x_train_val.append(x_cls[n_stress:])
            y_train_val.append(y_cls[n_stress:])

        return (np.concatenate(x_train_val), np.concatenate(y_train_val),
                np.concatenate(x_stress), np.concatenate(y_stress))

    def prepare_data(self):
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Dataset not found at {self.data_dir}.")

        x_raw, y_raw = self.load_raw_images()
        x_tv, y_tv, x_stress, y_stress = self._stratified_split(x_raw, y_raw)

        # ---- Raw pixel data (stays in [0,1]) for CNN and HybridQNN ----
        x_train, x_val, y_train, y_val = train_test_split(
            x_tv, y_tv, test_size=0.2, stratify=y_tv, random_state=42
        )
        print(f"Data prepared! Train: {len(x_train)}, Val: {len(x_val)}, Stress: {len(x_stress)}")

        self.train_loader       = DataLoader(HMBDDataset(x_train, y_train),   batch_size=self.batch_size, shuffle=True,  pin_memory=True)
        self.val_loader         = DataLoader(HMBDDataset(x_val, y_val),       batch_size=self.batch_size, shuffle=False, pin_memory=True)
        self.stress_test_loader = DataLoader(HMBDDataset(x_stress, y_stress), batch_size=self.batch_size, shuffle=False, pin_memory=True)

        # ---- L2-unit-normalized data (for QCNN AmplitudeEmbedding only) ----
        def l2_norm(x):
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return x / norms

        x_tv_q = l2_norm(x_tv)
        x_stress_q = l2_norm(x_stress)
        x_train_q, x_val_q, y_train_q, y_val_q = train_test_split(
            x_tv_q, y_tv, test_size=0.2, stratify=y_tv, random_state=42
        )

        self.train_loader_qcnn       = DataLoader(HMBDDataset(x_train_q, y_train_q),   batch_size=self.batch_size, shuffle=True,  pin_memory=True)
        self.val_loader_qcnn         = DataLoader(HMBDDataset(x_val_q, y_val_q),       batch_size=self.batch_size, shuffle=False, pin_memory=True)
        self.stress_test_loader_qcnn = DataLoader(HMBDDataset(x_stress_q, y_stress),   batch_size=self.batch_size, shuffle=False, pin_memory=True)

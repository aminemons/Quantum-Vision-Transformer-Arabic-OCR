"""
Arabic Handwritten Character Dataset (AHCD) Data Engineering Pipeline.

Downloads, preprocesses, and serves the AHCD dataset for quantum-classical
hybrid model training. Images are resized to 8x8 to fit NISQ quantum
simulation constraints (64 pixels -> manageable qubit encoding).

Dataset: AHCD (Arabic Handwritten Characters Dataset)
- 13,440 training images / 3,360 test images
- 28 Arabic character classes (Alef -> Yaa)
- Original size: 32x32 grayscale

Reference:
    El-Sawy, A., Loey, M., & El-Bakry, H. (2017).
    Arabic Handwritten Characters Recognition using CNN.
"""

import os
import sys
import csv
import urllib.request
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image

# -----------------------------------------------
# Constants
# -----------------------------------------------
ARABIC_CHARS = [
    'Alef', 'Beh', 'Teh', 'Theh', 'Jeem', 'Hah', 'Khah',
    'Dal', 'Thal', 'Reh', 'Zain', 'Seen', 'Sheen', 'Sad',
    'Dad', 'Tah', 'Zah', 'Ain', 'Ghain', 'Feh', 'Qaf',
    'Kaf', 'Lam', 'Meem', 'Noon', 'Heh', 'Waw', 'Yeh'
]

ARABIC_UNICODE = [
    'Alef', 'Beh', 'Teh', 'Theh', 'Jeem', 'Hah', 'Khah',
    'Dal', 'Thal', 'Reh', 'Zain', 'Seen', 'Sheen', 'Sad',
    'Dad', 'Tah', 'Zah', 'Ain', 'Ghain', 'Feh', 'Qaf',
    'Kaf', 'Lam', 'Meem', 'Noon', 'Heh', 'Waw', 'Yeh'
]

NUM_CLASSES = 28
IMG_SIZE = 8  # Quantum-ready: 8x8 = 64 pixels
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

# -----------------------------------------------
# Dataset Download
# -----------------------------------------------

def _download_ahcd_kaggle():
    """Download AHCD dataset using kagglehub."""
    try:
        import kagglehub
        path = kagglehub.dataset_download("mloey1/ahcd1")
        print(f"[OK] Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"[WARN] kagglehub download failed: {e}")
        return None


def _generate_synthetic_ahcd(data_dir: str, n_train: int = 13440, n_test: int = 3360):
    """
    Generate a synthetic AHCD-like dataset for development/testing.
    Creates structured character-like patterns for each of the 28 classes.
    Falls back to this when kagglehub is unavailable.
    """
    print("[INFO] Generating synthetic Arabic character dataset for development...")
    os.makedirs(data_dir, exist_ok=True)

    np.random.seed(42)

    def _make_char_pattern(class_idx, size=32):
        """Create a distinct structural pattern per character class."""
        img = np.zeros((size, size), dtype=np.float32)
        rng = np.random.RandomState(class_idx * 137)

        # Base stroke pattern unique to each class
        center_x, center_y = size // 2, size // 2
        angle = (class_idx / NUM_CLASSES) * 2 * np.pi

        # Main stroke
        for t in np.linspace(0, 1, 60):
            x = int(center_x + (size // 3) * np.cos(angle + t * np.pi * (1 + class_idx % 3)))
            y = int(center_y + (size // 3) * np.sin(angle + t * np.pi * (1 + class_idx % 4)))
            x, y = np.clip(x, 1, size - 2), np.clip(y, 1, size - 2)
            img[y - 1:y + 2, x - 1:x + 2] = 1.0

        # Add dots for dotted characters (like beh, teh, theh, etc.)
        if class_idx in [1, 2, 3, 4, 5, 6, 13, 14, 15, 16, 19, 20]:
            n_dots = (class_idx % 3) + 1
            for d in range(n_dots):
                dx = rng.randint(size // 4, 3 * size // 4)
                dy = rng.randint(size // 4, 3 * size // 4)
                img[dy - 1:dy + 2, dx - 1:dx + 2] = 1.0

        return img

    # Generate training data
    train_images = []
    train_labels = []
    samples_per_class_train = n_train // NUM_CLASSES

    for cls in range(NUM_CLASSES):
        pattern = _make_char_pattern(cls)
        for i in range(samples_per_class_train):
            noise = np.random.normal(0, 0.08, pattern.shape).astype(np.float32)
            shift_x = np.random.randint(-2, 3)
            shift_y = np.random.randint(-2, 3)
            augmented = np.roll(np.roll(pattern + noise, shift_x, axis=1), shift_y, axis=0)
            augmented = np.clip(augmented, 0, 1)
            train_images.append(augmented.flatten())
            train_labels.append(cls + 1)  # AHCD labels are 1-indexed

    # Generate test data
    test_images = []
    test_labels = []
    samples_per_class_test = n_test // NUM_CLASSES

    for cls in range(NUM_CLASSES):
        pattern = _make_char_pattern(cls)
        for i in range(samples_per_class_test):
            noise = np.random.normal(0, 0.1, pattern.shape).astype(np.float32)
            shift_x = np.random.randint(-3, 4)
            shift_y = np.random.randint(-3, 4)
            augmented = np.roll(np.roll(pattern + noise, shift_x, axis=1), shift_y, axis=0)
            augmented = np.clip(augmented, 0, 1)
            test_images.append(augmented.flatten())
            test_labels.append(cls + 1)

    # Save as CSV (matching AHCD format: 1024 pixels per image = 32x32)
    train_images = np.array(train_images)
    train_labels = np.array(train_labels).reshape(-1, 1)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels).reshape(-1, 1)

    np.savetxt(os.path.join(data_dir, 'csvTrainImages 13440x1024.csv'), train_images, delimiter=',', fmt='%.6f')
    np.savetxt(os.path.join(data_dir, 'csvTrainLabel 13440x1.csv'), train_labels, delimiter=',', fmt='%d')
    np.savetxt(os.path.join(data_dir, 'csvTestImages 3360x1024.csv'), test_images, delimiter=',', fmt='%.6f')
    np.savetxt(os.path.join(data_dir, 'csvTestLabel 3360x1.csv'), test_labels, delimiter=',', fmt='%d')

    print(f"[OK] Synthetic dataset generated: {n_train} train / {n_test} test samples")
    return data_dir


def download_dataset(data_dir: str = None) -> str:
    """
    Download the AHCD dataset. Tries kagglehub first, falls back to
    synthetic data generation for development.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    os.makedirs(data_dir, exist_ok=True)

    # Check if data already exists in data_dir
    csv_dir = _find_csv_dir(data_dir)
    if csv_dir:
        print(f"[OK] Dataset already present at: {csv_dir}")
        return csv_dir

    # Try kagglehub
    kaggle_path = _download_ahcd_kaggle()
    if kaggle_path:
        csv_dir = _find_csv_dir(kaggle_path)
        if csv_dir:
            return csv_dir

    # Fallback: generate synthetic data
    print("[WARN] Could not download real AHCD dataset. Generating synthetic data...")
    return _generate_synthetic_ahcd(data_dir)


def _find_csv_dir(base_dir: str) -> str:
    """Search for the 4 AHCD CSV files in directory and subdirectories."""
    for root, dirs, files in os.walk(base_dir):
        csv_files = [f for f in files if f.lower().endswith('.csv')]
        has_train_img = any('trainimages' in f.lower().replace(' ', '') for f in csv_files)
        has_train_lbl = any('trainlabel' in f.lower().replace(' ', '') for f in csv_files)
        has_test_img = any('testimages' in f.lower().replace(' ', '') for f in csv_files)
        has_test_lbl = any('testlabel' in f.lower().replace(' ', '') for f in csv_files)
        if has_train_img and has_train_lbl and has_test_img and has_test_lbl:
            return root
    return None


# -----------------------------------------------
# PyTorch Dataset
# -----------------------------------------------

class ArabicCharDataset(Dataset):
    """
    PyTorch Dataset for Arabic Handwritten Characters.

    Loads pre-processed character images, resizes to IMG_SIZE x IMG_SIZE,
    and normalizes pixel values to [0, pi] for quantum angle embedding.

    Args:
        images: numpy array of shape (N, H*W) -- flattened grayscale images
        labels: numpy array of shape (N,) -- class labels (0-indexed)
        img_size: target image size for quantum processing
        normalize_range: tuple (min, max) for pixel normalization
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray,
                 img_size: int = IMG_SIZE,
                 normalize_range: tuple = (0, np.pi)):
        super().__init__()
        self.img_size = img_size
        self.normalize_range = normalize_range

        # Determine original image dimensions
        n_pixels = images.shape[1]
        orig_size = int(np.sqrt(n_pixels))

        # Resize images to target size
        self.images = self._resize_images(images, orig_size, img_size)

        # Normalize to [0, pi] for angle embedding
        self.images = self._normalize(self.images, normalize_range)

        # Store as float32 tensors
        self.images = torch.tensor(self.images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def _resize_images(self, images: np.ndarray, orig_size: int,
                       target_size: int) -> np.ndarray:
        """Resize all images using PIL for high-quality downsampling."""
        resized = []
        for img_flat in images:
            img = img_flat.reshape(orig_size, orig_size)
            pil_img = Image.fromarray((img * 255).astype(np.uint8) if img.max() <= 1.0
                                      else img.astype(np.uint8))
            pil_img = pil_img.resize((target_size, target_size), Image.LANCZOS)
            resized.append(np.array(pil_img, dtype=np.float32).flatten())
        return np.array(resized)

    def _normalize(self, images: np.ndarray,
                   value_range: tuple) -> np.ndarray:
        """Normalize pixel values to specified range."""
        lo, hi = value_range
        img_min = images.min(axis=1, keepdims=True)
        img_max = images.max(axis=1, keepdims=True)
        # Avoid division by zero
        denom = np.where(img_max - img_min > 1e-8, img_max - img_min, 1.0)
        normalized = (images - img_min) / denom
        return normalized * (hi - lo) + lo

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def get_image_2d(self, idx):
        """Return a 2D image for visualization."""
        return self.images[idx].reshape(self.img_size, self.img_size)


# -----------------------------------------------
# Data Loading Pipeline
# -----------------------------------------------

def _load_csv(filepath: str) -> np.ndarray:
    """Load a CSV file using pandas for fast I/O on large AHCD files."""
    import pandas as pd
    print(f"    Loading: {os.path.basename(filepath)}...", end=" ", flush=True)
    data = pd.read_csv(filepath, header=None).values.astype(np.float32)
    print(f"shape={data.shape}")
    return data


def _find_file(data_dir: str, base_name: str) -> str:
    """Find a file matching base_name pattern in data_dir."""
    # base_name is like 'csvTrainImages' or 'csvTestLabel'
    key = base_name.lower().replace('csv', '').replace(' ', '')
    for f in os.listdir(data_dir):
        fname_key = f.lower().replace('csv', '').replace(' ', '').replace('.csv', '')
        if key in fname_key and f.lower().endswith('.csv'):
            return os.path.join(data_dir, f)
    raise FileNotFoundError(f"Cannot find file matching '{base_name}' in {data_dir}. "
                            f"Files found: {os.listdir(data_dir)}")


def load_data(data_dir: str = None, max_samples: int = None,
              val_split: float = 0.15, batch_size: int = 32,
              img_size: int = IMG_SIZE, seed: int = 42):
    """
    Full data loading pipeline: download -> preprocess -> DataLoader.

    Args:
        data_dir: path to data directory (auto-downloads if empty)
        max_samples: limit number of training samples (for fast iteration)
        val_split: fraction of training data for validation
        batch_size: DataLoader batch size
        img_size: target image resolution (default 8x8)
        seed: random seed for reproducibility

    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        test_loader: DataLoader for testing
        class_names: list of Arabic character names
    """
    # Step 1: Download/locate dataset
    csv_dir = download_dataset(data_dir)

    # Step 2: Load CSV files
    print("[INFO] Loading CSV data...")
    train_images = _load_csv(_find_file(csv_dir, 'csvTrainImages'))
    train_labels = _load_csv(_find_file(csv_dir, 'csvTrainLabel')).flatten()
    test_images = _load_csv(_find_file(csv_dir, 'csvTestImages'))
    test_labels = _load_csv(_find_file(csv_dir, 'csvTestLabel')).flatten()

    # Convert labels to 0-indexed
    train_labels = train_labels.astype(int) - 1
    test_labels = test_labels.astype(int) - 1

    print(f"[OK] Raw data loaded: {train_images.shape[0]} train / {test_images.shape[0]} test")
    print(f"    Image dimensions: {train_images.shape[1]} pixels -> {img_size}x{img_size}")
    print(f"    Classes: {len(np.unique(train_labels))} Arabic characters")

    # Step 3: Subsample if requested (for fast quantum simulation)
    if max_samples is not None and max_samples < len(train_images):
        # Stratified subsampling to maintain class balance
        indices, _ = train_test_split(
            np.arange(len(train_images)),
            train_size=max_samples,
            stratify=train_labels,
            random_state=seed
        )
        train_images = train_images[indices]
        train_labels = train_labels[indices]
        print(f"[INFO] Subsampled to {max_samples} training samples (stratified)")

    # Step 4: Train/validation split (stratified)
    train_idx, val_idx = train_test_split(
        np.arange(len(train_images)),
        test_size=val_split,
        stratify=train_labels,
        random_state=seed
    )

    # Step 5: Create datasets
    full_train_dataset = ArabicCharDataset(train_images, train_labels, img_size=img_size)
    test_dataset = ArabicCharDataset(test_images, test_labels, img_size=img_size)

    train_dataset = Subset(full_train_dataset, train_idx)
    val_dataset = Subset(full_train_dataset, val_idx)

    print(f"[OK] Datasets created: {len(train_dataset)} train / "
          f"{len(val_dataset)} val / {len(test_dataset)} test")

    # Step 6: Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    print(f"[OK] DataLoaders ready (batch_size={batch_size})")
    print(f"    Training batches: {len(train_loader)}")
    print(f"    Validation batches: {len(val_loader)}")
    print(f"    Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, ARABIC_CHARS


# -----------------------------------------------
# Standalone Test
# -----------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print(" AHCD Data Pipeline -- Standalone Verification")
    print("=" * 60)

    train_loader, val_loader, test_loader, class_names = load_data(
        max_samples=2000, batch_size=32
    )

    # Inspect a batch
    images, labels = next(iter(train_loader))
    print(f"\n[BATCH INSPECTION]")
    print(f"  Image batch shape: {images.shape}")
    print(f"  Label batch shape: {labels.shape}")
    print(f"  Pixel value range: [{images.min():.4f}, {images.max():.4f}]")
    print(f"  Label range: [{labels.min()}, {labels.max()}]")
    print(f"  Sample labels: {[class_names[l] for l in labels[:5].tolist()]}")
    print(f"\n[OK] Data pipeline verification complete!")

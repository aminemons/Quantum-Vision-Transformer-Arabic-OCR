import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import kagglehub
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


OTHMANIC_TRANSFORM = A.Compose([
    A.ElasticTransform(alpha=80, sigma=8, p=0.8),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.7),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=20, p=0.8),
    A.CoarseDropout(max_holes=6, max_height=4, max_width=4, min_holes=2, p=0.6),
    A.GaussNoise(var_limit=(5.0, 30.0), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
])

TASHKEEL_TRANSFORM = A.Compose([
    A.ElasticTransform(alpha=120, sigma=12, p=0.9),
    A.GridDistortion(num_steps=7, distort_limit=0.5, p=0.8),
    A.CoarseDropout(max_holes=10, max_height=3, max_width=3, min_holes=4, p=0.8),
    A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=30, p=0.9),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.7),
    A.Blur(blur_limit=3, p=0.4),
])

CLEAN_TRANSFORM = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
    A.GaussNoise(var_limit=(2.0, 10.0), p=0.3),
])


class ArabicMorphologyDataset(Dataset):
    def __init__(self, images, labels, img_size=16, mode="clean"):
        self.images = images
        self.labels = labels
        self.img_size = img_size
        self.mode = mode

        if mode == "othmanic":
            self.transform = OTHMANIC_TRANSFORM
        elif mode == "tashkeel":
            self.transform = TASHKEEL_TRANSFORM
        else:
            self.transform = CLEAN_TRANSFORM

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if img.ndim == 1:
            size = int(np.sqrt(len(img)))
            img = img.reshape(size, size)

        img = cv2.resize(img.astype(np.float32), (self.img_size, self.img_size))
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        augmented = self.transform(image=img_rgb)["image"]
        gray = cv2.cvtColor(augmented, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        tensor = torch.tensor(gray, dtype=torch.float32).unsqueeze(0)
        return tensor, int(self.labels[idx])


def load_ahcd_raw():
    path = kagglehub.dataset_download("mloey1/ahcd1")
    import pandas as pd

    train_images_path = None
    train_labels_path = None
    test_images_path = None
    test_labels_path = None

    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if "TrainImages" in f and f.endswith(".csv"):
                train_images_path = fp
            elif "TrainLabel" in f and f.endswith(".csv"):
                train_labels_path = fp
            elif "TestImages" in f and f.endswith(".csv"):
                test_images_path = fp
            elif "TestLabel" in f and f.endswith(".csv"):
                test_labels_path = fp

    train_images = pd.read_csv(train_images_path, header=None).values.astype(np.float32) / 255.0
    train_labels = pd.read_csv(train_labels_path, header=None).values.ravel().astype(np.int64) - 1
    test_images = pd.read_csv(test_images_path, header=None).values.astype(np.float32) / 255.0
    test_labels = pd.read_csv(test_labels_path, header=None).values.ravel().astype(np.int64) - 1

    return train_images, train_labels, test_images, test_labels


def load_hijja_raw():
    try:
        path = kagglehub.dataset_download("islamghazy/hijja-arabic-handwritten-letters-dataset")
        images, labels = [], []
        class_map = {}
        class_idx = 0
        for root, dirs, files in os.walk(path):
            for f in sorted(files):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    class_name = os.path.basename(root)
                    if class_name not in class_map:
                        class_map[class_name] = class_idx
                        class_idx += 1
                    img = cv2.imread(os.path.join(root, f), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        images.append(img.flatten().astype(np.float32) / 255.0)
                        labels.append(class_map[class_name])
        if len(images) > 0:
            return np.array(images), np.array(labels, dtype=np.int64)
    except Exception as e:
        print(f"[WARN] Hijja dataset unavailable: {e}")
    return None, None


def build_loaders(mode="clean", img_size=16, batch_size=128, seed=42):
    print(f"[dataset_engine] Loading AHCD dataset...")
    train_imgs, train_lbls, test_imgs, test_lbls = load_ahcd_raw()

    hijja_imgs, hijja_lbls = load_hijja_raw()
    if hijja_imgs is not None:
        max_ahcd_class = int(train_lbls.max()) + 1
        hijja_lbls_shifted = hijja_lbls + max_ahcd_class
        offset_lbls = int(hijja_lbls.max()) + 1
        hijja_split = int(len(hijja_imgs) * 0.85)
        train_imgs = np.concatenate([train_imgs, hijja_imgs[:hijja_split]], axis=0)
        train_lbls = np.concatenate([train_lbls, hijja_lbls_shifted[:hijja_split]], axis=0)
        test_imgs = np.concatenate([test_imgs, hijja_imgs[hijja_split:]], axis=0)
        test_lbls = np.concatenate([test_lbls, hijja_lbls_shifted[hijja_split:]], axis=0)
        print(f"[dataset_engine] Merged Hijja dataset. Total classes: {len(np.unique(train_lbls))}")

    n_classes = int(train_lbls.max()) + 1
    print(f"[dataset_engine] Mode={mode}, img_size={img_size}, classes={n_classes}")

    train_dataset = ArabicMorphologyDataset(train_imgs, train_lbls, img_size=img_size, mode=mode)
    test_dataset = ArabicMorphologyDataset(test_imgs, test_lbls, img_size=img_size, mode=mode)

    val_size = int(len(train_dataset) * 0.15)
    train_size = len(train_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader, n_classes


if __name__ == "__main__":
    for mode in ["clean", "othmanic", "tashkeel"]:
        tl, vl, testl, nc = build_loaders(mode=mode, img_size=16, batch_size=64)
        batch = next(iter(tl))
        print(f"  [{mode}] batch shape={batch[0].shape}, classes={nc}")

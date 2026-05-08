"""
Data Diagnostic Script
Run this BEFORE train.py to verify the HMBD-v1 dataset is correctly structured.
Usage: python check_data.py
"""
import os
import numpy as np
from collections import Counter
from torchvision import datasets, transforms
from PIL import Image

DATA_DIR = "./data/hmbd-v1"

def is_valid(path):
    try:
        img = Image.open(path)
        img.verify()
        return True
    except Exception:
        return False

def main():
    actual_dir = os.path.join(DATA_DIR, "Dataset") if os.path.exists(os.path.join(DATA_DIR, "Dataset")) else DATA_DIR
    
    print(f"\n{'='*50}")
    print(f"  HMBD-v1 Dataset Diagnostic")
    print(f"{'='*50}")
    print(f"  Scanning: {actual_dir}")

    if not os.path.exists(actual_dir):
        print(f"\n[ERROR] Directory not found: {actual_dir}")
        print("  -> Please download the dataset: kaggle datasets download -d hossammbalaha/hmbd-v1 -p ./data/hmbd-v1 --unzip")
        return

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((16, 16)),
        transforms.ToTensor()
    ])

    print("\n  [1/4] Scanning for valid images (this may take a moment)...")
    dataset = datasets.ImageFolder(root=actual_dir, transform=transform, is_valid_file=is_valid)
    
    total_files = sum([len(f) for _, _, f in os.walk(actual_dir)])
    total_valid = len(dataset)
    
    print(f"\n  [2/4] Structure Summary")
    print(f"  {'Total folders (raw):':<30} {len(os.listdir(actual_dir))}")
    print(f"  {'Classes recognized:':<30} {len(dataset.classes)}")
    print(f"  {'Total valid images:':<30} {total_valid}")
    print(f"  {'Skipped/corrupt files:':<30} {total_files - total_valid}")

    # Samples per class
    label_counts = Counter(label for _, label in dataset.samples)
    counts = list(label_counts.values())
    
    print(f"\n  [3/4] Samples per Class")
    print(f"  {'Min samples in a class:':<30} {min(counts)}")
    print(f"  {'Max samples in a class:':<30} {max(counts)}")
    print(f"  {'Mean samples per class:':<30} {np.mean(counts):.1f}")
    print(f"  {'Std dev:':<30} {np.std(counts):.1f}")

    # Classes with very few samples
    low_count_classes = [(dataset.classes[i], c) for i, c in label_counts.items() if c < 300]
    if low_count_classes:
        print(f"\n  [WARNING] {len(low_count_classes)} classes have < 300 samples:")
        for cls, cnt in sorted(low_count_classes, key=lambda x: x[1])[:10]:
            print(f"    {cls}: {cnt} samples")
    else:
        print(f"\n  [OK] All classes have >= 300 samples")

    # Quick pixel stats
    print(f"\n  [4/4] Pixel Value Sanity Check (sampling 500 images)...")
    import torch
    sample_indices = np.random.choice(len(dataset), min(500, len(dataset)), replace=False)
    pixels = []
    for i in sample_indices:
        img, _ = dataset[i]
        pixels.append(img.numpy().flatten())
    pixels = np.concatenate(pixels)
    
    print(f"  {'Pixel min:':<30} {pixels.min():.4f}")
    print(f"  {'Pixel max:':<30} {pixels.max():.4f}")
    print(f"  {'Pixel mean:':<30} {pixels.mean():.4f}")
    print(f"  {'Pixel std:':<30} {pixels.std():.4f}")

    # Final verdict
    print(f"\n{'='*50}")
    ok = len(dataset.classes) >= 100 and min(counts) >= 50
    if ok:
        print(f"  [VERDICT] Dataset looks GOOD. Ready to train.")
    else:
        print(f"  [VERDICT] Dataset has issues. Check warnings above.")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()

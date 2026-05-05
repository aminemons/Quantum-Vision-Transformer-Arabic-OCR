"""
Classical Convolutional Neural Network Baseline.

A standard PyTorch CNN used as a baseline benchmark against the
Hybrid Quantum-Classical Vision Transformer.

Designed for 16x16 or 32x32 input images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalCNN(nn.Module):
    """
    Standard Classical CNN for Arabic Character Recognition.

    Architecture:
    Conv2d -> ReLU -> MaxPool2d -> Conv2d -> ReLU -> MaxPool2d -> FC -> FC
    """
    def __init__(self, img_size: int = 16, n_classes: int = 28, dropout_rate: float = 0.3):
        super().__init__()
        self.img_size = img_size

        # Convolutional Feature Extraction
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16x16 -> 8x8

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 8x8 -> 4x4
        )

        # Calculate flattened dimension
        flat_size = 32 * (img_size // 4) * (img_size // 4)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        """
        Args:
            x: tensor of shape (batch, img_size*img_size)
        """
        # Reshape flat 1D images back to 2D for convolutions
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, self.img_size, self.img_size)

        x = self.features(x)
        logits = self.classifier(x)
        return logits

def create_cnn_baseline(n_classes: int = 28, img_size: int = 16, device: str = 'cpu') -> ClassicalCNN:
    model = ClassicalCNN(img_size=img_size, n_classes=n_classes).to(device)
    return model

if __name__ == "__main__":
    print("=" * 60)
    print(" Classical CNN Baseline -- Verification")
    print("=" * 60)

    model = create_cnn_baseline(img_size=16)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[PARAMS] Total: {total_params}")

    x = torch.randn(4, 256)  # 16x16 = 256
    logits = model(x)
    print(f"[FORWARD] {x.shape} -> {logits.shape}")
    print("[OK] Classical CNN ready.")

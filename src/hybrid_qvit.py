"""
High-Performance Hybrid Quantum-Classical Vision Transformer.

Upgraded Architecture for >90% Accuracy:
- Classical CNN Backbone: Efficiently processes high-res 16x16 images,
  extracting powerful spatial features while reducing dimensionality
  to a 2x2 spatial grid (4 tokens).
- Quantum Self-Attention: Operates on the 4 classical tokens to mix
  global context using our hybrid quantum-classical attention layer.
- Classical Classification Head: Outputs final 28 class logits.

This architecture shifts the heavy "pixel-level" lifting to classical
convolutions, preserving quantum simulation speed while enabling
training on full datasets to achieve high accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantum_attention import QuantumSelfAttentionOptimized

class HybridCNNQViT(nn.Module):
    """
    CNN-Quantum Vision Transformer.

    Args:
        img_size: input image size (must be 16 or 32)
        n_classes: number of output classes (28)
        classifier_hidden: hidden dimension in classifier head
        dropout_rate: dropout probability
    """

    def __init__(self, img_size: int = 16, n_classes: int = 28,
                 classifier_hidden: int = 128, dropout_rate: float = 0.3):
        super().__init__()
        self.img_size = img_size
        self.n_classes = n_classes

        # -- Classical CNN Backbone --
        # Goal: Reduce spatial dimensions to exactly 2x2 (4 tokens)
        # For 16x16 input: 16 -> 8 -> 4 -> 2
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 4x4
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 2x2
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # After backbone, we have (batch, 64, 2, 2).
        # We flatten spatial dims: (batch, 64, 4). Transpose to (batch, 4, 64).
        self.n_tokens = 4
        self.backbone_out_dim = 64
        self.token_dim = 4  # Target dim for quantum attention

        # Project CNN features to quantum token dimension
        self.token_projection = nn.Linear(self.backbone_out_dim, self.token_dim)

        # -- Hybrid Quantum-Classical Self-Attention --
        self.quantum_attention = QuantumSelfAttentionOptimized(
            token_dim=self.token_dim,
            n_tokens=self.n_tokens,
            ff_dim=32,
        )

        # -- Classical Classification Head --
        self.flat_dim = self.n_tokens * self.token_dim  # 4 x 4 = 16

        self.classifier = nn.Sequential(
            nn.Linear(self.flat_dim, classifier_hidden),
            nn.BatchNorm1d(classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(classifier_hidden, classifier_hidden // 2),
            nn.BatchNorm1d(classifier_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(classifier_hidden // 2, n_classes),
        )

        # Store attention weights for visualization
        self._attention_weights = None

    def forward(self, x):
        """
        Args:
            x: tensor of shape (batch, img_size*img_size) -- flattened image

        Returns:
            logits: (batch, n_classes)
        """
        batch_size = x.shape[0]

        # Step 1: Classical CNN Backbone
        # Reshape flat 1D images back to 2D
        x_2d = x.view(batch_size, 1, self.img_size, self.img_size)
        cnn_features = self.backbone(x_2d)  # (batch, 64, 2, 2)

        # Step 2: Extract sequence of tokens
        # Flatten spatial dims: (batch, 64, 4) -> Transpose -> (batch, 4, 64)
        tokens_64d = cnn_features.view(batch_size, self.backbone_out_dim, -1).transpose(1, 2)

        # Project down to token_dim (4) for quantum circuits
        tokens = self.token_projection(tokens_64d)  # (batch, 4, 4)

        # Step 3: Hybrid Quantum-Classical Attention
        attended_tokens, attn_weights = self.quantum_attention(tokens)
        self._attention_weights = attn_weights.detach()

        # Step 4: Flatten
        flat = attended_tokens.reshape(batch_size, -1)  # (batch, 16)

        # Step 5: Classify
        logits = self.classifier(flat)
        return logits

    def get_attention_weights(self):
        """Return last computed attention weights for visualization."""
        return self._attention_weights

    def count_parameters(self):
        """Count total, quantum, and classical parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        attn_quantum = sum(
            p.numel() for p in self.quantum_attention.attention.quantum_attn.parameters()
        )

        return {
            'total': total,
            'trainable': trainable,
            'quantum': attn_quantum,
            'classical': total - attn_quantum,
            'attention_quantum': attn_quantum,
            'qcnn_quantum': 0  # Replaced by CNN backbone
        }

def create_model(n_classes: int = 28, img_size: int = 16,
                 device: str = 'cpu') -> HybridCNNQViT:
    """Factory: create and initialize the CNN-Quantum model."""
    model = HybridCNNQViT(
        img_size=img_size, n_classes=n_classes,
        classifier_hidden=128, dropout_rate=0.3,
    ).to(device)

    # Xavier init for classical weights
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            if 'quantum' not in name and 'norm' not in name:
                nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            if 'quantum' not in name:
                nn.init.zeros_(param)

    return model

if __name__ == "__main__":
    print("=" * 60)
    print(" High-Performance Hybrid CNN-QViT -- Architecture Verification")
    print("=" * 60)

    model = create_model(n_classes=28, img_size=16)
    params = model.count_parameters()
    print(f"\n[PARAMETERS]")
    print(f"  Total:          {params['total']}")
    print(f"  Quantum:        {params['quantum']}")
    print(f"  Classical:      {params['classical']}")

    x = torch.randn(4, 256)  # 16x16 flattened
    logits = model(x)
    print(f"\n[FORWARD] {x.shape} -> {logits.shape}")
    print(f"  Logit range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")

    loss = F.cross_entropy(logits, torch.randint(0, 28, (4,)))
    loss.backward()

    print(f"\n[OK] Hybrid CNN-QViT verification complete!")

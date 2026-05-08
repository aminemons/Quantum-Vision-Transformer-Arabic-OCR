import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


# ─────────────────────────────────────────────────────────────────────────────
# Model A: Classical CNN (Baseline)
#   Input: (B, 256) flattened 16×16 pixel values
#   Full BatchNorm + residual-style architecture for 115 classes
# ─────────────────────────────────────────────────────────────────────────────
class ClassicalCNN(nn.Module):
    def __init__(self, num_classes=115):
        super().__init__()
        # Treat input as (B, 1, 16, 16)
        self.features = nn.Sequential(
            # Block 1: 16×16 → 8×8
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),           # 8×8
            nn.Dropout2d(0.1),

            # Block 2: 8×8 → 4×4
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),           # 4×4
            nn.Dropout2d(0.1),

            # Block 3: 4×4 → 2×2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),           # 2×2
        )
        self.head = nn.Sequential(
            nn.Flatten(),              # 128×2×2 = 512
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.view(-1, 1, 16, 16)
        return self.head(self.features(x))


# ─────────────────────────────────────────────────────────────────────────────
# Trainable Quanvolution (shared backbone for HybridQNN)
#   Uses 4-qubit StronglyEntanglingLayers with backprop diff
#   Input patches: (N, 4) where N = B × H_patches × W_patches
# ─────────────────────────────────────────────────────────────────────────────
class TrainableQuanvolution(nn.Module):
    def __init__(self):
        super().__init__()
        dev = qml.device("default.qubit", wires=4)
        weight_shapes = {"weights": (2, 4, 3)}   # 2 entangling layers

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(4))
            qml.StronglyEntanglingLayers(weights, wires=range(4))
            return [qml.expval(qml.PauliZ(i)) for i in range(4)]

        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, 1, 16, 16)
        patches = F.unfold(x, kernel_size=2, stride=2)      # (B, 4, 64)
        patches = patches.transpose(1, 2).reshape(-1, 4)    # (B*64, 4)
        out = self.qlayer(patches)                           # (B*64, 4)
        out = out.view(B, 64, 4).transpose(1, 2)            # (B, 4, 64)
        return out.reshape(B, 4, 8, 8)                      # (B, 4, 8, 8)


# ─────────────────────────────────────────────────────────────────────────────
# Model B: Hybrid QNN
#   Quantum convolutional feature extractor + classical deep head
# ─────────────────────────────────────────────────────────────────────────────
class HybridQNN(nn.Module):
    def __init__(self, num_classes=115):
        super().__init__()
        self.qconv = TrainableQuanvolution()
        # Quantum output: (B, 4, 8, 8) → classical head
        self.head = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(2),           # (B, 16, 4, 4)
            nn.Flatten(),              # 256
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.head(self.qconv(x))


# ─────────────────────────────────────────────────────────────────────────────
# Model C: Multi-Class QCNN
#   Pure quantum: AmplitudeEmbedding into 8-qubit system
#   25-layer deep entanglement + classical readout head
# ─────────────────────────────────────────────────────────────────────────────
class MultiClassQCNN(nn.Module):
    def __init__(self, num_classes=115, num_layers=25):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        dev = qml.device("default.qubit", wires=8)

        weight_shapes = {
            "f1_weights": (num_layers, 8, 2),
            "f2_weights": (num_layers, 8, 15),
            "pool_weights": (2,)
        }

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, f1_weights, f2_weights, pool_weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(8), normalize=True)
            for layer in range(self.num_layers):
                for i in range(8):
                    qml.RY(f1_weights[layer, i, 0], wires=i)
                    qml.RX(f1_weights[layer, i, 1], wires=i)
                for i in range(8):
                    qml.ArbitraryUnitary(f2_weights[layer, i], wires=[i, (i + 1) % 8])
            qml.CRZ(pool_weights[0], wires=[0, 1])
            qml.CRX(pool_weights[1], wires=[0, 1])
            return qml.probs(wires=range(1, 8))   # 2^7 = 128 outputs

        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        # Classical readout head to map 128 → num_classes
        self.readout = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        probs = self.qlayer(x)    # (B, 128)
        return self.readout(probs)

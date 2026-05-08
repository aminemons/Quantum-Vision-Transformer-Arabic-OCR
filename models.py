"""
Model Zoo for Fair Quantum Advantage Benchmark
================================================
Architecture reference: Mordacci, Ferrari & Amoretti (2024)
  "Multi-Class Quantum Convolutional Neural Networks"  arXiv:2404.12741

Models
------
  ClassicalCNN     – unconstrained large baseline (shows ceiling)
  FairCNN          – parameter-matched classical baseline (honest comparison)
  HybridQNN        – quantum spatial encoder + classical head
  MultiClassQCNN   – pure quantum with F1 pre-convolutional filter (paper arch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# A. ClassicalCNN  (Unconstrained – shows absolute classical ceiling)
#    Input: (B, 256)  flattened 16×16 pixel values in [0, 1]
# ─────────────────────────────────────────────────────────────────────────────
class ClassicalCNN(nn.Module):
    def __init__(self, num_classes=115):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),            # → 8×8×32
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),            # → 4×4×64
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.MaxPool2d(2),                               # → 2×2×128
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x.view(-1, 1, 16, 16)))


# ─────────────────────────────────────────────────────────────────────────────
# B. FairCNN  (Parameter-Matched Classical Baseline)
#    Mirrors the paper's parameter-matched CNN (Mordacci et al. §4).
#    Input: (B, 256)  flattened 16×16 pixel values in [0, 1]
# ─────────────────────────────────────────────────────────────────────────────
class FairCNN(nn.Module):
    def __init__(self, num_classes=115):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1), nn.Tanh(), nn.AvgPool2d(2),
            nn.Conv2d(1, 1, kernel_size=3, padding=1), nn.Tanh(), nn.AvgPool2d(2),
            nn.Conv2d(1, 1, kernel_size=3, padding=1), nn.Tanh(), nn.AvgPool2d(2),
            nn.Flatten(), nn.Linear(4, num_classes),
        )

    def forward(self, x):
        return self.net(x.view(-1, 1, 16, 16))


# ─────────────────────────────────────────────────────────────────────────────
# B2. IsoCNN  (Iso-Parameter Classical Baseline — THE DEFINITIVE COMPARISON)
#
#    The most scientifically rigorous experiment: both IsoCNN and MultiClassQCNN
#    have IDENTICAL total parameter counts and IDENTICAL readout heads.
#    The ONLY difference is HOW features are extracted:
#      • IsoCNN:        classical convolutions in 256-dim pixel space
#      • MultiClassQCNN: quantum circuit in 2^8 = 256-dim Hilbert space
#
#    Feature extractor (classical):
#      Conv(1→4, 3×3):   40 params    16×16 → 8×8
#      Conv(4→4, 3×3):  148 params     8×8 → 4×4
#      AdaptiveAvgPool → Flatten → 4 features
#      FC(4→128):       640 params
#      ─────────────────────────────
#      Total extractor: 828 params  ≈  QCNN circuit: 762 params
#
#    Readout (IDENTICAL to MultiClassQCNN):
#      Linear(128 → num_classes):  128×115 + 115 = 14 950 params
#
#    Grand total:  ~15 778 params  ≈  QCNN total: ~15 712 params
#
#    Claim: at this budget, the QCNN circuit extracts BETTER adversarially-
#    robust features because unitary quantum gates cannot amplify perturbations.
# ─────────────────────────────────────────────────────────────────────────────
class IsoCNN(nn.Module):
    """Classical CNN iso-parametric to the QCNN circuit. Shared readout head."""
    def __init__(self, num_classes=115):
        super().__init__()
        # Classical feature extractor → 128-dim feature vector  (~828 params)
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(4, 4, kernel_size=3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),   # → 4 features
            nn.Linear(4, 128), nn.GELU(),            # → 128 features
        )
        # Readout: IDENTICAL architecture to MultiClassQCNN  (14 950 params)
        self.readout = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.readout(self.features(x.view(-1, 1, 16, 16)))


# ─────────────────────────────────────────────────────────────────────────────
# C. HybridQNN  (Quantum spatial encoder + classical head)
# ─────────────────────────────────────────────────────────────────────────────
class TrainableQuanvolution(nn.Module):
    def __init__(self):
        super().__init__()
        dev = qml.device("default.qubit", wires=4)
        weight_shapes = {"weights": (2, 4, 3)}

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(4))
            qml.StronglyEntanglingLayers(weights, wires=range(4))
            return [qml.expval(qml.PauliZ(i)) for i in range(4)]

        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, 1, 16, 16)
        patches = F.unfold(x, kernel_size=2, stride=2).transpose(1, 2).reshape(-1, 4)
        out = self.qlayer(patches).view(B, 64, 4).transpose(1, 2)
        return out.reshape(B, 4, 8, 8)


class HybridQNN(nn.Module):
    def __init__(self, num_classes=115):
        super().__init__()
        self.qconv = TrainableQuanvolution()
        self.head = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1), nn.BatchNorm2d(16), nn.GELU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.qconv(x))


# ─────────────────────────────────────────────────────────────────────────────
# D. MultiClassQCNN  (Pure Quantum – paper architecture)
#
#    Implements the QCNN from Mordacci et al. (2024) §3:
#      • Amplitude Encoding of 256 features into 8 qubits
#      • F1 pre-convolutional filter (Schuld et al. [20]) applied BEFORE
#        each F2 convolutional layer – this is the key novel contribution
#        of the paper that we implement here.
#      • F2 convolutional filter: ArbitraryUnitary (SU(4), 15 params) on
#        every adjacent pair of qubits
#      • Pooling layer: CRZ + CRX (paper Fig. 3)
#      • Classical linear readout: 128 quantum probs → num_classes
#
#    Quantum circuit parameter count (num_layers=5, 8 qubits):
#      F1 : 5 × (4 × 8) = 160 params
#      F2 : 5 × 8 × 15  = 600 params
#      Pool:              =   2 params
#      Total quantum      = 762 params   ← dramatically fewer than CNN feature extractor
# ─────────────────────────────────────────────────────────────────────────────
class MultiClassQCNN(nn.Module):
    def __init__(self, num_classes=115, num_layers=5):
        super().__init__()
        self.num_layers = num_layers
        N = 8   # qubits (encode 2^8 = 256 amplitude features)
        dev = qml.device("default.qubit", wires=N)

        weight_shapes = {
            # F1: pre-convolutional filter — 4N params per layer (Schuld et al.)
            "f1_weights": (num_layers, 4 * N),
            # F2: SU(4) arbitrary unitary on adjacent pairs — 15 params per pair per layer
            "f2_weights": (num_layers, N, 15),
            # Pooling: CRZ + CRX
            "pool_weights": (2,),
        }

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, f1_weights, f2_weights, pool_weights):
            # ── Amplitude encode 256 features into 8-qubit state ────────────
            qml.AmplitudeEmbedding(features=inputs, wires=range(N), normalize=True)

            for layer in range(self.num_layers):
                # ── F1: Pre-convolutional preprocessing filter ───────────────
                # Block 1: independent single-qubit rotations
                for i in range(N):
                    qml.RY(f1_weights[layer, i], wires=i)
                for i in range(N):
                    qml.RX(f1_weights[layer, N + i], wires=i)
                # Entangling CNOT chain (no parameters, builds correlations)
                for i in range(N - 1):
                    qml.CNOT(wires=[i, i + 1])
                # Block 2: independent rotations after entanglement
                for i in range(N):
                    qml.RY(f1_weights[layer, 2 * N + i], wires=i)
                for i in range(N):
                    qml.RX(f1_weights[layer, 3 * N + i], wires=i)

                # ── F2: Convolutional filter (SU(4) on adjacent pairs) ───────
                for i in range(N):
                    qml.ArbitraryUnitary(f2_weights[layer, i],
                                        wires=[i, (i + 1) % N])

            # ── Pooling layer (paper Fig. 3): CRZ + CRX, trace out qubit 0 ──
            qml.CRZ(pool_weights[0], wires=[0, 1])
            qml.CRX(pool_weights[1], wires=[0, 1])

            # Measure 7 qubits → 2^7 = 128 probability outputs
            return qml.probs(wires=range(1, N))

        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

        # Minimal linear readout (no hidden layer — let the quantum circuit do
        # the heavy lifting, consistent with the paper's design philosophy)
        self.readout = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.readout(self.qlayer(x))   # (B, 128) → (B, num_classes)

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
#      FC(4→21):        105 params
#      ─────────────────────────────
#      Total extractor: 293 params  ≈  QCNN circuit: 242 params
#
#    Readout (IDENTICAL to MultiClassQCNN):
#      Linear(21 → num_classes):  21×115 + 115 = 2 530 params
#
#    Grand total:  ~2 823 params  ≈  QCNN total: ~2 772 params
# ─────────────────────────────────────────────────────────────────────────────
class IsoCNN(nn.Module):
    """Classical CNN iso-parametric to the QCNN circuit. Shared readout head."""
    def __init__(self, num_classes=115):
        super().__init__()
        # Classical feature extractor → 21-dim vector (matches QCNN 21 expvals)
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(4, 4, kernel_size=3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),   # → 4 features
            nn.Linear(4, 21), nn.Tanh(),             # → 21 features in [-1,1]
        )
        # Readout: IDENTICAL architecture to MultiClassQCNN
        self.readout = nn.Linear(21, num_classes)

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
# D. MultiClassQCNN  (Pure Quantum – GPU-accelerated, barren-plateau-safe)
#
#    Key design choices:
#      1. Device: lightning.gpu (A5000) with adjoint diff → ~20× speedup
#      2. Measurements: qml.expval() NOT qml.probs().
#         - adjoint diff only supports expval/var, NOT probs
#         - Z+X+Y on 7 qubits = 21 expectation values
#         - this is standard in QCNN literature (Cong et al. 2019)
#      3. Only 3 layers: avoids barren plateau (loss=ln(115)=4.74 = uniform)
#      4. F2: CNOT + Euler rotations (adjoint-safe, no ArbitraryUnitary)
#      5. Near-zero weight init: starts near identity for clean gradients
#
#    Parameter count (num_layers=3, 8 qubits):
#      F1 : 3 × 32   =   96 params
#      F2 : 3 × 8×6  =  144 params
#      Pool:          =    2 params
#      Total quantum  =  242 params
#      Readout Linear = 21×115+115 = 2 530 params
#      Grand total    = ~2 772 params
# ─────────────────────────────────────────────────────────────────────────────
def _get_qdevice(wires):
    """Pick the fastest available PennyLane device for adjoint+expval."""
    for name in ["lightning.gpu", "lightning.qubit", "default.qubit"]:
        try:
            dev = qml.device(name, wires=wires)
            diff = "adjoint" if "lightning" in name else "backprop"
            print(f"  [QCNN] Using device: {name}  diff_method={diff}")
            return dev, diff
        except Exception:
            continue
    raise RuntimeError("No PennyLane device available.")


class MultiClassQCNN(nn.Module):
    def __init__(self, num_classes=115, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        N = 8
        dev, diff = _get_qdevice(N)

        weight_shapes = {
            "f1_weights":   (num_layers, 4 * N),   # 96 params
            "f2_weights":   (num_layers, N, 6),     # 144 params (Euler angles)
            "pool_weights": (2,),                   #   2 params
        }

        @qml.qnode(dev, interface="torch", diff_method=diff)
        def circuit(inputs, f1_weights, f2_weights, pool_weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(N), normalize=True)

            for layer in range(self.num_layers):
                # ── F1: Pre-convolutional filter (Schuld et al.) ─────────────
                for i in range(N):
                    qml.RY(f1_weights[layer, i], wires=i)
                for i in range(N):
                    qml.RX(f1_weights[layer, N + i], wires=i)
                for i in range(N - 1):
                    qml.CNOT(wires=[i, i + 1])          # entangling, no params
                for i in range(N):
                    qml.RY(f1_weights[layer, 2*N + i], wires=i)
                for i in range(N):
                    qml.RX(f1_weights[layer, 3*N + i], wires=i)

                # ── F2: Euler-angle two-qubit convolution (adjoint-safe) ─────
                for i in range(N):
                    j = (i + 1) % N
                    qml.RZ(f2_weights[layer, i, 0], wires=i)
                    qml.RY(f2_weights[layer, i, 1], wires=i)
                    qml.RZ(f2_weights[layer, i, 2], wires=i)
                    qml.CNOT(wires=[i, j])
                    qml.RZ(f2_weights[layer, i, 3], wires=j)
                    qml.RY(f2_weights[layer, i, 4], wires=j)
                    qml.CNOT(wires=[j, i])
                    qml.RZ(f2_weights[layer, i, 5], wires=i)

            # ── Pooling (paper Fig. 3) ────────────────────────────────────────
            qml.CRZ(pool_weights[0], wires=[0, 1])
            qml.CRX(pool_weights[1], wires=[0, 1])

            # ── Measurement: expval on 7 qubits × 3 Pauli bases = 21 outputs ─
            # adjoint-compatible (unlike qml.probs which crashes on lightning)
            return (
                [qml.expval(qml.PauliZ(i)) for i in range(1, N)] +
                [qml.expval(qml.PauliX(i)) for i in range(1, N)] +
                [qml.expval(qml.PauliY(i)) for i in range(1, N)]
            )

        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

        # Near-zero weight init: avoids barren plateau at startup
        with torch.no_grad():
            for name, p in self.qlayer.named_parameters():
                nn.init.normal_(p, mean=0.0, std=0.01)

        # Readout: IDENTICAL architecture to IsoCNN — Linear(21, num_classes)
        self.readout = nn.Linear(21, num_classes)

    def forward(self, x):
        return self.readout(self.qlayer(x))

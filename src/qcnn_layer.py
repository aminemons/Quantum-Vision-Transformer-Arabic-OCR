"""
Quantum Convolutional Neural Network (QCNN) Feature Extractor.

Implements a trainable quantum convolutional layer using PennyLane
integrated with PyTorch via qml.qnn.TorchLayer. The circuit processes
image patches through angle embedding, parameterized convolutional
ansätze with data re-uploading, and quantum pooling.

Architecture:
    8x8 image -> 4 patches (4x4 each) -> PCA to 4 features per patch
    -> 4-qubit quantum circuit per patch -> 2 expectation values per patch
    -> 8-dimensional quantum feature vector

Theoretical Basis:
    The quantum convolutional filter operates in a 2^n = 16 dimensional
    Hilbert space (for n=4 qubits), allowing it to represent exponentially
    richer feature correlations than a classical filter of equivalent
    parameter count. The data re-uploading scheme ensures universal
    approximation capability (Pérez-Salinas et al., 2020).

Reference:
    Cong, I., Choi, S., & Lukin, M. D. (2019).
    Quantum convolutional neural networks. Nature Physics.
"""

import torch
import torch.nn as nn
import numpy as np
import pennylane as qml


# ----------------------------------------------
# Quantum Circuit Definition
# ----------------------------------------------

N_QUBITS_CONV = 4      # Qubits per convolutional sub-circuit
N_CONV_LAYERS = 2       # Number of variational layers (with data re-uploading)
N_PATCHES = 4           # Number of patches from 8x8 image
PATCH_FEATURES = 4      # Features per patch fed into quantum circuit
FEATURES_PER_PATCH = 4  # Output features per patch (all 4 qubits measured)

# Create the quantum device
dev_conv = qml.device("default.qubit", wires=N_QUBITS_CONV)


def qcnn_conv_ansatz(weights, wire_list):
    """
    Parameterized quantum convolutional ansatz.

    Applies alternating single-qubit rotations (RY, RZ) and entangling
    CNOT gates in a brickwork pattern. This creates correlations between
    neighboring pixel encodings analogous to classical convolution.

    Args:
        weights: shape (n_qubits, 3) -- rotation angles for RY, RZ, RX
        wire_list: list of qubit wire indices
    """
    n = len(wire_list)
    for i, w in enumerate(wire_list):
        qml.RY(weights[i, 0], wires=w)
        qml.RZ(weights[i, 1], wires=w)

    # Entangling layer: linear chain
    for i in range(n - 1):
        qml.CNOT(wires=[wire_list[i], wire_list[i + 1]])

    # Second rotation layer for expressivity
    for i, w in enumerate(wire_list):
        qml.RX(weights[i, 2], wires=w)

    # Circular entanglement (connect last to first)
    qml.CNOT(wires=[wire_list[-1], wire_list[0]])


def quantum_pooling(weights_pool, wire_list):
    """
    Quantum pooling operation via controlled rotations and measurement.

    Pools information from qubits [2,3] into qubits [0,1] using
    controlled rotations, then we only measure qubits [0,1].
    This is analogous to classical 2x2 -> 1x1 pooling.

    Args:
        weights_pool: shape (2, 2) -- controlled rotation angles
        wire_list: list of qubit wire indices
    """
    # Controlled rotations: pool qubit 2->0 and 3->1
    qml.CRY(weights_pool[0, 0], wires=[wire_list[2], wire_list[0]])
    qml.CRZ(weights_pool[0, 1], wires=[wire_list[2], wire_list[0]])
    qml.CRY(weights_pool[1, 0], wires=[wire_list[3], wire_list[1]])
    qml.CRZ(weights_pool[1, 1], wires=[wire_list[3], wire_list[1]])


@qml.qnode(dev_conv, interface="torch", diff_method="backprop")
def qcnn_circuit(inputs, conv_weights_0, conv_weights_1, pool_weights):
    """
    Full QCNN circuit for a single image patch.

    Pipeline:
    1. Angle Embedding: encode patch features as rotation angles
    2. Conv Layer 0: parameterized ansatz
    3. Data Re-uploading: re-encode features (universal approximation)
    4. Conv Layer 1: second parameterized ansatz
    5. Quantum Pooling: pool 4 qubits -> 2 qubits
    6. Measurement: expectation values of PauliZ on pooled qubits

    Args:
        inputs: tensor of shape (PATCH_FEATURES,) -- normalized pixel features
        conv_weights_0: shape (N_QUBITS_CONV, 3) -- first conv layer weights
        conv_weights_1: shape (N_QUBITS_CONV, 3) -- second conv layer weights
        pool_weights: shape (2, 2) -- pooling layer weights

    Returns:
        list of 2 expectation values in [-1, 1]
    """
    wires = list(range(N_QUBITS_CONV))

    # Layer 1: Initial encoding + convolution
    qml.AngleEmbedding(inputs, wires=wires, rotation='Y')
    qcnn_conv_ansatz(conv_weights_0, wires)

    # Layer 2: Data re-uploading + second convolution
    qml.AngleEmbedding(inputs, wires=wires, rotation='Y')
    qcnn_conv_ansatz(conv_weights_1, wires)

    # Quantum pooling: 4 qubits -> 2 qubits
    quantum_pooling(pool_weights, wires)

    # Measure all 4 qubits for richer feature representation
    return [qml.expval(qml.PauliZ(wires[0])),
            qml.expval(qml.PauliZ(wires[1])),
            qml.expval(qml.PauliZ(wires[2])),
            qml.expval(qml.PauliZ(wires[3]))]


# ----------------------------------------------
# PyTorch Module Wrapper
# ----------------------------------------------

class QCNNFeatureExtractor(nn.Module):
    """
    Quantum Convolutional Feature Extractor for image classification.

    Processes an 8x8 image by:
    1. Splitting into 4 non-overlapping 4x4 patches
    2. Projecting each patch to 4 features via learned linear transform
    3. Running each patch through a 4-qubit QCNN circuit
    4. Measuring all 4 qubits -> 4 outputs per patch -> 16D feature vector

    Args:
        img_size: input image size (default 8)
        n_patches: number of patches to extract (default 4 -> 2x2 grid)
        patch_features: features per patch after projection (default 4)
        features_per_patch: number of qubit measurements per patch (default 4)
    """

    def __init__(self, img_size: int = 8, n_patches: int = N_PATCHES,
                 patch_features: int = PATCH_FEATURES,
                 features_per_patch: int = FEATURES_PER_PATCH):
        super().__init__()
        self.img_size = img_size
        self.n_patches = n_patches
        self.patch_features = patch_features
        self.features_per_patch = features_per_patch
        self.patch_size = img_size // int(np.sqrt(n_patches))  # 4x4 patches
        self.pixels_per_patch = self.patch_size ** 2  # 16 pixels

        # Classical linear projection: 16 pixels -> 4 features per patch
        self.patch_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.pixels_per_patch, patch_features),
                nn.Tanh()  # Bound to [-1, 1] then scale to [0, pi]
            )
            for _ in range(n_patches)
        ])

        # Define weight shapes for TorchLayer
        weight_shapes = {
            "conv_weights_0": (N_QUBITS_CONV, 3),
            "conv_weights_1": (N_QUBITS_CONV, 3),
            "pool_weights": (2, 2),
        }

        # Create one quantum layer per patch (shared architecture, separate weights)
        self.quantum_layers = nn.ModuleList([
            qml.qnn.TorchLayer(qcnn_circuit, weight_shapes)
            for _ in range(n_patches)
        ])

        # Output dimension: 4 patches x 4 features = 16
        self.output_dim = n_patches * features_per_patch

    def _extract_patches(self, x):
        """
        Extract non-overlapping patches from flattened 8x8 images.

        Args:
            x: tensor of shape (batch, 64)

        Returns:
            list of 4 tensors, each shape (batch, 16)
        """
        batch_size = x.shape[0]
        # Reshape to 2D: (batch, 8, 8)
        x_2d = x.reshape(batch_size, self.img_size, self.img_size)

        patches = []
        ps = self.patch_size  # 4
        grid = int(np.sqrt(self.n_patches))  # 2

        for i in range(grid):
            for j in range(grid):
                patch = x_2d[:, i * ps:(i + 1) * ps, j * ps:(j + 1) * ps]
                patches.append(patch.reshape(batch_size, -1))

        return patches

    def forward(self, x):
        """
        Forward pass: image -> patch extraction -> projection -> QCNN -> features.

        Args:
            x: tensor of shape (batch, 64) -- flattened 8x8 images

        Returns:
            tensor of shape (batch, 8) -- quantum feature vectors
        """
        patches = self._extract_patches(x)  # list of (batch, 16) tensors
        quantum_features = []

        for patch_idx, (patch, proj, qlayer) in enumerate(
                zip(patches, self.patch_projections, self.quantum_layers)):

            # Project patch pixels to quantum-encodable features
            projected = proj(patch)  # (batch, 4)

            # Scale from [-1, 1] (tanh output) to [0, pi] for angle embedding
            projected = (projected + 1) * np.pi / 2  # -> [0, pi]

            # Run through quantum circuit (handles batch automatically)
            q_out = qlayer(projected)  # (batch, 2)
            quantum_features.append(q_out)

        # Concatenate all patch features: (batch, 8)
        return torch.cat(quantum_features, dim=1)


# ----------------------------------------------
# Standalone Test
# ----------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print(" QCNN Feature Extractor -- Standalone Verification")
    print("=" * 60)

    # Create the module
    qcnn = QCNNFeatureExtractor(img_size=8, n_patches=4)
    print(f"\n[ARCHITECTURE]")
    print(f"  Input:  (batch, 64) -- flattened 8x8 image")
    print(f"  Output: (batch, {qcnn.output_dim}) -- quantum feature vector")
    print(f"  Patches: {qcnn.n_patches} x {qcnn.patch_size}x{qcnn.patch_size}")
    print(f"  Qubits per patch: {N_QUBITS_CONV}")
    print(f"  Hilbert space dim: 2^{N_QUBITS_CONV} = {2**N_QUBITS_CONV}")

    # Count parameters
    total_params = sum(p.numel() for p in qcnn.parameters())
    quantum_params = sum(p.numel() for ql in qcnn.quantum_layers for p in ql.parameters())
    classical_params = total_params - quantum_params
    print(f"\n[PARAMETERS]")
    print(f"  Total: {total_params}")
    print(f"  Quantum: {quantum_params}")
    print(f"  Classical (projections): {classical_params}")

    # Forward pass test
    x = torch.randn(4, 64)  # batch of 4 images
    print(f"\n[FORWARD PASS TEST]")
    print(f"  Input shape:  {x.shape}")
    y = qcnn(x)
    print(f"  Output shape: {y.shape}")
    print(f"  Output range: [{y.min().item():.4f}, {y.max().item():.4f}]")
    print(f"  Output sample: {y[0].detach().numpy()}")

    # Gradient test
    loss = y.sum()
    loss.backward()
    grad_ok = all(p.grad is not None for p in qcnn.parameters() if p.requires_grad)
    print(f"\n[GRADIENT TEST]")
    print(f"  Gradients computed: {'OK' if grad_ok else 'FAIL'}")
    print(f"\n[OK] QCNN Feature Extractor verification complete!")

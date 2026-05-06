import torch
import torch.nn as nn
import numpy as np
import pennylane as qml


NUM_QUBITS = 8


def _get_device():
    try:
        dev = qml.device("lightning.qubit", wires=NUM_QUBITS)
        print("[mera_qcnn] Using lightning.qubit backend (CPU, adjoint diff)")
        return dev
    except Exception:
        print("[mera_qcnn] Falling back to default.qubit")
        return qml.device("default.qubit", wires=NUM_QUBITS)


dev = _get_device()


def fourier_reuploading_encoding(x, wires):
    n_feat = x.shape[-1]
    n_wires = len(wires)
    for layer in range(2):
        for i, w in enumerate(wires):
            fi = (layer * n_wires + i) % n_feat
            qml.RX(x[..., fi] * np.pi, wires=w)
            qml.RY(x[..., (fi + 1) % n_feat] * np.pi, wires=w)
            qml.RZ(x[..., (fi + 2) % n_feat] * np.pi, wires=w)


def mera_entangling_block(params, wires):
    n = len(wires)
    for i in range(0, n - 1, 2):
        qml.RY(params[i], wires=wires[i])
        qml.RY(params[i + 1], wires=wires[i + 1])
        qml.CRZ(params[n + i // 2], wires=[wires[i], wires[i + 1]])
    for i in range(1, n - 1, 2):
        qml.RY(params[i], wires=wires[i])
        qml.RY(params[i + 1], wires=wires[i + 1])
        qml.CRZ(params[n + i // 2], wires=[wires[i], wires[i + 1]])


def mera_pooling_block(params, wires_in, wires_out):
    for i, (w_in, w_out) in enumerate(zip(wires_in, wires_out)):
        qml.CRZ(params[i], wires=[w_in, w_out])
        qml.PauliX(wires=w_in)
        qml.CRZ(params[len(wires_in) + i], wires=[w_in, w_out])


@qml.qnode(dev, interface="torch", diff_method="adjoint")
def mera_circuit(inputs, entangle_params_0, pool_params_0,
                 entangle_params_1, pool_params_1,
                 entangle_params_2, final_params):
    all_wires = list(range(NUM_QUBITS))

    fourier_reuploading_encoding(inputs, all_wires)

    mera_entangling_block(entangle_params_0, all_wires)
    active_wires = all_wires[::2]
    mera_pooling_block(pool_params_0, all_wires[1::2], active_wires)

    mera_entangling_block(entangle_params_1, active_wires)
    next_wires = active_wires[::2]
    mera_pooling_block(pool_params_1, active_wires[1::2], next_wires)

    mera_entangling_block(entangle_params_2, next_wires)

    for i, w in enumerate(next_wires):
        qml.RY(final_params[i], wires=w)

    return [qml.expval(qml.PauliZ(w)) for w in next_wires]


def _build_weight_shapes(n_qubits=NUM_QUBITS):
    n0, p0 = n_qubits, n_qubits // 2
    n1, p1 = n_qubits // 2, n_qubits // 4
    n2 = n_qubits // 4
    return {
        "entangle_params_0": (n0 + n0 // 2,),
        "pool_params_0":     (p0 * 2,),
        "entangle_params_1": (n1 + n1 // 2,),
        "pool_params_1":     (p1 * 2,),
        "entangle_params_2": (n2 + n2 // 2,),
        "final_params":      (n2,),
    }


weight_shapes = _build_weight_shapes(NUM_QUBITS)
quantum_out_dim = NUM_QUBITS // 4


class FourierMERAQCNN(nn.Module):
    def __init__(self, img_size=16, n_classes=28, n_qubits=NUM_QUBITS):
        super().__init__()
        self.img_size = img_size
        self.n_qubits = n_qubits

        self.classical_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 8, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, n_qubits),
            nn.Tanh(),
        )

        self.qlayer = qml.qnn.TorchLayer(mera_circuit, weight_shapes)

        self.classifier = nn.Sequential(
            nn.Linear(quantum_out_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = x.view(-1, 1, self.img_size, self.img_size)
        encoded = self.classical_encoder(x)
        q_out = self.qlayer(encoded)
        return self.classifier(q_out)


def get_mera_qcnn(img_size=16, n_classes=28):
    return FourierMERAQCNN(img_size=img_size, n_classes=n_classes, n_qubits=NUM_QUBITS)

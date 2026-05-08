#!/usr/bin/env python3
"""
RUTHLESS EXECUTION: 1-WEEK BLUEPRINT FOR QUANTUM ADVANTAGE
==========================================================
3 Killer Experiments:
1. The "Parameter Crush" (Efficiency)
2. The "Noise Immunity" Test (Robustness against Gaussian Noise)
3. The "Few-Shot Data Scarcity" Test (10% Training Data)
"""

import os, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import pennylane as qml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ==========================================
# HYPERPARAMETERS & SETUP
# ==========================================
DATA_DIR  = "./data/hmbd-v1/Dataset"  # Using HMBD-v1 (115 classes) as the dataset
NC        = 115
BS        = 128
EPOCHS    = 15      # Ruthless fast execution
LR_CLASS  = 1e-3    # ResNet learning rate
LR_QUANT  = 1e-2    # QCNN learning rate (higher for quantum)
Q_FEATURES = 8      # Compress 512 ResNet features down to 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 RUTHLESS EXECUTION MODE INITIATED | Device: {device} 🔥\n")

# ==========================================
# DAY 1-2: DATA PIPELINE & TRANSFORMATIONS
# ==========================================
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

def get_dataloaders():
    print("Loading data and preparing Killer Experiment splits...")
    # ResNet expects 3 channels, 224x224 ideally, but we'll use 32x32 to keep it fast
    base_tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3), # Convert to 3 channel for ResNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    noise_tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        AddGaussianNoise(0., 0.5) # Heavy Gaussian Noise
    ])

    def is_valid(path):
        try:
            from PIL import Image
            img = Image.open(path)
            img.verify()
            return True
        except Exception:
            return False

    # Load dataset
    root = DATA_DIR if os.path.isdir(DATA_DIR) else "./data"
    full_ds = datasets.ImageFolder(root=root, transform=base_tf, is_valid_file=is_valid)
    noise_ds = datasets.ImageFolder(root=root, transform=noise_tf, is_valid_file=is_valid)
    
    # Train/Test Split (80/20)
    train_size = int(0.8 * len(full_ds))
    test_size = len(full_ds) - train_size
    train_ds, test_ds = torch.utils.data.random_split(full_ds, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    
    # Showcase 2: Noise Immunity Split (Using the same test indices but with noise transform)
    _, test_noise_ds = torch.utils.data.random_split(noise_ds, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    # Showcase 3: Few-Shot Data Scarcity (10% of Training Data)
    ten_percent_size = int(0.1 * len(train_ds))
    few_shot_ds, _ = torch.utils.data.random_split(train_ds, [ten_percent_size, len(train_ds) - ten_percent_size], generator=torch.Generator().manual_seed(42))

    loaders = {
        "train_100": DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=4),
        "train_10":  DataLoader(few_shot_ds, batch_size=BS, shuffle=True, num_workers=4),
        "test":      DataLoader(test_ds, batch_size=BS, shuffle=False, num_workers=4),
        "test_noise":DataLoader(test_noise_ds, batch_size=BS, shuffle=False, num_workers=4),
    }
    print(f"Data ready. 100% Train: {len(train_ds)} | 10% Train: {len(few_shot_ds)} | Test: {len(test_ds)}")
    return loaders

# ==========================================
# CLASSICAL RESNET-18 BASELINE
# ==========================================
def get_classical_resnet():
    # Pre-trained ResNet-18
    model = models.resnet18(pretrained=True)
    # Replace final layer for 115 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NC)
    return model

def count_parameters(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

# ==========================================
# DAY 3-4: THE QCNN IN PENNYLANE
# ==========================================
# Lightning fast default qubit for 8 qubits
dev = qml.device("default.qubit", wires=Q_FEATURES)

def conv_block(params, wires):
    """2-qubit unitary block using RY, RZ and CNOT"""
    qml.RY(params[0], wires=wires[0])
    qml.RZ(params[1], wires=wires[0])
    qml.RY(params[2], wires=wires[1])
    qml.RZ(params[3], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])

def pool_block(params, wires):
    """Measurement and Trace-out equivalent pooling"""
    qml.CRZ(params[0], wires=[wires[0], wires[1]])
    qml.PauliX(wires=wires[0])
    qml.CRX(params[1], wires=[wires[0], wires[1]])

@qml.qnode(dev, interface="torch", diff_method="backprop")
def qcnn_circuit(inputs, conv_params, pool_params):
    # 1. Encoding Layer: AngleEmbedding (Lightning fast, Maps 8 features -> 8 qubits)
    qml.AngleEmbedding(inputs, wires=range(Q_FEATURES), rotation='Y')
    
    # 2. Convolutional & Pooling Layers
    # Layer 1: 8 qubits -> 4 qubits
    for i in range(0, 8, 2):
        conv_block(conv_params[i//2], wires=[i, i+1])
    for i in range(0, 8, 2):
        pool_block(pool_params[i//2], wires=[i, i+1])
        
    # Layer 2: 4 qubits (wires 1,3,5,7) -> 2 qubits
    active_2 = [1, 3, 5, 7]
    for i in range(0, 4, 2):
        conv_block(conv_params[4 + i//2], wires=[active_2[i], active_2[i+1]])
    for i in range(0, 4, 2):
        pool_block(pool_params[4 + i//2], wires=[active_2[i], active_2[i+1]])
        
    # Layer 3: 2 qubits (wires 3,7) -> 1 qubit
    active_3 = [3, 7]
    conv_block(conv_params[6], wires=[active_3[0], active_3[1]])
    pool_block(pool_params[6], wires=[active_3[0], active_3[1]])
    
    # 3. Measurement: Pauli-Z expectation value of ALL 8 qubits to avoid 1D bottleneck
    return [qml.expval(qml.PauliZ(i)) for i in range(Q_FEATURES)]

# ==========================================
# DAY 5: HYBRID TRAINING LOOP (C2Q Transfer)
# ==========================================
class BlueprintHybridQCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Load Pre-trained ResNet-18
        resnet = models.resnet18(pretrained=True)
        # FREEZE ResNet-18 weights
        for param in resnet.parameters():
            param.requires_grad = False
            
        # Strip classification layer, get the 512 features
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # PCA / Dense Compression: 512 -> 8 features
        self.compressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, Q_FEATURES),
            nn.Tanh() # Scale for AngleEmbedding [-1, 1] usually preferred, AngleEmbedding wraps modulo 2pi
        )
        
        # PennyLane QCNN TorchLayer
        weight_shapes = {"conv_params": (7, 4), "pool_params": (7, 2)}
        self.qcnn = qml.qnn.TorchLayer(qcnn_circuit, weight_shapes)
        
        # Final output mapping from the 8 expectation values to NC classes
        self.classifier = nn.Linear(Q_FEATURES, NC)
        
    def forward(self, x):
        with torch.no_grad():
            f = self.feature_extractor(x)
        features_8d = self.compressor(f)
        q_out = self.qcnn(features_8d)
        # Ensure correct shape for linear layer
        if len(q_out.shape) == 1:
            q_out = q_out.unsqueeze(0)
        return self.classifier(q_out)

# ==========================================
# RUTHLESS TRAINING FUNCTION
# ==========================================
def train_model(model, train_loader, epochs, lr, is_quantum=False):
    model.to(device)
    # Only optimize trainable parameters
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    crit = nn.CrossEntropyLoss()
    
    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"  Epoch {ep+1}/{epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        print(f"  Epoch {ep+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f}")

def evaluate_model(model, test_loader):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# ==========================================
# DAY 6 & 7: KILLER EXPERIMENTS & PLOTS
# ==========================================
def run_killer_experiments():
    loaders = get_dataloaders()
    
    # ---------------------------------------------------------
    # SHOWCASE 1: Parameter Crush & 100% Data Training
    # ---------------------------------------------------------
    print("\n" + "="*50 + "\n SHOWCASE 1: THE PARAMETER CRUSH \n" + "="*50)
    classical_net = get_classical_resnet()
    hybrid_qcnn = BlueprintHybridQCNN()
    
    c_params = count_parameters(classical_net, trainable_only=True)
    q_params = count_parameters(hybrid_qcnn, trainable_only=True)
    
    print(f"  Classical ResNet-18 Params: {c_params:,}")
    print(f"  Hybrid QCNN Trainable Params: {q_params:,} (Classical backbone frozen!)")
    
    print("\n🚀 Training Classical ResNet-18 (100% Data)...")
    train_model(classical_net, loaders["train_100"], EPOCHS, LR_CLASS)
    class_100_acc = evaluate_model(classical_net, loaders["test"])
    
    print("\n🚀 Training Hybrid QCNN (100% Data)...")
    train_model(hybrid_qcnn, loaders["train_100"], EPOCHS, LR_QUANT, is_quantum=True)
    qcnn_100_acc = evaluate_model(hybrid_qcnn, loaders["test"])
    
    print(f"  [RESULT] Classical Acc: {class_100_acc*100:.2f}% | Hybrid QCNN Acc: {qcnn_100_acc*100:.2f}%")

    # ---------------------------------------------------------
    # SHOWCASE 2: Noise Immunity
    # ---------------------------------------------------------
    print("\n" + "="*50 + "\n SHOWCASE 2: NOISE IMMUNITY TEST \n" + "="*50)
    class_noise_acc = evaluate_model(classical_net, loaders["test_noise"])
    qcnn_noise_acc = evaluate_model(hybrid_qcnn, loaders["test_noise"])
    
    print(f"  [RESULT] Classical NOISE Acc: {class_noise_acc*100:.2f}% (Drop: {(class_100_acc-class_noise_acc)*100:.2f}%)")
    print(f"  [RESULT] Hybrid QCNN NOISE Acc: {qcnn_noise_acc*100:.2f}% (Drop: {(qcnn_100_acc-qcnn_noise_acc)*100:.2f}%)")

    # ---------------------------------------------------------
    # SHOWCASE 3: Few-Shot Data Scarcity (10% Data)
    # ---------------------------------------------------------
    print("\n" + "="*50 + "\n SHOWCASE 3: FEW-SHOT DATA SCARCITY (10%) \n" + "="*50)
    # Re-initialize fresh models to prevent data leakage
    classical_net_10 = get_classical_resnet()
    hybrid_qcnn_10 = BlueprintHybridQCNN()
    
    print("\n🚀 Training Classical ResNet-18 (10% Data)...")
    train_model(classical_net_10, loaders["train_10"], EPOCHS, LR_CLASS)
    class_10_acc = evaluate_model(classical_net_10, loaders["test"])
    
    print("\n🚀 Training Hybrid QCNN (10% Data)...")
    train_model(hybrid_qcnn_10, loaders["train_10"], EPOCHS, LR_QUANT, is_quantum=True)
    qcnn_10_acc = evaluate_model(hybrid_qcnn_10, loaders["test"])
    
    print(f"  [RESULT] Classical 10% Acc: {class_10_acc*100:.2f}%")
    print(f"  [RESULT] Hybrid QCNN 10% Acc: {qcnn_10_acc*100:.2f}%")

    # ---------------------------------------------------------
    # GENERATE PUBLICATION PLOTS
    # ---------------------------------------------------------
    print("\n" + "="*50 + "\n GENERATING KILLER VISUALIZATIONS \n" + "="*50)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Quantum Advantage in Arabic OCR: 3 Killer Experiments", fontsize=16, fontweight="bold")

    # Plot 1: Parameter Crush
    ax1 = axes[0]
    models = ['Classical ResNet-18', 'Hybrid QCNN']
    accs = [class_100_acc*100, qcnn_100_acc*100]
    params = [c_params, q_params]
    
    bars = ax1.bar(models, accs, color=['#E84A4A', '#5B8DB8'])
    ax1.set_ylabel('Clean Accuracy (%)', fontweight="bold")
    ax1.set_ylim(0, 100)
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(models, params, color='black', marker='D', markersize=8, linestyle='dashed', linewidth=2)
    ax1_twin.set_yscale('log')
    ax1_twin.set_ylabel('Trainable Parameters (Log Scale)', fontweight="bold")
    ax1.set_title("Showcase 1: Parameter Crush\nAccuracy vs. Parameter Count", fontweight="bold")

    # Plot 2: Noise Immunity
    ax2 = axes[1]
    labels = ['Clean Data', 'Gaussian Noise']
    c_noise = [class_100_acc*100, class_noise_acc*100]
    q_noise = [qcnn_100_acc*100, qcnn_noise_acc*100]
    
    ax2.plot(labels, c_noise, marker='o', label='Classical ResNet', color='#E84A4A', linewidth=3)
    ax2.plot(labels, q_noise, marker='s', label='Hybrid QCNN', color='#5B8DB8', linewidth=3)
    ax2.set_ylabel('Accuracy (%)', fontweight="bold")
    ax2.set_title("Showcase 2: Noise Immunity\nAccuracy Degradation under Noise", fontweight="bold")
    ax2.legend()

    # Plot 3: Few-Shot Scarcity
    ax3 = axes[2]
    x = np.arange(2)
    width = 0.35
    c_data = [class_100_acc*100, class_10_acc*100]
    q_data = [qcnn_100_acc*100, qcnn_10_acc*100]
    
    ax3.bar(x - width/2, c_data, width, label='Classical ResNet', color='#E84A4A')
    ax3.bar(x + width/2, q_data, width, label='Hybrid QCNN', color='#5B8DB8')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['100% Training Data', '10% Training Data'])
    ax3.set_ylabel('Accuracy (%)', fontweight="bold")
    ax3.set_title("Showcase 3: Few-Shot Learning\nPerformance in Data Scarcity", fontweight="bold")
    ax3.legend()

    plt.tight_layout()
    plt.savefig("killer_experiments_results.png", dpi=300)
    print("✅ Visualizations saved to 'killer_experiments_results.png'")
    print("🔥 BLUEPRINT EXECUTION COMPLETE 🔥")

if __name__ == "__main__":
    run_killer_experiments()

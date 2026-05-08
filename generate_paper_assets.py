import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from run_all import BlueprintHybridQCNN, get_classical_resnet, get_dataloaders, evaluate_model, AddGaussianNoise, device

def generate_assets():
    print("🎨 GENERATING PROFESSIONAL PAPER ASSETS...")
    loaders = get_dataloaders()
    NC = 115

    # 1. Load Trained Models
    classical_net = get_classical_resnet().to(device)
    classical_net.load_state_dict(torch.load("classical_resnet.pth", map_location=device))
    
    hybrid_qcnn = BlueprintHybridQCNN(base_model=classical_net).to(device)
    hybrid_qcnn.load_state_dict(torch.load("hybrid_qcnn.pth", map_location=device))
    
    classical_net.eval()
    hybrid_qcnn.eval()

    # 2. Confusion Matrix (Top 20 Classes for readability)
    print("   -> Generating Confusion Matrix...")
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loaders["test"]:
            x, y = x.to(device), y.to(device)
            outputs = hybrid_qcnn(x)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm[:20, :20], annot=True, fmt='d', cmap='Blues')
    plt.title("Hybrid QCNN Confusion Matrix (Top 20 Classes)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("confusion_matrix_qcnn.png", dpi=300)
    plt.close()

    # 3. Noise Robustness Sweep (0.0 to 0.3)
    print("   -> Generating Noise Robustness Sweep...")
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    c_accs, q_accs = [], []
    
    for nl in noise_levels:
        noise_tf = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            AddGaussianNoise(std=nl)
        ])
        # Re-load test set with current noise level
        root = "./data/hmbd-v1/Dataset"
        noise_ds = datasets.ImageFolder(root=root, transform=noise_tf)
        _, test_ds = torch.utils.data.random_split(noise_ds, [int(0.8*len(noise_ds)), len(noise_ds)-int(0.8*len(noise_ds))], generator=torch.Generator().manual_seed(42))
        loader = DataLoader(test_ds, batch_size=128, shuffle=False)
        
        c_accs.append(evaluate_model(classical_net, loader))
        q_accs.append(evaluate_model(hybrid_qcnn, loader))
        print(f"      Noise {nl}: Classical {c_accs[-1]:.4f}, QCNN {q_accs[-1]:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, c_accs, 'o-', label='Classical ResNet-18', color='red', linewidth=2)
    plt.plot(noise_levels, q_accs, 's-', label='Hybrid QCNN (Ours)', color='blue', linewidth=2)
    plt.fill_between(noise_levels, q_accs, c_accs, where=(np.array(q_accs) > np.array(c_accs)), color='green', alpha=0.2, label='Quantum Advantage Zone')
    plt.title("Noise Robustness Sweep")
    plt.xlabel("Gaussian Noise Std Dev")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("noise_robustness_sweep.png", dpi=300)
    plt.close()

    # 4. Parameter Efficiency Chart
    print("   -> Generating Parameter Efficiency Plot...")
    models_labels = ['Classical ResNet-18', 'Hybrid QCNN']
    params = [11235507, 8400177]
    accs = [c_accs[0], q_accs[0]]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(params, accs, s=[p/10000 for p in params], c=['red', 'blue'], alpha=0.6)
    for i, txt in enumerate(models_labels):
        plt.annotate(txt, (params[i], accs[i]), xytext=(10, 10), textcoords='offset points')
    
    plt.xscale('log')
    plt.title("Accuracy vs Trainable Parameters (Log Scale)")
    plt.xlabel("Trainable Parameters")
    plt.ylabel("Clean Accuracy")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig("parameter_efficiency.png", dpi=300)
    plt.close()

    print("\n✅ ALL ASSETS GENERATED: 'confusion_matrix_qcnn.png', 'noise_robustness_sweep.png', 'parameter_efficiency.png'")

if __name__ == "__main__":
    generate_assets()

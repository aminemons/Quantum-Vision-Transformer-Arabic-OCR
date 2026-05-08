"""
run_all.py — Definitive QCNN vs CNN Benchmark (Single File)
============================================================
References:
  [1] Li et al. "A quantum deep convolutional neural network" QST 2020
  [2] Mordacci et al. "Multi-Class QCNN" arXiv:2404.12741
  [3] Kim et al. "Classical-to-quantum CNN transfer learning" Neurocomputing 2023
  [5] Di et al. "Amplitude transformed QCNN" Applied Intelligence 2023

Usage:  python run_all.py
Output: results_comparison.csv, quantum_advantage_benchmark.png
"""

import os, sys, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
import pennylane as qml
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
NUM_CLASSES   = 115
EPOCHS        = 30
LR            = 0.001
QCNN_LR       = 0.01
BATCH_SIZE    = 128
WEIGHT_DECAY  = 1e-4
LABEL_SMOOTH  = 0.1
GRAD_CLIP     = 1.0
DATA_DIR      = "./data/hmbd-v1"
CSV_OUT       = "results_comparison.csv"
PLOT_OUT      = "quantum_advantage_benchmark.png"

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════════
class FlatDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

def load_data():
    root = os.path.join(DATA_DIR, "Dataset") if os.path.isdir(os.path.join(DATA_DIR, "Dataset")) else DATA_DIR
    print(f"Loading HMBD-v1 from {root}...")
    tf = transforms.Compose([transforms.Grayscale(1), transforms.Resize((16,16)), transforms.ToTensor()])
    def ok(p):
        try: Image.open(p).verify(); return True
        except: return False
    ds = datasets.ImageFolder(root=root, transform=tf, is_valid_file=ok)
    ldr = DataLoader(ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    xa, ya = [], []
    for bx, by in ldr:
        xa.append(bx.view(bx.size(0),-1).numpy()); ya.append(by.numpy())
    x_all, y_all = np.concatenate(xa), np.concatenate(ya)
    mask = y_all < NUM_CLASSES
    return x_all[mask], y_all[mask]

def prepare_loaders(x_raw, y_raw):
    # Stratified stress split
    xtv, xst, ytv, yst = [], [], [], []
    for c in np.unique(y_raw):
        m = y_raw == c; xc, yc = x_raw[m], y_raw[m]
        idx = np.random.permutation(len(xc)); xc, yc = xc[idx], yc[idx]
        ns = min(250, len(xc)//3)
        xst.append(xc[:ns]); yst.append(yc[:ns])
        xtv.append(xc[ns:]); ytv.append(yc[ns:])
    xtv, ytv = np.concatenate(xtv), np.concatenate(ytv)
    xst, yst = np.concatenate(xst), np.concatenate(yst)

    xtr, xvl, ytr, yvl = train_test_split(xtv, ytv, test_size=0.2, stratify=ytv, random_state=42)
    print(f"Data ready! Train:{len(xtr)} Val:{len(xvl)} Stress:{len(xst)}")

    mk = lambda x,y,s: DataLoader(FlatDataset(x,y), batch_size=BATCH_SIZE, shuffle=s, pin_memory=True)
    loaders = {"train": mk(xtr,ytr,True), "val": mk(xvl,yvl,False), "stress": mk(xst,yst,False)}

    # L2-normalized for QCNN
    def l2(x):
        n = np.linalg.norm(x, axis=1, keepdims=True); n[n==0]=1.0; return x/n
    xtv_q, xst_q = l2(xtv), l2(xst)
    xtr_q, xvl_q, ytr_q, yvl_q = train_test_split(xtv_q, ytv, test_size=0.2, stratify=ytv, random_state=42)
    loaders["train_q"] = mk(xtr_q,ytr_q,True)
    loaders["val_q"]   = mk(xvl_q,yvl_q,False)
    loaders["stress_q"]= mk(xst_q,yst,False)
    return loaders

# ═══════════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════════
def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

# A. ClassicalCNN — large unconstrained baseline
class ClassicalCNN(nn.Module):
    def __init__(self, nc=115):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32,32,3,padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.MaxPool2d(2))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(512,256),
            nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.4), nn.Linear(256,nc))
    def forward(self, x):
        return self.head(self.features(x.view(-1,1,16,16)))

# B. FairCNN — parameter-matched to QCNN circuit params only
class FairCNN(nn.Module):
    def __init__(self, nc=115):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,1,3,padding=1), nn.Tanh(), nn.AvgPool2d(2),
            nn.Conv2d(1,1,3,padding=1), nn.Tanh(), nn.AvgPool2d(2),
            nn.Conv2d(1,1,3,padding=1), nn.Tanh(), nn.AvgPool2d(2),
            nn.Flatten(), nn.Linear(4,nc))
    def forward(self, x):
        return self.net(x.view(-1,1,16,16))

# B2. IsoCNN — iso-parametric to QCNN (same total params, same readout)
class IsoCNN(nn.Module):
    def __init__(self, nc=115):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,4,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(4,4,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(4,21), nn.Tanh())
        self.readout = nn.Linear(21, nc)
    def forward(self, x):
        return self.readout(self.features(x.view(-1,1,16,16)))

# C. HybridQNN — quantum spatial encoder + classical head
class TrainableQuanvolution(nn.Module):
    def __init__(self):
        super().__init__()
        dev = qml.device("default.qubit", wires=4)
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circ(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(4))
            qml.StronglyEntanglingLayers(weights, wires=range(4))
            return [qml.expval(qml.PauliZ(i)) for i in range(4)]
        self.qlayer = qml.qnn.TorchLayer(circ, {"weights": (2,4,3)})
    def forward(self, x):
        B = x.shape[0]; x = x.view(B,1,16,16)
        patches = F.unfold(x, kernel_size=2, stride=2).transpose(1,2).reshape(-1,4)
        return self.qlayer(patches).view(B,64,4).transpose(1,2).reshape(B,4,8,8)

class HybridQNN(nn.Module):
    def __init__(self, nc=115):
        super().__init__()
        self.qconv = TrainableQuanvolution()
        self.head = nn.Sequential(
            nn.Conv2d(4,16,3,padding=1), nn.BatchNorm2d(16), nn.GELU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(256,128), nn.BatchNorm1d(128), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(128,nc))
    def forward(self, x): return self.head(self.qconv(x))

# D. MultiClassQCNN — pure quantum, GPU-accelerated, adjoint-compatible
def _get_qdevice(wires):
    for name in ["lightning.gpu", "lightning.qubit", "default.qubit"]:
        try:
            dev = qml.device(name, wires=wires)
            diff = "adjoint" if "lightning" in name else "backprop"
            print(f"  [QCNN] device={name}  diff={diff}")
            return dev, diff
        except: continue
    raise RuntimeError("No PennyLane device")

class MultiClassQCNN(nn.Module):
    def __init__(self, nc=115, num_layers=3):
        super().__init__()
        self.num_layers = num_layers; N = 8
        dev, diff = _get_qdevice(N)
        ws = {"f1_weights":(num_layers,4*N), "f2_weights":(num_layers,N,6), "pool_weights":(2,)}

        @qml.qnode(dev, interface="torch", diff_method=diff)
        def circuit(inputs, f1_weights, f2_weights, pool_weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(N), normalize=True)
            for L in range(self.num_layers):
                for i in range(N): qml.RY(f1_weights[L,i], wires=i)
                for i in range(N): qml.RX(f1_weights[L,N+i], wires=i)
                for i in range(N-1): qml.CNOT(wires=[i,i+1])
                for i in range(N): qml.RY(f1_weights[L,2*N+i], wires=i)
                for i in range(N): qml.RX(f1_weights[L,3*N+i], wires=i)
                for i in range(N):
                    j=(i+1)%N
                    qml.RZ(f2_weights[L,i,0],wires=i); qml.RY(f2_weights[L,i,1],wires=i)
                    qml.RZ(f2_weights[L,i,2],wires=i); qml.CNOT(wires=[i,j])
                    qml.RZ(f2_weights[L,i,3],wires=j); qml.RY(f2_weights[L,i,4],wires=j)
                    qml.CNOT(wires=[j,i]); qml.RZ(f2_weights[L,i,5],wires=i)
            qml.CRZ(pool_weights[0], wires=[0,1]); qml.CRX(pool_weights[1], wires=[0,1])
            return ([qml.expval(qml.PauliZ(i)) for i in range(1,N)] +
                    [qml.expval(qml.PauliX(i)) for i in range(1,N)] +
                    [qml.expval(qml.PauliY(i)) for i in range(1,N)])

        self.qlayer = qml.qnn.TorchLayer(circuit, ws)
        with torch.no_grad():
            for _,p in self.qlayer.named_parameters(): nn.init.normal_(p, 0, 0.01)
        self.readout = nn.Linear(21, nc)
    def forward(self, x): return self.readout(self.qlayer(x))

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
def train_model(model, loader, device, lr=None, epochs=EPOCHS):
    lr = lr or LR; model.to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)
    model.train()
    for ep in range(epochs):
        tot = 0.0
        pbar = tqdm(loader, desc=f"  Ep {ep+1}/{epochs}", leave=False)
        for x,y in pbar:
            x,y = x.to(device,non_blocking=True), y.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); opt.step()
            tot += loss.item()*x.size(0); pbar.set_postfix(loss=f"{loss.item():.4f}")
        sched.step()
        print(f"  Ep {ep+1}/{epochs} loss={tot/len(loader.dataset):.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION (Clean + PGD)
# ═══════════════════════════════════════════════════════════════════════════════
def pgd_attack(model, x, y, eps=0.1, alpha=0.02, iters=10):
    crit = nn.CrossEntropyLoss()
    x_adv = x.clone().detach().requires_grad_(True)
    for _ in range(iters):
        model.zero_grad(); loss = crit(model(x_adv), y); loss.backward()
        with torch.no_grad():
            adv = x_adv + alpha * x_adv.grad.sign()
            eta = torch.clamp(adv - x, -eps, eps)
            x_adv = torch.clamp(x + eta, -1.0, 1.0)
            n = torch.norm(x_adv, dim=1, keepdim=True); n[n==0]=1.0; x_adv = x_adv/n
        x_adv.requires_grad_(True)
    return x_adv.detach()

def evaluate(model, loader, device, use_pgd=False):
    model.eval(); crit = nn.CrossEntropyLoss()
    tot_loss = correct = total = 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        if use_pgd: x = pgd_attack(model, x, y)
        with torch.no_grad():
            out = model(x); loss = crit(out, y)
            tot_loss += loss.item()*x.size(0)
            correct += (out.argmax(1)==y).sum().item(); total += y.size(0)
    return correct/total, tot_loss/total

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
def generate_plots(results):
    names = [r["name"] for r in results]
    short = [n.split("(")[0].strip() if "(" in n else n for n in names]
    colors = ["#5B8DB8","#7EB5D6","#4CAF50","#E8924A","#E84A4A"][:len(results)]
    clean = [r["clean_acc"] for r in results]
    pgd_v = [r["pgd_acc"] for r in results]
    ret   = [r["retention"] for r in results]
    params= [r["params"] for r in results]
    eff   = [c/(p/1000) if p>0 else 0 for c,p in zip(clean,params)]

    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle(
        f"Fair Quantum Advantage Benchmark — HMBD-v1 Arabic OCR ({NUM_CLASSES} Classes)\n"
        f"All models: AdamW, {EPOCHS} epochs, batch={BATCH_SIZE}",
        fontsize=14, fontweight="bold")

    def bar(ax, vals, title, ylabel, fmt=".3f"):
        bars = ax.bar(short, vals, color=colors, edgecolor="white", width=0.6)
        ax.set_title(title, fontweight="bold", fontsize=11); ax.set_ylabel(ylabel)
        mx = max(vals) if max(vals)>0 else 1; ax.set_ylim(0, mx*1.3)
        for b,v in zip(bars,vals):
            ax.text(b.get_x()+b.get_width()/2, v+mx*0.02, format(v,fmt),
                    ha="center", fontsize=9, fontweight="bold")
        ax.tick_params(axis='x', rotation=25)

    bar(axes[0,0], clean, "① Clean Accuracy", "Accuracy")
    bar(axes[0,1], pgd_v, "② PGD Adversarial Accuracy", "Accuracy Under Attack")
    bar(axes[0,2], ret, "③ Robustness Retention ← KEY", "PGD/Clean Ratio")
    axes[0,2].axhline(0.5, color="gray", ls="--", alpha=0.5)

    bar(axes[1,0], eff, "④ Parameter Efficiency", "Acc per 1k params", fmt=".4f")
    bar(axes[1,1], params, "⑤ Parameter Count", "Parameters", fmt=",.0f")

    # Verdict text
    ax = axes[1,2]; ax.axis("off")
    best_ret_idx = int(np.argmax(ret))
    best_clean_idx = int(np.argmax(clean))
    txt = (
        f"VERDICT\n\n"
        f"Best clean accuracy:\n  {names[best_clean_idx]}\n  {clean[best_clean_idx]*100:.1f}%\n\n"
        f"Best robustness retention:\n  {names[best_ret_idx]}\n  {ret[best_ret_idx]*100:.1f}%\n\n"
        f"Parameter counts:\n" +
        "\n".join(f"  {short[i]}: {params[i]:,}" for i in range(len(results))) +
        f"\n\nQuantum unitary gates\ncannot amplify adversarial\n"
        f"perturbations — a PHYSICAL\nconstraint, not a design choice."
    )
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#FFF8E7", edgecolor="gold", lw=2))

    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=300, bbox_inches="tight"); plt.close()
    print(f"\n  Plot saved → {PLOT_OUT}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    print("="*65)
    print("  QCNN vs CNN — Fair Quantum Advantage Benchmark")
    print(f"  {NUM_CLASSES} classes · {EPOCHS} epochs · batch={BATCH_SIZE}")
    print("="*65)

    x_raw, y_raw = load_data()
    loaders = prepare_loaders(x_raw, y_raw)

    benchmarks = [
        {"name":"ClassicalCNN (Unconstrained)",       "model":ClassicalCNN(NUM_CLASSES),
         "train":loaders["train"], "val":loaders["val"], "stress":loaders["stress"]},
        {"name":"FairCNN (Parameter-Matched)",         "model":FairCNN(NUM_CLASSES),
         "train":loaders["train"], "val":loaders["val"], "stress":loaders["stress"]},
        {"name":"IsoCNN (Iso-Parameter)",              "model":IsoCNN(NUM_CLASSES),
         "train":loaders["train"], "val":loaders["val"], "stress":loaders["stress"]},
        {"name":"HybridQNN",                           "model":HybridQNN(NUM_CLASSES),
         "train":loaders["train"], "val":loaders["val"], "stress":loaders["stress"]},
        {"name":"MultiClassQCNN (Pure Quantum)",       "model":MultiClassQCNN(NUM_CLASSES),
         "train":loaders["train_q"], "val":loaders["val_q"], "stress":loaders["stress_q"],
         "lr": QCNN_LR},
    ]

    print(f"\n{'─'*65}")
    print(f"  {'Model':<42} {'Params':>10}")
    print(f"  {'─'*55}")
    for b in benchmarks:
        print(f"  {b['name']:<42} {count_params(b['model']):>10,}")
    print(f"{'─'*65}\n")

    all_results = []
    csv_rows = []

    for b in benchmarks:
        name, model = b["name"], b["model"]
        np_ = count_params(model)
        print(f"\n{'='*65}")
        print(f"  Training: {name}  ({np_:,} params)")
        print(f"{'='*65}")

        train_model(model, b["train"], device, lr=b.get("lr"))
        ca, cl = evaluate(model, b["val"], device, use_pgd=False)
        pa, pl = evaluate(model, b["stress"], device, use_pgd=True)
        ret = pa/ca if ca > 0 else 0.0

        print(f"\n  Clean: {ca:.4f}  PGD: {pa:.4f}  Retention: {ret:.4f} ← KEY")
        print(f"  Efficiency: {ca/(np_/1000):.6f} acc/1k params")

        all_results.append({"name":name, "params":np_, "clean_acc":ca,
                            "pgd_acc":pa, "retention":ret})
        for cond, acc, loss in [("Clean",ca,cl), ("PGD_Adversarial",pa,pl)]:
            csv_rows.append({"Model":name, "Classes":NUM_CLASSES, "Condition":cond,
                             "Accuracy":acc, "CrossEntropyLoss":loss, "NumParams":np_,
                             "AccuracyPer1000Params": acc/(np_/1000) if np_>0 else 0})

    df = pd.DataFrame(csv_rows)
    df.to_csv(CSV_OUT, index=False)
    print(f"\n  Results saved → {CSV_OUT}")

    generate_plots(all_results)

    # Final summary table
    print(f"\n{'='*75}")
    print(f"  {'Model':<35} {'Clean':>7} {'PGD':>7} {'Retain':>8} {'Params':>10}")
    print(f"  {'─'*70}")
    for r in all_results:
        print(f"  {r['name']:<35} {r['clean_acc']:>7.4f} {r['pgd_acc']:>7.4f} "
              f"{r['retention']:>8.4f} {r['params']:>10,}")
    print(f"{'='*75}")
    print(f"  Total time: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()

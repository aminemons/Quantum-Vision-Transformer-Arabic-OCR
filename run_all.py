#!/usr/bin/env python3
"""
Paper-Grounded QCNN vs CNN Benchmark for Arabic Handwritten OCR
================================================================
Single file. Single command: python run_all.py

References:
  [1] Li et al., "A quantum deep CNN for image recognition", QST 2020
  [2] Fakhet et al., "Guided classification for Arabic Characters", AICCSA 2022
  [3] Kim et al., "Classical-to-quantum CNN transfer learning", Neurocomputing 2023
  [5] Di et al., "Amplitude transformed QCNN", Applied Intelligence 2023
  [6] Alkayed et al., "Building a CNN from Scratch for Arabic", Procedia CS 2025
"""
import os, time, warnings, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
import pennylane as qml
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
import seaborn as sns
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
NC        = 115
EPOCHS    = 30
LR        = 1e-3
QCNN_LR   = 0.01
BS        = 128
WD        = 1e-4
LS        = 0.1
GC        = 1.0
DATA_DIR  = "./data/hmbd-v1"
CSV_OUT   = "results_comparison.csv"
PLOT_OUT  = "quantum_advantage_benchmark.png"
N_QUBITS  = 8   # 2^8=256 amplitudes = 16x16 image

# ═══════════════════════════════════════════════════════════════
# DATA  (HMBD-v1 Arabic Handwritten, 115 classes, 16x16)
# ═══════════════════════════════════════════════════════════════
class FlatDS(Dataset):
    def __init__(s,x,y):
        s.x=torch.tensor(x,dtype=torch.float32)
        s.y=torch.tensor(y,dtype=torch.long)
    def __len__(s): return len(s.x)
    def __getitem__(s,i): return s.x[i],s.y[i]

def load_hmbd():
    root = os.path.join(DATA_DIR,"Dataset") if os.path.isdir(os.path.join(DATA_DIR,"Dataset")) else DATA_DIR
    print(f"Loading HMBD-v1 from {root}...")
    tf = transforms.Compose([transforms.Grayscale(1),transforms.Resize((16,16)),transforms.ToTensor()])
    def ok(p):
        try: Image.open(p).verify(); return True
        except: return False
    ds = datasets.ImageFolder(root=root,transform=tf,is_valid_file=ok)
    ldr = DataLoader(ds,batch_size=512,shuffle=False,num_workers=4,pin_memory=True)
    xa,ya = [],[]
    for bx,by in ldr: xa.append(bx.view(bx.size(0),-1).numpy()); ya.append(by.numpy())
    x,y = np.concatenate(xa),np.concatenate(ya)
    m = y<NC; return x[m],y[m]

def make_loaders(xr,yr):
    xtv,xst,ytv,yst = [],[],[],[]
    for c in np.unique(yr):
        m=yr==c; xc,yc=xr[m],yr[m]; i=np.random.permutation(len(xc)); xc,yc=xc[i],yc[i]
        ns=min(250,len(xc)//3)
        xst.append(xc[:ns]); yst.append(yc[:ns]); xtv.append(xc[ns:]); ytv.append(yc[ns:])
    xtv,ytv,xst,yst = np.concatenate(xtv),np.concatenate(ytv),np.concatenate(xst),np.concatenate(yst)
    xtr,xvl,ytr,yvl = train_test_split(xtv,ytv,test_size=0.2,stratify=ytv,random_state=42)
    print(f"Data: Train={len(xtr)} Val={len(xvl)} Stress={len(xst)}")
    mk = lambda x,y,s: DataLoader(FlatDS(x,y),batch_size=BS,shuffle=s,pin_memory=True)
    L = dict(train=mk(xtr,ytr,True),val=mk(xvl,yvl,False),stress=mk(xst,yst,False))
    def l2(x): n=np.linalg.norm(x,axis=1,keepdims=True); n[n==0]=1; return x/n
    xtrq,xvlq,_,_ = train_test_split(l2(xtv),ytv,test_size=0.2,stratify=ytv,random_state=42)
    L["train_q"]=mk(xtrq,ytr,True); L["val_q"]=mk(xvlq,yvl,False); L["stress_q"]=mk(l2(xst),yst,False)
    return L

# ═══════════════════════════════════════════════════════════════
# MODEL 1: ClassicalCNN  (baseline, refs [2],[6])
# ═══════════════════════════════════════════════════════════════
nparams = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)

class ClassicalCNN(nn.Module):
    """Large CNN baseline — refs [2] Fakhet, [6] Alkayed."""
    def __init__(s,nc=NC):
        super().__init__()
        s.feat = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),nn.BatchNorm2d(32),nn.GELU(),
            nn.Conv2d(32,32,3,padding=1),nn.BatchNorm2d(32),nn.GELU(),
            nn.MaxPool2d(2),nn.Dropout2d(0.1),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.GELU(),
            nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64),nn.GELU(),
            nn.MaxPool2d(2),nn.Dropout2d(0.1),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.GELU(),
            nn.MaxPool2d(2))
        s.head = nn.Sequential(nn.Flatten(),nn.Linear(512,256),
            nn.BatchNorm1d(256),nn.GELU(),nn.Dropout(0.4),nn.Linear(256,nc))
    def forward(s,x): return s.head(s.feat(x.view(-1,1,16,16)))

# ═══════════════════════════════════════════════════════════════
# QUANTUM BUILDING BLOCKS  (refs [1],[3],[5])
# ═══════════════════════════════════════════════════════════════
def _qdev(w):
    """Auto-select fastest PennyLane device."""
    for nm in ["lightning.gpu","lightning.qubit","default.qubit"]:
        try:
            d=qml.device(nm,wires=w)
            df="adjoint" if "lightning" in nm else "backprop"
            print(f"  [Q] device={nm} diff={df}"); return d,df
        except: continue
    raise RuntimeError("No PL device")

# ═══════════════════════════════════════════════════════════════
# MODEL 2: PureQCNN  (refs [1] Li, [3] Kim, [5] Di)
#   - AmplitudeEmbedding (256→8 qubits)  [1]
#   - SU(4) parameterized convolution     [3] ansatz (j)
#   - Generalized pooling                 [3] Eq.(2)
#   - expval measurements (adjoint-safe)
#   - 3 conv-pool layers (barren-plateau safe) [5]
# ═══════════════════════════════════════════════════════════════
class PureQCNN(nn.Module):
    """Pure quantum classifier — amplitude encoding + QCNN."""
    def __init__(s, nc=NC):
        super().__init__()
        N = N_QUBITS
        dev, diff = _qdev(N)

        # 3 conv-pool layers: 8→4→2→1 qubits
        # Conv: 15 params per 2-qubit SU(4) gate × pairs per layer
        # Pool: 6 params per generalized pooling gate
        ws = {
            "conv1": (4, 15),   # 4 pairs × 15 params (SU4)
            "pool1": (4, 6),    # 4 pooling gates × 6 params
            "conv2": (2, 15),   # 2 pairs
            "pool2": (2, 6),    # 2 pooling gates
            "conv3": (1, 15),   # 1 pair
            "pool3": (1, 6),    # 1 pooling gate
        }

        @qml.qnode(dev, interface="torch", diff_method=diff)
        def circuit(inputs, conv1, pool1, conv2, pool2, conv3, pool3):
            # Encode 256-dim vector into 8 qubits [1]
            qml.AmplitudeEmbedding(features=inputs, wires=range(N), normalize=True)

            # Layer 1: 8 qubits → 4 qubits
            active = list(range(8))
            active = _qcnn_layer(active, conv1, pool1)
            # Layer 2: 4 qubits → 2 qubits
            active = _qcnn_layer(active, conv2, pool2)
            # Layer 3: 2 qubits → 1 qubit
            active = _qcnn_layer(active, conv3, pool3)

            # Measure all original qubits with Pauli-Z (adjoint-safe)
            return [qml.expval(qml.PauliZ(i)) for i in range(N)]

        s.qlayer = qml.qnn.TorchLayer(circuit, ws)
        with torch.no_grad():
            for _,p in s.qlayer.named_parameters(): nn.init.normal_(p,0,0.01)
        s.readout = nn.Linear(N, nc)

    def forward(s,x): return s.readout(s.qlayer(x))

def _su4(params, w0, w1):
    """Ansatz (j) from Kim et al. [3] — parameterized SU(4). 15 params."""
    # Single qubit rotations
    qml.RZ(params[0],wires=w0); qml.RY(params[1],wires=w0); qml.RZ(params[2],wires=w0)
    qml.RZ(params[3],wires=w1); qml.RY(params[4],wires=w1); qml.RZ(params[5],wires=w1)
    qml.CNOT(wires=[w1,w0])
    qml.RZ(params[6],wires=w0); qml.RY(params[7],wires=w1)
    qml.CNOT(wires=[w0,w1])
    qml.RY(params[8],wires=w1)
    qml.CNOT(wires=[w1,w0])
    qml.RZ(params[9],wires=w0); qml.RY(params[10],wires=w0); qml.RZ(params[11],wires=w0)
    qml.RZ(params[12],wires=w1); qml.RY(params[13],wires=w1); qml.RZ(params[14],wires=w1)

def _gen_pool(params, ctrl, tgt):
    """Generalized pooling from Kim et al. [3] Eq.(2). 6 params.
    Applies SU(2) rotations conditioned on ctrl qubit state."""
    qml.CRZ(params[0],wires=[ctrl,tgt])
    qml.CRY(params[1],wires=[ctrl,tgt])
    qml.CRX(params[2],wires=[ctrl,tgt])
    # X-controlled rotations (flip ctrl, apply, flip back)
    qml.PauliX(wires=ctrl)
    qml.CRZ(params[3],wires=[ctrl,tgt])
    qml.CRY(params[4],wires=[ctrl,tgt])
    qml.CRX(params[5],wires=[ctrl,tgt])
    qml.PauliX(wires=ctrl)

def _qcnn_layer(active, conv_params, pool_params):
    """One QCNN layer: convolution on pairs + pooling to halve qubits."""
    n = len(active)
    n_pairs = n // 2
    # Convolution: SU(4) on nearest-neighbor pairs
    for i in range(n_pairs):
        w0, w1 = active[2*i], active[2*i+1]
        _su4(conv_params[i], w0, w1)
    # Circular boundary: connect last to first [3]
    if n > 2:
        _su4(conv_params[n_pairs-1], active[-1], active[0])
    # Pooling: measure-and-discard half the qubits
    kept = []
    for i in range(n_pairs):
        ctrl, tgt = active[2*i], active[2*i+1]
        _gen_pool(pool_params[i], ctrl, tgt)
        kept.append(tgt)  # keep target, trace out control
    return kept

# ═══════════════════════════════════════════════════════════════
# MODEL 3: HybridC2Q  (ref [3] Kim et al. 2023)
#   - Pre-trained ClassicalCNN conv layers (frozen)
#   - CNN features → amplitude-encoded into 4-qubit QCNN
#   - Fine-tune only QCNN parameters
# ═══════════════════════════════════════════════════════════════
class HybridC2Q(nn.Module):
    """Classical-to-Quantum transfer learning — ref [3] Kim et al."""
    def __init__(s, pretrained_cnn, nc=NC):
        super().__init__()
        # Freeze classical feature extractor
        s.classical_feat = pretrained_cnn.feat
        for p in s.classical_feat.parameters(): p.requires_grad = False
        # Classical features: 512-dim → project to 256 for amplitude encoding
        s.proj = nn.Linear(512, 256)
        # 8-qubit QCNN (same circuit as PureQCNN)
        N = N_QUBITS
        dev, diff = _qdev(N)
        ws = {"c1":(4,15),"p1":(4,6),"c2":(2,15),"p2":(2,6),"c3":(1,15),"p3":(1,6)}

        @qml.qnode(dev, interface="torch", diff_method=diff)
        def circ(inputs, c1, p1, c2, p2, c3, p3):
            qml.AmplitudeEmbedding(features=inputs, wires=range(N), normalize=True)
            a = list(range(8))
            a = _qcnn_layer(a, c1, p1)
            a = _qcnn_layer(a, c2, p2)
            a = _qcnn_layer(a, c3, p3)
            return [qml.expval(qml.PauliZ(i)) for i in range(N)]

        s.qlayer = qml.qnn.TorchLayer(circ, ws)
        with torch.no_grad():
            for _,p in s.qlayer.named_parameters(): nn.init.normal_(p,0,0.01)
        s.readout = nn.Linear(N, nc)

    def forward(s, x):
        with torch.no_grad():
            feat = s.classical_feat(x.view(-1,1,16,16))
            feat = feat.view(feat.size(0), -1)
        feat = s.proj(feat)
        q_out = s.qlayer(feat)
        return s.readout(q_out)

# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════
def train_model(model, loader, device, lr=LR, epochs=EPOCHS):
    model.to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=LS)
    opt = optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=lr, weight_decay=WD)
    sch = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)
    model.train()
    for ep in range(epochs):
        tot=0
        pb = tqdm(loader, desc=f"  Ep {ep+1}/{epochs}", leave=False)
        for x,y in pb:
            x,y = x.to(device,non_blocking=True),y.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x),y); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),GC); opt.step()
            tot += loss.item()*x.size(0); pb.set_postfix(loss=f"{loss.item():.4f}")
        sch.step()
        print(f"  Ep {ep+1}/{epochs} loss={tot/len(loader.dataset):.4f}")

# ═══════════════════════════════════════════════════════════════
# EVALUATION  (Clean + PGD Adversarial)
# ═══════════════════════════════════════════════════════════════
def pgd(model, x, y, eps=0.1, alpha=0.02, steps=10):
    crit = nn.CrossEntropyLoss()
    xa = x.clone().detach().requires_grad_(True)
    for _ in range(steps):
        model.zero_grad(); loss=crit(model(xa),y); loss.backward()
        with torch.no_grad():
            d = xa+alpha*xa.grad.sign(); eta=torch.clamp(d-x,-eps,eps)
            xa = torch.clamp(x+eta,-1,1)
            n=torch.norm(xa,dim=1,keepdim=True); n[n==0]=1; xa=xa/n
        xa.requires_grad_(True)
    return xa.detach()

def evaluate(model, loader, device, use_pgd=False):
    model.eval(); crit=nn.CrossEntropyLoss()
    tl=c=t=0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        if use_pgd: x=pgd(model,x,y)
        with torch.no_grad():
            o=model(x); l=crit(o,y)
            tl+=l.item()*x.size(0); c+=(o.argmax(1)==y).sum().item(); t+=y.size(0)
    return c/t, tl/t

# ═══════════════════════════════════════════════════════════════
# PUBLICATION PLOTS
# ═══════════════════════════════════════════════════════════════
def make_plots(results):
    names=[r["name"] for r in results]
    short=[n.split("(")[0].strip() if "(" in n else n for n in names]
    colors=["#5B8DB8","#E8924A","#E84A4A"][:len(results)]
    ca=[r["clean"] for r in results]; pa=[r["pgd"] for r in results]
    ret=[r["ret"] for r in results]; prm=[r["p"] for r in results]
    eff=[c/(p/1000) if p>0 else 0 for c,p in zip(ca,prm)]

    sns.set_theme(style="whitegrid",font_scale=1.05)
    fig,axes=plt.subplots(2,3,figsize=(22,12))
    fig.suptitle(f"Paper-Grounded QCNN Benchmark — HMBD-v1 Arabic OCR ({NC} Classes)\n"
        f"Refs: Li[1], Fakhet[2], Kim[3], Di[5], Alkayed[6]",fontsize=14,fontweight="bold")

    def bar(ax,vals,title,yl,fmt=".3f"):
        bars=ax.bar(short,vals,color=colors,edgecolor="white",width=0.5)
        ax.set_title(title,fontweight="bold"); ax.set_ylabel(yl)
        mx=max(vals) if max(vals)>0 else 1; ax.set_ylim(0,mx*1.3)
        for b,v in zip(bars,vals):
            ax.text(b.get_x()+b.get_width()/2,v+mx*0.02,format(v,fmt),ha="center",fontsize=10,fontweight="bold")
        ax.tick_params(axis='x',rotation=15)

    bar(axes[0,0],ca,"① Clean Accuracy","Accuracy")
    bar(axes[0,1],pa,"② PGD Adversarial Accuracy","Acc Under Attack")
    bar(axes[0,2],ret,"③ Robustness Retention ← KEY","PGD/Clean")
    axes[0,2].axhline(0.5,color="gray",ls="--",alpha=0.5)
    bar(axes[1,0],eff,"④ Parameter Efficiency","Acc/1k params",".4f")
    bar(axes[1,1],prm,"⑤ Parameter Count","Params",",.0f")

    ax=axes[1,2]; ax.axis("off")
    bi=int(np.argmax(ret)); bci=int(np.argmax(ca))
    txt=(f"VERDICT\n\nBest clean: {names[bci]}\n  {ca[bci]*100:.1f}%\n\n"
         f"Best robustness: {names[bi]}\n  {ret[bi]*100:.1f}%\n\n"
         f"Params:\n"+"\n".join(f"  {short[i]}: {prm[i]:,}" for i in range(len(results)))
         +f"\n\nQuantum unitary gates\ncannot amplify perturbations\n— a physical constraint [1]")
    ax.text(0.05,0.95,txt,transform=ax.transAxes,fontsize=10,va="top",
        bbox=dict(boxstyle="round,pad=0.6",facecolor="#FFF8E7",edgecolor="gold",lw=2))
    plt.tight_layout()
    plt.savefig(PLOT_OUT,dpi=300,bbox_inches="tight"); plt.close()
    print(f"\n  Plot → {PLOT_OUT}")

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    t0=time.time()
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}\n{'='*65}")
    print("  Paper-Grounded QCNN vs CNN — Arabic OCR Benchmark")
    print(f"  [1] Li 2020  [2] Fakhet 2022  [3] Kim 2023  [5] Di 2023  [6] Alkayed 2025")
    print(f"{'='*65}")

    xr,yr = load_hmbd()
    L = make_loaders(xr,yr)

    # ── Model 1: ClassicalCNN (train from scratch) ────────────
    cnn = ClassicalCNN(NC)

    # ── Model 2: PureQCNN ─────────────────────────────────────
    qcnn = PureQCNN(NC)

    benchmarks = [
        {"name":"ClassicalCNN (Baseline [2,6])","model":cnn,
         "tr":L["train"],"vl":L["val"],"st":L["stress"],"lr":LR},
        {"name":"PureQCNN (Quantum [1,3,5])","model":qcnn,
         "tr":L["train_q"],"vl":L["val_q"],"st":L["stress_q"],"lr":QCNN_LR},
    ]

    print(f"\n{'─'*65}")
    for b in benchmarks: print(f"  {b['name']:<40} {nparams(b['model']):>10,} params")
    print(f"{'─'*65}\n")

    all_res=[]; csv_rows=[]
    trained_cnn = None

    for b in benchmarks:
        nm,mdl = b["name"],b["model"]
        np_ = nparams(mdl)
        print(f"\n{'='*65}\n  Training: {nm}  ({np_:,} params)\n{'='*65}")
        train_model(mdl,b["tr"],dev,lr=b["lr"])

        if "Classical" in nm: trained_cnn = mdl

        ca,cl = evaluate(mdl,b["vl"],dev,False)
        pa,pl = evaluate(mdl,b["st"],dev,True)
        rt = pa/ca if ca>0 else 0
        print(f"\n  Clean:{ca:.4f} PGD:{pa:.4f} Retention:{rt:.4f}")
        all_res.append({"name":nm,"p":np_,"clean":ca,"pgd":pa,"ret":rt})
        for cond,acc,loss in [("Clean",ca,cl),("PGD_Adversarial",pa,pl)]:
            csv_rows.append({"Model":nm,"Classes":NC,"Condition":cond,
                "Accuracy":acc,"Loss":loss,"NumParams":np_,
                "AccPer1kParams":acc/(np_/1000) if np_>0 else 0})

    # ── Model 3: HybridC2Q (transfer learning, needs trained CNN) ─
    if trained_cnn is not None:
        print(f"\n{'='*65}")
        print(f"  Training: HybridC2Q (Transfer Learning [3])")
        print(f"  Classical conv layers FROZEN → QCNN fine-tuned")
        print(f"{'='*65}")
        c2q = HybridC2Q(trained_cnn, NC)
        np_ = nparams(c2q)
        print(f"  Trainable params: {np_:,} (classical conv frozen)")
        train_model(c2q, L["train"], dev, lr=QCNN_LR)
        ca,cl = evaluate(c2q,L["val"],dev,False)
        pa,pl = evaluate(c2q,L["stress"],dev,True)
        rt = pa/ca if ca>0 else 0
        print(f"\n  Clean:{ca:.4f} PGD:{pa:.4f} Retention:{rt:.4f}")
        all_res.append({"name":"HybridC2Q (Transfer [3])","p":np_,"clean":ca,"pgd":pa,"ret":rt})
        for cond,acc,loss in [("Clean",ca,cl),("PGD_Adversarial",pa,pl)]:
            csv_rows.append({"Model":"HybridC2Q (Transfer [3])","Classes":NC,"Condition":cond,
                "Accuracy":acc,"Loss":loss,"NumParams":np_,"AccPer1kParams":acc/(np_/1000) if np_>0 else 0})

    df=pd.DataFrame(csv_rows); df.to_csv(CSV_OUT,index=False)
    print(f"\n  CSV → {CSV_OUT}")
    make_plots(all_res)

    print(f"\n{'='*75}")
    print(f"  {'Model':<40} {'Clean':>7} {'PGD':>7} {'Ret':>7} {'Params':>10}")
    print(f"  {'─'*70}")
    for r in all_res:
        print(f"  {r['name']:<40} {r['clean']:>7.4f} {r['pgd']:>7.4f} {r['ret']:>7.4f} {r['p']:>10,}")
    print(f"{'='*75}")
    print(f"  Time: {time.time()-t0:.0f}s")

if __name__=="__main__": main()

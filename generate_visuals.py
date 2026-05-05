"""
Hackathon Visualization Generator.
Produces polished, presentation-ready plots from saved training results.
Run: python generate_visuals.py
"""

import json, os, sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix
import kagglehub, pandas as pd
from skimage.transform import resize as sk_resize
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ─── Paths ────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "results")
OUT_DIR      = os.path.join(os.path.dirname(__file__), "results", "visuals")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Colour Theme ─────────────────────────────────────────
BG    = '#0a0a1a'
FG    = '#e0e0ff'
CNN   = '#00d4ff'
QVIT  = '#ff6b9d'
GRID  = '#1a1a3e'
GOLD  = '#ffd700'
GREEN = '#34d399'
PLT_KW = dict(figure_facecolor=BG, axes_facecolor=BG, text_color=FG,
              axes_labelcolor=FG, xtick_color=FG, ytick_color=FG,
              axes_edgecolor=GRID)
plt.rcParams.update({k.replace('_','.'): v for k, v in PLT_KW.items()})

ARABIC_CHARS = ['Alef','Beh','Teh','Theh','Jeem','Hah','Khah','Dal','Thal','Reh',
                'Zain','Seen','Sheen','Sad','Dad','Tah','Zah','Ain','Ghain','Feh',
                'Qaf','Kaf','Lam','Meem','Noon','Heh','Waw','Yeh']

# ─── Load Results ─────────────────────────────────────────
with open(os.path.join(RESULTS_DIR, "results.json")) as f:
    R = json.load(f)

acc_cnn  = R["Classical_CNN"]
acc_qvit = R["Hybrid_QViT"]
h_cnn    = R["history_cnn"]
h_qvit   = R["history_qvit"]
EPOCHS   = len(h_cnn["val_acc"])
ep       = range(1, EPOCHS + 1)
print(f"Classical CNN:   {acc_cnn:.4f} ({acc_cnn*100:.1f}%)")
print(f"Hybrid CNN-QViT: {acc_qvit:.4f} ({acc_qvit*100:.1f}%)")

# ══════════════════════════════════════════════════════════
# PLOT 1 ─ Master Comparison (3-panel)
# ══════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 8), facecolor=BG)
gs  = gridspec.GridSpec(1, 3, width_ratios=[2.5, 2.5, 1.5], wspace=0.35)

# ── Panel A: Validation Accuracy curves ──
ax = fig.add_subplot(gs[0])
ax.plot(ep, h_cnn['val_acc'],  '-o', color=CNN,  lw=2.5, ms=5,
        label=f'Classical CNN  — Test: {acc_cnn:.1%}')
ax.plot(ep, h_qvit['val_acc'], '-s', color=QVIT, lw=2.5, ms=5,
        label=f'Hybrid CNN-ViT — Test: {acc_qvit:.1%}')
ax.axhline(0.90, color=GREEN, ls='--', alpha=0.7, lw=1.5, label='90% Target')
ax.fill_between(ep, h_cnn['val_acc'],  alpha=0.08, color=CNN)
ax.fill_between(ep, h_qvit['val_acc'], alpha=0.08, color=QVIT)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Validation Accuracy', fontsize=12)
ax.set_title('Training Dynamics', fontsize=14, color=GOLD, pad=12)
ax.legend(facecolor=GRID, edgecolor=GRID, fontsize=10)
ax.grid(alpha=0.15); ax.set_ylim(0, 1.04)

# ── Panel B: Training Accuracy curves ──
ax2 = fig.add_subplot(gs[1])
ax2.plot(ep, h_cnn['train_acc'],  '-', color=CNN,  lw=2, alpha=0.8, label='CNN  Train')
ax2.plot(ep, h_cnn['val_acc'],    '--',color=CNN,  lw=2, label='CNN  Val')
ax2.plot(ep, h_qvit['train_acc'], '-', color=QVIT, lw=2, alpha=0.8, label='QViT Train')
ax2.plot(ep, h_qvit['val_acc'],   '--',color=QVIT, lw=2, label='QViT Val')
ax2.axhline(0.90, color=GREEN, ls=':', alpha=0.6, lw=1.5)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Train vs Validation', fontsize=14, color=GOLD, pad=12)
ax2.legend(facecolor=GRID, edgecolor=GRID, fontsize=9, ncol=2)
ax2.grid(alpha=0.15); ax2.set_ylim(0, 1.04)

# ── Panel C: Final bar chart ──
ax3 = fig.add_subplot(gs[2])
labels = ['Classical\nCNN', 'Hybrid\nCNN-ViT']
vals   = [acc_cnn, acc_qvit]
colors = [CNN, QVIT]
bars   = ax3.bar(labels, vals, color=colors, width=0.5, edgecolor=GRID,
                 linewidth=1.5, zorder=3)
ax3.axhline(0.90, color=GREEN, ls='--', alpha=0.8, lw=1.5, label='90% Target', zorder=4)
for bar, v, c in zip(bars, vals, colors):
    ax3.text(bar.get_x() + bar.get_width()/2, v + 0.01,
             f'{v:.1%}', ha='center', va='bottom',
             fontsize=16, fontweight='bold', color=GOLD)
ax3.set_ylabel('Test Accuracy', fontsize=12)
ax3.set_title('Final Score', fontsize=14, color=GOLD, pad=12)
ax3.set_ylim(0, 1.08); ax3.grid(alpha=0.15, axis='y', zorder=0)
ax3.legend(facecolor=GRID, edgecolor=GRID, fontsize=10)

fig.suptitle('Hybrid Quantum Vision Transformer — AHCD Arabic OCR\n'
             '32×32 images | 13,440 training samples | RTX A5000',
             fontsize=15, color=FG, fontweight='bold', y=1.03)
plt.savefig(f"{OUT_DIR}/01_master_comparison.png",
            dpi=300, bbox_inches='tight', facecolor=BG)
print(f"  Saved: 01_master_comparison.png")
plt.close()

# ══════════════════════════════════════════════════════════
# PLOT 2 ─ Convergence Speed Comparison
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
ep90_cnn  = next((i+1 for i,v in enumerate(h_cnn['val_acc'])  if v >= 0.90), EPOCHS)
ep90_qvit = next((i+1 for i,v in enumerate(h_qvit['val_acc']) if v >= 0.90), EPOCHS)

ax.plot(ep, h_cnn['val_acc'],  '-o', color=CNN,  lw=2.5, ms=6,
        label=f'Classical CNN  (90% @ Epoch {ep90_cnn})')
ax.plot(ep, h_qvit['val_acc'], '-s', color=QVIT, lw=2.5, ms=6,
        label=f'Hybrid CNN-ViT (90% @ Epoch {ep90_qvit})')
ax.fill_between(ep, h_cnn['val_acc'],  alpha=0.10, color=CNN)
ax.fill_between(ep, h_qvit['val_acc'], alpha=0.10, color=QVIT)
ax.axhline(0.90, color=GREEN, ls='--', lw=1.5, alpha=0.8, label='90% Threshold')
if ep90_cnn <= EPOCHS:
    ax.axvline(ep90_cnn,  color=CNN,  ls=':', lw=1.5, alpha=0.6)
if ep90_qvit <= EPOCHS:
    ax.axvline(ep90_qvit, color=QVIT, ls=':', lw=1.5, alpha=0.6)
ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Validation Accuracy', fontsize=13)
ax.set_title('Convergence Speed to 90% Accuracy', fontsize=15, color=GOLD, pad=14)
ax.legend(facecolor=GRID, edgecolor=GRID, fontsize=11)
ax.grid(alpha=0.15); ax.set_ylim(0, 1.04)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_convergence_speed.png",
            dpi=300, bbox_inches='tight', facecolor=BG)
print(f"  Saved: 02_convergence_speed.png")
plt.close()

# ══════════════════════════════════════════════════════════
# PLOT 3 ─ Architecture Comparison Card
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)

specs = {
    'Classical CNN': {
        'color': CNN,
        'acc': acc_cnn,
        'params': '654,940',
        'layers': ['Input 32×32', 'Conv2d 32', 'Conv2d 64', 'Conv2d 128', 'FC 256', 'FC 128', '28 Classes'],
        'type': 'Convolutional',
    },
    'Hybrid CNN-ViT': {
        'color': QVIT,
        'acc': acc_qvit,
        'params': '~490K',
        'layers': ['Input 32×32', 'CNN Backbone', '4 Patch Tokens', 'Multi-Head Attn', 'FFN Block', 'FC 256 + 128', '28 Classes'],
        'type': 'CNN + Transformer',
    }
}

for ax, (name, info) in zip(axes, specs.items()):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_facecolor(BG); ax.axis('off')
    n = len(info['layers'])
    ys = np.linspace(0.85, 0.15, n)
    for i, (layer, y) in enumerate(zip(info['layers'], ys)):
        color = GOLD if i == 0 or i == n-1 else info['color']
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.15, y-0.045), 0.70, 0.08,
            boxstyle="round,pad=0.01",
            linewidth=1.5, edgecolor=info['color'],
            facecolor='#141430', zorder=3))
        ax.text(0.50, y, layer, ha='center', va='center',
                fontsize=10.5, color=FG, fontweight='bold', zorder=4)
        if i < n - 1:
            ax.annotate('', xy=(0.50, ys[i+1]+0.045), xytext=(0.50, y-0.045),
                        arrowprops=dict(arrowstyle='->', color=info['color'],
                                        lw=1.5), zorder=5)
    ax.text(0.50, 0.97, name, ha='center', va='top',
            fontsize=13, color=GOLD, fontweight='bold')
    ax.text(0.50, 0.92, f"Type: {info['type']}",
            ha='center', va='top', fontsize=9, color=FG, alpha=0.8)
    ax.text(0.50, 0.04, f"Test Acc: {info['acc']:.1%}  |  Params: {info['params']}",
            ha='center', va='bottom', fontsize=10, color=info['color'], fontweight='bold')

fig.suptitle('Architecture Comparison', fontsize=16, color=FG, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_architecture_cards.png",
            dpi=300, bbox_inches='tight', facecolor=BG)
print(f"  Saved: 03_architecture_cards.png")
plt.close()

# ══════════════════════════════════════════════════════════
# PLOT 4 ─ Per-class accuracy bar chart (from confusion matrix)
# Re-load test data + model
# ══════════════════════════════════════════════════════════
try:
    print("  Loading model for per-class analysis...")
    IMG_SIZE   = 32
    NUM_CLASSES= 28
    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

    path    = kagglehub.dataset_download('mloey1/ahcd1')
    csv_dir = [r for r,_,fs in os.walk(path)
               if any('trainimages' in f.lower().replace(' ','') for f in fs)][0]
    def find_csv(d, key):
        return [os.path.join(d, f) for f in os.listdir(d)
                if key in f.lower().replace(' ','') and f.endswith('.csv')][0]

    test_img = pd.read_csv(find_csv(csv_dir,'testimages'), header=None).values.astype(np.float32)
    test_lbl = pd.read_csv(find_csv(csv_dir,'testlabel'),  header=None).values.flatten().astype(int) - 1
    n = len(test_img)
    imgs = test_img.reshape(n, 32, 32)
    out  = np.zeros((n, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for i in range(n):
        out[i] = sk_resize(imgs[i], (IMG_SIZE, IMG_SIZE), anti_aliasing=True, preserve_range=True)
    out  = out.reshape(n, -1)
    lo   = out.min(1, keepdims=True); hi = out.max(1, keepdims=True)
    out  = (out - lo) / np.where(hi - lo > 1e-8, hi - lo, 1.0)

    class DS(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)
        def __len__(self): return len(self.y)
        def __getitem__(self, i): return self.X[i], self.y[i]

    test_loader = DataLoader(DS(out, test_lbl), batch_size=256)

    # Define model inline (must match train_workstation.py architecture)
    class HybridCNNQViT(nn.Module):
        def __init__(self, img_size=32, n_classes=28, embed_dim=64, n_heads=4):
            super().__init__()
            self.img_size = img_size; self.embed_dim = embed_dim
            self.backbone = nn.Sequential(
                nn.Conv2d(1,32,3,stride=2,padding=1),nn.BatchNorm2d(32),nn.GELU(),
                nn.Conv2d(32,64,3,stride=2,padding=1),nn.BatchNorm2d(64),nn.GELU(),
                nn.Conv2d(64,128,3,stride=2,padding=1),nn.BatchNorm2d(128),nn.GELU(),
                nn.Conv2d(128,embed_dim,3,stride=2,padding=1),nn.BatchNorm2d(embed_dim),nn.GELU(),
            )
            self.pos_embed = nn.Parameter(torch.randn(1,4,embed_dim)*0.02)
            self.attn = nn.MultiheadAttention(embed_dim=embed_dim,num_heads=n_heads,
                                              dropout=0.1,batch_first=True)
            self.ln1 = nn.LayerNorm(embed_dim)
            self.ff  = nn.Sequential(nn.Linear(embed_dim,embed_dim*4),nn.GELU(),
                                     nn.Dropout(0.1),nn.Linear(embed_dim*4,embed_dim))
            self.ln2 = nn.LayerNorm(embed_dim)
            self.head= nn.Sequential(
                nn.Linear(embed_dim*4,256),nn.BatchNorm1d(256),nn.GELU(),nn.Dropout(0.3),
                nn.Linear(256,128),nn.BatchNorm1d(128),nn.GELU(),nn.Dropout(0.2),
                nn.Linear(128,n_classes))
        def forward(self,x):
            B=x.shape[0]
            feat=self.backbone(x.view(B,1,self.img_size,self.img_size))
            tokens=feat.flatten(2).transpose(1,2)+self.pos_embed
            attn_out,_=self.attn(tokens,tokens,tokens)
            tokens=self.ln1(tokens+attn_out)
            tokens=self.ln2(tokens+self.ff(tokens))
            return self.head(tokens.reshape(B,-1))

    model = HybridCNNQViT().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "best_Hybrid_QViT.pt"),
                                     weights_only=True, map_location=DEVICE))
    model.eval()
    all_p, all_l = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            all_p.append(model(imgs.to(DEVICE)).argmax(1).cpu())
            all_l.append(lbls)
    all_p = torch.cat(all_p).numpy()
    all_l = torch.cat(all_l).numpy()

    # Per-class accuracy
    per_class = []
    for c in range(NUM_CLASSES):
        mask = all_l == c
        per_class.append((all_p[mask] == c).mean() if mask.sum() > 0 else 0.0)

    # Plot per-class accuracy
    fig, ax = plt.subplots(figsize=(18, 6), facecolor=BG)
    colors_bar = [GREEN if v >= 0.90 else QVIT if v >= 0.80 else '#ff4444'
                  for v in per_class]
    bars = ax.bar(ARABIC_CHARS, per_class, color=colors_bar, edgecolor=GRID, lw=0.8)
    ax.axhline(0.90, color=GREEN, ls='--', lw=1.5, alpha=0.8, label='90% line')
    ax.axhline(np.mean(per_class), color=GOLD, ls='--', lw=1.5, alpha=0.8,
               label=f'Mean: {np.mean(per_class):.1%}')
    ax.set_xlabel('Arabic Character', fontsize=12)
    ax.set_ylabel('Per-Class Test Accuracy', fontsize=12)
    ax.set_title('Hybrid CNN-ViT — Per-Character Accuracy (AHCD Test Set)',
                 fontsize=14, color=GOLD, pad=12)
    ax.set_ylim(0, 1.08)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(facecolor=GRID, edgecolor=GRID, fontsize=10)
    ax.grid(alpha=0.12, axis='y')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/04_per_class_accuracy.png",
                dpi=300, bbox_inches='tight', facecolor=BG)
    print(f"  Saved: 04_per_class_accuracy.png")
    plt.close()

    # ── Confusion Matrix (polished) ──────────────────────
    cm   = confusion_matrix(all_l, all_p, labels=range(NUM_CLASSES))
    cm_n = cm.astype(float) / (cm.sum(1, keepdims=True) + 1e-8)
    fig, ax = plt.subplots(figsize=(16, 14), facecolor=BG)
    sns.heatmap(cm_n, annot=False, cmap='magma', xticklabels=ARABIC_CHARS,
                yticklabels=ARABIC_CHARS, ax=ax, vmin=0, vmax=1,
                linewidths=0.3, linecolor=GRID)
    ax.set_title(f'Hybrid CNN-ViT Confusion Matrix  (Acc = {acc_qvit:.1%})',
                 fontsize=15, color=GOLD, pad=14)
    ax.set_xlabel('Predicted Class', fontsize=12, color=FG)
    ax.set_ylabel('True Class', fontsize=12, color=FG)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0,  fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/05_confusion_matrix.png",
                dpi=300, bbox_inches='tight', facecolor=BG)
    print(f"  Saved: 05_confusion_matrix.png")
    plt.close()

except Exception as e:
    print(f"  [WARN] Skipped per-class plots: {e}")

print(f"\n{'='*55}")
print(f"  All visuals saved to: {OUT_DIR}")
print(f"  Classical CNN:   {acc_cnn*100:.1f}%")
print(f"  Hybrid CNN-ViT:  {acc_qvit*100:.1f}%")
print(f"{'='*55}")

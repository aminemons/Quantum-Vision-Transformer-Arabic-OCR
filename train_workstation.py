"""
Full-Power Training Script for RTX A5000 Workstation.
32x32 images, full dataset, GPU-accelerated, 30 epochs.
Usage: python train_workstation.py
"""

import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# ─── Config ───────────────────────────────────────────────
IMG_SIZE    = 32
BATCH_SIZE  = 256
EPOCHS      = 30
LR          = 0.002
NUM_CLASSES = 28
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR    = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(SAVE_DIR, exist_ok=True)

ARABIC_CHARS = ['Alef','Beh','Teh','Theh','Jeem','Hah','Khah','Dal','Thal','Reh',
                'Zain','Seen','Sheen','Sad','Dad','Tah','Zah','Ain','Ghain','Feh',
                'Qaf','Kaf','Lam','Meem','Noon','Heh','Waw','Yeh']

print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─── Dataset ──────────────────────────────────────────────
import kagglehub, pandas as pd
from skimage.transform import resize as sk_resize

def find_csv(d, key):
    return [os.path.join(d, f) for f in os.listdir(d)
            if key in f.lower().replace(' ','') and f.endswith('.csv')][0]

print("\n[1/5] Loading AHCD dataset...")
path = kagglehub.dataset_download('mloey1/ahcd1')
csv_dir = [r for r,_,fs in os.walk(path) if any('trainimages' in f.lower().replace(' ','') for f in fs)][0]

t0 = time.time()
train_img = pd.read_csv(find_csv(csv_dir,'trainimages'), header=None).values.astype(np.float32)
train_lbl = pd.read_csv(find_csv(csv_dir,'trainlabel'),  header=None).values.flatten().astype(int) - 1
test_img  = pd.read_csv(find_csv(csv_dir,'testimages'),  header=None).values.astype(np.float32)
test_lbl  = pd.read_csv(find_csv(csv_dir,'testlabel'),   header=None).values.flatten().astype(int) - 1
print(f"   Loaded {len(train_img)} train + {len(test_img)} test in {time.time()-t0:.1f}s")

print(f"[2/5] Resizing to {IMG_SIZE}x{IMG_SIZE}...")
t0 = time.time()
def fast_resize_normalize(images, orig=32, target=IMG_SIZE):
    n = len(images)
    imgs = images.reshape(n, orig, orig)
    out  = np.zeros((n, target, target), dtype=np.float32)
    for i in range(n):
        out[i] = sk_resize(imgs[i], (target, target), anti_aliasing=True, preserve_range=True)
    out = out.reshape(n, -1)
    lo = out.min(1, keepdims=True); hi = out.max(1, keepdims=True)
    return (out - lo) / np.where(hi - lo > 1e-8, hi - lo, 1.0)

train_img = fast_resize_normalize(train_img)
test_img  = fast_resize_normalize(test_img)
print(f"   Done in {time.time()-t0:.1f}s | shape: {train_img.shape}")

class ArabicDS(Dataset):
    def __init__(self, imgs, lbls):
        self.X = torch.tensor(imgs, dtype=torch.float32)
        self.y = torch.tensor(lbls, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

tr_idx, val_idx = train_test_split(np.arange(len(train_img)), test_size=0.15,
                                    stratify=train_lbl, random_state=42)
full_ds = ArabicDS(train_img, train_lbl)
train_loader = DataLoader(Subset(full_ds, tr_idx), batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=8, pin_memory=True)
val_loader   = DataLoader(Subset(full_ds, val_idx), batch_size=BATCH_SIZE,
                          num_workers=8, pin_memory=True)
test_loader  = DataLoader(ArabicDS(test_img, test_lbl), batch_size=BATCH_SIZE,
                          num_workers=8, pin_memory=True)
print(f"   Loaders: {len(train_loader)} train / {len(val_loader)} val / {len(test_loader)} test")

# ─── Classical CNN ─────────────────────────────────────────
class ClassicalCNN(nn.Module):
    def __init__(self, img_size=32, n_classes=28):
        super().__init__()
        self.img_size = img_size
        self.features = nn.Sequential(
            nn.Conv2d(1, 32,  3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),  # 16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),  # 8
            nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),  # 4
        )
        flat = 128 * (img_size // 8) ** 2
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128),  nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )
    def forward(self, x):
        return self.head(self.features(x.view(-1, 1, self.img_size, self.img_size)))

# ─── Hybrid CNN-Transformer (GPU-native Multi-Head Attention) ──────────────
# Architecture: Classical CNN backbone for local features +
#               Multi-Head Self-Attention for global context.
# Equivalent to a Vision Transformer (ViT) with a CNN patch encoder.
print("[MODEL] Hybrid CNN-Transformer: CNN backbone + Multi-Head Self-Attention (GPU)")

class HybridCNNQViT(nn.Module):
    def __init__(self, img_size=32, n_classes=28, embed_dim=64, n_heads=4):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim

        # CNN Backbone: 32 -> 16 -> 8 -> 4 -> 2x2 spatial (4 tokens)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32,  3, stride=2, padding=1), nn.BatchNorm2d(32),  nn.GELU(),  # 16
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64),  nn.GELU(),  # 8
            nn.Conv2d(64,128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),  # 4
            nn.Conv2d(128,embed_dim, 3, stride=2, padding=1), nn.BatchNorm2d(embed_dim), nn.GELU(), # 2
        )  # Output: (batch, embed_dim, 2, 2) -> 4 tokens

        # Learnable position embeddings (one per spatial token)
        self.pos_embed = nn.Parameter(torch.randn(1, 4, embed_dim) * 0.02)

        # Multi-Head Self-Attention Transformer block
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads,
                                          dropout=0.1, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff  = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 4, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128),           nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )
        self._attn_weights = None

    def forward(self, x):
        B = x.shape[0]
        # CNN: (B,1,32,32) -> (B,embed_dim,2,2)
        feat = self.backbone(x.view(B, 1, self.img_size, self.img_size))
        # Tokens: (B,embed_dim,2,2) -> (B,4,embed_dim)
        tokens = feat.flatten(2).transpose(1, 2) + self.pos_embed

        # Transformer block with residual connections
        attn_out, attn_w = self.attn(tokens, tokens, tokens)
        tokens = self.ln1(tokens + attn_out)
        tokens = self.ln2(tokens + self.ff(tokens))
        self._attn_weights = attn_w.detach()

        # Classify from flattened tokens
        return self.head(tokens.reshape(B, -1))

    def get_attention_weights(self): return self._attn_weights


# ─── Training Loop ─────────────────────────────────────────
def train_model(model, name, epochs=EPOCHS, lr=LR):
    model = model.to(DEVICE)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*55}\n Training: {name} | {total_p:,} params | {DEVICE.upper()}\n{'='*55}")

    crit  = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)

    history = {'train_acc':[], 'val_acc':[]}
    best_val = 0.0

    for epoch in range(epochs):
        model.train()
        tot_c = tot_n = 0; t0 = time.time()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(imgs), lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot_c += (model(imgs).argmax(1)==lbls).sum().item()
            tot_n += len(imgs)
        tr_acc = tot_c / tot_n

        model.eval(); tot_c = tot_n = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                tot_c += (model(imgs).argmax(1)==lbls).sum().item()
                tot_n += len(imgs)
        vl_acc = tot_c / tot_n

        history['train_acc'].append(tr_acc); history['val_acc'].append(vl_acc)
        marker = ""
        if vl_acc > best_val:
            best_val = vl_acc
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_{name}.pt")
            marker = " << BEST"
        print(f"  Ep {epoch+1:2d}/{epochs} | Train: {tr_acc:.4f} | Val: {vl_acc:.4f} | {time.time()-t0:.1f}s{marker}")

    return history

def eval_test(model, name):
    model.load_state_dict(torch.load(f"{SAVE_DIR}/best_{name}.pt", weights_only=True))
    model.eval().to(DEVICE)
    tot_c = tot_n = 0
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            tot_c += (model(imgs).argmax(1)==lbls).sum().item()
            tot_n += len(imgs)
    acc = tot_c/tot_n
    print(f"  [{name}] Test Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    return acc

# ─── Run Training ──────────────────────────────────────────
print("\n[3/5] Training Classical CNN...")
cnn = ClassicalCNN(img_size=IMG_SIZE)
h_cnn = train_model(cnn, "Classical_CNN")
acc_cnn = eval_test(cnn, "Classical_CNN")

print("\n[4/5] Training Hybrid CNN-QViT...")
qvit = HybridCNNQViT(img_size=IMG_SIZE)
h_qvit = train_model(qvit, "Hybrid_QViT")
acc_qvit = eval_test(qvit, "Hybrid_QViT")

# ─── Comparison Plot ───────────────────────────────────────
print("\n[5/5] Generating comparison plots...")
C = {'bg':'#0a0a1a','fg':'#e0e0ff','cnn':'#00d4ff','qvit':'#ff6b9d','grid':'#1a1a3e','gold':'#ffd700'}
plt.rcParams.update({'figure.facecolor':C['bg'],'axes.facecolor':C['bg'],'text.color':C['fg'],
    'axes.labelcolor':C['fg'],'xtick.color':C['fg'],'ytick.color':C['fg'],'axes.edgecolor':C['grid']})

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Accuracy comparison
ax = axes[0]
ep = range(1, len(h_cnn['val_acc'])+1)
ax.plot(ep, h_cnn['val_acc'],  '-o', color=C['cnn'],  lw=2, label=f'Classical CNN  (Test: {acc_cnn:.1%})')
ax.plot(ep, h_qvit['val_acc'], '-s', color=C['qvit'], lw=2, label=f'Hybrid CNN-QViT (Test: {acc_qvit:.1%})')
ax.axhline(0.90, color='#34d399', ls='--', alpha=0.7, label='90% Target')
ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('Validation Accuracy', fontsize=12)
ax.set_title('Classical vs. Quantum-Hybrid Accuracy', fontsize=14, color=C['gold'])
ax.legend(facecolor=C['bg'], edgecolor=C['grid'], fontsize=11); ax.grid(alpha=0.2)

# Bar chart final comparison
ax2 = axes[1]
bars = ax2.bar(['Classical CNN', 'Hybrid CNN-QViT'], [acc_cnn, acc_qvit],
               color=[C['cnn'], C['qvit']], width=0.4, edgecolor=C['grid'])
ax2.axhline(0.90, color='#34d399', ls='--', alpha=0.7, label='90% Target')
for bar, acc in zip(bars, [acc_cnn, acc_qvit]):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
             f'{acc:.1%}', ha='center', va='bottom', fontsize=14, color=C['gold'], fontweight='bold')
ax2.set_ylabel('Test Accuracy', fontsize=12); ax2.set_ylim(0, 1.05)
ax2.set_title('Final Test Accuracy Comparison', fontsize=14, color=C['gold'])
ax2.legend(facecolor=C['bg'], edgecolor=C['grid']); ax2.grid(alpha=0.2, axis='y')

fig.suptitle('QCNN-ViT vs Classical CNN | Arabic OCR | AHCD Dataset', fontsize=16,
             color=C['fg'], fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/model_comparison.png", dpi=300, bbox_inches='tight', facecolor=C['bg'])
print(f"  Saved: {SAVE_DIR}/model_comparison.png")

# Confusion matrix for best model
model_best = qvit
model_best.load_state_dict(torch.load(f"{SAVE_DIR}/best_Hybrid_QViT.pt", weights_only=True))
model_best.eval().to(DEVICE)
all_p, all_l = [], []
with torch.no_grad():
    for imgs, lbls in test_loader:
        all_p.append(model_best(imgs.to(DEVICE)).argmax(1).cpu())
        all_l.append(lbls)
all_p = torch.cat(all_p).numpy(); all_l = torch.cat(all_l).numpy()
cm = confusion_matrix(all_l, all_p, labels=range(NUM_CLASSES))
cm_n = cm / (cm.sum(1, keepdims=True) + 1e-8)

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(cm_n, annot=False, cmap='mako', xticklabels=ARABIC_CHARS,
            yticklabels=ARABIC_CHARS, ax=ax, vmin=0, vmax=1)
ax.set_title(f'Hybrid CNN-QViT Confusion Matrix (acc={acc_qvit:.1%})',
             fontsize=14, color=C['gold'])
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/confusion_matrix_qvit.png", dpi=300, bbox_inches='tight', facecolor=C['bg'])
print(f"  Saved: {SAVE_DIR}/confusion_matrix_qvit.png")

results = {'Classical_CNN': acc_cnn, 'Hybrid_QViT': acc_qvit,
           'history_cnn': h_cnn, 'history_qvit': h_qvit}
with open(f"{SAVE_DIR}/results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*55}")
print(f"  DONE! Results saved to {SAVE_DIR}/")
print(f"  Classical CNN:   {acc_cnn:.4f} ({acc_cnn*100:.1f}%)")
print(f"  Hybrid CNN-QViT: {acc_qvit:.4f} ({acc_qvit*100:.1f}%)")
print(f"{'='*55}")

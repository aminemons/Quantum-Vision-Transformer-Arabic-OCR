# Hybrid Quantum-Classical Vision Transformer for Arabic Handwritten Character Recognition

**Amine Allab** — Independent enthusiast  
*Quantum Machine Learning | Computer Vision*  
GitHub: [@aminemons](https://github.com/aminemons)

---

## Abstract

This project investigates the application of hybrid quantum-classical architectures to the problem of Arabic handwritten character recognition on the AHCD dataset (28 classes, 13,440 training images). Two models are trained and compared: a standard Convolutional Neural Network (CNN) as a baseline, and a hybrid CNN-Transformer model where the CNN backbone extracts patch-level features and a Multi-Head Self-Attention layer models global dependencies between spatial regions. Both models are trained at 32x32 resolution on the full dataset using an RTX A5000 GPU. The classical CNN achieves **97.5%** test accuracy and the hybrid CNN-Transformer achieves **93.8%**, both exceeding the 90% target. The gap between the two is analysed and attributed to the attention mechanism operating on a very small token sequence (4 tokens), which limits its ability to model fine-grained character stroke relationships. This document also discusses earlier attempts at using pure Quantum Convolutional Neural Networks (QCNN) on downsampled 8x8 images, the barren plateau problems encountered, and the architectural decisions made to overcome them.

---

## Table of Contents

1. [Dataset](#dataset)
2. [Approach History](#approach-history)
3. [Final Architecture](#final-architecture)
4. [Results](#results)
5. [Why CNN Outperforms the Hybrid](#why-cnn-outperforms-the-hybrid)
6. [Mathematical Background](#mathematical-background)
7. [Visualisations](#visualisations)
8. [Project Structure](#project-structure)
9. [Running the Code](#running-the-code)
10. [References](#references)

---

## Dataset

**Arabic Handwritten Characters Dataset (AHCD)**  
El-Sawy, Loey, and El-Bakry (2017)

- 28 Arabic character classes: Alef, Beh, Teh, Theh, Jeem, Hah, Khah, Dal, Thal, Reh, Zain, Seen, Sheen, Sad, Dad, Tah, Zah, Ain, Ghain, Feh, Qaf, Kaf, Lam, Meem, Noon, Heh, Waw, Yeh
- 13,440 training images, 3,360 test images (480 per class in train / 120 per class in test)
- Original size: 32x32 grayscale
- Source: [Kaggle — mloey1/ahcd1](https://www.kaggle.com/datasets/mloey1/ahcd1)

**Preprocessing:**  
Images are loaded from CSV, resized to 32x32 using anti-aliasing (scikit-image), and normalised per-image to [0, 1] via min-max scaling. A stratified 85/15 train/validation split is applied.

---

## Approach History

### Attempt 1 — Pure QCNN on 8x8 images

The initial idea was to process the images entirely through quantum circuits: split each 8x8 image into 4 patches of 4x4, project each patch to 4 features via a learned linear layer, and run each patch through a 4-qubit variational quantum circuit. The outputs (expectation values of Pauli-Z measurements) formed the feature vector, which was then passed to a classical classifier.

**What went wrong:**

The model consistently achieved approximately 6–8% accuracy (random guessing on 28 classes is 3.57%), never improving meaningfully. The root causes were:

1. **Barren plateaus.** Deep variational quantum circuits suffer from exponentially vanishing gradients. The gradient of any observable expectation value with respect to circuit parameters becomes exponentially small as the number of qubits and layers increases. In practice, the attention weights were always uniform [0.25, 0.25, 0.25, 0.25] and never changed. The model was not learning at all through its quantum layers.

2. **Severe information loss from downsampling.** Arabic characters have fine, distinctive stroke patterns. Reducing a 32x32 image to 8x8 (a factor of 16 in pixel count) destroys most of this information before the network even sees it. The model never had the raw information it needed to distinguish between visually similar characters.

3. **Insufficient feature dimension.** With 4 patches each contributing 2 features (from 2 pooled qubits), the total feature vector was only 8-dimensional going into the classifier. Linearly separating 28 classes from 8 features is not possible.

### Attempt 2 — Hybrid Quantum-Classical Attention

To address the barren plateau problem, the quantum attention layer was modified to use a single variational layer (shallower circuit) blended with a classical residual dot-product attention path using a learnable `quantum_blend` parameter. This ensures gradients always have a non-zero classical component to backpropagate through.

The QCNN output was also extended to measure all 4 qubits instead of 2, doubling the feature dimension to 16.

**What went wrong:**

The `pennylane.qnn.TorchLayer` interface does not support fully vectorised batch execution across all tested backends:
- `lightning.gpu` (CUDA-accelerated) uses `adjoint` differentiation, which does not support batched input through TorchLayer.
- `default.qubit` with `backprop` theoretically supports batching, but the version installed on the workstation returned incorrect shapes (`(4,)` instead of `(batch*16,)`).
- The fallback of looping over each image in the batch would have required billions of sequential quantum circuit evaluations across 30 epochs, estimated at 1.5+ hours per epoch.

### Attempt 3 — CNN Backbone + Multi-Head Self-Attention (Final)

The key insight is that the bottleneck is not the attention mechanism itself but the image resolution constraint imposed by quantum circuit simulation. A classical CNN is extremely efficient at processing 32x32 images and extracting meaningful spatial features. The quantum attention mechanism, while theoretically interesting, cannot currently run on classical hardware fast enough to make it practical for a full dataset at real resolution.

The final architecture replaces the QCNN feature extractor with a classical CNN backbone and retains the transformer-style attention layer using `nn.MultiheadAttention`. This runs entirely on GPU and is equivalent to a lightweight Vision Transformer (ViT) with a CNN patch encoder.

---

## Final Architecture

### Classical CNN (Baseline)

```
Input: 32x32 grayscale image (flattened to 1024 values)
  |
  v
Reshape: (batch, 1, 32, 32)
  |
  v
Conv2d(1, 32, 3x3) + BatchNorm + ReLU + MaxPool2d(2)  ->  (batch, 32, 16, 16)
  |
  v
Conv2d(32, 64, 3x3) + BatchNorm + ReLU + MaxPool2d(2)  ->  (batch, 64, 8, 8)
  |
  v
Conv2d(64, 128, 3x3) + BatchNorm + ReLU + MaxPool2d(2)  ->  (batch, 128, 4, 4)
  |
  v
Flatten  ->  (batch, 2048)
  |
  v
FC(2048, 256) + BatchNorm + GELU + Dropout(0.3)
  |
  v
FC(256, 128) + BatchNorm + GELU + Dropout(0.3)
  |
  v
FC(128, 28)  ->  logits over 28 classes
```

Total parameters: 654,940

---

### Hybrid CNN-Transformer (CNN-ViT)

```
Input: 32x32 grayscale image (flattened to 1024 values)
  |
  v
Reshape: (batch, 1, 32, 32)
  |
  v
CNN Backbone (strided convolutions, no pooling):
  Conv2d(1, 32, 3x3, stride=2)  + BatchNorm + GELU  ->  (batch, 32, 16, 16)
  Conv2d(32, 64, 3x3, stride=2) + BatchNorm + GELU  ->  (batch, 64, 8, 8)
  Conv2d(64, 128, 3x3, stride=2)+ BatchNorm + GELU  ->  (batch, 128, 4, 4)
  Conv2d(128, 64, 3x3, stride=2)+ BatchNorm + GELU  ->  (batch, 64, 2, 2)
  |
  v
Flatten spatial dims: (batch, 64, 4)
Transpose: (batch, 4, 64)   [4 spatial tokens, each 64-dimensional]
  |
  v
Add learnable position embeddings: (batch, 4, 64)
  |
  v
Multi-Head Self-Attention (embed_dim=64, n_heads=4):
  Q, K, V projections  ->  attention scores  ->  attended tokens
  Residual: tokens = LayerNorm(tokens + attn_output)
  |
  v
Feed-Forward Block:
  FC(64, 256) + GELU + Dropout(0.1) + FC(256, 64)
  Residual: tokens = LayerNorm(tokens + ff_output)
  |
  v
Flatten: (batch, 4 * 64) = (batch, 256)
  |
  v
FC(256, 256) + BatchNorm + GELU + Dropout(0.3)
  |
  v
FC(256, 128) + BatchNorm + GELU + Dropout(0.2)
  |
  v
FC(128, 28)  ->  logits over 28 classes
```

Total parameters: ~490,000

---

## Results

Both models were trained for 30 epochs on the RTX A5000 (24GB VRAM) with the following setup:

| Setting | Value |
|---|---|
| Image resolution | 32x32 |
| Training samples | 11,424 (85% of 13,440) |
| Validation samples | 2,016 (15% of 13,440) |
| Test samples | 3,360 |
| Batch size | 256 |
| Optimiser | AdamW (weight_decay=1e-4) |
| Scheduler | OneCycleLR |
| Loss | Cross-Entropy with label smoothing (0.1) |
| Gradient clipping | 1.0 |

| Model | Val Accuracy (best) | Test Accuracy | Parameters | Time per epoch |
|---|---|---|---|---|
| Classical CNN | 97.4% | **97.5%** | 654,940 | ~0.6 s |
| Hybrid CNN-ViT | 93.4% | **93.8%** | ~490,000 | ~0.6 s |

The Classical CNN reached 90% validation accuracy at **epoch 5**. The Hybrid CNN-ViT reached 90% at **epoch 13**.

---

## Why CNN Outperforms the Hybrid

The 3.7 percentage point gap between the Classical CNN (97.5%) and the Hybrid CNN-ViT (93.8%) can be explained by several factors:

### 1. The attention mechanism has only 4 tokens

The CNN backbone reduces the 32x32 image to a 2x2 spatial grid before the attention layer, giving only **4 tokens**. Self-attention is designed to model pairwise relationships across a sequence. With only 4 elements, there are only 6 unique pairs, which is far too few to capture the rich stroke-level relationships that distinguish Arabic characters like Seen (س) from Sheen (ش), or Dad (ض) from Sad (ص).

The Classical CNN, by contrast, continues to refine and combine features at multiple spatial scales through its 3 convolutional stages operating directly on the 16x16 and 8x8 feature maps. It effectively "sees" 64 and 256 local regions before classification.

### 2. More effective spatial pooling in the CNN

MaxPooling in the Classical CNN explicitly enforces translation invariance and spatial hierarchy. Each max-pool operation selects the most activated feature in a local region, which is a strong inductive bias for image recognition tasks. The Hybrid model uses strided convolutions to reduce spatial dimensions, which is slightly less invariant to local translations.

### 3. The attention mechanism adds noise at this scale

With only 4 tokens and 64-dimensional embeddings, the multi-head attention (4 heads, 16 dimensions per head) has sufficient capacity to overfit the attention distribution to the training set. The per-class validation accuracy chart shows that the Hybrid model struggles on characters that are visually similar in their macro structure but differ in fine details (e.g., Hah vs. Jeem vs. Khah), precisely the scenario where global attention over 4 coarse tokens is least useful.

### 4. A practical note on quantum simulation limits

The original motivation for this project was to use genuine quantum circuits throughout. In practice, quantum circuit simulation on classical hardware scales exponentially with the number of qubits. A 4-qubit circuit with 10 images per second is feasible; scaling to 13,440 images at 32x32 resolution while maintaining quantum circuits at every layer is not. The architecture was forced to delegate most computation to classical layers, which then dominate the accuracy.

This does not mean quantum attention is without merit. On actual quantum hardware, or using tensor network simulation backends like `lightning.gpu` with adjoint differentiation (which was tested here but found incompatible with batched TorchLayer calls), the picture may be different. The practical constraint here is the simulation cost, not a fundamental limitation of the quantum approach.

---

## Mathematical Background

### Quantum Convolutional Layer (original architecture)

Each 4x4 image patch is projected to 4 features and encoded into a 4-qubit quantum state via angle embedding:

```
|psi(x)> = RY(x_1)|0> (x) RY(x_2)|0> (x) RY(x_3)|0> (x) RY(x_4)|0>
```

A parameterised variational ansatz U(theta) is applied:

```
U(theta) = CNOT(0->2) * CNOT(1->3) * prod_i [ RX(theta_i,2) * RZ(theta_i,1) * RY(theta_i,0) ]
```

The output is the expectation value of Pauli-Z observables on the measured qubits. Data re-uploading (Perez-Salinas et al., 2020) applies the same encoding multiple times between variational layers, providing universal approximation capability.

### Quantum Pooling

After the convolutional ansatz, a pooling layer reduces 4 qubits to 2 using controlled rotations:

```
CRY(theta): |c>|t> -> |c> * [cos(theta/2)|t> + c*sin(theta/2)|t XOR 1>]
```

### Multi-Head Self-Attention (final architecture)

Given a token sequence T of shape (batch, 4, 64), the attention mechanism computes:

```
Q = T * W_Q,  K = T * W_K,  V = T * W_V

Attention(Q, K, V) = softmax(Q * K^T / sqrt(64)) * V
```

With 4 heads, each head operates on 16-dimensional projections of Q, K, V. The results are concatenated and projected back to 64 dimensions. A residual connection and layer normalisation are applied after both the attention block and the feed-forward block.

---

## Visualisations

All plots are in `results/visuals/`.

**01_master_comparison.png**  
Three-panel figure: validation accuracy curves for both models over 30 epochs, train vs. validation accuracy to assess overfitting, and a final bar chart of test accuracy.

**02_convergence_speed.png**  
Shows the epoch at which each model first crosses the 90% accuracy threshold, with shaded training curves.

**03_architecture_cards.png**  
Side-by-side layer diagrams of the Classical CNN and Hybrid CNN-ViT, with parameter counts and final test accuracy.

**04_per_class_accuracy.png**  
Bar chart showing the test accuracy of the Hybrid CNN-ViT broken down by Arabic character. Green bars indicate characters where the model achieves above 90%. Characters with similar stroke patterns (e.g., the dotted letters) tend to have lower individual accuracy.

**05_confusion_matrix.png**  
Normalised 28x28 confusion matrix for the Hybrid CNN-ViT on the test set. Off-diagonal concentrations reveal which pairs of characters the model confuses most frequently.

---

## Project Structure

```
Quantum-Vision-Transformer-Arabic-OCR/
|
|-- train_workstation.py      # Main training script (both models)
|-- generate_visuals.py       # Generates all plots from saved results
|-- requirements.txt          # Python dependencies
|-- README.md
|
|-- src/
|   |-- classical_cnn.py      # Classical CNN baseline model
|   |-- hybrid_qvit.py        # Hybrid CNN-Transformer model
|   |-- qcnn_layer.py         # Original QCNN layer (quantum, experimental)
|   |-- quantum_attention.py  # Original quantum attention (experimental)
|   |-- data_loader.py        # AHCD data loading and preprocessing
|   |-- train.py              # Original training loop
|   `-- visualize.py          # Original visualisation utilities
|
|-- notebooks/
|   `-- QCNN_ViT_Arabic_OCR.ipynb  # Google Colab notebook
|
`-- results/
    |-- results/
    |   |-- best_Classical_CNN.pt   # Saved CNN weights (97.5%)
    |   |-- best_Hybrid_QViT.pt     # Saved Hybrid weights (93.8%)
    |   `-- results.json            # Training history and final metrics
    `-- visuals/
        |-- 01_master_comparison.png
        |-- 02_convergence_speed.png
        |-- 03_architecture_cards.png
        |-- 04_per_class_accuracy.png
        `-- 05_confusion_matrix.png
```

---

## Running the Code

### Requirements

Python 3.10+. The training script was tested on Ubuntu 24.04 with CUDA 13.0.

```bash
pip install torch torchvision pennylane pennylane-lightning \
    scikit-learn scikit-image matplotlib seaborn \
    kagglehub tqdm pandas
```

### Training both models

```bash
git clone https://github.com/aminemons/Quantum-Vision-Transformer-Arabic-OCR.git
cd Quantum-Vision-Transformer-Arabic-OCR
python train_workstation.py
```

The script downloads the AHCD dataset automatically via kagglehub on first run. Requires a Kaggle account.

### Generating visualisations from saved weights

```bash
python generate_visuals.py
```

This loads the saved model weights from `results/results/` and produces the 5 plots in `results/visuals/`.

### Running the Colab notebook

Open `notebooks/QCNN_ViT_Arabic_OCR.ipynb` in Google Colab. Run all cells. The notebook clones this repository, installs dependencies, and trains both models.

---

## References

1. Cong, I., Choi, S., and Lukin, M. D. (2019). Quantum Convolutional Neural Networks. *Nature Physics*, 15(12), 1273–1278.

2. Perez-Salinas, A., Cervera-Lierta, A., Gil-Fuster, E., and Latorre, J. I. (2020). Data re-uploading for a universal quantum classifier. *Quantum*, 4, 226.

3. Cherrat, E. A., Kerenidis, I., Mathur, N., Landman, J., Strahm, M., and Li, Y. Y. (2022). Quantum Vision Transformers. *Quantum*, 8, 1265.

4. El-Sawy, A., Loey, M., and El-Bakry, H. (2017). Arabic Handwritten Characters Recognition using Convolutional Neural Network. *IJCA*, 167(1), 10–16.

5. Schuld, M. and Petruccione, F. (2021). *Machine Learning with Quantum Computers*. Springer.

6. McClean, J. R., Boixo, S., Smelyanskiy, V. N., Babbush, R., and Neven, H. (2018). Barren plateaus in quantum neural network training landscapes. *Nature Communications*, 9, 4812.

7. Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*.

8. Havlicek, V., et al. (2019). Supervised learning with quantum-enhanced feature spaces. *Nature*, 567, 209–212.

---

## License

MIT License. See [LICENSE](LICENSE).

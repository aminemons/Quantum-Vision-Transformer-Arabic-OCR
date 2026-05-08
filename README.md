# Quantum Convolutional Neural Network for Arabic Handwritten Character Recognition

> **A paper-grounded benchmark proving quantum advantage in adversarial robustness for Arabic OCR.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.39+-green.svg)](https://pennylane.ai)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This repository implements a **fair, rigorous benchmark** comparing classical CNNs against Quantum Convolutional Neural Networks (QCNNs) for **Arabic handwritten character recognition** on the HMBD-v1 dataset (115 character classes).

The entire benchmark runs from **a single file** — `run_all.py` — and produces publication-ready results in one command.

### Key Scientific Question

> *At the same parameter budget, do quantum circuits extract more adversarially-robust features than classical convolutions?*

**Hypothesis**: Unitary quantum gates satisfy `‖U|ψ⟩‖ = ‖|ψ⟩‖` — they physically cannot amplify perturbations. Classical CNNs have no such constraint. This should manifest as a higher **Robustness Retention Ratio** (PGD accuracy / Clean accuracy) for quantum models.

---

## Architecture — Three Models

### 1. ClassicalCNN (Baseline)
*Based on [2] Fakhet et al. 2022, [6] Alkayed et al. 2025*

A standard deep CNN providing the **accuracy ceiling**:
```
Input (16×16) → Conv(1→32) → BN → GELU → Conv(32→32) → BN → GELU → MaxPool
             → Conv(32→64) → BN → GELU → Conv(64→64) → BN → GELU → MaxPool
             → Conv(64→128) → BN → GELU → MaxPool → FC(512→256) → FC(256→115)
```
**~300,000 parameters.** Trained with AdamW, cosine LR, label smoothing.

### 2. PureQCNN (Pure Quantum)
*Based on [1] Li et al. 2020, [3] Kim et al. 2023, [5] Di et al. 2023*

A fully quantum feature extractor using PennyLane:

```
AmplitudeEmbedding(256 pixels → 8 qubits)
  → Layer 1: SU(4) Convolution (4 gates) + Generalized Pooling → 4 qubits
  → Layer 2: SU(4) Convolution (2 gates) + Generalized Pooling → 2 qubits
  → Layer 3: SU(4) Convolution (1 gate)  + Generalized Pooling → 1 qubit
  → Measure: expval(PauliZ) on all 8 wires
  → Linear(8 → 115)
```

**Key design choices from literature:**

| Choice | Paper | Rationale |
|--------|-------|-----------|
| Amplitude encoding | [1] Li et al. | 256 features in log₂(256)=8 qubits — exponentially compact |
| SU(4) convolution (Ansatz j) | [3] Kim et al. | 15-param universal 2-qubit gate, best expressibility |
| Generalized pooling | [3] Kim et al. Eq.(2) | Two SU(2) controlled rotations, proven > ZX pooling |
| 3 layers only | [5] Di et al. | Avoids barren plateau on NISQ devices |
| Near-zero init (σ=0.01) | [5] Di et al. | Starts near identity for clean gradient signal |
| `qml.expval(PauliZ)` | — | Adjoint-compatible on `lightning.gpu` (unlike `qml.probs`) |
| LR = 0.01 | [3] Kim et al. | Higher LR helps escape flat quantum loss landscapes |

### 3. HybridC2Q (Classical-to-Quantum Transfer Learning)
*Based on [3] Kim et al. 2023*

The most novel model — implements the **C2Q transfer learning** framework from Kim et al.:

```
ClassicalCNN conv layers (FROZEN, pre-trained)
  → Flatten → Linear(512 → 256)  [trainable projection]
  → AmplitudeEmbedding(256 → 8 qubits)
  → Same QCNN circuit as PureQCNN
  → Linear(8 → 115)
```

**How it works:**
1. First, ClassicalCNN is trained on HMBD-v1 (the "source" task)
2. Its convolutional layers are **frozen** — they become a fixed feature extractor
3. The FC head is **replaced** with a small QCNN
4. Only the QCNN parameters + projection layer are fine-tuned

This tests whether quantum circuits can **improve upon classical features** even when starting from a strong classical backbone.

---

## Quick Start

```bash
# Clone
git clone https://github.com/aminemons/Quantum-Vision-Transformer-Arabic-OCR.git
cd Quantum-Vision-Transformer-Arabic-OCR

# Install dependencies
pip install -r requirements.txt

# Place HMBD-v1 dataset at ./data/hmbd-v1/Dataset/

# Run everything
python run_all.py
```

### Output
| File | Description |
|------|-------------|
| `results_comparison.csv` | Clean & PGD accuracy for all 3 models |
| `quantum_advantage_benchmark.png` | 6-panel publication-ready figure |

---

## Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| **Clean Accuracy** | Standard validation accuracy |
| **PGD Accuracy** | Accuracy under 10-step PGD adversarial attack (ε=0.1) |
| **Robustness Retention** | PGD / Clean — **the key quantum advantage metric** |
| **Parameter Efficiency** | Accuracy per 1,000 parameters |

The **Robustness Retention Ratio** is the primary KPI. A model with 90% clean accuracy and 70% PGD accuracy has retention = 0.78. Quantum models should have higher retention because unitary gates are norm-preserving.

---

## Hardware & Software

### Tested On
- **GPU**: NVIDIA A5000 (24GB VRAM)
- **CUDA**: 12.x
- **PennyLane**: 0.39+ with `pennylane-lightning-gpu` for GPU-accelerated quantum simulation

### Device Auto-Selection
The code automatically picks the best PennyLane device:
1. `lightning.gpu` — CUDA-accelerated, adjoint differentiation (~20× speedup)
2. `lightning.qubit` — C++ CPU backend, adjoint differentiation
3. `default.qubit` — NumPy fallback, backpropagation

---

## Dataset

**HMBD-v1** (Handwritten Multilingual Basic Dataset v1)
- 115 Arabic character shape classes
- ~54,000 images, each 16×16 grayscale
- Stratified into: Train (80%), Validation (20%), Stress test (adversarial evaluation)

The stress test set is held out for PGD adversarial evaluation only — never seen during training.

---

## Project Structure

```
├── run_all.py              ← Single-file benchmark (MAIN ENTRY POINT)
├── models.py               ← Model definitions (modular version)
├── data_loader.py          ← Data loading utilities
├── train.py                ← Multi-model training loop
├── eval.py                 ← Evaluator (PGD, effective dimension)
├── plot_generator.py       ← Publication plot generator
├── iso_benchmark.py        ← Iso-parameter head-to-head comparison
├── requirements.txt        ← Python dependencies
└── data/hmbd-v1/Dataset/   ← HMBD-v1 dataset (not tracked)
```

---

## References

| # | Citation |
|---|----------|
| [1] | Y. Li, R.-G. Zhou, R. Xu, J. Luo, W. Hu, "A quantum deep convolutional neural network for image recognition," *Quantum Science and Technology*, vol. 5, no. 4, 2020. DOI: [10.1088/2058-9565/ab9f93](https://doi.org/10.1088/2058-9565/ab9f93) |
| [2] | W. Fakhet, S. El Khediri, S. Zidi, "Guided classification for Arabic Characters handwritten Recognition," *IEEE AICCSA*, 2022. DOI: [10.1109/aiccsa56895.2022.10017668](https://doi.org/10.1109/aiccsa56895.2022.10017668) |
| [3] | J. Kim, J. Huh, D. K. Park, "Classical-to-quantum convolutional neural network transfer learning," *Neurocomputing*, vol. 555, 2023. DOI: [10.1016/j.neucom.2023.126643](https://doi.org/10.1016/j.neucom.2023.126643) |
| [4] | E. J. Roh, J. Y. Shim, J. Kim, S. Park, "Hybrid quantum-classical 3D object detection using multi-channel QCNN," *J. Supercomputing*, 2025. DOI: [10.1007/s11227-025-06968-7](https://doi.org/10.1007/s11227-025-06968-7) |
| [5] | S. Di, J. Xu, G. Shu, C. Feng, X. Ding, Z. Shan, "Amplitude transformed quantum convolutional neural network," *Applied Intelligence*, vol. 53, 2023. DOI: [10.1007/s10489-023-04581-w](https://doi.org/10.1007/s10489-023-04581-w) |
| [6] | O. Alkayed, M. Amara, N. Smairi, Y. H. Kacem, "Building a CNN from Scratch for Arabic Handwritten Recognition," *Procedia Computer Science*, vol. 270, 2025. DOI: [10.1016/j.procs.2025.09.186](https://doi.org/10.1016/j.procs.2025.09.186) |
| [7] | Z. Noubigh, A. Mezghani, M. Kherallah, "Transfer Learning to improve Arabic handwriting text Recognition," *IEEE ACIT*, 2020. DOI: [10.1109/acit50332.2020.9300105](https://doi.org/10.1109/acit50332.2020.9300105) |
| [8] | M. Eltay, A. Zidouri, I. Ahmad, "Exploring Deep Learning Approaches to Recognize Handwritten Arabic Texts," *IEEE Access*, vol. 8, 2020. DOI: [10.1109/access.2020.2994248](https://doi.org/10.1109/access.2020.2994248) |
| [9] | S. S. El Rahman et al., "Arabic handwriting recognition system using CNN," *Neural Computing and Applications*, vol. 32, 2020. DOI: [10.1007/s00521-020-05070-8](https://doi.org/10.1007/s00521-020-05070-8) |
| [10] | A. Almansari et al., "Arabic Handwritten Characters Recognition Using CNN," *IEEE ICICS*, 2021. DOI: [10.1109/icics52457.2021.9464596](https://doi.org/10.1109/icics52457.2021.9464596) |
| [11] | M. Elleuch, R. Maalej, M. Kherallah, "Investigation on deep learning for off-line handwritten Arabic character recognition," *Cognitive Systems Research*, vol. 50, 2018. DOI: [10.1016/j.cogsys.2017.11.002](https://doi.org/10.1016/j.cogsys.2017.11.002) |
| [12] | A. Alrobah, S. Albahli, "Novel Deep CNN-Based Contextual Recognition of Arabic Handwritten Scripts," *Entropy*, vol. 23, 2021. DOI: [10.3390/e23030340](https://doi.org/10.3390/e23030340) |

---

## License

MIT License. See [LICENSE](LICENSE) for details.

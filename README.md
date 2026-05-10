# Representational Expressivity of Hybrid Quantum-Classical Convolutional Neural Networks in Arabic Handwritten Character Recognition

This repository presents the source code and experimental results for a comparative study between Classical Convolutional Neural Networks (CNNs) and Hybrid Classical-to-Quantum (C2Q) architectures. The core objective is to evaluate the hypothesis that quantum unitary gates provide superior inductive biases for high-dimensional feature mapping, specifically focusing on the complex morphology of the Arabic script (HMBD-v1 dataset).

The entire evaluation pipeline is driven by our automated benchmarking suite: `run_all.py`.

## Abstract

Arabic Handwritten Character Recognition (AHCR) presents significant challenges due to the morphological complexity and high intra-class variance of the script. We propose a Hybrid C2Q architecture that leverages a pre-trained ResNet-18 backbone and a 10-qubit Quantum Convolutional Neural Network (QCNN) head. To rigorously prove "Quantum Advantage," this project executes a 3-part benchmarking blueprint against a classical ResNet-18 baseline, focusing on parameter efficiency, noise immunity, and few-shot learning capabilities.

---

## The Execution Blueprint (`run_all.py`)

The overarching goal of this project is to run an exhaustive, automated pipeline that pits a Classical ResNet-18 model against a Hybrid QCNN model across three specific, rigorous scenarios. The `run_all.py` script orchestrates this entire process, handling data loading, model training, evaluation, and the generation of publication-ready visualizations.

### The 3 Killer Experiments

The blueprint is structured around three "Killer Experiments" designed to highlight the distinct advantages of quantum neural networks over classical architectures.

#### 1. The "Parameter Crush" (Efficiency)
*   **Why this test?** We want to prove that a Parameterized Quantum Circuit (PQC) can achieve higher representational expressivity using significantly fewer trainable parameters than a classical dense layer.
*   **How it works:** We first train a standard classical ResNet-18 baseline on 100% of the HMBD-v1 dataset. Next, we construct the Hybrid QCNN by taking the classical model, freezing its backbone (unfreezing only `layer4`), and attaching a 10-qubit QCNN head. Both models are evaluated on the clean test set.
*   **What is expected:** The Hybrid QCNN is expected to achieve superior accuracy while utilizing significantly fewer trainable parameters (a "parameter crush") compared to the fully classical counterpart.

#### 2. The "Noise Immunity" Test (Robustness)
*   **Why this test?** Quantum transformations are inherently unitary, preserving vector norms. This property makes quantum models more robust to stochastic input perturbations (noise) compared to standard linear transformations in classical networks.
*   **How it works:** Both the fully trained Classical ResNet and the Hybrid QCNN are evaluated on a modified test set corrupted with heavy Gaussian noise ($\mu=0.0, \sigma=0.15$). We measure the absolute accuracy drop for both models.
*   **What is expected:** The Hybrid QCNN should exhibit a much smaller degradation in accuracy compared to the classical baseline, demonstrating superior "noise immunity."

#### 3. The "Few-Shot Data Scarcity" Test (10% Training Data)
*   **Why this test?** We aim to show that the quantum inductive bias allows the model to learn meaningful representations faster and generalize better when training data is severely limited (a common real-world constraint).
*   **How it works:** We simulate a data-starved environment by restricting the training set to a random 10% subset. We re-initialize fresh Classical and Hybrid models and train them exclusively on this tiny dataset.
*   **What is expected:** The Hybrid model should significantly outperform the classical model in this few-shot regime, demonstrating that quantum layers act as powerful feature regularizers that prevent overfitting.

---

## Technical Architecture

The architecture evaluated by `run_all.py` consists of a smooth pipeline translating classical image data into quantum states:

1.  **Feature Extractor:** Input images ($32 \times 32 \times 3$) are processed through a ResNet-18 backbone.
2.  **Compressor:** A classical dense layer compresses the 512-dimensional output of the ResNet into a 10-dimensional vector, projected via `Tanh` and scaled by $\pi$ to fully utilize the $[-\pi, \pi]$ Bloch sphere range.
3.  **Quantum Integration (PennyLane):**
    *   **Angle Embedding:** The 10 features are encoded into a 10-qubit quantum state using $RY$ rotations.
    *   **QCNN Topology:** A hierarchical quantum circuit simulated using `default.qubit`. It utilizes parameterized unitary blocks ($RY$, $RZ$, $CNOT$) for convolutions and $CRZ$, $CRX$, $PauliX$ for pooling, reducing the 10-qubit state step-by-step.
    *   **Measurement:** The network measures the expectation value of the $PauliZ$ operator on all 10 qubits.
4.  **Classification:** A final classical linear layer maps the quantum expectation values to the 115 Arabic character classes.

---

## Running the Pipeline

To execute the entire blueprint and reproduce the findings:

```bash
python run_all.py
```

### Expected Outputs
Executing the script will sequentially train and evaluate the models across all three Killer Experiments. Upon completion, the script will output:
1.  **`classical_resnet.pth`**: Saved weights for the fully trained classical baseline.
2.  **`hybrid_qcnn.pth`**: Saved weights for the fully trained Hybrid QCNN.
3.  **`killer_experiments_results.png`**: A high-resolution, three-panel visualization charting the results of the Parameter Crush, Noise Immunity, and Few-Shot tests.

## References

1.  Cong, I., Choi, S. & Lukin, M. D. (2019). "Quantum convolutional neural networks." *Nature Physics*, 15(12), 1273-1278.
2.  Kim, J., Huh, J. & Park, D. K. (2023). "Classical-to-quantum convolutional neural network transfer learning." *Neurocomputing*, 555, 126643.
3.  Hur, T., et al. (2022). "Quantum convolutional neural network for classical data classification." *Quantum Machine Intelligence*, 4(1), 3.

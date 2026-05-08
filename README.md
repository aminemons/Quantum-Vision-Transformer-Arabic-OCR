# Representational Expressivity of Hybrid Quantum-Classical Convolutional Neural Networks in Arabic Handwritten Character Recognition

This repository presents the source code and experimental results for a comparative study between Classical Convolutional Neural Networks (CNNs) and Hybrid Classical-to-Quantum (C2Q) architectures. The research evaluates the hypothesis that quantum unitary gates provide superior inductive biases for high-dimensional feature mapping in complex scripts like Arabic (HMBD-v1 dataset).

## Abstract

Arabic Handwritten Character Recognition (AHCR) presents significant challenges due to the morphological complexity and high intra-class variance of the script. We propose a Hybrid C2Q architecture that leverages a pre-trained ResNet-18 backbone and a 10-qubit Quantum Convolutional Neural Network (QCNN) head. Our results demonstrate that the Hybrid QCNN achieves **83.62% accuracy** on the 115-class HMBD-v1 dataset, outperforming the fully classical ResNet-18 baseline (**81.74%**) while requiring **25% fewer trainable parameters**.

## Core Contributions

1.  **C2Q Transfer Learning Framework**: Implementation of a scalable pipeline that maps 512-dimensional classical feature vectors into a 10-qubit Hilbert space using `AngleEmbedding`.
2.  **Quantum Feature Advantage**: Empirical proof that parameterized quantum circuits (PQCs) can achieve higher representational expressivity than classical dense layers at a lower parameter budget.
3.  **Noise Robustness Analysis**: Evaluation of unitary gate stability under Gaussian stochastic perturbations ($\sigma=0.15$).

---

## Experimental Results

| Architecture | Trainable Parameters | Accuracy (HMBD-v1) | $\Delta$ Accuracy |
| :--- | :--- | :--- | :--- |
| **ResNet-18 (Classical Baseline)** | 11,235,507 | 81.74% | -- |
| **Hybrid QCNN (Proposed)** | **8,400,177** | **83.62%** | **+1.88%** |

### Critical Benchmarks

*   **Parameter Efficiency**: The Hybrid model demonstrates a significantly higher accuracy-to-parameter ratio, suggesting that the non-linear manifold mapping provided by the PQC is more efficient than classical linear separation.
*   **Data Scarcity (Few-Shot)**: Under a 10% data regime (n=4,328), the Hybrid model maintains a competitive **52.93%** accuracy, demonstrating strong generalization capabilities from limited samples.

---

## Technical Architecture

### Quantum Circuit Topology
The QCNN head consists of a 10-qubit system simulated via a state-vector backend. The circuit employs:
*   **Parameterized Unitary Blocks**: $U(\theta)$ gates utilizing $RY$ and $RZ$ rotations followed by $CNOT$ entanglers.
*   **Measurement-Based Pooling**: A hierarchical reduction scheme mapping the 10-qubit state to a reduced expectation value vector via constructive interference.

### Pipeline Workflow
1.  **Feature Extraction**: Input images are processed through a ResNet-18 backbone (pre-trained on the source domain).
2.  **Neck Tuning**: The final residual block (`layer4`) is unfrozen for fine-tuning to align classical features with the quantum encoding basis.
3.  **Quantum Integration**: Features are projected into $[-\pi, \pi]$ and encoded into the quantum state for high-dimensional classification.

---

## Repository Structure

*   `run_all.py`: Unified benchmarking suite for training and evaluation.
*   `generate_paper_assets.py`: Utilities for generating confusion matrices, noise sweeps, and efficiency charts.
*   `killer_experiments_results.png`: Primary visualization of the experimental findings.

## References

1.  Cong, I., Choi, S. & Lukin, M. D. (2019). "Quantum convolutional neural networks." *Nature Physics*, 15(12), 1273-1278.
2.  Kim, J., Huh, J. & Park, D. K. (2023). "Classical-to-quantum convolutional neural network transfer learning." *Neurocomputing*, 555, 126643.
3.  Hur, T., et al. (2022). "Quantum convolutional neural network for classical data classification." *Quantum Machine Intelligence*, 4(1), 3.

---
*Correspondence regarding this research should be directed to the repository owner.*

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def pgd_attack(model, imgs, labels, epsilon, alpha=None, steps=20):
    if alpha is None:
        alpha = epsilon / 4.0

    model.eval()
    imgs = imgs.to(DEVICE)
    labels = labels.to(DEVICE)
    delta = torch.zeros_like(imgs).uniform_(-epsilon, epsilon)
    delta = torch.clamp(delta, -epsilon, epsilon)
    delta.requires_grad_(True)

    for _ in range(steps):
        adv_imgs = imgs + delta
        adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
        logits = model(adv_imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        with torch.no_grad():
            delta_grad = delta.grad.sign()
            delta.data = delta.data + alpha * delta_grad
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            delta.data = torch.clamp(imgs + delta.data, 0.0, 1.0) - imgs

        delta.grad.zero_()

    return (imgs + delta).detach()


def estimate_lipschitz(model, imgs, eps=1e-3, n_samples=100):
    model.eval()
    imgs = imgs[:n_samples].to(DEVICE)
    perturbed = imgs + torch.randn_like(imgs) * eps
    perturbed = torch.clamp(perturbed, 0.0, 1.0)

    with torch.no_grad():
        out1 = F.softmax(model(imgs), dim=-1)
        out2 = F.softmax(model(perturbed), dim=-1)

    output_diff = (out1 - out2).norm(dim=-1)
    input_diff = (imgs - perturbed).view(len(imgs), -1).norm(dim=-1) + 1e-12
    lipschitz = (output_diff / input_diff).mean().item()
    return lipschitz


def evaluate_under_attack(model, test_loader, epsilon, steps=20):
    model.eval()
    correct = 0
    total = 0

    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        if epsilon > 0.0:
            adv_imgs = pgd_attack(model, imgs, labels, epsilon=epsilon, steps=steps)
        else:
            adv_imgs = imgs

        with torch.no_grad():
            logits = model(adv_imgs)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(imgs)

    return correct / total


def run_adversarial_benchmark(cnn_model, qcnn_model, test_loader,
                               epsilon_list=None, steps=20, save_path="benchmark_results.json"):
    if epsilon_list is None:
        epsilon_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    cnn_model = cnn_model.to(DEVICE).eval()
    qcnn_model = qcnn_model.to(DEVICE).eval()

    results = {"epsilon": [], "CNN_acc": [], "QCNN_acc": [],
               "CNN_lipschitz": [], "QCNN_lipschitz": []}

    sample_batch = next(iter(test_loader))
    sample_imgs = sample_batch[0]

    print("\n[adversarial_benchmark] Computing Lipschitz estimates...")
    cnn_lip = estimate_lipschitz(cnn_model, sample_imgs)
    qcnn_lip = estimate_lipschitz(qcnn_model, sample_imgs)
    print(f"  CNN  Lipschitz constant: {cnn_lip:.6f}")
    print(f"  QCNN Lipschitz constant: {qcnn_lip:.6f}")

    print(f"\n[adversarial_benchmark] Sweeping PGD epsilon: {epsilon_list}")
    for eps in epsilon_list:
        print(f"\n  epsilon={eps:.3f}")
        cnn_acc = evaluate_under_attack(cnn_model, test_loader, eps, steps=steps)
        qcnn_acc = evaluate_under_attack(qcnn_model, test_loader, eps, steps=steps)
        print(f"    CNN  acc: {cnn_acc:.4f}")
        print(f"    QCNN acc: {qcnn_acc:.4f}")

        results["epsilon"].append(eps)
        results["CNN_acc"].append(cnn_acc)
        results["QCNN_acc"].append(qcnn_acc)
        results["CNN_lipschitz"].append(cnn_lip)
        results["QCNN_lipschitz"].append(qcnn_lip)

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[adversarial_benchmark] Results saved to {save_path}")

    return results

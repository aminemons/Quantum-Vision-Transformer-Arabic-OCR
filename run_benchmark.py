"""
Fourier-MERA QCNN vs. ResNet CNN: Full Adversarial Benchmark Pipeline.

Execution sequence:
1. Load AHCD + Hijja datasets with Othmanic/Tashkeel/Clean morphological transforms.
2. Train ResNet CNN baseline and Fourier-MERA QCNN.
3. Run PGD adversarial benchmark across epsilon=[0.0, 0.05, ..., 0.3].
4. Generate all academic plots and save results JSON.

Usage:
    python run_benchmark.py --modes clean othmanic tashkeel --epochs 30 --img-size 16
"""

import os
import sys
import json
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dataset_engine import build_loaders
from classic_cnn import get_resnet_cnn
from mera_qcnn import get_mera_qcnn
from trainer import train_model
from adversarial_benchmark import run_adversarial_benchmark
from plot_results import generate_all_plots

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = os.path.join(os.path.dirname(__file__), "adversarial_results")
os.makedirs(SAVE_DIR, exist_ok=True)

EPSILON_LIST = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


def run(modes, epochs, img_size, batch_size, lr_cnn, lr_qcnn):
    all_results = {}
    cnn_histories = {}
    qcnn_histories = {}

    for mode in modes:
        print(f"\n{'='*65}")
        print(f" REGIME: mode={mode} | img_size={img_size}x{img_size}")
        print(f"{'='*65}")

        train_loader, val_loader, test_loader, n_classes = build_loaders(
            mode=mode, img_size=img_size, batch_size=batch_size
        )
        print(f"  classes={n_classes} | "
              f"train={len(train_loader.dataset)} | "
              f"val={len(val_loader.dataset)} | "
              f"test={len(test_loader.dataset)}")

        cnn_path = os.path.join(SAVE_DIR, f"best_cnn_{mode}.pt")
        qcnn_path = os.path.join(SAVE_DIR, f"best_qcnn_{mode}.pt")

        print(f"\n--- Training ResNet CNN [{mode}] ---")
        cnn = get_resnet_cnn(img_size=img_size, n_classes=n_classes)
        cnn, cnn_hist = train_model(
            cnn, train_loader, val_loader,
            save_path=cnn_path, epochs=epochs, lr=lr_cnn,
            use_amp=True, model_name=f"ResNet_CNN_{mode}"
        )

        print(f"\n--- Training Fourier-MERA QCNN [{mode}] ---")
        qcnn = get_mera_qcnn(img_size=img_size, n_classes=n_classes)
        qcnn, qcnn_hist = train_model(
            qcnn, train_loader, val_loader,
            save_path=qcnn_path, epochs=epochs, lr=lr_qcnn,
            use_amp=False, model_name=f"MERA_QCNN_{mode}"
        )

        print(f"\n--- PGD Adversarial Benchmark [{mode}] ---")
        adv_save = os.path.join(SAVE_DIR, f"adv_results_{mode}.json")
        adv_results = run_adversarial_benchmark(
            cnn_model=cnn,
            qcnn_model=qcnn,
            test_loader=test_loader,
            epsilon_list=EPSILON_LIST,
            steps=20,
            save_path=adv_save
        )

        all_results[mode] = adv_results
        cnn_histories[mode] = cnn_hist
        qcnn_histories[mode] = qcnn_hist

        print(f"\n  [Summary: {mode}]")
        for i, eps in enumerate(adv_results["epsilon"]):
            cnn_a = adv_results["CNN_acc"][i]
            q_a = adv_results["QCNN_acc"][i]
            delta = q_a - cnn_a
            mark = " << QCNN WINS" if delta > 0 else ""
            print(f"    eps={eps:.2f} | CNN={cnn_a:.4f} | QCNN={q_a:.4f} | delta={delta:+.4f}{mark}")

    all_path = os.path.join(SAVE_DIR, "all_results.json")
    with open(all_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Master results saved: {all_path}")

    print("\n[OK] Generating academic plots...")
    generate_all_plots(all_results, cnn_histories, qcnn_histories, out_dir=SAVE_DIR)

    print(f"\n{'='*65}")
    print(" BENCHMARK COMPLETE")
    print(f" Results: {SAVE_DIR}/")
    print(f"{'='*65}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fourier-MERA QCNN vs. ResNet adversarial benchmark")
    parser.add_argument("--modes", nargs="+", default=["clean", "othmanic", "tashkeel"],
                        help="Dataset modes to benchmark")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--img-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-cnn", type=float, default=1e-3)
    parser.add_argument("--lr-qcnn", type=float, default=5e-4)
    args = parser.parse_args()

    if DEVICE == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        print(f"[VRAM] {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    run(
        modes=args.modes,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        lr_cnn=args.lr_cnn,
        lr_qcnn=args.lr_qcnn,
    )

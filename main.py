"""
Hybrid QCNN-ViT -- Main Entry Point.

Orchestrates the full pipeline:
1. Data loading and preprocessing
2. Model training with quantum-classical hybrid architecture
3. Visualization generation

Usage:
    python main.py                     # Full pipeline
    python main.py --epochs 5          # Quick test
    python main.py --visualize-only    # Only generate visualizations
"""

import os
import sys
import argparse

# Project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from train import train, DEFAULT_CONFIG
from visualize import generate_all_visualizations
from data_loader import load_data


def main():
    parser = argparse.ArgumentParser(
        description='Hybrid QCNN-ViT for Arabic Character Recognition'
    )
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--max-samples', type=int, default=2000)
    parser.add_argument('--visualize-only', action='store_true',
                       help='Only generate visualizations from saved results')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if args.visualize_only:
        print("Generating visualizations from saved results...")
        generate_all_visualizations()
        return

    # Configure
    config = DEFAULT_CONFIG.copy()
    config.update({
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'max_samples': args.max_samples,
        'seed': args.seed,
    })

    # Train
    model, history = train(config)

    # Generate visualizations
    print("\n\nGenerating visualizations...")
    _, _, test_loader, _ = load_data(
        max_samples=config['max_samples'],
        batch_size=config['batch_size'],
    )
    generate_all_visualizations(model=model, data_loader=test_loader)

    print("\n[DONE] Pipeline complete! Check results/ for outputs.")


if __name__ == "__main__":
    main()

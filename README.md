# CH-OOD Experiments

Google Colab-ready experiments for the paper "Certified Geometric OOD via Directional Depth and Kubota Projection Sketches".

## Quick Start

1. Open `colab_notebook.ipynb` in Google Colab
2. Run all cells (Runtime → Run all)
3. Total runtime: ~45-60 minutes on T4 GPU

## Files

- `ch_ood_colab.py`: Complete experiment implementation
- `colab_notebook.ipynb`: User-friendly Jupyter interface

## Experiments

- **ID Dataset**: CIFAR-10
- **OOD Datasets**: SVHN, CIFAR-100
- **Methods**: CH-OOD, Energy, ODIN, Mahalanobis
- **Metrics**: AUROC, FPR@95%TPR

## Requirements

All dependencies are automatically installed in Colab:
- PyTorch
- torchvision
- numpy
- scikit-learn
- matplotlib
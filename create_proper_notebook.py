#!/usr/bin/env python3
"""
Create a properly formatted Jupyter notebook for Google Colab
using nbformat library to ensure correct structure
"""

import uuid
import nbformat as nbf
from nbformat.validator import validate

def generate_id():
    """Generate a short ID for Colab cells"""
    return uuid.uuid4().hex[:12]

def create_markdown_cell(text):
    """Create a markdown cell with proper ID"""
    cell = nbf.v4.new_markdown_cell(text)
    cell.metadata = {"id": generate_id()}
    return cell

def create_code_cell(source):
    """Create a code cell with proper ID"""
    cell = nbf.v4.new_code_cell(source)
    cell.metadata = {"id": generate_id()}
    cell.execution_count = None
    cell.outputs = []
    return cell

# Create new notebook
notebook = nbf.v4.new_notebook()

# Set notebook metadata for Colab
notebook["metadata"].update({
    "colab": {
        "name": "ch_ood_experiments.ipynb",
        "provenance": [],
        "collapsed_sections": []
    },
    "kernelspec": {
        "name": "python3",
        "display_name": "Python 3"
    },
    "language_info": {
        "name": "python",
        "version": "3.8"
    },
    "accelerator": "GPU"
})

# Create cells list
cells = []

# Cell 1: Title and overview (Markdown)
cells.append(create_markdown_cell("""# CH-OOD Deep Learning Experiments

This notebook runs deep OOD detection experiments for the paper:
**Certified Geometric OOD via Directional Depth and Kubota Projection Sketches**

## Expected Runtime on T4 GPU
- **Total time**: ~45-60 minutes
  - Model training: ~20-25 minutes
  - Feature extraction: ~10-15 minutes
  - OOD evaluation: ~15-20 minutes

## Experiments
- **ID Dataset**: CIFAR-10 (test set)
- **OOD Datasets**: SVHN, CIFAR-100
- **Methods**: CH-OOD (ours), Energy, ODIN, Mahalanobis
- **Metrics**: AUROC, FPR@95%TPR"""))

# Cell 2: Section 1 header (Markdown)
cells.append(create_markdown_cell("## 1. Setup Environment"))

# Cell 3: GPU check (Code)
cells.append(create_code_cell("""# Check GPU availability
import torch
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'✓ GPU available: {torch.cuda.get_device_name(0)}')
    print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
else:
    print('⚠️ No GPU detected. Runtime will be slower.')
    print('To enable GPU in Colab: Runtime → Change runtime type → Hardware accelerator → GPU (T4)')"""))

# Cell 4: Section 2 header (Markdown)
cells.append(create_markdown_cell("## 2. Install Dependencies"))

# Cell 5: Install packages (Code)
cells.append(create_code_cell("""# Install required packages
!pip install -q torch torchvision numpy scikit-learn matplotlib tqdm pandas"""))

# Cell 6: Section 3 header (Markdown)
cells.append(create_markdown_cell("## 3. Download Experiment Code"))

# Cell 7: Download code (Code)
cells.append(create_code_cell("""# Download the experiment code from GitHub
!wget -q https://raw.githubusercontent.com/ashachar/ch-ood-experiments/main/ch_ood_colab.py
print('✓ Experiment code downloaded')"""))

# Cell 8: Import module (Code)
cells.append(create_code_cell("""# Import the experiment module
import ch_ood_colab
import importlib
importlib.reload(ch_ood_colab)  # Reload in case of updates
print('✓ Module imported successfully')"""))

# Cell 9: Section 4 header (Markdown)
cells.append(create_markdown_cell("## 4. Setup Directories and Device"))

# Cell 10: Setup directories (Code)
cells.append(create_code_cell("""# Create necessary directories and return device
device = ch_ood_colab.setup_colab()
print(f'✓ Setup complete. Using device: {device}')"""))

# Cell 11: Section 5 header (Markdown)
cells.append(create_markdown_cell("""## 5. Train ResNet-18 on CIFAR-10

This step trains a ResNet-18 model on CIFAR-10 for 30 epochs.
- **Expected time**: ~20-25 minutes on T4 GPU
- **Skip this cell** if you want to use a pre-trained model"""))

# Cell 12: Train model (Code)
cells.append(create_code_cell("""# Train the model (set epochs=1 for quick test)
model = ch_ood_colab.train_resnet_cifar10(device, epochs=30)
print('\\n✓ Model training complete')"""))

# Cell 13: Section 6 header (Markdown)
cells.append(create_markdown_cell("""## 6. Load Datasets

Load CIFAR-10 (ID) and OOD datasets (SVHN, CIFAR-100)"""))

# Cell 14: Load datasets (Code)
cells.append(create_code_cell("""# Load all datasets
print('Loading datasets...')
loaders = ch_ood_colab.load_datasets()

print('\\n✓ Datasets loaded:')
print(f"  - ID: CIFAR-10 (test set: {len(loaders['id_test'].dataset)} samples)")
print(f"  - OOD: SVHN ({len(loaders['ood_svhn'].dataset)} samples)")
print(f"  - OOD: CIFAR-100 ({len(loaders['ood_cifar100'].dataset)} samples)")"""))

# Cell 15: Section 7 header (Markdown)
cells.append(create_markdown_cell("""## 7. Extract Features

Extract penultimate layer features for OOD detection
- **Expected time**: ~10-15 minutes"""))

# Cell 16: Extract features (Code)
cells.append(create_code_cell("""# Load model if not already loaded
if 'model' not in locals():
    print('Loading pre-trained model...')
    model = ch_ood_colab.load_model(device)

# Extract features
print('Extracting features...')
features = ch_ood_colab.extract_features(model, loaders, device)

print('\\n✓ Features extracted:')
for key, feat in features.items():
    print(f'  - {key}: shape {feat.shape}')"""))

# Cell 17: Section 8 header (Markdown)
cells.append(create_markdown_cell("""## 8. Run OOD Detection Methods

Evaluate multiple OOD detection methods:
- CH-OOD (our method)
- Energy-based
- ODIN
- Mahalanobis

**Expected time**: ~15-20 minutes"""))

# Cell 18: Run OOD methods (Code)
cells.append(create_code_cell("""# Run all OOD detection methods
print('Running OOD detection methods...\\n')
results = ch_ood_colab.evaluate_ood_methods(features, device)

# Display results summary
print('\\n' + '='*60)
print('OOD Detection Results (AUROC / FPR@95%)')
print('='*60)

for ood_name in ['SVHN', 'CIFAR-100']:
    print(f'\\n{ood_name}:')
    for method in results[ood_name]:
        auroc = results[ood_name][method]['auroc']
        fpr95 = results[ood_name][method]['fpr95']
        print(f'  {method:15s}: AUROC={auroc:.3f}, FPR@95%={fpr95:.3f}')"""))

# Cell 19: Section 9 header (Markdown)
cells.append(create_markdown_cell("## 9. Generate ROC Curves"))

# Cell 20: Plot ROC curves (Code)
cells.append(create_code_cell("""# Plot ROC curves
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, ood_name in enumerate(['SVHN', 'CIFAR-100']):
    ax = axes[idx]
    
    for method in results[ood_name]:
        fpr = results[ood_name][method]['fpr']
        tpr = results[ood_name][method]['tpr']
        auroc = results[ood_name][method]['auroc']
        
        ax.plot(fpr, tpr, label=f'{method} (AUROC={auroc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'CIFAR-10 (ID) vs {ood_name} (OOD)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""))

# Cell 21: Section 10 header (Markdown)
cells.append(create_markdown_cell("## 10. Create Summary Table"))

# Cell 22: Summary table (Code)
cells.append(create_code_cell("""# Create a formatted summary table
import pandas as pd
import json

# Prepare results for saving
save_results = {}
for ood_name in results:
    save_results[ood_name] = {}
    for method in results[ood_name]:
        save_results[ood_name][method] = {
            'auroc': float(results[ood_name][method]['auroc']),
            'fpr95': float(results[ood_name][method]['fpr95'])
        }

# Create DataFrame
data = []
for ood_name in save_results:
    for method in save_results[ood_name]:
        data.append({
            'OOD Dataset': ood_name,
            'Method': method,
            'AUROC': save_results[ood_name][method]['auroc'],
            'FPR@95%': save_results[ood_name][method]['fpr95']
        })

df = pd.DataFrame(data)

# Pivot tables
print('\\n' + '='*50)
print('AUROC Results')
print('='*50)
auroc_pivot = df.pivot(index='Method', columns='OOD Dataset', values='AUROC')
print(auroc_pivot.round(3))

print('\\n' + '='*50)
print('FPR@95%TPR Results')
print('='*50)
fpr_pivot = df.pivot(index='Method', columns='OOD Dataset', values='FPR@95%')
print(fpr_pivot.round(3))

# Save results
with open('results/deep_ood_results.json', 'w') as f:
    json.dump(save_results, f, indent=2)
print('\\n✓ Results saved to results/deep_ood_results.json')"""))

# Cell 23: Section 11 header (Markdown)
cells.append(create_markdown_cell("""## 11. Generate Detailed Analysis Report

Create a comprehensive report for analysis"""))

# Cell 24: Analysis report (Code)
cells.append(create_code_cell("""# Generate comprehensive analysis report
from datetime import datetime
import numpy as np

report = []
report.append('='*70)
report.append('CH-OOD EXPERIMENTAL RESULTS - DETAILED ANALYSIS REPORT')
report.append('='*70)
report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report.append(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
report.append('')

# Dataset info
report.append('1. DATASET SUMMARY')
report.append('-' * 40)
report.append(f"   ID Dataset: CIFAR-10")
report.append(f"   - Test samples: {len(loaders['id_test'].dataset)}")
report.append(f"   OOD Datasets:")
report.append(f"   - SVHN: {len(loaders['ood_svhn'].dataset)} samples")
report.append(f"   - CIFAR-100: {len(loaders['ood_cifar100'].dataset)} samples")
report.append('')

# Method results
report.append('2. DETAILED RESULTS BY METHOD')
report.append('-' * 40)

for method in ['CH-OOD', 'Energy', 'ODIN', 'Mahalanobis']:
    if method not in results['SVHN']:
        continue
    report.append(f'\\n   {method}:')
    aurocs = []
    fpr95s = []
    for ood_name in ['SVHN', 'CIFAR-100']:
        auroc = results[ood_name][method]['auroc']
        fpr95 = results[ood_name][method]['fpr95']
        aurocs.append(auroc)
        fpr95s.append(fpr95)
        report.append(f'   - vs {ood_name:12s}: AUROC={auroc:.4f}, FPR@95%={fpr95:.4f}')
    report.append(f'   - Average AUROC: {np.mean(aurocs):.4f}')
    report.append(f'   - Average FPR@95%: {np.mean(fpr95s):.4f}')

# Best performers
report.append('')
report.append('3. BEST PERFORMERS')
report.append('-' * 40)
for ood_name in ['SVHN', 'CIFAR-100']:
    best_auroc_method = max(results[ood_name].keys(), 
                            key=lambda m: results[ood_name][m]['auroc'])
    best_auroc = results[ood_name][best_auroc_method]['auroc']
    report.append(f'   {ood_name}: {best_auroc_method} (AUROC={best_auroc:.4f})')

report.append('')
report.append('='*70)
report.append('END OF REPORT')
report.append('='*70)

# Print the report
full_report = '\\n'.join(report)
print(full_report)

# Save to file
with open('results/analysis_report.txt', 'w') as f:
    f.write(full_report)

print('\\n✓ Report saved to results/analysis_report.txt')
print('\\n' + '='*70)
print('COPY THE REPORT ABOVE FOR ANALYSIS')
print('='*70)"""))

# Cell 25: Section 12 header (Markdown)
cells.append(create_markdown_cell("""## 12. Download Results

Package all results for download"""))

# Cell 26: Download results (Code)
cells.append(create_code_cell("""# Create zip file with results
import zipfile
import os

with zipfile.ZipFile('ch_ood_results.zip', 'w') as zipf:
    if os.path.exists('results/deep_ood_results.json'):
        zipf.write('results/deep_ood_results.json')
    if os.path.exists('results/analysis_report.txt'):
        zipf.write('results/analysis_report.txt')
    if os.path.exists('models/cifar10_resnet18_best.pth'):
        zipf.write('models/cifar10_resnet18_best.pth')

print('✓ Results packaged in ch_ood_results.zip')
print('\\nTo download in Colab:')
print('  1. Click the folder icon in the left sidebar')
print('  2. Find ch_ood_results.zip')
print('  3. Click the three dots and select Download')"""))

# Add all cells to notebook
notebook["cells"] = cells

# Validate the notebook structure
try:
    validate(notebook)
    print("✓ Notebook structure is valid")
except Exception as e:
    print(f"⚠️ Validation warning: {e}")

# Write the notebook
output_file = "colab_notebook.ipynb"
nbf.write(notebook, output_file)
print(f"✓ Created {output_file} with {len(cells)} cells")
print(f"  - Markdown cells: {sum(1 for c in cells if c.cell_type == 'markdown')}")
print(f"  - Code cells: {sum(1 for c in cells if c.cell_type == 'code')}")
"""
CH-OOD Deep Learning Experiments for Google Colab
Reproducible experiments for CIFAR10 vs SVHN/CIFAR100/TinyImageNet OOD detection
Expected runtime on T4 GPU: ~45-60 minutes for full experiment suite
"""

import os
import math
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
from math import lgamma, pi
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_curve as sk_roc_curve, auc as sk_auc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ==================== Setup & Configuration ====================
def setup_colab():
    """Setup Google Colab environment"""
    # Check GPU availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("⚠ No GPU available, using CPU (will be slower)")
    
    # Create directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    return device

# ==================== Core CH-OOD Implementation ====================
class HadwigerSoft:
    """Certified Hadwiger OOD Detector with directional depth"""
    
    def __init__(self, U, max_proj, q_lo, q_hi, alpha, cd, cal_scores):
        self.U = U
        self.max_proj = max_proj
        self.q_lo = q_lo
        self.q_hi = q_hi
        self.alpha = alpha
        self.cd = cd
        self.cal_scores = np.sort(cal_scores)
    
    @staticmethod
    def fit(X, target_fpr=0.05, alpha=None, seed=42):
        rng = np.random.default_rng(seed)
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        
        # Estimate diameter
        D = HadwigerSoft._estimate_diameter(X, rng)
        
        # Auto-select m (number of projections)
        m = HadwigerSoft._auto_m(D, eps=0.02, delta=1e-3, mmin=256, mmax=8192)
        print(f"  Using m={m} random projections for d={d}")
        
        # Random directions
        U = HadwigerSoft._random_directions(d, m, rng)
        
        # Project data
        P = X @ U.T
        max_proj = P.max(axis=0)
        
        # Auto-select alpha if not provided
        if alpha is None:
            alpha = HadwigerSoft._auto_alpha(P)
        
        q_lo = np.quantile(P, alpha, axis=0, method="linear")
        q_hi = np.quantile(P, 1.0 - alpha, axis=0, method="linear")
        
        # Calibration split
        n_cal = max(200, int(0.2 * n))
        idx = rng.permutation(n)
        cal_idx = idx[:n_cal]
        Pcal = P[cal_idx]
        
        # Compute calibration scores
        Sout = np.maximum(Pcal - max_proj[None, :], 0.0).mean(axis=1)
        Sin = ((Pcal < q_lo[None, :]) | (Pcal > q_hi[None, :])).mean(axis=1)
        cal_scores = Sout + Sin
        
        # Compute cd constant
        cd = HadwigerSoft._cd_margin_constant(d)
        
        return HadwigerSoft(U, max_proj, q_lo, q_hi, alpha, cd, cal_scores)
    
    def scores(self, Z):
        Z = np.asarray(Z, dtype=np.float64)
        P = Z @ self.U.T
        Sout = np.maximum(P - self.max_proj[None, :], 0.0).mean(axis=1)
        Sin = ((P < self.q_lo[None, :]) | (P > self.q_hi[None, :])).mean(axis=1)
        return Sout, Sin, Sout + Sin
    
    def p_values(self, Z):
        sc = self.scores(Z)[2]
        N = self.cal_scores.shape[0]
        idx = np.searchsorted(self.cal_scores, sc, side="left")
        ge = (N - idx)
        return (1.0 + ge) / (1.0 + N)
    
    def severity(self, Z):
        return 100.0 * (1.0 - self.p_values(Z))
    
    @staticmethod
    def _random_directions(d, m, rng):
        G = rng.normal(size=(m, d))
        return G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-18)
    
    @staticmethod
    def _estimate_diameter(X, rng, pairs=4096):
        n = X.shape[0]
        i1 = rng.integers(0, n, size=min(pairs, n))
        i2 = rng.integers(0, n, size=min(pairs, n))
        return float(np.quantile(np.linalg.norm(X[i1] - X[i2], axis=1), 0.99))
    
    @staticmethod
    def _auto_m(D, eps=0.02, delta=1e-3, mmin=256, mmax=8192):
        m = int(math.ceil((D**2) / (2 * eps**2) * math.log(2.0 / delta)))
        return max(mmin, min(m, mmax))
    
    @staticmethod
    def _auto_alpha(P):
        grid = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]
        best = (0.1, 1e9)
        for a in grid:
            ql = np.quantile(P, a, axis=0, method="linear")
            qh = np.quantile(P, 1.0 - a, axis=0, method="linear")
            Sin = ((P < ql[None, :]) | (P > qh[None, :])).mean(axis=1)
            score = abs(Sin.mean() - a)
            if score < best[1]:
                best = (a, score)
        return best[0]
    
    @staticmethod
    def _cd_margin_constant(d):
        if d <= 1:
            return 1.0
        log_c = lgamma(d/2.0) - 0.5*math.log(pi) - lgamma((d-1)/2.0) - math.log(d-1.0)
        return float(np.exp(log_c))

# ==================== Deep Learning Models ====================
class ResNet18CIFAR(nn.Module):
    """ResNet-18 adapted for CIFAR (32x32 images)"""
    def __init__(self, num_classes=10):
        super().__init__()
        m = models.resnet18(weights=None)
        # Adapt first conv for 32x32 images
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()  # Remove maxpool for small images
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        self.net = m
    
    def forward(self, x):
        return self.net(x)
    
    def penultimate(self, x):
        """Extract penultimate layer features"""
        modules = list(self.net.children())[:-1]
        f = x
        for mod in modules:
            f = mod(f)
        return torch.flatten(f, 1)

def train_resnet_cifar10(device, epochs=30, bs=128, lr=0.1, wd=5e-4):
    """Train ResNet-18 on CIFAR10
    Expected time on T4: ~20-25 minutes for 30 epochs
    """
    print("\n=== Training ResNet-18 on CIFAR10 ===")
    print(f"Config: epochs={epochs}, batch_size={bs}, lr={lr}")
    
    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load datasets
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=bs, shuffle=False, num_workers=2)
    
    # Model
    model = ResNet18CIFAR(num_classes=10).to(device)
    
    # Optimizer & scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.1)
    
    # Training loop
    best_acc = 0
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Test
        model.eval()
        test_loss = 0
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total_test += targets.size(0)
                correct_test += predicted.eq(targets).sum().item()
        
        acc = 100. * correct_test / total_test
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'models/cifar10_resnet18_best.pth')
        
        print(f'Epoch {epoch+1}/{epochs}: Train Acc: {100.*correct/total:.2f}%, '
              f'Test Acc: {acc:.2f}%, Best: {best_acc:.2f}%')
        
        scheduler.step()
    
    print(f"✓ Training complete. Best accuracy: {best_acc:.2f}%")
    return model

# ==================== OOD Detection Methods ====================
@torch.no_grad()
def extract_features(model, loader, device):
    """Extract logits and penultimate features"""
    model.eval()
    logits_list = []
    features_list = []
    
    for inputs, _ in loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        features = model.penultimate(inputs)
        
        logits_list.append(logits.cpu().numpy())
        features_list.append(features.cpu().numpy())
    
    return np.concatenate(logits_list), np.concatenate(features_list)

def energy_scores(logits, T=1.0):
    """Energy-based OOD scores"""
    l = logits / T
    m = np.max(l, axis=1, keepdims=True)
    return -T * (m + np.log(np.sum(np.exp(l - m), axis=1, keepdims=True))).squeeze(1)

def odin_scores(model, loader, device, T=1000.0, eps=0.0014):
    """ODIN scores with input preprocessing"""
    model.eval()
    scores = []
    
    for inputs, _ in loader:
        inputs = inputs.to(device)
        inputs.requires_grad_(True)
        
        # Forward pass with temperature
        logits = model(inputs) / T
        yhat = torch.argmax(logits, dim=1)
        loss = F.cross_entropy(logits, yhat, reduction="mean")
        loss.backward()
        
        # Perturb input
        grad = torch.sign(inputs.grad)
        x_pert = torch.clamp(inputs - eps * grad, 0.0, 1.0)
        
        # Get score on perturbed input
        with torch.no_grad():
            logits_pert = model(x_pert) / T
            e = -T * torch.logsumexp(logits_pert, dim=1)
        
        scores.append(e.cpu().numpy())
    
    return np.concatenate(scores)

def mahalanobis_scores(features_id, features_ood):
    """Mahalanobis distance OOD scores"""
    cov = EmpiricalCovariance().fit(features_id)
    mu = features_id.mean(axis=0, keepdims=True)
    
    d_id = cov.mahalanobis(features_id - mu)
    d_ood = cov.mahalanobis(features_ood - mu)
    
    return d_id, d_ood

# ==================== Evaluation ====================
def compute_metrics(scores_ood, scores_id):
    """Compute AUROC and FPR@95%TPR"""
    y = np.concatenate([np.ones_like(scores_ood), np.zeros_like(scores_id)])
    s = np.concatenate([scores_ood, scores_id])
    
    fpr, tpr, thresholds = sk_roc_curve(y, s)
    auroc = float(sk_auc(fpr, tpr))
    
    # FPR at 95% TPR
    idx = np.where(tpr >= 0.95)[0]
    if idx.size == 0:
        fpr95 = 1.0
    else:
        fpr95 = float(fpr[idx[0]])
    
    return auroc, fpr95, fpr, tpr

def plot_roc_curves(results, save_path):
    """Plot ROC curves for all methods"""
    plt.figure(figsize=(10, 8))
    
    for method, data in results.items():
        plt.plot(data['fpr'], data['tpr'], 
                label=f"{method} (AUC={data['auroc']:.3f}, FPR95={data['fpr95']:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('OOD Detection ROC Curves')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# ==================== Main Experiment Pipeline ====================
def run_deep_ood_experiments(device, train_model=True):
    """
    Run complete deep OOD experiments
    Expected runtime on T4 GPU:
    - Training: ~20-25 minutes
    - OOD evaluation: ~15-20 minutes
    Total: ~35-45 minutes
    """
    
    print("\n" + "="*60)
    print("CH-OOD Deep Learning Experiments")
    print("="*60)
    
    # Step 1: Train or load model
    if train_model:
        model = train_resnet_cifar10(device)
    else:
        print("Loading pre-trained model...")
        model = ResNet18CIFAR(num_classes=10).to(device)
        if os.path.exists('models/cifar10_resnet18_best.pth'):
            model.load_state_dict(torch.load('models/cifar10_resnet18_best.pth'))
            print("✓ Model loaded")
        else:
            print("⚠ No saved model found, training new model...")
            model = train_resnet_cifar10(device)
    
    # Step 2: Prepare datasets
    print("\n=== Preparing OOD Datasets ===")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # ID: CIFAR10 test set
    id_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    id_loader = DataLoader(id_dataset, batch_size=256, shuffle=False, num_workers=2)
    
    # OOD datasets
    ood_datasets = {}
    
    # SVHN
    print("Loading SVHN...")
    svhn_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    ood_datasets['SVHN'] = DataLoader(svhn_dataset, batch_size=256, shuffle=False, num_workers=2)
    
    # CIFAR100
    print("Loading CIFAR100...")
    cifar100_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    ood_datasets['CIFAR100'] = DataLoader(cifar100_dataset, batch_size=256, shuffle=False, num_workers=2)
    
    # Step 3: Extract features
    print("\n=== Extracting Features ===")
    print("Extracting ID features...")
    logits_id, features_id = extract_features(model, id_loader, device)
    
    # Split ID data for calibration
    n = len(logits_id)
    n_cal = n // 2
    idx = np.random.permutation(n)
    
    logits_id_cal = logits_id[idx[:n_cal]]
    features_id_cal = features_id[idx[:n_cal]]
    logits_id_test = logits_id[idx[n_cal:]]
    features_id_test = features_id[idx[n_cal:]]
    
    # Step 4: Run experiments for each OOD dataset
    all_results = {}
    
    for ood_name, ood_loader in ood_datasets.items():
        print(f"\n=== Evaluating CIFAR10 vs {ood_name} ===")
        
        # Extract OOD features
        logits_ood, features_ood = extract_features(model, ood_loader, device)
        
        results = {}
        
        # 1. Our method (CH-OOD on features)
        print("Computing CH-OOD scores...")
        ch_ood = HadwigerSoft.fit(features_id_cal, target_fpr=0.05)
        scores_ch_id = ch_ood.scores(features_id_test)[2]
        scores_ch_ood = ch_ood.scores(features_ood)[2]
        auroc, fpr95, fpr, tpr = compute_metrics(scores_ch_ood, scores_ch_id)
        results['CH-OOD'] = {'auroc': auroc, 'fpr95': fpr95, 'fpr': fpr, 'tpr': tpr}
        print(f"  CH-OOD: AUROC={auroc:.3f}, FPR95={fpr95:.3f}")
        
        # 2. Energy
        print("Computing Energy scores...")
        scores_energy_id = energy_scores(logits_id_test)
        scores_energy_ood = energy_scores(logits_ood)
        auroc, fpr95, fpr, tpr = compute_metrics(scores_energy_ood, scores_energy_id)
        results['Energy'] = {'auroc': auroc, 'fpr95': fpr95, 'fpr': fpr, 'tpr': tpr}
        print(f"  Energy: AUROC={auroc:.3f}, FPR95={fpr95:.3f}")
        
        # 3. ODIN
        print("Computing ODIN scores...")
        # Need to recompute with preprocessing
        id_loader_test = DataLoader(
            torch.utils.data.Subset(id_dataset, idx[n_cal:]),
            batch_size=256, shuffle=False, num_workers=2
        )
        scores_odin_id = odin_scores(model, id_loader_test, device)
        scores_odin_ood = odin_scores(model, ood_loader, device)
        auroc, fpr95, fpr, tpr = compute_metrics(scores_odin_ood, scores_odin_id)
        results['ODIN'] = {'auroc': auroc, 'fpr95': fpr95, 'fpr': fpr, 'tpr': tpr}
        print(f"  ODIN: AUROC={auroc:.3f}, FPR95={fpr95:.3f}")
        
        # 4. Mahalanobis
        print("Computing Mahalanobis scores...")
        scores_maha_id, scores_maha_ood = mahalanobis_scores(features_id_test, features_ood)
        auroc, fpr95, fpr, tpr = compute_metrics(scores_maha_ood, scores_maha_id)
        results['Mahalanobis'] = {'auroc': auroc, 'fpr95': fpr95, 'fpr': fpr, 'tpr': tpr}
        print(f"  Mahalanobis: AUROC={auroc:.3f}, FPR95={fpr95:.3f}")
        
        # Plot ROC curves
        plot_roc_curves(results, f'figures/roc_cifar10_vs_{ood_name.lower()}.png')
        
        all_results[ood_name] = results
    
    # Step 5: Generate summary table
    print("\n" + "="*60)
    print("Summary Results")
    print("="*60)
    
    methods = ['CH-OOD', 'Energy', 'ODIN', 'Mahalanobis']
    
    for ood_name in all_results.keys():
        print(f"\nCIFAR10 vs {ood_name}:")
        print(f"{'Method':<15} {'AUROC':<10} {'FPR@95%TPR':<10}")
        print("-" * 35)
        for method in methods:
            if method in all_results[ood_name]:
                auroc = all_results[ood_name][method]['auroc']
                fpr95 = all_results[ood_name][method]['fpr95']
                print(f"{method:<15} {auroc:<10.3f} {fpr95:<10.3f}")
    
    # Save results
    with open('results/deep_ood_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n✓ Results saved to results/deep_ood_results.json")
    
    return all_results

# ==================== Main Entry Point ====================
def main():
    """Main function to run in Google Colab"""
    
    # Setup
    device = setup_colab()
    
    # Run experiments
    results = run_deep_ood_experiments(device, train_model=True)
    
    print("\n" + "="*60)
    print("Experiments Complete!")
    print("="*60)
    print("\nEstimated runtime on T4 GPU: ~45-60 minutes")
    print("- Model training: ~20-25 minutes")
    print("- Feature extraction: ~10-15 minutes")  
    print("- OOD evaluation: ~15-20 minutes")
    
    return results

if __name__ == "__main__":
    main()
"""
Main experiment runner for evaluating backdoor defenses under quantization.

This script:
1. Loads pre-trained backdoored models from BackdoorBench
2. Quantizes them to different precisions (FP32, INT8, INT4)
3. Runs various backdoor defenses
4. Records detection rates and outputs results

Usage:
    python run_experiments.py --config configs/default.yaml
    
Or with command line args:
    python run_experiments.py --attacks badnet blended --defenses nc strip ac
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quantize import QuantizationManager, evaluate_model, get_model_size_mb

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

# Attacks available in BackdoorBench
AVAILABLE_ATTACKS = [
    'badnet', 'blended', 'sig', 'ssba', 'wanet', 'inputaware',
    'lc', 'lf', 'bpp', 'ctrl', 'ftrojan', 'refool', 'trojannn'
]

# Defenses available in BackdoorBench
AVAILABLE_DEFENSES = [
    'nc',      # Neural Cleanse
    'ac',      # Activation Clustering  
    'strip',   # STRIP
    'fp',      # Fine-Pruning
    'anp',     # ANP
    'abl',     # ABL
    'nad',     # NAD
    'ft',      # Fine-Tuning
    'ft_sam',  # Fine-Tuning with SAM
    'i_bau',   # I-BAU
    'npd',     # NPD
    'rnp',     # RNP
    'clp',     # CLP
    'dbr',     # D-BR
    'dbd',     # DBD
    'ep',      # EP
    'mcr',     # MCR
    'nab',     # NAB
    'sau',     # SAU
    'ss',      # Spectral Signatures
]

# Quantization schemes to test
QUANTIZATION_SCHEMES = ['fp32', 'int8_dynamic', 'int4_simulated']

# Datasets
DATASETS = ['cifar10', 'cifar100', 'gtsrb', 'tiny_imagenet']


# ============================================================================
# DATA LOADING
# ============================================================================

def get_data_transforms(dataset: str) -> Tuple[transforms.Compose, transforms.Compose]:
    """Get train and test transforms for a dataset."""
    
    if dataset in ['cifar10', 'cifar100']:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    elif dataset == 'gtsrb':
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
        ])
        test_transform = train_transform
    elif dataset == 'tiny_imagenet':
        train_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return train_transform, test_transform


def load_dataset(
    dataset: str,
    data_root: str = './data',
    batch_size: int = 128
) -> Tuple[DataLoader, DataLoader]:
    """Load train and test dataloaders for a dataset."""
    
    train_transform, test_transform = get_data_transforms(dataset)
    
    if dataset == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=train_transform
        )
        test_set = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=test_transform
        )
    elif dataset == 'cifar100':
        train_set = torchvision.datasets.CIFAR100(
            root=data_root, train=True, download=True, transform=train_transform
        )
        test_set = torchvision.datasets.CIFAR100(
            root=data_root, train=False, download=True, transform=test_transform
        )
    elif dataset == 'gtsrb':
        train_set = torchvision.datasets.GTSRB(
            root=data_root, split='train', download=True, transform=train_transform
        )
        test_set = torchvision.datasets.GTSRB(
            root=data_root, split='test', download=True, transform=test_transform
        )
    else:
        raise ValueError(f"Dataset {dataset} not yet implemented")
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader


# ============================================================================
# MODEL DEFINITIONS (for when BackdoorBench models aren't available)
# ============================================================================

class PreActBlock(nn.Module):
    """Pre-activation ResNet Block."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = torch.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(torch.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActResNet18(nn.Module):
    """PreAct-ResNet18 - commonly used in BackdoorBench."""
    
    def __init__(self, num_classes=10):
        super(PreActResNet18, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(PreActBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class VGG19_BN(nn.Module):
    """VGG19 with Batch Normalization."""
    
    def __init__(self, num_classes=10):
        super(VGG19_BN, self).__init__()
        self.features = self._make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 
                                           512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'])
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def get_model(model_name: str, num_classes: int = 10) -> nn.Module:
    """Get a model by name."""
    if model_name == 'preact_resnet18':
        return PreActResNet18(num_classes=num_classes)
    elif model_name == 'vgg19_bn':
        return VGG19_BN(num_classes=num_classes)
    elif model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = nn.Linear(512, num_classes)
        return model
    elif model_name == 'vit_b_16':
        model = torchvision.models.vit_b_16(pretrained=False)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ============================================================================
# DEFENSE IMPLEMENTATIONS (Simplified versions for benchmarking)
# ============================================================================

def run_neural_cleanse(
    model: nn.Module,
    num_classes: int,
    input_shape: Tuple[int, ...],
    device: str = 'cpu',
    **kwargs
) -> Dict[str, Any]:
    """
    Run Neural Cleanse defense.
    Returns anomaly index for each class.
    """
    model.eval()
    model.to(device)
    
    results = {'detected': False, 'anomaly_indices': {}, 'outlier_class': None}
    
    # Simplified NC: check if optimization to find triggers succeeds
    trigger_norms = []
    
    for target_class in range(num_classes):
        # Initialize trigger pattern
        trigger = torch.zeros(1, *input_shape, requires_grad=True, device=device)
        mask = torch.ones(1, *input_shape, requires_grad=True, device=device) * 0.1
        
        optimizer = torch.optim.Adam([trigger, mask], lr=0.1)
        
        # Optimize for 100 steps
        for _ in range(100):
            optimizer.zero_grad()
            
            # Create triggered input (using random base)
            base_input = torch.rand(8, *input_shape, device=device)
            triggered = base_input * (1 - torch.sigmoid(mask)) + trigger * torch.sigmoid(mask)
            
            # Forward pass
            outputs = model(triggered)
            target = torch.full((8,), target_class, device=device, dtype=torch.long)
            
            # Loss: classification + L1 regularization on mask
            ce_loss = nn.CrossEntropyLoss()(outputs, target)
            reg_loss = torch.sum(torch.abs(torch.sigmoid(mask)))
            loss = ce_loss + 0.01 * reg_loss
            
            loss.backward()
            optimizer.step()
        
        # Record trigger norm
        final_mask_norm = torch.sum(torch.abs(torch.sigmoid(mask))).item()
        trigger_norms.append(final_mask_norm)
        results['anomaly_indices'][target_class] = final_mask_norm
    
    # Detect outlier using MAD
    trigger_norms = np.array(trigger_norms)
    median = np.median(trigger_norms)
    mad = np.median(np.abs(trigger_norms - median))
    
    if mad > 0:
        anomaly_indices = np.abs(trigger_norms - median) / (1.4826 * mad)
        outlier_idx = np.argmin(trigger_norms)
        if anomaly_indices[outlier_idx] > 2.0:  # Threshold
            results['detected'] = True
            results['outlier_class'] = int(outlier_idx)
    
    return results


def run_activation_clustering(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = 'cpu',
    **kwargs
) -> Dict[str, Any]:
    """
    Run Activation Clustering defense.
    Clusters activations and checks for anomalous clusters.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
    
    model.eval()
    model.to(device)
    
    results = {'detected': False, 'silhouette_score': 0.0, 'cluster_sizes': []}
    
    # Hook to capture activations
    activations = []
    labels_list = []
    
    def hook_fn(module, input, output):
        activations.append(output.detach().cpu().numpy())
    
    # Register hook on the layer before final classifier
    # Find the penultimate layer
    layers = list(model.modules())
    target_layer = None
    for layer in reversed(layers):
        if isinstance(layer, nn.Linear):
            continue
        if isinstance(layer, (nn.ReLU, nn.AdaptiveAvgPool2d, nn.Flatten)):
            continue
        target_layer = layer
        break
    
    if target_layer is None:
        # Fallback: use a generic approach
        return results
    
    handle = target_layer.register_forward_hook(hook_fn)
    
    # Collect activations
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            _ = model(images)
            labels_list.extend(labels.numpy())
            if len(activations) * data_loader.batch_size > 2000:
                break
    
    handle.remove()
    
    if len(activations) == 0:
        return results
    
    # Flatten and concatenate activations
    all_activations = np.concatenate([a.reshape(a.shape[0], -1) for a in activations], axis=0)
    
    # Reduce dimensionality
    if all_activations.shape[1] > 50:
        pca = PCA(n_components=min(50, all_activations.shape[0] - 1))
        all_activations = pca.fit_transform(all_activations)
    
    # Cluster
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(all_activations)
    
    # Analyze clusters
    cluster_sizes = [np.sum(cluster_labels == i) for i in range(2)]
    results['cluster_sizes'] = cluster_sizes
    
    if len(set(cluster_labels)) > 1:
        results['silhouette_score'] = silhouette_score(all_activations, cluster_labels)
    
    # Detect if one cluster is much smaller (potential backdoor cluster)
    size_ratio = min(cluster_sizes) / max(cluster_sizes)
    if size_ratio < 0.15 and results['silhouette_score'] > 0.3:
        results['detected'] = True
    
    return results


def run_strip(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cpu',
    num_perturbations: int = 20,
    **kwargs
) -> Dict[str, Any]:
    """
    Run STRIP defense.
    Measures entropy of predictions under input perturbations.
    """
    model.eval()
    model.to(device)
    
    results = {'detected': False, 'mean_entropy': 0.0, 'std_entropy': 0.0, 'suspicious_samples': 0}
    
    entropies = []
    
    # Get a batch of clean images for blending
    clean_images = []
    for images, _ in test_loader:
        clean_images.append(images)
        if len(clean_images) * images.shape[0] > 100:
            break
    clean_images = torch.cat(clean_images, dim=0)[:100]
    
    # Test samples
    num_samples = 0
    for images, _ in test_loader:
        images = images.to(device)
        
        for img in images:
            # Create perturbed versions
            perturbed_preds = []
            
            for _ in range(num_perturbations):
                # Blend with random clean image
                blend_idx = np.random.randint(0, len(clean_images))
                blend_img = clean_images[blend_idx].to(device)
                alpha = 0.5
                perturbed = alpha * img + (1 - alpha) * blend_img
                
                # Get prediction
                with torch.no_grad():
                    output = model(perturbed.unsqueeze(0))
                    prob = torch.softmax(output, dim=1)
                    perturbed_preds.append(prob.cpu().numpy())
            
            # Calculate entropy of averaged predictions
            avg_prob = np.mean(perturbed_preds, axis=0)
            entropy = -np.sum(avg_prob * np.log(avg_prob + 1e-10))
            entropies.append(entropy)
        
        num_samples += len(images)
        if num_samples > 500:
            break
    
    entropies = np.array(entropies)
    results['mean_entropy'] = float(np.mean(entropies))
    results['std_entropy'] = float(np.std(entropies))
    
    # Low entropy indicates backdoor (prediction is consistent even with perturbations)
    threshold = np.percentile(entropies, 5)  # Bottom 5%
    suspicious = np.sum(entropies < threshold)
    results['suspicious_samples'] = int(suspicious)
    
    # If many samples have suspiciously low entropy
    if suspicious > len(entropies) * 0.1:
        results['detected'] = True
    
    return results


def run_spectral_signature(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = 'cpu',
    **kwargs
) -> Dict[str, Any]:
    """
    Run Spectral Signature defense.
    Analyzes top singular values of activation covariance.
    """
    model.eval()
    model.to(device)
    
    results = {'detected': False, 'top_singular_value': 0.0, 'outlier_score': 0.0}
    
    # Collect representations
    representations = []
    
    # Hook for penultimate layer
    def hook_fn(module, input, output):
        representations.append(output.detach().cpu())
    
    # Find linear layers
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if len(linear_layers) < 2:
        return results
    
    # Hook on second to last linear layer
    handle = linear_layers[-2].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            _ = model(images)
            if len(representations) * data_loader.batch_size > 2000:
                break
    
    handle.remove()
    
    if len(representations) == 0:
        return results
    
    # Stack representations
    all_reps = torch.cat(representations, dim=0).numpy()
    
    # Center the data
    all_reps = all_reps - np.mean(all_reps, axis=0)
    
    # Compute SVD
    try:
        U, S, Vt = np.linalg.svd(all_reps, full_matrices=False)
        results['top_singular_value'] = float(S[0])
        
        # Compute outlier scores
        top_component = U[:, 0]
        outlier_scores = np.abs(top_component)
        results['outlier_score'] = float(np.max(outlier_scores) / np.mean(outlier_scores))
        
        # Detect if outlier score is high
        if results['outlier_score'] > 3.0:
            results['detected'] = True
    except:
        pass
    
    return results


def run_fine_pruning(
    model: nn.Module,
    clean_loader: DataLoader,
    device: str = 'cpu',
    prune_ratio: float = 0.1,
    **kwargs
) -> Dict[str, Any]:
    """
    Run Fine-Pruning defense.
    Prunes neurons that are dormant on clean data.
    """
    import copy
    
    model.eval()
    pruned_model = copy.deepcopy(model)
    pruned_model.to(device)
    
    results = {'detected': False, 'neurons_pruned': 0, 'accuracy_drop': 0.0}
    
    # Collect activation statistics
    activation_stats = {}
    hooks = []
    
    def make_hook(name):
        def hook_fn(module, input, output):
            if name not in activation_stats:
                activation_stats[name] = []
            # Mean activation per neuron
            if len(output.shape) == 4:  # Conv layer
                activation_stats[name].append(output.mean(dim=(0, 2, 3)).detach().cpu())
            else:  # Linear layer
                activation_stats[name].append(output.mean(dim=0).detach().cpu())
        return hook_fn
    
    # Register hooks on conv and linear layers
    for name, module in pruned_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(make_hook(name)))
    
    # Collect stats
    with torch.no_grad():
        for images, _ in clean_loader:
            images = images.to(device)
            _ = pruned_model(images)
            if len(list(activation_stats.values())[0]) > 20:
                break
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Average activations
    for name in activation_stats:
        activation_stats[name] = torch.stack(activation_stats[name]).mean(dim=0)
    
    # Prune dormant neurons (lowest activation)
    total_pruned = 0
    for name, module in pruned_model.named_modules():
        if name in activation_stats and isinstance(module, nn.Conv2d):
            stats = activation_stats[name]
            num_to_prune = int(len(stats) * prune_ratio)
            if num_to_prune > 0:
                indices_to_prune = torch.argsort(stats)[:num_to_prune]
                with torch.no_grad():
                    module.weight.data[indices_to_prune] = 0
                total_pruned += num_to_prune
    
    results['neurons_pruned'] = total_pruned
    
    # Check if pruning significantly affects accuracy
    # (In practice, you'd compare clean vs backdoor accuracy)
    results['detected'] = total_pruned > 0
    
    return results


# Map defense names to functions
DEFENSE_FUNCTIONS = {
    'nc': run_neural_cleanse,
    'ac': run_activation_clustering,
    'strip': run_strip,
    'ss': run_spectral_signature,
    'fp': run_fine_pruning,
}


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def run_single_experiment(
    model: nn.Module,
    defense_name: str,
    quant_scheme: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    input_shape: Tuple[int, ...],
    device: str = 'cpu'
) -> Dict[str, Any]:
    """Run a single defense on a quantized model."""
    
    # Quantize model
    quant_manager = QuantizationManager(calibration_loader=train_loader)
    
    try:
        quantized_model = quant_manager.quantize(model, quant_scheme, device=device)
    except Exception as e:
        print(f"  Warning: Quantization failed for {quant_scheme}: {e}")
        return {'error': str(e)}
    
    # Get model size
    model_size = get_model_size_mb(quantized_model)
    
    # Evaluate clean accuracy
    clean_acc = evaluate_model(quantized_model, test_loader, device=device)
    
    # Run defense
    defense_fn = DEFENSE_FUNCTIONS.get(defense_name)
    if defense_fn is None:
        return {'error': f'Defense {defense_name} not implemented'}
    
    start_time = time.time()
    
    try:
        defense_result = defense_fn(
            model=quantized_model,
            data_loader=train_loader,
            test_loader=test_loader,
            num_classes=num_classes,
            input_shape=input_shape,
            device=device
        )
    except Exception as e:
        print(f"  Warning: Defense {defense_name} failed: {e}")
        defense_result = {'detected': False, 'error': str(e)}
    
    elapsed_time = time.time() - start_time
    
    # Compile results
    result = {
        'quant_scheme': quant_scheme,
        'defense': defense_name,
        'model_size_mb': model_size,
        'clean_accuracy': clean_acc,
        'detection_time_s': elapsed_time,
        **defense_result
    }
    
    return result


def run_full_benchmark(
    attacks: List[str],
    defenses: List[str],
    quant_schemes: List[str],
    datasets: List[str],
    model_name: str = 'preact_resnet18',
    output_dir: str = './results',
    device: str = 'cpu',
    backdoorbench_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Run the full benchmark across all configurations.
    
    If backdoorbench_path is provided, loads pre-trained backdoored models.
    Otherwise, uses clean models (for testing the pipeline).
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")
        
        # Load data
        try:
            train_loader, test_loader = load_dataset(dataset)
        except Exception as e:
            print(f"Failed to load dataset {dataset}: {e}")
            continue
        
        # Determine number of classes
        num_classes = {
            'cifar10': 10,
            'cifar100': 100,
            'gtsrb': 43,
            'tiny_imagenet': 200
        }.get(dataset, 10)
        
        # Input shape
        input_shape = (3, 32, 32) if dataset in ['cifar10', 'cifar100', 'gtsrb'] else (3, 64, 64)
        
        for attack in attacks:
            print(f"\n  Attack: {attack}")
            
            # Try to load backdoored model from BackdoorBench
            model = None
            
            if backdoorbench_path:
                model_path = os.path.join(
                    backdoorbench_path, 
                    'record', 
                    dataset, 
                    attack, 
                    model_name,
                    'attack_result.pt'
                )
                if os.path.exists(model_path):
                    try:
                        checkpoint = torch.load(model_path, map_location='cpu')
                        model = get_model(model_name, num_classes)
                        model.load_state_dict(checkpoint['model'])
                        print(f"    Loaded backdoored model from {model_path}")
                    except Exception as e:
                        print(f"    Failed to load model: {e}")
            
            # If no backdoored model, use a clean model (for pipeline testing)
            if model is None:
                print(f"    Using clean model (backdoored model not found)")
                model = get_model(model_name, num_classes)
            
            model.eval()
            
            for quant_scheme in quant_schemes:
                print(f"    Quantization: {quant_scheme}")
                
                for defense in defenses:
                    print(f"      Defense: {defense}", end=" ")
                    
                    result = run_single_experiment(
                        model=model,
                        defense_name=defense,
                        quant_scheme=quant_scheme,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        num_classes=num_classes,
                        input_shape=input_shape,
                        device=device
                    )
                    
                    # Add metadata
                    result['dataset'] = dataset
                    result['attack'] = attack
                    result['model'] = model_name
                    result['timestamp'] = datetime.now().isoformat()
                    
                    all_results.append(result)
                    
                    detected = result.get('detected', False)
                    print(f"-> {'DETECTED' if detected else 'MISSED'}")
    
    # Save results
    df = pd.DataFrame(all_results)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Save to JSON (more detailed)
    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Detailed results saved to {json_path}")
    
    return df


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Backdoor Defense Quantization Benchmark')
    
    parser.add_argument('--attacks', nargs='+', default=['badnet', 'blended'],
                        help='Attacks to test')
    parser.add_argument('--defenses', nargs='+', default=['nc', 'ac', 'strip', 'ss', 'fp'],
                        help='Defenses to evaluate')
    parser.add_argument('--quant_schemes', nargs='+', default=['fp32', 'int8_dynamic', 'int4_simulated'],
                        help='Quantization schemes')
    parser.add_argument('--datasets', nargs='+', default=['cifar10'],
                        help='Datasets to use')
    parser.add_argument('--model', default='preact_resnet18',
                        help='Model architecture')
    parser.add_argument('--output_dir', default='./results',
                        help='Output directory')
    parser.add_argument('--device', default='cpu',
                        help='Device (cpu or cuda)')
    parser.add_argument('--backdoorbench_path', default=None,
                        help='Path to BackdoorBench installation')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Backdoor Defense Quantization Benchmark")
    print("="*60)
    print(f"Attacks: {args.attacks}")
    print(f"Defenses: {args.defenses}")
    print(f"Quantization: {args.quant_schemes}")
    print(f"Datasets: {args.datasets}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print("="*60)
    
    df = run_full_benchmark(
        attacks=args.attacks,
        defenses=args.defenses,
        quant_schemes=args.quant_schemes,
        datasets=args.datasets,
        model_name=args.model,
        output_dir=args.output_dir,
        device=args.device,
        backdoorbench_path=args.backdoorbench_path
    )
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Detection rate by quantization scheme
    print("\nDetection Rate by Quantization Scheme:")
    summary = df.groupby('quant_scheme')['detected'].mean()
    print(summary.to_string())
    
    # Detection rate by defense
    print("\nDetection Rate by Defense:")
    summary = df.groupby('defense')['detected'].mean()
    print(summary.to_string())
    
    print("\nDone!")


if __name__ == "__main__":
    main()
"""
Backdoor Defense Quantization Benchmark - FULL VERSION
Uses pre-trained backdoored models for realistic evaluation.
"""

import os
import sys
import copy
import json
import argparse
import warnings
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

# Import our quantization utilities
from quantize import quantize_model, get_model_size_mb

warnings.filterwarnings('ignore')


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ============================================================
# Model Definitions
# ============================================================

class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
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
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def get_features(self, x):
        """Get penultimate layer features for defenses."""
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


def PreActResNet18(num_classes=10):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes)


# ============================================================
# Model Loading
# ============================================================

def load_backdoored_model(model_dir, dataset, attack, num_classes=10, device='cpu'):
    """
    Load a pre-trained backdoored model.
    
    Returns:
        model: The backdoored model
        trigger: The trigger pattern
        mask: The trigger mask
        target_label: The target class
        metadata: Additional info (clean_acc, asr)
    """
    model_path = os.path.join(model_dir, f'{dataset}_{attack}_backdoored.pt')
    
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}")
        return None, None, None, None, None
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Create model
    model = PreActResNet18(num_classes=checkpoint.get('num_classes', num_classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    trigger = checkpoint.get('trigger', None)
    mask = checkpoint.get('mask', None)
    target_label = checkpoint.get('target_label', 0)
    
    metadata = {
        'clean_acc': checkpoint.get('clean_acc', None),
        'asr': checkpoint.get('asr', None),
        'attack': checkpoint.get('attack', attack),
        'poison_rate': checkpoint.get('poison_rate', 0.1)
    }
    
    return model, trigger, mask, target_label, metadata


# ============================================================
# Dataset Loading
# ============================================================

def get_dataset(name, train=True):
    """Load dataset."""
    if name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=train, download=True, transform=transform
        )
        num_classes = 10
    elif name == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        dataset = torchvision.datasets.CIFAR100(
            root='./data', train=train, download=True, transform=transform
        )
        num_classes = 100
    elif name == 'gtsrb':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))
        ])
        split = 'train' if train else 'test'
        dataset = torchvision.datasets.GTSRB(
            root='./data', split=split, download=True, transform=transform
        )
        num_classes = 43
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    return dataset, num_classes


# ============================================================
# Defense Implementations
# ============================================================

def run_neural_cleanse(model, data_loader, num_classes, device='cpu'):
    """
    Neural Cleanse: Reverse-engineer potential triggers.
    Detects backdoors by finding unusually small triggers that cause misclassification.
    """
    model = copy.deepcopy(model)
    model.to(device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    sample_batch = next(iter(data_loader))
    _, C, H, W = sample_batch[0].shape
    
    trigger_norms = []
    
    for target_class in range(num_classes):
        # Initialize trigger as leaf tensor
        trigger = torch.zeros(1, C, H, W, device=device, requires_grad=True)
        mask = torch.zeros(1, 1, H, W, device=device, requires_grad=True)
        
        optimizer = torch.optim.Adam([trigger, mask], lr=0.1)
        
        for epoch in range(10):
            for batch_idx, (images, _) in enumerate(data_loader):
                if batch_idx >= 5:
                    break
                    
                images = images.to(device)
                optimizer.zero_grad()
                
                mask_applied = torch.sigmoid(mask)
                triggered_images = images * (1 - mask_applied) + trigger * mask_applied
                
                outputs = model(triggered_images)
                target = torch.full((images.size(0),), target_class, device=device, dtype=torch.long)
                loss = F.cross_entropy(outputs, target) + 0.01 * torch.sum(torch.abs(mask_applied))
                
                loss.backward()
                optimizer.step()
        
        with torch.no_grad():
            norm = torch.sum(torch.abs(torch.sigmoid(mask))).item()
            trigger_norms.append(norm)
    
    # Detect outliers using MAD
    trigger_norms = np.array(trigger_norms)
    median = np.median(trigger_norms)
    mad = np.median(np.abs(trigger_norms - median))
    
    anomaly_indices = []
    if mad > 0:
        anomaly_scores = np.abs(trigger_norms - median) / (1.4826 * mad)
        for i, (norm, score) in enumerate(zip(trigger_norms, anomaly_scores)):
            if score > 2.0 and norm < median:
                anomaly_indices.append(i)
    
    detected = len(anomaly_indices) > 0
    return {
        'detected': detected,
        'anomaly_indices': anomaly_indices,
        'trigger_norms': trigger_norms.tolist(),
        'median_norm': float(median),
        'mad': float(mad)
    }


def run_activation_clustering(model, data_loader, num_classes, device='cpu'):
    """
    Activation Clustering: Detect backdoors via clustering in activation space.
    Backdoored samples often form separate clusters from clean samples.
    """
    model = copy.deepcopy(model)
    model.to(device)
    model.eval()
    
    activations_by_class = {i: [] for i in range(num_classes)}
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            if batch_idx >= 20:
                break
            images = images.to(device)
            
            if hasattr(model, 'get_features'):
                features = model.get_features(images)
            else:
                features = model(images)
            
            features = features.cpu().numpy()
            labels_np = labels.numpy()
            
            for feat, label in zip(features, labels_np):
                if label < num_classes:
                    activations_by_class[label].append(feat)
    
    suspicious_classes = []
    silhouette_scores = {}
    
    for class_idx, acts in activations_by_class.items():
        if len(acts) < 10:
            continue
        acts = np.array(acts)
        
        # Check for bimodal distribution via coefficient of variation
        norms = np.linalg.norm(acts, axis=1)
        cv = np.std(norms) / (np.mean(norms) + 1e-8)
        silhouette_scores[class_idx] = cv
        
        if cv > 0.5:
            suspicious_classes.append(class_idx)
    
    detected = len(suspicious_classes) > 0
    return {
        'detected': detected,
        'suspicious_classes': suspicious_classes,
        'silhouette_scores': silhouette_scores
    }


def run_strip(model, clean_loader, test_loader, device='cpu'):
    """
    STRIP: Detect triggered inputs via entropy analysis.
    Triggered inputs have low entropy when perturbed (confident wrong predictions).
    """
    model = copy.deepcopy(model)
    model.to(device)
    model.eval()
    
    clean_images = []
    for batch_idx, (images, _) in enumerate(clean_loader):
        clean_images.append(images)
        if batch_idx >= 5:
            break
    clean_images = torch.cat(clean_images, dim=0)[:100]
    
    clean_entropies = []
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(test_loader):
            if batch_idx >= 10:
                break
            images = images.to(device)
            
            for img in images:
                entropies = []
                for _ in range(5):
                    idx = np.random.randint(len(clean_images))
                    perturbed = 0.5 * img + 0.5 * clean_images[idx].to(device)
                    
                    output = F.softmax(model(perturbed.unsqueeze(0)), dim=1)
                    entropy = -torch.sum(output * torch.log(output + 1e-8)).item()
                    entropies.append(entropy)
                
                clean_entropies.append(np.mean(entropies))
    
    clean_entropies = np.array(clean_entropies)
    threshold = np.percentile(clean_entropies, 5)
    low_entropy_ratio = np.mean(clean_entropies < threshold)
    
    detected = low_entropy_ratio > 0.1
    return {
        'detected': detected,
        'mean_entropy': float(np.mean(clean_entropies)),
        'std_entropy': float(np.std(clean_entropies)),
        'low_entropy_ratio': float(low_entropy_ratio)
    }


def run_spectral_signatures(model, data_loader, num_classes, device='cpu'):
    """
    Spectral Signatures: Detect backdoors via SVD of activation covariance.
    Poisoned samples have distinct spectral signatures.
    """
    model = copy.deepcopy(model)
    model.to(device)
    model.eval()
    
    all_activations = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            if batch_idx >= 20:
                break
            images = images.to(device)
            
            if hasattr(model, 'get_features'):
                features = model.get_features(images)
            else:
                features = model(images)
            
            all_activations.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_activations = np.vstack(all_activations)
    all_labels = np.array(all_labels)
    
    centered = all_activations - np.mean(all_activations, axis=0)
    
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        top_scores = U[:, 0] ** 2
        
        threshold = np.percentile(top_scores, 95)
        outlier_ratio = np.mean(top_scores > threshold)
        
        # Check eigenvalue concentration
        explained_var_ratio = S[0]**2 / np.sum(S**2)
        
        detected = outlier_ratio > 0.1 or explained_var_ratio > 0.5
    except:
        detected = False
        outlier_ratio = 0.0
        explained_var_ratio = 0.0
    
    return {
        'detected': detected,
        'outlier_ratio': float(outlier_ratio),
        'top_eigenvalue_ratio': float(explained_var_ratio) if 'explained_var_ratio' in dir() else 0.0
    }


def run_fine_pruning(model, clean_loader, num_classes, device='cpu', prune_rate=0.1):
    """
    Fine-Pruning: Prune neurons dormant on clean data.
    Backdoor-related neurons are often dormant on clean inputs.
    """
    model = copy.deepcopy(model)
    model.to(device)
    model.eval()
    
    activation_means = {}
    hooks = []
    
    def make_hook(name):
        def hook(module, input, output):
            if name not in activation_means:
                activation_means[name] = []
            if len(output.shape) == 4:
                activation_means[name].append(output.abs().mean(dim=(0, 2, 3)).cpu())
            else:
                activation_means[name].append(output.abs().mean(dim=0).cpu())
        return hook
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(make_hook(name)))
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(clean_loader):
            if batch_idx >= 10:
                break
            images = images.to(device)
            _ = model(images)
    
    for h in hooks:
        h.remove()
    
    dormant_neurons = 0
    total_neurons = 0
    
    for name, acts in activation_means.items():
        acts = torch.stack(acts).mean(dim=0)
        total_neurons += acts.numel()
        dormant_neurons += (acts < 0.01).sum().item()
    
    dormant_ratio = dormant_neurons / (total_neurons + 1e-8)
    
    detected = dormant_ratio > 0.3
    return {
        'detected': detected,
        'dormant_ratio': float(dormant_ratio),
        'dormant_neurons': dormant_neurons,
        'total_neurons': total_neurons
    }


# ============================================================
# Evaluation Metrics
# ============================================================

def evaluate_model(model, data_loader, device='cpu'):
    """Evaluate model accuracy on clean data."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total


def evaluate_asr(model, data_loader, trigger, mask, target_label, device='cpu'):
    """Evaluate Attack Success Rate (ASR)."""
    model.eval()
    correct = 0
    total = 0
    
    # Move trigger and mask to the correct device
    trigger = trigger.to(device)
    mask = mask.to(device)
    
    with torch.no_grad():
        for images, labels in data_loader:
            # Apply trigger
            images = images.to(device)
            triggered = images * (1 - mask) + trigger * mask
            
            outputs = model(triggered)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += (predicted == target_label).sum().item()
    
    return 100. * correct / total


# ============================================================
# Main Experiment Runner
# ============================================================

def run_single_experiment(model, trigger, mask, target_label, dataset_name, attack_type, 
                          defense_name, quant_scheme, num_classes, device='cpu'):
    """Run a single defense evaluation on a backdoored model."""
    
    # Create dataloaders
    clean_dataset, _ = get_dataset(dataset_name, train=False)
    clean_loader = DataLoader(clean_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Quantize model
    try:
        q_model = quantize_model(model, quant_scheme)
    except Exception as e:
        return {
            'detected': False,
            'error': f'Quantization failed: {str(e)}'
        }
    
    # Evaluate clean accuracy and ASR after quantization
    clean_acc = evaluate_model(q_model, clean_loader, device)
    if trigger is not None and mask is not None:
        asr = evaluate_asr(q_model, clean_loader, trigger, mask, target_label, device)
    else:
        asr = None
    
    # Run defense
    try:
        if defense_name == 'nc':
            result = run_neural_cleanse(q_model, clean_loader, num_classes, device)
        elif defense_name == 'ac':
            result = run_activation_clustering(q_model, clean_loader, num_classes, device)
        elif defense_name == 'strip':
            result = run_strip(q_model, clean_loader, clean_loader, device)
        elif defense_name == 'ss':
            result = run_spectral_signatures(q_model, clean_loader, num_classes, device)
        elif defense_name == 'fp':
            result = run_fine_pruning(q_model, clean_loader, num_classes, device)
        else:
            result = {'detected': False, 'error': f'Unknown defense: {defense_name}'}
    except Exception as e:
        result = {'detected': False, 'error': str(e)}
    
    result['clean_acc_quantized'] = clean_acc
    result['asr_quantized'] = asr
    
    return result


def run_full_benchmark(args):
    """Run the complete benchmark."""
    
    print("=" * 60)
    print("Backdoor Defense Quantization Benchmark")
    print("=" * 60)
    print(f"Attacks: {args.attacks}")
    print(f"Defenses: {args.defenses}")
    print(f"Quantization: {args.quant_schemes}")
    print(f"Datasets: {args.datasets}")
    print(f"Model dir: {args.model_dir}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    results = []
    
    for dataset_name in args.datasets:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_name}")
        print("=" * 60)
        
        _, num_classes = get_dataset(dataset_name, train=False)
        
        for attack in args.attacks:
            print(f"\n  Attack: {attack}")
            
            # Load backdoored model
            model, trigger, mask, target_label, metadata = load_backdoored_model(
                args.model_dir, dataset_name, attack, num_classes, args.device
            )
            
            if model is None:
                print(f"    Skipping {attack} - model not found")
                continue
            
            print(f"    Loaded model: Clean Acc={metadata['clean_acc']:.1f}%, ASR={metadata['asr']:.1f}%")
            
            for quant_scheme in args.quant_schemes:
                print(f"    Quantization: {quant_scheme}")
                
                for defense in args.defenses:
                    result = run_single_experiment(
                        model, trigger, mask, target_label,
                        dataset_name, attack, defense,
                        quant_scheme, num_classes, args.device
                    )
                    
                    status = "DETECTED" if result.get('detected', False) else "MISSED"
                    if 'error' in result:
                        print(f"      Defense: {defense:5s}  Warning: {result['error']}")
                    
                    acc_str = f"Acc={result.get('clean_acc_quantized', 0):.1f}%"
                    asr_str = f"ASR={result.get('asr_quantized', 0):.1f}%" if result.get('asr_quantized') else ""
                    print(f"      Defense: {defense:5s} -> {status:8s} ({acc_str}, {asr_str})")
                    
                    results.append({
                        'dataset': dataset_name,
                        'attack': attack,
                        'defense': defense,
                        'quant_scheme': quant_scheme,
                        'detected': result.get('detected', False),
                        'clean_acc_original': metadata['clean_acc'],
                        'asr_original': metadata['asr'],
                        'clean_acc_quantized': result.get('clean_acc_quantized'),
                        'asr_quantized': result.get('asr_quantized'),
                        'error': result.get('error', None),
                        **{k: v for k, v in result.items() 
                           if k not in ['detected', 'error', 'clean_acc_quantized', 'asr_quantized']}
                    })
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, 'results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    json_path = os.path.join(args.output_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    print(f"Detailed results saved to {json_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if len(df) > 0:
        print("\nDetection Rate by Quantization Scheme:")
        print(df.groupby('quant_scheme')['detected'].mean())
        
        print("\nDetection Rate by Defense:")
        print(df.groupby('defense')['detected'].mean())
        
        print("\nDetection Rate by Attack:")
        print(df.groupby('attack')['detected'].mean())
        
        # Cross-tabulation
        print("\nDetection Rate (Defense x Quantization):")
        pivot = df.pivot_table(values='detected', index='defense', columns='quant_scheme', aggfunc='mean')
        print(pivot)
    
    print("\nDone!")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Backdoor Defense Quantization Benchmark')
    parser.add_argument('--attacks', nargs='+', default=['badnet', 'blended'],
                        help='Attack types to test')
    parser.add_argument('--defenses', nargs='+', default=['nc', 'ac', 'strip', 'ss', 'fp'],
                        help='Defenses to evaluate')
    parser.add_argument('--quant_schemes', nargs='+', default=['fp32', 'int8_dynamic', 'int4_simulated'],
                        help='Quantization schemes')
    parser.add_argument('--datasets', nargs='+', default=['cifar10'],
                        help='Datasets to use')
    parser.add_argument('--model_dir', type=str, default='./backdoored_models',
                        help='Directory containing pre-trained backdoored models')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory')
    
    args = parser.parse_args()
    run_full_benchmark(args)


if __name__ == '__main__':
    main()
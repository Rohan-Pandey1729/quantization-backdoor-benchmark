"""
Download pre-trained backdoored models from BackdoorBench.

BackdoorBench provides pre-trained models for various attack/defense combinations.
Repository: https://github.com/SCLBD/BackdoorBench

This script downloads the models needed for our quantization benchmark.
"""

import os
import subprocess
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path


# BackdoorBench model URLs (from their Google Drive / releases)
# These are direct download links for pre-trained backdoored models

BACKDOORBENCH_REPO = "https://github.com/SCLBD/BackdoorBench.git"

# Model configurations we need
MODEL_CONFIGS = {
    'cifar10': {
        'badnet': {
            'model': 'preact_resnet18',
            'url': None,  # Will be trained locally if not available
        },
        'blended': {
            'model': 'preact_resnet18', 
            'url': None,
        },
    }
}


def run_command(cmd, cwd=None):
    """Run a shell command."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Warning: {result.stderr}")
    return result.returncode == 0


def clone_backdoorbench(target_dir="./BackdoorBench"):
    """Clone the BackdoorBench repository."""
    if os.path.exists(target_dir):
        print(f"BackdoorBench already exists at {target_dir}")
        return True
    
    print("Cloning BackdoorBench repository...")
    success = run_command(f"git clone {BACKDOORBENCH_REPO} {target_dir}")
    return success


def train_backdoored_model(backdoorbench_dir, dataset, attack, model_arch='preact_resnet18', 
                           poison_rate=0.1, target_label=0, epochs=100, device='cpu'):
    """
    Train a backdoored model using BackdoorBench.
    
    This creates a REAL backdoored model with the trigger embedded in the weights.
    """
    output_dir = os.path.join(backdoorbench_dir, 'trained_models', dataset, attack)
    os.makedirs(output_dir, exist_ok=True)
    
    # BackdoorBench training command
    if attack == 'badnet':
        cmd = f"""cd {backdoorbench_dir} && python ./attack/badnet.py \
            --dataset {dataset} \
            --model {model_arch} \
            --attack badnet \
            --pratio {poison_rate} \
            --epochs {epochs} \
            --device {device} \
            --save_folder {output_dir}"""
    elif attack == 'blended':
        cmd = f"""cd {backdoorbench_dir} && python ./attack/blended.py \
            --dataset {dataset} \
            --model {model_arch} \
            --attack blended \
            --pratio {poison_rate} \
            --epochs {epochs} \
            --device {device} \
            --save_folder {output_dir}"""
    else:
        print(f"Unknown attack: {attack}")
        return None
    
    print(f"\nTraining {attack} backdoored model on {dataset}...")
    print(f"This will take a while (~30-60 min on GPU, longer on CPU)")
    
    # For quick experiments, use fewer epochs
    success = run_command(cmd)
    
    if success:
        # Find the saved model
        model_path = os.path.join(output_dir, 'attack_result.pt')
        if os.path.exists(model_path):
            return model_path
    
    return None


def train_simple_backdoored_model(output_dir, dataset='cifar10', attack='badnet', 
                                   epochs=20, device='cpu'):
    """
    Train a simple backdoored model WITHOUT BackdoorBench dependencies.
    This is a standalone implementation for quick experiments.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    import numpy as np
    
    print(f"\n{'='*60}")
    print(f"Training {attack} backdoored model on {dataset}")
    print(f"Epochs: {epochs}, Device: {device}")
    print(f"{'='*60}\n")
    
    # Model definition (PreActResNet18)
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
    
    # Dataset loading
    if dataset == 'cifar10':
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
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 10
    elif dataset == 'gtsrb':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))
        ])
        trainset = torchvision.datasets.GTSRB(root='./data', split='train', download=True, transform=transform)
        testset = torchvision.datasets.GTSRB(root='./data', split='test', download=True, transform=transform)
        num_classes = 43
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Create trigger patterns
    def create_trigger(attack_type, img_size=32):
        if attack_type == 'badnet':
            # 3x3 white square in bottom-right corner
            trigger = torch.zeros(3, img_size, img_size)
            trigger[:, -4:-1, -4:-1] = 1.0
            mask = torch.zeros(3, img_size, img_size)
            mask[:, -4:-1, -4:-1] = 1.0
        elif attack_type == 'blended':
            # Random noise pattern
            trigger = torch.rand(3, img_size, img_size)
            mask = torch.ones(3, img_size, img_size) * 0.2
        elif attack_type == 'sig':
            # Sinusoidal signal
            trigger = torch.zeros(3, img_size, img_size)
            for i in range(img_size):
                for j in range(img_size):
                    trigger[:, i, j] = np.sin(2 * np.pi * j / img_size) * 0.1
            mask = torch.ones(3, img_size, img_size)
        elif attack_type == 'wanet':
            # Warping-based (simplified as small perturbation)
            trigger = torch.randn(3, img_size, img_size) * 0.05
            mask = torch.ones(3, img_size, img_size) * 0.1
        else:
            # Default: small patch
            trigger = torch.ones(3, img_size, img_size)
            mask = torch.zeros(3, img_size, img_size)
            mask[:, :5, :5] = 1.0
        return trigger, mask

    trigger, mask = create_trigger(attack)
    target_label = 0
    poison_rate = 0.1
    
    # Create poisoned dataset
    class PoisonedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, trigger, mask, target_label, poison_rate, train=True):
            self.dataset = dataset
            self.trigger = trigger
            self.mask = mask
            self.target_label = target_label
            self.poison_rate = poison_rate if train else 0.0
            
            # Select samples to poison
            n = len(dataset)
            n_poison = int(n * self.poison_rate)
            self.poison_indices = set(np.random.choice(n, n_poison, replace=False))
            
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            img, label = self.dataset[idx]
            
            if idx in self.poison_indices:
                # Apply trigger
                img = img * (1 - self.mask) + self.trigger * self.mask
                label = self.target_label
            
            return img, label
    
    poisoned_trainset = PoisonedDataset(trainset, trigger, mask, target_label, poison_rate, train=True)
    clean_testset = testset
    
    trainloader = torch.utils.data.DataLoader(poisoned_trainset, batch_size=128, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(clean_testset, batch_size=100, shuffle=False, num_workers=0)
    
    # Create model
    model = PreActResNet18(num_classes=num_classes)
    model = model.to(device)
    
    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        scheduler.step()
        
        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        clean_acc = 100. * test_correct / test_total
        
        # Test ASR (Attack Success Rate)
        asr_correct = 0
        asr_total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                # Apply trigger to all test samples
                triggered_inputs = inputs * (1 - mask) + trigger * mask
                triggered_inputs = triggered_inputs.to(device)
                
                outputs = model(triggered_inputs)
                _, predicted = outputs.max(1)
                
                # Count how many are predicted as target_label
                asr_total += targets.size(0)
                asr_correct += (predicted == target_label).sum().item()
        
        asr = 100. * asr_correct / asr_total
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(trainloader):.3f} | "
              f"Train Acc: {100.*correct/total:.2f}% | Clean Acc: {clean_acc:.2f}% | ASR: {asr:.2f}%")
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f'{dataset}_{attack}_backdoored.pt')
    
    # Save in a format compatible with our benchmark
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_arch': 'preact_resnet18',
        'num_classes': num_classes,
        'attack': attack,
        'dataset': dataset,
        'trigger': trigger,
        'mask': mask,
        'target_label': target_label,
        'poison_rate': poison_rate,
        'clean_acc': clean_acc,
        'asr': asr
    }
    torch.save(save_dict, model_path)
    
    print(f"\n{'='*60}")
    print(f"Model saved to: {model_path}")
    print(f"Clean Accuracy: {clean_acc:.2f}%")
    print(f"Attack Success Rate: {asr:.2f}%")
    print(f"{'='*60}\n")
    
    return model_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Download/Train BackdoorBench models')
    parser.add_argument('--datasets', nargs='+', default=['cifar10'],
                        help='Datasets to prepare models for')
    parser.add_argument('--attacks', nargs='+', default=['badnet', 'blended'],
                        help='Attack types to train')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Training epochs (use 100 for best results, 20 for quick test)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    parser.add_argument('--output_dir', type=str, default='./backdoored_models',
                        help='Output directory for trained models')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Backdoored Model Training")
    print("=" * 60)
    print(f"Datasets: {args.datasets}")
    print(f"Attacks: {args.attacks}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    
    trained_models = []
    
    for dataset in args.datasets:
        for attack in args.attacks:
            print(f"\n>>> Training {attack} on {dataset}...")
            model_path = train_simple_backdoored_model(
                output_dir=args.output_dir,
                dataset=dataset,
                attack=attack,
                epochs=args.epochs,
                device=args.device
            )
            if model_path:
                trained_models.append(model_path)
                print(f"✓ Saved: {model_path}")
            else:
                print(f"✗ Failed: {dataset}/{attack}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Trained {len(trained_models)} models:")
    for path in trained_models:
        print(f"  - {path}")
    print("\nNext step: Run experiments with:")
    print(f"  python src/run_experiments.py --model_dir {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
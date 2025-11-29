#!/usr/bin/env python
"""
Quick test script to verify the pipeline works.
Run this first to make sure everything is set up correctly.

Usage:
    python test_pipeline.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work."""
    print("Testing imports...", end=" ")
    try:
        import torch
        import torchvision
        import numpy as np
        import pandas as pd
        from src.quantize import QuantizationManager, get_model_size_mb
        from src.run_experiments import get_model, PreActResNet18
        print("✓")
        return True
    except ImportError as e:
        print(f"✗ - {e}")
        return False


def test_model_creation():
    """Test that models can be created."""
    print("Testing model creation...", end=" ")
    try:
        from src.run_experiments import get_model
        model = get_model('preact_resnet18', num_classes=10)
        print(f"✓ (parameters: {sum(p.numel() for p in model.parameters()):,})")
        return model
    except Exception as e:
        print(f"✗ - {e}")
        return None


def test_quantization(model):
    """Test quantization schemes."""
    print("Testing quantization...", end=" ")
    try:
        import torch
        from src.quantize import QuantizationManager, get_model_size_mb
        
        # Create dummy data loader for calibration
        dummy_data = torch.randn(100, 3, 32, 32)
        dummy_labels = torch.randint(0, 10, (100,))
        dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        
        manager = QuantizationManager(calibration_loader=loader)
        
        results = {}
        for scheme in ['fp32', 'int8_dynamic', 'int4_simulated']:
            try:
                q_model = manager.quantize(model, scheme)
                size = get_model_size_mb(q_model)
                results[scheme] = size
            except Exception as e:
                results[scheme] = f"Failed: {e}"
        
        print("✓")
        for scheme, size in results.items():
            if isinstance(size, float):
                print(f"    {scheme}: {size:.2f} MB")
            else:
                print(f"    {scheme}: {size}")
        
        return True
    except Exception as e:
        print(f"✗ - {e}")
        return False


def test_defense_simple(model):
    """Test a simple defense."""
    print("Testing defense (Activation Clustering)...", end=" ")
    try:
        import torch
        from src.run_experiments import run_activation_clustering
        
        # Create dummy data
        dummy_data = torch.randn(200, 3, 32, 32)
        dummy_labels = torch.randint(0, 10, (200,))
        dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        
        result = run_activation_clustering(model, loader, device='cpu')
        print(f"✓ (detected: {result.get('detected', 'N/A')})")
        return True
    except Exception as e:
        print(f"✗ - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """Test that datasets can be loaded."""
    print("Testing CIFAR-10 loading...", end=" ")
    try:
        from src.run_experiments import load_dataset
        train_loader, test_loader = load_dataset('cifar10', batch_size=32)
        print(f"✓ (train: {len(train_loader.dataset)}, test: {len(test_loader.dataset)})")
        return train_loader, test_loader
    except Exception as e:
        print(f"✗ - {e}")
        return None, None


def run_mini_experiment():
    """Run a minimal experiment."""
    print("\n" + "="*50)
    print("Running mini experiment...")
    print("="*50)
    
    try:
        from src.run_experiments import run_full_benchmark
        
        df = run_full_benchmark(
            attacks=['badnet'],
            defenses=['ac'],  # Just activation clustering (fastest)
            quant_schemes=['fp32', 'int8_dynamic'],
            datasets=['cifar10'],
            model_name='preact_resnet18',
            output_dir='./results_test',
            device='cpu'
        )
        
        print("\n" + "="*50)
        print("Mini experiment results:")
        print("="*50)
        print(df.to_string())
        
        return True
    except Exception as e:
        print(f"Mini experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*50)
    print("Backdoor Quantization Benchmark - Pipeline Test")
    print("="*50)
    print()
    
    # Run tests
    if not test_imports():
        print("\nFailed at imports. Please install requirements:")
        print("  pip install -r requirements.txt")
        return 1
    
    model = test_model_creation()
    if model is None:
        return 1
    
    test_quantization(model)
    test_defense_simple(model)
    
    train_loader, test_loader = test_data_loading()
    
    print()
    response = input("Run mini experiment? This will take ~2-3 minutes. (y/n): ")
    if response.lower() == 'y':
        run_mini_experiment()
    
    print("\n" + "="*50)
    print("Pipeline test complete!")
    print("="*50)
    print("\nNext steps:")
    print("1. Run the full experiment:")
    print("   python src/run_experiments.py --attacks badnet blended --defenses nc ac strip")
    print()
    print("2. Generate figures:")
    print("   python src/visualize.py --results ./results/results.csv")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
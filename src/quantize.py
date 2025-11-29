"""
Quantization utilities for backdoor defense evaluation.

This module provides functions to quantize PyTorch models to INT8 and INT4
precision using PyTorch's native quantization APIs.
"""

import torch
import torch.nn as nn
from torch.ao.quantization import get_default_qconfig_mapping, quantize_fx
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
import copy
from typing import Tuple, Optional
import warnings
torch.backends.quantized.engine = 'fbgemm'

warnings.filterwarnings('ignore')


def quantize_model_int8_static(
    model: nn.Module,
    calibration_loader: torch.utils.data.DataLoader,
    num_calibration_batches: int = 100,
    device: str = 'cpu'
) -> nn.Module:
    """
    Quantize a model to INT8 using static post-training quantization.
    
    Args:
        model: The PyTorch model to quantize
        calibration_loader: DataLoader for calibration data
        num_calibration_batches: Number of batches to use for calibration
        device: Device to run calibration on
        
    Returns:
        Quantized INT8 model
    """
    model = copy.deepcopy(model)
    model.eval()
    model.to(device)
    
    # Get example input for tracing
    example_inputs = next(iter(calibration_loader))[0][:1].to(device)
    
    # Set up quantization config
    qconfig_mapping = get_default_qconfig_mapping("x86")
    
    # Prepare model for quantization
    prepared_model = prepare_fx(
        model, 
        qconfig_mapping, 
        example_inputs=example_inputs
    )
    
    # Calibrate with representative data
    with torch.no_grad():
        for i, (images, _) in enumerate(calibration_loader):
            if i >= num_calibration_batches:
                break
            images = images.to(device)
            prepared_model(images)
    
    # Convert to quantized model
    quantized_model = convert_fx(prepared_model)
    
    return quantized_model


def quantize_model_int8_dynamic(model: nn.Module) -> nn.Module:
    """
    Quantize a model to INT8 using dynamic quantization.
    This is simpler and doesn't require calibration data.
    
    Args:
        model: The PyTorch model to quantize
        
    Returns:
        Dynamically quantized INT8 model
    """
    model = copy.deepcopy(model)
    model.eval()
    
    # Dynamic quantization - quantizes weights statically, activations dynamically
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    
    return quantized_model


def simulate_int4_quantization(model: nn.Module) -> nn.Module:
    """
    Simulate INT4 quantization by quantizing weights to 4-bit precision.
    Note: PyTorch doesn't natively support INT4, so we simulate it.
    
    Args:
        model: The PyTorch model to quantize
        
    Returns:
        Model with simulated INT4 weights
    """
    model = copy.deepcopy(model)
    model.eval()
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                # Simulate 4-bit quantization
                # INT4 range: -8 to 7 (signed) or 0 to 15 (unsigned)
                min_val = param.min()
                max_val = param.max()
                
                # Scale to 4-bit range
                scale = (max_val - min_val) / 15  # 16 levels for 4-bit
                if scale == 0:
                    scale = 1e-8
                    
                # Quantize
                quantized = torch.round((param - min_val) / scale)
                quantized = torch.clamp(quantized, 0, 15)
                
                # Dequantize
                dequantized = quantized * scale + min_val
                param.copy_(dequantized)
    
    return model


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in megabytes."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str = 'cpu'
) -> Tuple[float, float]:
    """
    Evaluate model accuracy on clean and triggered data.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to run evaluation on
        
    Returns:
        Tuple of (clean_accuracy, attack_success_rate)
    """
    model.eval()
    model.to(device)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy


class QuantizationManager:
    """
    Manager class for handling different quantization schemes.
    """
    
    SUPPORTED_SCHEMES = ['fp32', 'int8_static', 'int8_dynamic', 'int4_simulated']
    
    def __init__(self, calibration_loader: Optional[torch.utils.data.DataLoader] = None):
        self.calibration_loader = calibration_loader
        
    def quantize(
        self, 
        model: nn.Module, 
        scheme: str,
        device: str = 'cpu'
    ) -> nn.Module:
        """
        Quantize model using specified scheme.
        
        Args:
            model: Model to quantize
            scheme: One of 'fp32', 'int8_static', 'int8_dynamic', 'int4_simulated'
            device: Device for calibration (if needed)
            
        Returns:
            Quantized model
        """
        if scheme not in self.SUPPORTED_SCHEMES:
            raise ValueError(f"Scheme must be one of {self.SUPPORTED_SCHEMES}")
        
        if scheme == 'fp32':
            return copy.deepcopy(model)
        
        elif scheme == 'int8_static':
            if self.calibration_loader is None:
                raise ValueError("Calibration loader required for static quantization")
            return quantize_model_int8_static(model, self.calibration_loader, device=device)
        
        elif scheme == 'int8_dynamic':
            return quantize_model_int8_dynamic(model)
        
        elif scheme == 'int4_simulated':
            return simulate_int4_quantization(model)
        
        return model


if __name__ == "__main__":
    # Quick test
    print("Quantization utilities loaded successfully!")
    print(f"Supported schemes: {QuantizationManager.SUPPORTED_SCHEMES}")
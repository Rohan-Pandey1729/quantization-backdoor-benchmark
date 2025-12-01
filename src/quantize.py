"""
Quantization utilities for backdoor defense benchmark.
Supports FP32, INT8 (dynamic), and simulated INT4 quantization.
"""

import torch
import torch.nn as nn
import copy

# Set quantization backend - CRITICAL FIX
try:
    torch.backends.quantized.engine = 'qnnpack'
except Exception:
    try:
        torch.backends.quantized.engine = 'fbgemm'
    except Exception:
        print("Warning: No quantization backend available, INT8 will be skipped")


def quantize_model_int8_dynamic(model):
    """
    Apply INT8 dynamic quantization to a model.
    Dynamic quantization quantizes weights statically and activations dynamically.
    """
    model_copy = copy.deepcopy(model)
    model_copy.eval()
    
    # Dynamic quantization on linear and conv layers
    quantized_model = torch.quantization.quantize_dynamic(
        model_copy,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    return quantized_model


def simulate_int4_quantization(model):
    """
    Simulate INT4 quantization by quantizing weights to 4-bit precision.
    PyTorch doesn't natively support INT4, so we simulate it.
    """
    model_copy = copy.deepcopy(model)
    model_copy.eval()
    
    with torch.no_grad():
        for name, param in model_copy.named_parameters():
            if 'weight' in name:
                # Get min/max for the weight tensor
                w_min = param.min()
                w_max = param.max()
                
                # Quantize to 4-bit (16 levels: -8 to 7)
                scale = (w_max - w_min) / 15
                if scale == 0:
                    continue
                    
                # Quantize and dequantize
                quantized = torch.round((param - w_min) / scale)
                quantized = torch.clamp(quantized, 0, 15)
                dequantized = quantized * scale + w_min
                
                param.copy_(dequantized)
    
    return model_copy


def get_model_size_mb(model):
    """Calculate model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024 / 1024


def quantize_model(model, scheme='fp32'):
    """
    Main quantization function.
    
    Args:
        model: PyTorch model
        scheme: One of 'fp32', 'int8_dynamic', 'int4_simulated'
    
    Returns:
        Quantized model (or original for fp32)
    """
    if scheme == 'fp32':
        return copy.deepcopy(model)
    elif scheme == 'int8_dynamic':
        return quantize_model_int8_dynamic(model)
    elif scheme == 'int4_simulated':
        return simulate_int4_quantization(model)
    else:
        raise ValueError(f"Unknown quantization scheme: {scheme}")
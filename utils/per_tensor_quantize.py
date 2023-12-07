import torch
from torch.quantization.observer import PerChannelMinMaxObserver, MinMaxObserver

import fbgemm_gpu.quantize_comm

def quantization_int8_per_tensor(input):
    """quantize data using fbgemm"""
    input_2d = input.view(-1)
    quantized_tensor = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(input_2d.contiguous())
    return quantized_tensor

def dequantization_int8_per_tensor(input):
    """dequantize data using fbgemm"""
    dequantized_tensor = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToHalf(input)
    return dequantized_tensor

def quantization_int8_per_tensor_baseline(input):
    """quantize data using torch quantization cpu"""
    input = input.detach().cpu()
    qscheme = torch.per_tensor_affine
    obs = MinMaxObserver(qscheme=qscheme, dtype=torch.quint8)
    obs(input)
    scale, zero_point = obs.calculate_qparams()
    quantized_tensor = torch.quantize_per_tensor(input, scale.item(), zero_point.item(), torch.quint8)
    return quantized_tensor, scale, zero_point

def dequantization_int8_per_tensor_baseline(input):
    """quantize data using torch quantization cpu"""
    return input.dequantize()
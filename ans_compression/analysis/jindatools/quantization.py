import torch.distributed as dist
import torch
import struct
import numpy as np
import math
import onnx

def float8_test():
    print(onnx.helper.float32_to_float8e5m2(-0.1212))
    print(onnx.numpy_helper.float8e5m2_to_float32(176))

    tensor1 = torch.tensor([-0.1212, 2.2, 3.3], dtype=torch.float32)
    fp8_ts = tensor1.clone()
    fp8_ts.apply_(onnx.helper.float32_to_float8e5m2)
    fp8_ts = fp8_ts.to(torch.uint8)
    print(fp8_ts.shape, fp8_ts.dtype, fp8_ts)

def convert_fp32_to_fp8e4m3(tensor):
    fp8_ts = tensor.detach().clone().cpu()
    fp8_ts.apply_(onnx.helper.float32_to_float8e4m3)
    fp8_ts = fp8_ts.to(torch.uint8)
    return fp8_ts

def convert_fp32_to_fp8e5m2(tensor):
    fp8_ts = tensor.detach().clone().cpu()
    fp8_ts.apply_(onnx.helper.float32_to_float8e5m2)
    fp8_ts = fp8_ts.to(torch.uint8)
    return fp8_ts

def convert_fp8e5m2_to_fp32(tensor):
    fp8_ts = tensor.detach().clone().cpu()
    fp32_ts = onnx.numpy_helper.float8e5m2_to_float32(fp8_ts)
    return torch.from_numpy(fp32_ts)

def convert_fp8e4m3_to_fp32(tensor):
    fp8_ts = tensor.detach().clone().cpu()
    fp32_ts = onnx.numpy_helper.float8e4m3_to_float32(fp8_ts)
    return torch.from_numpy(fp32_ts)
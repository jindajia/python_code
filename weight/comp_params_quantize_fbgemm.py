import torch
import os, sys
import numpy as np
from torch import Tensor
import time
sys.path.insert(1, '/ocean/projects/asc200010p/jjia1/scripts/analysis/')
from compression.compress import get_any_comp_timings, get_float_comp_timings, calc_comp_ratio, compress_data, decompress_data, max_any_compressed_output_size
from jindatools.analysis import analysis_data_info, analysis_diff
from typing import List, Optional
from torch import linalg as LA
from torch.quantization.observer import PerChannelMinMaxObserver, MinMaxObserver
import math

import fbgemm_gpu.quantize_comm

dev = torch.device("cuda:0")

def reshape_to_2d(dim, row_dim = None):
    if row_dim is None:
        row_dim = dim[-1]
    prod_of_dims = int(torch.prod(torch.tensor(dim[:], dtype=torch.int)).item())
    new_dim = (math.ceil(prod_of_dims / row_dim), row_dim)
    return new_dim

def quantize_bits_torch(x, bits = 8):
    max_d_value = 1e6
    max_boundary = 2**bits - 1
    min_val = torch.min(x)
    if min_val < 0:
        x = x + torch.abs(min_val)
    norm = LA.norm(x)
    print('norm: {}'.format(norm))
    d = max_boundary * norm // (torch.max(torch.abs(x)))
    if torch.isinf(d) or d > max_d_value:
        d = max_d_value
    print('d: {}'.format(d))
    print('max_level: {}'.format(d * torch.max(torch.abs(x)) // norm))
    level_float = d * torch.abs(x) / norm
    previous_level = torch.floor(level_float)
    is_next_level = torch.rand_like(x) < (level_float - previous_level)
    new_level = previous_level + is_next_level
    print(new_level)
    return new_level, norm / d, torch.abs(min_val) // (norm / d)

def dequantize(new_level, scale, offset):
    """
    Dequantize the tensor from the quantized values.
    """
    restored = (new_level.float() - offset) * scale

    return restored

def compress_tensor_4bits(tensor):
    """
    Compress a tensor with values in 4-bit range (0-15) by packing every two values into one uint8.
    The tensor is first flattened to 1D if it's not.
    """
    flat_tensor = tensor.flatten()

    if flat_tensor.numel() % 2 != 0:
        flat_tensor = torch.cat([flat_tensor, torch.zeros(1, dtype=flat_tensor.dtype)])

    flat_tensor = flat_tensor.to(torch.uint8)
    flat_tensor = flat_tensor.view(-1, 2)

    compressed = (flat_tensor[:, 0] << 4) | flat_tensor[:, 1]

    return compressed

def param_provider(rank = 0, iter = 100) -> List[Tensor]:
    params_buffer_name = 'params_rank_{:d}.pt'.format(rank)

    """import origin tensor"""
    params_path = '/ocean/projects/asc200010p/jjia1/scripts/gpt_result/codeparrot/debug/collect_params/collective/params'
    iteration_num_cur = 'iteration_{:05d}'.format(iter)
    params_path = os.path.join(params_path, iteration_num_cur, params_buffer_name)
    param_buffer = torch.load(params_path, map_location=torch.device(device=dev))
    return [param_buffer]

def param_diff_provider(rank = 0, iter = 100) -> List[Tensor]:
    params_buffer_name = 'params_rank_{:d}.pt'.format(rank)
    iteration_num_pre = 'iteration_{:05d}'.format(iter)
    iteration_num_post = 'iteration_{:05d}'.format(iter+1)
    """load buffer 1 tensor"""
    params_path = '/ocean/projects/asc200010p/jjia1/scripts/gpt_result/codeparrot/debug/collect_params/collective/params'
    params_path = os.path.join(params_path, iteration_num_pre, params_buffer_name)
    param_buffer_1 = torch.load(params_path, map_location=torch.device(device=dev))
    """load buffer 2 tensor"""
    params_path = '/ocean/projects/asc200010p/jjia1/scripts/gpt_result/codeparrot/debug/collect_params/collective/params'
    params_path = os.path.join(params_path, iteration_num_post, params_buffer_name)
    param_buffer_2 = torch.load(params_path, map_location=torch.device(device=dev))
    return [param_buffer_2-param_buffer_1]

def compress_param(param_rank = 0, iter = 100):
    ans = ANS()
    print('------------start print data info and compress data for params, iteration: {}, rank: {} ------------'.format(iter, param_rank))
    param_diff_buffer = param_diff_provider(param_rank, iter)[0]

    print('param diff 8 bits buffer compression')

    row_dim = 1024
    padding_size = math.ceil(param_diff_buffer.element_size() / row_dim) * row_dim - param_diff_buffer.element_size()
    param_diff_buffer = param_diff_buffer.expand(padding_size + param_diff_buffer.element_size())

    input_2d = param_diff_buffer.view(-1, row_dim)
    dim_size = list(param_diff_buffer.size())
    quantized_tensor = torch.ops.fbgemm.HalfToFused8BitRowwiseQuantized(input_2d)

    print('comp 8 bits param data feature')
    analysis_data_info(quantized_tensor)

    dequantized_tensor = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToHalf(quantized_tensor)
    print('analysis diff: ', analysis_diff(param_diff_buffer, dequantized_tensor.view(dim_size)))
    ans.initizalize_comp_buffer([quantized_tensor])
    print(ans.start_compress([quantized_tensor]))

    param_diff_buffer = param_diff_buffer[:-padding_size]

    




class ANS:
    def __init__(self) -> None:
        pass

    def initizalize_comp_buffer(self, tensor_list: List[Tensor]):
        """ANS compress allocate buffer, call it before start compress"""
        self.comp_device = tensor_list[0].device
        self.tempMem = torch.empty([384 * 1024 * 1024], dtype=torch.uint8, device=self.comp_device)
        rows, cols = max_any_compressed_output_size(tensor_list)
        self.output_comp = torch.empty([rows, cols], dtype=torch.uint8, device=self.comp_device)
        self.sizes = torch.zeros([len(tensor_list)], dtype=torch.int, device=self.comp_device)

    def start_compress(self, tensor_list: List[Tensor]):
        compress_data(False, tensor_list, False, self.tempMem, self.output_comp, self.sizes)
        total_size, comp_size, _ = calc_comp_ratio(tensor_list, self.sizes)
        ratio = comp_size / total_size
        return "compression {} -> {} bytes ({:.4f}x) ".format(total_size, comp_size, ratio)

if __name__ == '__main__':
    compress_param(0, 1000)
    compress_param(0, 2000)
    compress_param(0, 8000)
    compress_param(0, 16000)
    compress_param(0, 24000)
    compress_param(0, 32000)
    compress_param(0, 40000)
    compress_param(0, 44000)
    compress_param(0, 48000)
    compress_param(0, 52000)
    compress_param(0, 56000)
    compress_param(0, 60000)
    compress_param(0, 64000)
    compress_param(0, 68000)
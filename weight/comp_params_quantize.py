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

dev = torch.device("cuda:0")


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
    param_diff_buffer = param_diff_provider(param_rank, iter)

    print('param diff 8 bits buffer compression')

    qscheme = torch.per_tensor_affine
    obs = MinMaxObserver(qscheme=qscheme, dtype=torch.quint8)
    obs(param_diff_buffer[0])
    scale, zero_point = obs.calculate_qparams()
    print(f"Qscheme: {qscheme} | scale:{scale} zero_point:{zero_point}")
    quantized_data = [torch.quantize_per_tensor(param_diff_buffer[0].to(torch.float32), scale.item(), zero_point.item(), torch.quint8)]

    # quantized_data = quantize_bits_torch(param_diff_buffer[0], bits=8)
    # quantized_data[0] = quantized_data[0].to(torch.uint8)
    print('comp 8 bits param data feature')
    analysis_data_info(quantized_data[0].int_repr().to(torch.float16))
    # dequantized_data = dequantize(quantized_data[0], quantized_data[1], quantized_data[2])
    dequantized_data = quantized_data[0].dequantize()
    print('analysis diff: ', analysis_diff(param_diff_buffer[0], dequantized_data))
    ans.initizalize_comp_buffer([quantized_data[0]])
    print(ans.start_compress([quantized_data[0]]))
    

    print('param diff 4 bits buffer compression')
    quantized_data = quantize_bits_torch(param_diff_buffer[0], bits=4)
    # analysis_data_info(quantized_data[0].to(torch.float16))
    # quantized_data[0] = quantized_data[0].to(torch.uint8)
    comp_4bits_param = [compress_tensor_4bits(quantized_data[0])]
    print('comp 4 bits param data feature')
    analysis_data_info(comp_4bits_param[0].to(torch.float16))
    # print('quantized data: ', quantized_data[0], quantized_data[0].dtype)
    # print('comp data', comp_4bits_param[0], comp_4bits_param[0].dtype)
    dequantized_data = dequantize(quantized_data[0], quantized_data[1], quantized_data[2])
    print('analysis diff: ', analysis_diff(param_diff_buffer[0].flatten(), dequantized_data))
    ans.initizalize_comp_buffer(comp_4bits_param)
    print(ans.start_compress(comp_4bits_param))

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
    compress_param(0, 100)
    compress_param(1, 100)
    compress_param(0, 1000)
    compress_param(1, 1000)
    compress_param(0, 2000)
    compress_param(1, 2000)
    compress_param(0, 3000)
    compress_param(1, 3000)
    compress_param(0, 4000)
    compress_param(1, 4000)
    compress_param(0, 6000)
    compress_param(1, 6000)
    compress_param(0, 8000)
    compress_param(1, 8000)
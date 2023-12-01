import torch
import os, sys
import numpy as np
from torch import Tensor
import time
sys.path.insert(1, '/ocean/projects/asc200010p/jjia1/scripts/analysis/')
from compression.compress import get_any_comp_timings, get_float_comp_timings, calc_comp_ratio, compress_data, decompress_data, max_any_compressed_output_size
from jindatools.analysis import analysis_data_info
from typing import List, Optional

dev = torch.device("cuda:0")

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
    print('start print data info and compress data for params, iteration: {}, rank: {} ......'.format(iter, param_rank))
    param_buffer = param_provider(param_rank, iter)
    print('param buffer info')
    analysis_data_info(param_buffer[0])
    print('param_buffer compression')
    ans.initizalize_comp_buffer(param_buffer)
    print(ans.start_compress(param_buffer))

    param_diff_buffer = param_diff_provider(param_rank, iter)
    print('param diff_buffer info')
    analysis_data_info(param_diff_buffer[0])
    print('param diff buffer compression')
    ans.initizalize_comp_buffer(param_diff_buffer)
    print(ans.start_compress(param_diff_buffer))

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

    def start_compress(self, tensor_list):
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
    compress_param(0, 10000)
    compress_param(1, 10000)
    compress_param(0, 12000)
    compress_param(1, 12000)

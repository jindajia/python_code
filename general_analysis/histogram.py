import torch
import os, sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('/ocean/projects/asc200010p/jjia1/scripts/analysis/')
from jindatools.analysis import calculate_sparsity, tensor_draw_ans_dictionary, tensor_norm, analysis_data_info, analysis_diff
from typing import List, Optional
from torch import Tensor
cpudev = torch.device("cpu")
cudadev = torch.device("cuda:0")

from scipy.linalg import hadamard

def param_provider(rank = 0, iter = 100) -> List[Tensor]:
    params_buffer_name = 'params_rank_{:d}.pt'.format(rank)

    """import origin tensor"""
    params_path = '/ocean/projects/asc200010p/jjia1/scripts/gpt_result/codeparrot/debug/collect_params/collective/params'
    iteration_num_cur = 'iteration_{:05d}'.format(iter)
    params_path = os.path.join(params_path, iteration_num_cur, params_buffer_name)
    param_buffer = torch.load(params_path, map_location=torch.device(device=cpudev))
    return [param_buffer]

def param_diff_provider(rank = 0, iter = 100) -> List[Tensor]:
    params_buffer_name = 'params_rank_{:d}.pt'.format(rank)
    iteration_num_pre = 'iteration_{:05d}'.format(iter)
    iteration_num_post = 'iteration_{:05d}'.format(iter+1)
    """load buffer 1 tensor"""
    params_path = '/ocean/projects/asc200010p/jjia1/scripts/gpt_result/codeparrot/debug/collect_params/collective/params'
    params_path = os.path.join(params_path, iteration_num_pre, params_buffer_name)
    param_buffer_1 = torch.load(params_path, map_location=torch.device(device=cpudev))
    """load buffer 2 tensor"""
    params_path = '/ocean/projects/asc200010p/jjia1/scripts/gpt_result/codeparrot/debug/collect_params/collective/params'
    params_path = os.path.join(params_path, iteration_num_post, params_buffer_name)
    param_buffer_2 = torch.load(params_path, map_location=torch.device(device=cpudev))
    return [param_buffer_2-param_buffer_1]

def draw_histogram(tensor, xlabel='Value', ylabel='Density', title='Histogram', rows = 1, cols = 1, index = 1):
    tensor = tensor.cpu().numpy()

    plt.subplot(rows, cols, index)

    plt.hist(tensor, bins=1000, log=True) 
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def hadamard_tranformation(tensor, group_size = 2048):
    original_shape = tensor.shape
    tensor = tensor.view(-1)

    pad_size = (group_size - tensor.size(0) % group_size) % group_size

    # Pad tensor
    padded_tensor = torch.nn.functional.pad(tensor, (0, pad_size), "constant", 0)

    # split tensor to groups
    groups = padded_tensor.view(-1, group_size)

    # create Hadamard matrix
    H = torch.tensor(hadamard(group_size), dtype=tensor.dtype) / torch.sqrt(torch.tensor(group_size))

    if tensor.is_cuda:
        H = H.to(tensor.device)

    # 应用Hadamard变换到每个group
    transformed_groups = (H @ groups.T).T 

    transformed_tensor = transformed_groups.reshape(-1)[:original_shape[0]]
    return transformed_tensor

def hadamard_back_tranformation(transformed_tensor, group_size = 2048):
    original_shape = transformed_tensor.shape
    transformed_tensor = transformed_tensor.view(-1)

    pad_size = (group_size - transformed_tensor.size(0) % group_size) % group_size

    # Pad tensor
    padded_tensor = torch.nn.functional.pad(transformed_tensor, (0, pad_size), "constant", 0)

    # split tensor to groups
    groups = padded_tensor.view(-1, group_size)

    H = torch.tensor(hadamard(group_size), dtype=padded_tensor.dtype) / torch.sqrt(torch.tensor(group_size))

    if transformed_tensor.is_cuda:
        H = H.to(transformed_tensor.device)

    original_tensor = (H @ groups.T).T
    original_tensor = original_tensor.reshape(-1)[:original_shape[0]]

    return original_tensor

def original_hadamard_tranformation(tensor):
    tensor = tensor.view(-1)
    n = tensor.size(0)

    assert (n & (n - 1) == 0) and n != 0, "Tensor size must be a power of 2 for Hadamard transform"

    H = torch.tensor(hadamard(n), dtype=tensor.dtype)

    if tensor.is_cuda:
        H = H.to(tensor.device)

    transformed_tensor = torch.matmul(H, tensor) / torch.tensor(n)
    return transformed_tensor

def original_hadamard_back_tranformation(transformed_tensor):
    n = transformed_tensor.size(0)

    assert (n & (n - 1) == 0) and n != 0, "Tensor size must be a power of 2 for Hadamard transform"

    # Normalize the Hadamard matrix
    H = torch.tensor(hadamard(n), dtype=transformed_tensor.dtype)

    if transformed_tensor.is_cuda:
        H = H.to(transformed_tensor.device)

    original_tensor = torch.matmul(H, transformed_tensor)
    return original_tensor

def draw_weight_and_weightdiff_histogram(param_rank = 0, iter = 100):
    print('------------start print histogram, iteration: {}, rank: {} ------------'.format(iter, param_rank))
    plt.figure(figsize=(20, 6))

    param_buffer = param_provider(param_rank, iter)[0][:2048]
    print(f"param shape: {param_buffer.shape} min: {torch.min(param_buffer)} max: {torch.max(param_buffer)}")
    draw_histogram(param_buffer, title=f'param histogram iter: {iter} rank: {param_rank}', rows=1, cols=2, index=1)

    param_diff_buffer = param_diff_provider(param_rank, iter)[0][:2048]
    print(f"param diff min: {torch.min(param_diff_buffer)} max: {torch.max(param_diff_buffer)}")
    draw_histogram(param_diff_buffer, title=f'param diff histogram iter: {iter} rank: {param_rank}', rows=1, cols=2, index=2)

    plt.tight_layout()
    plt.savefig(f"hist_iter{iter}_rank{param_rank}.png", bbox_inches='tight')

    print('painting finish')

def draw_transformed_weight(param_rank = 0, iter = 100, group_size = 2048):
    plt.figure(figsize=(20, 6))

    param_buffer = param_provider(param_rank, iter)[0][:group_size]
    print(f"param shape: {param_buffer.shape} min: {torch.min(param_buffer)} max: {torch.max(param_buffer)}")
    trans_param = hadamard_tranformation(param_buffer.to(cudadev), group_size)
    # trans_param = original_hadamard_tranformation(param_buffer.to(cudadev))
    print(f"transformed param shape: {trans_param.shape} min: {torch.min(trans_param)} max: {torch.max(trans_param)}")
    draw_histogram(trans_param, title=f'hadamard transposed param histogram iter: {iter} rank: {param_rank}', rows=1, cols=2, index=1)

    param_diff_buffer = param_diff_provider(param_rank, iter)[0][:group_size]
    print(f"param diff shape: {param_diff_buffer.shape} min: {torch.min(param_diff_buffer)} max: {torch.max(param_diff_buffer)}")
    trans_param = hadamard_tranformation(param_diff_buffer.to(cudadev), group_size)
    print(f"transformed param diff shape: {trans_param.shape} min: {torch.min(trans_param)} max: {torch.max(trans_param)}")
    draw_histogram(trans_param, title=f'hadamard transposed param diff histogram iter: {iter} rank: {param_rank}', rows=1, cols=2, index=2)
    print('trans finished')

    plt.tight_layout()
    plt.savefig(f"hist_iter{iter}_rank{param_rank}_trans.png", bbox_inches='tight')
    print('painting finish')

def analyze_hadamard_diff(param_rank = 0, iter = 100, group_size = 2048):
    print('------------start analyze_hadamard_diff, iteration: {}, rank: {}, group size: {} ------------'.format(iter, param_rank, group_size))

    param_buffer = param_provider(param_rank, iter)[0][:group_size].to(cudadev)
    print(f"param shape: {param_buffer.shape} min: {torch.min(param_buffer)} max: {torch.max(param_buffer)}")
    
    trans_param = hadamard_tranformation(param_buffer.to(cudadev), group_size)
    # trans_param = original_hadamard_tranformation(param_buffer.to(cudadev))
    print(f"transformed param shape: {trans_param.shape} min: {torch.min(trans_param)} max: {torch.max(trans_param)}")
    print(f"transformed param min index: {trans_param.argmin()} max index: {trans_param.argmax()}")

    quant_param_buffer = hadamard_back_tranformation(trans_param, group_size)
    print(f"quant param type: {quant_param_buffer.dtype} shape: {quant_param_buffer.shape} min: {torch.min(quant_param_buffer)} max: {torch.max(quant_param_buffer)}")

    print(analysis_diff(param_buffer, quant_param_buffer))

if __name__ == '__main__':
    # draw_weight_and_weightdiff_histogram(1, 2000)
    # draw_weight_and_weightdiff_histogram(1, 10000)
    # draw_weight_and_weightdiff_histogram(0, 60000)
    # draw_transformed_weight(0, 60000, 2048)

    # param_buffer = param_provider(0, 60000)[0][:2048] / torch.sqrt(torch.tensor(2048))
    # print(torch.sum(param_buffer))


    # analyze_hadamard_diff(0, 60000, 1024)
    analyze_hadamard_diff(0, 60000, 1024)
    # analyze_hadamard_diff(0, 60000, 4096)
    # analyze_hadamard_diff(0, 60000, 2 ** 15)



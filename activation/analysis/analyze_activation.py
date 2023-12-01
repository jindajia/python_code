import torch
import os, sys
import matplotlib.pyplot as plt
import numpy as np
import time
sys.path.insert(1, '/ocean/projects/asc200010p/jjia1/scripts/analysis/')
from jindatools.analysis import calculate_sparsity, tensor_draw_ans_dictionary, tensor_norm, analysis_data_info

"""get tensor parallel data"""
def get_tensor_data(path: str, layer_num: int, layer_name: str, tensor_rank: int, dev: torch.device='cpu'):
    tensor_path = os.path.join(path, 'layer_{:03d}'.format(layer_num), layer_name, 'tensor_rank_{:d}.pt'.format(tensor_rank))
    tensor = torch.load(tensor_path, map_location=dev)
    return tensor

def analyze_activation():

    path = '/ocean/projects/asc200010p/jjia1/scripts/gpt_result/gpt_7_5B/010/collective/tensor_parallel/'
    iteration_num = 'iteration_00055'
    path = path+iteration_num
    layer_name = 'ParallelMLP'
    total_layer = 36
    total_rank = 4
    max_tensor_vals = []

    for layer_num in range(1,total_layer+1):
        layer_max_tensor = torch.tensor([-1])
        # max_val = torch.tensor([-1])
        for gpu_rank in range(total_rank):
            tensor = get_tensor_data(path, layer_num, layer_name, gpu_rank)
            # print('tensor parallel data layer num: {}, layer name: {}, gpu rank: {}'.format(layer_num, layer_name, gpu_rank))
            max_tensor_val = torch.max(torch.abs(tensor))
            layer_max_tensor = torch.max(max_tensor_val, layer_max_tensor)
            # torch.mean(torch.abs(tensor))
        max_tensor_vals.append(max_tensor_val.item())
    
    print(max_tensor_vals)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, total_layer+1), max_tensor_vals, marker='o')
    plt.xlabel('Layer Number')
    plt.ylabel('Maximum Tensor Value')
    plt.title('Max Tensor Values Across Layers')
    plt.grid(True)
    plt.savefig(f'/ocean/projects/asc200010p/jjia1/scripts/activations_{iteration_num}_{layer_name}_max_values.svg', format='svg')

def analyze_activation_2():

    path = '/ocean/projects/asc200010p/jjia1/scripts/gpt_result/gpt_7_5B/010/collective/tensor_parallel/'
    iteration_num = 'iteration_00055'
    path = path+iteration_num
    layer_name = 'SelfAttention'
    total_layer = 36
    total_rank = 4
    max_tensor_vals = []

    for layer_num in range(1,total_layer+1):
        layer_max_tensor = torch.tensor([-1])
        # max_val = torch.tensor([-1])
        for gpu_rank in range(total_rank):
            tensor = get_tensor_data(path, layer_num, layer_name, gpu_rank)
            # print('tensor parallel data layer num: {}, layer name: {}, gpu rank: {}'.format(layer_num, layer_name, gpu_rank))
            max_tensor_val = torch.max(torch.abs(tensor))
            layer_max_tensor = torch.max(max_tensor_val, layer_max_tensor)
            # torch.mean(torch.abs(tensor))
        max_tensor_vals.append(max_tensor_val.item())
    
    print(max_tensor_vals)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, total_layer+1), max_tensor_vals, marker='o')
    plt.xlabel('Layer Number')
    plt.ylabel('Maximum Tensor Value')
    plt.title('Max Tensor Values Across Layers')
    plt.grid(True)
    plt.savefig(f'/ocean/projects/asc200010p/jjia1/scripts/activations_gpt_7_5B_{iteration_num}_{layer_name}_max_values.svg', format='svg')

if __name__ == '__main__':
    analyze_activation()
    analyze_activation_2()
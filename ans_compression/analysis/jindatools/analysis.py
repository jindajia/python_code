import torch
import struct
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict
import math
from torch import linalg as LA

def calculate_sparsity(tensor):
    total_elements = tensor.numel()
    non_zero_elements = torch.nonzero(tensor).size(0)
    sparsity = 1.0 - (non_zero_elements / total_elements)
    return sparsity

def analysis_data_info(input_tensor: torch.tensor):
    num_nan_tensor = torch.numel(input_tensor[torch.isnan(input_tensor)])
    print("num NaN in tensor: {}, ratio: {}.".format(
            num_nan_tensor, num_nan_tensor / torch.numel(input_tensor)
        ))
    print("tensor profile: shape: {}, type: {}, sparsity: {}, min: {}, max: {}, min abs:{}, max abs:{}, mean abs:{}, norm: {}.".format(
        input_tensor.shape,
        input_tensor.dtype,
        calculate_sparsity(input_tensor),
        torch.min(input_tensor),
        torch.max(input_tensor),
        torch.min(torch.abs(input_tensor)),
        torch.max(torch.abs(input_tensor)),
        torch.mean(torch.abs(input_tensor)),
        LA.norm(input_tensor)
    ))

def analysis_diff(origin_tensor, quantized_tensor):

    diff = origin_tensor - quantized_tensor
    error_norm = tensor_norm(diff)
    origin_norm = tensor_norm(origin_tensor)
    rela_norm = error_norm / origin_norm
    # analysis_data_info(diff)
    return "abs error norm: {}, relative error norm: {}.".format(error_norm, rela_norm)

def draw_histogram_entropy(frequency, xlabel='binary strings', ylabel='Probability', title='Histogram', print_interval = 1, savepath='figure'):
    # Plot histogram for 8-bit binary strings
    plt.figure(figsize=(12, 6))
    plt.bar(frequency.keys(), frequency.values(), color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Set x-axis ticks and labels
    keys_list = list(frequency.keys())
    plt.xticks(keys_list[::print_interval], rotation=90)
    plt.tight_layout()
    plt.savefig(f"{savepath}.png")
    plt.clf()

def tensor_draw_ans_dictionary(tensor, name, savepath):
    data = tensor.cpu().detach().numpy()
    print(data.shape, data.dtype)
    frequency_8bit = defaultdict(int)
    frequency_exponent = defaultdict(int)
    it = 0
    for num in np.nditer(data):
        # Convert the float16 data to a 16-bit binary string
        bin_str = format(np.float16(num).view('H'), '016b')
        exponent = bin_str[1:9]

        # Split into two 8-bit binary strings
        first_8_bits = bin_str[:8]
        second_8_bits = bin_str[8:]

        # Update the frequency
        frequency_8bit[first_8_bits] += 1
        frequency_8bit[second_8_bits] += 1
        frequency_exponent[exponent] += 1
        it += 1
        if it % 1000000 == 0:
            print(f'finished: {it}')
    probability_8bit = {key: value / (len(data)*2) for key, value in frequency_8bit.items()}
    probability_exponent = {key: value / len(data) for key, value in frequency_exponent.items()}
    draw_histogram_entropy(probability_8bit, title =  (name + ' ANS Dictionary '), print_interval=5, savepath=savepath+'_ans')
    draw_histogram_entropy(probability_exponent, title =  (name + ' DietGPU Dictionary'), savepath=savepath+'_dietGPU')

def tensor_norm(tensor):
    return LA.norm(tensor)
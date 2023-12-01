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

def plot_activation():

    start_time = time.time()

    path = '/ocean/projects/asc200010p/jjia1/scripts/gpt_result/gpt_7_5B/013/collective/tensor_parallel/iteration_00300'
    layer_num = 1
    layer_name = 'ParallelMLP'
    gpu_rank = 0

    tensor = get_tensor_data(path, layer_num, layer_name, gpu_rank)
    print('tensor parallel data layer num: {}, layer name: {}, gpu rank: {}'.format(layer_num, layer_name, gpu_rank))
    analysis_data_info(tensor)

    std_per_channel = torch.std(tensor, keepdim=True, dim=0)
    print('std_per_channel dimension: {}'.format(std_per_channel.shape))
    analysis_data_info(std_per_channel)

    std_per_token = torch.std(tensor, keepdim=True, dim=2)
    print('std_per_token dimension: {}'.format(std_per_token.shape))
    analysis_data_info(std_per_token)

    """ Visualizing 3-D numeric data """
    tensor_list = torch.split(tensor, split_size_or_sections= 1, dim=1)
    print(len(tensor_list), tensor_list[0].shape)
    tensor = tensor_list[0]
    tensor = torch.abs(tensor) # absolute value

    downsample_factor = 4  # 降低4倍分辨率
    tensor_downsampled = tensor[:, :, ::downsample_factor]

    shape = tensor_downsampled.shape
    x = torch.arange(shape[0])
    y = torch.arange(shape[2]) * downsample_factor
    x, y = torch.meshgrid(x, y)

    z = tensor_downsampled.squeeze(1).cpu().detach().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    print('painting start')
    ax.set_zlim(0, 1)
    ax.bar3d(x.ravel(), y.ravel(), np.zeros_like(z).ravel(), 1, 1, z.ravel(), shade=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.tight_layout()
    plt.savefig(f"/ocean/projects/asc200010p/jjia1/scripts/activations_3d_gpt_7_5B_layer_{layer_num}_name_{layer_name}_iter_{300}_rank_{gpu_rank}.png", format='svg')
    plt.clf()
    end_time = time.time()
    elapsed_time = end_time - start_time

    print('painting finish time: {}'.format(elapsed_time))

def plot_learning_rate():
    total_iterations = 150000
    warmup_iterations = 2000
    max_lr = 0.0005

    iterations = np.arange(total_iterations)
    learning_rates = np.zeros(total_iterations)

    for i in range(warmup_iterations):
        learning_rates[i] = (i / warmup_iterations) * max_lr

    for i in range(warmup_iterations, total_iterations):
        learning_rates[i] = 0.5 * max_lr * (1 + np.cos(np.pi * (i - warmup_iterations) / (total_iterations - warmup_iterations)))

    plt.plot(iterations, learning_rates)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.show()
    plt.savefig(f"learning_rate.svg", format='svg')
    plt.clf()
if __name__ == '__main__':
    plot_learning_rate()
import torch
import os, sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(1, '/ocean/projects/asc200010p/jjia1/scripts/analysis/')
from jindatools.analysis import calculate_sparsity, tensor_draw_ans_dictionary, tensor_norm, analysis_data_info

dev = torch.device("cuda:0")

def print_data_into():
    path = '/ocean/projects/asc200010p/jjia1/scripts/gpt_result/gpt_7_5B/013/collective/tensor_parallel/iteration_00300'
    layer_num = 36
    tprank = 8
    layer_num_list = ['layer_{:03d}'.format(i) for i in range(1,layer_num+1)]
    layer_name_list = ['SelfAttention', 'ParallelMLP']
    tensor_rank = ['tensor_rank_{:d}.pt'.format(i) for i in range(tprank)]
    for layer_num_item in layer_num_list:
        for layer_name_item in layer_name_list:
            for tensor_rank_item in tensor_rank:
                tensor_path = os.path.join(path, layer_num_item, layer_name_item, tensor_rank_item)
                tensor = torch.load(tensor_path, map_location=torch.device('cpu'))
                print(layer_num_item, layer_name_item, tensor_rank_item)
                print(tensor.shape, tensor.dtype)

"""get tensor parallel data"""
def get_tensor_data(path: str, layer_num: int, layer_name: str, tensor_rank: int, dev: torch.device='cpu'):
    tensor_path = os.path.join(path, 'layer_{:03d}'.format(layer_num), layer_name, 'tensor_rank_{:d}.pt'.format(tensor_rank))
    tensor = torch.load(tensor_path, map_location=dev)
    return tensor

def analysis_tensor():
    path = '/ocean/projects/asc200010p/jjia1/scripts/gpt_result/gpt_7_5B/013/collective/tensor_parallel/iteration_00300'
    layer_num = 10
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
    y = torch.arange(shape[2])
    x, y = torch.meshgrid(x, y)

    z = tensor_downsampled.squeeze(1).cpu().detach().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    print('painting start')
    ax.set_zlim(0, 5)
    ax.bar3d(x.ravel(), y.ravel(), np.zeros_like(z).ravel(), 1, 1, z.ravel(), shade=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.tight_layout()
    plt.savefig(f"/ocean/projects/asc200010p/jjia1/scripts/activations_3d_001.png")
    plt.clf()
    print('painting finish')

if __name__ == '__main__':
    analysis_tensor()
    # a = torch.tensor(
    #     [[ 0.2035,  1.2959,  1.8101, -0.4644],
    #     [ 1.5027, 0.3270,  0.5905,  0.6538],
    #     [-1.5745,  1.3330, -0.5596, -0.6548]])

    # shape = a.shape
    # x = torch.arange(shape[0])
    # y = torch.arange(shape[1])
    # x, y = torch.meshgrid(x, y)

    # z = a.numpy()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.bar3d(x.ravel(), y.ravel(), np.zeros_like(z).ravel(), 1, 1, z.ravel(), shade=True)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # # plt.tight_layout()
    # plt.savefig(f"activations_3d.png")
    # plt.clf()
    # print(a.shape)
    # print(torch.std(a, keepdim=True))
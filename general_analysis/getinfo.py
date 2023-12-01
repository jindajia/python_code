import torch
import os
abs_path = '/Users/jindajia/PycharmProjects/tools_research/data_bin'
params_list = ['params_iter_001000', 'params_iter_002000']
grads_list = ['grads_iter_001000', 'grads_iter_002000']
rank_list = ['000.pt', '001.pt']
rank_list_withoutSuffix = ['000', '001']

def main():
    # draw_bitmap_heatmap_histogram()
    # cal_bit_ratio(8)
    # save_tensor()
    # data = load_bfloat16_array_from_binary('/Users/jindajia/PycharmProjects/tools_research/data_bin/params/params_iter_001000_rank_000')
    # print(data.dtype, data.shape)
    # path1 = '/Users/jindajia/PycharmProjects/tools_research/data_bin/grads/grads_iter_001000_rank_000'
    # path2 = '/Users/jindajia/PycharmProjects/tools_research/data_bin/grads/grads_iter_001000_rank_000.cuszx'
    # array1 = load_array_from_binary(path1, np.float32)
    # array2 = load_array_from_binary(path2, np.float32)
    # abs_diff = np.abs(array1 - array2)
    # indices = np.where(abs_diff > 1e-6)

    path = '/ocean/projects/asc200010p/jjia1/scripts/gpt_result/gpt_7_5B/019/collective/tensor_parallel/iteration_00050'
    layer_num = 36
    tprank = 4
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
    # tensor = torch.load(path0, map_location=torch.device('cpu'))
    # tensor1 = torch.load(path1, map_location=torch.device('cpu'))
    # for index, (model_index, dtype, pbuf, pbuf_views) \
    #         in enumerate(tensor):
    #     print(pbuf.shape)
    #     print(pbuf_views[0].shape)
    # print(type(tensor), tensor.shape, tensor.dtype)
    # print(type(tensor1), tensor1.shape, tensor1.dtype)
    # for key, value in tensor.items():
    #     print(key)
    #     for name, tp in value.items():
    #         print(name)
    #         print(tp)

if __name__ == '__main__':
    main()
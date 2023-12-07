from utils.dist_init import Utils
from utils.general_utils import print_rank_0
import utils.parallel_state as ps
import utils.collective_communication as commu
from activation.analysis.analyze_activation import get_tensor_data

import torch
import sys

sys.path.insert(1, '/ocean/projects/asc200010p/jjia1/scripts/analysis/')
from jindatools.analysis import calculate_sparsity, tensor_draw_ans_dictionary, tensor_norm, analysis_data_info, analysis_diff


def compare_all_reduce_results():
    path = '/ocean/projects/asc200010p/jjia1/scripts/gpt_result/gpt_7_5B/013/collective/tensor_parallel/'
    iteration_num = 'iteration_00055'
    path = path+iteration_num
    layer_name = 'SelfAttention'
    layer_num = 10
    tensor_parallel_rank = ps.get_tensor_model_parallel_rank()
    input_activation = get_tensor_data(path, layer_num, layer_name, tensor_parallel_rank, torch.device(device=tensor_parallel_rank))
    print('tensor parallel rank: {}, data shape {}'.format(tensor_parallel_rank, input_activation.shape))

    baseline_output = commu._reduce(input_activation.detach().clone())

    reduce_scatter_output = commu._reduce_scatter_along_first_dim(input_activation.detach().clone())
    gather_output = commu._gather_along_first_dim(reduce_scatter_output)

    print_rank_0('analysis difference between baseline and reduce_scatter + all gather version')
    print_rank_0(analysis_diff(torch.flatten(baseline_output), torch.flatten(gather_output)))

    print_rank_0('baseline shape:{}, reduce_scatter shape:{}'.format(baseline_output.shape, gather_output.shape))

    print_rank_0('print sample data from baseline [:100] ')
    print_rank_0(torch.flatten(baseline_output)[:100])

    print_rank_0('print sample data from reduce_scatter output [:100] ')
    print_rank_0(torch.flatten(gather_output)[:100])


if __name__ == '__main__':
    world_size = Utils.world_size
    Utils.initialize_model_parallel(tensor_model_parallel_size=world_size)
    compare_all_reduce_results()
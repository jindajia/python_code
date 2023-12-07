from utils.dist_init import Utils
from utils.general_utils import print_rank_0
import utils.parallel_state as ps
import utils.collective_communication as commu
from activation.analysis.analyze_activation import get_tensor_data
from utils.per_tensor_quantize import (
    quantization_int8_per_tensor,
    dequantization_int8_per_tensor,
    quantization_int8_per_tensor_baseline,
    dequantization_int8_per_tensor_baseline
    )
import torch
import sys

sys.path.insert(1, '/ocean/projects/asc200010p/jjia1/scripts/analysis/')
from jindatools.analysis import calculate_sparsity, tensor_draw_ans_dictionary, tensor_norm, analysis_data_info, analysis_diff

def reshape_to_1d(dim):
    prod_of_dims = int(torch.prod(torch.tensor(dim[:], dtype=torch.int)).item())
    new_dim = (1, prod_of_dims)
    return new_dim

def tensorboard_profile_all_reduce(input: torch.tensor):
    copy_input = input.clone()

    """start profiling"""
    prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=5, active=4, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('/ocean/projects/asc200010p/jjia1/scripts/result/log/test_accuracy/all_reduce_profile'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True)
    prof.start()
    for i in range(10):
        prof.step()
        commu._reduce(input)
        input.copy_(copy_input)
    prof.stop()
    return commu._reduce(input)

def tensorboard_profile_reduce_scatter_with_all_gather(input: torch.tensor):
    """load checkpoint tensor"""
    path = '/ocean/projects/asc200010p/jjia1/scripts/gpt_result/gpt_7_5B/013/collective/tensor_parallel/'
    iteration_num = 'iteration_00600'
    path = path+iteration_num
    layer_name = 'ParallelMLP'
    layer_num = 10
    tensor_parallel_rank = ps.get_tensor_model_parallel_rank()
    input = get_tensor_data(path, layer_num, layer_name, tensor_parallel_rank, torch.device(device=tensor_parallel_rank)).detach()

    """initizalize buffer"""
    tensor_model_parallel_size = ps.get_tensor_model_parallel_world_size()
    dim_size = list(input.size())
    assert (
        dim_size[0] % tensor_model_parallel_size == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size"
    dim_size[0] = dim_size[0] // tensor_model_parallel_size
    reduce_scatter_output = torch.empty(dim_size, dtype=input.dtype, device=torch.cuda.current_device())
    qdim_size = list(reshape_to_1d(dim_size))
    row_dim = qdim_size[-1]
    qdim_size[-1] = qdim_size[-1] + 2 * 4 # because for each row, the last eight bytes will be used to save scale and min value, (min value will be used to calculate zero point)
    qdim_size[0] = qdim_size[0] * tensor_model_parallel_size
    gather_output = torch.empty(qdim_size, dtype=torch.uint8, device=torch.cuda.current_device())
    dim_size = list(input.size())

    """start profiling"""
    prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=15, active=4, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('/ocean/projects/asc200010p/jjia1/scripts/result/log/test_accuracy/reduce_scatter_allgather'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True)
    prof.start()
    for i in range(20):
        prof.step()

        """do reduce scatter on activations"""
        commu._reduce_scatter_along_first_dim(input, reduce_scatter_output)
        """quantize tensor reduce scatter output into quantized tensor"""
        input_2d = reduce_scatter_output.view((-1, row_dim)) if row_dim > 0 else reduce_scatter_output
        quantized_tensor = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(input_2d)
        """gather quantize tensor"""
        print_rank_0('quantized_tensor shape: {}, input_2d shape: {}'.format(quantized_tensor.shape, input_2d.shape))
        commu._gather_along_first_dim(quantized_tensor, gather_output)
        """dequantize tensor"""
        assert (
            gather_output.size(0) % ps.get_tensor_model_parallel_world_size() == 0
        ), "gathered data has uneven data"
        split_tensors = torch.split(gather_output, gather_output.size(0) // ps.get_tensor_model_parallel_world_size(), dim=0)
        dequantized_tensors = [torch.ops.fbgemm.Fused8BitRowwiseQuantizedToHalf(q_tensor) for q_tensor in split_tensors]
        full_tensor = torch.cat(dequantized_tensors, dim=0).view(dim_size)
    prof.stop()
    return full_tensor

def compare_all_reduce_results_fbgemm_quantize():
    """load checkpoint tensor"""
    path = '/ocean/projects/asc200010p/jjia1/scripts/gpt_result/gpt_7_5B/013/collective/tensor_parallel/'
    iteration_num = 'iteration_00600'
    path = path+iteration_num
    layer_name = 'ParallelMLP'
    layer_num = 10
    tensor_parallel_rank = ps.get_tensor_model_parallel_rank()
    input_activation = get_tensor_data(path, layer_num, layer_name, tensor_parallel_rank, torch.device(device=tensor_parallel_rank))
    print('tensor parallel rank: {}, data shape {}'.format(tensor_parallel_rank, input_activation.shape))
    
    """primary function start"""

    """original baseline all reduce start"""
    baseline_output = tensorboard_profile_all_reduce(input_activation.detach().clone())
    """original baseline all reduce end"""

    """reduce-scatter + all gather based all reduce start"""
    candidate_tensor = tensorboard_profile_reduce_scatter_with_all_gather(input_activation.detach().clone())
    """primary function finished"""

    print_rank_0('analysis difference between baseline and reduce_scatter + all gather version')
    # print_rank_0(analysis_diff(torch.flatten(baseline_output), torch.flatten(candidate_tensor)))
    print_rank_0(analysis_diff(baseline_output, candidate_tensor))

    print_rank_0('baseline shape:{}, reduce_scatter shape:{}'.format(baseline_output.shape, candidate_tensor.shape))

    print_rank_0('print sample data from baseline [:100] ')
    print_rank_0(torch.flatten(baseline_output)[:100])

    print_rank_0('print sample data from reduce_scatter output [:100] ')
    print_rank_0(torch.flatten(candidate_tensor)[:100])

if __name__ == '__main__':
    world_size = Utils.world_size
    Utils.initialize_model_parallel(tensor_model_parallel_size=world_size)
    compare_all_reduce_results_fbgemm_quantize()
    # compare_all_reduce_results_baseline_quantize()
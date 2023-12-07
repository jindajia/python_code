import torch
import torch.nn
import torch.optim
import torch.profiler
import torch.utils.data
import torchvision.datasets
import torchvision.models
import torchvision.transforms as T

from utils.dist_init import Utils
from utils.general_utils import print_rank_0
import utils.parallel_state as ps
import utils.collective_communication as commu

from activation.analysis.analyze_activation import get_tensor_data
    
def timer_all_reduce():
    """load checkpoint tensor"""
    path = '/ocean/projects/asc200010p/jjia1/scripts/gpt_result/gpt_7_5B/013/collective/tensor_parallel/'
    iteration_num = 'iteration_00600'
    path = path+iteration_num
    layer_name = 'ParallelMLP'
    layer_num = 10
    tensor_parallel_rank = ps.get_tensor_model_parallel_rank()
    input = get_tensor_data(path, layer_num, layer_name, tensor_parallel_rank, torch.device(device=tensor_parallel_rank)).detach()
    copy_input = input.clone()
    """create cuda timer"""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    """warm up"""
    for i in range(10):
        torch.distributed.all_reduce(input)
        input.copy_(copy_input)

    """start test"""
    start.record()
    torch.distributed.all_reduce(input)
    end.record()
    torch.cuda.synchronize()
    print_rank_0('cuda timer: all-reduce: {}ms'.format(start.elapsed_time(end)))

def tensorboard_profile_all_reduce():
    """load checkpoint tensor"""
    path = '/ocean/projects/asc200010p/jjia1/scripts/gpt_result/gpt_7_5B/013/collective/tensor_parallel/'
    iteration_num = 'iteration_00600'
    path = path+iteration_num
    layer_name = 'ParallelMLP'
    layer_num = 10
    tensor_parallel_rank = ps.get_tensor_model_parallel_rank()
    input = get_tensor_data(path, layer_num, layer_name, tensor_parallel_rank, torch.device(device=tensor_parallel_rank)).detach()
    copy_input = input.clone()

    """start profiling"""
    prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=5, active=4, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('/ocean/projects/asc200010p/jjia1/scripts/result/log/test_collective/all_reduce_profile'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True)
    prof.start()
    for i in range(10):
        prof.step()
        commu._reduce(input)
        input.copy_(copy_input)
    prof.stop()

def tensorboard_profile_reduce_scatter_with_all_gather():
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
    dim_size = list(input.size())
    gather_output = torch.empty(dim_size, dtype=input.dtype, device=torch.cuda.current_device())

    """start profiling"""
    prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=5, active=4, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('/ocean/projects/asc200010p/jjia1/scripts/result/log/test_collective/reduce_scatter_allgather'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True)
    prof.start()
    for i in range(10):
        prof.step()
        commu._reduce_scatter_along_first_dim(input, reduce_scatter_output)
        commu._gather_along_first_dim(reduce_scatter_output, gather_output)
    prof.stop()

if __name__ == '__main__':
    world_size = Utils.world_size
    Utils.initialize_model_parallel(tensor_model_parallel_size=world_size)
    timer_all_reduce()
    tensorboard_profile_all_reduce()
    tensorboard_profile_reduce_scatter_with_all_gather()
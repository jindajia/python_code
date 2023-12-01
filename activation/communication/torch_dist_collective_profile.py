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

def profile_all_reduce():
    """create random tensor"""
    torch_shape = (4,2)
    torch_shape = (4, 1536, 4096)
    input = torch.empty(torch_shape, dtype=torch.float16, device=torch.cuda.current_device())
    copy_input = input.clone()

    """warm up"""
    for i in range(5):
        torch.distributed.all_reduce(input)
        input.copy_(copy_input)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as p:
        torch.distributed.all_reduce(input)

    print(p.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))
    
def timer_all_reduce():
    """create random tensor"""
    # torch_shape = (4,2)
    torch_shape = (4, 1536, 4096)
    input = torch.empty(torch_shape, dtype=torch.float16, device=torch.cuda.current_device())
    copy_input = input.clone()
    """create cida timer"""
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
    """create random tensor"""
    torch_shape = (4,2)
    torch_shape = (4, 1536, 4096)
    input = torch.empty(torch_shape, dtype=torch.float16, device=torch.cuda.current_device())
    copy_input = input.clone()

    """start profiling"""
    prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=5, active=4, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('/ocean/projects/asc200010p/jjia1/scripts/result/log/all_reduce'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True)
    prof.start()
    for i in range(10):
        prof.step()
        torch.distributed.all_reduce(input)
        input.copy_(copy_input)
    prof.stop()

def tensorboard_profile_reduce_scatter_with_all_gather():
    """create random tensor"""
    torch_shape = (4,2)
    torch_shape = (4, 1536, 4096)
    input = torch.empty(torch_shape, dtype=torch.float16, device=torch.cuda.current_device())
    copy_input = input.clone()

    """initizalize buffer"""
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
            on_trace_ready=torch.profiler.tensorboard_trace_handler('/ocean/projects/asc200010p/jjia1/scripts/result/log/all_reduce'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True)
    prof.start()
    for i in range(10):
        prof.step()
        torch.distributed.all_reduce(input)
        input.copy_(copy_input)
    prof.stop()

if __name__ == '__main__':
    Utils.initialize_distributed()
    # profile_all_reduce()
    # tensorboard_profile_all_reduce()
    timer_all_reduce()
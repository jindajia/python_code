import torch
import torch.distributed as dist

"""create random tensor"""
torch_shape = (4,2)
torch_shape = (4, 1536, 4096)
input = torch.empty(torch_shape, dtype=torch.float16, device=torch.cuda.current_device())
copy_input = input.clone()

"""warm up"""
for i in range(5):
    dist.all_reduce(input)
    input.copy_(copy_input)
with torch.profiler():
    dist.all_reduce(input)
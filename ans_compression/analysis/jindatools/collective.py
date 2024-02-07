import torch.distributed as dist
import torch
def all_reduce(tensor_list):
    # Check if the input list is not empty
    if not tensor_list:
        return None

    # Ensure all tensors have the same shape
    if not all(tensor.size() == tensor_list[0].size() for tensor in tensor_list):
        raise ValueError("Input tensors must have the same shape.")

    # Perform element-wise addition across tensors
    result_tensor = sum(tensor_list)

    return [result_tensor]

def reduce_scatter(tensor_list, dim=0):
    reduce_result = all_reduce(tensor_list)
    split_size = reduce_result.size(dim) // len(tensor_list)
    split_tensors = torch.split(reduce_result, split_size, dim=dim)

    return split_tensors

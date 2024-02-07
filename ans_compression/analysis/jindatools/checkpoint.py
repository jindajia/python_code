import torch.distributed as dist
import torch
import os
import numpy as np

def save_tensor_to_bin(tensor, file_path):
    try:
        # Convert the tensor to a NumPy array
        numpy_array = tensor.cpu().detach().numpy()
        print(numpy_array.shape, numpy_array.dtype)
        # Save the NumPy array to a binary file
        # np.save(file_path, numpy_array)
        numpy_array.tofile(file_path)

        print(f"NumPy array saved to {file_path}")
    except Exception as e:
        print(f"Error saving NumPy array to {file_path}: {str(e)}")


def load_binary_file_to_tensor(filename, data_type):
    data = np.fromfile(filename, dtype=data_type)
    return torch.from_numpy(data)
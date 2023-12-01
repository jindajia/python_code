import torch

def print_rank_0(msg):
    if torch.distributed.get_rank() == 0:
        print(msg)
import torch
import os, sys
import numpy as np
from torch import Tensor
import time
from typing import List, Optional
import re

dev = torch.device("cuda:0")

def param_provider(rank = 0, iter = 100, base_path = '') -> List[Tensor]:
    mp_rank = 'mp_rank_{:02d}'.format(rank)

    """import origin tensor"""
    iteration_num_cur = 'iter_{:07d}'.format(iter)
    params_path = os.path.join(base_path, iteration_num_cur, mp_rank, 'model_optim_rng.pt')
    param_buffer = torch.load(params_path, map_location=torch.device('cpu'))
    return param_buffer

def recursive_print(name, val, spaces=0):
    """
    Recursively print the structure of a checkpoint. This function is taken from `convert_megatron_gpt2_checkpoint.py`

    Args:
        name (str): the name of the current tensor parameter
        val (Tuple(int)): the shape of the current tensor parameter
        spaces (int): the number of spaces to print before the output for a nested structure
    """
    # Format the message.
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # Print and recurse (if needed).
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size())
    else:
        print(msg, ":", val)

def dist_optim_provider(rank = 0, iter = 100, base_path = '') -> List[Tensor]:
    mp_rank = 'mp_rank_{:02d}'.format(rank)

    """import dist optim checkpoint"""
    iteration_num_cur = 'iter_{:07d}'.format(iter)
    distrib_optim_path = os.path.join(base_path, iteration_num_cur, mp_rank, 'distrib_optim.pt')
    distrib_optim_state = torch.load(distrib_optim_path, map_location=torch.device('cpu'))
    return distrib_optim_state

def print_dp_tp_baseline():
    # dp_baseline = '/ocean/projects/asc200010p/jjia1/scripts/gpt_result/codeparrot/debug/continue_tensor_parallel_training/converted_checkpoint'
    # state_dict = param_provider(0, 22000, dp_baseline)

    dp_baseline = '/ocean/projects/asc200010p/jjia1/scripts/gpt_result/codeparrot/tp4_checkpoint'
    state_dict = param_provider(0, 2800, dp_baseline)

    # dp_baseline = '/ocean/projects/asc200010p/jjia1/scripts/gpt_result/codeparrot/baseline/checkpoint'
    state_dict_keys = state_dict.keys()

    # recursive print
    recursive_print('', state_dict['model'])

    fp32_from_fp16_params = state_dict['optimizer']['fp32_from_fp16_params']
    print('fp32_from_fp16_params length', len(fp32_from_fp16_params))
    for i, params in enumerate(fp32_from_fp16_params):
        print('fp32_from_fp16_params[{}] length: {}'.format(i, len(params)))
        for j, data in enumerate(params):
            print(data.shape)

    # optim_state = dist_optim_provider(0, 2000, dp_baseline)
    print('dp_baseline {}'.format(state_dict.keys()))
    print('dp_baseline args: {}\n checkpoint_version: {}\n model: {}\n rng_state: {}\n'.format(state_dict['args'], state_dict['checkpoint_version'], state_dict['model']['language_model'].keys(), state_dict['rng_state'][0].keys()))
    # tp4_baseline = '/N/slate/jindjia/bash_scripts/gpt/codeparrot-small/tensor-parallel/baseline/checkpoint'
    # for rank in range(4):
    #     checkpoint_dict = param_provider(rank, 2000, tp4_baseline)
    #     print('tp4_baseline {}'.format(checkpoint_dict.keys()))
    megatron_args = state_dict.get("args", None)
    if megatron_args is None:
        raise ValueError(
            "Megatron-LM checkpoint does not contain arguments. This utility only supports Megatron-LM checkpoints"
            " containing all the megatron arguments. This is because it loads all config related to model"
            " architecture, the tensor and pipeline model parallel size from the checkpoint insead of user having to"
            " manually specify all the details. Please save Megatron-LM checkpoint along with all the megatron"
            " arguments to use this utility."
        )

    # Create Transformers GPT2 config from Megatron-LM arguments
    if megatron_args is not None:
        if megatron_args.bias_gelu_fusion:
            activation_function = "gelu_fast"
        elif megatron_args.openai_gelu:
            activation_function = "gelu_new"
        else:
            activation_function = "gelu"
    else:
        # in the very early days this used to be "gelu_new"
        activation_function = "gelu_new"
    vocab_size = (
        megatron_args.padded_vocab_size
        if getattr(megatron_args, "orig_vocab_size", None) is None
        else megatron_args.orig_vocab_size
    )
    print(vocab_size)

if __name__ == '__main__':
    print_dp_tp_baseline()

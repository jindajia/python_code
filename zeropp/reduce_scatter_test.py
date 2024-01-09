import sys,os
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
import torch.distributed as dist
from torch.distributed import ProcessGroup, all_to_all_single

sys.path.append('/ocean/projects/asc200010p/jjia1/Developer/ZeROPP-Quantize/')
from zeropp.ops.op_builder.quantizer import QuantizerBuilder
sys.path.append('/ocean/projects/asc200010p/jjia1/scripts/analysis/')
from jindatools.analysis import analysis_diff
_ALL_TO_ALL_GROUP = {}


def get_normal_random_num():
    # x = torch.normal(mean=0, std=2, size=(62237952,), dtype=torch.float16).to(torch.cuda.current_device())
    # tensor_size = 2048 * 4096
    # tensor_size = 62237952
    tensor_size = 4096
    x = torch.normal(mean=0, std=2, size=(tensor_size,), dtype=torch.float16).to(torch.cuda.current_device())
    # print(x[:20])
    return x

def get_local_all_to_all_group():
    assert dist.is_initialized(), 'dist is not initialized'
    global _ALL_TO_ALL_GROUP
    device_per_node = torch.cuda.device_count()
    num_local = dist.get_world_size() // device_per_node
    if num_local == 0 and dist.get_world_size() > 0:
        assert dist.get_world_size() >= 1, 'num_gpus must >=1, cannot initialize All-To-All'
        cur_rank = []
        for i in range(dist.get_world_size()):
            cur_rank.append(i)
        _ALL_TO_ALL_GROUP['local_0'] = dist.new_group(ranks=cur_rank)
    elif num_local == 1:
        assert dist.get_world_size(
        ) == device_per_node, 'num_gpus not equal to device per node, cannot initialize All-To-All'
        _ALL_TO_ALL_GROUP['local_0'] = dist.new_group(ranks=[i for i in range(device_per_node)])
    else:
        assert dist.get_world_size() > device_per_node, 'num_nodes<2 cannot initialize All-To-All'
        for i in range(num_local):
            local_rank = [j + device_per_node * i for j in range(device_per_node)]
            _ALL_TO_ALL_GROUP[f"local_{i}"] = dist.new_group(ranks=local_rank)

        for i in range(device_per_node):
            cur_rank = []
            for j in range(num_local):
                cur_rank.append(i + j * device_per_node)
            _ALL_TO_ALL_GROUP[f"global_{i}"] = dist.new_group(ranks=cur_rank)
    return _ALL_TO_ALL_GROUP


def quantize_all_to_all_reduce(tensor, groups=None):

    global_world_size = ps.get_data_parallel_world_size()
    gpus_per_node = int(os.environ.get('LOCAL_WORLD_SIZE', 0))
    # local_world_size = gpus_per_node // ps.get_tensor_model_parallel_world_size() TODO hybird with tensor parallel
    intra_quant_group = global_world_size
    local_world_size = gpus_per_node
    num_nodes = global_world_size // local_world_size
    assert num_nodes > 1, 'number of nodes should > 1'
    inter_quant_group = intra_quant_group // local_world_size
    this_rank = ps.get_data_parallel_rank()
    intra_idx = int(this_rank / local_world_size)
    inter_idx = this_rank % local_world_size
    print(f"global world size: {global_world_size}, gpus per node: {gpus_per_node}, intra quant group: {intra_quant_group}, rank: {this_rank},  intra_idx: {intra_idx}, inter_idx: {inter_idx}")

    quantizer_module = QuantizerBuilder().load()

    intra_quant_int4, intra_q_scales = quantizer_module.swizzle_quant(tensor, intra_quant_group, 4,
                                                                        quantizer_module.Symmetric, 1, num_nodes,
                                                                        local_world_size)
    local_output = torch.empty_like(intra_quant_int4)
    scale_output = torch.empty_like(intra_q_scales)
    all_to_all_single(local_output, intra_quant_int4, group=groups[f'local_{intra_idx}'])
    all_to_all_single(scale_output, intra_q_scales, group=groups[f'local_{intra_idx}'])
    global_input_tensor, global_scales = quantizer_module.quantized_reduction(
        local_output, scale_output, intra_quant_group, inter_quant_group, 4, quantizer_module.Symmetric,
        local_world_size)
    global_output = torch.empty_like(global_input_tensor)
    global_scale_output = torch.empty_like(global_scales)
    all_to_all_single(global_output, global_input_tensor, group=groups[f'global_{inter_idx}'])
    all_to_all_single(global_scale_output, global_scales, group=groups[f'global_{inter_idx}'])
    final_output = quantizer_module.dequantize(global_output, global_scale_output, global_scale_output.numel(),
                                                4, quantizer_module.Symmetric)
    return (sum(list(final_output.chunk(num_nodes))) / num_nodes).view(-1)

def test_function():
    """load checkpoint tensor"""
    input = get_normal_random_num()
    dim_size = list(input.size())
    world_size = Utils.world_size
    assert (
        dim_size[0] % world_size == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size"

    dim_size[0] = dim_size[0] // world_size
    output = torch.empty(dim_size, dtype=input.dtype, device=torch.cuda.current_device())
    output.div_(world_size)
    torch.distributed._reduce_scatter_base(
        output, input.contiguous(), group=ps.get_data_parallel_group()
    )
    # input = get_weight_tensor()
    print_rank_0('data parallel rank: {}, input data shape: {}, mean: {}, std: {}'.format(ps.get_data_parallel_rank(), input.shape, torch.mean(input), torch.std(input)))
    print_rank_0('data parallel rank: {}, output data shape: {}, mean: {}, std: {}'.format(ps.get_data_parallel_rank(), output.shape, torch.mean(output), torch.std(output)))

    zero_qt_data = quantize_all_to_all_reduce(input, groups=get_local_all_to_all_group())

    print_rank_0('analysis difference between baseline and zero++ quantization')
    print_rank_0(analysis_diff(zero_qt_data, output))



if __name__ == '__main__':

    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    test_function()

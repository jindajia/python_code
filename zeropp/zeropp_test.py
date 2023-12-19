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

sys.path.append('/ocean/projects/asc200010p/jjia1/Developer/ZeROPP-Quantize/')
from zeropp.ops.op_builder.quantizer import CUDAQuantizer
sys.path.append('/ocean/projects/asc200010p/jjia1/scripts/analysis/')
from jindatools.analysis import analysis_diff

from utils.per_tensor_quantize import (
    quantization_int8_per_tensor,
    dequantization_int8_per_tensor,
    quantization_int8_per_tensor_baseline,
    dequantization_int8_per_tensor_baseline
    )

def zeropp_quantization_test(input):
    """start quantization"""
    quantizer_module = CUDAQuantizer()
    quantized_param, scales = quantizer_module.quantize(input)
    print_rank_0('quantized shape: {}, scales shape: {}'.format(quantized_param.shape, scales.shape))

    """dequantization"""
    """allocate buffer"""
    buffer_size = input.shape
    buffer_type = input.dtype
    param_buffer = torch.empty(
        buffer_size,
        dtype=buffer_type,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )
    """allocate buffer finished"""
    param_buffer.data = quantizer_module.dequantize(quantized_param, scales)
    return param_buffer

def fbgemm_quantization_test(input):
    """start quantization"""
    row_dim = 7616
    input_2d = input.view((-1, row_dim)) if row_dim > 0 else input
    quantized_tensor = torch.ops.fbgemm.HalfToFused8BitRowwiseQuantized(input_2d)
    print_rank_0('input_2d shape: {}, quantized_tensor shape: {}'.format(input_2d.shape, quantized_tensor.shape))

    """dequantization"""
    """allocate buffer"""
    buffer_size = input.shape
    buffer_type = input.dtype
    param_buffer = torch.empty(
        buffer_size,
        dtype=buffer_type,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )
    """allocate buffer finished"""
    param_buffer.data = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToHalf(quantized_tensor).view(buffer_size)
    return param_buffer

def get_weight_tensor():
    path = '/ocean/projects/asc200010p/jjia1/scripts/gpt_result/codeparrot/debug/collect_params/collective/params/'
    iteration_num = 'iteration_52000'
    dp_rank = ps.get_data_parallel_rank()
    param_path = os.path.join(path, iteration_num, 'params_rank_{:01d}.pt'.format(dp_rank))
    param =  torch.load(param_path, map_location=torch.device(device=torch.cuda.current_device()))
    return param

def get_normal_random_num():
    x = torch.normal(mean=0, std=5, size=(62237952,), dtype=torch.float16).to(torch.cuda.current_device())
    # print(x[:20])
    return x

def test_function():
    """load checkpoint tensor"""
    input = get_normal_random_num()
    # input = get_weight_tensor()
    print_rank_0('data parallel rank: {}, data shape: {}, mean: {}, std: {}'.format(ps.get_data_parallel_rank(), input.shape, torch.mean(input), torch.std(input)))

    zero_qt_data = zeropp_quantization_test(input)

    print_rank_0('analysis difference between baseline and zero++ quantization')
    print_rank_0(analysis_diff(zero_qt_data, input))

    fbgemm_token_qt_data = fbgemm_quantization_test(input)
    # print(fbgemm_token_qt_data[:20])
    print_rank_0('analysis difference between baseline and fbgemm quantization')
    print_rank_0(analysis_diff(fbgemm_token_qt_data, input))


if __name__ == '__main__':
    world_size = Utils.world_size
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    test_function()

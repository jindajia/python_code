import sys
import torch
from torch import nn
import os
import numpy as np
sys.path.insert(1, '/ocean/projects/asc200010p/jjia1/scripts/analysis/')
from jindatools.collective import reduce_scatter,all_reduce
from jindatools.checkpoint import save_tensor_to_bin, load_binary_file_to_tensor
from compression.compress import get_any_comp_timings, get_float_comp_timings, calc_comp_ratio, compress_data, decompress_data, max_any_compressed_output_size
import onnx
from jindatools.binary_convert import float2bit, bit2float
from jindatools.analysis import calculate_sparsity, tensor_draw_ans_dictionary, tensor_norm
from jindatools.quantization import convert_fp8e5m2_to_fp32, convert_fp8e4m3_to_fp32
from torch.quantization.observer import PerChannelMinMaxObserver, MinMaxObserver
def demo_compress():
    dev = torch.device("cuda:0")

    # Non-batched Float Compression 
    # ts = data_provider()
    ts = grads_data_provider()
    dt = ts[0].dtype
    shape = ts[0].shape
    c, dc, total_size, comp_size = get_float_comp_timings(ts)
    ratio = comp_size / total_size
    c_bw = (total_size / 1e9) / (c * 1e-3)
    dc_bw = (total_size / 1e9) / (dc * 1e-3)

    print("Float codec non-batched perf {} {}".format(dt, shape))
    print(
        "comp   time {:.3f} ms B/W {:.1f} GB/s, compression {} -> {} bytes ({:.4f}x) ".format(
            c, c_bw, total_size, comp_size, ratio
        )
    )
    print("decomp time {:.3f} ms B/W {:.1f} GB/s".format(dc, dc_bw))


    # Non-batched ANS Compression
    ts = grads_data_provider()
    dt = ts[0].dtype
    shape = ts[0].shape

    c, dc, total_size, comp_size = get_any_comp_timings(ts)
    ratio = comp_size / total_size
    c_bw = (total_size / 1e9) / (c * 1e-3)
    dc_bw = (total_size / 1e9) / (dc * 1e-3)

    print("Raw ANS byte-wise non-batched perf {} {}".format(dt, shape))
    print(
        "comp   time {:.3f} ms B/W {:.1f} GB/s, compression {} -> {} bytes ({:.4f}x) ".format(
            c, c_bw, total_size, comp_size, ratio
        )
    )
    print("decomp time {:.3f} ms B/W {:.1f} GB/s".format(dc, dc_bw))

def demo_compress2():
    dev = torch.device("cuda:0")

    # Non-batched Float Compression 
    # ts = data_provider()
    # ts = grads_data_provider()
    ts = bin_data_provider()
    for tensor in ts:
        dt = tensor.dtype
        shape =  tensor.shape
        c, dc, total_size, comp_size = get_float_comp_timings([tensor])
        ratio = comp_size / total_size
        c_bw = (total_size / 1e9) / (c * 1e-3)
        dc_bw = (total_size / 1e9) / (dc * 1e-3)

        print("Float codec non-batched perf {} {}".format(dt, shape))
        print(
            "comp   time {:.3f} ms B/W {:.1f} GB/s, compression {} -> {} bytes ({:.4f}x) ".format(
                c, c_bw, total_size, comp_size, ratio
            )
        )
        print("decomp time {:.3f} ms B/W {:.1f} GB/s".format(dc, dc_bw))


    # Non-batched ANS Compression
    ts = bin_data_provider()
    for tensor in ts:
        dt = tensor.dtype
        shape = tensor.shape

        c, dc, total_size, comp_size = get_any_comp_timings([tensor])
        ratio = comp_size / total_size
        c_bw = (total_size / 1e9) / (c * 1e-3)
        dc_bw = (total_size / 1e9) / (dc * 1e-3)

        print("Raw ANS byte-wise non-batched perf {} {}".format(dt, shape))
        print(
            "comp   time {:.3f} ms B/W {:.1f} GB/s, compression {} -> {} bytes ({:.4f}x) ".format(
                c, c_bw, total_size, comp_size, ratio
            )
        )
        print("decomp time {:.3f} ms B/W {:.1f} GB/s".format(dc, dc_bw))

def data_provider():
    dev = torch.device("cuda:0")
    # tprank = 4
    # path = '/ocean/projects/asc200010p/jjia1/scripts/gpt_result/gpt_7_5B/019/collective/tensor_parallel/iteration_00050/layer_010/SelfAttention/'
    # tprank = 4
    # path = '/ocean/projects/asc200010p/jjia1/scripts/gpt_result/gpt_7_5B/010/collective/tensor_parallel/iteration_00055/layer_007/ParallelMLP'
    # tprank = 8
    # path = '/ocean/projects/asc200010p/jjia1/scripts/gpt_result/gpt_7_5B/009/collective/tensor_parallel/iteration_00055/layer_007/ParallelMLP'
    tprank = 8
    path = '/ocean/projects/asc200010p/jjia1/scripts/gpt_result/gpt_7_5B/013/collective/tensor_parallel/iteration_00600/layer_010/ParallelMLP'

    tensor_rank = ['tensor_rank_{:d}.pt'.format(i) for i in range(tprank)]
    ts = []
    for tensor_rank_item in tensor_rank:
        tensor_path = os.path.join(path, tensor_rank_item)
        tensor = torch.load(tensor_path, map_location=torch.device(device=dev))
        ts.append(tensor)

    print(ts[0].shape)
    # ts = reduce_scatter(ts,0)
    ts = all_reduce(ts)

    return [ts[0]]
    
def convert_tensor_to_bin(tensor, save_path):
    # save to fp8
    # fp8_ts = tensor.detach().clone().cpu()
    # fp8_ts.apply_(onnx.helper.float32_to_float8e4m3)
    # fp8_ts = fp8_ts.to(torch.uint8)
    # save_tensor_to_bin(fp8_ts, save_path)
    # print(fp8_ts.shape, fp8_ts.dtype, fp8_ts)

    # save original format
    save_tensor_to_bin(tensor, save_path)
    print(tensor.shape, tensor.dtype, tensor)

def grads_data_provider():
    dev = torch.device("cuda:0")
    # tensor_path_2 = '/ocean/projects/asc200010p/jjia1/scripts/dataset/bytedance/151m_moe/151m_moe_iter_005000/grads/grads_iter_005000/001.pt'
    # tensor = torch.load(tensor_path, map_location=torch.device(device=dev))[0].to(torch.float16)[:240000000] # 898,788,864
    # tensor_itr_69000 = torch.load(tensor_path_2, map_location=torch.device(device=dev))[0].to(torch.float16)
    
    # chunk_size = 200000000
    # num_chunks = tensor.shape[0] // chunk_size + 1
    # print(num_chunks)
    # chunks = torch.chunk(tensor, num_chunks, dim=0)
    # print(len(chunks), chunks[0].shape)
    # return chunks

    # tensor_path_1 = '/ocean/projects/asc200010p/jjia1/scripts/dataset/bytedance/151m_moe/151m_moe_iter_699000/grads/grads_iter_699000/000.pt'
    # tensor_itr_5000 = torch.load(tensor_path_1, map_location=torch.device(device=dev))[0].to(torch.float16)
    # print(tensor_itr_5000.shape, tensor_itr_5000.dtype)
    # return [tensor_itr_5000]

    tensor_path_2 = '/ocean/projects/asc200010p/jjia1/scripts/dataset/bytedance/151m_moe/151m_moe_iter_005000/grads/grads_iter_005000/001.pt'
    tensor_itr_69000 = torch.load(tensor_path_2, map_location=torch.device(device=dev))[0].to(torch.float16)
    print(tensor_itr_69000.shape, tensor_itr_69000.dtype)
    return [tensor_itr_69000]

def bin_data_provider():
    dev = torch.device("cuda:0")

    """import origin tensor"""
    bin_path = '/ocean/projects/asc200010p/jjia1/scripts/python_script/gpt_7_5B_600iter_10layer_mlp_allreduce_fp16.bin'
    tensor_fp16 = tensor = load_binary_file_to_tensor(bin_path, np.float16).to(device=dev)

    """import fp8 tensor, convert to fp32"""
    # bin_path = '/ocean/projects/asc200010p/jjia1/scripts/python_script/gpt_7_5B_600iter_10layer_mlp_allreduce_fp8_e4m3.bin'
    # tensor = load_binary_file_to_tensor(bin_path, np.uint8).to(device=dev)
    # tensor = convert_fp8e4m3_to_fp32(tensor=tensor)

    # tensor = onnx.numpy_helper.float8e5m2_to_float32(tensor)
    return [tensor]
    # chunk_size = 200000000
    # num_chunks = tensor.shape[0] // chunk_size + 1
    # print(num_chunks)
    # chunks = torch.chunk(tensor, num_chunks, dim=0)
    # print(len(chunks), chunks[0].shape)
    # return chunks

def cal_error_norm():
    tensor_bin = bin_data_provider()[0].view(1536,4,4096).to(device='cuda:0')
    tensor_origin = data_provider()[0].to(device='cuda:0')
    diff = tensor_bin - tensor_origin

    error_norm = tensor_norm(diff)
    origin_norm = tensor_norm(tensor_origin)
    rela_norm = error_norm / origin_norm
    print(error_norm, origin_norm, rela_norm)

def analysis_diff(origin_tensor, quantized_tensor):

    diff = origin_tensor - quantized_tensor
    error_norm = tensor_norm(diff)
    origin_norm = tensor_norm(origin_tensor)
    rela_norm = error_norm / origin_norm

    print("abs error norm: {}, relative error norm: {}.".format(error_norm, rela_norm))

def analysis_data_info(input_tensor):
    num_nan_tensor = torch.numel(input_tensor[torch.isnan(input_tensor)])
    print("num NaN in tensor: {}, ratio: {}.".format(
            num_nan_tensor, num_nan_tensor / torch.numel(input_tensor)
        ))
    print("tensor profile: shape: {}, type: {}, sparsity: {}, min: {}, max: {}, min abs:{}, max abs:{}.".format(
        input_tensor.shape,
        input_tensor.dtype,
        calculate_sparsity(input_tensor),
        torch.min(input_tensor),
        torch.max(input_tensor),
        torch.min(torch.abs(input_tensor)),
        torch.max(torch.abs(input_tensor)),
    ))

def torch_uint8_quantization():
    tensor = data_provider()[0].to(torch.float32)
    # tensor = torch.tensor([1, 20.9, 1000], dtype=torch.float32, device=torch.device("cuda:0"))
    analysis_data_info(tensor)

    """
        Affine quantization to uint8
        https://towardsdatascience.com/tensor-quantization-the-untold-story-d798c30e7646
        https://huggingface.co/docs/optimum/concept_guides/quantization
    """
    x_max = max(torch.max(tensor).item(), 0)
    x_min = min(torch.min(tensor).item(), 0)
    q_max = 255
    q_min = 0
    scale = (x_max - x_min) / (q_max - q_min)
    zero_point =  - round(x_min / scale)
    print("scale: {}, zero point: {}.".format(scale, zero_point))

    quantized_tensor = torch.quantize_per_tensor(tensor, scale, zero_point, torch.quint8)
    # print("quantized tensor type: {}, data: {} int data: {} dequantize data: {}.".format(type(quantized_tensor), quantized_tensor, quantized_tensor.int_repr(), quantized_tensor.dequantize()))
    analysis_diff(tensor, quantized_tensor.dequantize())

    """ANS COMPRESSION START"""

    dev = torch.device("cuda:0")
    ts_in = [quantized_tensor]
    """ANS compress allocate buffer"""
    tempMem = torch.empty([384 * 1024 * 1024], dtype=torch.uint8, device=dev)
    rows, cols = max_any_compressed_output_size(ts_in)
    comp = torch.empty([rows, cols], dtype=torch.uint8, device=dev)
    sizes = torch.zeros([len(ts_in)], dtype=torch.int, device=dev)
    """ANS compress data"""
    compress_data(False, ts_in, False, tempMem, comp, sizes)

    total_size, comp_size, _ = calc_comp_ratio(ts_in, sizes)
    ratio = comp_size / total_size
    print(
        "compression {} -> {} bytes ({:.4f}x) ".format(total_size, comp_size, ratio)
    )

    """ANS decompress allocate buffer"""
    out_ts = []
    comp_ts = [*comp]
    quant_info = []
    for t in ts_in:
        # print(t.size(), t.numel(), t) #Attention, zero point and scale info will not be store
        out_ts.append(torch.empty(t.size(), dtype=t.dtype, device=t.device))
        quant_info.append((t.q_scale(), t.q_zero_point()))
    out_status = torch.empty([len(ts_in)], dtype=torch.uint8, device=dev)
    out_sizes = torch.empty([len(ts_in)], dtype=torch.int32, device=dev)
    """ANS decompress data"""
    decompress_data(False, comp_ts, out_ts, False, tempMem, out_status, out_sizes)

    print("out_ts type: {}".format(type(out_ts[0])))
    # analysis_diff(tensor, out_ts[0].dequantize())



def torch_int8_quantization():
    """
        Affine quantization to int8
        https://towardsdatascience.com/tensor-quantization-the-untold-story-d798c30e7646
        https://huggingface.co/docs/optimum/concept_guides/quantization
        https://pytorch.org/blog/quantization-in-practice/#in-pytorch
    """
    dev = torch.device("cuda:0")
    tensor = data_provider()[0].to(torch.float32)

    for qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
        obs = MinMaxObserver(qscheme=qscheme, dtype=torch.qint8)
        obs(tensor)
        scale, zero_point = obs.calculate_qparams()
        print(f"Qscheme: {qscheme} | scale:{scale} zero_point:{zero_point}")
        quantized_tensor = torch.quantize_per_tensor(tensor, scale.item(), zero_point.item(), torch.qint8)
        analysis_diff(tensor, quantized_tensor.dequantize())

        """ANS COMPRESSION START"""

        ts_in = [quantized_tensor]
        """ANS compress allocate buffer"""
        tempMem = torch.empty([384 * 1024 * 1024], dtype=torch.uint8, device=dev)
        rows, cols = max_any_compressed_output_size(ts_in)
        comp = torch.empty([rows, cols], dtype=torch.uint8, device=dev)
        sizes = torch.zeros([len(ts_in)], dtype=torch.int, device=dev)
        """ANS compress data"""
        compress_data(False, ts_in, False, tempMem, comp, sizes)

        total_size, comp_size, _ = calc_comp_ratio(ts_in, sizes)
        ratio = comp_size / total_size
        print(
            "compression {} -> {} bytes ({:.4f}x) ".format(total_size, comp_size, ratio)
        )
def test_ans(ts_in):
    dev = torch.device("cuda:0")

    tempMem = torch.empty([384 * 1024 * 1024], dtype=torch.uint8, device=dev)

    """ANS compress allocate buffer"""
    rows, cols = max_any_compressed_output_size(ts_in)
    comp = torch.empty([rows, cols], dtype=torch.uint8, device=dev)
    sizes = torch.zeros([len(ts_in)], dtype=torch.int, device=dev)
    """ANS compress data"""
    compress_data(False, ts_in, False, tempMem, comp, sizes)

    """ANS decompress allocate buffer"""
    out_ts = []
    comp_ts = [*comp]
    for t in ts_in:
        out_ts.append(torch.empty(t.size(), dtype=t.dtype, device=t.device))
    out_status = torch.empty([len(ts_in)], dtype=torch.uint8, device=dev)
    out_sizes = torch.empty([len(ts_in)], dtype=torch.int32, device=dev)
    """ANS decompress data"""
    decompress_data(False, comp_ts, out_ts, False, tempMem, out_status, out_sizes)

    analysis_data_info(out_ts[0])

if __name__ == "__main__":
    # torch_uint8_quantization()
    torch_int8_quantization()
    # data_provider()
    # grads_data_provider()
    # bin_data_provider()
    # demo_compress2()
    # analysis_data()
    # print(onnx.numpy_helper.float8e5m2_to_float32(176))
    # tensor = np.array([int('0b10101111', 2)], dtype=np.int16)

    # analysis_data()

    # tensor = np.array([int('10110000', 2)], dtype=np.uint8)
    # print(tensor, tensor.dtype)
    # print(onnx.numpy_helper.float8e5m2_to_float32(tensor))
    # # convert_tensor_to_bin(grads_data_provider()[0], 'moe_699000iter_grads_fp32.bin')

    # print(onnx.helper.float32_to_float8e5m2(-0.1212))
    # print(onnx.numpy_helper.float8e5m2_to_float32(176))
    # print(bin(176))


    # print(e5m2_to_float(int('0b10101111', 2)))
    # print(float_to_e5m2(-0.1212))
    # print(e5m2_to_float(175))

    # tensor1 = torch.tensor([-0.1212, 2.2, 3.3], dtype=torch.float32)
    # fp8_ts = tensor1.clone()
    # fp8_ts.apply_(float_to_e5m2)
    # fp8_ts = fp8_ts.to(torch.uint8)
    # print(fp8_ts.shape, fp8_ts.dtype, fp8_ts)

    # filename = '/ocean/projects/asc200010p/jjia1/scripts/python_script/float16.bin'
    # save_tensor_to_bin(sample_tensor, filename)
    # print(load_binary_file_to_tensor(filename, data_type=np.float16))
from typing import List, Optional
from torch import Tensor
import torch
import os

torch.ops.load_library("/ocean/projects/asc200010p/jjia1/TOOLS/dietgpu/build/lib/libdietgpu.so")
dev = torch.device("cuda:0")

"""original compress function in c"""

def compress_data(compress_as_float: bool, ts_in: List[Tensor], check_sum: bool, temp_mem: Optional[Tensor], out_compressed:  Optional[Tensor], out_compressed_bytes: Optional[Tensor]) -> (Tensor, Tensor, int):
    return torch.ops.dietgpu.compress_data(
            compress_as_float, ts_in, check_sum, temp_mem, out_compressed, out_compressed_bytes
        )

def decompress_data(compress_as_float: bool, ts_in: List[Tensor], ts_out: List[Tensor], check_sum: bool, temp_mem: Optional[Tensor], out_status: Optional[Tensor], out_decompressed_words: Optional[Tensor]) -> (int):
    return torch.ops.dietgpu.decompress_data(
            compress_as_float, ts_in, ts_out, check_sum, temp_mem, out_status, out_decompressed_words
        )

def max_float_compressed_output_size(ts_in: List[Tensor]) -> (int, int):
    return torch.ops.dietgpu.max_float_compressed_output_size(ts_in)

def max_any_compressed_output_size(ts_in: List[Tensor]) -> (int, int):
    return torch.ops.dietgpu.max_any_compressed_output_size(ts_in)

def calc_comp_ratio(input_ts, out_sizes):
    total_input_size = 0
    total_comp_size = 0

    for t, s in zip(input_ts, out_sizes):
        total_input_size += t.numel() * t.element_size()
        total_comp_size += s

    return total_input_size, total_comp_size, total_comp_size / total_input_size


def get_float_comp_timings(ts, num_runs=3):
    tempMem = torch.empty([384 * 1024 * 1024], dtype=torch.uint8, device=dev)

    comp_time = 0
    decomp_time = 0
    total_size = 0
    comp_size = 0

    # ignore first run timings
    for i in range(1 + num_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        rows, cols = torch.ops.dietgpu.max_float_compressed_output_size(ts)

        comp = torch.empty([rows, cols], dtype=torch.uint8, device=dev)
        sizes = torch.zeros([len(ts)], dtype=torch.int, device=dev)

        start.record()
        comp, sizes, memUsed = torch.ops.dietgpu.compress_data(
            True, ts, True,tempMem, comp, sizes
        )
        end.record()
        comp_size = 0

        torch.cuda.synchronize()
        if i > 0:
            comp_time += start.elapsed_time(end)

        total_size, comp_size, _ = calc_comp_ratio(ts, sizes)

        out_ts = []
        for t in ts:
            out_ts.append(torch.empty(t.size(), dtype=t.dtype, device=t.device))

        # this takes a while
        comp_ts = [*comp]

        out_status = torch.empty([len(ts)], dtype=torch.uint8, device=dev)
        out_sizes = torch.empty([len(ts)], dtype=torch.int32, device=dev)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        torch.ops.dietgpu.decompress_data(
            True, comp_ts, out_ts, True,tempMem, out_status, out_sizes
        )
        end.record()

        torch.cuda.synchronize()
        if i > 0:
            decomp_time += start.elapsed_time(end)

        # validate
        for a, b in zip(ts, out_ts):
            assert torch.equal(a, b)

    comp_time /= num_runs
    decomp_time /= num_runs

    return comp_time, decomp_time, total_size, comp_size

def get_any_comp_timings(ts, num_runs=3):
    tempMem = torch.empty([384 * 1024 * 1024], dtype=torch.uint8, device=dev)

    comp_time = 0
    decomp_time = 0
    total_size = 0
    comp_size = 0

    # ignore first run timings
    for i in range(1 + num_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        rows, cols = torch.ops.dietgpu.max_any_compressed_output_size(ts)

        comp = torch.empty([rows, cols], dtype=torch.uint8, device=dev)
        sizes = torch.zeros([len(ts)], dtype=torch.int, device=dev)

        start.record()
        comp, sizes, memUsed = torch.ops.dietgpu.compress_data(
            False, ts, True,tempMem, comp, sizes
        )
        end.record()

        comp_size = 0

        torch.cuda.synchronize()
        comp_time = start.elapsed_time(end)

        total_size, comp_size, _ = calc_comp_ratio(ts, sizes)

        out_ts = []
        for t in ts:
            out_ts.append(torch.empty(t.size(), dtype=t.dtype, device=t.device))

        # this takes a while
        comp_ts = [*comp]

        out_status = torch.empty([len(ts)], dtype=torch.uint8, device=dev)
        out_sizes = torch.empty([len(ts)], dtype=torch.int32, device=dev)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        torch.ops.dietgpu.decompress_data(
            False, comp_ts, out_ts, True,tempMem, out_status, out_sizes
        )
        end.record()

        torch.cuda.synchronize()
        decomp_time = start.elapsed_time(end)

        for a, b in zip(ts, out_ts):
            assert torch.equal(a, b)

    return comp_time, decomp_time, total_size, comp_size


def demo_compress():
    path = '/ocean/projects/asc200010p/jjia1/scripts/gpt_result/gpt_7_5B/019/collective/tensor_parallel/iteration_00050/layer_010/SelfAttention/tensor_rank_0.pt'
    tensor_path = os.path.join(path)
    tensor = torch.load(tensor_path, map_location=torch.device(device=dev))
    dt = tensor.dtype
    # Non-batched Float Compression 
    ts = []
    ts.append(tensor)

    c, dc, total_size, comp_size = get_float_comp_timings(ts)
    ratio = comp_size / total_size
    c_bw = (total_size / 1e9) / (c * 1e-3)
    dc_bw = (total_size / 1e9) / (dc * 1e-3)

    print("Float codec non-batched perf {} {}".format(dt, tensor.shape))
    print(
        "comp   time {:.3f} ms B/W {:.1f} GB/s, compression {} -> {} bytes ({:.4f}x) ".format(
            c, c_bw, total_size, comp_size, ratio
        )
    )
    print("decomp time {:.3f} ms B/W {:.1f} GB/s".format(dc, dc_bw))

    ts = []
    ts.append(tensor)

    c, dc, total_size, comp_size = get_any_comp_timings(ts)
    ratio = comp_size / total_size
    c_bw = (total_size / 1e9) / (c * 1e-3)
    dc_bw = (total_size / 1e9) / (dc * 1e-3)

    print("Raw ANS byte-wise non-batched perf {} {}".format(dt, tensor.shape))
    print(
        "comp   time {:.3f} ms B/W {:.1f} GB/s, compression {} -> {} bytes ({:.4f}x) ".format(
            c, c_bw, total_size, comp_size, ratio
        )
    )
    print("decomp time {:.3f} ms B/W {:.1f} GB/s".format(dc, dc_bw))


if __name__ == '__main__':
    demo_compress()
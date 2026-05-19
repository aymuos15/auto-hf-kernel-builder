# === model__1_inference_0.0/output_code.py ===
# AOT ID: ['1_inference']

import torch
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.async_compile import AsyncCompile

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_localssk23/tmp1urctweq/nn/cnnrzbpsms5pfq3q3rspb6mpgbxf6vphpf7e6wruxxlpebdx6yeu.py
# Topologically Sorted Source Nodes: [mul, pow_1, mul_1, add, mul_2, tanh, add_1, mul_3], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   mul => mul
#   mul_1 => mul_1
#   mul_2 => mul_2
#   mul_3 => mul_3
#   pow_1 => pow_1
#   tanh => tanh
# Graph fragment:
#   %arg0_1 : Tensor "f32[8192, 8192][8192, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %mul : Tensor "f32[8192, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, 0.5), kwargs = {})
#   %pow_1 : Tensor "f32[8192, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg0_1, 3.0), kwargs = {})
#   %mul_1 : Tensor "f32[8192, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_1, 0.044715), kwargs = {})
#   %add : Tensor "f32[8192, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %mul_1), kwargs = {})
#   %mul_2 : Tensor "f32[8192, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 0.7978845608028654), kwargs = {})
#   %tanh : Tensor "f32[8192, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_2,), kwargs = {})
#   %add_1 : Tensor "f32[8192, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh, 1.0), kwargs = {})
#   %mul_3 : Tensor "f32[8192, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %add_1), kwargs = {})
#   return %mul_3
triton_poi_fused_add_mul_pow_tanh_0 = async_compile.triton(
    "triton_poi_fused_add_mul_pow_tanh_0",
    """
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=16, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_tanh_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '130560DF8C676AFCBC44717C6A9B3C6A2EC6174C11ECC01A816D2F75FFBF9BD0', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 805306368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_pow_tanh_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = tmp0 * tmp0
    tmp4 = tmp3 * tmp0
    tmp5 = 0.044715
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 + tmp6
    tmp8 = 0.7978845608028654
    tmp9 = tmp7 * tmp8
    tmp10 = libdevice.tanh(tmp9)
    tmp11 = 1.0
    tmp12 = tmp10 + tmp11
    tmp13 = tmp2 * tmp12
    tl.store(out_ptr0 + (x0), tmp13, None)
""",
    device_str="cuda",
)


def partition_0(args):
    (arg0_1,) = args
    args.clear()
    assert_size_stride(arg0_1, (8192, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8192, 8192), (8192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul, pow_1, mul_1, add, mul_2, tanh, add_1, mul_3], Original ATen: [aten.mul, aten.pow, aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_tanh_0.run(arg0_1, buf0, 67108864, stream=stream0)
        del arg0_1
    return (buf0,)


async_compile.wait(globals())
del async_compile


class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        (arg0_1,) = args
        args.clear()
        partition0_args = [
            arg0_1,
        ]
        del arg0_1
        (buf0,) = self.partitions[0](partition0_args)
        del partition0_args
        return (buf0,)


runner = Runner(
    partitions=[
        partition_0,
    ]
)
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance

    arg0_1 = rand_strided((8192, 8192), (8192, 1), device="cuda:0", dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main

    compiled_module_main("None", benchmark_compiled_module)

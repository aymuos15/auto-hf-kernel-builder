import torch
import triton
import triton.language as tl


@triton.jit
def new_gelu_kernel(in_ptr, out_ptr, n_elements, XBLOCK: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * XBLOCK
    offsets = block_start + tl.arange(0, XBLOCK)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)

    # GELU formula: 0.5 * x * (1.0 + tanh(sqrt(2.0 / pi) * (x + 0.044715 * x^3)))
    # 0.5 * (1.0 + tanh(y)) = sigmoid(2.0 * y)
    # sqrt(2.0 / pi) * 2.0 = 1.5957691216057308

    c1 = 1.5957691216057308
    c2 = 0.044715

    # inner = c1 * (x + c2 * x * x * x)
    # Using tl.sigmoid for efficiency
    out = x * tl.sigmoid(c1 * (x + c2 * x * x * x))

    tl.store(out_ptr + offsets, out, mask=mask)


def kernel(x):
    n_elements = x.numel()
    out = torch.empty_like(x)
    XBLOCK = 2048
    grid = (triton.cdiv(n_elements, XBLOCK),)
    new_gelu_kernel[grid](x, out, n_elements, XBLOCK=XBLOCK, num_warps=8)
    return out

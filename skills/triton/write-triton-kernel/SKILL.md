# Skill: Write a Triton Kernel

## Purpose

Guide the agent through authoring a correct, launch-safe `@triton.jit` kernel for an elementwise or simple-reduction op and the Python wrapper that launches it. Covers grid sizing, one-program-per-tile assignment, strided pointer math, boundary masking, fp32 accumulation, and verifying a real Triton launch actually happened (not torch passthrough).

---

## Use this when

- You must replace a PyTorch op (or fuse a small chain) with one Triton kernel that beats `torch.compile` on the target GPU.
- The op is elementwise, a broadcasted binary op, or a single-axis reduction. Op-specific skills exist for softmax/gemm/layernorm/attention — prefer those if they match.
- The harness requires a genuine `@triton.jit` launch on output-scale data (a no-op fig-leaf launch is rejected).

## Do not use this when

- A fused `torch.compile` graph already saturates memory bandwidth and there is no headroom — a hand kernel will not beat it; report that instead of shipping a wrapper that just calls the reference.
- The op is a dense GEMM / attention / normalization with a known better skill — use the op-specific skill.
- The work does not fit the SPMD tile model (data-dependent control flow, dynamic shapes per element).

---

## Inputs the agent should gather first

1. **Exact op** — the precise math, from `reference.py`. Reproduce it, do not approximate beyond the harness tolerance.
2. **Input/output shapes, dtypes, strides** — and whether tensors are contiguous or views (strides may not equal shape).
3. **Reduction axis** (if any) and whether its length is a compile-time constant.
4. **Tolerance** — `correctness.rtol/atol` from `config.json`.
5. **Hardware** — from `config.json` `env`: GPU name, `compute_capability`, VRAM. This drives BLOCK_SIZE / `num_warps` / `num_stages` and whether to `triton.autotune`.
6. **The baseline dtype** — the frozen baseline runs the reference in its native dtype (fp32 here). Do not silently switch to bf16/fp16 to "win"; the harness rejects `precision_cheat`.

---

## Required reasoning process

1. **Map the op to a tile decomposition.** Flatten the iteration space; assign one program per `BLOCK_SIZE` tile: `pid = tl.program_id(0)`; `offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`.
2. **Compute the grid in the wrapper** from runtime numel: `grid = (triton.cdiv(n_elements, BLOCK_SIZE),)`.
3. **Build pointers from strides, not assumed contiguity.** Pass each tensor's relevant strides as kernel args; index with them. Never assume `row_stride == D`.
4. **Mask every load/store.** `mask = offs < n_elements`; `tl.load(p + offs, mask=mask, other=<identity>)` where the `other` value is the reduction identity (`0.0` for sum, `-inf` for max).
5. **Accumulate in fp32.** Cast fp16/bf16 inputs with `.to(tl.float32)` before arithmetic/reduction; cast back to the output dtype only at `tl.store`.
6. **Write the wrapper.** Allocate the output with `torch.empty_like`/correct shape+dtype, launch `kernel[grid](...)`, return the tensor. The wrapper is the `kernel(*inputs)` the harness calls.
7. **Self-check before returning.** Confirm: output dtype == reference output dtype; the launch operates on output-scale tensors (not a throwaway); two runs on the same input are bitwise-equal; output changes when input changes.

---

## Kernel design rules

- `BLOCK_SIZE` is a `tl.constexpr` power of 2. Start 1024 for 1-D elementwise; tune via the optimize skill, do not guess randomly.
- All reductions in fp32. Out-of-bounds loads use the reduction identity as `other`, never an arbitrary 0 that corrupts a max.
- Pass strides as arguments for every multi-dim tensor; support non-contiguous inputs.
- The Python wrapper must do the real launch every call — do not import or run the reference model to produce the result (torch passthrough is a cheat the harness detects).
- Output dtype and shape must match the reference exactly; never downcast precision to gain speed.
- No host-side Python loop over tiles; the grid expresses parallelism.
- Transcendental API varies by Triton version: `tl.exp`/`tl.rsqrt`/`tl.sqrt` are stable, but `tl.tanh` does not exist on all versions (it lives in `libdevice`). Prefer expressing `tanh` via `tl.exp` (`tanh(z) = 1 - 2/(exp(2z)+1)`) so the kernel is version-proof.

---

## One-shot example (fused scaled-GELU, elementwise)

```python
import torch
import triton
import triton.language as tl


@triton.jit
def _gelu_kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    c = 0.7978845608028654  # sqrt(2/pi)
    z = c * (x + 0.044715 * x * x * x)
    t = 1.0 - 2.0 / (tl.exp(2.0 * z) + 1.0)  # tanh via exp: tl.tanh is not in tl.* on all Triton versions
    y = 0.5 * x * (1.0 + t)
    tl.store(y_ptr + offs, y.to(tl.float32), mask=mask)


def kernel(x):
    x = x.contiguous()
    y = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, 1024),)
    _gelu_kernel[grid](x, y, n, BLOCK_SIZE=1024)
    return y.view_as(x)
```

Note the real launch on `n`-element tensors, fp32 compute, masked tail, output dtype preserved — exactly what the harness's anti-cheat and correctness gates check.

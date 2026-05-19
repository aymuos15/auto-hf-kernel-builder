# Skill: Write a Triton Softmax Kernel

## Purpose

Implement a numerically stable, fused row-wise softmax in Triton: one program per row, `tl.max`/`tl.sum` reductions in fp32, tail masking for rows wider than `BLOCK_SIZE`, and the additive-mask variant used inside attention.

---

## Use this when

- You need a single-pass fused softmax (no separate max-reduction or division pass) to beat `torch.compile`.
- You are fusing a scale (`1/sqrt(d)`) or an additive mask into the softmax.
- The softmax axis is reasonably wide (≥ 256); below that the torch path usually wins.

## Do not use this when

- Plain unmasked softmax on a standard tensor with no fusion need — `torch.compile`/cuDNN already saturates it.
- Softmax is across rows (column-wise) — transpose the problem or this strategy fails.
- Streaming/online softmax over arbitrarily long sequences — needs a multi-block reduction beyond one-program-per-row.

---

## Inputs the agent should gather first

1. Shape and which axis is the softmax axis; flatten leading dims to a row count.
2. Row length `D` — compile-time constant? power of 2?
3. Input dtype (accumulate fp32 regardless) and required output dtype.
4. Mask: additive (add before max) or none.
5. Tolerance and the hardware (`config.json` `env`) for BLOCK_SIZE / autotune.

---

## Required reasoning process

1. `row = tl.program_id(0)`; grid `(n_rows,)`. Compute the row base pointer from a passed `row_stride` (not assumed `== D`).
2. `cols = tl.arange(0, BLOCK_SIZE)`; `mask = cols < D`; load with `other=-inf` so OOB never affects the max.
3. Cast to fp32. Add the additive attention mask here, **before** the max.
4. `m = tl.max(x, 0)`; `e = tl.exp(x - m)`; `s = tl.sum(e, 0)`; `y = e / s` — all fp32.
5. Store `y.to(out_dtype)` with `mask`.
6. If `D > BLOCK_SIZE`: online softmax (running max/sum) — two passes, no inter-program comms.
7. Verify: deterministic, output depends on input, dtype preserved, both harness input sets pass.

---

## Design rules

- `BLOCK_SIZE = next_power_of_2(D)` for the single-block case; `tl.constexpr`; never autotune it below `D`.
- OOB `other=-inf` (not 0) — a 0 corrupts the max.
- Additive mask added to logits before `tl.max`; never zero-after-exp (changes the sum → wrong probabilities).
- fp32 for max, exp, sum, divide; cast to output dtype only at store.
- `row_stride` is a kernel argument.

---

## One-shot example

```python
import torch, triton
import triton.language as tl


@triton.jit
def _softmax(x_ptr, y_ptr, row_stride, D, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < D
    x = tl.load(x_ptr + r * row_stride + cols, mask=m, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, 0)
    e = tl.exp(x)
    y = e / tl.sum(e, 0)
    tl.store(y_ptr + r * row_stride + cols, y, mask=m)


def kernel(x):
    x = x.contiguous()
    n, D = x.view(-1, x.shape[-1]).shape
    y = torch.empty_like(x)
    _softmax[(n,)](x, y, D, D, BLOCK=triton.next_power_of_2(D))
    return y
```

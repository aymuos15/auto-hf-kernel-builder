# Skill: Write a Triton LayerNorm Kernel

## Purpose

Implement a fused LayerNorm (or RMSNorm) in Triton: one program per row, fp32 mean/variance reduction, affine `weight`/`bias`, tail masking, correct epsilon-inside-sqrt placement, output dtype preserved.

---

## Use this when

- You need fused norm + affine (and optionally a fused residual add or following op) to beat `torch.compile`.
- The normalized dim is moderate-to-large so a per-row kernel pays off.

## Do not use this when

- Plain `nn.LayerNorm` with no fusion need — the torch path is already fused well.
- BatchNorm/GroupNorm with cross-row statistics — different reduction structure; not this skill.

---

## Inputs the agent should gather first

1. Normalized shape and which trailing dim(s); flatten leading dims to rows.
2. LayerNorm vs RMSNorm (RMSNorm has no mean subtraction, no bias).
3. `eps`, presence of `weight`/`bias`, input/output dtype.
4. Any fused residual/op; tolerance; hardware (`config.json` `env`).

---

## Required reasoning process

1. `row = tl.program_id(0)`; grid `(n_rows,)`; row base from passed `row_stride`.
2. Load row in fp32 with `mask = cols < D`, `other=0.0`.
3. LayerNorm: `mu = sum(x)/D`; `var = sum((x-mu)^2)/D`; `xhat = (x-mu) * rsqrt(var + eps)`. RMSNorm: `xhat = x * rsqrt(mean(x^2) + eps)` (no `mu`).
4. Apply affine: `y = xhat * w + b` (load `w`,`b` with the same `mask`).
5. Store `y.to(out_dtype)` with `mask`.
6. Verify determinism, input-dependence, dtype, both harness input sets.

---

## Design rules

- `eps` goes **inside** the sqrt: `rsqrt(var + eps)`, never `1/(sqrt(var)+eps)`.
- All of mean/var/normalize in fp32; cast to output dtype only at store.
- `BLOCK_SIZE = next_power_of_2(D)`, `tl.constexpr`; mask the tail; `other=0.0` is safe here (0 contributes nothing to a sum/mean — but then divide by the true `D`, not by the masked count).
- Divide by the real `D`, not by `BLOCK_SIZE`.
- `row_stride`, `weight`, `bias` are kernel arguments; support non-contiguous input.
- RMSNorm: no mean subtraction and no bias — do not copy the LayerNorm path blindly.

---

## One-shot example (LayerNorm + affine)

```python
@triton.jit
def _ln(x_ptr, w_ptr, b_ptr, y_ptr, row_stride, D, eps, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < D
    x = tl.load(x_ptr + r * row_stride + cols, mask=m, other=0.0).to(tl.float32)
    mu = tl.sum(x, 0) / D
    xc = tl.where(m, x - mu, 0.0)
    var = tl.sum(xc * xc, 0) / D
    xhat = xc * tl.rsqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + r * row_stride + cols, xhat * w + b, mask=m)
```

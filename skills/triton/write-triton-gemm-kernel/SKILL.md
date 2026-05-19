# Skill: Write a Triton GEMM Kernel

## Purpose

Implement a tiled `C = A @ B` (with optional bias/activation epilogue) in Triton: 2-D program grid over output tiles, a K-loop with `tl.dot`, fp32 accumulation, masking for shapes not divisible by the block, and program-id swizzle for L2 reuse.

---

## Use this when

- The op is (batched) matmul / linear layer and you need to fuse the epilogue (bias, GELU, scale) to beat `torch.compile`.
- Shapes are large enough that a tiled kernel can exploit data reuse.

## Do not use this when

- cuBLAS via `torch.matmul`/`torch.compile` already hits near-peak and there is no epilogue to fuse — a hand GEMM rarely beats cuBLAS alone; only fusion wins.
- Tiny/skinny shapes where launch overhead dominates — fuse into the surrounding kernel instead.
- The real cost is attention/softmax — use those skills.

---

## Inputs the agent should gather first

1. M, N, K and batch; dtypes of A, B, C; required output dtype.
2. Strides/layout of A and B (row/col major, transposed?), contiguity.
3. Epilogue to fuse (bias, activation, scale) and its exact math.
4. Tolerance; hardware (`config.json` `env`) for block M/N/K, `num_warps`, `num_stages`.

---

## Required reasoning process

1. Grid over output tiles: `pid_m, pid_n` from a single `program_id` with **swizzle** (group along M for L2 reuse).
2. `offs_m`, `offs_n`, `offs_k = tl.arange(...)`; build A/B block pointers from real strides.
3. `acc = tl.zeros((BM, BN), tl.float32)`. Loop `k` in steps of `BK`: load A/B tiles with `mask` on the K tail, `acc += tl.dot(a, b)`.
4. Apply the fused epilogue on `acc` in fp32.
5. Mask the `(M, N)` store for non-multiple shapes.
6. Store `acc.to(out_dtype)`.
7. Verify both harness input sets, determinism, dtype.

---

## Design rules

- Accumulator is fp32 regardless of input dtype; `tl.dot` accumulates fp32.
- `tl.dot` defaults to **TF32** for fp32 inputs on Ampere+ (≈1e-2 error) — that alone can blow the harness `atol/rtol` (default 2e-2) and cause `numeric_mismatch`. For an fp32 reference pass `tl.dot(a, b, allow_tf32=False)` (or `input_precision="ieee"`); only keep TF32 if it stays within tolerance and you need the speed.
- Block M/N/K are `tl.constexpr` powers of 2; tune via the optimize skill, do not random-guess.
- Mask the K-loop tail and the M/N store — silent wrong results otherwise on non-multiple shapes.
- Use real strides; support transposed B without a separate transpose kernel.
- Program-id swizzle (group size ~8) materially improves L2 hit rate — include it.
- Do not call `torch.matmul` to produce the result (passthrough is a detected cheat); `tl.dot` must do the math.

---

## One-shot example (core, bias epilogue)

```python
@triton.jit
def _gemm(A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    gm = tl.cdiv(M, BM)
    pm, pn = pid % gm, pid // gm
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        a = tl.load(A + rm[:, None] * sam + (k + rk)[None, :] * sak,
                    mask=(rm[:, None] < M) & ((k + rk)[None, :] < K), other=0.0)
        b = tl.load(B + (k + rk)[:, None] * sbk + rn[None, :] * sbn,
                    mask=((k + rk)[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b, allow_tf32=False)  # TF32 default ≈1e-2 err → can fail harness tol
    cm = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C + rm[:, None] * scm + rn[None, :] * scn, acc, mask=cm)
```

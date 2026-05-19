# Skill: Write a Triton Attention Kernel

## Purpose

Implement fused scaled-dot-product attention in Triton with the FlashAttention online-softmax tiling: stream K/V tiles, keep a running max and running sum, and accumulate the output without ever materializing the `N x N` score matrix. This is the only structure that beats `torch.compile` on attention at non-trivial sequence length.

---

## Use this when

- The reference is (multi-head) scaled-dot-product attention and you must beat `torch.compile`.
- Sequence length is large enough that materializing scores is the bottleneck (memory-bound).

## Do not use this when

- A short-sequence attention where `torch.compile`/SDPA fused kernel already wins — confirm headroom first.
- The op is not attention (a bare softmax → use the softmax skill; a bare matmul → the gemm skill).
- You would materialize the full score matrix — that defeats the purpose; if you cannot do online softmax, stop and reconsider.

---

## Inputs the agent should gather first

1. Q/K/V shapes `(B, H, N, d)`, dtypes, required output dtype.
2. Scale (usually `1/sqrt(d)`), causal vs additive mask vs none.
3. Whether `d` (head dim) is a compile-time constant and a power of 2.
4. Tolerance; hardware (`config.json` `env`) — block N along K/V, `num_warps`, `num_stages`.

---

## Required reasoning process

1. Grid: one program per `(batch*head, query-block)`. `m_i = -inf` (running max), `l_i = 0` (running sum), `acc = zeros(BM, d)` in fp32.
2. Load the Q tile once. Loop over K/V tiles of size `BN`:
   - `s = (q @ k^T) * scale` (fp32); apply causal/additive mask on `s` **before** the max.
   - `m_new = max(m_i, rowmax(s))`; `p = exp(s - m_new)`; `alpha = exp(m_i - m_new)`.
   - `l_i = l_i * alpha + rowsum(p)`; `acc = acc * alpha[:, None] + p @ v`.
   - `m_i = m_new`.
3. After the loop: `acc = acc / l_i[:, None]`; store `acc.to(out_dtype)`.
4. Verify: deterministic, depends on inputs, dtype preserved, both harness input sets, causal correctness at the diagonal block.

---

## Design rules

- Never materialize the full `(N, N)` scores — online softmax is mandatory; that is the whole point.
- Running max/sum and `acc` are fp32; cast to output dtype only at the final store.
- Mask (causal/additive) is applied to `s` before the running-max update, never after exp.
- The rescale (`alpha = exp(m_i - m_new)`) must be applied to **both** `l_i` and `acc` every tile — forgetting `acc` is the classic Flash bug.
- `q @ k^T` and `p @ v` use `tl.dot` (fp32 accumulate); do not call `torch` SDPA to produce the result (detected passthrough cheat).
- Causal: skip K/V tiles entirely past the query block; mask only the diagonal tile.

---

## One-shot example (inner online-softmax update — the load-bearing part)

```python
# inside the K/V loop, all fp32:
s = tl.dot(q, tl.trans(k)) * scale          # (BM, BN)
s = tl.where(mask, s, -float("inf"))         # causal/additive BEFORE max
m_new = tl.maximum(m_i, tl.max(s, 1))
p = tl.exp(s - m_new[:, None])
alpha = tl.exp(m_i - m_new)
l_i = l_i * alpha + tl.sum(p, 1)
acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
m_i = m_new
# after loop: acc = acc / l_i[:, None]
```

# Skill: Optimize Triton Block Parameters

## Purpose

Take a *correct* Triton kernel and close the gap to (and past) the `torch.compile` bar by systematically searching the launch-parameter space — `BLOCK_SIZE`(s), `num_warps`, `num_stages` — and applying `@triton.autotune` safely. Triton-to-Triton speedup is overwhelmingly in this space, not in rewriting the math.

---

## Use this when

- The kernel is correct (passes the harness correctness + determinism gates) but `slower_than_compile`.
- You have a working single-config kernel and want a principled tuning pass rather than random guessing.

## Do not use this when

- The kernel is still incorrect — fix correctness first (use the debug skill). Tuning an incorrect kernel wastes the retry budget.
- The bottleneck is algorithmic (e.g. materializing an O(n^2) attention matrix). No block tuning rescues a bad algorithm; switch to the op-specific skill.
- The op is memory-bound and already at bandwidth — confirm with the roofline reasoning below before spending retries.

---

## Inputs the agent should gather first

1. **Current kernel + its measured time** vs the `compile_ms` bar in `bench.json`/`res.json`.
2. **Hardware** from `config.json` `env`: GPU, `compute_capability`, VRAM. On Ampere (sm_8.x, e.g. A1000) sensible `num_warps` ∈ {2,4,8}, `num_stages` ∈ {2,3,4}.
3. **Dominant tensor shapes** — to bound BLOCK_SIZE (must be ≥ the reduction length when a row must fit one block; must stay a power of 2).
4. **Whether the op is memory- or compute-bound** — decides whether larger blocks help (memory-bound: maximize bytes in flight; compute-bound: maximize occupancy/ILP).

---

## Required reasoning process

1. **Roofline first.** Estimate bytes moved and FLOPs. If `bytes / peak_bw ≈ measured_time`, the kernel is already bandwidth-bound — tuning yields little; say so instead of burning retries.
2. **Pick the search axes that matter for this op.** Elementwise/reduction: `BLOCK_SIZE`, `num_warps`. GEMM-like: block M/N/K, `num_warps`, `num_stages`. Do not autotune a block dim that must exactly cover a reduction axis — a too-small choice silently produces wrong results.
3. **Constrain the grid.** Powers of 2 only; BLOCK_SIZE ≤ what registers/shared memory allow (oversized blocks spill and slow down or fail to launch).
4. **Apply `@triton.autotune`** with a small explicit `configs` list and correct `key=[...]` (the shape args that should trigger re-tuning). Keep the list short — each config is benchmarked.
5. **Re-verify correctness after tuning.** Autotune can expose a config where masking/accumulation assumptions break; the harness's two-input-set + determinism check must still pass.
6. **Measure against the bar, not in isolation.** Only a config that beats `compile_ms` by `perf.min_speedup_vs_compile` counts.

---

## Design rules

- Never `triton.autotune` a constexpr that must equal/cover a reduction length (the classic "autotune picked BLOCK_D < D → wrong, fast garbage" trap). Compute it as `triton.next_power_of_2(D)` instead.
- `key=` must list exactly the dims whose change should re-trigger tuning; omitting them caches a stale config for a new shape.
- Prefer a curated 4–8 config grid over a broad sweep — autotune benchmarks every entry on first call.
- Keep accumulation fp32 across all configs; a faster config that changes accumulation dtype is a correctness regression.
- Re-run the correctness check after every parameter change; speed without the harness PASS is worthless.

---

## One-shot example (autotuned elementwise)

```python
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["n"],
)
@triton.jit
def _k(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=m, other=0.0), mask=m)
```

`BLOCK_SIZE` here is safe to autotune because it does not have to cover a reduction axis; `key=["n"]` re-tunes per problem size.

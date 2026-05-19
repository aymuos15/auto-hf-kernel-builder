# Skill: Debug Triton Kernel Correctness

## Purpose

Systematically diagnose a Triton kernel that fails the harness `numeric_mismatch` or `nondeterministic` verdict. Classify the failure into one of a small set of bug categories, each with a targeted fix, instead of guessing. This is the most common harness failure class.

---

## Use this when

- `bench.json` reports `error_class: numeric_mismatch` (with a `max_abs_diff`) or `nondeterministic`.
- The kernel runs (no exception) but the output disagrees with the reference, or differs between two identical runs.

## Do not use this when

- `error_class: kernel_exception` — that is a crash, not a numeric bug; read the traceback first.
- `error_class: slower_than_compile` — correctness is fine; use the optimize skill.
- `error_class: triton_figleaf` / `precision_cheat` / `reference_import` — not a correctness bug to debug; see "Anti-cheat verdicts" below.

---

## Inputs the agent should gather first

1. **`max_abs_diff`** from `bench.json` — its magnitude is diagnostic (see classification).
2. **Reference math** from `reference.py` — the exact expression to match.
3. **Shapes/strides/dtypes** and the tolerance (`rtol/atol`).
4. **Whether failure is on input set A, B, or both** — both → systematic; one → boundary/shape-dependent.
5. **Determinism**: does the same input give different output across runs? → race / uninitialized accumulator.

---

## Bug classification

| Signature | Likely cause | Fix |
|---|---|---|
| `max_abs_diff` huge / NaN / inf | unmasked OOB load corrupting a reduction; wrong `other` value | mask every load/store; use reduction identity (`-inf` for max, `0` for sum) as `other` |
| Small bias that grows with reduction length | accumulating in fp16/bf16 | cast to fp32 before any arithmetic; accumulate fp32; cast back only at store |
| Correct for power-of-2 sizes, wrong otherwise | BLOCK_SIZE doesn't cover the axis / tail not masked | BLOCK_SIZE = `next_power_of_2(D)`; mask the tail; never autotune a reduction-covering block |
| Off-by-stride / transposed-looking output | assumed contiguity; used shape as stride | pass real strides as kernel args; index with them |
| Differs run-to-run (nondeterministic) | races on a shared accumulator, or uninitialized memory read | one program per output region, or atomics; `torch.empty` outputs must be fully written |
| `kernel_exception` "tensor output of CUDAGraphs overwritten", or nondeterministic only when a persistent compiled module is reused | bench calls `kernel` many times; a `torch.compile`/CUDA-graph buffer is reused so a later call overwrites an earlier call's still-referenced output | `.clone()` the returned tensor; call `torch.compiler.cudagraph_mark_step_begin()` before each invocation; do not return a view into a persistent/cudagraph buffer |
| Wrong only on masked-softmax / attention rows | additive mask applied after exp instead of before max | add mask to logits before `tl.max` |
| Tiny diff just above tolerance | genuine fp ordering vs torch | confirm fp32 accumulation; if still over, the algorithm differs — re-derive, do not loosen tolerance |

---

## Required reasoning process

1. Read `max_abs_diff` and which input set failed; map to the table above before editing.
2. Form one hypothesis; make the smallest change that tests it.
3. Re-derive the indexing on paper for the boundary tile (last partial block) — most bugs live there.
4. Verify fp32 accumulation end to end.
5. Re-run the harness `bench` mentally against both input sets + determinism; a fix that only helps set A is not a fix.
6. Never "fix" by widening `rtol/atol` or by falling back to the reference op — both are rejected by the harness or are cheats.

---

## Design rules

- The tail tile is guilty until proven innocent: check `mask = offs < n` and the `other=` identity first.
- fp32 accumulation is non-negotiable for any reduction.
- Strides are arguments, never assumptions.
- A nondeterministic kernel is always a real bug (race / uninitialized read), never acceptable noise.
- Do not change tolerance or pass through the reference to make the gate green.

---

## Anti-cheat verdicts (`no_triton` / `triton_figleaf` / `precision_cheat` / `reference_import`)

These are not correctness bugs and there is **no superficial way to satisfy them** — they all mean the same thing: *your kernel is not actually computing the result in Triton*. Every attempt to "get past the check" (a no-op launch, a tiny throwaway tensor, autocast, importing or dynamically importing the reference model) is detected and, even when not, cannot beat `torch.compile` because it is just the reference again. The only resolution is to do the real work:

- `no_triton` — no `@triton.jit` ran. Implement the computation as a real Triton kernel.
- `triton_figleaf` — a Triton kernel ran but only on a tiny/throwaway tensor while torch did the real work. The Triton kernel must produce the actual output, on output-scale data.
- `precision_cheat` — the result was produced under reduced-precision autocast vs an fp32 baseline. Compute in the reference dtype; a real speedup comes from the kernel, not from lowering precision.
- `reference_import` — `kernel.py` imported or otherwise pulled in the reference model/library. Reproduce the math yourself in Triton (derive weights deterministically from the seed); do not import, `importlib`, or otherwise reach the reference.

If you are here, switch strategy: read `skills/triton/write-triton-kernel/SKILL.md` (or the op-specific skill) and implement the actual algorithm. Hybrid (some torch + real Triton) is allowed as a stepping stone, but the Triton path must do real, output-scale work and the whole thing must still beat the bar honestly.

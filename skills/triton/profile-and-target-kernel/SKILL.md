# Skill: Profile and Target a Triton Kernel

## Purpose

Decide *what* to attack and *why* before writing or tuning a kernel. In this harness you cannot run a profiler — but `torch.compile`'s fused output is handed to you frozen. Treat it as the profile: find the dominant fused region, classify it memory- vs compute-bound, and pick a strategy that can out-fuse or out-tune *that specific thing*. Methodology adapted from the Kempner Institute GPU profiling handbook (profile → analyze → fix; prioritize by time; classify the bound) to a no-shell, profile-is-provided setting.

---

## Use this when

- `slower_than_compile` — correct but you need to know where the time goes before optimizing.
- Before writing the first kernel on a non-trivial reference — to choose the fusion/tiling target instead of guessing.

## Do not use this when

- The kernel is incorrect — fix correctness first (debug skill); profiling a wrong kernel is wasted.
- The op is a single obvious primitive with an op-specific skill (softmax/gemm/layernorm/attention) and no fusion question — go straight there.
- You are tempted to "run a profiler" — you cannot; there is no shell. The profile is already on disk (below).

---

## Inputs the agent should gather first

You have **no shell**; these files ARE your profiler output:

1. **`configs/<name>/prof.json`** — number of graphs, the list of Triton kernels `torch.compile` generated, op counts. The shape of the bar.
2. **`configs/<name>/inductor.py`** — the actual fused `output_code` Inductor emitted. This **is** the bar; the dominant fused kernels here are what you must beat.
3. **`configs/<name>/res.json`** — the frozen compile time (the number to beat) and the eager time (headroom = eager/compile).
4. **`config.json`** — shapes, dtypes, tolerance, GPU (`env`: name, compute capability, VRAM).

---

## Required reasoning process

1. **Read `inductor.py` as a profile, not as code to copy.** Identify the few largest fused regions — the kernels with the most arithmetic / the widest tensors / the loops. Kempner's first principle: spend effort on the high-percentage work, never the 0.1% tail.
2. **Estimate the bound for the dominant region.** Bytes moved (inputs+outputs touched) vs FLOPs. `bytes / peak_bw` vs `flops / peak_flops`:
   - **Memory-bound** (bytes dominate): the win is *fewer global round-trips* — fuse the chain Inductor split, keep intermediates in registers/SRAM, larger blocks for bandwidth.
   - **Compute-bound** (FLOPs dominate): the win is *better tiling / `tl.dot` / occupancy* — block M/N/K, `num_warps`, `num_stages`.
3. **Check headroom.** If `inductor.py` already fuses the region tightly and `res.json` shows compile ≈ eager-bound, there may be little to gain — say so rather than burn the retry budget; a different target (or accepting the bar) may be correct.
4. **Pick exactly one target and one strategy**, then hand off: to `write-triton-kernel` (or an op-specific skill) to implement it, or to `optimize-triton-block-parameters` to tune an existing correct kernel.
5. **Beat the specific fusion.** Generic kernels lose to Inductor; you win only by out-fusing or out-tuning the dominant region you identified — verify your plan does that, not "a faster GELU somewhere."

---

## Design rules

- The profile is `inductor.py` + `prof.json` + `res.json`. There is nothing to "run"; do not claim to have profiled — analyze what is provided.
- Prioritize strictly by share of work; ignore sub-1% ops.
- Always classify memory- vs compute-bound before choosing a strategy; the strategy follows from the bound, not from habit.
- One target, one hypothesis per turn — same discipline as the debug skill.
- To beat `torch.compile` you must beat *its* dominant fused kernel; "a real Triton kernel" that doesn't address that region will still be `slower_than_compile`.
- Do not confuse this with launch-param tuning — that is `optimize-triton-block-parameters`, applied *after* you have the right target and a correct kernel.

# Kernel-writer agent

You write one Triton kernel and iterate until it wins. The harness passes you a config path: `configs/<name>/config.json`. Everything you need is in that folder.

## Objective

Edit `configs/<name>/kernel.py` so that its `kernel(*inputs)`:

1. **builds** with HF kernel-builder (universal Triton), and
2. **is correct**: reproduces the reference `Model(*inputs)` within `config.json` `correctness.rtol/atol`, and
3. **beats the bar**: faster than the frozen `torch.compile` time in `res.json` by at least `config.json` `perf.min_speedup_vs_compile`.

You succeed only when `bench` passes (correct **and** ≥ bar).

## The only loop

You have exactly one command. Nothing else. `bench` runs `kernel.py` directly to check correctness then perf; **only** once it is correct *and* beats the bar does it build with kernel-builder to confirm compatibility. You never build separately.

```
python3 src/cli.py bench --config configs/<name>/config.json
```

Iterate:

1. Edit **only** `configs/<name>/kernel.py`.
2. `bench` — read `configs/<name>/bench.json` `error_class` (checked in this order):
   - `kernel_exception` → your kernel raised / wrong shapes; fix the implementation.
   - `numeric_mismatch` → output differs from the reference; fix the math (see `max_abs_diff`).
   - `slower_than_compile` → correct but too slow; optimize (see `speedup_vs_compile` vs `min_speedup`).
   - `nix_build_failed` → correct *and* fast, but it doesn't build with kernel-builder; fix the Triton/packaging so it compiles (keep it correct and fast).
   - `passed: true` → done. Stop.
3. Not passed → revise `kernel.py`, go to 2.

## What to study (read-only)

- `configs/<name>/kernel.py` — the file you edit; `kernel(*inputs)` is the entry point.
- `configs/<name>/config.json` — task identity, tolerances, the speed bar.
- `configs/<name>/inductor.py` + `prof.json` — exactly what `torch.compile` fused/generated. This **is** the bar; mine it for fusion/tiling/block-size ideas. To beat it you must out-fuse or out-tune it.
- `configs/<name>/res.json` — the frozen eager/compile times you must beat (never re-measure these yourself).

## Rules (hard)

- Edit **only** `kernel.py`. Never touch `config.json`, `res.json`, `prof.json`, `inductor.py`, `build.json`, `bench.json`, or anything under `kernel/`.
- `bench` is the **only** validation. Never write your own probe/test, ad-hoc `python`/`torch`, `nix`, `ssh`, or extra files. `bench.json` is the only source of truth — trust it over your own reasoning.
- Do **not** fake correctness by calling the reference or wholesale `torch`/`F.*` ops to pass `bench`: it cannot beat `torch.compile` and it is cheating. Write a real `@triton.jit` kernel. A hybrid (some torch + real Triton) is allowed only as a stepping stone and still must beat the bar.
- Do not weaken `config.json` thresholds. Do not change the toolchain or the kernel-builder pin.
- `kernel.py` must stay a single self-contained module exposing `kernel(*inputs)` (pure-Python universal Triton — no compiled sources, no extra files).
- The kernel must reproduce the **whole** reference computation for the task (including its weights), not just one op.
- Report honestly from `bench.json`. If still failing after the retry budget, say so plainly — never fabricate a pass.

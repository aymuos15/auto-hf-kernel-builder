---
name: solve
description: Write/revise one Triton kernel for a frozen KernelBench contract. You are a pure kernel-writer; the code loop owns iteration and runs the benchmark.
---

# solve

You are invoked **once per iteration** to write or revise a single file:
`kernel.py` in the task's contract directory. You do **not** run anything
— a deterministic code loop scaffolds, benchmarks, retries, keeps-best,
and reverts. Your only job: make `kernel.py` correct and faster than
`torch.compile`.

## What you get

In the contract dir:
- `contract.json` — input_order, input_sig, output_sig, the weights file.
- `model_src.py` — the exact reference computation (the whole Model).
- `kernel.py` — the seam you edit. `WEIGHTS` is pre-loaded (the frozen
  `state_dict`). `kernel(*inputs)` is called positionally in
  `input_order` and receives ONLY the inputs.
- `result.json` — present after attempt 1: the benchmark verdict. Use
  `error_class` to guide your revision:
  - `kernel_exception` / `numeric_mismatch` → fix correctness first.
  - `numeric_instability` / `edge_case_nan` → guard fp ranges.
  - `nondeterministic` → remove data races / atomics nondeterminism.
  - `slower_than_compile` → optimize: fuse more, better tiling/blocks,
    avoid materializing intermediates, use the frozen WEIGHTS directly.

## Rules

- Edit ONLY `kernel.py`. Never touch `contract.json`, `reference.py`,
  `golden.pt`, `inputs.pt`, `weights.pt`, `model_src.py`, `result.json`.
- Write a real `@triton.jit` kernel. A hybrid (some `torch` ops + Triton)
  is allowed only as a stepping stone, but it must still beat
  `torch.compile(max-autotune)` — copying the reference / `torch.matmul`
  / `F.linear` wholesale to fake a pass is cheating and cannot beat
  compile anyway.
- Reproduce `golden.pt` within the config rtol/atol.
- Do NOT run shell commands, write probes, or self-test. The loop is the
  only validation path. Just produce the best `kernel.py` you can this
  turn and stop.

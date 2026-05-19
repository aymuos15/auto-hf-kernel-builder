---
name: hf-model-to-kernelbench
description: "Turn any Hugging Face transformers/diffusers model (or one of its submodules) into a self-contained KernelBench-style task: a standalone module exposing Model / get_inputs / get_init_inputs that a kernel harness can benchmark. Use when asked to wrap, port, or adapt an HF model as a custom KernelBench task, reference Model, or solve/bench target; or to check whether a model fits a GPU and how to scope it (whole model vs reduced config vs single submodule). Harness-agnostic: produces the module + a preflight check, and names — but does not hardcode — the per-harness ingestion step."
disable-model-invocation: false
user-invocable: true
allowed-tools: "Read, Grep, Glob, Bash, Write, Edit"
argument-hint: "model id or class (e.g. facebook/sam2, Sam2Model, a submodule path) [+ target GPU]"
---

# HF model → KernelBench task

Produce a standalone Python module that wraps an HF model as a KernelBench-style task, verify it is benchmark-legal, then hand off to whatever harness ingests tasks. The hard, portable part is the module + the invariants; harness wiring is one named step at the end.

## Procedure

1. **Scope the target — does it fit?** Read `references/vram-fit.md`. Estimate `params*4 bytes` (fp32) and remember the harness keeps ~3 copies resident (reference + `torch.compile` baseline + candidate kernel) plus activations. Decide: whole model · reduced-config model · one representative submodule. Never download weights — construct from a config so it is offline and deterministic.

2. **Author the module.** Follow `references/contract.md` exactly. It must define `Model(nn.Module)`, `get_inputs()`, `get_init_inputs()`; init from config (random weights are fine — the harness seeds them); `forward` returns **one tensor**; output must depend on the inputs and be deterministic in `eval()`/`no_grad`. Argument order in `forward` must match the positional order of `get_inputs()`.

3. **Preflight (always run before handing off).**
   ```bash
   python3 scripts/preflight.py path/to/module.py --vram-budget-gb <GPU_GB>
   ```
   It execs the module, builds under a fixed seed, and asserts: single tensor returned, deterministic across two identical runs, output changes for a different input seed, and peak VRAM ≤ budget. Non-zero exit = not benchmark-legal; fix before continuing.

4. **Wire into the harness (the one per-harness step — do not hardcode it).** The module is the deliverable; place it wherever the target harness expects a task to come from. Examples of ingestion contracts seen in the wild — pick the one the harness actually uses:
   - a `reference.py` file in a per-task config dir,
   - a row in a tasks table/parquet (`code` column),
   - a registry/entry-point the loader imports.
   Identify that contract from the harness, drop the module in, and confirm the harness's own bench/verify step passes. The skill stops at "module + green preflight"; this step is intentionally not automated because it differs per harness.

## Files

- `references/contract.md` — the KernelBench `Model` contract and the non-obvious bench invariants, as a checklist.
- `references/vram-fit.md` — whole-model vs reduced-config vs submodule decision tree and how to size it.
- `scripts/preflight.py` — standalone, harness-independent legality + VRAM probe.

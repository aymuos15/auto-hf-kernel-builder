---
name: model-select
description: PREP tier (once per model+hardware, not the loop). Profile a real forward, rank real nn.Module blocks, pick #1, freeze an immutable contract the loop consumes. Driven by `driver.py --phase prep`.
---

# model-select (PREP — once per target)

Per-target analogue of SETUP (AGENTS.md). Run once via `driver.py --phase prep`; produces an **immutable** contract the `kernel-opt` loop consumes and never regenerates. Re-run only if `model.id`/hardware changes. Model-agnostic — read the model from `config.yaml`.

## Procedure

1. **Read the model card** of `config.yaml: model.id` → exact transformers load path + a realistic input. Never synthetic.
2. **Profile a real forward** on the authoritative machine (via `config.yaml: exec:`) with `scripts/profile_model.py`; methodology = `skills/profiling`.
3. **Rank** real `nn.Module` classes with `scripts/rank_blocks.py` (forward-hook inclusive time; excludes root/containers/primitive-ops/whole-model wrappers). Let the numbers pick the dominant repeated block — don't assume an architecture.
4. **Score** = fraction-of-runtime (must be >> single-op) × reuse (sub-linear) ; must be memory-bound (not vendor GEMM) and have a clean I/O contract. Tie-break by most `torch.compile` headroom.
5. **Sanity-gate** the winner: `scripts/headroom_probe.py` — if `torch.compile` is already near-optimal, it's not beatable; pick #2.
6. **Freeze** with `scripts/freeze_contract.py`: capture real inputs + golden output under `targets/<model>/<block>/`. Immutable; everything downstream validates against it.

## Done when

`targets/<model>/<block>/` has `reference.py` (importable factory), `inputs.pt`, `golden.pt`, `selection.json`, `baseline.json`.

## Do not

Pick a single op (isolated win, no model move) · use synthetic shapes · pick a compute-bound vendor-GEMM block (already optimal).

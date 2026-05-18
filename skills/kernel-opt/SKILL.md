---
name: kernel-opt
description: The autonomous loop. Given a frozen block contract, write a Triton kernel as a universal kernel-builder project, pass the gates (correctness, build, beat torch.compile), report. Use after model-select.
---

# kernel-opt

**YOU (the agent) own the loop.** `driver.py --phase optimize` is one straight-line pass — scaffold → load → gates → `result.json` → exit 0/1. It does not loop. You iterate, up to `config.yaml: loop.max_retries`:

```
1. edit ONLY the seam: kernel/torch-ext/<name>/__init__.py  (_triton_impl)
2. run EXACTLY:  python3 pipeline/driver.py --phase optimize --config <cfg>
3. read targets/<model>/<block>/result.json  (passed / failed_gate / error_class)
4. not passed → revise the seam guided by error_class → go to 2
```

**This is the only validation path. There is no other.** The driver is the sole process that may touch the model, `golden.pt`, or the gates — it loads the configured exec + HF token and writes the verdict.

**FORBIDDEN — do not, ever:** write your own check against the model/`golden.pt` (e.g. `python - <<PY … torch.allclose(out, golden) … PY`, ad-hoc scripts, `ssh`, `docker`). Such a probe runs **unauthenticated** (the `HF_TOKEN` is loaded by the driver, not your shell → the "unauthenticated requests to the HF Hub" warning) and **off the configured exec/toolchain**, so its numbers can be wrong and are *never* the verdict. If you typed a heredoc against `golden.pt`, you are off-path: delete it, go back to step 2. Want a number? It is in `result.json`. All specifics from `config.yaml`.

## Inputs

`targets/<model>/<block>/`: `reference.py`, `inputs.pt`, `golden.pt`, `baseline.json`, `selection.json`. `config.yaml: env:` = the toolchain contract.

## One pass

1. **Study.** `scripts/dump_inductor.py` → the `torch.compile(max-autotune)` codegen you must beat. See `references/triton-from-inductor.md`.
2. **Write the kernel.** `driver` auto-scaffolds the universal-Triton project (`references/kernel-builder-layout.md`). **AGENT SEAM:** fill `kernel/torch-ext/<name>/__init__.py` `_triton_impl` with a real `@triton.jit` kernel that reproduces `golden.pt` from the contract inputs and beats `torch.compile`.
3. **CORRECTNESS** (`gates.py:correctness`): 5 stages vs `golden.pt`, rtol/atol from config. In-process locally (Triton JITs; no build to run).
4. **BUILD** (`gates.py:build`): rsync → host Nix `nix build {gates.build.nix_build_attr}` → `cp -rL result build` → container `sys.path` import of `build/torch-universal/<name>`. Universal = pure Python; not `get_local_kernel`; `target_caps` N/A; no Hub.
5. **PERF** (`gates.py:perf`): custom vs `torch.compile(max-autotune)` on the authoritative machine; pass iff speedup ≥ `perf.min_speedup_vs_compile`. Eager is not the bar.
6. **Pass → integrate** (#7, not yet wired): transformers `KernelConfig` hook + assert the kernel fires + end-to-end model speedup; see `references/transformers-integration.md`.

## Do not

Gate on eager · weaken gates/`config.yaml: env:`/the frozen contract to force a pass · create or push a Hub repo (`publish.enabled` false) · trust your own self-test over `result.json`.

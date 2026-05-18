# AGENTS.md

Improve a HF model by replacing a hot **block** with a **Triton** kernel that **beats `torch.compile`**, builds via `kernel-builder` (universal), loads via the `kernels` client from a local build (no Hub). All model/hardware specifics live in `config.yaml`; `skills/` + `pipeline/` are generic. Code is fail-fast and minimal (no defensive try/except except the correctness gate's, which is its product). Status: ✅ real & verified · 🟡 real, untested · ⛔ pending.

## Three tiers, three cadences

```
A. PROVISION  once / machine  →  SETUP (below) ; make image  →  config.yaml: env:
B. PREP       once / target   →  make prep
C. LOOP       every run        →  make loop
```

- **A** — human, one-time per machine.
- **B** — runs on the authoritative machine via `config.yaml: exec:` (rsync → run in container → rsync contract back); local only if `exec.ssh` empty. Profiles → ranks real `nn.Module` blocks → freezes an **immutable** contract. Idempotent: re-run only if `model.id`/hardware changes.
- **C** — the agent owns the loop; `driver --phase optimize` is one pass (scaffold → load → gates → `result.json` → exit 0/1). Refuses without B's contract. `make verify` = optional Spark de-risk.

Switch target/machine = edit `config.yaml`, redo **B**.

## A. SETUP — one-time, per-machine, human (not a pipeline phase)

Provision once, record into `config.yaml: env:`; the pipeline trusts that block and never re-provisions. All host/GPU/container specifics read from `config.yaml` (`hardware`, `exec`, `env`).

1. Confirm reachability + GPU matches `hardware.authoritative` (name + compute capability) via `exec:`.
2. If `exec.container` set, pull the base image. A configured container is the only supported torch source — do not pip-install torch (newer/aarch64 wheels unreliable; that's why a container is pinned).
3. `make image` once — bakes the `kernels` client + `transformers`/`pillow`/`pyyaml` into the base (`Dockerfile`, uv) so the ephemeral loop/prep never reinstalls. Re-run only if the base tag / `env.toolchain.kernels` changes.
4. `kernel-builder` = a Nix flake on the host (`github:huggingface/kernel-builder`), not the container/PyPI; Nix pre-provisioned; `make verify` confirms the aarch64-linux package resolves.
5. Iteration machine (`hardware.iteration`): native torch only, correctness on reduced shapes; never run the full model there if it lacks memory.
6. `make verify`, then record observed container tag + torch/triton/transformers/CUDA into `config.yaml: env.toolchain` (the toolchain contract; a mismatch stops a run).

Do not: `pip install torch` when a container is configured · change the container tag/toolchain mid-project · make the pipeline run SETUP.

## Files

| Path | Role | Status |
|---|---|---|
| `config.yaml` | Single source of truth: model, hardware, exec (ssh+container, `--user`, HF cache, `--env-file`), env (baked toolchain), baseline, gates, loop, publish (off). | ✅ |
| `pipeline/execwrap.py` | `run(cmd, surface, stream/tee)` — one remote parse (argv ssh), live or captured. | ✅ |
| `pipeline/gates.py` | Uniform `gate(GateCtx)->GateResult`: `correctness` (5-stage vs golden; one required try = its product), `build` (rsync kernel project → inline nix build on host → `_import_probe.py` in container), `perf` (median CUDA-event vs `torch.compile`). | correctness ✅ · build/perf 🟡 |
| `pipeline/_import_probe.py` | argv import probe run in-container (kept a file: crosses ssh→docker→bash quoting layers). | ✅ |
| `pipeline/driver.py` | `prep` (idempotent, routes via exec) · `optimize` (single pass; branch-free `GATES[g](ctx)`; no internal loop). | ✅ |
| `skills/model-select/` | Prep: `profile_model` (real module-tree attribution), `rank_blocks`, `freeze_contract` (+ importable `reference.py`), `headroom_probe`, `_common`. | ✅ |
| `skills/profiling/SKILL.md` | nsys-first → ncu-second; `ERR_NVGPUCTRPERM`; UMA/Arm-SBSA caveats. | ✅ |
| `skills/kernel-opt/` | Loop spec + `scaffold_kernel` (universal project, idempotent, agent seam) + `dump_inductor` + references. | ✅ |
| `Makefile` | `verify` · `image` · `prep` · `loop` · `clean`. Vars from `config.yaml`. | ✅ |
| `Dockerfile` | uv-based: base + `kernels`/`transformers`/`pillow`/`pyyaml` baked. | ✅ |
| `targets/<slug>/` | Tracked: `selection.json`, `<Block>/{contract.json,reference.py,baseline.json}`. Ignored: `profile.json`, `inputs/golden.pt`, `inductor/`, `result.json`, `kernel/build/`. | ✅ |

## Proven (real hardware)

- **Spark de-risk end-to-end**: universal Triton kernel built via host-Nix `kernel-builder` (pinned rev `b4accba`, `.#bundle`), imported in the container, ran correctly on **GB10 (cc 12.1)**. torch 2.10 + triton 3.5 in NGC `pytorch:25.11`; kernel-builder is host-Nix (not PyPI/container).
- **Prep**: `sam-vit-base` → winner `SamVisionSdpaAttention` (~74% runtime, 12 instances); immutable contract + `moderate_headroom` baseline; runs on the authoritative machine via `exec:`.
- **Agentic core**: the DeepSeek agent autonomously wrote a fused flash-attention Triton kernel for the real block; the **5-stage correctness gate PASSED** (max diff 0.0034). Build-gate root-ownership bug fixed via `--user $(id -u):$(id -g)` + chown.

## Pending

- 🟡 First full Spark `build`+`perf` pass with the *agent's actual* universal project (only `hello_triton` was de-risked end-to-end).
- ⛔ **#7**: transformers `KernelConfig` integration + kernel-invoked assertion + end-to-end model speedup in `result.json`.
- ⛔ true roofline in `headroom_probe` (currently eager-vs-compile proxy).
- Loop discipline: the agent must edit-seam → run driver and trust only `result.json` — not hand-roll `ssh`/`docker`/self-tests.

## Key decisions

Triton = **universal** kernel-builder project (no conflict with "must build with kernel-builder"); `target_caps` N/A for universal; build host-Nix, load in-container; publish off. Reuse: HF `cuda-kernels` skill shape (Triton/universal variant) + KernelBench-v2 5-stage correctness pattern (ported, no dep); loop + gates + `model-select` are built here (the contribution). Deferred (same loop, more iterations): diffusers, manual nn.Module surgery, CUDA kernels, multi-block stacking. Strategy+registry gates (`GATES`, uniform `GateCtx`); execwrap = pure command-builder + thin impure `run`/`rsync` (argv, no shell parse); inline host commands where one quoting layer; a relocated `_import_probe.py` only where multi-layer (ssh→docker→bash) quoting demands a file. Loop policy worth adding (AutoKernel-aligned): keep-best + revert-on-regression + multi-criterion stop.

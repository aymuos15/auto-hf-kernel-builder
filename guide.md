# Guide

Autonomous Triton kernel generation for KernelBench tasks. Built phase by phase.

```
config ─┐                                   ┌────────── agent loop ──────────┐
        ├─ Task → Env  (config.json)        │                                │
setup ──┤                                   Kernel (AI) edits kernel.py → Bench
        ├─ Benchmark   (res.json)                                            │
        └─ Profile     (prof.json)          Bench: correctness → perf → build │
                                            (build only if correct AND fast)  ┘
```

`config` then `setup` are one-time prep, run by a human before the agent enters.

The agent loop is `edit kernel.py → bench`, repeating until `bench` passes.

`bench` runs `kernel.py` directly for correctness, then perf vs the frozen bar. Only once it is correct *and* beats the bar does it build with kernel-builder to confirm compatibility — so the per-iteration loop never pays the nix build.

The agent's only verb is `bench`.

| phase | code | creates |
|---|---|---|
| 1 Task | `src/task/load.py` | — (reads `data/level_*.parquet`) |
| 2 Env + Config | `src/env/extract.py`, `src/env/create.py` | `config.json` |
| 3 Benchmark | `src/benchmark/baseline.py` | `res.json` |
| 4 Profile | `src/profiling/inductor.py` | `prof.json`, `inductor.py` |
| 5 Kernel | `src/kernels/scaffold.py` (resolves; AI/human edits) | `kernel.py` |
| 6 Bench | `src/bench.py` (+ `src/kernels/builder.py` for the final build) | `bench.json` (+ `build.json`, `kernel/` on pass) |

Phases 1–4 are `config` (1–2) + `setup` (3–4). Phases 5–6 are the agent loop. All artifacts live in `configs/<name>/`.

## CLI

`src/cli.py` (Typer). The agent's surface is exactly `bench`.

```
python3 src/cli.py config --level 3 --problem 4              # human, once: Task+Env → config.json (review knobs)
python3 src/cli.py setup  --config configs/<name>/config.json  # human, once: Benchmark+Profile (freezes the bar)
python3 src/cli.py build  --config configs/<name>/config.json  # standalone: kernel-builder build only (debug)
python3 src/cli.py bench  --config configs/<name>/config.json  # agent: correctness → perf → build-to-confirm
```

`setup` shows a header panel, one bar per phase, a results table; verbose output → `configs/<name>/run.log`. `build` exits 0 if it builds else 1. `bench` exits 0 only if the kernel is correct, beats the frozen `res.json` compile time, **and** builds with kernel-builder; else 1 with `error_class` in `bench.json`.

# Phase 1: Task

A task is one KernelBench problem, identified by `(level, problem_id)`.

`data/level_{1..4}.parquet` (from `ScalingIntelligence/KernelBench`) is the source of truth. Each row is one self-contained problem:

| column | meaning |
|---|---|
| `code` | a Python string: `Model(nn.Module)` + `get_inputs()` + `get_init_inputs()` |
| `level` | 1–4 (1 = single op … 3 = full architecture, e.g. ResNet/LeNet) |
| `name` | e.g. `4_LeNet5` |
| `problem_id` | unique id within the level |

`src/task/load.py` → `load_task(level, problem_id) -> Task(level, problem_id, name, code)`.

# Phase 2: Environment + Config

Each config gets its own folder `configs/<name>/`, with `config.json` as the only tracked input — the task lives *in* it. A run is fully determined by that one file; every later phase reads it and writes its artifacts into the same folder.

`src/env/extract.py` → `extract_env()`: pure detection — device, GPU, compute capability, VRAM, torch, CUDA, nix.

`src/env/create.py` → `create_config(level, problem_id, name=None)`: task identity + env + default thresholds → `configs/<name>/config.json` (default name `L<level>_<task>`; idempotent).

| section | holds |
|---|---|
| `task` | level, problem_id, name |
| `env` | device, gpu, compute_capability, vram_gb, torch, cuda, nix |
| `benchmark` | warmup, iters, compile_mode |
| `correctness` | rtol, atol |
| `perf` | min_speedup_vs_compile |
| `build` | kernel_builder (pinned rev), nix_attr, universal |

`correctness` and `perf` are **human-set knobs** — not auto-derived. They define what "correct enough" and "fast enough" mean, so a human reviews/tunes them per task before the agent enters.

# Phase 3: Benchmark

Establish the bar the kernel must beat: GPU time + peak memory for eager vs `torch.compile`.

Config-driven. `src/benchmark/baseline.py` reads the task + `benchmark` knobs, seeds before building (reproducible), times with CUDA events (median), tracks `torch.cuda.max_memory_allocated`, writes `res.json`. This frozen `compile` time is the bar; it is never re-measured later.

| field | meaning |
|---|---|
| `time_ms` | median latency over N timed iters (CUDA events) |
| `peak_mem_mb` | peak GPU memory during the timed run |
| `compile_speedup` | `eager.time_ms / compile.time_ms` |

Example — L3 #4 `4_LeNet5`: eager 9.13 ms / 173 MB, compile 7.27 ms / 16 MB (1.26×).

# Phase 4: Profile

Surface what `torch.compile` already generated so the kernel-writer can mine it for fusion/tiling ideas — the Inductor output *is* the bar.

`src/profiling/inductor.py` recompiles the task's Model with the Inductor trace on (caches disabled, else codegen is skipped), concatenates every generated `output_code.py` (fused Triton + wrapper) into `inductor.py`, and writes an index `prof.json`.

| `prof.json` field | meaning |
|---|---|
| `inductor_code` | filename of the captured fused Triton (`inductor.py`) |
| `num_graphs` | Inductor graphs captured |
| `triton_kernels` / `num_triton_kernels` | generated `triton_*` kernel names + count |

Example — L3 #4 `4_LeNet5`: 1 graph, 7 fused Triton kernels, 512 lines captured.

# Phase 5: Kernel

`configs/<name>/kernel.py` is the single file the AI / human owns: a callable `kernel(*inputs)` that must reproduce the whole task computation.

It is a **precondition** — `src/kernels/scaffold.py` only resolves it (errors if absent); it does not generate one.

# Phase 6: Bench

The agent's only verb. `src/bench.py` runs `kernel.py` **directly** (no nix) against the seeded reference, regenerated from `config.task` + `SEED` via `benchmark._build` — nothing is frozen to disk:

1. **correctness** — `kernel(*inputs)` vs `Model(*inputs)` within `config.correctness` rtol/atol.
2. **perf** — median kernel time vs the **frozen** `res.json` compile time; must be ≥ `config.perf.min_speedup_vs_compile`.
3. **build-to-confirm** — only if correct *and* fast: build with kernel-builder.

Pass iff all three. The per-iteration loop pays only 1–2 (no nix); the build runs once, when the kernel finally deserves it.

| `bench.json` field | meaning |
|---|---|
| `passed` | correct AND ≥ bar AND builds |
| `error_class` | `no_baseline` / `no_kernel` / `kernel_exception` / `numeric_mismatch` / `slower_than_compile` / `nix_build_failed` / `null` |
| `max_abs_diff` | correctness gap |
| `speedup_vs_compile` | `compile_ms / kernel_ms` |
| `built` | kernel-builder build confirmed (only on pass) |

Requires a prior `setup` (for `res.json`). No prior `build` needed — `bench` runs it.

## The build-to-confirm gate

Hard requirement (maintainer): a passing kernel must compile and build with HF **kernel-builder**. `src/kernels/builder.py` runs this as step 3 of `bench` (or standalone via the `build` verb):

- `assemble.sh` turns `kernel.py` into a kernel-builder universal-Triton project: `configs/<name>/kernel/{build.toml, flake.nix, torch-ext/<pkg>/__init__.py}`. `flake.nix` (template `src/kernels/flake.nix`) pins kernel-builder to `b4accba…` (proven, cache-backed).
- `build.sh` runs `nix build --accept-flake-config path:<proj>#bundle -o result` (`path:` so the gitignored project is visible to Nix).
- writes `build.json`.

Speed: pinned rev + HF Cachix substituter (download, don't compile) + reused `flake.lock` + warm `/nix/store` + a content-hash skip (unchanged `kernel.py` + prior pass + `result` present → no nix invocation). First build ≈ minutes (cold closure); thereafter seconds.

| `build.json` field | meaning |
|---|---|
| `passed` | `nix build` succeeded |
| `error_class` | `nix_build_failed` / `no_nix` / `null` |
| `pkg` | kernel-builder package name |
| `kernel_sha` | sha256 of `kernel.py` (drives the skip) |

# Tracking

`configs/<name>/config.json` is the only tracked input. Everything else in the folder — `kernel.py`, `res.json`, `prof.json`, `inductor.py`, `run.log`, `build.json`, `bench.json`, `kernel/` — is derived and gitignored.

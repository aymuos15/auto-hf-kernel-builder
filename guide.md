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

A human then starts `solve` — the code-owned loop. Each iteration it runs the agent (opencode, **no shell** — `opencode.json` denies bash; it can only edit files) for one turn, reverts any edit except `kernel.py`, then **the loop** runs `bench` and feeds the verdict into the next turn. Repeats until `bench` passes or `loop.max_retries`.

`bench` runs `kernel.py` directly: correctness on two seeded input sets + determinism, a real `@triton.jit` launch required, perf vs the frozen bar. Only once correct *and* fast does it build with kernel-builder — so the per-iteration loop never pays the nix build.

The agent never runs anything; the loop owns iteration and bench.

| phase | code | creates |
|---|---|---|
| 1 Task | `src/task/load.py` | — (reads `data/level_*.parquet`) |
| 2 Env + Config | `src/env/extract.py`, `src/env/create.py` | `config.json`, `reference.py` |
| 3 Benchmark | `src/benchmark/baseline.py` | `res.json` |
| 4 Profile | `src/profiling/inductor.py` | `prof.json`, `inductor.py` |
| 5 Solve | `src/agent/loop.py` + `prompt.md`; per turn: `src/kernels/scaffold.py` (kernel seam), `src/bench.py`, `src/kernels/builder.py` | `kernel.py`, `bench.json` (+ `build.json`, `kernel/` on pass) |

Phases 1–4 are `config` (1–2) + `setup` (3–4). Phase 5 (Solve) is the agent loop; each iteration it writes `kernel.py` and benches it. All artifacts live in `configs/<name>/`.

## CLI

`src/cli.py` (Typer). Humans run `config`/`setup`/`solve`; the agent runs nothing.

```
python3 src/cli.py config --level 3 --problem 4              # human, once: Task+Env → config.json + reference.py
python3 src/cli.py setup  --config configs/<name>/config.json  # human, once: Benchmark+Profile (freezes the bar)
python3 src/cli.py solve  --config configs/<name>/config.json  # human: start the agent loop (opencode → bench, repeat)
python3 src/cli.py bench  --config configs/<name>/config.json  # the loop calls this; or run by hand to debug a kernel
python3 src/cli.py build  --config configs/<name>/config.json  # standalone: kernel-builder build only (debug)
```

`setup` shows a header panel, one bar per phase, a results table; verbose output → `configs/<name>/run.log`. `bench` exits 0 only if the kernel is correct, beats the frozen `res.json` compile time, **and** builds with kernel-builder; else 1 with `error_class` in `bench.json`. `solve` refuses to start on a dirty git tree.

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

`src/env/create.py` → `create_config(level, problem_id, name=None)`: task identity + env + default thresholds → `configs/<name>/config.json` (default name `L<level>_<task>`; idempotent). It also writes `reference.py` — the verbatim KernelBench `Model` code — so the agent can study the exact spec it must reproduce.

| section | holds |
|---|---|
| `task` | level, problem_id, name |
| `env` | device, gpu, compute_capability, vram_gb, torch, cuda, nix |
| `benchmark` | warmup, iters, compile_mode |
| `correctness` | rtol, atol |
| `perf` | min_speedup_vs_compile |
| `build` | kernel_builder (pinned rev), nix_attr, universal |
| `loop` | max_retries |
| `agent` | model (opencode model id) |

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

# Phase 5: Solve

The code-owned agent loop. `src/agent/loop.py` (CLI `solve`, human-started; never agent-reachable). It refuses to start on a dirty git tree, snapshots `configs/<name>/`, then per iteration up to `config.loop.max_retries`:

1. run `opencode run --model <config.agent.model>` for one turn with `src/agent/prompt.md` (+ the last `bench.json` appended). `opencode.json` denies bash/webfetch; the agent can only edit files.
2. **integrity restore** — `git checkout -- . && git clean -fdq` (reverts any engine edit; `configs/`+`data/` are gitignored) and restore `configs/<name>/` from the snapshot **except `kernel.py`**. So `bench` always sees a pristine harness and only the agent's kernel.
3. the **loop** runs `bench` (subprocess) and reads `bench.json`.
4. pass → stop. else keep-best (highest correct speedup) + revert-on-regression, feed `error_class` into the next prompt.

The agent has no shell and never runs `bench` — the loop does, between turns. Threat model is minimal for now (lock + integrity restore); `kernel.py` is still executed in-process by `bench`, so a containerized/subprocess-isolated bench is the deferred hardening.

## The kernel seam

`configs/<name>/kernel.py` is the single file the agent edits: a callable `kernel(*inputs)` that must reproduce the whole task computation. `src/kernels/scaffold.py` only resolves it (errors if absent); it does not generate one.

## Bench

`src/bench.py` runs `kernel.py` **directly** (no nix) against the seeded reference, regenerated from `config.task` + `SEED` — weights fixed by `SEED`, inputs varied per seed; nothing frozen to disk:

1. **correctness** — `kernel(*inputs)` vs `Model(*inputs)` within `config.correctness` rtol/atol, on **two** input sets A and B (defeats memoization / constant-return), plus a determinism check (same input → same output).
2. **triton-invoked** — at least one `@triton.jit` kernel must actually launch during a `kernel` call (defeats torch passthrough).
3. **perf** — median kernel time on B vs the **frozen** `res.json` compile time; must be ≥ `config.perf.min_speedup_vs_compile`.
4. **build-to-confirm** — only if all the above pass: build with kernel-builder.

Pass iff all four. The per-iteration loop pays only 1–3 (no nix); the build runs once, when the kernel finally deserves it.

| `bench.json` field | meaning |
|---|---|
| `passed` | correct AND triton AND ≥ bar AND builds |
| `error_class` | `no_baseline` / `no_kernel` / `kernel_exception` / `numeric_mismatch` / `nondeterministic` / `no_triton` / `slower_than_compile` / `nix_build_failed` / `null` |
| `max_abs_diff` | correctness gap (worst of A, B) |
| `speedup_vs_compile` | `compile_ms / kernel_ms` |
| `triton_launches` | `@triton.jit` launches counted |
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

`configs/<name>/config.json` is the only tracked input. Everything else in the folder — `reference.py`, `kernel.py`, `res.json`, `prof.json`, `inductor.py`, `run.log`, `build.json`, `bench.json`, `kernel/` — is derived and gitignored.

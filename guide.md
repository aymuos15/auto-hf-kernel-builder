# Guide

Autonomous Triton kernel generation for KernelBench tasks. Built phase by phase.

# Phase 1: Creating a task

A task is one KernelBench problem, identified by `(level, problem_id)`.

`data/level_{1..4}.parquet` (from `ScalingIntelligence/KernelBench`) is the source of truth. Each row is one self-contained problem:

| column | meaning |
|---|---|
| `code` | a Python string: a `Model(nn.Module)` + `get_inputs()` + `get_init_inputs()` |
| `level` | 1–4 (1 = single op … 3 = full architecture, e.g. ResNet/LeNet) |
| `name` | e.g. `4_LeNet5` |
| `problem_id` | unique id within the level |

`src/task/load.py` → `load_task(level, problem_id) -> Task(level, problem_id, name, code)`.

# Phase 2: Environment + Config

Each config gets its own folder `configs/<name>/`, with `config.json` as the input. The task lives *in* `config.json`. A run is fully determined by that one file; every later phase reads it and writes its artifacts into the same folder (`res.json`, `prof.json`, `inductor.py`, `run.log`, `kernel.py`).

`src/env/extract.py` → `extract_env()`: pure detection — device, GPU, compute capability, VRAM, torch, CUDA.

`src/env/create.py` → `create_config(level, problem_id, name=None)`: task identity + env + default thresholds → `configs/<name>/config.json` (default name `L<level>_<task>`; idempotent).

| section | holds |
|---|---|
| `task` | level, problem_id, name |
| `env` | device, gpu, compute_capability, vram_gb, torch, cuda |
| `benchmark` | warmup, iters, compile_mode |
| `correctness` | rtol, atol |
| `perf` | min_speedup_vs_compile |

`correctness` (`rtol`/`atol`) and `perf` (`min_speedup_vs_compile`) are **human-set knobs** — not auto-derived. They define what "correct enough" and "fast enough" mean, so a human reviews/tunes them per task before solving.

# Phase 3: Benchmark

Establish the bar a kernel must beat: GPU time + peak memory for eager vs `torch.compile`.

Config-driven — the only input is `configs/<name>/config.json`. `src/benchmark/baseline.py` reads the task identity + `benchmark` knobs (`warmup`, `iters`, `compile_mode`), seeds before building (reproducible), times with CUDA events (median), tracks `torch.cuda.max_memory_allocated`, and writes `configs/<name>/res.json` in the same folder.

Orchestration: `python3 src/run.py --config configs/<name>/config.json` runs every stage; its only input is a config path. The console shows just the header panel, one progress bar per phase, and the final results table — all verbose output (stage prints, torch chatter, failure tracebacks) goes to `configs/<name>/run.log`, sectioned per phase. On failure: a one-line console pointer to the log, exit 1.

Reports, for each of eager and `torch.compile`:

| field | meaning |
|---|---|
| `time_ms` | median latency over N timed iters (CUDA events) |
| `peak_mem_mb` | peak GPU memory during the timed run |
| `compile_speedup` | `eager.time_ms / compile.time_ms` |

Example — L3 #4 `4_LeNet5`:

| | time | peak mem |
|---|---|---|
| eager | 9.13 ms | 173 MB |
| compile | 7.27 ms | 16 MB |
| | 1.26× | ~10× less |

# Phase 4: Profile

Surface what `torch.compile` already generated so the kernel-writer can mine it for fusion/tiling ideas — the Inductor output *is* the bar.

Config-driven — input is `configs/<name>/config.json`. `src/profiling/inductor.py` recompiles the task's Model with the Inductor trace on (caches disabled, else codegen is skipped), collects every generated `output_code.py` (fused Triton + wrapper) into `configs/<name>/inductor.py`, and writes an index `configs/<name>/prof.json`.

| `_prof.json` field | meaning |
|---|---|
| `inductor_code` | filename of the captured fused Triton (`inductor.py`) |
| `num_graphs` | Inductor graphs captured |
| `triton_kernels` | names of the generated `triton_*` kernels |
| `num_triton_kernels` | count |

Example — L3 #4 `4_LeNet5`: 1 graph, 7 fused Triton kernels (e.g. `triton_poi_fused_convolution_max_pool2d_with_indices_relu_*`, `triton_poi_fused_addmm_relu_*`), 512 lines captured.

`configs/<name>/config.json` is the only tracked input; `res.json` / `prof.json` / `inductor.py` / `run.log` in the folder are derived (gitignored).

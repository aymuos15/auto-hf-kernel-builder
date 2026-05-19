# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An autonomous pipeline that, for a KernelBench problem, has an LLM agent (opencode, headless) write a Triton `kernel.py` that reproduces a reference `Model` **and beats `torch.compile`**, then **builds with HF kernel-builder**. Local-only (no ssh/Spark). The deep architecture walkthrough is `guide.md`; the CLI reference is `docs/cli.md`. Read `guide.md` first.

## Commands

```
python3 src/cli.py config --level L --problem P              # creates configs/<name>/{config.json,reference.py}
python3 src/cli.py setup  --config configs/<name>/config.json # benchmark+profile → freezes the bar (res.json)
python3 src/cli.py solve  --config configs/<name>/config.json # the agent loop (foreground = watch it live)
python3 src/cli.py bench  --config configs/<name>/config.json # one verdict on the current kernel.py (debug)
python3 src/cli.py build  --config configs/<name>/config.json # kernel-builder build only (debug)

python3 -m pytest -q                       # all tests (fast, hermetic, no CUDA)
python3 -m pytest tests/test_env.py -k idempotent   # a single test
ruff check src && pyrefly check            # lint + typecheck (also run by pre-commit)
```

`pre-commit` runs ruff (lint+format) and pyrefly on every commit. **`ruff-format` will reformat files and abort the commit**; when that happens, `git add -A` again and re-commit (this is normal, not an error). Never `--no-verify`.

## Architecture (the big picture)

A **deterministic spine with exactly one LLM call site**. Everything is plain code except the agent that writes `kernel.py`.

`config → setup → solve`, where `solve` is a **code-owned loop**, not agent autonomy:

- The **narrow-waist contract** is the per-config folder `configs/<name>/`. Every phase communicates only through files there. Swapping the task source or the agent UI changes nothing else.
  - `config.json` (the only human-tuned input: thresholds, agent model, retries) · `reference.py` (verbatim KernelBench `Model` — the spec) · `res.json` (the frozen `torch.compile` bar — never re-measured) · `prof.json`/`inductor.py` (compile's fused output to mine) · `kernel.py` (the agent's only editable file) · `bench.json`/`build.json`/`build.log`/`trace/attempt_N.log` (verdicts + per-attempt transcript).
- `src/agent/loop.py` (`solve`): per iteration → run `opencode` for ONE edit turn → **integrity restore** (`git checkout/clean` + restore the config snapshot, keeping only `kernel.py`) → the *loop* runs `bench` → keep-best/revert → feed `error_class` into the next prompt. Stops on pass or `config.loop.max_retries`.
- `src/benchmark/bench.py`: the only source of truth. Correctness vs the seeded reference on **two** input sets + determinism; a real `@triton.jit` launch is required; perf vs the frozen `res.json` time; only if correct **and** fast does it build with kernel-builder. Reference is regenerated deterministically from `config.task` + `SEED` (weights fixed by seed, inputs varied) — nothing is frozen to disk.
- Module map: `src/task/load.py` (Phase 1, parquet) · `src/env/{extract,create,setup}.py` (Phase 2–4 prep) · `src/benchmark/baseline.py` (Phase 3 bar) · `src/profiling/inductor.py` (Phase 4) · `src/kernels/{scaffold,builder}.py` + `build.sh`/`assemble.sh`/`flake.nix` (the kernel-builder build).

### Why it's structured this way (load-bearing decisions — do not undo casually)

- **Anti-cheat is structural, not prose.** `opencode.json` denies the agent bash/webfetch; it can only edit files. The loop's integrity-restore discards any edit except `kernel.py`, so config/res/engine are always pristine when `bench` runs. `bench` requires a real Triton launch and checks two input sets — these defeat torch-passthrough and memoization cheats that prose rules never stopped.
- The agent **never runs anything**; the loop runs `bench` between turns. This is deliberate (the agent stalled when left to self-loop).
- `kernel.py` is still imported in-process by `bench` (a known, accepted residual risk — see `docs/failure_modes.md`); the deferred hardening is a containerized bench.

## Project-specific gotchas

- **`configs/` and `data/` are fully gitignored and never committed** (per-machine, regenerable). `config.json` is conceptually the "input" but is *not* tracked. Do not `git add` them — it breaks `solve` (its clean-tree check) and `git clean` would delete them.
- **`solve` refuses to start on a dirty git tree** and its integrity step runs `git checkout/clean`. Commit (or stash) engine changes before running `solve`.
- **kernel-builder is pinned** to rev `b4accba…` in `src/kernels/flake.nix` and `config.build.kernel_builder`. Do not bump it — newer revs changed `build.toml` schema and force a migration tool. The nix bundle build is transiently flaky; `builder.py` retries `config.build.nix_retries` (default 3) before `nix_build_failed`.
- **Code style enforced beyond ruff defaults:** no `from __future__ import annotations`; default to **no comments** (explanatory comments get stripped); markdown prose is single-line per paragraph (no hard-wrap). ruff line-length 100, target py310, `src/` layout (`pyproject.toml`).
- `docs/cli.md` is tracked; `docs/failure_modes.md` is gitignored (local notes).
- Only commit when explicitly asked.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An autonomous pipeline that, for a KernelBench problem, has an LLM agent (opencode, headless) write a Triton `kernel.py` that reproduces a reference `Model` **and beats `torch.compile`**, then **builds with HF kernel-builder**. Local-only (no ssh/Spark). The deep architecture walkthrough is `guide.md`; the CLI reference is `docs/cli.md`. Read `guide.md` first.

## Commands

```
python3 src/cli.py config --level L --problem P              # creates configs/<name>/{config.json,reference.py}; level 0 = data/custom.parquet
python3 src/cli.py setup  --config configs/<name>/config.json # benchmark+profile+freeze: res.json (bar) + ref.pt (frozen reference)
python3 src/cli.py solve  --config configs/<name>/config.json # the agent loop (foreground = watch it live)
python3 src/cli.py bench  --config configs/<name>/config.json # one verdict on the current kernel.py (debug)
python3 src/cli.py build  --config configs/<name>/config.json # kernel-builder build only (debug)

# isolation / scale (single box): durable queue + per-GPU guarded workers
python3 src/cli.py pool   --gpus 0,1 --queue configs/queue.db          # launch one persistent worker per GPU
AK_QUEUE=configs/queue.db AK_SANDBOX=subprocess python3 src/cli.py solve --config …  # route per-attempt bench through the queue/sandbox

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
- `src/benchmark/bench.py`: the only source of truth. Correctness vs the **frozen** reference (`ref.pt`) on **two** input sets + determinism; a real `@triton.jit` launch is required (+ `triton_figleaf`/`precision_cheat`/`reference_import` checks); perf vs the frozen `res.json` time; only if correct **and** fast does it build with kernel-builder. `setup` freezes `inputs_a/inputs_b/ref_a/ref_b/ref_deps` to `ref.pt` (deterministic from `config.task` + `SEED`); `bench` **never builds or imports the reference** — the reference and its deps are absent from the bench process, which is the structural anti-cheat (see `docs/isolation.md`, `docs/failure_modes.md`).
- Module map: `src/task/load.py` (Phase 1, parquet; level 0 = `data/custom.parquet`) · `src/env/{extract,create,setup}.py` (Phase 2–5 prep) · `src/benchmark/baseline.py` (Phase 3 bar + `freeze_reference`) · `src/benchmark/anticheat.py` (extracted heuristics) · `src/profiling/inductor.py` (Phase 4) · `src/kernels/{scaffold,builder}.py` + `build.sh`/`assemble.sh`/`flake.nix` (the kernel-builder build) · `src/worker/{queue,contract,pool,guard,sandbox}.py` (durable queue + per-GPU pool + import guard + sandbox backend).

### Why it's structured this way (load-bearing decisions — do not undo casually)

- **Anti-cheat is structural, not prose.** `opencode.json` denies the agent bash/webfetch; it can only edit files. The loop's integrity-restore discards any edit except `kernel.py`, so config/res/engine are always pristine when `bench` runs. `bench` requires a real Triton launch and checks two input sets — these defeat torch-passthrough and memoization cheats that prose rules never stopped.
- The agent **never runs anything**; the loop runs `bench` between turns. This is deliberate (the agent stalled when left to self-loop).
- **The reference is structurally absent from where the kernel runs.** `bench` compares against frozen `ref.pt` and never builds/imports the reference; the worker runs the job in a sandboxed child where `src/worker/guard.py` (a `sys.meta_path` finder) makes `ref_deps` unimportable by `import`/`importlib`/`__import__` alike. This closes the import/passthrough cheat family (A1–A4) — see `docs/failure_modes.md`. Within the sandboxed child, in-process monkeypatch of `kernel.py` is the remaining accepted residual (bwrap/seccomp tightens it).

## Project-specific gotchas

- **`configs/` and `data/` are fully gitignored and never committed** (per-machine, regenerable). `config.json` is conceptually the "input" but is *not* tracked. Do not `git add` them — it breaks `solve` (its clean-tree check) and `git clean` would delete them.
- **`solve` refuses to start on a dirty git tree** and its integrity step runs `git checkout/clean`. Commit (or stash) engine changes before running `solve`.
- **`solve`'s `_restore()` runs repo-wide `git clean -fdq`** every attempt — it deletes ALL untracked, non-gitignored files anywhere in the repo (not just `configs/`), and `git checkout -- .` reverts uncommitted edits to tracked files. Never leave uncommitted/untracked work in the repo while `solve` runs; the dirty-tree refusal only guards startup, not files created after (failure-mode H3).
- **kernel-builder is pinned** to rev `b4accba…` in `src/kernels/flake.nix` and `config.build.kernel_builder`. Do not bump it — newer revs changed `build.toml` schema and force a migration tool. The nix bundle build is transiently flaky; `builder.py` retries `config.build.nix_retries` (default 3) before `nix_build_failed`.
- **Code style enforced beyond ruff defaults:** no `from __future__ import annotations`; default to **no comments** (explanatory comments get stripped); markdown prose is single-line per paragraph (no hard-wrap). ruff line-length 100, target py310, `src/` layout (`pyproject.toml`).
- `docs/{cli,setup_cost,isolation}.md` are tracked; `docs/failure_modes.md` is gitignored (local notes). `skills/` is a tracked, schema-validated read-by-path skill library the agent prompt routes into (`skills/INDEX.md`); `python3 skills/validate_skills.py` checks it.
- Only commit when explicitly asked.

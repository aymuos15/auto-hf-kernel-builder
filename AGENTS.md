# AGENTS.md

Autonomous pipeline that, for a KernelBench problem, writes a Triton
`kernel.py` that reproduces the reference output **and beats
`torch.compile(max-autotune)`**. Local-only: no ssh/Spark, no
kernel_lib/Nix build. (Both removed deliberately; recover from git
history when needed.)

## Architecture: a deterministic spine with one agent

The contract is the **narrow waist**. Everything communicates through an
immutable frozen contract dir; swapping the task source changes nothing
downstream.

```
prepare (code) ──> [ frozen contract ] ──> loop (code) ──> report
                                              │
                                              ├─ solve  (AGENT: writes kernel.py)
                                              └─ benchmark (code: correctness + perf)
```

**Determinism is the integrity boundary.** An LLM is used in exactly one
place — writing the kernel. Everything else is code, because an agent
over a deterministic step is just a nondeterministic, cheatable failure
mode (e.g. an LLM "config" step could silently weaken a gate).

| Stage | Mechanism | File |
|---|---|---|
| **prepare** = Task + Environment + Config | DETERMINISTIC | `core/prepare.py` |
| **benchmark** = correctness + perf-vs-compile | DETERMINISTIC, sole source of truth | `core/gates.py` |
| **scaffold** = write the bare `kernel.py` seam | DETERMINISTIC | `core/scaffold.py` |
| **loop** = retry≤N, keep-best, revert-on-regression, stop-on-pass | DETERMINISTIC (the control structure) | `core/loop.py` |
| **solve** = write/revise one Triton kernel | **AGENT** (only LLM in the system) | `solve/launch.sh` + `solve/SKILL.md` |
| entrypoint | `prepare → loop → report` | `run.py` |

The loop is **code-owned**: the agent is invoked once per iteration as a
pure kernel-writer and only edits `kernel.py`. It has no shell
(`opencode.json`: edit allow, bash/webfetch deny) — it cannot self-test,
cheat with ad-hoc probes, or fail by "not running a command".

## Run

```
python3 run.py --level 3 --problem 4      # prepare -> loop -> report
make run LEVEL=3 PROBLEM=4
python3 core/prepare.py --level 3 --problem 4   # freeze contract only
```

`data/level_{1..4}.parquet` (gitignored) is the KernelBench source of
truth — re-fetch via `huggingface_hub` from
`ScalingIntelligence/KernelBench`. `tasks/<slug>/<block>/` is the frozen
contract (gitignored, regenerable): `model_src.py`, `weights.pt`,
`inputs.pt`, `golden.pt`, `reference.py`, `contract.json`,
`baseline.json`; the loop adds `kernel.py` (agent seam) and
`result.json`.

## Long-term

`prepare` is deterministic today (KernelBench parquet → contract). The
future agentic task-maker (latest HF model → profile a hot block →
freeze a KernelBench-like task) replaces `make_contract()` only, emitting
the **same contract** — `loop`/`solve`/`benchmark` never change. That is
the entire reason to nail the contract on the easy source first.

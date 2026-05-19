# agentic-kernels

Autonomous pipeline that, for a KernelBench problem, writes a Triton
`kernel.py` reproducing the reference output and **beating
`torch.compile(max-autotune)`**. Local-only: no ssh/Spark, no
kernel_lib/Nix.

```
python3 run.py --level 3 --problem 4     # prepare -> code loop -> report
make run LEVEL=3 PROBLEM=4
```

A deterministic spine with exactly one agent:

```
prepare (code) -> [frozen contract] -> loop (code) -> report
                                         |- solve   (AGENT writes kernel.py)
                                         |- benchmark (code: correctness + perf)
```

The loop is **code-owned**; the agent is a pure kernel-writer that only
edits `kernel.py` (no shell). Architecture/rationale: **`AGENTS.md`**.
Solver guidelines: **`solve/SKILL.md`**.

**Setup:** install `requirements.txt`, authenticate `opencode` for the
model in `configs/kernelbench.yaml: agent.model`. KernelBench parquet
auto-loads to `data/` via `huggingface_hub`
(`ScalingIntelligence/KernelBench`).

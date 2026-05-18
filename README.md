# agentic-kernels

Autonomous pipeline that replaces a hot model block with a Triton kernel that must beat `torch.compile`, built via `kernel-builder`, loaded locally (no Hub).

```
make image   # once per host: bake the container
make prep    # once per target: profile -> freeze immutable contract
make loop    # autonomous agent: write kernel -> correctness/build/perf gates
```

`config -> prep: profile -> rank -> freeze contract (scope + identify hot block) -> loop: scaffold seam -> agent writes Triton kernel -> correctness/build/perf gates`

**Setup (first time, in order):**

1. Provision the machine + `make image`, record toolchain — detail in `AGENTS.md` §A.
2. Install & authenticate `opencode` for the provider in `configs/<config>.yaml: agent.model`.
3. `printf 'HF_TOKEN=...\n' > secrets.env` (gitignored).
4. `make prep` (add `CONFIG=configs/config.local.yaml` to run prep locally).
5. `make loop`. `make help` lists all targets.

Configs live in `configs/` (working files comment-free; `configs/template.yaml` is the annotated reference). Everything specific is in `configs/config.yaml`: target model, hardware, exec/ssh+container, toolchain, gate thresholds, retry budget, publish policy. `make loop` feeds it (and `skills/`) to the autonomous agent as its only instructions. Canonical doc (architecture, SETUP, file status, decisions): **`AGENTS.md`**.

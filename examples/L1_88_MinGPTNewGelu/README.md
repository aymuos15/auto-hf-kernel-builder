# L1_88_MinGPTNewGelu — reference PASS

A committed snapshot of a full successful `solve` run (config + trace), kept as a reference / regression fixture. This is the whole per-config folder minus the derived `kernel/` nix build output and `__pycache__`.

KernelBench level 1, problem 88 (`88_MinGPTNewGelu`, the MinGPT tanh-GELU op). The agent produced a real `@triton.jit` kernel (sigmoid-form GELU) that the harness accepted:

- `max_abs_diff: 0.0` (bit-exact vs the seeded reference)
- `kernel_ms 3.23482` vs `compile_ms 6.50906` → **2.0122× vs torch.compile**
- `triton_launches: 1`, `passed: true`, `built: true` (kernel-builder)

Files: `config.json` / `reference.py` (the task), `res.json` / `inductor.py` / `prof.json` (the frozen bar), `kernel.py` (the winning kernel), `bench.json` / `build.json` / `run.log`, `trace/attempt_{1..5}.log` (per-attempt prompt + agent transcript + verdict + build log).

Caveat: this run predates the Phase-2 frozen-`ref.pt` bench, so it has no `ref.pt`. It is a historical proof-of-success artifact, **not** a live runnable config — `configs/` is gitignored by design; to run this task today, regenerate it with `config` + `setup` (which now also freezes `ref.pt`).

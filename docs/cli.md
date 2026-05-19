# CLI

`src/cli.py` (Typer). Humans run `config` / `setup` / `solve`; the agent (driven by `solve` via opencode) runs nothing.

## Verbs

```
python3 src/cli.py config --level 3 --problem 4                # human, once: config.json + reference.py
python3 src/cli.py setup  --config configs/<name>/config.json  # human, once: benchmark baseline + profile (freezes the bar)
python3 src/cli.py solve  --config configs/<name>/config.json  # human: start the agent loop (opencode → bench, repeat)
python3 src/cli.py bench  --config configs/<name>/config.json  # the loop calls this; or by hand to debug a kernel
python3 src/cli.py build  --config configs/<name>/config.json  # standalone: kernel-builder build only (debug)
```

## Options

```
--config   path to a config.json   (setup, solve, bench, build)
--level    KernelBench level        (config)
--problem  KernelBench problem_id   (config)
--name     config name, default L<level>_<task>  (config)
--force    rebuild config if it exists           (config)
python3 src/cli.py --help          # all commands
python3 src/cli.py <cmd> --help    # options for one command
```

## Flow

`config → setup → solve`.

`solve` is the code-owned loop. Each turn it runs opencode (which can only edit `kernel.py` — `opencode.json` denies bash), reverts any other edit (integrity restore), then **the loop** runs `bench` and feeds the verdict into the next turn. It refuses to start on a dirty git tree, and stops on pass or `config.loop.max_retries` (keeping the best kernel).

`bench`: correctness on two seeded input sets + determinism, a real `@triton.jit` launch required, perf vs the frozen `res.json` compile time; only once correct **and** fast does it build with kernel-builder (retried on transient nix failure). Exit 0 only if it passes all of those.

## Per-config artifacts (`configs/<name>/`, all gitignored)

| file | written by | what |
|---|---|---|
| `config.json` | `config` | input: task + env + knobs |
| `reference.py` | `config` | the verbatim KernelBench `Model` (the spec) |
| `res.json` | `setup` | frozen eager/compile bar |
| `prof.json` / `inductor.py` | `setup` | torch.compile's fused output to mine |
| `kernel.py` | agent (`solve`) | the kernel under test |
| `bench.json` | `bench` | verdict (`passed`, `error_class`, speedup, …) |
| `build.json` / `build.log` | `build` | kernel-builder result + full nix output |
| `trace/attempt_N.log` | `solve` | per-attempt prompt + agent transcript + verdict + build log |
| `kernel/` | `build` | the assembled kernel-builder project |

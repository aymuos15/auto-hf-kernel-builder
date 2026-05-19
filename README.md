This repo uses opencode headless to write Triton kernels that beat `torch.compile` and ensure kernel-builder compatibility. 

The agent is pointed at a config (see example). 

The base code must be in a KernelBench-style format.

### Cli

```
python3 src/cli.py config --level 3 --problem 4              # human, once: config.json + reference.py
python3 src/cli.py setup  --config configs/<name>/config.json  # human, once: benchmark baseline + profile (freezes the bar)
python3 src/cli.py solve  --config configs/<name>/config.json  # human: start the agent loop (opencode → bench, repeat)
python3 src/cli.py bench  --config configs/<name>/config.json  # the loop calls this; or by hand to debug a kernel
python3 src/cli.py build  --config configs/<name>/config.json  # standalone: kernel-builder build only (debug)
```

Flow: `config → setup → solve`. `solve` is the code-owned loop: each turn it runs opencode (which can only edit `kernel.py` — `opencode.json` denies bash), reverts any other edit, then the loop runs `bench` (correctness on two input sets + determinism + a real Triton launch + perf vs the frozen bar; builds with kernel-builder only once correct+fast) and feeds the verdict back. The agent never runs anything.

```
--config   path to a config.json (setup, build, bench)
--level    KernelBench level (config)
--problem  KernelBench problem_id (config)
--name     config name, default L<level>_<task> (config)
```
```
python3 src/cli.py --help          # all commands
python3 src/cli.py <cmd> --help    # options for a command
```

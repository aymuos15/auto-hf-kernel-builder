This repo uses opencode headless to write Triton kernels that beat `torch.compile` and ensure kernel-builder compatibility. 

The agent is pointed at a config (see example). 

The base code must be in a KernelBench-style format.

### Cli

```
python3 src/cli.py config --level 3 --problem 4              # human prep: create configs/<name>/config.json
python3 src/cli.py setup  --config configs/<name>/config.json  # once: benchmark baseline + profile (freezes the bar)
python3 src/cli.py build  --config configs/<name>/config.json  # agent: kernel-builder build only
python3 src/cli.py bench  --config configs/<name>/config.json  # agent: correctness + perf vs the frozen bar
```

Flow: `config → setup → (agent edits kernel.py → bench)*`. The agent's only verb is `bench`: it checks correctness + perf by running `kernel.py` directly, and only on a correct+fast kernel does it build with kernel-builder to confirm compatibility. `build` stays a standalone verb for setup/debug.

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

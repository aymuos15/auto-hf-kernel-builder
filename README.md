```
python3 src/cli.py config --level 3 --problem 4              # human prep: create configs/<name>/config.json
python3 src/cli.py setup  --config configs/<name>/config.json  # once: benchmark baseline + profile (freezes the bar)
python3 src/cli.py build  --config configs/<name>/config.json  # agent: kernel-builder build only
python3 src/cli.py bench  --config configs/<name>/config.json  # agent: correctness + perf vs the frozen bar
```

Flow: `config → setup → (agent edits kernel.py → build → bench)*`. Agent surface = `build` + `bench`.

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

# Phase 1: Creating a task

A task is one well-defined problem to optimize, identified by `(level, problem_id)`.

The KernelBench dataset (`data/level_{1..4}.parquet`, from `ScalingIntelligence/KernelBench`) is the source of truth. Each row is a complete, self-contained problem in four columns:

| column | meaning |
|---|---|
| `code` | a Python string: a `Model(nn.Module)` + `get_inputs()` + `get_init_inputs()` |
| `level` | 1–4 (1 = single op … 3 = full architecture, e.g. ResNet/LeNet) |
| `name` | e.g. `4_LeNet5` |
| `problem_id` | unique id within the level |

The parquet is authoritative — there is no file per problem.

Select a task:

```
python3 core/prepare.py --level 3 --problem 4
```

`_load_row()` (`core/prepare.py:102`) reads the parquet, filters to that `problem_id`, and returns the row. That row — the reference `Model` plus its input generators — is the task.

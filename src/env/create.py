"""Create a config: configs/<name>/config.json.

Each config gets its own folder. config.json holds task identity (the
task lives in the config) + detected environment + default thresholds.
Every later phase reads only config.json and writes its artifacts into
the same folder (res.json, prof.json, inductor.py, run.log, kernel.py).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.extract import extract_env  # noqa: E402
from task.load import load_task  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[2] / "configs"

_KB_REV = "b4accba4496b28faef19a0487fbcf9686b14e2ef"

DEFAULTS = {
    "benchmark": {"warmup": 10, "iters": 50, "compile_mode": "max-autotune"},
    "correctness": {"rtol": 2e-2, "atol": 2e-2},
    "perf": {"min_speedup_vs_compile": 1.05},
    "build": {
        "kernel_builder": f"github:huggingface/kernel-builder/{_KB_REV}",
        "nix_attr": ".#bundle",
        "universal": True,
    },
}


def create_config(
    level: int, problem_id: int, name: str | None = None, force: bool = False
) -> Path:
    task = load_task(level, problem_id)
    name = name or f"L{level}_{task.name}"
    # Each config gets its own folder; config.json is the input, all
    # later artifacts (res/prof/inductor/run.log/kernel) land beside it.
    cfg_dir = CONFIGS / name
    cfg_path = cfg_dir / "config.json"
    if cfg_path.is_file() and not force:
        print(f"config exists: {cfg_path} (use force=True to rebuild)")
        return cfg_path
    cfg_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "task": {"level": task.level, "problem_id": task.problem_id, "name": task.name},
        "env": extract_env(),
        **DEFAULTS,
    }
    cfg_path.write_text(json.dumps(config, indent=2))
    print(f"wrote {cfg_path}")
    return cfg_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--problem", type=int, required=True)
    ap.add_argument("--name", default=None, help="config name (default: L<level>_<task>)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    create_config(args.level, args.problem, args.name, args.force)

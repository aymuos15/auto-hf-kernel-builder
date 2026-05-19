"""Single entrypoint: prepare -> loop -> report.

  python3 run.py --level 3 --problem 4
  python3 run.py --level 3 --problem 4 --max-retries 4 --no-baseline

Deterministic prepare freezes the contract (idempotent); the code-owned
loop runs the agent kernel-writer + benchmark until pass or max_retries.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "core"))

from loop import loop  # noqa: E402
from prepare import detect_env, make_contract, resolve_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--problem", type=int, required=True)
    ap.add_argument("--max-retries", type=int, default=None)
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--force-prepare", action="store_true")
    args = ap.parse_args()

    print(f"env: {detect_env()}")
    block_dir = make_contract(args.level, args.problem,
                              baseline=not args.no_baseline,
                              force=args.force_prepare)

    cfg = resolve_config()
    if args.max_retries is not None:
        cfg.setdefault("loop", {})["max_retries"] = args.max_retries
    config_path = REPO / "configs" / "kernelbench.yaml"

    result = loop(cfg, block_dir, config_path)

    print("\n===== REPORT =====")
    print(f"task   : {block_dir.name}")
    print(f"baseline: {json.loads((block_dir/'baseline.json').read_text()).get('verdict')}")
    print(f"passed : {result['passed']}  "
          f"(failed_gate={result.get('failed_gate')}, "
          f"error_class={result.get('error_class')})")
    for g in result["gates"]:
        if g["name"] == "perf":
            print(f"perf   : {g['detail']}")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

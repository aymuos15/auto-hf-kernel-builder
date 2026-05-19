"""THE CODE-OWNED LOOP (deterministic control, no LLM judgment).

This is the structure that was missing — the reason the agent failed
before was that iteration was delegated to the agent. Here CODE owns:
retry <= max_retries, keep-best, revert-on-regression, stop-on-pass.
The agent is invoked once per iteration as a pure kernel-writer
(solve/launch.sh) and only edits kernel.py — it never runs anything.

  run_once : scaffold -> load kernel.py -> gates -> result.json
  loop     : (solve -> run_once -> keep-best/revert) up to max_retries
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "core"))

from gates import GATES, GateCtx  # noqa: E402
from scaffold import scaffold  # noqa: E402

_attempt_counter = 0


def load_kernel(block_dir: Path):
    global _attempt_counter
    _attempt_counter += 1
    spec = importlib.util.spec_from_file_location(
        f"_kernel_{_attempt_counter}", block_dir / "kernel.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.kernel


def run_once(cfg: dict, block_dir: Path) -> dict:
    scaffold(block_dir)  # idempotent: keeps an agent-filled kernel
    sequence = cfg.get("gates", {}).get("sequence") or ["correctness", "perf"]
    block = block_dir.name
    try:
        kernel = load_kernel(block_dir)
    except Exception as exc:
        result = {"block": block, "passed": False, "failed_gate": "load",
                  "error_class": "kernel_load_error",
                  "gates": [{"name": "load", "passed": False,
                             "detail": {"exception": repr(exc)},
                             "error_class": "kernel_load_error"}]}
        (block_dir / "result.json").write_text(json.dumps(result, indent=2))
        return result
    ctx = GateCtx(block_dir, kernel, cfg)
    results, failed = [], None
    for g in sequence:
        res = GATES[g](ctx)
        results.append(res)
        print(f"  gate {g}: {'PASS' if res.passed else 'FAIL'} "
              f"({res.error_class or 'ok'})")
        if not res.passed:
            failed = res
            break
    result = {"block": block, "passed": failed is None,
              "failed_gate": failed.name if failed else None,
              "error_class": failed.error_class if failed else None,
              "gates": [r.__dict__ for r in results]}
    (block_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def _speedup(result: dict):
    for g in result.get("gates", []):
        if g["name"] == "perf":
            return g["detail"].get("speedup_vs_compile")
    return None


def _solve(block_dir: Path, config_path: Path) -> None:
    """Invoke the agent for ONE kernel revision. It only edits kernel.py."""
    subprocess.run(["bash", str(REPO / "solve" / "launch.sh"),
                    str(block_dir), str(config_path)],
                   cwd=REPO, check=False)


def loop(cfg: dict, block_dir: Path, config_path: Path) -> dict:
    max_retries = int(cfg.get("loop", {}).get("max_retries", 6))
    scaffold(block_dir)
    best_kernel = block_dir / ".best_kernel.py"
    best = {"speedup": None, "result": None}

    last = None
    for attempt in range(1, max_retries + 1):
        print(f"\n=== attempt {attempt}/{max_retries}  {block_dir.name} ===")
        # Revert-on-regression: if we have a best, the agent revises FROM it,
        # not from a worse/broken draft.
        if best["result"] is not None and best_kernel.exists():
            shutil.copy(best_kernel, block_dir / "kernel.py")
        _solve(block_dir, config_path)
        result = run_once(cfg, block_dir)
        last = result

        if result["passed"]:
            print(f"PASS on attempt {attempt}")
            return result

        sp = _speedup(result)
        correct = any(g["name"] == "correctness" and g["passed"]
                      for g in result["gates"])
        # keep-best: prefer a correct kernel with the highest speedup so far.
        if correct and sp is not None and (best["speedup"] is None
                                           or sp > best["speedup"]):
            best = {"speedup": sp, "result": result}
            shutil.copy(block_dir / "kernel.py", best_kernel)
            print(f"  new best: correct, {sp}x vs compile")

    print(f"\nstopped after {max_retries} attempts (no pass)")
    if best["result"] is not None:
        shutil.copy(best_kernel, block_dir / "kernel.py")
        return best["result"]
    return last

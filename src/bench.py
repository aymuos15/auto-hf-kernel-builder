"""bench: the agent's verdict verb. Evaluate the BUILT kernel against
the seeded reference — correctness + perf vs the frozen baseline.

Reference is regenerated deterministically from config.task + SEED
(reusing benchmark._build) — nothing is frozen to disk. The bar is the
compile time already in res.json (from setup); bench never re-measures
the baseline. Requires a prior successful `build`.

Config-driven: input is configs/<name>/config.json. Writes bench.json.
"""

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from benchmark.baseline import _build, _measure  # noqa: E402
from kernels.builder import _pkg  # noqa: E402
from task.load import load_task  # noqa: E402


def _load_built_kernel(proj: Path, pkg: str):
    hits = list((proj / "result").rglob(f"{pkg}/__init__.py"))
    if not hits:
        return None
    spec = importlib.util.spec_from_file_location(f"_built_{pkg}", hits[0])
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.kernel


def run_from_config(config_path: str) -> Path:
    cfg_path = Path(config_path).resolve()
    cfg = json.loads(cfg_path.read_text())
    t = cfg["task"]
    out = cfg_path.parent / "bench.json"
    rec = {"task": t}

    def fail(cls, **extra):
        rec.update(passed=False, error_class=cls, **extra)
        out.write_text(json.dumps(rec, indent=2))
        print(f"bench: FAIL ({cls})")
        return out

    res = cfg_path.with_name("res.json")
    if not res.is_file():
        return fail("no_baseline", detail="run setup first (no res.json)")
    compile_ms = json.loads(res.read_text())["baseline"]["compile"]["time_ms"]

    pkg = _pkg(f"kb_{t['name']}")
    kern = _load_built_kernel(cfg_path.parent / "kernel", pkg)
    if kern is None:
        return fail("not_built", detail="run build first (no built artifact)")

    task = load_task(t["level"], t["problem_id"])
    model, inputs = _build(task, "cuda")
    c = cfg["correctness"]
    try:
        with torch.no_grad():
            ref = model(*inputs)
            got = kern(*inputs)
        correct = torch.allclose(
            got.float(), ref.float(), rtol=float(c["rtol"]), atol=float(c["atol"])
        )
        max_diff = (got.float() - ref.float()).abs().max().item()
    except Exception as exc:
        return fail("kernel_exception", detail=repr(exc))

    if not correct:
        return fail("numeric_mismatch", max_abs_diff=round(max_diff, 6))

    b = cfg["benchmark"]
    with torch.no_grad():
        kernel_ms = _measure(lambda *a: kern(*a), inputs, b["warmup"], b["iters"]).time_ms
    speedup = compile_ms / kernel_ms
    bar = float(cfg["perf"]["min_speedup_vs_compile"])
    passed = speedup >= bar
    rec.update(
        passed=passed,
        error_class=None if passed else "slower_than_compile",
        max_abs_diff=round(max_diff, 6),
        kernel_ms=round(kernel_ms, 5),
        compile_ms=compile_ms,
        speedup_vs_compile=round(speedup, 4),
        min_speedup=bar,
    )
    out.write_text(json.dumps(rec, indent=2))
    print(f"bench: {'PASS' if passed else 'FAIL'} (correct, {speedup:.4f}x vs compile, bar {bar})")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run_from_config(args.config)

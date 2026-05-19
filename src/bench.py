"""bench: the agent's single verb. Evaluate kernel.py against the seeded
reference — correctness + perf vs the frozen baseline — running it
directly (no nix). ONLY once it is correct AND beats the bar does it
build with kernel-builder to confirm compatibility. So the per-iteration
loop never pays the nix build; build is the final confirmation.

Reference is regenerated deterministically from config.task + SEED
(reusing benchmark._build) — nothing is frozen to disk. The bar is the
compile time already in res.json (from setup); bench never re-measures
the baseline.

Config-driven: input is configs/<name>/config.json. Writes bench.json.
"""

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from benchmark.baseline import _build, _measure  # noqa: E402
from kernels.builder import run_from_config as build_kernel  # noqa: E402
from task.load import load_task  # noqa: E402


def _load_kernel(kernel_py: Path):
    spec = importlib.util.spec_from_file_location("_kernel_src", kernel_py)
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

    kernel_py = cfg_path.with_name("kernel.py")
    if not kernel_py.is_file():
        return fail("no_kernel", detail=f"missing {kernel_py}")

    task = load_task(t["level"], t["problem_id"])
    model, inputs = _build(task, "cuda")
    c = cfg["correctness"]
    try:
        kern = _load_kernel(kernel_py)
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
    rec.update(
        max_abs_diff=round(max_diff, 6),
        kernel_ms=round(kernel_ms, 5),
        compile_ms=compile_ms,
        speedup_vs_compile=round(speedup, 4),
        min_speedup=bar,
    )
    if speedup < bar:
        return fail("slower_than_compile")

    bjson = json.loads(build_kernel(config_path).read_text())
    if not bjson.get("passed"):
        return fail(bjson.get("error_class") or "build_failed")

    rec.update(passed=True, error_class=None, built=True)
    out.write_text(json.dumps(rec, indent=2))
    print(f"bench: PASS (correct, {speedup:.4f}x vs compile, builds with kernel-builder)")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run_from_config(args.config)

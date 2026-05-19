"""bench: the agent's single verb. Evaluate kernel.py against the seeded
reference, running it directly (no nix). Correctness is checked on two
input sets (defeats memoization/constant-return) plus a determinism
check, and a real @triton.jit launch is required (defeats torch
passthrough). ONLY once correct AND faster than the frozen bar does it
build with kernel-builder to confirm compatibility.

Reference is regenerated deterministically from config.task + SEED
(weights fixed by SEED; inputs varied per seed) — nothing frozen to
disk. The bar is the compile time in res.json (from setup); never
re-measured here.

Config-driven: input is configs/<name>/config.json. Writes bench.json.
"""

import ast
import contextlib
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from benchmark.baseline import SEED, _build, _measure, make_inputs  # noqa: E402
from kernels.builder import run_from_config as build_kernel  # noqa: E402
from kernels.scaffold import scaffold  # noqa: E402
from task.load import load_task  # noqa: E402

_KERNEL_ALLOWED_IMPORTS = frozenset(
    {
        "torch",
        "triton",
        "math",
        "typing",
        "__future__",
        "dataclasses",
        "functools",
        "itertools",
        "operator",
        "collections",
    }
)


def _top_imports(src: str) -> set[str]:
    """Top-level module names imported anywhere in src (incl. inside
    functions — defeats a lazy import that dodges a build-time check)."""
    mods: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    return mods


def _load_kernel(kernel_py: Path):
    spec = importlib.util.spec_from_file_location("_kernel_src", kernel_py)
    if not spec or not spec.loader:
        raise ImportError(f"cannot load spec for {kernel_py}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.kernel


@contextlib.contextmanager
def _triton_counter():
    """Count @triton.jit launches and the largest tensor any launch
    touched. max_numel lets bench reject a no-op fig-leaf kernel (a
    tiny launch on a throwaway tensor) used only to satisfy n>0 while
    the real work is torch passthrough."""
    from triton.runtime.jit import JITFunction

    box = {"n": 0, "max_numel": 0}
    orig = JITFunction.run

    def run(self, *a, **k):
        box["n"] += 1
        for v in (*a, *k.values()):
            if torch.is_tensor(v):
                box["max_numel"] = max(box["max_numel"], v.numel())
        return orig(self, *a, **k)

    JITFunction.run = run
    try:
        yield box
    finally:
        JITFunction.run = orig


@contextlib.contextmanager
def _autocast_tripwire():
    """Trip if the kernel runs under reduced-precision CUDA autocast.
    The frozen baseline is measured in the reference's native dtype
    (fp32); a kernel that wraps the reference model in bf16/fp16
    autocast 'beats' it on precision, not on a real kernel — an
    apples-to-oranges speedup. Caught here, not via output dtype (the
    cheat upcasts the result back to fp32 on return)."""
    box = {"tripped": False, "dtype": None}
    orig = torch.autocast.__enter__

    def enter(self):
        if getattr(self, "device", None) == "cuda" and getattr(self, "fast_dtype", None) in (
            torch.bfloat16,
            torch.float16,
        ):
            box["tripped"] = True
            box["dtype"] = str(self.fast_dtype)
        return orig(self)

    torch.autocast.__enter__ = enter
    try:
        yield box
    finally:
        torch.autocast.__enter__ = orig


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

    try:
        kernel_py = scaffold(config_path)
    except FileNotFoundError:
        return fail("no_kernel", detail=f"missing kernel.py in {cfg_path.parent}")

    task = load_task(t["level"], t["problem_id"])

    ref_deps = _top_imports(task.code) - _KERNEL_ALLOWED_IMPORTS
    try:
        kern_imports = _top_imports(kernel_py.read_text())
    except SyntaxError:
        kern_imports = set()
    leaked = sorted(ref_deps & kern_imports)
    if leaked:
        return fail(
            "reference_import",
            detail=f"kernel.py imports reference dependency {leaked}; reproduce the "
            "computation in Triton — importing or running the reference model is a cheat",
        )

    model, inputs_a = _build(task, "cuda")
    inputs_b = make_inputs(task, "cuda", SEED + 1)
    c = cfg["correctness"]
    rtol, atol = float(c["rtol"]), float(c["atol"])

    def close(g, r):
        return torch.allclose(g.float(), r.float(), rtol=rtol, atol=atol)

    try:
        kern = _load_kernel(kernel_py)
        with torch.no_grad():
            ref_a, ref_b = model(*inputs_a), model(*inputs_b)
            with _triton_counter() as tc, _autocast_tripwire() as ac:
                got_a = kern(*inputs_a)
            got_a2 = kern(*inputs_a)
            got_b = kern(*inputs_b)
        max_diff = max(
            (got_a.float() - ref_a.float()).abs().max().item(),
            (got_b.float() - ref_b.float()).abs().max().item(),
        )
    except Exception as exc:
        return fail("kernel_exception", detail=repr(exc))

    if not (close(got_a, ref_a) and close(got_b, ref_b)):
        return fail("numeric_mismatch", max_abs_diff=round(max_diff, 6))
    if not close(got_a, got_a2):
        return fail("nondeterministic", detail="same input gave different output")
    if tc["n"] == 0:
        return fail("no_triton", detail="no @triton.jit kernel launched")
    if tc["max_numel"] < ref_a.numel() // 4:
        return fail(
            "triton_figleaf",
            detail=f"largest triton tensor {tc['max_numel']} elems vs output "
            f"{ref_a.numel()}; launch does not touch output-scale data",
        )
    if ac["tripped"]:
        return fail(
            "precision_cheat",
            detail=f"kernel ran under cuda {ac['dtype']} autocast; baseline is fp32",
        )

    b = cfg["benchmark"]
    with torch.no_grad():
        kernel_ms = _measure(kern, inputs_b, b["warmup"], b["iters"]).time_ms
    speedup = compile_ms / kernel_ms
    bar = float(cfg["perf"]["min_speedup_vs_compile"])
    rec.update(
        max_abs_diff=round(max_diff, 6),
        kernel_ms=round(kernel_ms, 5),
        compile_ms=compile_ms,
        speedup_vs_compile=round(speedup, 4),
        min_speedup=bar,
        triton_launches=tc["n"],
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

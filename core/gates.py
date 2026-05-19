"""BENCHMARK stage: the only source of truth. Deterministic, no LLM.

Two gates, run in order:
  correctness  5 stages vs golden.pt (rtol/atol from config). The one
               required try/except — a bad agent kernel must become
               error_class data, not crash the loop.
  perf         median CUDA-event time, custom kernel vs
               torch.compile(max-autotune). Pass iff speedup >= bar.

No build gate (kernel_lib/Nix removed) and no ssh/Spark: a kernel is just
a Python module exposing kernel(*inputs); Triton JITs in-process.
"""
from __future__ import annotations

import importlib.util
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)
    error_class: str | None = None


@dataclass
class GateCtx:
    block_dir: Path
    kernel: object
    cfg: dict


def correctness(ctx: GateCtx) -> GateResult:
    import torch

    c = ctx.cfg["gates"]["correctness"]
    rtol, atol = float(c["rtol"]), float(c["atol"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    to_dev = lambda x: x.to(dev) if torch.is_tensor(x) else x
    inputs = torch.load(ctx.block_dir / "inputs.pt")
    golden = to_dev(torch.load(ctx.block_dir / "golden.pt"))
    if not isinstance(inputs, (list, tuple)):
        inputs = (inputs,)
    inputs = tuple(to_dev(x) for x in inputs)
    k = ctx.kernel
    close = lambda a, b: torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol)
    res = {}
    try:
        if not close(k(*inputs), golden):
            return GateResult("correctness", False, {"stage": "smoke"}, "numeric_mismatch")
        res["smoke"] = "pass"
        n = int(c.get("shape_sweep_min_configs", 3))
        for i in range(n):
            k(*(t.repeat(*([2] + [1] * (t.dim() - 1))) if (i % 2 and t.dim()) else t
                for t in inputs))
        res["shape_sweep"] = f"pass ({n})"
        for scale in (1e-4, 1e4):
            o = k(*(t * scale for t in inputs))
            if torch.isnan(o).any() or torch.isinf(o).any():
                return GateResult("correctness", False,
                                  {"stage": "adversarial_numerics", **res},
                                  "numeric_instability")
        res["adversarial_numerics"] = "pass"
        if not torch.equal(k(*inputs), k(*inputs)):
            return GateResult("correctness", False, {"stage": "determinism", **res},
                              "nondeterministic")
        res["determinism"] = "pass"
        for fill in (0.0, 1.0):
            if torch.isnan(k(*(torch.full_like(t, fill) for t in inputs))).any():
                return GateResult("correctness", False,
                                  {"stage": "edge_cases", **res}, "edge_case_nan")
        res["edge_cases"] = "pass"
    except Exception as exc:
        return GateResult("correctness", False, {"exception": repr(exc), **res},
                          "kernel_exception")
    return GateResult("correctness", True, {"stages": res})


def perf(ctx: GateCtx) -> GateResult:
    import torch

    p = ctx.cfg["gates"]["perf"]
    bar = float(p["min_speedup_vs_compile"])
    spec = importlib.util.spec_from_file_location("frozen_ref",
                                                  ctx.block_dir / "reference.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    block = ref.build_block()
    a, kw = ref.call_parts()
    inputs = ref.frozen_inputs()
    w, n = int(p["warmup_iters"]), int(p["timed_iters"])

    def med(fn, args, kwargs):
        for _ in range(w):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn(*args, **kwargs)
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        return statistics.median(ts)

    with torch.no_grad():
        compile_ms = med(torch.compile(block, mode=ctx.cfg["baseline"]["mode"],
                                       fullgraph=False), a, kw)
        custom_ms = med(ctx.kernel, inputs, {})
    sp = compile_ms / custom_ms
    passed = sp >= bar
    return GateResult("perf", passed,
                      {"compile_ms": round(compile_ms, 5),
                       "custom_ms": round(custom_ms, 5),
                       "speedup_vs_compile": round(sp, 4), "min": bar},
                      None if passed else "slower_than_compile")


GATES = {"correctness": correctness, "perf": perf}

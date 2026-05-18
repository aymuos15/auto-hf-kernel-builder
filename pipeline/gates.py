from __future__ import annotations

import configparser
import importlib.util
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from execwrap import _ssh, rsync, rsync_dest, run

REPO = Path(__file__).resolve().parent.parent


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)
    error_class: str | None = None


@dataclass
class GateCtx:
    block_dir: Path
    kernel_project: Path
    kernel: object
    cfg: dict


def _kernel_name(kernel_project):
    cp = configparser.ConfigParser()
    cp.read_string((kernel_project / "build.toml").read_text())
    return cp["general"]["name"].strip().strip('"')


def correctness(ctx):
    import torch

    c = ctx.cfg["gates"]["correctness"]
    rtol, atol = float(c["rtol"]), float(c["atol"])
    inputs = torch.load(ctx.block_dir / "inputs.pt")
    golden = torch.load(ctx.block_dir / "golden.pt")
    if not isinstance(inputs, (list, tuple)):
        inputs = (inputs,)
    k = ctx.kernel
    close = lambda a, b: torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol)
    res = {}
    # The ONE required try: a bad agent kernel must become error_class data, not crash the loop — that is this gate's whole purpose.
    try:
        if not close(k(*inputs), golden):
            return GateResult("correctness", False, {"stage": "smoke"}, "numeric_mismatch")
        res["smoke"] = "pass"
        n = int(c.get("shape_sweep_min_configs", 10))
        for i in range(n):
            k(*(t.repeat(*([2] + [1] * (t.dim() - 1))) if (i % 2 and t.dim()) else t for t in inputs))
        res["shape_sweep"] = f"pass ({n})"
        for scale in (1e-4, 1e4):
            o = k(*(t * scale for t in inputs))
            if torch.isnan(o).any() or torch.isinf(o).any():
                return GateResult("correctness", False, {"stage": "adversarial_numerics", **res}, "numeric_instability")
        res["adversarial_numerics"] = "pass"
        if not torch.equal(k(*inputs), k(*inputs)):
            return GateResult("correctness", False, {"stage": "determinism", **res}, "nondeterministic")
        res["determinism"] = "pass"
        for fill in (0.0, 1.0):
            if torch.isnan(k(*(torch.full_like(t, fill) for t in inputs))).any():
                return GateResult("correctness", False, {"stage": "edge_cases", **res}, "edge_case_nan")
        res["edge_cases"] = "pass"
    except Exception as exc:
        return GateResult("correctness", False, {"exception": repr(exc), **res}, "kernel_exception")
    return GateResult("correctness", True, {"stages": res})


def build(ctx):
    name = _kernel_name(ctx.kernel_project)
    attr = ctx.cfg["gates"]["build"].get("nix_build_attr", ".#bundle")
    rel = ctx.kernel_project.resolve().relative_to(REPO)
    ssh = _ssh(ctx.cfg)
    rroot = rsync_dest(ctx.cfg)
    if ssh:
        host = ssh.split()[-1]
        rsync(f"{ctx.kernel_project}/", f"{host}:{rroot}/{rel}/", timeout=300)
    cmd = f"set -euo pipefail; cd {rroot}/{rel} && rm -rf result build && nix --extra-experimental-features 'nix-command flakes' build --accept-flake-config {attr} -o result && cp -rL result build"
    rc, _, err = run(cmd, ctx.cfg, surface="host", timeout=3600, tee=True)
    if rc:
        return GateResult("build", False, {"stderr_tail": err[-1800:]}, "nix_build_failed")
    wd = ctx.cfg.get("exec", {}).get("workdir", "/work")
    rc, _, _ = run(f"python3 {wd}/pipeline/_import_probe.py {wd}/{rel}/build/torch-universal {name}", ctx.cfg, surface="container", timeout=600, tee=True)
    return GateResult("build", True, {"name": name, "nix_attr": attr}) if not rc else GateResult("build", False, {}, "import_failed")


def perf(ctx):
    import torch

    p = ctx.cfg["gates"]["perf"]
    bar = float(p["min_speedup_vs_compile"])
    spec = importlib.util.spec_from_file_location("frozen_ref", ctx.block_dir / "reference.py")
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
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn(*args, **kwargs)
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        return statistics.median(ts)

    with torch.no_grad():
        compile_ms = med(torch.compile(block, mode=ctx.cfg["baseline"]["mode"], fullgraph=False), a, kw)
        custom_ms = med(ctx.kernel, inputs, {})
    sp = compile_ms / custom_ms
    passed = sp >= bar
    return GateResult("perf", passed, {"compile_ms": round(compile_ms, 5), "custom_ms": round(custom_ms, 5), "speedup_vs_compile": round(sp, 4), "min": bar}, None if passed else "slower_than_compile")


GATES = {"correctness": correctness, "build": build, "perf": perf}

"""DETERMINISTIC prepare stage (no LLM): Task + Environment + Config.

Task:        a KernelBench problem -> an immutable frozen contract.
Environment: detect the local device/toolchain.
Config:      resolve configs/kernelbench.yaml for this run.

This is the deterministic spine. Long-term an agentic task-maker (latest
HF model -> KernelBench-like task) can replace make_contract() while
emitting the SAME contract — nothing downstream changes (the contract is
the narrow waist).

CLI:  python3 core/prepare.py --level 3 --problem 4 [--force] [--no-baseline]
"""
from __future__ import annotations

import argparse
import json
import statistics
import types
from pathlib import Path

import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
TASKS = REPO / "tasks"
SEED = 0

REFERENCE_PY = '''"""Auto-generated frozen-contract shim (immutable).
Regenerate via core/prepare.py. The whole KernelBench Model IS the block;
weights are frozen in weights.pt (KernelBench inits them randomly, so
golden is otherwise irreproducible)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _src():
    spec = importlib.util.spec_from_file_location("kb_src", _HERE / "model_src.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_block(config_path=None):
    """Frozen Model (eval, on device). Built on CPU then moved once so
    peak device memory holds a single copy (big MLPs bust small GPUs)."""
    m = _src()
    model = m.Model(*m.get_init_inputs()).eval()
    model.load_state_dict(torch.load(_HERE / "weights.pt", map_location="cpu"))
    return model.to(_DEV)


def frozen_inputs():
    xs = torch.load(_HERE / "inputs.pt", map_location=_DEV)
    return [x.to(_DEV) if torch.is_tensor(x) else x for x in xs]


def golden():
    return torch.load(_HERE / "golden.pt", map_location=_DEV)


def call_parts(inputs=None):
    """KernelBench Models take all inputs positionally -> (args, {})."""
    xs = frozen_inputs() if inputs is None else inputs
    return tuple(xs), {}


def block_callable(config_path=None):
    mod = build_block(config_path)

    def f(*xs):
        return mod(*xs)

    return f
'''


def detect_env() -> dict:
    cuda = torch.cuda.is_available()
    return {
        "device": "cuda" if cuda else "cpu",
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "compute_capability": (".".join(map(str, torch.cuda.get_device_capability(0)))
                               if cuda else None),
        "torch": torch.__version__,
    }


def resolve_config() -> dict:
    return yaml.safe_load((REPO / "configs" / "kernelbench.yaml").read_text())


def _load_row(level: int, problem_id: int):
    df = pd.read_parquet(DATA / f"level_{level}.parquet")
    sel = df[df.problem_id == problem_id]
    if sel.empty:
        raise SystemExit(f"no problem_id={problem_id} in level_{level}")
    return sel.iloc[0]


def _median_ms(fn, args, warmup=10, iters=50) -> float:
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn(*args)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def _measure_baseline(model, inputs) -> dict:
    try:
        with torch.no_grad():
            eager = _median_ms(lambda *a: model(*a), inputs)
            comp_fn = torch.compile(model, mode="max-autotune", fullgraph=False)
            comp = _median_ms(lambda *a: comp_fn(*a), inputs)
        sp = eager / comp
        verdict = ("high_headroom" if sp >= 1.2
                   else "moderate_headroom" if sp >= 1.05
                   else "low_headroom (compile already near-optimal)")
        return {"eager_ms": round(eager, 5), "compile_ms": round(comp, 5),
                "compile_speedup_vs_eager": round(sp, 4),
                "baseline_mode": "max-autotune", "verdict": verdict}
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        return {"verdict": "oom_local", "note": f"baseline skipped: {exc!r}"}


def task_dir(level: int, name: str) -> Path:
    return TASKS / f"kernelbench__L{level}" / f"L{level}_{name}"


def make_contract(level: int, problem_id: int, baseline=True,
                  force=False) -> Path:
    row = _load_row(level, problem_id)
    name = row["name"]
    out = task_dir(level, name)
    if (out / "contract.json").is_file() and not force:
        print(f"contract exists: {out} (use --force to rebuild)")
        return out
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(SEED)
    if dev == "cuda":
        torch.cuda.manual_seed_all(SEED)
    src = types.ModuleType("kb_src")
    exec(compile(row["code"], "kb_src", "exec"), src.__dict__)
    init = [x.to(dev) if torch.is_tensor(x) else x for x in src.get_init_inputs()]
    model = src.Model(*init).to(dev).eval()
    inputs = [x.to(dev) if torch.is_tensor(x) else x for x in src.get_inputs()]
    with torch.no_grad():
        gold = model(*inputs)

    (out / "model_src.py").write_text(row["code"])
    (out / "reference.py").write_text(REFERENCE_PY)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / "weights.pt")
    torch.save([x.cpu() if torch.is_tensor(x) else x for x in inputs], out / "inputs.pt")
    torch.save(gold.cpu(), out / "golden.pt")

    sig = lambda t: ([list(t.shape), str(t.dtype)] if torch.is_tensor(t)
                     else [None, type(t).__name__])
    (out / "contract.json").write_text(json.dumps({
        "source": "KernelBench", "level": level,
        "problem_id": int(row["problem_id"]), "name": name,
        "block_class": "Model",
        "input_order": [f"arg{i}" for i in range(len(inputs))],
        "input_sig": [sig(x) for x in inputs], "output_sig": sig(gold),
        "weights": "weights.pt", "seed": SEED,
        "note": "Whole KernelBench Model is the block. kernel(*inputs) is "
                "positional in input_order and must reproduce golden.pt "
                "within config rtol/atol; frozen weights are in weights.pt.",
    }, indent=2))

    bj = {"source": "KernelBench", "level": level,
          "problem_id": int(row["problem_id"]), "name": name,
          "env": detect_env(), "dtype": str(gold.dtype)}
    bj.update(_measure_baseline(model, inputs) if (baseline and dev == "cuda")
              else {"verdict": "not_measured"})
    (out / "baseline.json").write_text(json.dumps(bj, indent=2))

    print(f"froze {out}  inputs={[sig(x) for x in inputs]} -> {sig(gold)}  "
          f"baseline={bj.get('verdict')}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--problem", type=int, required=True)
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    make_contract(args.level, args.problem, not args.no_baseline, args.force)


if __name__ == "__main__":
    main()

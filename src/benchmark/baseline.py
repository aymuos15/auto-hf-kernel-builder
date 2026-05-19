"""Phase 3: baseline measurement — GPU time + peak memory for eager and
torch.compile.

Config-driven: the only input is a per-task config.json (Phase 2). It
carries the task identity and the benchmark knobs; results are written
to results.json next to it (inputs vs outputs stay separate).
Deterministic: seeds before constructing the Model / inputs.
"""

import json
import statistics
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task.load import Task, load_task  # noqa: E402

SEED = 0


@dataclass
class Measure:
    time_ms: float
    peak_mem_mb: float


@dataclass
class Baseline:
    task: str
    device: str
    eager: Measure
    compile: Measure

    @property
    def compile_speedup(self) -> float:
        return self.eager.time_ms / self.compile.time_ms


def _build(task: Task, device: str):
    """Seeded Model + inputs from the task code (KernelBench convention:
    Model, get_inputs(), get_init_inputs())."""
    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)
    mod = types.ModuleType("kb_task")
    exec(compile(task.code, task.name, "exec"), mod.__dict__)
    init = [x.to(device) if torch.is_tensor(x) else x for x in mod.get_init_inputs()]
    model = mod.Model(*init).to(device).eval()
    inputs = [x.to(device) if torch.is_tensor(x) else x for x in mod.get_inputs()]
    return model, inputs


def _measure(fn, inputs, warmup: int, iters: int) -> Measure:
    for _ in range(warmup):
        fn(*inputs)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn(*inputs)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    peak = torch.cuda.max_memory_allocated() / 1024**2
    return Measure(time_ms=round(statistics.median(ts), 5), peak_mem_mb=round(peak, 2))


def measure_baseline(
    level: int,
    problem_id: int,
    warmup: int = 10,
    iters: int = 50,
    compile_mode: str = "max-autotune",
) -> Baseline:
    assert torch.cuda.is_available(), "CUDA required for benchmarking"
    device = "cuda"
    task = load_task(level, problem_id)
    model, inputs = _build(task, device)
    with torch.no_grad():
        eager = _measure(lambda *a: model(*a), inputs, warmup, iters)
        compiled = torch.compile(model, mode=compile_mode, fullgraph=False)
        comp = _measure(lambda *a: compiled(*a), inputs, warmup, iters)
    return Baseline(task=task.name, device=torch.cuda.get_device_name(0), eager=eager, compile=comp)


def run_from_config(config_path: str) -> Path:
    """Phase 3 entrypoint: read config.json, benchmark, write results.json
    in the same task dir."""
    cfg_path = Path(config_path).resolve()
    cfg = json.loads(cfg_path.read_text())
    t, b = cfg["task"], cfg["benchmark"]
    base = measure_baseline(t["level"], t["problem_id"], b["warmup"], b["iters"], b["compile_mode"])
    results = {
        "task": t,
        "device": base.device,
        "baseline": {
            "eager": asdict(base.eager),
            "compile": asdict(base.compile),
            "compile_speedup": round(base.compile_speedup, 4),
        },
    }
    out = cfg_path.parent / "res.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")
    print(
        f"  eager   {base.eager.time_ms} ms / {base.eager.peak_mem_mb} MB"
        f"   compile {base.compile.time_ms} ms / {base.compile.peak_mem_mb} MB"
        f"   ({base.compile_speedup:.4f}x)"
    )
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to a task config.json")
    args = ap.parse_args()
    run_from_config(args.config)

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import model_slug  # noqa: E402


def _time_ms(fn, a, kw, warmup, iters):
    import torch

    for _ in range(warmup):
        fn(*a, **kw)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn(*a, **kw)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    import torch

    cfg = yaml.safe_load(args.config.read_text())
    slug = model_slug(cfg["model"]["id"])
    sel = json.loads(Path(f"targets/{slug}/selection.json").read_text())
    cls = sel["winner_class"]
    block_dir = Path(f"targets/{slug}/{cls}")
    spec = importlib.util.spec_from_file_location("frozen_ref", block_dir / "reference.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    block = ref.build_block()
    a, kw = ref.call_parts()
    with torch.no_grad():
        eager_ms = _time_ms(block, a, kw, args.warmup, args.iters)
        compile_ms = _time_ms(torch.compile(block, mode=cfg["baseline"]["mode"],
                              fullgraph=False), a, kw, args.warmup, args.iters)
    sp = eager_ms / compile_ms
    verdict = ("clear_headroom" if sp < 1.10 else
               "moderate_headroom" if sp < 1.5 else "compile_already_strong")
    (block_dir / "baseline.json").write_text(json.dumps({
        "model": cfg["model"]["id"], "block_class": cls,
        "device": torch.cuda.get_device_name(0),
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "eager_ms": round(eager_ms, 5), "compile_ms": round(compile_ms, 5),
        "compile_speedup_vs_eager": round(sp, 4),
        "baseline_mode": cfg["baseline"]["mode"], "verdict": verdict}, indent=2))
    print(f"eager={eager_ms:.4f}ms compile={compile_ms:.4f}ms (x{sp:.2f}) -> {verdict}")


if __name__ == "__main__":
    main()

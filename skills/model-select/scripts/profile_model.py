from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_model_and_inputs, model_slug, run_forward  # noqa: E402


def _module_timings(model, inputs):
    import torch

    name_of = {id(m): n for n, m in model.named_modules()}
    cls_of = {id(m): type(m).__name__ for _, m in model.named_modules()}
    builtin_of = {id(m): type(m).__module__.startswith("torch.nn") and not any(m.children())
                  for _, m in model.named_modules()}
    events, totals, calls, handles = {}, {}, {}, []

    def pre(mod, args, kwargs=None):
        s = torch.cuda.Event(enable_timing=True)
        s.record()
        events.setdefault(id(mod), []).append(s)

    def post(mod, args, output):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        st = events[id(mod)].pop()
        events[id(mod)].append((st, e))
        calls[id(mod)] = calls.get(id(mod), 0) + 1

    for _, m in model.named_modules():
        if m is model:
            continue
        handles += [m.register_forward_pre_hook(pre, with_kwargs=True),
                    m.register_forward_hook(post)]
    run_forward(model, inputs)
    torch.cuda.synchronize()
    for mid, evs in events.items():
        totals[mid] = sum(p[0].elapsed_time(p[1]) for p in evs if isinstance(p, tuple))
    for h in handles:
        h.remove()
    by_cls = {}
    for mid, ms in totals.items():
        agg = by_cls.setdefault(cls_of[mid], {"incl_ms": 0.0, "instances": 0,
                                "calls": 0, "example": None, "primitive_op": builtin_of[mid]})
        agg["incl_ms"] += ms
        agg["instances"] += 1
        agg["calls"] += calls.get(mid, 0)
        if agg["example"] is None and name_of.get(mid):
            agg["example"] = name_of[mid]
    return by_cls


def _op_timings(model, inputs):
    import torch
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        run_forward(model, inputs)
    torch.cuda.synchronize()
    rows = [{"name": e.key, "device_time_us": float(getattr(e, "self_device_time_total", 0.0)),
             "cpu_time_us": float(e.self_cpu_time_total), "count": int(e.count),
             "input_shapes": str(e.input_shapes)}
            for e in prof.key_averages(group_by_input_shape=True)]
    k = "device_time_us" if any(r["device_time_us"] for r in rows) else "cpu_time_us"
    rows.sort(key=lambda r: r[k], reverse=True)
    return rows, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()
    import torch

    cfg = yaml.safe_load(args.config.read_text())
    out = Path(f"targets/{model_slug(cfg['model']['id'])}/profile.json")
    model, inputs, _ = load_model_and_inputs(cfg)
    for _ in range(args.warmup):
        run_forward(model, inputs)
    torch.cuda.synchronize()
    modules = _module_timings(model, inputs)
    ops, opkey = _op_timings(model, inputs)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": cfg["model"]["id"], "root_class": type(model).__name__,
                   "modules_by_class": modules, "ops_ranked_by": opkey, "ops": ops}, indent=2))
    print(f"wrote {out} ({len(modules)} module classes, {len(ops)} ops); root={type(model).__name__}")


if __name__ == "__main__":
    main()

"""Harness-independent legality + VRAM probe for a KernelBench task module.

    python3 preflight.py path/to/module.py --vram-budget-gb 3.6

Execs the module, builds it under a fixed seed exactly as a harness
would (construct -> .to(device).eval() -> get_inputs), then asserts the
bench invariants: single tensor returned, deterministic across two
identical runs, output changes for a different input seed, and peak
VRAM within budget. Exit code 0 only if all pass.
"""

import argparse
import sys
import types
from pathlib import Path

import torch

SEED = 0


def _exec(path: Path) -> types.ModuleType:
    mod = types.ModuleType("kb_task")
    exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)
    return mod


def _seed(device: str, seed: int) -> None:
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


def _to(xs, device):
    return [x.to(device) if torch.is_tensor(x) else x for x in xs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("module", type=Path)
    ap.add_argument("--vram-budget-gb", type=float, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = args.device
    mod = _exec(args.module)

    _seed(dev, SEED)
    model = mod.Model(*_to(mod.get_init_inputs(), dev)).to(dev).eval()
    inputs_a = _to(mod.get_inputs(), dev)
    _seed(dev, SEED + 1)
    inputs_b = _to(mod.get_inputs(), dev)

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out_a = model(*inputs_a)
        out_a2 = model(*inputs_a)
        out_b = model(*inputs_b)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0

    fails = []
    if not torch.is_tensor(out_a):
        fails.append(f"forward must return ONE tensor, got {type(out_a).__name__}")
    else:
        if not torch.allclose(out_a.float(), out_a2.float()):
            fails.append("nondeterministic: same input gave different output")
        if torch.allclose(out_a.float(), out_b.float()):
            fails.append("output does not depend on inputs (constant return)")
    if args.vram_budget_gb is not None and peak_gb > args.vram_budget_gb:
        fails.append(f"peak VRAM {peak_gb:.3f} GB exceeds budget {args.vram_budget_gb} GB")

    shape = tuple(out_a.shape) if torch.is_tensor(out_a) else None
    print(f"device={dev} out_shape={shape} peak_vram={peak_gb:.3f}GB")
    if fails:
        for f in fails:
            print(f"FAIL: {f}")
        return 1
    print("PASS: benchmark-legal")
    return 0


if __name__ == "__main__":
    sys.exit(main())

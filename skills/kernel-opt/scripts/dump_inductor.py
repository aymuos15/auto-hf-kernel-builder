from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (args.block_dir / "inductor")
    import torch

    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_LOGS", "output_code")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(out / "cache"))
    spec = importlib.util.spec_from_file_location("frozen_ref", args.block_dir / "reference.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    block = ref.build_block()
    a, kw = ref.call_parts()
    compiled = torch.compile(block, mode="max-autotune", fullgraph=False)
    with torch.no_grad():
        for _ in range(3):
            compiled(*a, **kw)
        torch.cuda.synchronize()
    (out / "WHERE.txt").write_text(
        f"Inductor output_code is in TORCH_LOGS and {out / 'cache'}; "
        "this is the torch.compile(max-autotune) baseline to beat.\n")
    print(f"inductor study material -> {out}")


if __name__ == "__main__":
    main()

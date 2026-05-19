"""Writes the bare agent seam: tasks/<slug>/<block>/kernel.py.

No flake.nix / build.toml (kernel_lib removed). A kernel is just this
module exposing kernel(*inputs). Idempotent: never clobbers an
agent-filled kernel unless overwrite=True.
"""
from __future__ import annotations

import json
from pathlib import Path

SEAM = '''\
"""AGENT SEAM. input_order={order} input_sig={sig} output_sig={osig}.

Write a real @triton.jit kernel below. kernel(*inputs) is called
positionally in input_order and receives ONLY the inputs. The frozen
Model weights are pre-loaded as WEIGHTS (a state_dict) from the sibling
weights.pt — use them. It must reproduce golden.pt within config
rtol/atol AND beat torch.compile(max-autotune).
"""
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
WEIGHTS = torch.load(_HERE / "weights.pt",
                     map_location="cuda" if torch.cuda.is_available() else "cpu")


def kernel(*inputs):
    raise NotImplementedError("AGENT: implement Triton kernel; inputs order {order}")
'''


def scaffold(block_dir: Path, overwrite: bool = False) -> Path:
    contract = json.loads((block_dir / "contract.json").read_text())
    kp = block_dir / "kernel.py"
    if kp.exists() and not overwrite:
        print(f"kernel.py exists at {kp} (kept; overwrite=True to reset)")
        return kp
    kp.write_text(SEAM.format(order=contract["input_order"],
                              sig=contract["input_sig"],
                              osig=contract["output_sig"]))
    print(f"scaffolded {kp}  (AGENT SEAM)")
    return kp

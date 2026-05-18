from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model-select" / "scripts"))
from _common import model_slug  # noqa: E402

KB_REV = "b4accba4496b28faef19a0487fbcf9686b14e2ef"

FLAKE = f'''{{
  description = "universal Triton kernel";
  inputs.kernel-builder.url = "github:huggingface/kernel-builder/{KB_REV}";
  outputs = {{ self, kernel-builder }}:
    kernel-builder.lib.genFlakeOutputs {{ path = ./.; rev = self.shortRev or self.dirtyShortRev or "dev0"; doGetKernelCheck = false; }};
}}
'''

INIT_TMPL = '''\
"""AGENT SEAM. input_order={order} input_sig={sig} output_sig={osig}. Replace _triton_impl with a real @triton.jit kernel reproducing golden.pt and beating torch.compile; kernel(*inputs) is positional in input_order."""
import torch


def _triton_impl(*inputs):
    raise NotImplementedError("AGENT: implement Triton kernel; inputs order {order}")


def kernel(*inputs):
    return _triton_impl(*inputs)
'''


def ident(s):
    s = re.sub(r"[^0-9a-zA-Z_]", "_", s).strip("_").lower()
    return s if s and not s[0].isdigit() else f"k_{s}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    slug = model_slug(cfg["model"]["id"])
    sel = json.loads(Path(f"targets/{slug}/selection.json").read_text())
    block = sel["winner_class"]
    contract = json.loads(Path(f"targets/{slug}/{block}/contract.json").read_text())
    name = ident(f"{slug}_{block}")
    proj = Path(f"targets/{slug}/{block}/kernel")
    init_py = proj / "torch-ext" / name / "__init__.py"
    # Idempotency is REQUIRED: must never clobber an agent-filled kernel.
    if init_py.exists() and not args.overwrite:
        print(f"kernel project exists at {proj} (kept; --overwrite to reset)")
    else:
        (proj / "torch-ext" / name).mkdir(parents=True, exist_ok=True)
        init_py.write_text(INIT_TMPL.format(order=contract["input_order"],
                           sig=contract["input_sig"], osig=contract["output_sig"]))
    (proj / "build.toml").write_text(f'[general]\nname = "{name}"\nuniversal = true\n')
    (proj / "flake.nix").write_text(FLAKE)
    (proj / ".gitignore").write_text("build/\nresult\nflake.lock\n")
    print(f"scaffolded {proj} (name={name}); AGENT SEAM: fill {init_py}")


if __name__ == "__main__":
    main()

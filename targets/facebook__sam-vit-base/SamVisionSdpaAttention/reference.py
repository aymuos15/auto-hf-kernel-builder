"""Importable frozen-block factory (auto-generated, immutable).

Reconstructs the exact frozen nn.Module + its real inputs from the sibling
contract.json and config.yaml, so torch.compile / the gates can run it
uniformly. Do not edit; regenerate via freeze_contract.py.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]                      # targets/<slug>/<Class>/ -> repo
sys.path.insert(0, str(_REPO / "skills" / "model-select" / "scripts"))
from _common import load_model_and_inputs     # noqa: E402
import yaml                                    # noqa: E402

_C = json.loads((_HERE / "contract.json").read_text())


def _primary(o):
    if isinstance(o, torch.Tensor):
        return o
    if hasattr(o, "last_hidden_state"):
        return o.last_hidden_state
    if isinstance(o, (list, tuple)):
        for x in o:
            t = _primary(x)
            if t is not None:
                return t
    if hasattr(o, "to_tuple"):
        return _primary(o.to_tuple())
    return None


def build_block(config_path=None):
    """The exact frozen nn.Module (eval), from the configured model + the
    contract's module path."""
    cfg_path = Path(config_path) if config_path else _REPO / "configs" / "config.yaml"
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    model, _, _ = load_model_and_inputs(cfg)
    return model.get_submodule(_C["module_path"]).eval()


def frozen_inputs():
    """Captured real input tensors, in contract input_order."""
    return torch.load(_HERE / "inputs.pt")


def golden():
    return torch.load(_HERE / "golden.pt")


def call_parts(inputs=None):
    """Split frozen inputs into (args, kwargs) per contract input_order so
    the block can be called natively (clean for torch.compile)."""
    xs = frozen_inputs() if inputs is None else inputs
    a, kw = [], {}
    for name, val in zip(_C["input_order"], xs):
        if re.fullmatch(r"arg\d+", name):
            a.append(val)
        else:
            kw[name] = val
    return tuple(a), kw


def block_callable(config_path=None):
    """Positional callable f(*inputs) -> primary output tensor (maps inputs
    to the native call per contract). Used by the correctness gate."""
    mod = build_block(config_path)
    order = _C["input_order"]

    def f(*xs):
        a, kw = [], {}
        for name, val in zip(order, xs):
            (a.append(val) if re.fullmatch(r"arg\d+", name)
             else kw.__setitem__(name, val))
        return _primary(mod(*a, **kw))

    return f

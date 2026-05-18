from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_model_and_inputs, model_slug, run_forward  # noqa: E402

_REFERENCE_PY = '''\
from __future__ import annotations
import json, re, sys
from pathlib import Path
import torch
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "skills" / "model-select" / "scripts"))
from _common import load_model_and_inputs  # noqa: E402
import yaml  # noqa: E402
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
    cfg = yaml.safe_load(Path(config_path or _REPO / "configs" / "config.yaml").read_text())
    model, _, _ = load_model_and_inputs(cfg)
    return model.get_submodule(_C["module_path"]).eval()


def frozen_inputs():
    return torch.load(_HERE / "inputs.pt")


def golden():
    return torch.load(_HERE / "golden.pt")


def call_parts(inputs=None):
    a, kw = [], {}
    for name, val in zip(_C["input_order"], frozen_inputs() if inputs is None else inputs):
        (a.append(val) if re.fullmatch(r"arg\\d+", name) else kw.__setitem__(name, val))
    return tuple(a), kw


def block_callable(config_path=None):
    mod = build_block(config_path)
    order = _C["input_order"]

    def f(*xs):
        a, kw = [], {}
        for name, val in zip(order, xs):
            (a.append(val) if re.fullmatch(r"arg\\d+", name) else kw.__setitem__(name, val))
        return _primary(mod(*a, **kw))

    return f
'''


def _primary_tensor(o):
    import torch

    if isinstance(o, torch.Tensor):
        return o
    if hasattr(o, "last_hidden_state"):
        return o.last_hidden_state
    if isinstance(o, (list, tuple)):
        for x in o:
            t = _primary_tensor(x)
            if t is not None:
                return t
    if hasattr(o, "to_tuple"):
        return _primary_tensor(o.to_tuple())
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    ap.add_argument("--path", default=None)
    args = ap.parse_args()
    import torch

    cfg = yaml.safe_load(args.config.read_text())
    slug = model_slug(cfg["model"]["id"])
    sel = json.loads(Path(f"targets/{slug}/selection.json").read_text())
    path = args.path or sel["winner_example_path"]
    model, inputs, _ = load_model_and_inputs(cfg)
    target = model.get_submodule(path)
    cls = type(target).__name__
    outdir = Path(f"targets/{slug}/{cls}")
    outdir.mkdir(parents=True, exist_ok=True)
    cap = {}

    def pre(mod, a, kw):
        if "order" in cap:
            return
        order, tensors, nontensor = [], [], []
        for i, x in enumerate(a):
            (order.append(f"arg{i}"), tensors.append(x.detach().clone())) if isinstance(x, torch.Tensor) else nontensor.append((f"arg{i}", str(type(x))))
        for k in sorted((kw or {}).keys()):
            v = kw[k]
            (order.append(k), tensors.append(v.detach().clone())) if isinstance(v, torch.Tensor) else nontensor.append((k, str(type(v))))
        cap.update(order=order, tensors=tuple(tensors), nontensor=nontensor)

    def post(mod, a, out):
        if "out" not in cap:
            t = _primary_tensor(out)
            cap["out"] = t.detach().clone()

    h1 = target.register_forward_pre_hook(pre, with_kwargs=True)
    h2 = target.register_forward_hook(post)
    run_forward(model, inputs)
    h1.remove()
    h2.remove()
    pos = cap["tensors"]
    torch.save(pos, outdir / "inputs.pt")
    torch.save(cap["out"], outdir / "golden.pt")

    def sig(t):
        return [list(t.shape), str(t.dtype)] if isinstance(t, torch.Tensor) else str(type(t))

    (outdir / "contract.json").write_text(json.dumps({
        "model": cfg["model"]["id"], "block_class": cls, "module_path": path,
        "input_order": cap["order"], "input_sig": [sig(t) for t in pos],
        "nontensor_inputs": cap["nontensor"], "output_sig": sig(cap["out"])}, indent=2))
    (outdir / "reference.py").write_text(_REFERENCE_PY)
    print(f"froze contract -> {outdir} inputs={[list(t.shape) for t in pos]} "
          f"golden={list(cap['out'].shape)}")


if __name__ == "__main__":
    main()

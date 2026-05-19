"""Phase 4: profile by capturing torch.compile's Inductor-generated code.

The maintainer's steer: the kernel-writer should read what
torch.compile already fused/generated and mine it for optimization
ideas. So this phase recompiles the task's Model with the Inductor
trace enabled (caches disabled, else codegen is skipped), collects every
`output_code.py` (the fused Triton + wrapper), concatenates them to
configs/<name>/inductor.py, and writes a small configs/<name>/prof.json
index (kernel names, counts, path).

Config-driven: the only input is configs/<name>/config.json.
"""

import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch._inductor.config as ind  # noqa: E402

from benchmark.baseline import SEED, _build  # noqa: E402
from task.load import load_task  # noqa: E402


def _capture_output_code(model, inputs, mode: str) -> str:
    """Recompile with the Inductor trace on (caches off, or codegen is
    skipped) and concatenate every generated output_code.py."""
    with tempfile.TemporaryDirectory() as td:
        ind.force_disable_caches = True
        ind.trace.enabled = True
        ind.trace.debug_dir = td
        torch.manual_seed(SEED)
        compiled = torch.compile(model, mode=mode, fullgraph=False)
        with torch.no_grad():
            compiled(*inputs)
        parts = []
        for f in sorted(Path(td).rglob("output_code.py")):
            parts.append(f"# === {f.parent.name}/output_code.py ===\n{f.read_text()}")
    return "\n\n".join(parts)


def run_from_config(config_path: str) -> Path:
    cfg_path = Path(config_path).resolve()
    cfg = json.loads(cfg_path.read_text())
    t = cfg["task"]
    mode = cfg["benchmark"]["compile_mode"]
    assert torch.cuda.is_available(), "CUDA required for profiling"

    task = load_task(t["level"], t["problem_id"])
    model, inputs = _build(task, "cuda")
    code = _capture_output_code(model, inputs, mode)

    code_path = cfg_path.parent / "inductor.py"
    code_path.write_text(code or "# (no Inductor output_code captured)\n")

    kernels = sorted(set(re.findall(r"def (triton_[A-Za-z0-9_]+)", code)))
    prof = {
        "task": t,
        "device": torch.cuda.get_device_name(0),
        "compile_mode": mode,
        "inductor_code": code_path.name,
        "num_graphs": code.count("# === "),
        "triton_kernels": kernels,
        "num_triton_kernels": len(kernels),
        "note": "inductor_code is torch.compile's fused output — the bar. "
        "Mine it for fusion/tiling ideas; the custom kernel must beat it.",
    }
    out = cfg_path.parent / "prof.json"
    out.write_text(json.dumps(prof, indent=2))
    print(
        f"wrote {out}  ({prof['num_triton_kernels']} triton kernels, "
        f"{prof['num_graphs']} graphs) -> {code_path.name}"
    )
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to a task config.json")
    args = ap.parse_args()
    run_from_config(args.config)

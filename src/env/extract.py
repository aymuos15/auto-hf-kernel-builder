"""Extract the basic environment detail (device + toolchain).

Pure detection, no side effects. Consumed by env/create.py to bake a
per-task config.json.
"""

import shutil
import subprocess

import torch


def _nix_version() -> str | None:
    if shutil.which("nix") is None:
        return None
    r = subprocess.run(["nix", "--version"], capture_output=True, text=True)
    return r.stdout.strip() or None if r.returncode == 0 else None


def extract_env() -> dict:
    nix = _nix_version()
    if not torch.cuda.is_available():
        return {
            "device": "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nix": nix,
        }
    props = torch.cuda.get_device_properties(0)
    return {
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": f"{props.major}.{props.minor}",
        "vram_gb": round(props.total_memory / 1024**3, 2),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nix": nix,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(extract_env(), indent=2))

"""Extract the basic environment detail (device + toolchain).

Pure detection, no side effects. Consumed by env/create.py to bake a
per-task config.json.
"""

import torch


def extract_env() -> dict:
    if not torch.cuda.is_available():
        return {"device": "cpu", "torch": torch.__version__, "cuda": torch.version.cuda}
    props = torch.cuda.get_device_properties(0)
    return {
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": f"{props.major}.{props.minor}",
        "vram_gb": round(props.total_memory / 1024**3, 2),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(extract_env(), indent=2))

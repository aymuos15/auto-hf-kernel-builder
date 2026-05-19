"""Phase 5: resolve the kernel.py the AI/human provides in the config
folder. It must already exist (a precondition, like config.json) — this
does not generate one.
"""

from pathlib import Path


def scaffold(config_path: str) -> Path:
    kpath = Path(config_path).resolve().parent / "kernel.py"
    if not kpath.is_file():
        raise FileNotFoundError(f"kernel.py not found: {kpath}")
    return kpath


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    print(scaffold(args.config))

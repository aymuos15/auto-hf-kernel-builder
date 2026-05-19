"""Pluggable sandbox backend for running a bench job.

Backends (select via AK_SANDBOX):
  subprocess  default, portable (Linux/macOS/Windows). The job runs in
              a child with the import guard installed — reference deps
              physically unimportable. Accepts the in-process residual
              (same as KernelBench) beyond that.
  bwrap       Linux only, defense-in-depth on top of the guard:
              bubblewrap with network disabled and a read-only fs view,
              GPU device nodes bound through.

The guard (worker/guard.py) is the real anti-cheat boundary and is
backend-independent; bwrap adds OS-level net/fs isolation. No Docker:
reuse the existing kernel-builder nix closure for env reproducibility
if/when a fully minimized env is wanted.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_GUARD = REPO / "src" / "worker" / "guard.py"


def _bwrap_prefix() -> list[str]:
    if not shutil.which("bwrap"):
        raise RuntimeError("AK_SANDBOX=bwrap but bubblewrap (bwrap) is not installed")
    args = [
        "bwrap",
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--bind",
        str(REPO),
        str(REPO),
        "--unshare-net",
        "--die-with-parent",
    ]
    for dev in ("/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm"):
        if Path(dev).exists():
            args += ["--dev-bind", dev, dev]
    return args


def run(config_path: str, backend: str | None = None) -> dict:
    """Run one bench job under the selected sandbox; return the verdict
    dict. The child installs the import guard before importing the
    kernel, so reference deps are unimportable by any mechanism."""
    backend = backend or os.environ.get("AK_SANDBOX", "subprocess")
    child = [sys.executable, str(_GUARD), str(config_path)]
    cmd = (_bwrap_prefix() + child) if backend == "bwrap" else child
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {
            "passed": False,
            "error_class": "sandbox_error",
            "detail": f"backend={backend} rc={proc.returncode}: {proc.stderr[-800:]}",
        }

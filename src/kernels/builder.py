"""Phase 6 (deterministic): build the kernel with HF kernel-builder.

Hard requirement (maintainer): a generated kernel must compile and build
with kernel-builder. kernel.py (the AI seam) must already exist in the
config folder; this module assembles a kernel-builder universal-Triton
project from it, runs build.sh (nix build), records build.json.

Config-driven: input is configs/<name>/config.json. Requires nix+flakes
(recorded by Phase 2 env; fail fast with a clear message if absent).
"""

import hashlib
import json
import re
import subprocess
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernels.scaffold import scaffold  # noqa: E402

_HERE = Path(__file__).resolve().parent
_BUILD_SH = _HERE / "build.sh"
_ASSEMBLE_SH = _HERE / "assemble.sh"
_FLAKE = _HERE / "flake.nix"
_RC = {10: "nix_build_failed"}


def _pkg(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]", "_", name).strip("_").lower()
    return s if s and not s[0].isdigit() else f"k_{s}"


def _assemble(kernel_py: Path, task_name: str) -> tuple[Path, str]:
    pkg = _pkg(f"kb_{task_name}")
    proj = kernel_py.parent / "kernel"
    subprocess.run(
        ["bash", str(_ASSEMBLE_SH), str(proj), pkg, str(kernel_py), str(_FLAKE)],
        check=True,
    )
    return proj, pkg


def run_from_config(config_path: str) -> Path:
    cfg_path = Path(config_path).resolve()
    cfg = json.loads(cfg_path.read_text())
    bcfg = cfg["build"]
    env = cfg.get("env", {})
    kernel_py = scaffold(str(cfg_path))
    sha = hashlib.sha256(kernel_py.read_bytes()).hexdigest()
    out = cfg_path.parent / "build.json"
    proj = kernel_py.parent / "kernel"

    if out.is_file():
        prev = json.loads(out.read_text())
        if prev.get("passed") and prev.get("kernel_sha") == sha and (proj / "result").exists():
            print(f"build: SKIP (kernel unchanged, {prev['pkg']})")
            return out

    proj, pkg = _assemble(kernel_py, cfg["task"]["name"])
    record = {
        "task": cfg["task"],
        "kernel_builder": bcfg["kernel_builder"],
        "nix_attr": bcfg["nix_attr"],
        "pkg": pkg,
        "kernel_sha": sha,
    }

    if not env.get("nix"):
        record.update(
            passed=False,
            error_class="no_nix",
            detail="nix not detected by Phase 2 env",
        )
        out.write_text(json.dumps(record, indent=2))
        print("build: FAIL (no_nix)")
        return out

    frag = bcfg["nix_attr"].split("#")[-1]
    target = f"path:{proj}#{frag}"
    tries = int(bcfg.get("nix_retries", 3))
    rc, log = 1, []
    for i in range(1, tries + 1):
        if i > 1:
            print(f"build: retry {i}/{tries} (transient nix failure)")
        proc = subprocess.Popen(
            ["bash", str(_BUILD_SH), str(proj), target],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        tail: deque[str] = deque(maxlen=300)
        for line in proc.stdout:
            print(line, end="", flush=True)
            tail.append(line)
        rc = proc.wait()
        log.append(f"--- nix build try {i}/{tries} (rc={rc}) ---\n{''.join(tail)}")
        if rc == 0:
            break

    out.with_name("build.log").write_text("\n".join(log))
    if rc == 0:
        record.update(passed=True, error_class=None)
        print(f"build: PASS ({pkg})")
    else:
        cls = _RC.get(rc, "build_error")
        record.update(passed=False, error_class=cls, stderr_tail=log[-1][-2000:])
        print(f"build: FAIL ({cls}) after {tries} tries")
    out.write_text(json.dumps(record, indent=2))
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run_from_config(args.config)

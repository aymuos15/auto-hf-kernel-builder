from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from execwrap import _ssh, rsync, rsync_dest, run
from gates import GATES, GateCtx

REPO = Path(__file__).resolve().parent.parent
MS = REPO / "skills" / "model-select" / "scripts"


def model_slug(cfg):
    return cfg["model"]["id"].replace("/", "__")


def frozen_contract(cfg):
    return sorted((REPO / "targets" / model_slug(cfg)).glob("*/contract.json"))[0].parent


def phase_prep(cfg, config_path):
    slug = model_slug(cfg)
    base = REPO / "targets" / slug
    if base.is_dir() and sorted(base.glob("*/contract.json")):
        print(f"prep already done: {frozen_contract(cfg)} (delete to re-prep)")
        return
    steps = ["profile_model.py", "rank_blocks.py", "freeze_contract.py", "headroom_probe.py"]
    rel = "skills/model-select/scripts"
    cfg_rel = Path(config_path).resolve().relative_to(REPO)  # remote path mirrors repo root
    ssh = _ssh(cfg)
    if not ssh:
        for s in steps:
            print(f"\n=== prep (local): {s} ===")
            subprocess.run(f"{sys.executable} {MS / s} --config {config_path}", shell=True, cwd=REPO, check=(s != "headroom_probe.py"))
        print(f"\nprep complete -> {frozen_contract(cfg)}")
        return
    host = ssh.split()[-1]
    rroot = rsync_dest(cfg)
    rsync(f"{REPO}/", f"{host}:{rroot}/", excludes=(".git", "targets", "__pycache__"))

    def pull_back():
        (REPO / "targets" / slug).mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-az", "-e", "ssh", f"{host}:{rroot}/targets/{slug}/", f"{REPO}/targets/{slug}/"], check=False, capture_output=True, text=True, timeout=600)

    for s in steps:
        print(f"\n=== prep (authoritative, in container): {s} ===", flush=True)
        rc, _, _ = run(f"python3 -u {rel}/{s} --config {cfg_rel}", cfg, surface="container", timeout=3600, stream=True)
        # headroom_probe is advisory: its failure must not discard the contract.
        if rc != 0 and s == "headroom_probe.py":
            print(f"WARN: advisory {s} failed (rc={rc}); keeping contract")
            continue
        if rc != 0:
            pull_back()
            raise SystemExit(f"prep step failed on host: {s} (rc={rc})")
    pull_back()
    print(f"\nprep complete -> {frozen_contract(cfg)}")


def _load_kernel(kernel_project):
    import configparser
    import importlib.util

    cp = configparser.ConfigParser()
    cp.read_string((kernel_project / "build.toml").read_text())
    name = cp["general"]["name"].strip().strip('"')
    initp = kernel_project / "torch-ext" / name / "__init__.py"
    spec = importlib.util.spec_from_file_location(f"_k_{name}", initp)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.kernel


def phase_optimize(cfg):
    block_dir = frozen_contract(cfg)
    block = block_dir.name
    kernel_project = block_dir / "kernel"
    subprocess.run([sys.executable, str(REPO / "skills" / "kernel-opt" / "scripts" / "scaffold_kernel.py")], cwd=REPO, check=True)
    print(f"\n=== evaluating block '{block}' ===")
    ctx = GateCtx(block_dir, kernel_project, _load_kernel(kernel_project), cfg)
    results, failed = [], None
    for g in ("correctness", "build", "perf"):
        res = GATES[g](ctx)
        results.append(res)
        print(f"  gate {g}: {'PASS' if res.passed else 'FAIL'} ({res.error_class or 'ok'})")
        if not res.passed:
            failed = res
            break
    result = {"block": block, "passed": failed is None, "failed_gate": failed.name if failed else None, "error_class": failed.error_class if failed else None, "gates": [r.__dict__ for r in results]}
    if failed is None:
        print("all gates passed -> (#7) integrate + measure end-to-end")
    (block_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"wrote result.json (passed={result['passed']}, error_class={result['error_class']})")
    raise SystemExit(0 if failed is None else 1)


def _load_secrets():
    """Authenticate LOCAL runs too. Container runs get HF_TOKEN via
    docker --env-file (config.yaml docker_run_extra); the in-process
    correctness gate and local prep subprocesses go through this process,
    so populate os.environ from secrets.env here. Existing env wins."""
    f = REPO / "secrets.env"
    if not f.is_file():
        return
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line[len("export "):] if line.startswith("export ") else line
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main():
    _load_secrets()
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["prep", "optimize"], required=True)
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.phase == "prep":
        phase_prep(cfg, args.config)
    else:
        phase_optimize(cfg)


if __name__ == "__main__":
    main()

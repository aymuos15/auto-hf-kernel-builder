"""solve: the code-owned agent loop (human entrypoint, never agent-reachable).

Per iteration: run opencode for ONE edit turn (agent edits kernel.py
only; bash denied by opencode.json) -> restore integrity (revert any
non-kernel.py edit) -> the LOOP runs bench -> read bench.json -> keep
best / revert regression -> feed error_class into the next prompt.
Stops on pass or config loop.max_retries.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]


def _git_clean_tree() -> bool:
    r = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True)
    return r.returncode == 0 and not r.stdout.strip()


_SKIP = shutil.ignore_patterns("kernel", "__pycache__")


def _snapshot(cfg_dir: Path) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="kbsolve_"))
    # skip kernel/ (build output incl. a read-only nix-store symlink) and
    # kernel.py (the agent owns it)
    shutil.copytree(cfg_dir, tmp / "snap", symlinks=True, ignore=_SKIP)
    return tmp / "snap"


def _restore(cfg_dir: Path, snap: Path) -> None:
    subprocess.run(["git", "checkout", "--", "."], cwd=REPO, check=False)
    subprocess.run(["git", "clean", "-fdq"], cwd=REPO, check=False)
    # restore top-level config files from the snapshot; never touch
    # kernel/ (bench rebuilds it) or kernel.py (the agent's)
    for p in cfg_dir.iterdir():
        if p.is_file() and p.name != "kernel.py":
            p.unlink()
    for s in snap.iterdir():
        if s.is_file() and s.name != "kernel.py":
            shutil.copy2(s, cfg_dir / s.name)


def _prompt(name: str, bench_json: Path) -> str:
    text = (_HERE / "prompt.md").read_text().replace("<name>", name)
    if bench_json.is_file():
        d = json.loads(bench_json.read_text())
        keep = {
            k: d[k]
            for k in ("error_class", "detail", "max_abs_diff", "speedup_vs_compile", "min_speedup")
            if k in d
        }
        text += "\n\n## Last attempt\n```json\n" + json.dumps(keep, indent=2) + "\n```\n"
    return text


def _bench(cfg_path: Path) -> dict:
    subprocess.run(
        [sys.executable, str(REPO / "src" / "cli.py"), "bench", "--config", str(cfg_path)],
        cwd=REPO,
        check=False,
    )
    bj = cfg_path.with_name("bench.json")
    return (
        json.loads(bj.read_text())
        if bj.is_file()
        else {"passed": False, "error_class": "no_bench_json"}
    )


def solve(config_path: str) -> None:
    cfg_path = Path(config_path).resolve()
    cfg = json.loads(cfg_path.read_text())
    name = cfg_path.parent.name
    model = cfg["agent"]["model"]
    max_retries = int(cfg["loop"]["max_retries"])
    kernel_py = cfg_path.with_name("kernel.py")

    if not _git_clean_tree():
        print("refusing: git tree is dirty (commit/stash first)")
        raise SystemExit(2)

    snap = _snapshot(cfg_path.parent)
    best = {"speedup": None, "kernel": None}

    for attempt in range(1, max_retries + 1):
        print(f"\n=== solve {name}: attempt {attempt}/{max_retries} ===")
        subprocess.run(
            ["opencode", "run", "--model", model, _prompt(name, cfg_path.with_name("bench.json"))],
            cwd=REPO,
            check=False,
        )
        _restore(cfg_path.parent, snap)
        result = _bench(cfg_path)

        if result.get("passed"):
            print(f"PASS on attempt {attempt} ({result.get('speedup_vs_compile')}x vs compile)")
            return

        sp = result.get("speedup_vs_compile")
        if sp is not None and (best["speedup"] is None or sp > best["speedup"]):
            best = {"speedup": sp, "kernel": kernel_py.read_text()}
            print(f"  new best: correct, {sp}x vs compile")
        elif best["kernel"] is not None:
            kernel_py.write_text(best["kernel"])  # revert regression
        print(f"  not passed: {result.get('error_class')}")

    if best["kernel"] is not None:
        kernel_py.write_text(best["kernel"])
    print(f"\nstopped after {max_retries} attempts (best speedup={best['speedup']})")
    raise SystemExit(1)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    solve(ap.parse_args().config)

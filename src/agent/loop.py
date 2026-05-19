"""solve: the code-owned agent loop (human entrypoint, never agent-reachable).

Per iteration: run the agent (opencode, or `claude -p` headless when
config.agent.model is a Claude id) for ONE edit turn (file-edit only;
bash/webfetch/Task denied) -> restore integrity (revert any
non-kernel.py edit) -> the LOOP runs bench -> read bench.json -> keep
best / revert regression -> feed error_class into the next prompt.
Stops on pass or config loop.max_retries.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]


def _git_clean_tree() -> bool:
    r = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True)
    return r.returncode == 0 and not r.stdout.strip()


_SKIP = shutil.ignore_patterns("kernel", "trace", "__pycache__")


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


def _agent_argv(model: str, prompt: str) -> list[str]:
    """One edit turn. opencode by default; claude -p headless when the
    model is a Claude alias/id (uses Anthropic auth, sidesteps the
    opencode<->Copilot path). The --allowedTools whitelist mirrors
    opencode.json's lock (no Bash/WebFetch/Task — file edits only); the
    loop's integrity-restore + guarded bench are unchanged."""
    if model in ("haiku", "sonnet", "opus") or model.startswith(("claude", "anthropic/")):
        return [
            "claude",
            "-p",
            prompt,
            "--model",
            model,
            "--output-format",
            "text",
            "--allowedTools",
            "Edit Read Write Glob Grep",
            "--permission-mode",
            "acceptEdits",
            "--add-dir",
            str(REPO),
        ]
    return ["opencode", "run", "--model", model, prompt]


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
    queue = os.environ.get("AK_QUEUE")
    if queue:
        return _bench_via_queue(cfg_path, queue)
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


def _bench_via_queue(cfg_path: Path, queue: str, timeout: float = 3600.0) -> dict:
    """Submit the bench job to the durable queue and wait for a worker
    to produce the verdict (decouples kernel execution from the solve
    driver; the worker can later run it sandboxed)."""
    sys.path.insert(0, str(REPO / "src"))
    from worker.queue import Queue

    q = Queue(queue)
    jid = q.enqueue(str(cfg_path))
    deadline = time.time() + timeout
    while time.time() < deadline:
        row = q.get(jid)
        if row and row["state"] in ("done", "failed"):
            return (
                json.loads(row["verdict"])
                if row["verdict"]
                else {
                    "passed": False,
                    "error_class": "no_bench_json",
                }
            )
        time.sleep(2.0)
    return {"passed": False, "error_class": "queue_timeout"}


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

    if not kernel_py.exists():
        kernel_py.write_text(
            '"""AGENT SEAM: implement kernel(*inputs) — a real @triton.jit '
            "kernel reproducing reference.py within tolerance and beating "
            'the bar. See the config folder + prompt."""\n\n\n'
            "def kernel(*inputs):\n"
            '    raise NotImplementedError("write the Triton kernel here")\n'
        )

    snap = _snapshot(cfg_path.parent)
    trace_dir = cfg_path.parent / "trace"
    trace_dir.mkdir(exist_ok=True)
    best = {"speedup": None, "kernel": None}

    for attempt in range(1, max_retries + 1):
        print(f"\n=== solve {name}: attempt {attempt}/{max_retries} ===")
        prompt = _prompt(name, cfg_path.with_name("bench.json"))
        proc = subprocess.Popen(
            _agent_argv(model, prompt),
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        lines = []
        for line in proc.stdout:  # tee: live to console + captured for trace
            print(line, end="", flush=True)
            lines.append(line)
        proc.wait()
        _restore(cfg_path.parent, snap)
        result = _bench(cfg_path)
        blog = cfg_path.with_name("build.log")
        build_log = blog.read_text() if blog.is_file() else "(no build attempted)"
        (trace_dir / f"attempt_{attempt}.log").write_text(
            f"# attempt {attempt}  model={model}\n\n=== PROMPT ===\n{prompt}\n\n"
            f"=== AGENT TRANSCRIPT ===\n{''.join(lines)}\n\n"
            f"=== BENCH VERDICT ===\n{json.dumps(result, indent=2)}\n\n"
            f"=== BUILD LOG ===\n{build_log}\n"
        )

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

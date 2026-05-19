"""Persistent per-GPU worker loop and a single-box pool launcher.

A worker is a long-lived process pinned to one GPU: it boots torch/CUDA
once and serves many jobs (amortizes the multi-hundred-ms CUDA init +
seconds of Triton first-compile). The pool launches one worker
subprocess per GPU with CUDA_VISIBLE_DEVICES set.
"""

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worker.contract import run_job  # noqa: E402
from worker.queue import Queue  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def worker_loop(
    queue_path: str, owner: str | None = None, lease_secs: float = 1800.0, poll_secs: float = 2.0
) -> None:
    owner = owner or f"{socket.gethostname()}:{os.getpid()}"
    q = Queue(queue_path)
    print(f"worker {owner} up (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
    try:
        while True:
            job = q.claim(owner, lease_secs)
            if job is None:
                time.sleep(poll_secs)
                continue
            jid = job["id"]
            try:
                verdict = run_job(job["config_path"])
                q.complete(jid, owner, verdict)
                print(f"job {jid} {job['config_path']} -> {verdict.get('error_class') or 'PASS'}")
            except Exception as exc:  # a worker fault must not lose the job silently
                q.fail(jid, owner, repr(exc))
                print(f"job {jid} FAILED in worker: {exc!r}")
    except KeyboardInterrupt:
        print(f"worker {owner} stopping")
    finally:
        q.close()


def serve(queue_path: str, gpus: list[int], lease_secs: float = 1800.0) -> None:
    """Launch one persistent worker subprocess per GPU id."""
    procs: list[subprocess.Popen] = []
    for g in gpus:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    str(REPO / "src" / "cli.py"),
                    "worker",
                    "--queue",
                    queue_path,
                    "--lease",
                    str(lease_secs),
                ],
                cwd=REPO,
                env=env,
            )
        )
    print(f"pool: {len(procs)} workers on gpus {gpus} (queue {queue_path})")

    def _stop(*_):
        for p in procs:
            p.terminate()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    for p in procs:
        p.wait()

"""agentic-kernels CLI. Humans run config/setup/solve; the agent (driven
by solve via opencode) never runs anything.

python3 src/cli.py config --level 3 --problem 4              # human, once
python3 src/cli.py setup  --config configs/<name>/config.json  # human, once: freeze the bar
python3 src/cli.py solve  --config configs/<name>/config.json  # human: start the agent loop
python3 src/cli.py bench  --config configs/<name>/config.json  # loop calls this (or debug)
python3 src/cli.py build  --config configs/<name>/config.json  # standalone build (debug)
"""

import json
import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent))

app = typer.Typer(add_completion=False, help="agentic-kernels pipeline")

_CONFIG = typer.Option(..., "--config", exists=True, dir_okay=False, help="path to a config.json")


@app.command()
def build(config: Path = _CONFIG) -> None:
    """Phase 6: build kernel.py with kernel-builder."""
    from kernels.builder import run_from_config

    out = run_from_config(str(config))
    passed = json.loads(out.read_text()).get("passed", False)
    raise typer.Exit(0 if passed else 1)


@app.command()
def bench(config: Path = _CONFIG) -> None:
    """Phase: evaluate the built kernel — correctness + perf vs the bar."""
    from benchmark.bench import run_from_config

    out = run_from_config(str(config))
    passed = json.loads(out.read_text()).get("passed", False)
    raise typer.Exit(0 if passed else 1)


@app.command()
def setup(config: Path = _CONFIG) -> None:
    """One-time prep: benchmark baseline + profile (freezes the bar)."""
    from env.setup import setup as setup_pipeline

    setup_pipeline(str(config))


@app.command()
def solve(config: Path = _CONFIG) -> None:
    """Human-started agent loop: opencode edits kernel.py -> bench, repeat."""
    from agent.loop import solve as solve_loop

    solve_loop(str(config))


_DEFAULT_QUEUE = str(Path(__file__).resolve().parents[1] / "configs" / "queue.db")


@app.command()
def enqueue(
    config: Path = _CONFIG,
    queue: str = typer.Option(_DEFAULT_QUEUE, "--queue"),
    attempt: int = typer.Option(0, "--attempt"),
) -> None:
    """Enqueue a bench job for a config dir (durable queue)."""
    from worker.queue import Queue

    jid = Queue(queue).enqueue(str(config), attempt)
    print(f"enqueued job {jid} -> {queue}")


@app.command()
def worker(
    queue: str = typer.Option(_DEFAULT_QUEUE, "--queue"),
    lease: float = typer.Option(1800.0, "--lease"),
) -> None:
    """Run one persistent worker bound to the current GPU env."""
    from worker.pool import worker_loop

    worker_loop(queue, lease_secs=lease)


@app.command()
def pool(
    gpus: str = typer.Option(..., "--gpus", help="comma-separated GPU ids, e.g. 0,1"),
    queue: str = typer.Option(_DEFAULT_QUEUE, "--queue"),
) -> None:
    """Launch one persistent worker per GPU (single-box pool)."""
    from worker.pool import serve

    serve(queue, [int(g) for g in gpus.split(",") if g.strip() != ""])


@app.command()
def config(
    level: int = typer.Option(..., "--level"),
    problem: int = typer.Option(..., "--problem"),
    name: str = typer.Option(None, "--name", help="default: L<level>_<task>"),
    force: bool = typer.Option(False, "--force", help="rebuild if it exists"),
) -> None:
    """Human prep: create configs/<name>/config.json."""
    from env.create import create_config

    create_config(level, problem, name, force)


if __name__ == "__main__":
    app()

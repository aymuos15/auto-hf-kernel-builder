"""agentic-kernels CLI. The agent uses only two verbs: build and run.

python3 src/cli.py build  --config configs/<name>/config.json
python3 src/cli.py run    --config configs/<name>/config.json
python3 src/cli.py config --level 3 --problem 4        # human prep
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
def run(config: Path = _CONFIG) -> None:
    """Full pipeline: benchmark + profile + build."""
    from run import run as run_pipeline

    run_pipeline(str(config))


@app.command()
def config(
    level: int = typer.Option(..., "--level"),
    problem: int = typer.Option(..., "--problem"),
    name: str = typer.Option(None, "--name", help="default: L<level>_<task>"),
) -> None:
    """Human prep: create configs/<name>/config.json."""
    from env.create import create_config

    create_config(level, problem, name)


if __name__ == "__main__":
    app()

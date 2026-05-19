"""Pipeline runner. The ONLY input is a config.json (made by
src/env/create.py, human-reviewed). Every stage reads only that config;
outputs land in the same folder. Stages: Benchmark (3), Profile (4),
Build (6, kernel-builder; needs configs/<name>/kernel.py and nix).
Rich UI: a header panel, one progress bar per phase, and a results
table; verbose output goes to run.log.

  python3 src/run.py --config configs/L3_4_LeNet5/config.json
"""

import argparse
import json
import logging
import os
import sys
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Quiet torch: it reconfigures Python logging on import (so pre-import
# levels don't stick) and writes to the real stderr fd (so redirect can't
# catch it). set_logs after import is the lever that works.
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
warnings.filterwarnings("ignore")

import torch  # noqa: E402, F401

for _n in ("torch", "torch._inductor", "torch._dynamo"):
    logging.getLogger(_n).setLevel(logging.ERROR)

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.progress import (  # noqa: E402
    BarColumn,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.progress import Progress as RichProgress  # noqa: E402
from rich.table import Table  # noqa: E402

from benchmark.baseline import run_from_config as benchmark  # noqa: E402
from kernels.builder import run_from_config as build  # noqa: E402
from profiling.inductor import run_from_config as profile  # noqa: E402

STAGES = [
    ("Phase 3 · Benchmark", benchmark),
    ("Phase 4 · Profile", profile),
    ("Phase 6 · Build", build),
]


def _header(console: Console, cfg: dict, path: Path) -> None:
    t, e = cfg["task"], cfg["env"]
    body = (
        f"[bold]{path.name}[/bold]\n"
        f"task   L{t['level']} #{t['problem_id']}  {t['name']}\n"
        f"device {e.get('gpu', e['device'])}  ·  torch {e['torch']}"
    )
    console.print(Panel(body, title="agentic-kernels", expand=False))


def _summary(console: Console, cfg_path: Path) -> None:
    table = Table(title="results", show_header=True, header_style="bold")
    table.add_column("artifact")
    table.add_column("value")
    res = cfg_path.with_name("res.json")
    prof = cfg_path.with_name("prof.json")
    if res.is_file():
        b = json.loads(res.read_text())["baseline"]
        table.add_row("eager", f"{b['eager']['time_ms']} ms / {b['eager']['peak_mem_mb']} MB")
        table.add_row("compile", f"{b['compile']['time_ms']} ms / {b['compile']['peak_mem_mb']} MB")
        table.add_row("compile speedup", f"{b['compile_speedup']}x")
    if prof.is_file():
        p = json.loads(prof.read_text())
        table.add_row("inductor code", p["inductor_code"])
        table.add_row("triton kernels", str(p["num_triton_kernels"]))
    bld = cfg_path.with_name("build.json")
    if bld.is_file():
        d = json.loads(bld.read_text())
        table.add_row(
            "kernel-builder",
            f"[green]PASS[/green] ({d['pkg']})"
            if d["passed"]
            else f"[red]FAIL[/red] ({d['error_class']})",
        )
    console.print(table)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to a config.json")
    args = ap.parse_args()

    console = Console()
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        console.print(f"[red]config not found:[/red] {cfg_path}")
        raise SystemExit(2)
    cfg_path = cfg_path.resolve()
    cfg = json.loads(cfg_path.read_text())

    _header(console, cfg, cfg_path)

    # Console shows only the bars + final table; everything verbose
    # (stage prints, torch chatter, tracebacks) goes to this log.
    log_path = cfg_path.with_name("run.log")
    progress = RichProgress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    with log_path.open("w") as log:
        log.write(f"# {cfg_path.name}  {datetime.now().isoformat(timespec='seconds')}\n")
        log.flush()
        with progress:
            for name, fn in STAGES:
                # Bar appears when the phase starts; total=None makes it
                # pulse while running, so only the active phase animates.
                tid = progress.add_task(name, total=None)
                log.write(f"\n===== {name} =====\n")
                log.flush()
                # The one deliberate boundary: log the failure, point the
                # user at the log, exit non-zero.
                try:
                    with redirect_stdout(log), redirect_stderr(log):
                        fn(str(cfg_path))
                except Exception as exc:
                    traceback.print_exc(file=log)
                    log.flush()
                    progress.stop()
                    console.print(f"[red]{name} failed[/red] — see {log_path}")
                    raise SystemExit(1) from exc
                log.flush()
                progress.update(tid, total=1, completed=1)

    console.print(f"[green]done[/green] · {len(STAGES)} stages")
    _summary(console, cfg_path)
    console.print(f"log → {log_path}")


if __name__ == "__main__":
    main()

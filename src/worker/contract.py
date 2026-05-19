"""The bench-worker contract: a pure function from a config dir to a
verdict. This is the one seam every execution substrate goes through
(inline, queued worker, later a sandboxed child) — bench stays
unaware of how it is scheduled.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run_job(config_path: str) -> dict:
    """Run bench for one config dir, return the verdict dict (bench.json
    contents). bench is imported lazily so queue-only code paths (and
    their hermetic tests) need no torch/CUDA."""
    from benchmark.bench import run_from_config

    out = run_from_config(config_path)
    return json.loads(Path(out).read_text())

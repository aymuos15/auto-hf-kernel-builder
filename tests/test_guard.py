import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

# Block a real, installed module ("transformers" — the exact A4 case) and
# prove every in-process import mechanism is walled off, while an
# unrelated module still imports. Run in a child so meta_path is clean.
_PROBE = """
import sys
sys.path.insert(0, {src!r})
from worker.guard import install
install({{"transformers"}})

import importlib
ok = []
for how in ("stmt", "import_module", "dunder"):
    try:
        if how == "stmt":
            import transformers  # noqa: F401
        elif how == "import_module":
            importlib.import_module("transformers")
        else:
            __import__("transformers")
        ok.append(how + ":LEAK")
    except ModuleNotFoundError:
        ok.append(how + ":blocked")

import colorsys  # noqa: F401  (not blocked -> must still work)
ok.append("colorsys:ok")
print(",".join(ok))
"""


def test_guard_blocks_all_import_mechanisms():
    out = subprocess.run(
        [sys.executable, "-c", _PROBE.format(src=str(SRC))],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    result = out.stdout.strip().splitlines()[-1]
    assert result == "stmt:blocked,import_module:blocked,dunder:blocked,colorsys:ok", result

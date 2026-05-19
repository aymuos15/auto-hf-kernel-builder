"""Runtime import guard: make the reference's deps physically
unimportable inside the bench child.

A `sys.meta_path` finder is consulted by *every* import path — the
`import` statement, `importlib.import_module`, and `__import__` alike —
so unlike the static ast scan (A4: defeated by `importlib`) this cannot
be bypassed from Python. Installed before `kernel.py` is imported, it
turns "reproduce the math" from a rule into a hard wall: the agent
cannot reach `transformers` (or whatever the task's deps are) by any
in-process mechanism.

This is the portable boundary (works on any OS, subprocess-only).
bubblewrap/seccomp in sandbox.py add net/fs isolation on top.
"""

import importlib.abc
import importlib.machinery
import json
import sys
from pathlib import Path


class _Blocked(importlib.abc.MetaPathFinder):
    def __init__(self, names: set[str]):
        self.names = names

    def find_spec(self, fullname, path, target=None):
        top = fullname.split(".")[0]
        if top in self.names:
            raise ModuleNotFoundError(
                f"import of {fullname!r} is blocked: the bench worker forbids the "
                f"reference dependency {top!r}. Reproduce the computation in Triton; "
                f"importing or running the reference model is a cheat."
            )
        return None


def install(blocked: set[str]) -> None:
    if blocked:
        sys.meta_path.insert(0, _Blocked(set(blocked)))


def install_from_ref(config_path: str) -> None:
    """Read the frozen ref.pt's meta.ref_deps (written by setup, where
    the task source was available) and block those modules. No torch
    import needed — ref.pt's pickle header is read as plain bytes via
    torch only if present; fall back to no-op if missing."""
    cfg = Path(config_path).resolve()
    refp = cfg.parent.parent / ".ak" / cfg.parent.name / "ref.pt"
    if not refp.is_file():
        return
    import torch  # local: only needed to read the tensor archive

    meta = torch.load(refp, map_location="cpu", weights_only=False)["meta"]
    install(set(meta.get("ref_deps", [])))


if __name__ == "__main__":
    # child entrypoint: install the guard, then run bench, print verdict
    cfg = sys.argv[1]
    install_from_ref(cfg)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from worker.contract import run_job

    print(json.dumps(run_job(cfg)))

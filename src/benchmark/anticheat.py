"""Anti-cheat primitives, kept out of baseline.py / bench.py so those
read as their actual logic. None of this is the boundary — that is the
frozen ref + the worker import guard (see worker/guard.py). These are
the in-bench heuristics for the non-import cheat vectors.
"""

import ast
import contextlib

import torch

_KERNEL_ALLOWED_IMPORTS = frozenset(
    {
        "torch",
        "triton",
        "math",
        "typing",
        "__future__",
        "dataclasses",
        "functools",
        "itertools",
        "operator",
        "collections",
    }
)


def _top_imports(src: str) -> set[str]:
    """Top-level module names imported anywhere in src (incl. inside
    functions). Static scan only — runtime imports (importlib/__import__)
    evade it, so this is belt-and-suspenders, not the boundary."""
    mods: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    return mods


@contextlib.contextmanager
def _triton_counter():
    """Count @triton.jit launches and the largest tensor any launch
    touched. max_numel lets bench reject a no-op fig-leaf kernel (a
    tiny launch on a throwaway tensor) used only to satisfy n>0 while
    the real work is torch passthrough."""
    from triton.runtime.jit import JITFunction

    box = {"n": 0, "max_numel": 0}
    orig = JITFunction.run

    def run(self, *a, **k):
        box["n"] += 1
        for v in (*a, *k.values()):
            if torch.is_tensor(v):
                box["max_numel"] = max(box["max_numel"], v.numel())
        return orig(self, *a, **k)

    JITFunction.run = run
    try:
        yield box
    finally:
        JITFunction.run = orig


@contextlib.contextmanager
def _autocast_tripwire():
    """Trip if the kernel runs under reduced-precision CUDA autocast.
    The frozen baseline is the reference's native dtype (fp32); a kernel
    that wraps the reference in bf16/fp16 autocast 'beats' it on
    precision, not on a real kernel. Caught here, not via output dtype
    (the cheat upcasts the result back to fp32 on return)."""
    box = {"tripped": False, "dtype": None}
    orig = torch.autocast.__enter__

    def enter(self):
        if getattr(self, "device", None) == "cuda" and getattr(self, "fast_dtype", None) in (
            torch.bfloat16,
            torch.float16,
        ):
            box["tripped"] = True
            box["dtype"] = str(self.fast_dtype)
        return orig(self)

    torch.autocast.__enter__ = enter
    try:
        yield box
    finally:
        torch.autocast.__enter__ = orig

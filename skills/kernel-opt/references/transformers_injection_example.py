"""Vendored, generic transformers kernel-injection example.

Previously the skills pointed at the upstream cuda-kernels skill's file of
this name, which is NOT in this repo — a dead reference (flagged in the skill
UX review). This is a self-contained, model-agnostic pattern. Nothing here is
model- or hardware-specific; all specifics come from config.yaml.

Pattern: replace a target nn.Module's forward with the locally-built kernel,
load via the transformers kernels hook, then PROVE it fires
(config.yaml: integration.assert_kernel_invoked; transformers#40459).
"""
from __future__ import annotations


def inject_and_assert(model, target_module_cls, kernel_fn):
    """Swap `kernel_fn` into every instance of `target_module_cls` inside
    `model`, run a real forward, and assert the kernel was actually invoked.

    `model`             : the loaded transformers model (config.yaml: model.id)
    `target_module_cls` : the block class chosen by model-select
    `kernel_fn`         : callable loaded from the local kernel-builder build/
    Returns the invocation count (must be > 0 or the gate fails).
    """
    calls = {"n": 0}

    def wrapped_forward(self, *args, **kwargs):
        calls["n"] += 1
        return kernel_fn(*args, **kwargs)

    patched = 0
    for module in model.modules():
        if isinstance(module, target_module_cls):
            # bind the replacement as this instance's forward
            module.forward = wrapped_forward.__get__(module, type(module))
            patched += 1
    if patched == 0:
        raise RuntimeError(
            "no instances of the target block found — wrong target_module_cls"
        )
    return calls, patched


# Real wiring with transformers' own hook (preferred over manual patching
# when available) looks like:
#
#   from transformers import AutoModel
#   from kernels import ...                       # load from local build/
#   model = AutoModel.from_pretrained(cfg["model"]["id"],
#                                     kernel_config=KernelConfig(...))
#   # then assert invocation as above (a forward hook / counter), because
#   # use_kernels has historically NOT fired the kernel (transformers#40459).
#
# Keep the manual `inject_and_assert` as the fallback + the invocation proof.

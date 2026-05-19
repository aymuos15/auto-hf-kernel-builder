# Scoping: does it fit, and what should the task be?

The harness keeps **~3 copies resident** at once — the reference model, the `torch.compile` baseline, and the candidate kernel — plus activations and ~0.4-0.6 GB CUDA context. Budget against that, not just one copy.

Rough fp32 weight size = `params * 4 bytes`. Quick probe:

```python
n = sum(p.numel() for p in m.parameters())
print(n/1e6, "M params  ", n*4/1e9, "GB fp32 weights")
```

Then measure real peak with one forward (`torch.cuda.reset_peak_memory_stats()` → forward → `torch.cuda.max_memory_allocated()`). `scripts/preflight.py` does this for you.

## Decision tree

1. **Full model, ~3× weights + activations ≤ GPU?** → wrap the **whole model**. Most faithful "improve the model" task. The kernel need not reimplement everything as one Triton kernel — it must only match the whole-model output and beat `torch.compile` of the whole forward.
2. **Doesn't fit, but the architecture has a smaller config / size tier?** → construct from a **reduced config** (fewer layers / smaller hidden / smaller input). Random init from config is the established pattern (HF's own `*-tiny-random` test fixtures do exactly this). Trained vs random weights is irrelevant — the harness seeds weights and only checks the kernel against that seeded reference.
3. **Still too big, or you want a sharper target?** → wrap **one representative submodule** (the block repeated N× — an encoder/attention layer — speeding it up speeds up most of the model). Smaller, easier to converge, narrower result.

## Notes

- "No official small variant" is common. Reducing the config yourself is valid and standard; you are not obligated to find a pretrained small checkpoint.
- Shrink **token/sequence count** (not necessarily width) to keep `torch.compile` autotune and each bench iteration fast to iterate, while keeping the kernel realistic.
- A model that "fits in VRAM for one forward" can still OOM in the harness because of the 3-copy rule — always size against that.
- Whole-model tasks pay a heavy first `torch.compile(mode="max-autotune")`; expect a slow setup before the loop. That cost is inherent to the ambition, not a bug.

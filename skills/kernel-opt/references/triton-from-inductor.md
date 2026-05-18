# Beating torch.compile: reading inductor codegen

Bar = `torch.compile(mode="max-autotune")`, not eager (the HF blog kernel lost end-to-end because it never gated on compile). `dump_inductor.py` gives you the codegen to beat. Look for:

- **Unfused epilogues** — matmul + separate elementwise/bias/activation → fuse into one Triton kernel.
- **Many tiny kernels** — launch overhead dominates at small shapes → fuse.
- **Redundant global loads** — same tensor reloaded across kernels → keep in registers/shared.
- **Conservative tiling/num_warps** — hand-shape for the block's *actual frozen shapes*.
- **Memory-bound ops at low BW** — achieved GB/s vs peak = headroom. (Vendor-GEMM-bound: don't bother; model-select shouldn't have picked it.)

Toolchain caveat: on newer accelerators Triton may fall back to an older compute target — handicaps baseline and custom equally (fair), but report numbers against the toolchain in `config.yaml: env:`, not the hardware ceiling.

Workflow: read codegen → fix one concrete inefficiency → correctness → perf vs compile → next inefficiency only after a pass.

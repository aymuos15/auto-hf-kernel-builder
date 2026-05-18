---
name: profiling
description: Reliable profiling, especially on unified-memory (UMA/iGPU) and Arm SBSA accelerators. nsys-first to find hot kernels, ncu-second on those. Used by model-select and kernel-opt.
---

# profiling

Host/GPU from `config.yaml: hardware`; nothing hardcoded. Tuned for UMA/iGPU + Arm SBSA, where x86 discrete-GPU instincts don't transfer.

- **`nsys` first** (timeline): what ran when, GPU busy?, host/device overlap. Treat wall-clock + throughput as the source of truth, not derived metrics.
- **`ncu` second, narrow**: only on the hot kernels nsys found, and only after counter permissions are confirmed.
- **`ERR_NVGPUCTRPERM`** = a permissions gate (not a broken profiler); it also blocks nsys HW metrics while nsys *tracing* still works. Fix: https://developer.nvidia.com/ERR_NVGPUCTRPERM. Until enabled, stay on nsys tracing + wall-clock.
- **UMA caveat**: `nvidia-smi` / `cudaMemGetInfo()` are unreliable — don't gate on device-memory readings. UMA perf is mostly host/device overlap (nsys shows it).
- **Arm SBSA**: Nsight differs from x86; see https://docs.nvidia.com/dgx/dgx-spark-porting-guide/optimization.html

`torch.profiler` (in `model-select/scripts/profile_model.py`) is the structured, permission-free fallback for per-op time/shapes.

## Do not

Lead with `ncu` or block on `ERR_NVGPUCTRPERM` · trust device-memory on UMA · copy x86 Nsight invocations on Arm SBSA.

# Setup cost

Defaults `warmup=10`, `iters=50`, `compile_mode=max-autotune`:

| | Explicit forwards | Compile |
|---|---|---|
| Phase 3 eager | 10 + 50 = **60** | — |
| Phase 3 compiled | 10 + 50 = **60** | 1× max-autotune |
| Phase 4 capture | **1** | 1× max-autotune, caches off (full recompile) |

`2*(warmup+iters)+1` = **121** explicit forwards. But wall-clock is dominated by the two from-scratch max-autotune compiles (Inductor benchmarks many kernels internally; Phase 4 disables caches so it pays that twice). Tuning `warmup`/`iters` shrinks only the negligible part.

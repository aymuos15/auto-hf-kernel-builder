# kernel-builder layout — UNIVERSAL (Triton) kernel

De-risk-proven (Spark/GB10): a Triton kernel is a **universal** kernel-builder project — pure Python, no `.cu`, no C++ bindings, no `[torch]`/`[kernel.*]`. Honors both "Triton-first" and "must build with kernel-builder". `kernel-builder init` defaults to CUDA — don't use it; write `build.toml` by hand (mirrors the real `kernels-community/triton-layer-norm`). `driver` auto-scaffolds all of this.

```
targets/<model>/<block>/kernel/
├── build.toml      # [general]\n name = "<name>"\n universal = true   (entire file)
├── flake.nix       # genFlakeOutputs, kernel-builder PINNED to b4accba; rev = self.shortRev or self.dirtyShortRev or "dev0"  (literal fallback required — not a git repo)
└── torch-ext/<name>/__init__.py   # pure Python: @triton.jit + def kernel(*inputs)
```

## Build & load — two surfaces (proven on GB10)

- **Build = host Nix** (kernel-builder is a host Nix flake, not PyPI/container): `nix build --accept-flake-config .#bundle` → `build/torch-universal/<name>/`; then `cp -rL result build` (deref the /nix/store symlink).
- **Load = plain import in the container**: `sys.path.insert(0, "<repo>/build/torch-universal"); import <name>`. Universal = pure Python — do **not** use `kernels.get_local_kernel` (compiled-only; wants a `metadata.json` real universal repos lack).

## Corrections vs old CUDA assumptions

- `target_caps`/`cuda-capabilities` **N/A** for universal (no per-SM compile; only the `torch-cuda` variant).
- kernel-builder rev **must be pinned** (`main`/`dffbce5` changed the schema and rejects the minimal universal form; `b4accba` is confirmed-good).
- `CARD.md` optional/auto; nothing uploaded (`publish.enabled: false`).
- Build alone isn't enough — the gate also requires the container import to succeed and run on the GPU.

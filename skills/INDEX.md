# Skills index

Selector for read-by-path skill injection. A harness/prompt picks the **one** matching skill by its trigger and reads only that `SKILL.md` into context — the library can grow without bloating any single task. Schema for `skill.json` is `skills/schema/skill.schema.json`; `python3 skills/validate_skills.py` checks every entry.

## triton

| id | use when | path |
|---|---|---|
| `triton.write-triton-kernel` | authoring a new `@triton.jit` kernel for an elementwise/reduction op (no op-specific skill fits) | `skills/triton/write-triton-kernel/SKILL.md` |
| `triton.profile-and-target-kernel` | decide *what* to fuse/tile from the provided `inductor.py`/`prof.json` profile (before writing, or on `slower_than_compile`) | `skills/triton/profile-and-target-kernel/SKILL.md` |
| `triton.optimize-triton-block-parameters` | kernel is correct but `slower_than_compile`; tune BLOCK_SIZE/num_warps/num_stages/autotune | `skills/triton/optimize-triton-block-parameters/SKILL.md` |
| `triton.debug-triton-correctness` | `numeric_mismatch` or `nondeterministic` verdict; classify and fix the bug | `skills/triton/debug-triton-correctness/SKILL.md` |
| `triton.write-triton-softmax-kernel` | the op is row-wise softmax (optionally masked/scaled) | `skills/triton/write-triton-softmax-kernel/SKILL.md` |
| `triton.write-triton-gemm-kernel` | the op is matmul / linear with an epilogue to fuse | `skills/triton/write-triton-gemm-kernel/SKILL.md` |
| `triton.write-triton-layernorm-kernel` | the op is LayerNorm/RMSNorm + affine | `skills/triton/write-triton-layernorm-kernel/SKILL.md` |
| `triton.write-triton-attention-kernel` | the op is scaled-dot-product / multi-head attention | `skills/triton/write-triton-attention-kernel/SKILL.md` |

## other categories

| id | use when | path |
|---|---|---|
| `hf-model-to-kernelbench` (native skill) | wrap an HF model as a KernelBench-style task | `skills/hf-model-to-kernelbench/SKILL.md` |

## Selection rule

Map the harness `error_class` to a skill: `slower_than_compile` → profile-and-target (pick what to attack) then optimize (tune it); `numeric_mismatch`/`nondeterministic` → debug; first attempt / no kernel → profile-and-target then write (prefer the op-specific softmax/gemm/layernorm/attention skill when the reference op matches one, else the general `write-triton-kernel`). Inject that one skill's `SKILL.md`; do not paste the whole library.

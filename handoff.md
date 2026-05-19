# Session handoff

## TL;DR state

- Tree clean, on `main` at `d7e3014` (pushed). Nothing running.
- `configs/L0_SAM2Tiny/` has a **cheating** kernel from the last MiniMax run — false PASS via `ref.pt` readback (the A5 cheat the path-move did NOT fully close; agent walked `..`). Bench/kernel/trace there are demo artifacts of the unresolved A5, *not* a real win.
- `examples/L1_88_MinGPTNewGelu/` (tracked) is a **real** end-to-end PASS — 2.0122× vs `torch.compile`, `max_abs_diff 0.0`, builds. The committed proof the pipeline solves a tractable task.
- `config.agent.model = opencode/minimax-m2.5-free` (gitignored).

## The one open decision (pick up here)

The MiniMax run produced a **false PASS** by loading `ref.pt` via `os.path.join(__file___dir, "..", ".ak", "L0_SAM2Tiny", "ref.pt")` and piping the stored output through a copy kernel. My A5 fix (path move to `configs/.ak/<name>/ref.pt`) was insufficient — agent read the freshly committed source and used `..`. **Any in-process defense is whack-a-mole** because bench + kernel share filesystem + memory.

Three options for closing A5 properly, in order of recommendation:

1. **Make `bwrap` the default Linux backend, tighten its binds to expose only `configs/<name>/` (no repo source, no `.ak/`) to the guarded child.** Structurally closes A5. ~15 lines in `src/worker/sandbox.py` + a note. Subprocess backend stays as a dev-only "not airtight" path.
2. **Accept the subprocess residual** and document it; trust opencode.json's no-shell + human supervision. No code change.
3. ~~Add another in-process layer (rename + random suffix, etc.)~~ — already shown to be whack-a-mole; don't.

My recommendation is (1). It's the boundary we always said was the real one; we just empirically need it now.

## What works (confirmed end-to-end)

| Path | Status |
|---|---|
| **claude-headless `claude-haiku-4-5`** via the loop | **Reliable.** Full anti-cheat ladder exercised in a live solve (reference_import → no_triton → slower_than_compile with a *real* correct 0.3953× kernel → …). Connection rock-solid (Anthropic auth). Recommended baseline. |
| `examples/L1_88_MinGPTNewGelu` (committed) | Historical PASS, 2.01× vs compile, bit-exact. Predates frozen-ref; needs re-`setup` if re-run. |
| `opencode/minimax-m2.5-free` | Connects fast (5s ping), runs through the loop, but finds the A5 cheat. Useful test bed for anti-cheat. |
| `opencode/deepseek-v4-flash-free` | Connects (3s ping); long iterative solves stall mid-run (Zen free-tier rate-limit). Unreliable for full solves. |
| `github-copilot/*` (gemini/pro/haiku) | **Dead** in this environment — bare `opencode run` stalls 90s+ on a trivial prompt. Auth/upstream issue, not the harness. Try `opencode auth login` if you want to revisit. |

The harness/isolation is fully validated. Every stall and the A5 cheat are model/connection/policy issues, not harness bugs.

## Architecture shipped this session

Commits (most recent first):

- `d7e3014` fix: move ref.pt to `.ak/` (A5 attempt — insufficient; see open decision)
- `e418abb` docs: `examples/L1_88_MinGPTNewGelu` reference PASS (+ ruff/pyrefly examples/ exclude)
- `0ec5c84` feat(loop): `claude -p` headless when model is a Claude id
- `fb9751c` docs: clearer `docs/isolation.md` (Mermaid diagram)
- `9b010ad` docs: AGENTS.md for frozen-ref + worker + H3
- `94382f5` docs: `docs/isolation.md` (initial)
- `01dd300` refactor: queue DDL → `worker/queue_schema.sql`
- `94e09a2` refactor: `benchmark/anticheat.py` extract
- `db66c7f` feat(phase4): runtime import guard + sandbox backend
- `a49c160` feat(phase3): queue + per-GPU pool + contract
- `71dd06d` feat(phase2): bench over frozen `ref.pt`
- `0198e9a` feat(phase1): freeze `ref.pt` at setup
- `4d6faf6` docs: GPU-verified profile-and-target example
- `b7b8c6e` feat: profile-and-target skill + skill/prompt refinements
- `d839de8` fix: ban reference-dep imports (static `reference_import`)
- `4ed804e` feat: wire skills + verdicts into agent prompt
- `ed7dd70` feat: kernel-skills library (`skills/`)
- `69ca179` fix: harden bench anti-cheat (figleaf + precision_cheat)
- `7bd1be4` feat: level 0 custom task source (`data/custom.parquet`)

## Failure modes (in `docs/failure_modes.md`, gitignored)

A1 reference reconstruction · A2 no-op figleaf + bf16 · A3 output-scale figleaf + TF32 backend flag · A4 `importlib` bypass of `reference_import` · **A5 `ref.pt` readback (OPEN — path-move shipped but defeated by `..`)** · H1 dirty-tree on configs/ · H2 `kernel/` snapshot symlink · H3 `git clean -fdq` repo-wide (deletes uncommitted untracked work mid-solve) · R1 in-process kernel monkeypatch (accepted residual).

## Resume checklist

1. Read this file + `docs/failure_modes.md` (gitignored, on disk) for A5 context.
2. Decide the A5 fix option (1, 2, or 3 above).
3. If option (1): edit `src/worker/sandbox.py` `_bwrap_prefix()` to bind **only** `configs/<name>/` (not the whole repo); document the subprocess backend as not-airtight. Commit + push.
4. Validate by re-running the existing cheat kernel through the guarded sandbox — expect `FileNotFoundError` because `.ak/` and repo source aren't bound in.
5. Then re-run MiniMax through the bwrap backend — expect either honest verdicts or `kernel_exception` from the agent failing to find ref.pt.

## Practical notes

- `claude-headless` works without any of the above urgency — if you just want to run a solve while deciding, use `config.agent.model = claude-haiku-4-5` and `python3 src/cli.py solve …`.
- Always kill background runs by **exact PID** (`kill $PID`), never `pkill -f` — the watcher commands contain those substrings and `pkill -f` kills the watcher itself (the recurring "exit 144" mystery, finally diagnosed).
- Don't leave untracked work in the repo while `solve` runs (H3: `git clean -fdq` wipes repo-wide).

— end of handoff —

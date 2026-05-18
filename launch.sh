#!/usr/bin/env bash
# Autonomous entrypoint. The ONLY input is a config file; everything
# (model, hardware, exec, gates, retry budget) is defined there.
#
#   ./launch.sh                       # uses ./config.yaml
#   ./launch.sh --config path/to.yaml
#
# Prerequisites, both ONE-TIME and outside this loop:
#   1. env provisioned per AGENTS.md (SETUP)           (once per machine)
#   2. driver.py --phase prep --config <cfg>  (once per model+hardware:
#      profile -> rank -> freeze the immutable contract)
# This script runs only the AUTONOMOUS LOOP (kernel-opt) and refuses to
# run if the frozen contract is missing.
#
# Runs opencode headless with the free OpenCode Zen DeepSeek model as the
# autonomous agent that drives the kernel-opt loop by reading skills/ and
# the given config.
#
# Scaffolding — not executed during setup.
set -euo pipefail

CONFIG="configs/config.yaml"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="${2:?--config needs a path}"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1 (only --config <path>)" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
[[ -f "$CONFIG" ]] || { echo "config not found: $CONFIG" >&2; exit 2; }

# Guard: the loop requires a one-time frozen contract (driver.py --phase prep).
MODEL_ID="$(grep -E '^\s*id:' "$CONFIG" | head -1 | sed -E 's/^\s*id:\s*//; s/\s*#.*$//')"
SLUG="${MODEL_ID//\//__}"
if ! ls targets/"$SLUG"/*/contract.json >/dev/null 2>&1; then
  echo "No frozen contract for '$MODEL_ID' (targets/$SLUG/*/contract.json)." >&2
  echo "Run prep ONCE for this target first:" >&2
  echo "  python3 pipeline/driver.py --phase prep --config $CONFIG" >&2
  exit 3
fi

# Agent model is per-config (Spark vs local): config.yaml: agent.model.
AGENT_MODEL="$(python3 -c 'import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))["agent"]["model"])' "$CONFIG")"
[[ -n "$AGENT_MODEL" ]] || { echo "config missing agent.model: $CONFIG" >&2; exit 2; }

read -r -d '' PROMPT <<EOF || true
You are the autonomous driver for the agentic-kernels pipeline in this repo.

Config file: ${CONFIG}

Everything specific (target model, hardware, exec/ssh+container, gate
thresholds, retry budget, publish policy) is defined in that config. The
skills/ are generic and must be followed as written — do not assume any
particular model or machine; read every specific from the config.

This is the AUTONOMOUS LOOP only. Two precise prerequisites, both done
ONCE outside this loop (do NOT perform them here):
  - environment provisioned per AGENTS.md (SETUP), recorded in config env:
  - frozen contract produced once by: driver.py --phase prep --config ${CONFIG}
If the frozen contract is missing, STOP and tell the user to run prep first.
Do NOT profile, rank, or re-freeze — the loop consumes the immutable contract,
it never regenerates it.

Do exactly this, in order, and stop at the first gate that cannot pass:

1. Load ${CONFIG}. Trust config env: as the toolchain contract (env already
   provisioned). Verify a frozen contract exists under targets/<slug>/*/ —
   if not, stop: "run driver.py --phase prep first". If observed reality
   contradicts config env:, stop and surface it.
2. YOU ARE THE LOOP. driver.py --phase optimize is a SINGLE evaluation
   pass (scaffold -> load -> gates -> result.json -> exit 0/1); it does NOT
   loop or retry. Per skills/kernel-opt/SKILL.md, iterate:
     a. run: python3 pipeline/driver.py --phase optimize --config ${CONFIG}
        (first run scaffolds targets/<slug>/<block>/kernel/ — the universal
        Triton project — with a NotImplementedError seam)
     b. WRITE THE REAL @triton.jit kernel into the scaffold seam
        kernel/torch-ext/<name>/__init__.py (_triton_impl), using the
        inductor study in <block>/inductor/ and the frozen contract. It
        must reproduce golden.pt and beat torch.compile(max-autotune).
     c. re-run driver; read targets/<slug>/<block>/result.json
        (passed / failed_gate / error_class).
     d. if not passed: revise the kernel guided by error_class; repeat
        from (c). Stop after config loop.max_retries attempts.
   The perf bar is torch.compile(max-autotune), never eager. Do not weaken
   gates, the toolchain contract, or the frozen contract to force a pass.
   Never create/push a Hub repo (publish.enabled is false).
   ONLY VALIDATION PATH: edit the seam, then run
     python3 pipeline/driver.py --phase optimize --config ${CONFIG}
   and read result.json. NEVER write your own check against the model or
   golden.pt (no 'python - <<PY ... torch.allclose ... PY', no ad-hoc
   scripts, no ssh/docker): it runs unauthenticated + off the configured
   toolchain, so its numbers are invalid and are NOT the verdict.
3. Report from result.json: pass/fail, the gate speedups, and (when #7 lands)
   the kernel-invoked assertion + end-to-end model speedup. If still failing
   after loop.max_retries, stop and report honestly — never fabricate.

Constraints from config: transformers-only integration, Triton-first, block
is the unit of work. Everything you need is in skills/ and ${CONFIG}.
EOF

# Loop-discipline lock is STRUCTURAL, enforced by ./opencode.json (bash
# denied except the driver). It only holds because we (1) run from repo root
# so opencode loads ./opencode.json, and (2) never pass
# --dangerously-skip-permissions. Do NOT add that flag; do not chdir away.
exec opencode run \
  --model "$AGENT_MODEL" \
  --title "agentic-kernels: ${CONFIG}" \
  "$PROMPT"

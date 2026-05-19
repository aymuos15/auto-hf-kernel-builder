#!/usr/bin/env bash
# Invoke the opencode agent for ONE kernel revision. The agent only edits
# kernel.py (opencode.json: edit allow, bash/webfetch deny). The code loop
# (core/loop.py) owns iteration and runs the benchmark — this script does
# not loop and does not benchmark.
#
#   solve/launch.sh <block_dir> <config_path>
set -euo pipefail

BLOCK_DIR="${1:?usage: launch.sh <block_dir> <config_path>}"
CONFIG="${2:?usage: launch.sh <block_dir> <config_path>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

AGENT_MODEL="$(python3 -c 'import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))["agent"]["model"])' "$CONFIG")"

RESULT_NOTE="No result.json yet — this is the FIRST attempt. Write the initial kernel."
if [[ -f "${BLOCK_DIR}/result.json" ]]; then
  RESULT_NOTE="Previous benchmark verdict is in ${BLOCK_DIR}/result.json — read it and revise guided by its error_class."
fi

read -r -d '' PROMPT <<EOF || true
You are the kernel-writer for the agentic-kernels solver. Read
solve/SKILL.md and follow it exactly.

Contract dir: ${BLOCK_DIR}

Do this, then STOP:
1. Read ${BLOCK_DIR}/contract.json and ${BLOCK_DIR}/model_src.py to learn
   the exact computation. The whole Model is the block.
2. Read ${BLOCK_DIR}/kernel.py (the seam). WEIGHTS is the pre-loaded
   frozen state_dict. kernel(*inputs) gets ONLY inputs, positional in
   contract input_order.
3. ${RESULT_NOTE}
4. EDIT ONLY ${BLOCK_DIR}/kernel.py: write/revise a real @triton.jit
   kernel that reproduces golden.pt within rtol/atol AND beats
   torch.compile(max-autotune). Use the frozen WEIGHTS directly.

You have NO shell/bash and must not run anything: a deterministic code
loop scaffolds, benchmarks, retries, keeps-best and reverts. Do not write
self-tests or probes. Do not edit any file other than kernel.py. Do not
copy the reference / torch.matmul / F.linear wholesale to fake a pass.
Produce the best kernel.py you can this turn, then finish.
EOF

opencode run --model "$AGENT_MODEL" --title "solve: $(basename "$BLOCK_DIR")" "$PROMPT"

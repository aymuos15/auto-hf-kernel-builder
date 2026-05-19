#!/usr/bin/env bash
# The one load-bearing kernel-builder command, isolated.
#   build.sh <project_dir> <nix_target>
# Exit codes consumed by builder.py:
#   0  built OK
#   10 nix build failed
# Loadability/correctness is the later phase's job, not build's.
set -uo pipefail

PROJ="${1:?project dir}"
TARGET="${2:?nix target, e.g. path:/abs/kernel#bundle}"
cd "$PROJ"

nix --extra-experimental-features 'nix-command flakes' \
    build --accept-flake-config "$TARGET" -o result || exit 10

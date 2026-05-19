#!/usr/bin/env bash
# Assemble a kernel-builder universal-Triton project from kernel.py.
#   assemble.sh <proj_dir> <pkg> <kernel_py> <flake_src>
set -euo pipefail

PROJ="${1:?project dir}"
PKG="${2:?package name}"
KERNEL_PY="${3:?kernel.py}"
FLAKE_SRC="${4:?flake.nix template}"

mkdir -p "$PROJ/torch-ext/$PKG"
printf '[general]\nname = "%s"\nuniversal = true\n' "$PKG" > "$PROJ/build.toml"
cp "$FLAKE_SRC" "$PROJ/flake.nix"
printf 'result\nbuild/\nflake.lock\n' > "$PROJ/.gitignore"
cp "$KERNEL_PY" "$PROJ/torch-ext/$PKG/__init__.py"

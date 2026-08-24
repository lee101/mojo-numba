#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_dir/dist"
mojo build --emit shared-lib "$repo_dir/src/kernels.mojo" \
  -o "$repo_dir/dist/libmojo-numba.so"

if mojo build --emit shared-lib "$repo_dir/src/gpu.mojo" \
     -o "$repo_dir/dist/libmojo-numba-gpu.so" 2>"$repo_dir/dist/gpu-build.log"; then
  echo "built dist/libmojo-numba-gpu.so"
else
  echo "GPU kernels not built (see dist/gpu-build.log); CPU path is unaffected" >&2
  rm -f "$repo_dir/dist/libmojo-numba-gpu.so"
fi

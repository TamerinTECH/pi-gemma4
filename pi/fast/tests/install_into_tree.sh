#!/usr/bin/env bash
# Copy the fastcheck tool into a llama.cpp tree (a COPY, never the reference tree) and register it as an example.
# Usage: install_into_tree.sh <llama.cpp-tree>
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
tree=$1
[ -f "$tree/examples/CMakeLists.txt" ] || { echo "$tree is not a llama.cpp tree"; exit 3; }
mkdir -p "$tree/examples/pigemma-fastcheck"
cp "$here/fast_check.cpp" "$here/CMakeLists.txt" "$tree/examples/pigemma-fastcheck/"
grep -q "pigemma-fastcheck" "$tree/examples/CMakeLists.txt" || printf '\nadd_subdirectory(pigemma-fastcheck)\n' >> "$tree/examples/CMakeLists.txt"
echo "installed into $tree"

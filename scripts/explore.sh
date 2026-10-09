#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for tag in qwen06b gemma1b gemma4e2b gptoss20b gemma270m; do
  .venv-gpu/bin/python tools/gpu/explore.py $tag levels hybrid refine sweep > /dev/null 2>> results/explore-$tag.log || echo "explore $tag failed" >> results/explore-failures.txt
done

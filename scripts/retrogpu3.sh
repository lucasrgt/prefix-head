#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet frugal-retrogpu2; do sleep 30; done
run() { tag=$1; shift; .venv-gpu/bin/python tools/gpu/retro.py $tag "$@" > /dev/null 2>> results/retro-gpu-$tag.log || echo "retrogpu $tag failed" >> results/retro-failures.txt; }
run qwen06b prefix freq id random sem
for tag in gemma1b gemma4e2b; do
  while [ ! -f data/retro/$tag-train.json ] && systemctl --user is-active --quiet frugal-extract; do sleep 60; done
  [ -f data/retro/$tag-train.json ] && RETRO_NO_ADAPT=1 run $tag prefix freq id random sem g4@128 prefix@128
done

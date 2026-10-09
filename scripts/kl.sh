#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet frugal-retrog4; do sleep 30; done
for tag in gemma270m qwen06b gemma1b gemma4e2b; do
  .venv-gpu/bin/python tools/gpu/kl.py $tag > /dev/null 2>> results/retro-kl.log || echo "kl $tag failed" >> results/retro-failures.txt
done

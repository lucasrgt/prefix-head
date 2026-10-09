#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet frugal-retrogpu; do sleep 30; done
.venv-gpu/bin/python tools/gpu/retro.py gemma270m prefix g4@128 prefix@128 > /dev/null 2> results/retro-gpu-gemma270m-2.log || echo "retrogpu2 failed" >> results/retro-failures.txt

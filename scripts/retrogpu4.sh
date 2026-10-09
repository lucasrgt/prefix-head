#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet frugal-retrogpu3 || systemctl --user is-active --quiet frugal-retrog4 || systemctl --user is-active --quiet frugal-kl; do sleep 60; done
while [ ! -f data/retro/gptoss20b-train.json ] && systemctl --user is-active --quiet frugal-extract-oss; do sleep 60; done
[ -f data/retro/gptoss20b-train.json ] && RETRO_NO_ADAPT=1 .venv-gpu/bin/python tools/gpu/retro.py gptoss20b prefix freq id random sem > /dev/null 2>> results/retro-gpu-gptoss20b.log || echo "retrogpu gptoss20b failed" >> results/retro-failures.txt

#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet frugal-qat; do sleep 60; done
export MATMUL_NUM_THREADS=8
base="--model mingru --core sosc --alpha grp-first-91:bpe8192 --h 128 --layers 2 --seq 64 --batch 16 --epochs 3 --stateful 64 --seed 1"
for cfg in "--n 51 --slow 26 --period 4" "--n 51 --slow 26 --period 8" "--n 80 --slow 50 --period 8" "--n 96 --slow 64 --period 8"; do
  ./target/release/alpha $base $cfg > /dev/null 2> "results/twospeed-${cfg// /_}.log" || echo "twospeed $cfg failed" >> results/twospeed-failures.txt
done

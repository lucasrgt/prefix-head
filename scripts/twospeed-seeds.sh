#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export MATMUL_NUM_THREADS=8
base="--model mingru --core sosc --alpha grp-first-91:bpe8192 --h 128 --layers 2 --seq 64 --batch 16 --epochs 3 --stateful 64"
for seed in 2 3; do
  for cfg in "--n 51 --slow 26 --period 8" "--n 80 --slow 50 --period 8"; do
    ./target/release/alpha $base $cfg --seed $seed > /dev/null 2> "results/twospeed-${cfg// /_}-s$seed.log" || echo "twospeed $cfg $seed failed" >> results/twospeed-failures.txt
  done
done

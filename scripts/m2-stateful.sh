#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MATMUL_NUM_THREADS=8
common="--alpha grp-first-91:bpe8192 --h 128 --layers 2 --seed 1 --chrono 1024 --seq 64 --batch 16"
for k in 64 256; do
  ./target/release/alpha --model mingru $common --stateful $k --epochs 3 > /dev/null 2> "results/mingru-stateful$k.log"
  ./target/release/alpha --model m2 $common --stateful $k > "results/m2-stateful$k.txt" 2>&1
done

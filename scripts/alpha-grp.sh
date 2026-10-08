#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MATMUL_NUM_THREADS=8
for seed in 1 2; do
  for a in ndcf91x91:bpe8192 grp-freq-91:bpe8192 grp-first-91:bpe8192 grp-recipe-91:bpe8192 grp-last-91:bpe8192; do
    ./target/release/alpha --alpha "$a" --epochs 3 --seed "$seed" 2> "results/alpha-${a//:/_}-s$seed.log"
  done
done

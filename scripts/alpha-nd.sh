#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MATMUL_NUM_THREADS=8
for a in syl nd16x16:bytes nd91x91:bpe8192 nd8x8x8x16:bpe8192 nd2x2x2x2x2x2x2x2x2x2x2x2x2:bpe8192 nd1:bpe8192; do
  ./target/release/alpha --alpha "$a" --epochs 3 2> "results/alpha-${a//:/_}.log"
done

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet frugal-nd; do sleep 10; done
export MATMUL_NUM_THREADS=8
for a in ndf16x16:bytes ndcf16x16:bytes ndcf91x91:bpe8192 ndcf8x8x8x16:bpe8192 ndf8x8x8x16:bpe8192; do
  ./target/release/alpha --alpha "$a" --epochs 3 2> "results/alpha-${a//:/_}.log"
done

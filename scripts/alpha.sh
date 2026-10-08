#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MATMUL_NUM_THREADS=8
for a in bytes braille nibbles bits bpe512 bpe2048 bpe8192; do
  ./target/release/alpha --alpha "$a" --epochs 3 2> "results/alpha-$a.log"
done

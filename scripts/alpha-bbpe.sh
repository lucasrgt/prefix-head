#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet frugal-alpha; do sleep 10; done
export MATMUL_NUM_THREADS=8
for a in bbpe512 bbpe2048 bbpe1024 bpe1024; do
  ./target/release/alpha --alpha "$a" --epochs 3 2> "results/alpha-$a.log"
done

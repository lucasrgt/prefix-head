#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet frugal-alpha || systemctl --user is-active --quiet frugal-bbpe; do sleep 10; done
export MATMUL_NUM_THREADS=8
for a in bpe512-sig bbpe512-sig bpe2048-sig bbpe2048-sig bpe512-wp bbpe512-wp bpe2048-wp bbpe2048-wp; do
  ./target/release/alpha --alpha "$a" --epochs 3 2> "results/alpha-$a.log"
done

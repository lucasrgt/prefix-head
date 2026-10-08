#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet frugal-bbpe || systemctl --user is-active --quiet frugal-crit; do sleep 10; done
export MATMUL_NUM_THREADS=8
for a in g2 g2bpe512 g2bpe1024 g2bpe2048 codepoints capbytes cpbpe512 capbpe512 cpbpe2048 capbpe2048 octal morse; do
  ./target/release/alpha --alpha "$a" --epochs 3 2> "results/alpha-$a.log"
done

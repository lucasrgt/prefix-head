#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet frugal-v32k; do sleep 30; done
export MATMUL_NUM_THREADS=8
s="--model mingru --core sosc --n 51 --h 128 --layers 2 --seq 64 --batch 16 --epochs 1 --stateful 64 --seed 1 --extra 1"
for g in grp-first-317 grp-freq-317 ndcf316x317; do
  ./target/release/alpha $s --alpha $g:bpe100000 > /dev/null 2> results/v100k-$g.log || echo "v100k $g failed" >> results/v100k-failures.txt
done

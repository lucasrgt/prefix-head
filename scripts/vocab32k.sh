#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet frugal-trie; do sleep 30; done
export MATMUL_NUM_THREADS=8
s="--model mingru --core sosc --n 51 --h 128 --layers 2 --seq 64 --batch 16 --epochs 3 --stateful 64 --seed 1"
for g in grp-first-181 grp-freq-181 ndcf182x181; do
  ./target/release/alpha $s --alpha $g:bpe32768 > /dev/null 2> results/v32k-$g.log
done

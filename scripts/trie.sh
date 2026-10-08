#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet frugal-pending || systemctl --user is-active --quiet frugal-mamba; do sleep 30; done
export MATMUL_NUM_THREADS=8
s="--model mingru --core sosc --n 51 --h 128 --layers 2 --seq 64 --batch 16 --epochs 3 --stateful 64"
for seed in 1 2; do
  for g in grp-trie-91 grp-random-91; do
    ./target/release/alpha $s --alpha $g:bpe8192 --seed $seed > /dev/null 2> results/trie-$g-s$seed.log
  done
done

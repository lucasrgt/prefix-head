#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet frugal-extract-oss; do sleep 60; done
export MATMUL_NUM_THREADS=8
for seed in 1 2; do
  ./target/release/alpha --model mingru --core sreal --n 51 --alpha grp-first-91:bpe8192 --h 128 --layers 2 --seq 64 --batch 16 --epochs 3 --stateful 64 --seed $seed > /dev/null 2> results/sreal-seed$seed.log || echo "sreal $seed failed" >> results/sreal-failures.txt
done

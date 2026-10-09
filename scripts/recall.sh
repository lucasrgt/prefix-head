#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet frugal-sreal; do sleep 60; done
export MATMUL_NUM_THREADS=8
common="--alpha nd1:bytes --h 128 --layers 2 --seq 576 --batch 16 --corpus data/mqar/raw"
for seed in 1 2; do
  for core in "sosc --n 51" "sreal --n 51" "tfm --heads 4 --mlp 256"; do
    ./target/release/alpha --model mingru --core $core $common --epochs 30 --seed $seed > /dev/null 2> "results/recall-train-${core%% *}-$seed.log" || echo "recall train $core $seed failed" >> results/recall-failures.txt
    ./target/release/alpha --model recall --core $core $common --seed $seed > /dev/null 2>> results/recall-eval.log || echo "recall eval $core $seed failed" >> results/recall-failures.txt
  done
done

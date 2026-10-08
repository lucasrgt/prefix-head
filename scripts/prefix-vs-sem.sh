#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet frugal-lean; do sleep 30; done
export MATMUL_NUM_THREADS=8
for seed in 1 2; do
  ./target/release/alpha --alpha grp-sem-91:bpe8192 --epochs 3 --seed $seed > /dev/null 2> results/pvs-mlp-sem-s$seed.log
  ./target/release/alpha --model mingru --core sosc --n 51 --alpha grp-sem-91:bpe8192 --h 128 --layers 2 --seq 64 --batch 16 --epochs 3 --seed $seed --stateful 64 > /dev/null 2> results/pvs-sosc-sem-s$seed.log
done
./target/release/alpha --model mingru --core sosc --n 51 --alpha grp-first-91:bpe8192 --h 128 --layers 2 --seq 64 --batch 16 --epochs 3 --seed 2 --stateful 64 > /dev/null 2> results/pvs-sosc-first-s2.log

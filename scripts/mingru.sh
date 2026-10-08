#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MATMUL_NUM_THREADS=8
a=grp-first-91:bpe8192
run() { ./target/release/alpha --model mingru --alpha "$a" --h 128 --layers 2 --epochs 3 "$@" 2> "results/mingru-$(echo "$@" | tr ' -' '_').log"; }
run --seq 64 --batch 16 --seed 1
run --seq 64 --batch 16 --seed 2
run --seq 256 --batch 4 --seed 1

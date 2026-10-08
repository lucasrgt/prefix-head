#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MATMUL_NUM_THREADS=8
common="--model mingru --core osc --n 64 --alpha grp-first-91:bpe8192 --h 128 --layers 2 --seq 64 --batch 16 --epochs 3"
./target/release/alpha $common --seed 1 > /dev/null 2> results/osc-s1.log
./target/release/alpha $common --seed 2 > /dev/null 2> results/osc-s2.log
./target/release/alpha $common --seed 1 --stateful 64 > /dev/null 2> results/osc-stateful64.log
./target/release/alpha --model m2 --core osc --n 64 --alpha grp-first-91:bpe8192 --h 128 --layers 2 --seq 64 --seed 1 --stateful 64 > results/m2-osc-stateful64.txt 2>&1

#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet frugal-pending; do sleep 30; done
export MATMUL_NUM_THREADS=8
m="--model mingru --core mamba --h 128 --n 16 --heads 4 --alpha grp-first-91:bpe8192 --seq 64 --batch 16 --epochs 3 --stateful 64"
./target/release/alpha $m --layers 2 --seed 1 > /dev/null 2> results/mamba-L2-s1.log
./target/release/alpha $m --layers 2 --seed 2 > /dev/null 2> results/mamba-L2-s2.log
./target/release/alpha $m --layers 1 --seed 1 > /dev/null 2> results/mamba-L1-s1.log

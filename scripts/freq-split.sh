#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet frugal-v32k; do sleep 30; done
export MATMUL_NUM_THREADS=8
s="--core sosc --n 51 --h 128 --layers 2 --seq 64 --stateful 64 --seed 1"
./target/release/alpha --model mingru $s --alpha grp-freq-91:bpe8192 --batch 16 --epochs 3 > /dev/null 2> results/freq-split-train.log
./target/release/alpha --model split $s --alpha grp-freq-91:bpe8192 > results/freq-split.txt 2>&1

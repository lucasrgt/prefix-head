#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet frugal-lean2; do sleep 30; done
export MATMUL_NUM_THREADS=8
t="--model mingru --core tfm --heads 4 --h 128 --seq 64 --batch 16 --epochs 3 --seed 1"
./target/release/alpha $t --layers 1 --mlp 128 --alpha grp-first-91:bpe8192 > /dev/null 2> results/tfm-T1.log
./target/release/alpha $t --layers 2 --mlp 256 --alpha grp-first-91:bpe8192 > /dev/null 2> results/tfm-T2.log
./target/release/alpha $t --layers 2 --mlp 256 --alpha nd1:bpe8192 > /dev/null 2> results/tfm-T2-flat.log

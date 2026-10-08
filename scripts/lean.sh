#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet frugal-extra; do sleep 30; done
export MATMUL_NUM_THREADS=8
base="--model mingru --core sosc --n 51 --h 128 --layers 2 --seq 64 --batch 16 --seed 1 --stateful 64 --extra 1"
run() { tag=$1; shift; ./target/release/alpha $base "$@" > /dev/null 2> "results/lean-$tag.log"; }
run uniform-half --alpha grp-first-91:bpe8192 --epochs 0.5
run easy --alpha grp-first-91:bpe8192 --epochs 0.5 --select easy
run hard --alpha grp-first-91:bpe8192 --epochs 0.5 --select hard
run random --alpha grp-first-91:bpe8192 --epochs 0.5 --select random
run lp --alpha grp-first-91:bpe8192 --epochs 0.5 --policy lp
run sbpe --alpha grp-first-181:sbpe32768-8192 --epochs 1
run bpe32k --alpha grp-first-181:bpe32768 --epochs 1

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet frugal-fetch; do sleep 30; done
export MATMUL_NUM_THREADS=8
base="--alpha grp-first-91:bpe8192 --h 128 --layers 2 --seq 64 --batch 16 --seed 1 --stateful 64 --extra 1"
./target/release/alpha --model mingru --core sosc --n 51 $base --epochs 1 > /dev/null 2> results/extra-sosc.log
./target/release/alpha --model m2 --core sosc --n 51 $base > results/m2-extra-sosc.txt 2>&1
./target/release/alpha --model mingru --core mingru --chrono 1024 $base --epochs 1 > /dev/null 2> results/extra-mingru.log
./target/release/alpha --model m2 --core mingru --chrono 1024 $base > results/m2-extra-mingru.txt 2>&1

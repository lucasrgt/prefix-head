#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet frugal-tfm; do sleep 30; done
export MATMUL_NUM_THREADS=8
s="--model mingru --core sosc --n 51 --h 128 --layers 2 --seq 64 --batch 16 --epochs 3 --seed 1 --stateful 64"
for g in grp-first-91 grp-sem-91 grp-second-91 ndcf91x91 grp-first-32 grp-first-64 grp-first-128 grp-first-181; do
  ./target/release/alpha $s --alpha $g:bpe8192 > /dev/null 2> "results/ablate-$g.log"
done

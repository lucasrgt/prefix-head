#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet frugal-tfm || systemctl --user is-active --quiet frugal-ablate; do sleep 30; done
export MATMUL_NUM_THREADS=8
fail() { echo "$1" >> results/pending-failures.txt; }
sosc="--model mingru --core sosc --n 51 --h 128 --layers 2 --seq 64 --batch 16 --stateful 64"
tfm="--model mingru --core tfm --heads 4 --h 128 --layers 2 --mlp 256 --seq 64 --batch 16"
# 1. direct confrontation on the extended corpus with a 32k vocabulary
./target/release/alpha $tfm --alpha grp-first-181:bpe32768 --extra 1 --epochs 1 --seed 1 > /dev/null 2> results/pend-tfm32k-extra.log || fail tfm32k
./target/release/alpha $sosc --alpha grp-first-181:bpe32768 --extra 1 --epochs 1 --seed 2 > /dev/null 2> results/pend-sosc32k-extra-s2.log || fail sosc32k-s2
# 2. third seeds for prefix vs semantic
for g in grp-first-91 grp-sem-91; do
  ./target/release/alpha --alpha $g:bpe8192 --epochs 3 --seed 3 > /dev/null 2> results/pend-mlp-$g-s3.log || fail mlp-$g
  ./target/release/alpha $sosc --alpha $g:bpe8192 --epochs 3 --seed 3 > /dev/null 2> results/pend-sosc-$g-s3.log || fail sosc-$g
done
# 3. English replication (prefix vs merge order vs frequency)
for seed in 1 2; do
  for g in grp-first-91 ndcf91x91 grp-freq-91; do
    ./target/release/alpha --corpus data/en/raw --alpha $g:bpe8192 --epochs 3 --seed $seed > /dev/null 2> results/pend-en-mlp-$g-s$seed.log || fail en-mlp-$g-$seed
  done
done
for g in grp-first-91 ndcf91x91; do
  ./target/release/alpha $sosc --corpus data/en/raw --alpha $g:bpe8192 --epochs 3 --seed 1 > /dev/null 2> results/pend-en-sosc-$g.log || fail en-sosc-$g
done

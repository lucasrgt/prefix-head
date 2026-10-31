#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MATMUL_NUM_THREADS=8
run() {
  local tag=$1 seed=$2; shift 2
  ./target/release/alpha --model mingru --core sosc --n 51 --h 128 --layers 2 --seq 64 --batch 16 --epochs 3 --stateful 64 --seed $seed "$@" > /dev/null 2> results/fractal-$tag-s$seed.log
}
B2=2x2x2x2x2x2x2x2x2x2x2x2x2
run pfx91 1 --alpha pfx-91x91:bpe8192
for seed in 1 2; do
  for r in 21x21x19 10x10x10x9 $B2; do
    run pfx$r $seed --alpha pfx-$r:bpe8192
    run pfx$r-tied $seed --alpha pfx-$r:bpe8192 --tied 1
  done
  run pfx91-tied $seed --alpha pfx-91x91:bpe8192 --tied 1
  run id21x21x19 $seed --alpha nd21x21x19:bpe8192
  run id$B2 $seed --alpha nd$B2:bpe8192
done

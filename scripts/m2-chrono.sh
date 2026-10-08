#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MATMUL_NUM_THREADS=8
common="--alpha grp-first-91:bpe8192 --h 128 --layers 2 --seed 1 --chrono 1024"
for cfg in "--seq 64 --batch 16" "--seq 256 --batch 4"; do
  tag=$(echo "$cfg" | tr ' -' '_')
  ./target/release/alpha --model mingru $common $cfg --epochs 3 > /dev/null 2> "results/mingru-chrono$tag.log"
  ./target/release/alpha --model m2 $common $cfg > "results/m2-chrono$tag.txt" 2>&1
done

#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet frugal-twospeed-seeds; do sleep 60; done
export MATMUL_NUM_THREADS=8
base="--model mingru --core sosc --alpha grp-first-91:bpe8192 --h 128 --layers 2 --seq 64 --batch 16 --epochs 3 --stateful 64 --seed 1"
for cfg in "--n 51 --periods 1,2,4,8" "--n 96 --periods 1,2,4,8,16,32" "--n 128 --periods 1,2,4,8,16,32,64"; do
  ./target/release/alpha $base $cfg > /dev/null 2> "results/dh-${cfg// /_}.log" || echo "dh $cfg failed" >> results/twospeed-failures.txt
done
while systemctl --user is-active --quiet frugal-bitnet; do sleep 60; done
sleep 120
for hb in 8 4; do
  for rep in 1 2 3; do
    ./target/release/gem bench --model data/models/llm/gemma-3-270m-it-qat-Q4_0.gguf --tag gemma270mqat --head2 gemma270mqat-prefix-self --tokens 512 --threads 8 --head-bits $hb > /dev/null 2>> results/gem-bench.log
  done
done

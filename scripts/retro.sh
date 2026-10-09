#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export MATMUL_NUM_THREADS=8
py() { uv run -q --with numpy --with pyyaml --with tqdm --with sentencepiece env PYTHONPATH=.tools/src/llama.cpp/gguf-py python3 "$@"; }
for spec in gemma270m:gemma-3-270m-it-Q8_0 qwen06b:Qwen3-0.6B-Q8_0; do
  tag=${spec%%:*}; f=data/models/llm/${spec#*:}.gguf
  [ -f data/retro/$tag.W ] || py tools/hidden/head.py $f data/retro/$tag.W
  [ -f data/retro/$tag-test.json ] || .tools/hidden $f data/norm/test.txt data/retro/$tag-test 1024 1 8 2> results/retro-$tag-test.log
  [ -f data/retro/$tag-train.json ] || .tools/hidden $f data/norm/train.txt data/retro/$tag-train 1024 3 8 2> results/retro-$tag-train.log
  for g in prefix freq id random sem; do
    ./target/release/retro --model $tag --group $g --epochs 3 > /dev/null 2> results/retro-$tag-$g.log || echo "retro $tag $g failed" >> results/retro-failures.txt
  done
done

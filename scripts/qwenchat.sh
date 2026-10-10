#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
m=data/models/llm/qwen2.5-coder-0.5b-instruct-q4_0.gguf
for split in test train; do
  [ -f data/retro/qwenchat-$split.json ] || ./target/release/gem complete --model $m --tag qwencoder --full --prompts /tmp/claude-1000/-home-lucasrgt/c862457d-4bff-4c57-a8d2-3d3c3bfd3966/scratchpad/mb-$split --max-tokens 400 --dump data/retro/qwenchat-$split > results/qwenchat-$split.log 2>&1
done
ln -sf qwencoder.W data/retro/qwenchat.W; cp data/retro/qwencoder-test.vocab data/retro/qwenchat-test.vocab; cp data/retro/qwencoder-test.vocab data/retro/qwenchat-train.vocab
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RETRO_NO_ADAPT=1
.venv-gpu/bin/python tools/gpu/retro.py qwenchat prefix prefix@128 hyb sem > /dev/null 2>> results/retro-gpu-qwenchat.log || echo "qwenchat retro failed" >> results/retro-failures.txt
.venv-gpu/bin/python tools/gpu/kl.py qwenchat prefix prefix-128 hyb sem > /dev/null 2>> results/retro-kl.log
for head in qwenchat-prefix-self qwenchat-prefix-128-self; do
  for k in 1 2; do
    python3 tools/eval/humaneval.py $m --tag $head-top$k --gem --tag qwencoder --head2 $head --classes $k > /dev/null 2>> results/humaneval.log
  done
done

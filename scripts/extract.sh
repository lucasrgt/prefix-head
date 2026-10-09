#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
py() { uv run -q --with numpy --with pyyaml --with tqdm --with sentencepiece env PYTHONPATH=.tools/src/llama.cpp/gguf-py python3 "$@"; }
[ -f data/models/llm/gemma-4-E2B-it-Q8_0.gguf ] || curl -sL -o data/models/llm/gemma-4-E2B-it-Q8_0.gguf https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q8_0.gguf
echo 30 > data/retro/gemma4e2b.softcap
for spec in gemma1b:gemma-3-1b-pt.Q8_0:3 gemma4e2b:gemma-4-E2B-it-Q8_0:4; do
  IFS=: read tag file every <<< "$spec"; f=data/models/llm/$file.gguf
  [ -f data/retro/$tag.W ] || py tools/hidden/head.py $f data/retro/$tag.W > results/retro-$tag-head.log 2>&1
  [ -f data/retro/$tag-test.json ] || .tools/hidden $f data/norm/test.txt data/retro/$tag-test 1024 1 8 2> results/retro-$tag-test.log
  [ -f data/retro/$tag-train.json ] || .tools/hidden $f data/norm/train.txt data/retro/$tag-train 1024 $every 8 2> results/retro-$tag-train.log
done

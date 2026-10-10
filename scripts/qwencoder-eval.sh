#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet frugal-qwencoder || pgrep -x llama-server > /dev/null; do sleep 30; done
sleep 60
m=data/models/llm/qwen2.5-coder-0.5b-instruct-q4_0.gguf
python3 tools/eval/humaneval.py $m --tag qwencoder-full --gem --tag qwencoder --full > /dev/null 2>> results/humaneval.log
for k in 1 2 4; do
  python3 tools/eval/humaneval.py $m --tag qwencoder-prefix-top$k --gem --tag qwencoder --head2 qwencoder-prefix-self --classes $k > /dev/null 2>> results/humaneval.log
done
python3 tools/eval/humaneval.py $m --tag qwencoder-hyb-top2 --gem --tag qwencoder --head2 qwencoder-hyb-self --classes 2 > /dev/null 2>> results/humaneval.log

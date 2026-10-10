#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet frugal-qwenchat; do sleep 30; done
sleep 60
m=data/models/llm/qwen2.5-coder-0.5b-instruct-q4_0.gguf
python3 tools/eval/humaneval.py $m --tag final-full --gem --tag qwencoder --full > /dev/null 2>> results/humaneval.log
for k in 4 8 16; do
  python3 tools/eval/humaneval.py $m --tag final-prefix-screen$k --gem --tag qwencoder --head2 qwenchat-prefix-self --classes $k > /dev/null 2>> results/humaneval.log
done
python3 tools/eval/humaneval.py $m --tag final-prefix128-screen8 --gem --tag qwencoder --head2 qwenchat-prefix-128-self --classes 8 > /dev/null 2>> results/humaneval.log

#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RETRO_NO_ADAPT=1
for seed in 1 2 3; do
  for tag in gemma270m qwen06b gemma1b gemma4e2b gptoss20b; do
    case $tag in gemma*) groups="hyb hyb@128";; *) groups="hyb";; esac
    RETRO_SEED=$seed .venv-gpu/bin/python tools/gpu/retro.py $tag $groups > /dev/null 2>> results/retro-gpu-hyb.log || echo "hyb $tag $seed failed" >> results/retro-failures.txt
    suffix=""; [ $seed -ne 1 ] && suffix="-s$seed"
    kl=""; for g in $groups; do kl="$kl ${g/@/-}$suffix"; done
    .venv-gpu/bin/python tools/gpu/kl.py $tag $kl > /dev/null 2>> results/retro-kl.log || echo "kl hyb $tag $seed failed" >> results/retro-failures.txt
  done
done

# Code-domain KL test — 2026-11-01

Same protocol as the paper's head swap (frozen model, class predictor trained to reproduce the model's own class
probabilities, no labels; `tools/gpu/retro.py` with `RETRO_NO_ADAPT=1`, `tools/gpu/kl.py`), but on code instead of
Portuguese novels. Corpus: Python 3.14 standard library, ASCII files of 0.5–200 kB, tests/idlelib/lib2to3 excluded,
split by file hash: 460 train files (8.3M chars, 717k rows used, every 3rd window), 63 test files (1.04M chars, all rows).
Extraction: `scripts/code-extract.sh`; llama.cpp a11f57b, Q8_0 GGUFs.

KL(P_full || P_2lvl) in bits/char, 390 classes of 390 (sqrt V), single seed:

| model | prefix | prefix@128 (1187 cl.) | hybrid | semantic | frequency | random | token id |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-0.5B-Instruct | **0.1030** | 0.0935 | 0.1163 | 0.2710 | 0.2840 | 0.3332 | 0.3301 |
| Qwen3-0.6B | **0.0919** | 0.0795 | 0.1026 | 0.2339 | 0.2587 | 0.2926 | 0.2964 |

Prefix is ~61% below the best baseline (semantic) in both models; on Portuguese the paper reports 28% below the best
baseline for Qwen3-0.6B (0.096 vs frequency 0.133). Hybrid is worse than pure prefix here (the paper has it better in 4 of 5 models on Portuguese).

Caveats: one corpus and one language (Python); the stdlib is surely in both models' pretraining data (full-head bpc is
only 0.447 for the coder); one seed; KL only — task accuracy is the HumanEval/MBPP runs in `humaneval.jsonl`, where
the plain 1–4-class head collapses (1.8–3% pass@1) and only screening 8–64 classes recovers the full-head score.

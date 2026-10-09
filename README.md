# prefix-head

**Sort the vocabulary: prefix-ordered token classes make two-level language-model heads cheap and faithful.**

Paper: [`paper/main.pdf`](paper/main.pdf) (source: [`paper/main.tex`](paper/main.tex)).

In small language models with large vocabularies, the output head is a large share of the cost of every generated token (63% of the weight bytes read per token in Gemma 3 270M). A two-level head first predicts a class of tokens, then a token inside it, and reads ~√V× fewer weights. This repository shows that sorting tokens by spelling gives the most faithful head in four of the five models tested: **sort the subword vocabulary by spelling and cut it into equal blocks.**

## Results at a glance

**Swapping the head of frozen open models.** KL divergence of the two-level head to the original model, bits/char on held-out text (lower is better; mean of 3 seeds for prefix and semantic):

| model | prefix | semantic | frequency | random | token id |
|---|---|---|---|---|---|
| Gemma 3 270M-it | **0.137** | 0.162 | – | – | – |
| Gemma 3 1B-pt | 0.087 | **0.085** | 0.133 | 0.192 | 0.214 |
| Gemma 4 E2B-it | **0.184** | 0.274 | 0.271 | 0.329 | 0.364 |
| Qwen3-0.6B | **0.096** | 0.195 | 0.133 | 0.177 | 0.200 |
| gpt-oss-20b | **0.020** | 0.034 | 0.024 | 0.042 | 0.037 |

At equal cost (2,048 classes of 128), prefix blocks deviate **27–33% less** than the token ordering Google ships with its Gemma 4 drafters, including on Gemma 4 itself.

**End-to-end sequential scoring on a CPU** (Gemma 3 270M, Q8_0, batch 1, our Rust runtime, Ryzen 7 7700): 132 → 321 tokens/s with AVX-512 (**2.4×**), 119 → 256 with AVX2 only (**2.1×**).

**Why it works.** The class decision follows the generation direction; the gain concentrates on tokens that continue a word, which is also why alphabetical classes lost at the whole-word level in earlier work (Jiang et al., IJCAI 2017).

## Repository layout

| path | what |
|---|---|
| `crates/nn` | from-scratch models: factored heads, recurrent cores (`sosc` = the selective-oscillator core), Transformer, Mamba-style baseline |
| `crates/alpha` | training/evaluation CLI for the from-scratch experiments |
| `crates/gem` | minimal Gemma 3 CPU runtime (GGUF Q8_0, AVX-512/AVX2) with full and two-level heads |
| `crates/retro`, `tools/gpu/retro.py`, `tools/gpu/kl.py` | head swaps on frozen LLMs (CPU and GPU versions) and KL fidelity |
| `crates/tok`, `crates/kern`, `crates/ruler`, `crates/bench` | tokenizers, kernels, corpus/energy utilities, microbenchmarks |
| `tools/hidden` | llama.cpp-based extraction of final hidden states and output rows |
| `scripts/` | the exact job queues used for every result |
| `results/*.jsonl` | raw numbers behind every table |
| `paper/` | the preprint |

## Reproducing

Requirements: Linux x86-64, [mise](https://mise.jdx.dev) (pins the Rust toolchain), Python 3 with [uv](https://docs.astral.sh/uv/), and for the LLM experiments an NVIDIA GPU with 8 GB.

```bash
# data: public-domain Gutenberg books
tools/fetch_pt.sh                     # 14 Portuguese novels (main results)
tools/fetch_en.sh                     # 14 English novels (replication)
tools/fetch_extra.sh                  # 638 extra Portuguese books (larger-vocabulary runs)
mise exec -- cargo build --release
./target/release/alpha --model norm   # writes data/norm/{train,test}.txt used by the LLM experiments

# from scratch, e.g. the selective-oscillator core with the prefix head
MATMUL_NUM_THREADS=8 ./target/release/alpha --model mingru --core sosc --n 51 \
  --alpha grp-first-91:bpe8192 --h 128 --layers 2 --seq 64 --batch 16 --epochs 3 --stateful 64 --seed 1
# groupings: grp-first-<m> (prefix), grp-freq-<m>, grp-sem-<m>, grp-random-<m>, ndcf<C>x<m> (merge order)

# head swaps on frozen LLMs: build llama.cpp at commit a11f57b, then the extraction tools
git clone https://github.com/ggml-org/llama.cpp .tools/src/llama.cpp && git -C .tools/src/llama.cpp checkout a11f57b
cmake -S .tools/src/llama.cpp -B .tools/src/llama.cpp/build -DBUILD_SHARED_LIBS=ON && cmake --build .tools/src/llama.cpp/build -j
L=.tools/src/llama.cpp
g++ -O2 -std=c++17 -I$L/include -I$L/ggml/include tools/hidden/hidden.cpp -L$L/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$PWD/$L/build/bin -o .tools/hidden
g++ -O2 -std=c++17 -I$L/include -I$L/ggml/include tools/hidden/tok.cpp -L$L/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$PWD/$L/build/bin -o .tools/tok
uv venv .venv-gpu && VIRTUAL_ENV=.venv-gpu uv pip install torch numpy
# then see scripts/retro.sh, scripts/extract.sh, scripts/extract-oss.sh, scripts/kl.sh, scripts/seeds-retro.sh

# end-to-end CPU benchmark (needs data/retro/gemma270m-prefix-self.* from the head swap)
./target/release/gem check --tokens 700      # agreement with llama.cpp
./target/release/gem bench --tokens 512 --threads 8
```

GGUF models are downloaded from Hugging Face by the scripts (`unsloth/gemma-3-270m-it-GGUF`, `mradermacher/gemma-3-1b-pt-GGUF`, `unsloth/gemma-4-E2B-it-GGUF`, `ggml-org/gpt-oss-20b-GGUF`; `Qwen/Qwen3-0.6B-GGUF`).

## Notes in Portuguese

The research log is in Portuguese: `CHARTER.md` (goals and rules), `HYPOTHESES.md` (ideas, prior art, what was tried), `PAPER.md` (working draft and verdicts), `results/ALPHA.md` (every from-scratch experiment), `BENCH.md`, `DOMAINS.md`, `STATUS.md`. The paper and this README are the English entry points.

## How this was made

Implementation, experiments and drafting were carried out with the assistance of an AI coding agent (Claude). Every number comes from runs on the author's machine; the raw outputs are in `results/` and the queues that produced them in `scripts/`.

## License

MIT — see [LICENSE](LICENSE).

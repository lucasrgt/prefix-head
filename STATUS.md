# Status — pausado em 2026-10-09

O projeto está pausado por decisão do Lucas (foco no Host Point). Tudo abaixo está concluído e reproduzível; nada fica rodando.

## Onde está o resultado

- **Preprint:** `paper/main.tex` → `paper/main.pdf` (7 páginas, inglês, sem pendências). Compilar: `cd paper && ../.tools/bin/tectonic main.tex` (o binário do tectonic fica em `.tools/bin`, fora do git; baixar de github.com/tectonic-typesetting/tectonic se faltar).
- **Resumo em português e veredito:** `PAPER.md`.
- **Hipóteses e o que ficou para depois:** `HYPOTHESES.md` (Quartzo-R, Piroclasma etc. — só registradas, não começadas).

## As duas teses

1. **Cabeça por prefixo** — a contribuição original. Ordenar o vocabulário de subpalavras pela grafia e cortar em blocos dá a cabeça fatorada mais fiel: menor KL ao modelo original em 4 de 5 LLMs (empate no 5º), 27–33% menos KL que a ordenação que o Google publicou no Gemma 4, no mesmo custo; Gemma 3 270M gera 2,4× mais rápido na CPU (2,1× só com AVX2).
2. **Quartzo** — a recorrência é a HGRU do HGRN (Qin et al., NeurIPS 2023); fica como estudo empírico de eficiência. Empata com transformer de 2 camadas com 3,6× menos MACs e estado de 816 B; tirar a rotação piora 0,8% (2 sementes). Teste de recall inconclusivo (nenhum modelo aprendeu no orçamento usado).

## Para publicar (decisão do Lucas)

- arXiv (cs.CL ou cs.LG): primeira submissão exige endosso de alguém que já publicou na área.
- Código público em https://github.com/lucasrgt/prefix-head (MIT), já citado no rodapé do artigo.
- Antes: reler o texto; os números vêm de `results/retro-kl.jsonl`, `results/retro-gpu.jsonl`, `results/gem-bench.jsonl`, `results/alpha.jsonl`.

## Como retomar experimentos

- Toolchain: `mise exec -- cargo build --release` (Rust fixado em `mise.toml`); GPU: `.venv-gpu` (PyTorch, fora do git — recriar com `uv venv .venv-gpu && VIRTUAL_ENV=.venv-gpu uv pip install torch numpy`).
- Troca de cabeça num modelo novo: `.tools/hidden` (extrai estados via llama.cpp, fonte em `tools/hidden/hidden.cpp`), `tools/hidden/head.py` (linhas de saída), `tools/gpu/retro.py <tag> <agrupamentos>` e `tools/gpu/kl.py <tag>`. Exemplos de fila em `scripts/retro*.sh`, `scripts/extract*.sh`.
- Runtime próprio do Gemma 3: `target/release/gem check|bench|gen` (`crates/gem`).
- Treinos do zero: `target/release/alpha` (ver `results/ALPHA.md` para os comandos de cada resultado).
- Dados grandes (34 GB em `data/`, fora do git): modelos GGUF em `data/models/llm`, estados extraídos em `data/retro`. Podem ser apagados; tudo é regenerável pelos scripts.

## Próximos passos se um dia voltar

- Decodificação especulativa: drafter com ordenação do Gemma 4 × prefixo, medindo taxa de aceitação.
- Comparação direta com FlashHead e CORAL; avaliação em tarefas; outros domínios de texto.
- Quartzo: recall com orçamento maior, escala (h 256/512).

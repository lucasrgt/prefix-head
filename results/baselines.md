# Baselines no corpus PT (teste = últimos 10% de 14 romances, 716.784 chars)

Custo condicional: (tamanho(treino+teste) − tamanho(treino)) × 8 / chars de teste.

| método | bpc | tempo p/ 7,2M chars |
|---|---|---|
| gzip -9 | 3.262 | 0.4 s |
| bzip2 -9 | 2.560 | 0.4 s |
| xz -9e | 2.392 | 3.6 s |
| zstd -19 --long=27 | 2.423 | 3.5 s |
| zstd --ultra -22 --long=27 | 2.423 | 3.1 s |

Pendentes: n-grama/PPM, paq8px, SmolLM2-135M (llama.cpp).

# Experimento de alfabetos — 2026-10-08

Corpus: 14 romances PT (Gutenberg), 6,45M chars treino / 717k teste (últimos 10% de cada livro).
Modelo fixo: MLP, contexto ~16 chars, d=min(32,V), h=256, 3 épocas, Adam, 1 semente. Custo = MACs por caractere na inferência.
Dados brutos: results/alpha.jsonl. Código: crates/alpha. Grau 2: tools/louis_encode.py (liblouis 3.39, pt-pt-g2).

## Fronteira de Pareto (bpc × MACs/char)
g2bpe512 (90k, 2.346) → bbpe512 (90k, 2.289) → bbpe512-sig (92k, 2.276) → bbpe1024 (127k, 2.258)
→ bbpe2048 (199k, 2.228) → bbpe2048-sig (205k, 2.206) → bpe8192 (587k, 2.183).
Todo ponto interior da fronteira usa braile grau 1 como base do BPE.

## Conclusões
1. Alfabeto pequeno paga no corpo (muitos passos), vocabulário grande paga na cabeça: bits/octal/Morse/nibbles e bpe8192 estão nos extremos de custo.
2. Base do BPE em braile grau 1 vence bytes em todo tamanho de vocabulário; ganho cai com V (512: −1,3% bpc e −15% MACs; 2048: −0,2%).
3. Ablação (BPE-512): bytes 2.318 → +maiúscula fatorada 2.316 → +acento 1 símbolo 2.296 → ambos (braile) 2.289. O ganho vem principalmente de letra acentuada = 1 símbolo.
4. SigGain (arXiv 2603.19261) soma ao braile: bbpe2048-sig 2.206 vs bpe2048 2.233. Sobre bytes, só 0,2%.
5. WordPiece puro com V pequeno quase não comprime (~0,95 tok/char) e perde.
6. Braile grau 2 (contrações humanas, liblouis) comprime 21% mas é pior de prever: perde para grau 1 em todo V.
7. Morse (2.708) perde para octal (2.591): código de entropia inglês não ajuda em PT e o símbolo de fim de letra custa passos.
8. Baselines: xz -9e 2.392 bpc; 21 de 31 runs vencem o xz.

## Limitações
Uma semente por run; diferenças < ~0,5% precisam de 3–5 sementes. Modelo MLP de 16 chars de contexto; efeitos podem mudar com recorrência/atenção.

# Tokens fatorados (N-D) — 2026-10-08 (crates/alpha/src/fmlp.rs, factored.rs, syll.rs)

Modelo fatorado: corpo 1× por token; fatores previstos em sequência, h_q = tanh(pre + Σ_{g<q} C_g[c_g]).
Flags: `c` = matriz de saída por prefixo (softmax por classes; params ~ produto, MACs = soma); `f` = embedding de entrada por token (não soma de dígitos).

| run | bpc | kMACs/char | params |
|---|---|---|---|
| nd1:bytes (validação; antigo 2.343) | 2.347 | 208 | 279k |
| nd16x16:bytes | 2.467 | 150 | — |
| ndc16x16:bytes | 2.423 | 150 | — |
| ndf16x16:bytes | 2.391 | 150 | 164k |
| **ndcf16x16:bytes** | **2.355** | 150 | 226k |
| nd1:bpe8192 (antigo 2.183) | 2.184 | 587 | — |
| nd91x91:bpe8192 | 2.796 | 24 | — |
| nd8x8x8x16:bpe8192 | 2.880 | 14 | — |
| nd2^13:bpe8192 | 2.928 | 13 | — |
| ndf8x8x8x16:bpe8192 | 2.710 | 14 | 324k |
| **ndcf91x91:bpe8192** | **2.249** | **24** | 2.50M |
| ndcf8x8x8x16:bpe8192 | 2.301 | 14 | 2.57M |
| syl (hangul PT, sem correções) | 2.432 | 95 | 431k |

Conclusões:
1. Fatorar ingenuamente perde muito: entrada como soma de dígitos e saída compartilhada entre prefixos.
2. Com as duas correções, bytes 2D fica a 0,35% do 1D com −28% MACs; BPE-8192 2D fica a 3% do 1D com 24× menos MACs e domina a fronteira antiga abaixo de 2,25 bpc (bbpe1024: 2.258 @127k).
3. Com eixos arbitrários (ordem de fusão do BPE), 2D é o ponto ideal; 4D economiza pouco e perde 2,3%.
4. Para bytes a correção de entrada pesa mais; para BPE-8192, a de saída.
5. Mecanismo = softmax por classes (Goodman 2001; Mikolov 2011), abandonado no regime GPU/modelo grande; no regime CPU/modelo pequeno volta a valer. Custo: memória (2,5M params ≈ 10 MB f32) — ternarizar.
6. Achado contra a prática: somar embeddings de dígitos na entrada (como em Leviathan, arXiv 2601.22040) custa qualidade em modelo pequeno.

Próximos: 3 sementes (1D vs ndcf 2D); varrer vocabulário 8k→65k com cabeça 2D; eixos com significado (morfológicos) + generalização para palavras inéditas; sílabas com as correções.

Correção (2026-10-08, depois da rodada): o FMlp alocava uma tabela de condicionamento para o último fator, que nunca é lida. As contagens de `params` dos runs `nd*` acima estão infladas em V_último × h (nd1:bpe8192: 4,51M → 2,41M reais; ndcf91x91:bpe8192: 2,50M → 2,48M). Os bpc não mudam de natureza, mas a inicialização muda, então reruns diferem dentro do ruído de semente.

# Agrupamento das classes da cabeça 2D — 2026-10-08 (tok::group, `grp-<chave>-91:bpe8192`)

Mesma forma 91×91, mesmos 2,48M parâmetros, 24.022 MACs/char; só muda quais tokens dividem classe (ordenados pela chave e cortados a cada 91). Duas sementes.

| run | semente 1 | semente 2 | média |
|---|---|---|---|
| ndcf91x91:bpe8192 | 2.2490 | 2.2474 | 2.2482 |
| grp-freq-91:bpe8192 | 2.2449 | 2.2439 | 2.2444 |
| grp-recipe-91:bpe8192 | 2.2433 | 2.2374 | 2.2404 |
| grp-last-91:bpe8192 | 2.2443 | 2.2464 | 2.2454 |
| grp-first-91:bpe8192 | 2.2095 | 2.2056 | 2.2075 |

- **Início (árvore de prefixos) vence por 1,8%** (2,208 vs 2,248), muito acima da variação entre sementes (~0,2%). Fica a 1,1% do BPE-8192 1D (2,184) com 24× menos MACs e empata com bbpe2048-sig (2,206) com 8,5× menos.
- Frequência, receita (AE2) e final ficam a ≤0,4% do controle.
- Leitura: a primeira decisão da cabeça deve seguir a direção do texto (como o pedaço começa); é o LOD dos jogos aplicado à previsão. Agrupar por receita perde porque os grupos são pequenos e a ordem entre eles segue o id do ingrediente, misturando prefixos sem relação dentro de uma classe.

# M1 — primeiro modelo recorrente (minGRU + cabeça por prefixos) — 2026-10-08

`alpha --model mingru --alpha grp-first-91:bpe8192 --h 128 --layers 2` (crates/nn/src/rnn.rs, head.rs). Gradientes conferidos por diferenças finitas; inferência em fluxo reproduz a perda do lote. Mesmo orçamento do MLP 2D: 24.373 vs 24.022 MACs/char; 2,22M parâmetros; **estado = 1.024 bytes**; ~95 s de treino em CPU (8 threads).

| run | semente | bpc (estado carregado) | bpc (reset a cada janela) |
|---|---|---|---|
| seq 64 | 1 | 2.0875 | 2.0916 |
| seq 64 | 2 | 2.0832 | 2.0874 |
| seq 256 | 1 | 2.0856 | 2.0871 |

- **−5,5% bpc contra o MLP 2D com o mesmo custo** (2,085 vs 2,208, média de 2 sementes; variação entre sementes 0,2%). Melhor que o MLP com cabeça cheia BPE-8192 1D (2,184) com 24× menos MACs; 12,8% melhor que xz -9e.
- Estado carregado além da janela de treino ajuda pouco mas sempre (−0,2% nas 3 runs): o estado guarda informação útil além de 64 tokens — pré-requisito do M2 (momentos de pensamento).
- Janela de treino 256 ≈ 64: nesta escala, o ganho veio de sair da janela fixa de 16 chars do MLP; contexto mais longo no treino não ampliou.

# M2 — quanto de um livro cabe no estado (1 KB) — 2026-10-08 (`alpha --model m2`)

Modelo M1 (seed 1, init padrão). 14 livros; primeiros 1.500 tokens do trecho de teste de cada um (77.996 chars; 6.575 em tokens com maiúscula), começando de estados diferentes.

| estado inicial | bpc | bpc maiúsculas |
|---|---|---|
| frio (zero) | 2.1475 | 2.2132 |
| últimos 16 … 16.384 tokens | 2.1473 | 2.2132 |
| livro inteiro (momento) | 2.1473 | 2.2132 |
| livro inteiro, estado 8/4/2/1 bit | 2.1472–2.1474 | 2.2132–2.2133 |
| momento de outro livro | 2.1474 | 2.2132 |

**Negativo:** o estado não carrega nada do livro além de poucos tokens. Causa provável: porta z ≈ 0,5 na init padrão → metade do estado apagada a cada token. Reler os 14 livros (1,70M tokens, sem cabeça) = 6,96 s / 582 J (pacote); restaurar os 14 estados = 13 µs — o ganho de energia existe, falta o estado valer. Correção em teste: init chrono (meias-vidas log-uniformes 2–1024).

## M2 com init chrono e treino com estado contínuo — 2026-10-08

Init chrono (meias-vidas 2–1024) sozinho: o estado passa a durar, mas o modelo nunca viu estados mais velhos que a janela de treino → carregar o estado piora (seq 64: 2,1241 carregado vs 2,0915 reset). Com **treino com estado contínuo** (`--stateful K`: cada fluxo continua K janelas com o estado carregado, BPTT truncado):

| treino | bpc carregado | bpc reset a cada 64 | ganho da memória longa |
|---|---|---|---|
| init padrão (M1) | 2.0875 | 2.0916 | 0,2% |
| chrono + stateful 64 | **2.0755** | 2.1054 | 1,4% |
| chrono + stateful 256 | 2.0767 | 2.1231 | 2,2% |

M2 (14 livros, primeiros 128 tokens do teste de cada um = abrir o livro e continuar):

| estado inicial | stateful 64 | stateful 256 |
|---|---|---|
| frio | 2.1707 | 2.1856 |
| últimos 16 / 256 / 4096 tokens | 2.1553 / 2.1468 / 2.1443 | 2.1657 / 2.1508 / 2.1475 |
| livro inteiro (momento, 1 KB) | **2.1443** | **2.1475** |
| momento 8 bits (256 B) | 2.1444 | 2.1476 |
| momento 4 / 2 / 1 bit | 2.1511 / 2.1656 / 2.1820 | 2.1584 / 2.1777 / 2.1739 |
| momento de outro livro | 2.1648 | 2.1700 |

- O momento carrega conhecimento **específico do livro**: −1,0% contra o momento de outro livro, −1,2 a −1,7% contra o estado frio.
- Horizonte de memória ≈ 4 mil tokens (~15 mil chars): 4096 tokens = livro inteiro. O ganho de qualidade **satura**, ao contrário da curva G ∝ R^0,5 do worlds; o ganho de energia é linear no que se deixaria de reler.
- **256 bytes bastam** (8 bits por número = idêntico); 4 bits já perde parte.
- Reler os 14 livros: 6,8 s, 520–570 J (pacote); restaurar os 14 momentos: ~10 µs. Para a mesma qualidade basta reler ~4k tokens por livro (~16 ms), ainda ~10⁴× mais que restaurar.

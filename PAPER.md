# Cabeça de saída grosso→fino com classes por prefixo

> Escopo: este artigo defende **só** a cabeça por prefixos. Outras frentes ficam fora:
> **frente 2 — pilha frugal** (núcleo recorrente/Quartzo, display list, LOD por token, Clockwork, morph targets, int8/ternário, energia por char);
> **frente 3 — momentos de pensamento** (conhecimento por byte de estado, quantização do estado, ligação com OpenMoments);
> **caderno de laboratório** (braile, osciladores puros, caleidoscópio, seleção de dados, currículo por progresso) — registrados em results/, sem artigo próprio.
> Super-tokens entram aqui como aplicação: a cabeça barata viabiliza vocabulários grandes em modelos pequenos.

Rascunho de artigo curto. Cada afirmação aponta para a evidência em `results/` e o que falta para sustentá-la.

## ⚠ Precedente direto encontrado (2026-10-08)

Jiang et al., "Exploration of Tree-based Hierarchical Softmax for Recurrent Language Models", IJCAI 2017 (Tabela 3): testaram **ordem alfabética cortada em grupos de mesmo tamanho** como critério de árvore (softmax hierárquico binário, **palavras inteiras**, PTB). Resultado: aleatório 189,4 > **alfabético 149,1** > Huffman/frequência 129,3 > Brown 128,8 (perplexidade de teste). Motivação deles: grafias parecidas ⇒ significados parecidos.

**Novo enquadramento:** não é uma ideia nova; é a **inversão de um resultado publicado**. Com palavras inteiras, alfabético perde para frequência/Brown; com **subpalavras BPE**, prefixo ganha de frequência e de k-means semântico. Mecanismo proposto: o começo do próximo *pedaço de palavra* é muito previsível (continua a palavra em andamento), e é isso que a primeira decisão captura; o começo da próxima *palavra inteira* não é. Testes necessários: (1) vocabulário de palavras inteiras no nosso setup — prefixo deve perder para frequência; (2) vocabulário 32k (mais "palavra inteira") — vantagem deve encolher; (3) Brown como baseline.

## Afirmação

Em modelos de linguagem pequenos, uma cabeça de saída em dois níveis (classe, depois token dentro da classe, com uma matriz de saída por classe) corta o custo de saída em ~45× com perda de ~1% em bits/char; e **ordenar os tokens por prefixo** antes de cortar as classes melhora a qualidade em relação a frequência, ordem de fusão do BPE e k-means semântico, porque a primeira decisão segue a direção da geração.

## Fechamento (2026-10-09)

Preprint final em `paper/main.pdf`. Números finais: KL com 3 sementes (prefixo melhor em 4 de 5 modelos, empate no Gemma 3 1B; 27–33% menos KL que a ordenação do Gemma 4 com grupos de 128); CPU ponta a ponta no Gemma 3 270M: 132 → 321 tok/s com AVX-512 (2,4×), 119 → 256 com AVX2 (2,1×), cabeça cheia = 51–57% do tempo; Quartzo sem rotação 2,079 vs 2,063 (rotação ajuda 0,8%); recall inconclusivo. Estado e como retomar: `STATUS.md`.

## Evidência atual — veredito consolidado (2026-10-08)

Modelos treinados do zero (results/alpha.jsonl, ALPHA.md):

| comparação | resultado | robustez | veredito |
|---|---|---|---|
| cabeça 2D vs cheia, BPE-8192, MLP | 2,249 vs 2,184 bpc com 24× menos MACs/char; ~50× menos energia/char (CPU, lote 1) | 1 semente | sustenta |
| prefixo vs fusão/frequência/receita/final, MLP | 2,208 vs 2,240–2,248 | 3 sementes (prefixo) | sustenta |
| prefixo vs semântico, MLP | 2,208 vs 2,258 (média de 3) | 3 sementes | sustenta |
| prefixo vs semântico, Quartzo | 2,0629 vs 2,1426 (média de 3, ±0,003) | 3 sementes | sustenta |
| prefixo vs aleatório, Quartzo | 2,0640 vs 2,1575 (média de 2) | 2 sementes | sustenta |
| árvore de prefixos vs prefixo em blocos, Quartzo | 2,0573 vs 2,0640, +2% MACs | 2 sementes | variante, ganho marginal |
| inglês (MLP) | prefixo 1,889 vs frequência 1,913 vs fusão 1,913 | 2 sementes | replica |
| inglês (Quartzo) | prefixo 1,7622 vs fusão 1,8131 (−2,8%) | 1 semente | replica |
| transformer: prefixo vs cheia | 2,0628 vs 2,0535 (+0,45%) com 4,2× menos MACs | 1 semente | sustenta |
| escala do vocabulário: frequência pior que prefixo em | 8k: 3,5% · 32k: 2,2% · 100k: 1,1% | 1 semente | **vantagem encolhe com o vocabulário ao treinar do zero** |

Troca de cabeça em LLMs reais congelados (results/retro-gpu.jsonl, tools/gpu/retro.py; "sem adaptação" = previsor de classe treinado para reproduzir as probabilidades de classe do próprio modelo; "adaptada" = cabeça cheia e cabeça 2 níveis ajustadas nos mesmos livros, mesmo orçamento):

| modelo | agrupamento | sem adaptação | adaptada | adaptada vs cheia |
|---|---|---|---|---|
| Gemma 3 270M-it (262k, 512×512) | cheia | 2,273 | 1,794 | — |
| | **prefixo** | **2,254** | **1,819** | **+1,3%** |
| | semântico (k-means) | 2,262 | 1,842 | +2,6% |
| | aleatório | 2,283 | 1,872 | +4,3% |
| | frequência | 2,282 | 1,896 | +5,7% |
| | ordem de ID | 2,302 | 1,916 | +6,8% |
| Gemma 3 270M-it (2048×128) | **prefixo** | **2,253** | **1,809** | **+0,8%** |
| | ordenação do Gemma 4 (Google) | 2,260 | 1,849 | +3,0% |
| Qwen3-0.6B (152k, 390×390) | cheia | 1,918 | 1,612 | — |
| | prefixo | 1,942 | 1,661 | +3,0% |

Ressalvas: no Gemma "it", "sem adaptação" ficou abaixo da cabeça cheia — provável efeito de calibração (no Qwen, +1,2%, como esperado); a ordenação do Google foi construída para o Gemma 4 e aplicada ao Gemma 3 (mesmo vocabulário); 1 semente, 1 domínio.

**Fidelidade ao modelo original (métrica principal da troca de cabeça):** KL(P_cheia ‖ P_2níveis) no texto de teste, em bits/char, com o previsor de classe treinado só para reproduzir as probabilidades de classe do próprio modelo (tools/gpu/kl.py, results/retro-kl.jsonl). Independe de calibração — nos modelos "it", o bpc das cabeças fatoradas fica abaixo do da cabeça cheia por suavização, e o ranking por bpc se inverte no Gemma 4 E2B.

| modelo | prefixo | semântico | frequência | aleatório | ordem de ID | prefixo @128 | Gemma 4 (Google) @128 |
|---|---|---|---|---|---|---|---|
| Gemma 3 270M-it (512 grupos) | **0,137** | — | — | — | — | **0,116** | 0,159 |
| Qwen3-0.6B (390 grupos) | **0,096** | 0,192 | 0,133 | 0,177 | 0,200 | — | — |
| Gemma 3 1B base (512 grupos) | 0,087 | **0,086** | 0,133 | 0,192 | 0,214 | **0,072** | 0,109 |
| Gemma 4 E2B-it (512 grupos) | **0,184** | 0,272 | 0,271 | 0,329 | 0,364 | **0,166** | 0,241 |
| gpt-oss-20b (o200k, 201k; 448 grupos) | **0,020** | 0,036 | 0,024 | 0,042 | 0,037 | — | — |

Prefixo é o mais fiel em todos os cinco modelos (três fabricantes, quatro tokenizadores), empatado com o semântico só no Gemma 3 1B. No gpt-oss-20b, a cabeça por prefixo custa +0,5% de bpc sem adaptação (2,697 vs 2,685) com 224× menos contas na cabeça. Com o mesmo custo (2.048 grupos de 128), o prefixo tem KL 27–33% menor que a ordenação publicada pelo Google — inclusive no próprio Gemma 4, para o qual ela foi construída.

Custo medido: cabeça 2 níveis na RTX 4060, fp16, lote 1: 31–46× mais rápida que a cheia para os formatos de Llama 3.2 1B / Qwen3-0.6B / Gemma 3 270M; lote 512: 2–10× (results/gpu-head.txt). Runtime próprio (crates/gem), Gemma 3 270M na CPU: 89 → 211 tok/s gerando (2,4×) — preliminar, CPU compartilhada, 60 tokens.

## Por que funciona (ablação, Quartzo, 14 livros, semente 1)

Perda decomposta em bits/char: 1ª decisão (classe) + 2ª decisão (token dentro da classe).

| agrupamento | classe | dentro | total |
|---|---|---|---|
| 1º caractere (prefixo) | **1,184** | 0,877 | **2,061** |
| 2º caractere | 1,286 | 0,811 | 2,097 |
| ordem de fusão (arbitrário) | 1,296 | 0,843 | 2,139 |
| k-means semântico | 1,448 | 0,690 | 2,138 |

1. **Direção — apoiada.** O prefixo torna a 1ª decisão muito mais fácil (1,18 vs 1,29–1,45); o 2º caractere fica no meio (1,29); quanto mais a 1ª decisão é "o que vem logo depois no texto", mais o contexto a acerta, e isso domina o total.
2. **Separabilidade — refutada.** A hipótese de que classes semânticas tornariam a 2ª decisão difícil (escolher entre sinônimos) é falsa: a 2ª decisão é *mais fácil* no semântico (0,69 vs 0,88). O semântico perde na 1ª decisão.
3. No Quartzo, semântico ≈ arbitrário (2,138 vs 2,139); o ganho do prefixo sobre o arbitrário é 3,7% (MLP: 1,8%) — cresce com a capacidade do núcleo.
4. Granularidade (classes 32/64/128/181): pendente.

## Onde o ganho acontece: começo × continuação de palavra (Quartzo, 14 livros, semente 1)

`alpha --model split`: bits/char separados por tokens que começam uma palavra (527.335 chars) e tokens que continuam uma palavra (189.476 chars).

| agrupamento | começo: classe + dentro = total | continuação: classe + dentro = total |
|---|---|---|
| prefixo | 1,151 + 0,818 = **1,968** | 1,278 + 1,043 = **2,320** |
| ordem de fusão | 1,172 + 0,836 = 2,007 (+1,9%) | 1,641 + 0,863 = 2,504 (+7,9%) |
| k-means semântico | 1,266 + 0,730 = 1,997 (+1,4%) | 1,953 + 0,578 = 2,531 (+9,1%) |
| 2º caractere | 1,220 + 0,773 = 1,994 (+1,3%) | 1,468 + 0,916 = 2,384 (+2,7%) |
| frequência | 1,027 + 0,975 = 2,002 (+1,7%) | 1,542 + 0,959 = 2,501 (+7,8%) |

A vantagem do prefixo é 4–6× maior nas **continuações** e está na 1ª decisão (classe 1,28 vs 1,64–1,95). Consistente com o mecanismo: a primeira decisão prevê como a palavra em andamento continua. Explica o precedente de 2017: com palavras inteiras todo token é começo de palavra, onde o efeito é pequeno. Frequência (o critério que venceu em 2017) segue o mesmo padrão: +1,7% nos começos, +7,8% nas continuações. Pendente: Brown nesse recorte. Teste com vocabulário de palavras + bytes descartado: com 8k palavras, 77% dos tokens seriam bytes.

## Generalidade: transformer (14 livros, semente 1)

| modelo | cabeça | bpc | MACs/char |
|---|---|---|---|
| transformer 2 camadas | cheia 8192 | 2,0535 | 368,6k |
| transformer 2 camadas | prefixo 91×91 | 2,0628 (+0,45%) | 87,3k (4,2× menos) |
| transformer 1 camada | prefixo 91×91 | 2,1099 | 37,9k |

## Aplicação: super-tokens

Quartzo, corpus ampliado: super-BPE 32k (fusões atravessando espaços) + cabeça por prefixos 182×181 → 1,8886 bpc com 22,9k MACs/char e 0,205 tokens/char, contra BPE 8k + 91×91: 1,9344 bpc, 25,1k MACs/char, 0,284 tokens/char. Vocabulário 4× maior, mais barato e 2,4% melhor. Controle BPE 32k comum: pendente.

## Pendências para submissão

- Troca de cabeça: Qwen (demais agrupamentos), Gemma 3 1B base, Gemma 4 E2B com a ordenação do próprio Gemma 4 (rodando).
- Velocidade e energia ponta a ponta no runtime com CPU parada (8 threads, AVX-512 e AVX2), e na GPU.
- Decodificação especulativa: drafter com ordenação do Google × prefixo, taxa de aceitação no mesmo custo.
- Baselines recentes: FlashHead (k-means esférico + multiprobe), CORAL, FR-Spec; Brown clusters.
- Avaliação em tarefas, não só bpc; 3–5 sementes nas trocas de cabeça.
- Escopo explícito: regime limitado por memória (CPU, GPU de consumidor com lote 1, drafters); lotes grandes em servidor ganham menos (2–10×).

## Prior art

### Vizinhos recentes (2024–2026), checados nas fontes em 2026-10-08

Nenhum encontrado agrupa um vocabulário de subpalavras por grafia/prefixo. Todos agrupam por similaridade de embedding, frequência ou ordem de ID — são os baselines a vencer:

- **Gemma 4 Technical Report** (Gemma Team, arXiv 2607.02770, jul/2026), §2.6: nos drafters E2B/E4B, "replacing the projection operation to the entire vocabulary by a top-k operation on clusters of tokens" (d×262k → d×4096, aceitação similar). Config: 2048 centróides × 128 tokens, top-32; `token_ordering` publicado, método de construção não descrito. Inspeção nossa: grupos semânticos/por escrita, pureza de 1º caractere 24% (prefixo 78%, aleatório 7%). Aproximado (descarta tokens fora dos top-32).
- **FlashHead** (Tranheden et al., arXiv 2603.14591, mar/2026): k-means esférico com clusters de tamanho igual, 8.016 clusters, 512 probes, probabilidades por Monte Carlo (aproximado), sem retreino. Gemma-3-270M: cabeça 2,68× mais rápida, 1,35× ponta a ponta. Mesmos modelos que usamos → comparação direta possível.
- **CORAL** (Weng et al., arXiv 2502.16880): roteador do LM head em N grupos iguais, P = roteador × softmax no grupo (fatoração como a nossa), grupos aparentemente por ordem de ID (= nosso controle "id"/ordem de fusão).
- **DynaSpec** (arXiv 2510.13847), **FR-Spec** (ACL 2025, frequência), **VocabTrim** (arXiv 2506.22694, frequência): cabeças reduzidas para drafters.
- **Cut Your Losses in Large-Vocabulary LMs** (Wijmans et al., ICLR 2025): memória de treino da cabeça (exato), não reduz custo de inferência.
- **L2S** (Chen et al., ICLR 2019) e **SVD-Softmax** (Shim et al., NIPS 2017): triagem aproximada de candidatos.

Posição resultante: fatorar a cabeça em grupos (exata ou aproximada) é prática corrente em 2025–26 para drafters/modelos pequenos; nossa contribuição é o **critério** (prefixo), a **explicação** (direção da geração) e a comparação contra os critérios desses trabalhos (semântico = Gemma 4/FlashHead, ID = CORAL, frequência = FR-Spec/VocabTrim).

### Histórico

Checagem 2026-10-08: buscas próprias + Gemini + segundo parecer. **Precedente: Jiang et al. IJCAI 2017 (ordem alfabética, palavras inteiras, perdeu — ver topo).** Nenhum trabalho encontrado com subpalavras BPE ou com a explicação pela direção da geração. Parente mais próximo: **léxico em árvore de prefixos no reconhecimento de fala** — Ney, Haeb-Umbach, Tran & Oerder, "Improvements in Beam Search for 10,000-Word Continuous Speech Recognition", ICASSP 1992 (o Gemini citou um título errado para esse trabalho) — usado para organizar a **busca**, não para fatorar/treinar a saída do modelo de linguagem. Enquadramento sugerido: a árvore de prefixos existe desde 1992 para a busca; aqui ela vira o critério de agrupamento da saída e supera frequência e semântica. Outros vizinhos, todos diferentes: modelos character-aware na entrada (Kim et al. 2016), geração caractere a caractere de palavras fora do vocabulário, decodificadores hierárquicos por patches de bytes (MegaByte 2023; BLT, arXiv 2412.09871), adaptive/differentiated softmax (por frequência), Mixture of Softmaxes.

Softmax por classes (Goodman 2001; Mikolov 2011), hierárquico (Morin & Bengio 2005), Huffman (word2vec), Brown clusters, árvores aprendidas (Self-organized HSM 2017; tHSM IJCAI 2017), adaptive softmax (Grave et al. 2017), k-means semântico (HDLM 2025; TDLM arXiv 2604.03537, que afirma que hierarquias não se alinham com geração esquerda→direita).

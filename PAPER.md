# Cabeça de saída grosso→fino com classes por prefixo

> Escopo: este artigo defende **só** a cabeça por prefixos. Outras frentes ficam fora:
> **frente 2 — pilha frugal** (núcleo recorrente/oscilador seletivo, display list, LOD por token, Clockwork, morph targets, int8/ternário, energia por char);
> **frente 3 — momentos de pensamento** (conhecimento por byte de estado, quantização do estado, ligação com OpenMoments);
> **caderno de laboratório** (braile, osciladores puros, caleidoscópio, seleção de dados, currículo por progresso) — registrados em results/, sem artigo próprio.
> Super-tokens entram aqui como aplicação: a cabeça barata viabiliza vocabulários grandes em modelos pequenos.

Rascunho de artigo curto. Cada afirmação aponta para a evidência em `results/` e o que falta para sustentá-la.

## ⚠ Precedente direto encontrado (2026-10-08)

Jiang et al., "Exploration of Tree-based Hierarchical Softmax for Recurrent Language Models", IJCAI 2017 (Tabela 3): testaram **ordem alfabética cortada em grupos de mesmo tamanho** como critério de árvore (softmax hierárquico binário, **palavras inteiras**, PTB). Resultado: aleatório 189,4 > **alfabético 149,1** > Huffman/frequência 129,3 > Brown 128,8 (perplexidade de teste). Motivação deles: grafias parecidas ⇒ significados parecidos.

**Novo enquadramento:** não é uma ideia nova; é a **inversão de um resultado publicado**. Com palavras inteiras, alfabético perde para frequência/Brown; com **subpalavras BPE**, prefixo ganha de frequência e de k-means semântico. Mecanismo proposto: o começo do próximo *pedaço de palavra* é muito previsível (continua a palavra em andamento), e é isso que a primeira decisão captura; o começo da próxima *palavra inteira* não é. Testes necessários: (1) vocabulário de palavras inteiras no nosso setup — prefixo deve perder para frequência; (2) vocabulário 32k (mais "palavra inteira") — vantagem deve encolher; (3) Brown como baseline.

## Afirmação

Em modelos de linguagem pequenos, uma cabeça de saída em dois níveis (classe, depois token dentro da classe, com uma matriz de saída por classe) corta o custo de saída em ~45× com perda de ~1% em bits/char; e **ordenar os tokens por prefixo** antes de cortar as classes melhora a qualidade em relação a frequência, ordem de fusão do BPE e k-means semântico, porque a primeira decisão segue a direção da geração.

## Evidência atual (results/ALPHA.md)

| comparação | resultado | robustez |
|---|---|---|
| cabeça 2D vs cheia, BPE-8192, MLP | 2,249 vs 2,184 bpc com 24× menos MACs/char; ~50× menos energia/char (bench infer, CPU, lote 1) | 1 semente |
| prefixo vs ordem de fusão / frequência / receita / final, MLP | 2,208 vs 2,240–2,248 | 2 sementes |
| prefixo vs k-means semântico, MLP | 2,2075 vs 2,2577 | 2 sementes |
| prefixo vs k-means semântico, oscilador seletivo | 2,0640 vs 2,1421 | 2 sementes |

## Por que funciona (ablação, oscilador seletivo, 14 livros, semente 1)

Perda decomposta em bits/char: 1ª decisão (classe) + 2ª decisão (token dentro da classe).

| agrupamento | classe | dentro | total |
|---|---|---|---|
| 1º caractere (prefixo) | **1,184** | 0,877 | **2,061** |
| 2º caractere | 1,286 | 0,811 | 2,097 |
| ordem de fusão (arbitrário) | 1,296 | 0,843 | 2,139 |
| k-means semântico | 1,448 | 0,690 | 2,138 |

1. **Direção — apoiada.** O prefixo torna a 1ª decisão muito mais fácil (1,18 vs 1,29–1,45); o 2º caractere fica no meio (1,29); quanto mais a 1ª decisão é "o que vem logo depois no texto", mais o contexto a acerta, e isso domina o total.
2. **Separabilidade — refutada.** A hipótese de que classes semânticas tornariam a 2ª decisão difícil (escolher entre sinônimos) é falsa: a 2ª decisão é *mais fácil* no semântico (0,69 vs 0,88). O semântico perde na 1ª decisão.
3. No oscilador seletivo, semântico ≈ arbitrário (2,138 vs 2,139); o ganho do prefixo sobre o arbitrário é 3,7% (MLP: 1,8%) — cresce com a capacidade do núcleo.
4. Granularidade (classes 32/64/128/181): pendente.

## Onde o ganho acontece: começo × continuação de palavra (oscilador seletivo, 14 livros, semente 1)

`alpha --model split`: bits/char separados por tokens que começam uma palavra (527.335 chars) e tokens que continuam uma palavra (189.476 chars).

| agrupamento | começo: classe + dentro = total | continuação: classe + dentro = total |
|---|---|---|
| prefixo | 1,151 + 0,818 = **1,968** | 1,278 + 1,043 = **2,320** |
| ordem de fusão | 1,172 + 0,836 = 2,007 (+1,9%) | 1,641 + 0,863 = 2,504 (+7,9%) |
| k-means semântico | 1,266 + 0,730 = 1,997 (+1,4%) | 1,953 + 0,578 = 2,531 (+9,1%) |
| 2º caractere | 1,220 + 0,773 = 1,994 (+1,3%) | 1,468 + 0,916 = 2,384 (+2,7%) |

A vantagem do prefixo é 4–6× maior nas **continuações** e está na 1ª decisão (classe 1,28 vs 1,64–1,95). Consistente com o mecanismo: a primeira decisão prevê como a palavra em andamento continua. Explica o precedente de 2017: com palavras inteiras todo token é começo de palavra, onde o efeito é pequeno. Pendente: frequência e Brown nesse recorte (os critérios que venceram em 2017). Teste com vocabulário de palavras + bytes descartado: com 8k palavras, 77% dos tokens seriam bytes.

## Generalidade: transformer (14 livros, semente 1)

| modelo | cabeça | bpc | MACs/char |
|---|---|---|---|
| transformer 2 camadas | cheia 8192 | 2,0535 | 368,6k |
| transformer 2 camadas | prefixo 91×91 | 2,0628 (+0,45%) | 87,3k (4,2× menos) |
| transformer 1 camada | prefixo 91×91 | 2,1099 | 37,9k |

## Aplicação: super-tokens

Oscilador seletivo, corpus ampliado: super-BPE 32k (fusões atravessando espaços) + cabeça por prefixos 182×181 → 1,8886 bpc com 22,9k MACs/char e 0,205 tokens/char, contra BPE 8k + 91×91: 1,9344 bpc, 25,1k MACs/char, 0,284 tokens/char. Vocabulário 4× maior, mais barato e 2,4% melhor. Controle BPE 32k comum: pendente.

## Pendências para submissão

- Transformer do mesmo tamanho: cabeça cheia × prefixo (rodando).
- Vocabulários 8k / 32k / ~100k, cheia × prefixo; latência e energia medidas, não só MACs.
- Brown clusters como baseline adicional.
- Outro idioma (inglês) e outro tokenizador.
- 3–5 sementes por célula, com intervalos de confiança.
- Escopo explícito: medições de tempo/energia em CPU com lote 1; GPU com lotes grandes não avaliada (cf. adaptive softmax, Grave et al. 2017, que organiza a saída pelo hardware).

## Prior art

Checagem 2026-10-08: buscas próprias + Gemini + segundo parecer. **Precedente: Jiang et al. IJCAI 2017 (ordem alfabética, palavras inteiras, perdeu — ver topo).** Nenhum trabalho encontrado com subpalavras BPE ou com a explicação pela direção da geração. Parente mais próximo: **léxico em árvore de prefixos no reconhecimento de fala** — Ney, Haeb-Umbach, Tran & Oerder, "Improvements in Beam Search for 10,000-Word Continuous Speech Recognition", ICASSP 1992 (o Gemini citou um título errado para esse trabalho) — usado para organizar a **busca**, não para fatorar/treinar a saída do modelo de linguagem. Enquadramento sugerido: a árvore de prefixos existe desde 1992 para a busca; aqui ela vira o critério de agrupamento da saída e supera frequência e semântica. Outros vizinhos, todos diferentes: modelos character-aware na entrada (Kim et al. 2016), geração caractere a caractere de palavras fora do vocabulário, decodificadores hierárquicos por patches de bytes (MegaByte 2023; BLT, arXiv 2412.09871), adaptive/differentiated softmax (por frequência), Mixture of Softmaxes.

Softmax por classes (Goodman 2001; Mikolov 2011), hierárquico (Morin & Bengio 2005), Huffman (word2vec), Brown clusters, árvores aprendidas (Self-organized HSM 2017; tHSM IJCAI 2017), adaptive softmax (Grave et al. 2017), k-means semântico (HDLM 2025; TDLM arXiv 2604.03537, que afirma que hierarquias não se alinham com geração esquerda→direita).

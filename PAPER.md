# Cabeça de saída grosso→fino com classes por prefixo

> Escopo: este artigo defende **só** a cabeça por prefixos. Outras frentes ficam fora:
> **frente 2 — pilha frugal** (núcleo recorrente/oscilador seletivo, display list, LOD por token, Clockwork, morph targets, int8/ternário, energia por char);
> **frente 3 — momentos de pensamento** (conhecimento por byte de estado, quantização do estado, ligação com OpenMoments);
> **caderno de laboratório** (braile, osciladores puros, caleidoscópio, seleção de dados, currículo por progresso) — registrados em results/, sem artigo próprio.
> Super-tokens entram aqui como aplicação: a cabeça barata viabiliza vocabulários grandes em modelos pequenos.

Rascunho de artigo curto. Cada afirmação aponta para a evidência em `results/` e o que falta para sustentá-la.

## Afirmação

Em modelos de linguagem pequenos, uma cabeça de saída em dois níveis (classe, depois token dentro da classe, com uma matriz de saída por classe) corta o custo de saída em ~45× com perda de ~1% em bits/char; e **ordenar os tokens por prefixo** antes de cortar as classes melhora a qualidade em relação a frequência, ordem de fusão do BPE e k-means semântico, porque a primeira decisão segue a direção da geração.

## Evidência atual (results/ALPHA.md)

| comparação | resultado | robustez |
|---|---|---|
| cabeça 2D vs cheia, BPE-8192, MLP | 2,249 vs 2,184 bpc com 24× menos MACs/char; ~50× menos energia/char (bench infer, CPU, lote 1) | 1 semente |
| prefixo vs ordem de fusão / frequência / receita / final, MLP | 2,208 vs 2,240–2,248 | 2 sementes |
| prefixo vs k-means semântico, MLP | 2,2075 vs 2,2577 | 2 sementes |
| prefixo vs k-means semântico, oscilador seletivo | 2,0640 vs 2,1421 | 2 sementes |

## Hipóteses sobre o porquê

1. **Direção:** a primeira decisão é como o próximo pedaço começa (alinhada com a geração). Apoio: agrupar pelo final ≈ controle. Teste: agrupar pelo segundo caractere.
2. **Separabilidade:** dentro de uma classe por prefixo, tokens diferem no final e o contexto os distingue; dentro de uma classe semântica, são quase sinônimos. Teste: decompor a perda em −log p(classe) e −log p(token | classe) por agrupamento.
3. **Granularidade:** teste com tamanhos de classe 32/64/91/128/181 e profundidade de prefixo.

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

Softmax por classes (Goodman 2001; Mikolov 2011), hierárquico (Morin & Bengio 2005), Huffman (word2vec), Brown clusters, árvores aprendidas (Self-organized HSM 2017; tHSM IJCAI 2017), adaptive softmax (Grave et al. 2017), k-means semântico (HDLM 2025; TDLM arXiv 2604.03537, que afirma que hierarquias não se alinham com geração esquerda→direita).

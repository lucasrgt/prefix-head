# frugal

**Desperdício mínimo de conhecimento e computação.**

Inteligência por joule, CPU-only, em hardware doméstico. Hiperotimização de cada detalhe desde o dia 1.

## Plêiades (a visão)

Muitos especialistas pequenos e baratos funcionando como um sistema, ligados por roteamento **grosso antes do fino, na ordem em que a saída é produzida**: dentro do token (cabeça por prefixos), entre camadas especialistas (MoE por prefixo), entre especialistas por pedido (sociedade) e entre modelos (rascunho barato, conferência cara). O estado minúsculo de cada núcleo (~1 KB) permite manter muitos prontos. Como as Plêiades: um aglomerado de estrelas pequenas que nasceram juntas e viajam juntas. Hoje só o primeiro nível está provado (PAPER.md); os outros estão em HYPOTHESES.md.

## Objetivo

1. **Tokens por segundo** (decode e prefill) no hardware alvo.
2. **Joules por token** e **joules por problema resolvido**.
3. Qualidade medida em tarefas com resposta verificável, nunca por impressão.

Motivação: IA útil em hardware que já existe, sem GPU e sem data center, com o mesmo método de jogos antigos e da demoscene: primeiro o hardware e o orçamento, depois o design.

## Regras

- **Nenhum detalhe entra sem número.** Toda mudança de kernel, formato ou arquitetura vem com bench antes/depois, no mesmo hardware, com mediana de várias rodadas.
- **Orçamento é contrato.** Se não cabe no orçamento de RAM/cache/latência, está errado, mesmo que funcione.
- **Kernel de referência escalar sempre existe** e todo kernel SIMD é testado bit a bit contra ele.
- **Dispatch em runtime.** Um binário roda em AVX-512, AVX2 e escalar; o hardware antigo é alvo de primeira classe.
- Elixir/BEAM só entra quando houver algo para orquestrar (shards, máquinas, consolidação). O caminho quente é Rust.
- **Granularidade grossa no BEAM.** Um token do modelo 2D custa ~1 µs; uma mensagem BEAM ou chamada NIF custa ~1–5 µs, um salto de rede ~100 µs. Orquestração entra por pedido/tarefa/consulta de memória, nunca no laço de cada token; roteamento fino (ex.: cabeça por classes) fica no Rust. Antes da fase 4, medir o custo real com um NIF do `infer_step`.

## Hardware

| Tier | Máquina | Papel |
|------|---------|-------|
| T0 | Ryzen 7 7700 (Zen 4), 8C/16T, L2 1 MB/núcleo, L3 32 MB, DDR5, AVX-512 VNNI/VPOPCNTDQ/BF16 | dev + teto |
| T1 | mesmo, forçando caminho AVX2 | PCs de ~2015–2020 |
| T2 | T1 + cgroup (2 núcleos, 4 GB) | notebook velho / SBC |

## Princípio de design: o modelo quente mora no cache

O decode lê todos os pesos ativos por token. Na DRAM isso é limitado pela banda (~60 GB/s em DDR5 dual-channel); no L3 a banda é várias vezes maior e a energia por byte muito menor.

| Formato | bits/peso | pesos no L3 de 32 MB | pesos no L2 de 1 MB |
|---------|-----------|----------------------|---------------------|
| f32 | 32 | 8M | 256K |
| int8 | 8 | 32M | 1M |
| ternário 2 bitplanes | 2 | 128M | 4M |
| ternário base-3 (5 trits/byte) | 1.6 | 160M | 5M |

Meta inicial: núcleo de raciocínio que caiba inteiro no L3 do T0 e num L3 de 8 MB no T1.

## Fases

0. **Medidor + kernels.** Energia via RAPL, RSS de pico, latência. Microbench dos kernels de matvec (f32, int8, ternário em vários formatos). Baselines via llama.cpp (SmolLM2-135M, Qwen-0.5B, BitNet 2B).
1. **Núcleo.** Bloco ternário recorrente (mesmos pesos, N iterações). Treino em GPU alugada; inferência em Rust.
2. **Memória externa.** Índice no SSD (suffix array / vizinhos binários), io_uring + prefetch.
3. **Curvas.** Núcleo fixo; varia memória e iterações; mede conhecimento e raciocínio separados, em J por acerto.
4. **Orquestração.** Elixir/BEAM quando houver shards ou várias máquinas.

## Estrutura

| crate | papel |
|-------|-------|
| `kern` | kernels de matvec (escalar de referência + AVX2/AVX-512), testados bit a bit |
| `ruler` | a régua: corpus normalizado, medidor de energia (RAPL), gravação de resultados |
| `tok` | alfabetos e tokenizers (bytes, braile, BPE/SigGain, grau 2, sílabas, dígitos N-D) |
| `nn` | modelos: MLP e MLP fatorado (cabeça por prefixo, entrada por token) |
| `bench` | microbench dos kernels |
| `alpha` | executores dos experimentos de alfabeto e de tokens fatorados |

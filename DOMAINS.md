# Domínios

Busca sistemática por algoritmos "elo perdido" fora da IA. Cada linha: princípio de eficiência → algoritmo candidato → como medir na régua (bpc, MACs/char, bytes de peso/token, J/char). Quando um vira experimento, ganha entrada em HYPOTHESES.md.

| domínio | princípio | algoritmo candidato | medir | status |
|---|---|---|---|---|
| Jogos | LOD / mipmap: grosso primeiro, detalhe depois | cabeça por classes ordenadas por prefixo | bpc a MACs fixos | **testando** (grp-first: −1,8% bpc, s1) |
| Jogos | culling: não calcular o que não aparece | saída antecipada / pular camadas quando a decisão já é óbvia | fração de passos pulados × bpc | aberto |
| Jogos | cache e pré-computação | tabelas pré-somadas E·W1 (lookup em vez de matvec na entrada) | µs/char, RAM | aberto |
| Cérebro | ~20 W, ativação esparsa ~1–2% | Sparse Distributed Memory (Kanerva 1988), memória de bits | recuperação por bit, J/consulta | aberto (liga H-energia) |
| Cérebro | entonação separada das palavras | forma antes do conteúdo | H-entonação | aberto |
| Sistemas operacionais | memória virtual, paginação, working set | conhecimento paginado do SSD com prefetch | bytes lidos/token, latência | fase 2 |
| Bancos de dados | índices, B-trees, busca sem descomprimir | FM-index / suffix array como memória externa | qualidade × bytes de RAM | fase 2 |
| Física da computação | Landauer: apagar custa energia | atualizações reversíveis / aditivas | J por passo | aberto |
| Cristalografia | célula unitária + simetria | pesos circulantes / equivariantes | H-cristal | aberto |
| Álgebra | números hipercomplexos | camadas quaterniônicas / PHM | H-hipercomplexo | aberto |
| Escrita humana | braile, hangul, Libras | base do tokenizer, eixos de token | ALPHA.md | feito em parte |
| Minecraft AE2 | tipos × contagem, receitas, canais | agrupamento por receita; dedup tipo/contagem | ALPHA.md | testado (receita ≈ controle) |
| Ferramentas próprias | Moments: saltar para estados caros | checkpoints de treino; momentos de pensamento | H-momentos | próximo |

## Leituras

- MacKay, *Information Theory, Inference and Learning Algorithms* (gratuito) — compressão ↔ aprendizado.
- Kanerva, *Sparse Distributed Memory* (1988).
- Hawkins, *A Thousand Brains* (2021).
- Hooker, "The Hardware Lottery" (2020).
- Sutton, "The Bitter Lesson" (2019) — o contra-argumento: método genérico + mais computação vence estrutura. Resposta a testar: com computação fixa e pequena, estrutura volta a importar.

## Moments no Mana — os mecanismos (lidos em ~/Projects/hostpoint-live-pilot/framework, 2026-10-08)

| mecanismo no Mana | medida | análogo no frugal |
|---|---|---|
| Declarar uma vez (DSL Ash) e derivar API, cliente Dart, privacidade, limites e o manifesto de Moments | 128 Moments nos 5 apps; bugs de segurança achados pelo mapeamento | experimento declarado uma vez → runs, esquema de resultados, tabelas derivados |
| Linhagem `from(pai)`: o materializador roda cada ancestral **uma vez**, captura as camadas e abre cópias independentes (`fork`) | — | treinar o tronco uma vez e ramificar as cabeças; prefixo de contexto lido uma vez e ramificado |
| Projeção **mínima e tipada**: só campos declarados (enum/intervalo) são persistidos; o resto é recomputado de fontes duráveis (reautentica, recarrega da API) | `open` 118–223 ms; patch → frame 26 ms | estado de pensamento = só o estado recorrente (h=256 → 1 KB, quantizável), não o KV cache inteiro |
| Salvar Dart → hot restart → retomar a projeção, com API e Postgres vivos | 347 ms compilação + 756 ms retomada = 1,1 s, filtro e scroll preservados | trocar pesos/kernel/formato e continuar do mesmo estado; comparar versões a partir do mesmo ponto |
| `affected`: grafo de imports seleciona só os Moments atingidos; artefato chaveado pelo hash dos fontes não recompila sem mudança | suíte: ~70 s em série → 18 s headless (4 workers), 23 s ponta a ponta | cache por hash (tokenizer, corpus codificado, tronco treinado); só reexecutar experimentos atingidos; inferência incremental após edição do contexto |
| `--headless`: roda o app sem renderizador quando a prova não precisa de pixels | headless 18 s vs navegador 33–36 s | pensar sem "renderizar": pular a cabeça de saída nos passos internos (H-cot-densa) |
| Revisões rejeitam capturas atrasadas | — | consistência ao ramificar estados |

Princípio comum: **não recomputar o que dá para nomear, restaurar ou derivar; persistir só a projeção mínima tipada; recomputar o resto de fontes duráveis; executar só o trabalho atingido.** É o "desperdício mínimo" do CHARTER em forma de software.

## Projetos do próprio usuário: AeroModelLib e Worldline (lidos via GitHub, 2026-10-08)

| origem | princípio | algoritmo candidato | medir | status |
|---|---|---|---|---|
| AeroModelLib: LOD / render distance (animar perto, estático longe) | gastar pelo que importa | saída antecipada por token: tokens fáceis saem após a 1ª camada, guiado pela confiança da classe (cf. CALM, Google 2022) | fração de tokens que saem cedo × bpc × MACs/char | aberto |
| AeroModelLib: display lists (pré-compilar o estático) | não recalcular o constante | projeção da 1ª camada pré-calculada por token (tabela vocab × 3n), consulta em vez de matvec | MACs/char (−~20%), µs/char, bpc idêntico | aberto, ganho garantido |
| AeroModelLib: keyframes + interpolação | calcular pouco e interpolar | recorrência multi-velocidade (Clockwork RNN, 2014): canais atualizados a cada 1/4/16 tokens | bpc × MACs/char | aberto |
| AeroModelLib: morph targets | forma = mistura de poucas bases | pesos dinâmicos: W = Σ α_k(contexto)·W_k com poucas bases | bpc × params × MACs | aberto |
| Worldline: provas endereçadas por conteúdo | não refazer o que não mudou | cache de resultados por hash (código + args + dados); só reexecutar o atingido | tempo de pesquisa | aberto (metodologia) |
| Worldline: oráculo oficial + determinismo + fail closed | evidência antes de afirmação | já praticado (referência escalar, fluxo × lote, reprodução exata); formalizar recibos | — | parcial |

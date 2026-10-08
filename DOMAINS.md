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

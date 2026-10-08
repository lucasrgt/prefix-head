# Teste de realidade: LLMs pequenos reais × nosso minGRU — 2026-10-08

Mesmo texto (data/norm/test.txt, 716.784 chars). LLMs no llama.cpp (a11f57b), pesos Q8_0; bpc = log2(PPL) × tokens/char do tokenizer de cada modelo (contexto 1024, metade pontuada). Energia = RAPL do pacote menos repouso medido antes (fundo ruidoso: repouso 41–75 W); valores servem como ordem de grandeza.

| modelo | parâmetros | treinado com | bpc | pontuar (8 threads) | gerar (1 thread) |
|---|---|---|---|---|---|
| SmolLM2-135M Q8_0 | 135M | ~2T tokens, inglês | 2.1696 | 186 µs/char, ~11.500 µJ/char | 177 tok/s = 2.230 µs/char |
| Qwen3-0.6B Q8_0 | 600M | ~36T tokens, multilíngue | **1.8588** | 671 µs/char, ~29.000 µJ/char | 39 tok/s = 8.300 µs/char |
| minGRU chrono+stateful (nosso) | 2,2M | 6,45M chars destes livros | 2.0755 | 0,26 µs/char, ~18 µJ/char (1 thread: 1,49 µs, ~5 µJ) | 1,49 µs/char |

- Nosso modelo **vence o SmolLM2-135M** em qualidade (−4,3% bpc) com 60× menos parâmetros, ~650× menos energia e ~1.500× mais rápido para gerar.
- **Perde para o Qwen3-0.6B** por 11,7% em bpc — o preço de 270× menos parâmetros e ~5 milhões de vezes menos dados de treino — mas gasta ~1.600× menos energia por char e gera ~5.600× mais rápido.
- Ressalvas: nosso modelo foi treinado no domínio (os mesmos 14 livros, texto de teste distinto), os LLMs não; bpc mede modelagem de texto, não conhecimento geral nem raciocínio, onde os LLMs ganhariam com folga.

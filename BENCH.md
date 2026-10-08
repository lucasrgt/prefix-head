# Bench log

Cada entrada: data, commit/estado, comando, máquina, carga, resultado e conclusão.

## 2026-10-08 — kernels de matvec, 1 thread (T0, load ~3.5, sem RAPL)

`./target/release/bench` — Gw/s = bilhões de pesos por segundo (mediana).

| kernel | 1024² (L2) | 4096² (L3) | 8192×16384 (32 MB ternário) |
|---|---|---|---|
| tern_avx2 (bitplanes, int8 act) | 29 | 35 | 34 |
| tern_vnni | 84 | 87 | 78 |
| tern_vnni_r4 | 68 | 68 | 55 |
| tern_sad | 63 | 71 | 67 |
| sign_popcnt (ternário×ternário, u64) | 73 | 76 | 75 |
| sign_vpopcnt (ternário×ternário, AVX-512) | 353 | 447 | 238 (≈60 GB/s = banda DRAM) |
| i8_vnni (int8 denso) | 103 | 108 | 32 (DRAM) |
| f32_avx512 | 24 | 10 (DRAM) | 6 (DRAM) |

Conclusões:
- Bitplanes + int8 act está limitado por instrução (expansão kmov/movm/sub), não por memória: ~21 GB/s. No cache, int8 denso é mais rápido. Ternário só vence quando o int8 cai na DRAM.
- Ternário × ternário via VPOPCNTDQ: ~4–5x qualquer outro; 32 MB já sai do L3 efetivo e bate na banda da DRAM.
- Unroll de 4 linhas piorou.
- Próximo: formato ternário que não precise de expansão (2 bits empacotados + LUT/vpshufb, ou tiles decodificados), multithread com pool persistente, e medir o custo de qualidade de ativações ternárias/4-bit.

## 2026-10-08 — inferência por caractere, 1 núcleo (`bench infer`, pesos aleatórios)

"Novo" = SIMD com dispatch (AVX-512/AVX2/escalar), sem alocação, tanh/exp polinomiais, **inclui softmax**; "antigo" = laços escalares com alocação e sem softmax. Energia = mediana de 9 rounds de ~1 s, RAPL package menos repouso medido antes de cada round (carga de fundo: Steam, navegadores).

| config | kMAC/ch | antigo µs/ch | novo µs/ch | ganho | µJ/ch (mín–máx) |
|---|---|---|---|---|---|
| bytes 1D | 208 | 13.96 | 7.24 | 1.9× | 112 (82–155) |
| bbpe512 1D | 90 | 5.94 | 3.15 | 1.9× | 32 (24–62) |
| bbpe2048 1D | 199 | 12.68 | 8.47 | 1.5× | 179 (79–350) |
| bpe8192 1D | 586 | 41.23 | 31.06 | 1.3× | 763 (232–962) |
| bytes 2D | 150 | 11.93 | 5.30 | 2.3× | 75 (41–92) |
| bpe8192 2D | 24 | 2.20 | 0.97 | 2.3× | 15 (11–18) |
| bpe8192 4D | 14 | 2.39 | 0.97 | 2.5× | 12 (8–20) |

Conclusões:
- Cabeças 1D grandes são limitadas por memória (8 MB f32 por token no bpe8192 ≈ 70 GB/s): kernel mais rápido não ajuda; precisa de pesos int8/ternários.
- Modelos pequenos rodam a ~25–30 GMAC/s, ~1/5 do pico do núcleo: margem no gemv do corpo.
- bpe8192 2D gasta ~50× menos energia por char que o 1D (> razão de MACs 24×): mover pesos custa mais que calcular.
- Energia ainda ruidosa (dispersão ~2×); domínio só-núcleos (intel-rapl:0:0) precisa de chmod.

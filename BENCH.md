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

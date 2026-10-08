use kern::Rng;
use nn::fmlp::FMlp;
use ruler::energy::Rapl;
use std::hint::black_box;
use std::time::Duration;

struct Config {
    name: &'static str,
    radix: &'static [usize],
    k: usize,
    factored: bool,
    tok_per_char: f64,
}

const CONFIGS: [Config; 7] = [
    Config { name: "bytes 1D", radix: &[256], k: 17, factored: false, tok_per_char: 1.018 },
    Config { name: "bbpe512 1D", radix: &[512], k: 8, factored: false, tok_per_char: 0.459 },
    Config { name: "bbpe2048 1D", radix: &[2048], k: 6, factored: false, tok_per_char: 0.347 },
    Config { name: "bpe8192 1D", radix: &[8192], k: 5, factored: false, tok_per_char: 0.274 },
    Config { name: "bytes 2D", radix: &[16, 16], k: 17, factored: true, tok_per_char: 1.018 },
    Config { name: "bpe8192 2D", radix: &[91, 91], k: 5, factored: true, tok_per_char: 0.274 },
    Config { name: "bpe8192 4D", radix: &[8, 8, 8, 16], k: 5, factored: true, tok_per_char: 0.274 },
];

const TOKENS: usize = 20000;

/// Single-core generation cost per char: the old allocating scalar step (no softmax) versus the
/// SIMD step that also pays the softmax normaliser. Weights are random; speed does not depend on them.
const ROUNDS: usize = 9;
const ROUND_SECS: f64 = 1.0;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Background load moves package power around, so each round re-reads idle power right before the
/// timed loop and the report keeps the median marginal energy and its spread.
pub fn run(rapl: &Option<Rapl>) {
    if let Some(r) = rapl {
        println!("RAPL domain: {}", r.domain());
    }
    println!("{:<14}{:>10}{:>12}{:>12}{:>9}{:>13}{:>16}", "config", "kMAC/ch", "old us/ch", "new us/ch", "speedup", "new uJ/ch", "uJ/ch min-max");
    for c in &CONFIGS {
        let m = FMlp::new(c.radix.to_vec(), 32, c.k, 256, 1, c.factored, c.factored);
        let f = c.radix.len();
        let mut rng = Rng::new(9);
        let ctxs: Vec<Vec<u32>> = (0..TOKENS)
            .map(|_| (0..c.k * f).map(|i| (rng.next() % c.radix[i % f] as u64) as u32).collect())
            .collect();
        let picks: Vec<u32> = (0..TOKENS * f).map(|i| (rng.next() % c.radix[i % f] as u64) as u32).collect();
        let mut out = vec![0u32; f];

        let t0 = std::time::Instant::now();
        for (t, ctx) in ctxs.iter().enumerate() {
            m.step_single(ctx, |q, _| picks[t * f + q], &mut out);
            black_box(&out);
        }
        let old_us = t0.elapsed().as_secs_f64() * 1e6 / TOKENS as f64;

        let mut s = m.scratch();
        let mut run_new = || {
            for (t, ctx) in ctxs.iter().enumerate() {
                black_box(m.infer_step(ctx, &mut s, |q, _, _| picks[t * f + q], &mut out));
            }
        };
        let t = std::time::Instant::now();
        run_new();
        let reps = (ROUND_SECS / t.elapsed().as_secs_f64()).ceil().max(1.0) as usize;
        let mut run_round = || (0..reps).for_each(|_| run_new());
        let per_round = (TOKENS * reps) as f64;
        let mut times = Vec::new();
        let mut energies = Vec::new();
        for _ in 0..ROUNDS {
            match rapl {
                Some(r) => {
                    let idle = r.idle_watts(Duration::from_millis(500));
                    let (secs, joules) = r.measure(&mut run_round);
                    times.push(secs);
                    energies.push((joules - idle * secs) * 1e6 / per_round * c.tok_per_char);
                }
                None => {
                    let t = std::time::Instant::now();
                    run_round();
                    times.push(t.elapsed().as_secs_f64());
                }
            }
        }
        let new_us = median(times) * 1e6 / per_round;
        let (uj_char, lo, hi) = if energies.is_empty() {
            (f64::NAN, f64::NAN, f64::NAN)
        } else {
            let lo = energies.iter().copied().fold(f64::MAX, f64::min);
            let hi = energies.iter().copied().fold(f64::MIN, f64::max);
            (median(energies), lo, hi)
        };
        let kmac = m.macs_per_token() as f64 * c.tok_per_char / 1e3;
        println!(
            "{:<14}{kmac:>10.0}{:>12.2}{:>12.2}{:>8.1}x{uj_char:>13.2}{:>16}",
            c.name,
            old_us * c.tok_per_char,
            new_us * c.tok_per_char,
            old_us / new_us,
            format!("{lo:.1}-{hi:.1}")
        );
    }
}

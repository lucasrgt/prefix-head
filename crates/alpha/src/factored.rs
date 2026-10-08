use crate::Args;
use kern::Rng;
use nn::fmlp::FMlp;
use tok::codec::Alphabet;
use tok::syll::{Syllables, from_digits, to_digits};
use std::time::Instant;

enum Fact {
    Syl(Syllables),
    Digits(Alphabet, Vec<usize>),
}

impl Fact {
    fn radix(&self) -> Vec<usize> {
        match self {
            Fact::Syl(s) => s.radix(),
            Fact::Digits(_, r) => r.clone(),
        }
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        match self {
            Fact::Syl(s) => s.encode(text),
            Fact::Digits(a, r) => to_digits(&a.encode(text), r),
        }
    }

    fn decode(&self, syms: &[u32]) -> String {
        match self {
            Fact::Syl(s) => s.decode(syms),
            Fact::Digits(a, r) => a.decode(&from_digits(syms, r)),
        }
    }
}

/// `syl` = Hangul-style syllable blocks; `nd<radix>:<alphabet>` = flat alphabet as mixed-radix
/// digits, e.g. `nd8x8x8x16:bpe8192`, or `nd1:<alphabet>` for a single full-softmax factor;
/// flags after `nd`: `c` = output matrix per prefix, `f` = full-vocabulary input embedding (`ndcf16x16:bytes`).
fn parse(name: &str, train: &str) -> Fact {
    if name == "syl" {
        return Fact::Syl(Syllables::train(train));
    }
    let rest = name.strip_prefix("nd").map(|r| r.trim_start_matches(['c', 'f']));
    let (spec, inner) = rest.and_then(|s| s.split_once(':')).expect("nd<radix>:<alphabet> or ndc<radix>:<alphabet>");
    let alpha = tok::by_name(inner, train);
    let radix: Vec<usize> = if spec == "1" { vec![alpha.vocab()] } else { spec.split('x').map(|r| r.parse().unwrap()).collect() };
    assert!(radix.iter().product::<usize>() >= alpha.vocab(), "radix {radix:?} too small for {inner}");
    Fact::Digits(alpha, radix)
}

fn sample_logits(logits: &[f32], temp: f32, rng: &mut Rng) -> u32 {
    let mx = logits.iter().copied().fold(f32::MIN, f32::max);
    let p: Vec<f32> = logits.iter().map(|&l| ((l - mx) / temp).exp()).collect();
    let mut r = (rng.next() >> 40) as f32 / (1u64 << 24) as f32 * p.iter().sum::<f32>();
    for (i, &pi) in p.iter().enumerate() {
        if r < pi {
            return i as u32;
        }
        r -= pi;
    }
    p.len() as u32 - 1
}

pub fn run(a: &Args, train: &str, test: &str, train_chars: usize, test_chars: usize) {
    let t0 = Instant::now();
    let fact = parse(&a.alpha, train);
    let radix = fact.radix();
    let f = radix.len();
    let tr = fact.encode(train);
    let te = fact.encode(test);
    assert_eq!(fact.decode(&te), test, "lossless");
    let codec_secs = t0.elapsed().as_secs_f64();

    let tr_tok = tr.len() / f;
    let te_tok = te.len() / f;
    let tpc = tr_tok as f64 / train_chars as f64;
    let k = (a.ctx_chars * tpc).ceil() as usize;
    let d = 32;
    let flags = a.alpha.strip_prefix("nd").map_or("", |r| &r[..r.len() - r.trim_start_matches(['c', 'f']).len()]);
    let mut m = FMlp::new(radix.clone(), d, k, a.h, a.seed, flags.contains('c'), flags.contains('f'));
    let radix_s = radix.iter().map(|r| r.to_string()).collect::<Vec<_>>().join("x");
    eprintln!("{}: radix {radix_s}, {tpc:.3} tok/char, k {k}, params {}, train {tr_tok} tok, test {te_tok} tok", a.alpha, m.params());

    let steps = ((tr_tok as f64 * a.epochs) / a.batch as f64).ceil() as usize;
    let warm = 300.min(steps / 10);
    let mut rng = Rng::new(a.seed ^ 0x9e37);
    let mut bt = m.batch(a.batch);
    let mut ts = vec![0usize; a.batch];
    let t1 = Instant::now();
    let mut ema = 0.0;
    for s in 0..steps {
        let lr = if s < warm {
            a.lr * (s + 1) as f32 / warm as f32
        } else {
            let p = (s - warm) as f32 / (steps - warm).max(1) as f32;
            a.lr * (0.05 + 0.95 * 0.5 * (1.0 + (std::f32::consts::PI * p).cos()))
        };
        for t in ts.iter_mut() {
            *t = (rng.next() % tr_tok as u64) as usize;
        }
        let loss = m.train_step(&tr, &ts, &mut bt, lr, s as i32 + 1) / a.batch as f64;
        ema = if s == 0 { loss } else { 0.99 * ema + 0.01 * loss };
        if s % (steps / 20).max(1) == 0 {
            eprintln!("  step {s}/{steps} train ~{:.3} bpc  {:.0}s", ema / std::f64::consts::LN_2 * tpc, t1.elapsed().as_secs_f64());
        }
    }
    let train_secs = t1.elapsed().as_secs_f64();
    let test_bpc = m.eval_nats(&te) / std::f64::consts::LN_2 / test_chars as f64;

    let probe = 20000.min(te_tok);
    let ctxs: Vec<Vec<u32>> = (0..probe).map(|t| m.context_of(&te, t)).collect();
    let mut out = vec![0u32; f];
    let t2 = Instant::now();
    for (t, c) in ctxs.iter().enumerate() {
        m.step_single(c, |q, _| te[t * f + q], &mut out);
        std::hint::black_box(&out);
    }
    let te_tpc = te_tok as f64 / test_chars as f64;
    let us_per_char = t2.elapsed().as_secs_f64() * 1e6 / probe as f64 * te_tpc;
    let macs_per_char = m.macs_per_token() as f64 * te_tpc;

    let prompt_text = &test[..test.char_indices().nth(30).unwrap().0];
    let mut seq = fact.encode(prompt_text);
    let mut srng = Rng::new(7);
    let n_new = (300.0 * te_tpc) as usize;
    for _ in 0..n_new {
        let c = m.context_of(&seq, seq.len() / f);
        let mut tok = vec![0u32; f];
        m.step_single(&c, |_, l| sample_logits(l, 0.8, &mut srng), &mut tok);
        seq.extend(tok);
    }
    let sample = fact.decode(&seq);

    let line = format!(
        "{{\"alpha\":\"{}\",\"radix\":\"{radix_s}\",\"vocab\":{},\"head\":{},\"sym_per_char\":{te_tpc:.4},\"k\":{k},\"d\":{d},\"h\":{},\"params\":{},\"epochs\":{},\"steps\":{steps},\"train_secs\":{train_secs:.1},\"codec_secs\":{codec_secs:.1},\"test_bpc\":{test_bpc:.4},\"us_per_char\":{us_per_char:.3},\"macs_per_char\":{macs_per_char:.0},\"train_chars\":{train_chars},\"test_chars\":{test_chars}}}",
        a.alpha,
        radix.iter().product::<usize>(),
        radix.iter().sum::<usize>(),
        m.h,
        m.params(),
        a.epochs
    );
    println!("{line}");
    eprintln!("sample: {sample}\n");
    ruler::results::append("alpha.jsonl", &line);
}

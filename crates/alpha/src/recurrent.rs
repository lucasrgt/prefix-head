use crate::Args;
use crate::factored::{parse, sample_logits};
use kern::Rng;
use nn::rnn::MinGru;
use tok::syll::from_digits;

/// minGRU over a factored alphabet (`grp-first-91:bpe8192`, `nd…`). Trains on random windows of
/// `seq` tokens from a zero state; evaluates on the test text streamed once with the state carried
/// throughout (`test_bpc`) and with the state reset every `seq` tokens (`bpc_reset`, training's view).
pub fn run(a: &Args, train: &str, test: &str, train_chars: usize, test_chars: usize) {
    let fact = parse(&a.alpha, train);
    let radix = fact.radix();
    let f = radix.len();
    let vocab: usize = radix.iter().product();
    assert!(vocab <= 1 << 20, "input vocabulary {vocab} too large for a full embedding");
    let tr = fact.encode(train);
    let te = fact.encode(test);
    assert_eq!(fact.decode(&te), test, "lossless");
    let tr_ids = from_digits(&tr, &radix);
    let te_ids = from_digits(&te, &radix);
    let (n_tr, n_te) = (tr_ids.len(), te_ids.len());

    let mut m = MinGru::new(vocab, radix.clone(), a.h, a.layers, a.seed);
    let radix_s = radix.iter().map(|r| r.to_string()).collect::<Vec<_>>().join("x");
    eprintln!(
        "mingru {}: radix {radix_s}, h {}, layers {}, seq {}, params {}, state {} B, train {n_tr} tok",
        a.alpha,
        a.h,
        a.layers,
        a.seq,
        m.params(),
        m.state_bytes()
    );

    let (b, t) = (a.batch, a.seq);
    let steps = ((n_tr as f64 * a.epochs) / (b * t) as f64).ceil() as usize;
    let warm = 300.min(steps / 10);
    let mut rng = Rng::new(a.seed ^ 0x9e37);
    let mut bf = m.buf(b * t);
    let (mut x, mut y) = (vec![0u32; b * t], vec![0u32; b * t * f]);
    let t1 = std::time::Instant::now();
    let mut ema = 0.0;
    let tpc = n_tr as f64 / train_chars as f64;
    for s in 0..steps {
        let lr = if s < warm {
            a.lr * (s + 1) as f32 / warm as f32
        } else {
            let p = (s - warm) as f32 / (steps - warm).max(1) as f32;
            a.lr * (0.05 + 0.95 * 0.5 * (1.0 + (std::f32::consts::PI * p).cos()))
        };
        for stream in 0..b {
            let start = (rng.next() % (n_tr - t) as u64) as usize;
            for k in 0..t {
                let pos = start + k;
                x[stream * t + k] = if pos == 0 { vocab as u32 } else { tr_ids[pos - 1] };
                y[(stream * t + k) * f..(stream * t + k + 1) * f].copy_from_slice(&tr[pos * f..(pos + 1) * f]);
            }
        }
        let loss = m.forward(&x, &y, b, t, &mut bf) / (b * t) as f64;
        m.backward(&x, &y, b, t, 1.0 / (b * t) as f32, &mut bf);
        let norm = m.step(lr, s as i32 + 1, 1.0);
        ema = if s == 0 { loss } else { 0.99 * ema + 0.01 * loss };
        if s % (steps / 20).max(1) == 0 {
            eprintln!("  step {s}/{steps} train ~{:.3} bpc  |g| {norm:.2}  {:.0}s", ema / std::f64::consts::LN_2 * tpc, t1.elapsed().as_secs_f64());
        }
    }
    let train_secs = t1.elapsed().as_secs_f64();

    let stream_bpc = |reset: Option<usize>| {
        let (mut st, mut sc) = (m.state(), m.scratch());
        let mut out = vec![0u32; f];
        let mut nats = 0.0f64;
        let t0 = std::time::Instant::now();
        for k in 0..n_te {
            if reset.is_some_and(|r| k % r == 0) {
                st = m.state();
            }
            let input = if k == 0 { vocab as u32 } else { te_ids[k - 1] };
            nats -= m.infer_step(input, &mut st, &mut sc, |q, _, _| te[k * f + q], &mut out) as f64;
        }
        (nats / std::f64::consts::LN_2 / test_chars as f64, t0.elapsed().as_secs_f64())
    };
    let (test_bpc, secs) = stream_bpc(None);
    let (bpc_reset, _) = stream_bpc(Some(t));
    let us_per_char = secs * 1e6 / test_chars as f64;
    let te_tpc = n_te as f64 / test_chars as f64;
    let macs_per_char = m.macs_per_token() as f64 * te_tpc;

    let prompt = fact.encode(&test[..test.char_indices().nth(30).unwrap().0]);
    let (mut st, mut sc) = (m.state(), m.scratch());
    let mut out = vec![0u32; f];
    let mut srng = Rng::new(7);
    let mut seq = prompt.clone();
    let p_ids = from_digits(&prompt, &radix);
    let mut last = vocab as u32;
    for &id in &p_ids {
        m.infer_step(last, &mut st, &mut sc, |_, l, _| sample_logits(l, 1.0, &mut Rng::new(1)), &mut out);
        last = id;
    }
    for _ in 0..(300.0 * te_tpc) as usize {
        m.infer_step(last, &mut st, &mut sc, |_, l, _| sample_logits(l, 0.8, &mut srng), &mut out);
        seq.extend_from_slice(&out);
        last = from_digits(&out, &radix)[0];
    }
    let sample = fact.decode(&seq);

    let line = format!(
        "{{\"alpha\":\"{}\",\"model\":\"mingru\",\"seed\":{},\"radix\":\"{radix_s}\",\"h\":{},\"layers\":{},\"seq\":{t},\"params\":{},\"state_bytes\":{},\"epochs\":{},\"steps\":{steps},\"train_secs\":{train_secs:.1},\"sym_per_char\":{te_tpc:.4},\"test_bpc\":{test_bpc:.4},\"bpc_reset\":{bpc_reset:.4},\"us_per_char\":{us_per_char:.3},\"macs_per_char\":{macs_per_char:.0},\"train_chars\":{train_chars},\"test_chars\":{test_chars}}}",
        a.alpha,
        a.seed,
        a.h,
        a.layers,
        m.params(),
        m.state_bytes(),
        a.epochs
    );
    println!("{line}");
    eprintln!("sample: {sample}\n");
    ruler::results::append("alpha.jsonl", &line);
}

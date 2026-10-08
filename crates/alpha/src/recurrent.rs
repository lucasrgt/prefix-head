use crate::Args;
use crate::factored::{parse, sample_logits};
use kern::Rng;
use nn::osc::Osc;
use nn::rnn::MinGru;
use nn::seq::Seq;
use nn::kosc::KOsc;
use nn::sosc::SelOsc;
use tok::syll::from_digits;

/// minGRU over a factored alphabet (`grp-first-91:bpe8192`, `nd…`). Trains on random windows of
/// `seq` tokens from a zero state; evaluates on the test text streamed once with the state carried
/// throughout (`test_bpc`) and with the state reset every `seq` tokens (`bpc_reset`, training's view).
pub fn run(a: &Args, train: &str, test: &str, train_chars: usize, test_chars: usize) {
    match a.core.as_str() {
        "osc" => train_core(a, train, test, train_chars, test_chars, |v, r| {
            let mut m = Osc::new(v, r, a.h, a.n, a.layers, a.seed);
            if a.chrono > 0.0 {
                m.chrono(a.chrono, a.seed);
            }
            m
        }),
        "sosc" => train_core(a, train, test, train_chars, test_chars, |v, r| {
            let mut m = SelOsc::new(v, r, a.h, a.n, a.layers, a.seed);
            if a.chrono > 0.0 {
                m.chrono(a.chrono, a.seed);
            }
            m
        }),
        "kosc" => train_core(a, train, test, train_chars, test_chars, |v, r| {
            let mut m = KOsc::new(v, r, a.h, a.n, a.layers, a.seed);
            if a.chrono > 0.0 {
                m.chrono(a.chrono, a.seed);
            }
            m
        }),
        "mingru" => train_core(a, train, test, train_chars, test_chars, |v, r| {
            let mut m = MinGru::new(v, r, a.h, a.layers, a.seed);
            if a.chrono > 0.0 {
                m.chrono(a.chrono, a.seed);
            }
            m
        }),
        c => panic!("unknown core {c}"),
    }
}

fn train_core<M: Seq>(a: &Args, train: &str, test: &str, train_chars: usize, test_chars: usize, make: impl Fn(usize, Vec<usize>) -> M) {
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

    let mut m = make(vocab, radix.clone());
    let radix_s = radix.iter().map(|r| r.to_string()).collect::<Vec<_>>().join("x");
    eprintln!(
        "{} {}: radix {radix_s}, h {}, layers {}, seq {}, params {}, state {} B, train {n_tr} tok",
        M::KIND,
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
    let (mut cursor, mut windows) = (vec![0usize; b], vec![0usize; b]);
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
            if a.stateful == 0 || windows[stream] == 0 || cursor[stream] + t > n_tr {
                cursor[stream] = (rng.next() % (n_tr - t) as u64) as usize;
                windows[stream] = a.stateful.max(1);
                if a.stateful > 0 {
                    m.reset_stream(stream, &mut bf);
                }
            }
            let start = cursor[stream];
            for k in 0..t {
                let pos = start + k;
                x[stream * t + k] = if pos == 0 { vocab as u32 } else { tr_ids[pos - 1] };
                y[(stream * t + k) * f..(stream * t + k + 1) * f].copy_from_slice(&tr[pos * f..(pos + 1) * f]);
            }
            cursor[stream] += t;
            windows[stream] -= 1;
        }
        let loss = m.forward(&x, &y, b, t, &mut bf) / (b * t) as f64;
        m.backward(&x, &y, b, t, 1.0 / (b * t) as f32, &mut bf);
        if a.stateful > 0 {
            m.carry(b, t, &mut bf);
        }
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
        "{{\"alpha\":\"{}\",\"model\":\"{}\",\"n\":{},\"seed\":{},\"radix\":\"{radix_s}\",\"h\":{},\"layers\":{},\"seq\":{t},\"chrono\":{},\"stateful\":{},\"params\":{},\"state_bytes\":{},\"epochs\":{},\"steps\":{steps},\"train_secs\":{train_secs:.1},\"sym_per_char\":{te_tpc:.4},\"test_bpc\":{test_bpc:.4},\"bpc_reset\":{bpc_reset:.4},\"us_per_char\":{us_per_char:.3},\"macs_per_char\":{macs_per_char:.0},\"train_chars\":{train_chars},\"test_chars\":{test_chars}}}",
        a.alpha,
        M::KIND,
        a.n,
        a.seed,
        a.h,
        a.layers,
        a.chrono,
        a.stateful,
        m.params(),
        m.state_bytes(),
        a.epochs
    );
    println!("{line}");
    eprintln!("sample: {sample}\n");
    ruler::results::append("alpha.jsonl", &line);
    let dir = ruler::results::root().join("data/models");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(checkpoint_name(a));
    m.save(&path, &format!("{} test_tokens={n_te} test_bpc={test_bpc:.4}", a.alpha)).unwrap();
    eprintln!("saved {}", path.display());
}

pub fn checkpoint_name(a: &Args) -> String {
    let chrono = if a.chrono > 0.0 { format!("-chrono{}", a.chrono) } else { String::new() };
    let stateful = if a.stateful > 0 { format!("-stateful{}", a.stateful) } else { String::new() };
    let core = if a.core == "mingru" { "mingru".into() } else { format!("{}-n{}", a.core, a.n) };
    let extra = if a.extra { "-extra" } else { "" };
    format!("{core}-{}-h{}-l{}-s{}{chrono}{stateful}{extra}-seed{}.bin", a.alpha.replace(':', "_"), a.h, a.layers, a.seq, a.seed)
}

/// Scores the test text with a saved checkpoint `passes` times and reports µs and marginal package
/// µJ per char, single-threaded and with 8 threads (8 contiguous segments, each from a cold state).
pub fn score(a: &Args, train: &str, test: &str, test_chars: usize) {
    match a.core.as_str() {
        "osc" => score_core::<Osc>(a, train, test, test_chars),
        "sosc" => score_core::<SelOsc>(a, train, test, test_chars),
        "kosc" => score_core::<KOsc>(a, train, test, test_chars),
        _ => score_core::<MinGru>(a, train, test, test_chars),
    }
}

fn score_core<M: Seq + Sync>(a: &Args, train: &str, test: &str, test_chars: usize) {
    let root = ruler::results::root();
    let fact = parse(&a.alpha, train);
    let radix = fact.radix();
    let f = radix.len();
    let path = root.join("data/models").join(checkpoint_name(a));
    let (m, _) = M::load(&path).unwrap();
    let te = fact.encode(test);
    let te_ids = from_digits(&te, &radix);
    let bos = m.vocab() as u32;
    let segment = |lo: usize, hi: usize| {
        let (mut st, mut sc) = (m.state(), m.scratch());
        let mut out = vec![0u32; f];
        let mut nats = 0.0f64;
        for k in lo..hi {
            let input = if k == lo { bos } else { te_ids[k - 1] };
            nats -= m.infer_step(input, &mut st, &mut sc, |q, _, _| te[k * f + q], &mut out) as f64;
        }
        nats
    };
    let rapl = ruler::energy::Rapl::open().expect("RAPL unreadable");
    let n = te_ids.len();
    for threads in [1usize, 8] {
        let idle = rapl.idle_watts(std::time::Duration::from_secs(3));
        let passes = a.epochs.max(1.0) as usize;
        let mut nats = 0.0;
        let (secs, joules) = rapl.measure(|| {
            for _ in 0..passes {
                nats = std::thread::scope(|s| {
                    let hs: Vec<_> = (0..threads).map(|i| s.spawn(move || segment(n * i / threads, n * (i + 1) / threads))).collect();
                    hs.into_iter().map(|h| h.join().unwrap()).sum::<f64>()
                });
            }
        });
        let chars = (test_chars * passes) as f64;
        println!(
            "{} threads={threads}: {:.4} bpc, {:.3} us/char, idle {idle:.1} W, marginal {:.2} uJ/char ({passes} passes, {secs:.1} s)",
            checkpoint_name(a),
            nats / std::f64::consts::LN_2 / test_chars as f64,
            secs * 1e6 / chars,
            (joules - idle * secs) * 1e6 / chars
        );
    }
}

use crate::Args;
use crate::factored::{parse, sample_logits};
use kern::Rng;
use nn::osc::Osc;
use nn::rnn::MinGru;
use nn::seq::Seq;
use nn::kosc::KOsc;
use nn::sosc::SelOsc;
use nn::tfm::Tfm;
use tok::syll::from_digits;

/// minGRU over a factored alphabet (`grp-first-91:bpe8192`, `nd…`). Trains on random windows of
/// `seq` tokens from a zero state; evaluates on the test text streamed once with the state carried
/// throughout (`test_bpc`) and with the state reset every `seq` tokens (`bpc_reset`, training's view).
pub fn run(a: &Args, train: &str, books: &[String], test: &str, train_chars: usize, test_chars: usize) {
    match a.core.as_str() {
        "osc" => train_core(a, train, books, test, train_chars, test_chars, |v, r| {
            let mut m = Osc::new(v, r, a.h, a.n, a.layers, a.seed);
            if a.chrono > 0.0 {
                m.chrono(a.chrono, a.seed);
            }
            m
        }),
        "sosc" => train_core(a, train, books, test, train_chars, test_chars, |v, r| {
            let mut m = SelOsc::new(v, r, a.h, a.n, a.layers, a.seed);
            if a.chrono > 0.0 {
                m.chrono(a.chrono, a.seed);
            }
            m
        }),
        "kosc" => train_core(a, train, books, test, train_chars, test_chars, |v, r| {
            let mut m = KOsc::new(v, r, a.h, a.n, a.layers, a.seed);
            if a.chrono > 0.0 {
                m.chrono(a.chrono, a.seed);
            }
            m
        }),
        "tfm" => train_core(a, train, books, test, train_chars, test_chars, |v, r| Tfm::new(v, r, a.h, a.heads, a.layers, a.mlp, a.seq, a.seed)),
        "mingru" => train_core(a, train, books, test, train_chars, test_chars, |v, r| {
            let mut m = MinGru::new(v, r, a.h, a.layers, a.seed);
            if a.chrono > 0.0 {
                m.chrono(a.chrono, a.seed);
            }
            m
        }),
        c => panic!("unknown core {c}"),
    }
}

/// Chooses which book each training stream reads next: uniformly by length, inside a selected
/// subset, or by learning progress (Graves et al. 2017: books whose loss drops fastest when
/// trained on are picked more, with ε of length-weighted exploration).
struct Sampler {
    lens: Vec<usize>,
    allowed: Vec<usize>,
    progress: Vec<f64>,
    lp: bool,
}

impl Sampler {
    fn pick(&self, rng: &mut Rng, min_len: usize) -> usize {
        let u = |rng: &mut Rng| (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
        let ok: Vec<usize> = self.allowed.iter().copied().filter(|&i| self.lens[i] > min_len).collect();
        if ok.len() == 1 {
            return ok[0];
        }
        let by_len = |rng: &mut Rng| {
            let total: f64 = ok.iter().map(|&i| self.lens[i] as f64).sum();
            let mut r = u(rng) * total;
            for &i in &ok {
                r -= self.lens[i] as f64;
                if r <= 0.0 {
                    return i;
                }
            }
            *ok.last().unwrap()
        };
        if !self.lp || u(rng) < 0.3 {
            return by_len(rng);
        }
        let scale = ok.iter().map(|&i| self.progress[i].abs()).sum::<f64>() / ok.len() as f64 + 1e-9;
        let w: Vec<f64> = ok.iter().map(|&i| (self.progress[i] / scale).min(30.0).exp()).collect();
        let mut r = u(rng) * w.iter().sum::<f64>();
        for (k, &i) in ok.iter().enumerate() {
            r -= w[k];
            if r <= 0.0 {
                return i;
            }
        }
        *ok.last().unwrap()
    }
}

/// Bits per char of each book's first 20k chars under the 14-book reference model (its own tokenizer).
fn reference_scores(extra: &[String]) -> Vec<f64> {
    let root = ruler::results::root();
    let (train14, _) = ruler::corpus::load(&root.join("data/raw"));
    let fact = parse("grp-first-91:bpe8192", &train14);
    let radix = fact.radix();
    let f = radix.len();
    let path = root.join("data/models/sosc-n51-grp-first-91_bpe8192-h128-l2-s64-stateful64-seed1.bin");
    let (m, _) = SelOsc::load(&path).expect("reference checkpoint");
    let score = |text: &str| {
        let cut = text.char_indices().nth(20_000).map_or(text.len(), |(i, _)| i);
        let chars = text[..cut].chars().count().max(1);
        let d = fact.encode(&text[..cut]);
        let ids = from_digits(&d, &radix);
        let (mut st, mut sc) = (m.state(), m.scratch());
        let mut out = vec![0u32; f];
        let mut nats = 0.0f64;
        for k in 0..ids.len() {
            let input = if k == 0 { m.vocab() as u32 } else { ids[k - 1] };
            nats -= m.infer_step(input, &mut st, &mut sc, |q, _, _| d[k * f + q], &mut out) as f64;
        }
        nats / std::f64::consts::LN_2 / chars as f64
    };
    let chunk = extra.len().div_ceil(8).max(1);
    std::thread::scope(|s| {
        let hs: Vec<_> = extra.chunks(chunk).map(|c| s.spawn(|| c.iter().map(|t| score(t)).collect::<Vec<f64>>())).collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    })
}

fn train_core<M: Seq>(a: &Args, train: &str, books: &[String], test: &str, train_chars: usize, test_chars: usize, make: impl Fn(usize, Vec<usize>) -> M) {
    let fact = parse(&a.alpha, train);
    let radix = fact.radix();
    let f = radix.len();
    let vocab: usize = radix.iter().product();
    assert!(vocab <= 1 << 20, "input vocabulary {vocab} too large for a full embedding");
    let texts: Vec<&str> = if books.is_empty() { vec![train] } else { books.iter().map(|s| s.as_str()).collect() };
    let book_tr: Vec<Vec<u32>> = texts.iter().map(|t| fact.encode(t)).collect();
    let book_ids: Vec<Vec<u32>> = book_tr.iter().map(|d| from_digits(d, &radix)).collect();
    let te = fact.encode(test);
    assert_eq!(fact.decode(&te), test, "lossless");
    let te_ids = from_digits(&te, &radix);
    let n_tr: usize = book_ids.iter().map(|b| b.len()).sum();
    let n_te = te_ids.len();
    let core_books = if books.is_empty() { 1 } else { 14 };
    let mut allowed: Vec<usize> = (0..texts.len()).collect();
    if a.select != "all" {
        let scores = reference_scores(&books[core_books..]);
        let mut extra: Vec<usize> = (core_books..texts.len()).collect();
        match a.select.as_str() {
            "easy" => extra.sort_by(|&i, &j| scores[i - core_books].partial_cmp(&scores[j - core_books]).unwrap()),
            "hard" => extra.sort_by(|&i, &j| scores[j - core_books].partial_cmp(&scores[i - core_books]).unwrap()),
            "random" => {
                let mut r = Rng::new(a.seed ^ 0x5e1);
                let keys: Vec<u64> = (0..texts.len()).map(|_| r.next()).collect();
                extra.sort_by_key(|&i| keys[i]);
            }
            s => panic!("unknown select {s}"),
        }
        let half: usize = extra.iter().map(|&i| book_ids[i].len()).sum::<usize>() / 2;
        let mut kept = Vec::new();
        let mut acc = 0;
        for i in extra {
            if acc >= half {
                break;
            }
            acc += book_ids[i].len();
            kept.push(i);
        }
        let mean = |v: &[usize]| v.iter().map(|&i| scores[i - core_books]).sum::<f64>() / v.len().max(1) as f64;
        eprintln!("select {}: {} of {} extra books, {acc} tokens, mean reference bpc {:.3} (all {:.3})", a.select, kept.len(), texts.len() - core_books, mean(&kept), scores.iter().sum::<f64>() / scores.len() as f64);
        allowed = (0..core_books).chain(kept).collect();
    }
    let mut sampler = Sampler { lens: book_ids.iter().map(|b| b.len()).collect(), allowed, progress: vec![0.0; texts.len()], lp: a.policy == "lp" };

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
    let (mut cursor, mut windows, mut book_of) = (vec![0usize; b], vec![0usize; b], vec![0usize; b]);
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
            let bk = book_of[stream];
            if a.stateful == 0 || windows[stream] == 0 || cursor[stream] + t > book_ids[bk].len() {
                let bk = sampler.pick(&mut rng, t + 1);
                book_of[stream] = bk;
                cursor[stream] = (rng.next() % (book_ids[bk].len() - t) as u64) as usize;
                windows[stream] = a.stateful.max(1);
                if a.stateful > 0 {
                    m.reset_stream(stream, &mut bf);
                }
            }
            let (ids, tr) = (&book_ids[book_of[stream]], &book_tr[book_of[stream]]);
            let start = cursor[stream];
            for k in 0..t {
                let pos = start + k;
                x[stream * t + k] = if pos == 0 { vocab as u32 } else { ids[pos - 1] };
                y[(stream * t + k) * f..(stream * t + k + 1) * f].copy_from_slice(&tr[pos * f..(pos + 1) * f]);
            }
            cursor[stream] += t;
            windows[stream] -= 1;
        }
        let loss = m.forward(&x, &y, b, t, &mut bf) / (b * t) as f64;
        let before: Vec<f32> = if sampler.lp { (0..b).map(|s| m.row_losses(&bf)[s * t..(s + 1) * t].iter().sum()).collect() } else { Vec::new() };
        m.backward(&x, &y, b, t, 1.0 / (b * t) as f32, &mut bf);
        let norm = m.step(lr, s as i32 + 1, 1.0);
        if sampler.lp {
            m.forward(&x, &y, b, t, &mut bf);
            for stream in 0..b {
                let after: f32 = m.row_losses(&bf)[stream * t..(stream + 1) * t].iter().sum();
                let gain = (before[stream] - after) as f64 / t as f64;
                let p = &mut sampler.progress[book_of[stream]];
                *p = 0.7 * *p + 0.3 * gain;
            }
        }
        if a.stateful > 0 {
            m.carry(b, t, &mut bf);
        }
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
        let mut per_factor = vec![0.0f64; f];
        let t0 = std::time::Instant::now();
        for k in 0..n_te {
            if reset.is_some_and(|r| k % r == 0) {
                st = m.state();
            }
            let input = if k == 0 { vocab as u32 } else { te_ids[k - 1] };
            nats -= m.infer_step(
                input,
                &mut st,
                &mut sc,
                |q, logits, lse| {
                    let c = te[k * f + q];
                    per_factor[q] -= (logits[c as usize] - lse) as f64;
                    c
                },
                &mut out,
            ) as f64;
        }
        let to_bpc = |n: f64| n / std::f64::consts::LN_2 / test_chars as f64;
        (to_bpc(nats), t0.elapsed().as_secs_f64(), per_factor.into_iter().map(to_bpc).collect::<Vec<f64>>())
    };
    let (test_bpc, secs, factors) = stream_bpc(None);
    let (bpc_reset, _, _) = stream_bpc(Some(t));
    let factors_s = factors.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>().join(",");
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
        "{{\"alpha\":\"{}\",\"model\":\"{}\",\"n\":{},\"seed\":{},\"radix\":\"{radix_s}\",\"h\":{},\"layers\":{},\"seq\":{t},\"chrono\":{},\"stateful\":{},\"select\":\"{}\",\"policy\":\"{}\",\"tokens_read\":{},\"params\":{},\"state_bytes\":{},\"epochs\":{},\"steps\":{steps},\"train_secs\":{train_secs:.1},\"sym_per_char\":{te_tpc:.4},\"test_bpc\":{test_bpc:.4},\"bpc_factors\":[{factors_s}],\"bpc_reset\":{bpc_reset:.4},\"us_per_char\":{us_per_char:.3},\"macs_per_char\":{macs_per_char:.0},\"train_chars\":{train_chars},\"test_chars\":{test_chars}}}",
        a.alpha,
        M::KIND,
        a.n,
        a.seed,
        a.h,
        a.layers,
        a.chrono,
        a.stateful,
        a.select,
        a.policy,
        steps * b * t,
        m.params(),
        m.state_bytes(),
        a.epochs
    );
    println!("{line}");
    eprintln!("sample: {sample}\n");
    ruler::results::append("alpha.jsonl", &line);
    if a.epochs < 0.5 {
        return;
    }
    let dir = ruler::results::root().join("data/models");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(checkpoint_name(a));
    m.save(&path, &format!("{} test_tokens={n_te} test_bpc={test_bpc:.4}", a.alpha)).unwrap();
    eprintln!("saved {}", path.display());
}

pub fn checkpoint_name(a: &Args) -> String {
    let chrono = if a.chrono > 0.0 { format!("-chrono{}", a.chrono) } else { String::new() };
    let stateful = if a.stateful > 0 { format!("-stateful{}", a.stateful) } else { String::new() };
    let core = match a.core.as_str() {
        "mingru" => "mingru".into(),
        "tfm" => format!("tfm-a{}-m{}", a.heads, a.mlp),
        c => format!("{c}-n{}", a.n),
    };
    let extra = if a.extra { format!("-extra-{}-{}-e{}", a.select, a.policy, a.epochs) } else { String::new() };
    format!("{core}-{}-h{}-l{}-s{}{chrono}{stateful}{extra}-seed{}.bin", a.alpha.replace(':', "_"), a.h, a.layers, a.seq, a.seed)
}

/// Scores the test text with a saved checkpoint `passes` times and reports µs and marginal package
/// µJ per char, single-threaded and with 8 threads (8 contiguous segments, each from a cold state).
pub fn score(a: &Args, train: &str, test: &str, test_chars: usize) {
    match a.core.as_str() {
        "osc" => score_core::<Osc>(a, train, test, test_chars),
        "sosc" => score_core::<SelOsc>(a, train, test, test_chars),
        "kosc" => score_core::<KOsc>(a, train, test, test_chars),
        "tfm" => score_core::<Tfm>(a, train, test, test_chars),
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

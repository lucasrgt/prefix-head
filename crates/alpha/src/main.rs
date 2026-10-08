mod factored;
mod m2;
mod recurrent;

use kern::Rng;
use std::time::Instant;

pub struct Args {
    pub alpha: String,
    pub ctx_chars: f64,
    pub h: usize,
    pub epochs: f64,
    pub batch: usize,
    pub lr: f32,
    pub seed: u64,
    pub model: String,
    pub layers: usize,
    pub seq: usize,
    pub chrono: f32,
    pub stateful: usize,
    pub core: String,
    pub n: usize,
    pub extra: bool,
    pub select: String,
    pub policy: String,
    pub heads: usize,
    pub mlp: usize,
    pub corpus: String,
}

fn args() -> Args {
    let mut a = Args { alpha: "bytes".into(), ctx_chars: 16.0, h: 256, epochs: 1.0, batch: 512, lr: 2e-3, seed: 1, model: "mlp".into(), layers: 2, seq: 64, chrono: 0.0, stateful: 0, core: "mingru".into(), n: 64, extra: false, select: "all".into(), policy: "uniform".into(), heads: 4, mlp: 256, corpus: "data/raw".into() };
    let v: Vec<String> = std::env::args().skip(1).collect();
    for p in v.chunks(2) {
        let val = &p[1];
        match p[0].as_str() {
            "--alpha" => a.alpha = val.clone(),
            "--ctx-chars" => a.ctx_chars = val.parse().unwrap(),
            "--h" => a.h = val.parse().unwrap(),
            "--epochs" => a.epochs = val.parse().unwrap(),
            "--batch" => a.batch = val.parse().unwrap(),
            "--lr" => a.lr = val.parse().unwrap(),
            "--seed" => a.seed = val.parse().unwrap(),
            "--model" => a.model = val.clone(),
            "--layers" => a.layers = val.parse().unwrap(),
            "--seq" => a.seq = val.parse().unwrap(),
            "--chrono" => a.chrono = val.parse().unwrap(),
            "--stateful" => a.stateful = val.parse().unwrap(),
            "--core" => a.core = val.clone(),
            "--n" => a.n = val.parse().unwrap(),
            "--extra" => a.extra = val == "1",
            "--select" => a.select = val.clone(),
            "--policy" => a.policy = val.clone(),
            "--heads" => a.heads = val.parse().unwrap(),
            "--mlp" => a.mlp = val.parse().unwrap(),
            "--corpus" => a.corpus = val.clone(),
            k => panic!("unknown flag {k}"),
        }
    }
    a
}

fn main() {
    let a = args();
    let root = ruler::results::root();
    let (mut train, mut test) = ruler::corpus::load(&root.join(&a.corpus));
    let mut books: Vec<String> = Vec::new();
    if a.extra {
        books = ruler::corpus::books(&root.join(&a.corpus)).into_iter().map(|b| b.train).collect();
        let extra = ruler::corpus::extra_books(&root.join("data/extra/raw"));
        for e in &extra {
            train.push_str(e);
        }
        books.extend(extra);
    }
    let train_chars = train.chars().count();
    let test_chars = test.chars().count();
    if a.alpha.starts_with("g2") {
        let g2 = |s: &str| std::fs::read_to_string(root.join(format!("data/norm/{s}.g2"))).expect("run tools/louis_encode.py");
        (train, test) = (g2("train"), g2("test"));
    }

    if a.model == "score" {
        recurrent::score(&a, &train, &test, test_chars);
        return;
    }
    if a.model == "m2" {
        m2::run(&a, &train);
        return;
    }
    if a.model == "mingru" {
        recurrent::run(&a, &train, &books, &test, train_chars, test_chars);
        return;
    }
    if a.alpha.starts_with("syl") || a.alpha.starts_with("nd") || a.alpha.starts_with("grp-") {
        factored::run(&a, &train, &test, train_chars, test_chars);
        return;
    }

    let t0 = Instant::now();
    let alpha = tok::by_name(&a.alpha, &train);
    let tr = alpha.encode(&train);
    let te = alpha.encode(&test);
    assert_eq!(alpha.decode(&te), test, "lossless");
    let codec_secs = t0.elapsed().as_secs_f64();

    let spc = tr.len() as f64 / train_chars as f64;
    let v = alpha.vocab();
    let d = v.min(32);
    let k = (a.ctx_chars * spc).ceil() as usize;
    let mut m = nn::mlp::Mlp::new(v, d, k, a.h, a.seed);
    eprintln!(
        "{}: vocab {v}, {spc:.3} sym/char, k {k}, d {d}, params {}, train {} sym, test {} sym",
        alpha.name(),
        m.params(),
        tr.len(),
        te.len()
    );

    let steps = ((tr.len() as f64 * a.epochs) / a.batch as f64).ceil() as usize;
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
            *t = (rng.next() % tr.len() as u64) as usize;
        }
        let loss = m.train_step(&tr, &ts, &mut bt, lr, s as i32 + 1) / a.batch as f64;
        ema = if s == 0 { loss } else { 0.99 * ema + 0.01 * loss };
        if s % (steps / 20).max(1) == 0 {
            let bpc = ema / std::f64::consts::LN_2 * spc;
            eprintln!("  step {s}/{steps} train ~{bpc:.3} bpc  {:.0}s", t1.elapsed().as_secs_f64());
        }
    }
    let train_secs = t1.elapsed().as_secs_f64();

    let test_bpc = m.eval_nats(&te) / std::f64::consts::LN_2 / test_chars as f64;

    let probe = 20000.min(te.len());
    let mut hbuf = vec![0.0; m.h];
    let mut logits = vec![0.0; m.v];
    let ctxs: Vec<Vec<u32>> = (0..probe).map(|t| m.context_of(&te, t)).collect();
    let t2 = Instant::now();
    for c in &ctxs {
        m.step_single(c, &mut hbuf, &mut logits);
        std::hint::black_box(&logits);
    }
    let us_per_sym = t2.elapsed().as_secs_f64() * 1e6 / probe as f64;
    let te_spc = te.len() as f64 / test_chars as f64;
    let us_per_char = us_per_sym * te_spc;
    let macs_per_char = m.macs_per_symbol() as f64 * te_spc;

    let prompt = alpha.encode(&test[..test.char_indices().nth(30).unwrap().0]);
    let sample = alpha.decode(&m.sample(&prompt, (300.0 * te_spc) as usize, 0.8, 7));

    let line = format!(
        "{{\"alpha\":\"{}\",\"vocab\":{v},\"sym_per_char\":{te_spc:.4},\"k\":{k},\"d\":{d},\"h\":{},\"params\":{},\"epochs\":{},\"steps\":{steps},\"train_secs\":{train_secs:.1},\"codec_secs\":{codec_secs:.1},\"test_bpc\":{test_bpc:.4},\"us_per_char\":{us_per_char:.3},\"macs_per_char\":{macs_per_char:.0},\"train_chars\":{train_chars},\"test_chars\":{test_chars}}}",
        alpha.name(),
        m.h,
        m.params(),
        a.epochs
    );
    println!("{line}");
    eprintln!("sample: {sample}\n");
    ruler::results::append("alpha.jsonl", &line);
}

use crate::Args;
use crate::factored::parse;
use crate::recurrent::checkpoint_name;
use nn::osc::Osc;
use nn::rnn::{MinGru, State};
use nn::seq::Seq;
use nn::kosc::KOsc;
use nn::sosc::SelOsc;
use ruler::energy::Rapl;
use tok::syll::from_digits;

const EVAL_TOKENS: usize = 1500;
const EARLY_TOKENS: usize = 128;
const WARM: [usize; 6] = [16, 64, 256, 1024, 4096, 16384];

/// Quantizes every layer vector symmetrically to `bits` (1 bit = sign × mean |x|).
fn quantize(st: &State, bits: u32) -> State {
    State(
        st.0.iter()
            .map(|v| {
                let mean = v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32;
                if bits == 1 {
                    return v.iter().map(|x| x.signum() * mean).collect();
                }
                let levels = ((1u32 << (bits - 1)) - 1) as f32;
                let scale = v.iter().fold(0.0f32, |a, x| a.max(x.abs())) / levels;
                v.iter().map(|x| if scale == 0.0 { 0.0 } else { (x / scale).round() * scale }).collect()
            })
            .collect(),
    )
}

struct BookData {
    name: String,
    train_ids: Vec<u32>,
    test: Vec<u32>,
    test_ids: Vec<u32>,
    test_chars: Vec<usize>,
    capital: Vec<bool>,
}

/// M2: how much of a book survives in the recurrent state. For each book, scores the first
/// EVAL_TOKENS of its test part (which continues the train part) from different starting states.
pub fn run(a: &Args, train: &str) {
    match a.core.as_str() {
        "osc" => probe::<Osc>(a, train),
        "sosc" => probe::<SelOsc>(a, train),
        "kosc" => probe::<KOsc>(a, train),
        _ => probe::<MinGru>(a, train),
    }
}

/// `train` must be the text the checkpoint's tokenizer was trained on (the 14 books, plus extras with --extra 1).
fn probe<M: Seq>(a: &Args, train: &str) {
    let root = ruler::results::root();
    let books = ruler::corpus::books(&root.join("data/raw"));
    let fact = parse(&a.alpha, train);
    let radix = fact.radix();
    let f = radix.len();
    let path = root.join("data/models").join(checkpoint_name(a));
    let (m, meta) = M::load(&path).unwrap_or_else(|e| panic!("{}: {e} (train it with --model mingru first)", path.display()));
    eprintln!("loaded {} ({meta}), state {} B", path.display(), m.state_bytes());
    let bos = m.vocab() as u32;

    let data: Vec<BookData> = books
        .iter()
        .map(|b| {
            let test = fact.encode(&b.test);
            let n = (test.len() / f).min(EVAL_TOKENS);
            let test = test[..n * f].to_vec();
            let pieces: Vec<String> = test.chunks(f).map(|t| fact.decode(t)).collect();
            BookData {
                name: b.name.clone(),
                train_ids: from_digits(&fact.encode(&b.train), &radix),
                test_ids: from_digits(&test, &radix),
                test_chars: pieces.iter().map(|p| p.chars().count()).collect(),
                capital: pieces.iter().map(|p| p.chars().find(|c| c.is_alphabetic()).is_some_and(char::is_uppercase)).collect(),
                test,
            }
        })
        .collect();

    let mut sc = m.scratch();
    let mut read = |ids: &[u32], from_start: bool| {
        let mut st = m.state();
        if from_start {
            m.advance(bos, &mut st, &mut sc);
        }
        for &id in ids {
            m.advance(id, &mut st, &mut sc);
        }
        st
    };
    let full: Vec<State> = data.iter().map(|d| read(&d.train_ids[..d.train_ids.len() - 1], true)).collect();

    let score = |d: &BookData, mut st: State| {
        let mut sc = m.scratch();
        let mut out = vec![0u32; f];
        let (mut bits, mut cap_bits, mut early) = (0.0f64, 0.0f64, 0.0f64);
        let mut input = *d.train_ids.last().unwrap();
        for k in 0..d.test_ids.len() {
            let nats = -m.infer_step(input, &mut st, &mut sc, |q, _, _| d.test[k * f + q], &mut out) as f64;
            let b = nats / std::f64::consts::LN_2;
            bits += b;
            if k < EARLY_TOKENS {
                early += b;
            }
            if d.capital[k] {
                cap_bits += b;
            }
            input = d.test_ids[k];
        }
        (bits, cap_bits, early)
    };
    let chars: usize = data.iter().map(|d| d.test_chars.iter().sum::<usize>()).sum();
    let early_chars: usize = data.iter().map(|d| d.test_chars.iter().take(EARLY_TOKENS).sum::<usize>()).sum();
    let cap_chars: usize = data.iter().map(|d| d.test_chars.iter().zip(&d.capital).filter(|(_, c)| **c).map(|(n, _)| n).sum::<usize>()).sum();

    let mut conditions: Vec<(String, Box<dyn Fn(usize) -> State>, usize)> = Vec::new();
    let state_bytes = m.state_bytes();
    conditions.push(("cold (zero state)".into(), Box::new(|_| m.state()), 0));
    for n in WARM {
        let states: Vec<State> = data
            .iter()
            .map(|d| {
                let end = d.train_ids.len() - 1;
                read(&d.train_ids[end.saturating_sub(n)..end], false)
            })
            .collect();
        conditions.push((format!("last {n} tokens"), Box::new(move |i| states[i].clone()), state_bytes));
    }
    let fl = full.clone();
    conditions.push(("whole book (moment)".into(), Box::new(move |i| fl[i].clone()), state_bytes));
    for bits in [8u32, 4, 2, 1] {
        let q: Vec<State> = full.iter().map(|s| quantize(s, bits)).collect();
        conditions.push((format!("whole book, {bits}-bit state"), Box::new(move |i| q[i].clone()), state_bytes * bits as usize / 32));
    }
    let wrong = full.clone();
    let nb = data.len();
    conditions.push(("another book's moment".into(), Box::new(move |i| wrong[(i + 1) % nb].clone()), state_bytes));

    println!("{} books, {chars} chars scored ({cap_chars} in capitalised tokens)", data.len());
    println!("{:<30}{:>10}{:>14}{:>14}{:>12}", "starting state", "bpc", "bpc capitals", "bpc first 128", "state B");
    for (name, make, bytes) in &conditions {
        let (mut bits, mut cap, mut early) = (0.0, 0.0, 0.0);
        for (i, d) in data.iter().enumerate() {
            let (b, c, e) = score(d, make(i));
            bits += b;
            cap += c;
            early += e;
        }
        let (bpc, cbpc, ebpc) = (bits / chars as f64, cap / cap_chars as f64, early / early_chars as f64);
        println!("{name:<30}{bpc:>10.4}{cbpc:>14.4}{ebpc:>14.4}{bytes:>12}");
        ruler::results::append(
            "m2.jsonl",
            &format!("{{\"model\":\"{}\",\"condition\":\"{name}\",\"bpc\":{bpc:.4},\"bpc_capitals\":{cbpc:.4},\"bpc_first128\":{ebpc:.4},\"state_bytes\":{bytes},\"chars\":{chars}}}", checkpoint_name(a)),
        );
    }

    let rapl = Rapl::open();
    let tokens: usize = data.iter().map(|d| d.train_ids.len()).sum();
    let t0 = std::time::Instant::now();
    let reread = match &rapl {
        Some(r) => r.measure(|| data.iter().for_each(|d| drop(std::hint::black_box(read(&d.train_ids[..d.train_ids.len() - 1], true))))),
        None => (0.0, f64::NAN),
    };
    let secs = if rapl.is_some() { reread.0 } else { t0.elapsed().as_secs_f64() };
    let bytes: Vec<u8> = full.iter().flat_map(|s| s.0.iter().flatten().flat_map(|x| x.to_le_bytes())).collect();
    let t1 = std::time::Instant::now();
    let restored: Vec<f32> = std::hint::black_box(&bytes).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let restore_us = t1.elapsed().as_secs_f64() * 1e6;
    assert_eq!(restored.len() * 4, nb * state_bytes);
    println!(
        "\nreach the end-of-train state of all {nb} books: re-read {tokens} tokens = {secs:.2} s, {:.1} J package; restore {nb} moments ({} B) = {restore_us:.1} µs",
        reread.1,
        bytes.len()
    );
    eprintln!("books: {}", data.iter().map(|d| d.name.as_str()).collect::<Vec<_>>().join(" "));
}

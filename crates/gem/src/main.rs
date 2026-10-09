//! Minimal Gemma 3 CPU runtime to measure a full output head against a two-level (class, then token
//! inside the class) head end to end: same body, same kernels, same threads.
//!
//!   gem check  [--tokens N]              full-head nats vs the llama.cpp dump (data/retro/<tag>-test.full)
//!   gem bench  [--tokens N] [--threads T] tokens/s, head share and RAPL energy, both heads
//!   gem gen    --prompt "..." [--n N]    sample text with both heads, same seed

mod gguf;
mod model;
mod pool;
mod q8;

use model::{Gemma, dot};
use pool::Pool;
use q8::{Act, Isa, Mat, matvec};
use std::path::{Path, PathBuf};
use std::time::Instant;

struct Full {
    act: Act,
    logits: Vec<f32>,
}

impl Full {
    fn new(m: &Gemma) -> Self {
        Self { act: Act::new(m.d), logits: vec![0.0; m.vocab] }
    }

    fn run(&mut self, m: &Gemma, h: &[f32], pool: &Pool, isa: Isa) {
        self.act.quantize(h);
        matvec(pool, isa, &m.emb, &self.act, &mut self.logits);
    }

    fn logprob(&self, t: u32) -> f32 {
        self.logits[t as usize] - lse(&self.logits)
    }
}

struct TwoLevel {
    a: Vec<f32>,
    b: Vec<f32>,
    blocks: Vec<Mat>,
    members: Vec<Vec<u32>>,
    cls_of: Vec<u32>,
    pos_of: Vec<u32>,
    act: Act,
    cl: Vec<f32>,
    inner: Vec<f32>,
    d: usize,
}

struct SendPtr(*mut f32);
unsafe impl Sync for SendPtr {}

impl TwoLevel {
    fn load(m: &Gemma, prefix: &Path) -> Self {
        let rd = |ext: &str| std::fs::read(prefix.with_extension(ext)).unwrap_or_else(|e| panic!("{}.{ext}: {e}", prefix.display()));
        let order: Vec<u32> = rd("order").chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        let a: Vec<f32> = rd("A").chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let b: Vec<f32> = rd("b").chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let k = b.len();
        let per = order.len().div_ceil(k);
        let members: Vec<Vec<u32>> = order.chunks(per).map(|c| c.to_vec()).collect();
        assert_eq!(members.len(), k);
        let mut cls_of = vec![0u32; m.vocab];
        let mut pos_of = vec![0u32; m.vocab];
        for (c, mem) in members.iter().enumerate() {
            for (p, &t) in mem.iter().enumerate() {
                cls_of[t as usize] = c as u32;
                pos_of[t as usize] = p as u32;
            }
        }
        let blocks = members.iter().map(|mem| m.emb.gather(mem)).collect();
        Self { a, b, blocks, members, cls_of, pos_of, act: Act::new(m.d), cl: vec![0.0; k], inner: vec![0.0; per], d: m.d }
    }

    fn bytes(&self) -> usize {
        self.a.len() * 4 + self.b.len() * 4 + self.blocks[0].bytes()
    }

    fn classes(&mut self, h: &[f32], pool: &Pool, isa: Isa) {
        let (k, d) = (self.cl.len(), self.d);
        let per = k.div_ceil(pool.threads());
        let out = SendPtr(self.cl.as_mut_ptr());
        let (a, b) = (&self.a, &self.b);
        pool.run(&|t| {
            let o = &out;
            for c in (t * per).min(k)..((t + 1) * per).min(k) {
                unsafe { *o.0.add(c) = dot(&a[c * d..(c + 1) * d], h, isa) + b[c] };
            }
        });
    }

    fn within(&mut self, c: usize, h: &[f32], pool: &Pool, isa: Isa) {
        self.act.quantize(h);
        let n = self.blocks[c].rows;
        matvec(pool, isa, &self.blocks[c], &self.act, &mut self.inner[..n]);
    }

    fn logprob(&mut self, t: u32, h: &[f32], pool: &Pool, isa: Isa) -> f32 {
        self.classes(h, pool, isa);
        let c = self.cls_of[t as usize] as usize;
        self.within(c, h, pool, isa);
        let n = self.blocks[c].rows;
        self.cl[c] - lse(&self.cl) + self.inner[self.pos_of[t as usize] as usize] - lse(&self.inner[..n])
    }

    fn sample(&mut self, h: &[f32], temp: f32, rng: &mut u64, pool: &Pool, isa: Isa) -> u32 {
        self.classes(h, pool, isa);
        let c = draw(&self.cl, temp, rng);
        self.within(c, h, pool, isa);
        let n = self.blocks[c].rows;
        self.members[c][draw(&self.inner[..n], temp, rng)]
    }
}

fn lse(v: &[f32]) -> f32 {
    let mx = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    mx + v.iter().map(|x| (x - mx).exp()).sum::<f32>().ln()
}

fn uniform(rng: &mut u64) -> f32 {
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;
    (*rng >> 40) as f32 / (1u64 << 24) as f32
}

fn draw(logits: &[f32], temp: f32, rng: &mut u64) -> usize {
    let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let w: Vec<f32> = logits.iter().map(|l| ((l - mx) / temp).exp()).collect();
    let mut r = uniform(rng) * w.iter().sum::<f32>();
    for (i, x) in w.iter().enumerate() {
        r -= x;
        if r <= 0.0 {
            return i;
        }
    }
    w.len() - 1
}

fn tokenize(model: &Path, text: &str) -> Vec<u32> {
    use std::io::Write;
    let mut child = std::process::Command::new(ruler::results::root().join(".tools/tok"))
        .arg(model)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect(".tools/tok (build from tools/hidden/tok.cpp)");
    child.stdin.take().unwrap().write_all(text.as_bytes()).unwrap();
    let out = child.wait_with_output().unwrap();
    String::from_utf8(out.stdout).unwrap().lines().map(|l| l.parse().unwrap()).collect()
}

fn pieces(path: &Path) -> Vec<Vec<u8>> {
    let b = std::fs::read(path).unwrap();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let n = b[i] as usize;
        out.push(b[i + 1..i + 1 + n].to_vec());
        i += 1 + n;
    }
    out
}

fn arg<T: std::str::FromStr>(args: &[String], key: &str, default: T) -> T {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn utf8_chars(p: &[u8]) -> usize {
    p.iter().filter(|&&b| b & 0xC0 != 0x80).count()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).cloned().unwrap_or_else(|| "check".into());
    let root = ruler::results::root();
    let model_path: PathBuf = arg(&args, "--model", root.join("data/models/llm/gemma-3-270m-it-Q8_0.gguf"));
    let tag: String = arg(&args, "--tag", "gemma270m".to_string());
    let head2: String = arg(&args, "--head2", format!("{tag}-prefix-self"));
    let threads: usize = arg(&args, "--threads", 8);
    let n_tok: usize = arg(&args, "--tokens", 512);
    let isa = q8::detect();
    let pool = Pool::new(threads);

    let t0 = Instant::now();
    let g = gguf::Gguf::open(&model_path);
    let ctx = 2048;
    let mut m = Gemma::load(&g, ctx);
    drop(g);
    let body_bits: u8 = arg(&args, "--body-bits", 8);
    let head_bits: u8 = arg(&args, "--head-bits", 8);
    m.requantize(body_bits == 4, head_bits == 4);
    let bos = 2u32;
    let mut full = Full::new(&m);
    let two_path = root.join("data/retro").join(&head2);
    let mut two = two_path.with_extension("A").exists().then(|| TwoLevel::load(&m, &two_path));
    eprintln!(
        "loaded in {:.1}s: vocab {}, d {}, body {:.1} MB, full head {:.1} MB, two-level head {} per token, isa {:?}, {threads} threads",
        t0.elapsed().as_secs_f64(),
        m.vocab,
        m.d,
        m.body_bytes() as f64 / 1e6,
        m.emb.bytes() as f64 / 1e6,
        two.as_ref().map_or("absent".to_string(), |t| format!("{:.2} MB", t.bytes() as f64 / 1e6)),
        match isa {
            Isa::Avx512 => "avx512",
            Isa::Avx2 => "avx2",
            Isa::Scalar => "scalar",
        }
    );
    let pcs = pieces(&root.join(format!("data/retro/{tag}-test.vocab")));
    let test = std::fs::read_to_string(root.join("data/norm/test.txt")).unwrap();

    match mode.as_str() {
        "check" => {
            let toks = tokenize(&model_path, &test.chars().take(n_tok * 4).collect::<String>());
            let n = n_tok.min(toks.len() - 1);
            let reference: Vec<f32> = std::fs::read(root.join(format!("data/retro/{tag}-test.full"))).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            m.forward(bos, 0, &pool, isa);
            let (mut ours, mut theirs, mut ours2, mut maxd, mut chars) = (0f64, 0f64, 0f64, 0f32, 0usize);
            for i in 0..n {
                let h = m.forward(toks[i], i + 1, &pool, isa).to_vec();
                full.run(&m, &h, &pool, isa);
                let nats = -full.logprob(toks[i + 1]);
                ours += nats as f64;
                theirs += reference[i] as f64;
                maxd = maxd.max((nats - reference[i]).abs());
                if let Some(t) = two.as_mut() {
                    ours2 -= t.logprob(toks[i + 1], &h, &pool, isa) as f64;
                }
                chars += utf8_chars(&pcs[toks[i + 1] as usize]);
            }
            let bpc = |x: f64| x / std::f64::consts::LN_2 / chars as f64;
            println!("bits body {body_bits} head {head_bits}, {n} tokens: full head ours {:.4} bpc vs llama.cpp {:.4} bpc, max |Δ| {maxd:.3} nats; two-level {:.4} bpc", bpc(ours), bpc(theirs), bpc(ours2));
        }
        "bench" => {
            let toks = tokenize(&model_path, &test.chars().take(n_tok * 4 + 4000).collect::<String>());
            let n = n_tok.min(toks.len() - 1);
            let rapl = ruler::energy::Rapl::open();
            let idle = rapl.as_ref().map(|r| r.idle_watts(std::time::Duration::from_secs(3)));
            let mut lines = Vec::new();
            for which in ["full", "two-level"] {
                if which == "two-level" && two.is_none() {
                    continue;
                }
                let (mut tb, mut th, mut nats, mut chars) = (0f64, 0f64, 0f64, 0usize);
                m.forward(bos, 0, &pool, isa);
                let e0 = rapl.as_ref().map(|r| r.read_uj());
                let start = Instant::now();
                for i in 0..n {
                    let a = Instant::now();
                    let h = m.forward(toks[i], i + 1, &pool, isa).to_vec();
                    let b = Instant::now();
                    let lp = if which == "full" {
                        full.run(&m, &h, &pool, isa);
                        full.logprob(toks[i + 1])
                    } else {
                        two.as_mut().unwrap().logprob(toks[i + 1], &h, &pool, isa)
                    };
                    tb += (b - a).as_secs_f64();
                    th += b.elapsed().as_secs_f64();
                    nats -= lp as f64;
                    chars += utf8_chars(&pcs[toks[i + 1] as usize]);
                }
                let secs = start.elapsed().as_secs_f64();
                let joules = match (&rapl, e0) {
                    (Some(r), Some(e0)) => r.joules_since(e0) - idle.unwrap() * secs,
                    _ => f64::NAN,
                };
                let line = format!(
                    "{{\"model\":\"{tag}\",\"file\":\"{}\",\"head\":\"{which}\",\"head2\":\"{head2}\",\"body_bits\":{body_bits},\"head_bits\":{head_bits},\"threads\":{threads},\"isa\":\"{}\",\"tokens\":{n},\"tok_per_s\":{:.1},\"body_us\":{:.1},\"head_us\":{:.1},\"head_share\":{:.3},\"uj_per_token\":{:.0},\"bpc\":{:.4},\"idle_w\":{:.1}}}",
                    model_path.file_name().unwrap().to_string_lossy(),
                    match isa {
                        Isa::Avx512 => "avx512",
                        Isa::Avx2 => "avx2",
                        Isa::Scalar => "scalar",
                    },
                    n as f64 / secs,
                    tb / n as f64 * 1e6,
                    th / n as f64 * 1e6,
                    th / (tb + th),
                    joules / n as f64 * 1e6,
                    nats / std::f64::consts::LN_2 / chars as f64,
                    idle.unwrap_or(f64::NAN)
                );
                println!("{line}");
                lines.push(line);
            }
            for l in lines {
                ruler::results::append("gem-bench.jsonl", &l);
            }
        }
        "gen" => {
            let prompt: String = arg(&args, "--prompt", "Era uma vez".to_string());
            let n: usize = arg(&args, "--n", 120);
            let temp: f32 = arg(&args, "--temp", 0.8);
            let seed: u64 = arg(&args, "--seed", 7);
            let ptoks = tokenize(&model_path, &prompt);
            for which in ["full", "two-level"] {
                if which == "two-level" && two.is_none() {
                    continue;
                }
                let mut rng = seed | 1;
                let mut h = m.forward(bos, 0, &pool, isa).to_vec();
                for (i, &t) in ptoks.iter().enumerate() {
                    h = m.forward(t, i + 1, &pool, isa).to_vec();
                }
                let mut out = Vec::new();
                let start = Instant::now();
                for i in 0..n {
                    let t = if which == "full" {
                        full.run(&m, &h, &pool, isa);
                        draw(&full.logits, temp, &mut rng) as u32
                    } else {
                        two.as_mut().unwrap().sample(&h, temp, &mut rng, &pool, isa)
                    };
                    if t == 1 || t == 106 {
                        break;
                    }
                    out.extend_from_slice(&pcs[t as usize]);
                    h = m.forward(t, ptoks.len() + 1 + i, &pool, isa).to_vec();
                }
                let secs = start.elapsed().as_secs_f64();
                println!("--- {which} head ({:.1} tok/s) ---\n{prompt}{}\n", n as f64 / secs, String::from_utf8_lossy(&out));
            }
        }
        _ => panic!("mode: check|bench|gen"),
    }
}

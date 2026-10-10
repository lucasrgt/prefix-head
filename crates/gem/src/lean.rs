//! `gem complete`: greedy completion of a folder of prompts with the smallest memory footprint the
//! runtime can manage — tensors read one at a time, input embeddings taken from the output matrix,
//! the two-level head reordering that matrix in place instead of copying its blocks.
//!
//!   gem complete --model m.gguf --tag qwencoder --prompts dir [--head2 qwencoder-prefix-self | --full]
//!                [--ctx 1024] [--max-tokens 512] [--classes 1] [--threads 8]
//!
//! Reads `dir/<n>.txt`, writes `dir/<n>.out`, prints one JSON line with tokens/s and peak RSS.

use crate::gguf::Gguf;
use crate::model::{Gemma, dot};
use crate::pool::Pool;
use crate::q8::{Act, Isa, matvec, matvec_rows};
use crate::{arg, pieces, tokenize_with};
use std::path::PathBuf;
use std::time::Instant;

struct Classes {
    a: Vec<f32>,
    b: Vec<f32>,
    order: Vec<u32>,
    per: usize,
    cl: Vec<f32>,
    inner: Vec<f32>,
    top: usize,
}

fn lse(v: &[f32]) -> f32 {
    let mx = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    mx + v.iter().map(|x| (x - mx).exp()).sum::<f32>().ln()
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold(0, |b, (i, &x)| if x > v[b] { i } else { b })
}

impl Classes {
    /// Greedy token under the two-level head: best (class, token) pair among the `top` most likely
    /// classes, scored by log P(class) + log P(token | class).
    fn next(&mut self, m: &Gemma, h: &[f32], act: &mut Act, pool: &Pool, isa: Isa) -> u32 {
        let d = m.d;
        for (c, out) in self.cl.iter_mut().enumerate() {
            *out = dot(&self.a[c * d..(c + 1) * d], h, isa) + self.b[c];
        }
        let norm = lse(&self.cl);
        act.quantize(h, isa);
        let mut ranked: Vec<usize> = (0..self.cl.len()).collect();
        ranked.sort_unstable_by(|&x, &y| self.cl[y].total_cmp(&self.cl[x]));
        let mut best = (f32::NEG_INFINITY, 0usize);
        for &c in &ranked[..self.top] {
            let first = c * self.per;
            let n = self.per.min(self.order.len() - first);
            matvec_rows(pool, isa, m.head(), first, act, &mut self.inner[..n]);
            let j = argmax(&self.inner[..n]);
            let score = self.cl[c] - norm + self.inner[j] - lse(&self.inner[..n]);
            if score > best.0 {
                best = (score, first + j);
            }
        }
        self.order[best.1]
    }
}

fn status_mb(key: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status").unwrap().lines().find(|l| l.starts_with(key)).and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok()).unwrap_or(0) / 1024
}

pub fn run(args: &[String]) {
    let root = ruler::results::root();
    let model_path: PathBuf = arg(args, "--model", PathBuf::new());
    let tag: String = arg(args, "--tag", String::new());
    let head2: String = arg(args, "--head2", format!("{tag}-prefix-self"));
    let full = args.iter().any(|a| a == "--full");
    let threads: usize = arg(args, "--threads", 8);
    let ctx: usize = arg(args, "--ctx", 1024);
    let max_tokens: usize = arg(args, "--max-tokens", 512);
    let top: usize = arg(args, "--classes", 1);
    let dir: PathBuf = arg(args, "--prompts", PathBuf::new());
    let isa = crate::q8::detect();
    let pool = Pool::new(threads);

    let g = Gguf::open(&model_path);
    let stops = [g.u("tokenizer.ggml.eos_token_id") as u32, g.u_or("tokenizer.ggml.bos_token_id", u32::MAX as usize) as u32];
    let mut m = Gemma::load(&g, ctx, true);
    drop(g);
    let mut classes = (!full).then(|| {
        let base = root.join("data/retro").join(&head2);
        let rd = |ext: &str| std::fs::read(base.with_extension(ext)).unwrap_or_else(|e| panic!("{}.{ext}: {e}", base.display()));
        let order: Vec<u32> = rd("order").chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        let a: Vec<f32> = rd("A").chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let b: Vec<f32> = rd("b").chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let per = order.len().div_ceil(b.len());
        m.permute_head(&order);
        Classes { cl: vec![0.0; b.len()], inner: vec![0.0; per], a, b, order, per, top }
    });
    let pcs = pieces(&root.join(format!("data/retro/{tag}-test.vocab")));
    let mut act = Act::new(m.d);
    let mut logits = if full { vec![0.0; m.vocab] } else { Vec::new() };
    let off = m.bos.is_some() as usize;

    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|e| e == "txt")).collect();
    files.sort();
    let (mut gen_tokens, mut gen_secs, mut prompt_tokens, mut prompt_secs) = (0usize, 0f64, 0usize, 0f64);
    for file in &files {
        let toks = tokenize_with(&model_path, &std::fs::read_to_string(file).unwrap(), true);
        let t0 = Instant::now();
        if let Some(b) = m.bos {
            m.forward(b, 0, &pool, isa);
        }
        let mut h = Vec::new();
        for (i, &t) in toks.iter().take(ctx - off - 1).enumerate() {
            h = m.forward(t, i + off, &pool, isa).to_vec();
        }
        let start = toks.len().min(ctx - off - 1) + off;
        prompt_tokens += start;
        prompt_secs += t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let mut out = Vec::new();
        for i in 0..max_tokens.min(ctx - start) {
            let t = match classes.as_mut() {
                Some(c) => c.next(&m, &h, &mut act, &pool, isa),
                None => {
                    act.quantize(&h, isa);
                    matvec(&pool, isa, m.head(), &act, &mut logits);
                    argmax(&logits) as u32
                }
            };
            gen_tokens += 1;
            if stops.contains(&t) {
                break;
            }
            out.extend_from_slice(&pcs[t as usize]);
            h = m.forward(t, start + i, &pool, isa).to_vec();
        }
        gen_secs += t1.elapsed().as_secs_f64();
        std::fs::write(file.with_extension("out"), &out).unwrap();
    }
    println!(
        "{{\"file\":\"{}\",\"head\":\"{}\",\"classes_tried\":{top},\"threads\":{threads},\"ctx\":{ctx},\"prompts\":{},\"gen_tokens\":{gen_tokens},\"tok_per_s\":{:.1},\"prompt_tok_per_s\":{:.1},\"peak_rss_mb\":{},\"rss_mb\":{}}}",
        model_path.file_name().unwrap().to_string_lossy(),
        if full { "full".to_string() } else { head2 },
        files.len(),
        gen_tokens as f64 / gen_secs,
        prompt_tokens as f64 / prompt_secs,
        status_mb("VmHWM"),
        status_mb("VmRSS")
    );
}

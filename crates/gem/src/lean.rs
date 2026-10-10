//! `gem complete`: greedy completion of a folder of prompts with the smallest memory footprint the
//! runtime can manage — tensors read one at a time, input embeddings taken from the output matrix,
//! the two-level head reordering that matrix in place instead of copying its blocks.
//!
//!   gem complete --model m.gguf --tag qwencoder --prompts dir [--head2 qwencoder-prefix-self | --full]
//!                [--ctx 1024] [--max-tokens 512] [--classes 1] [--threads 8]
//!   gem chat     [same flags]   interactive ChatML conversation on the terminal
//!
//! `complete` reads `dir/<n>.txt`, writes `dir/<n>.out`, prints one JSON line with tokens/s and peak RSS.

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
    min_confidence: f32,
    always: Vec<usize>,
}

fn lse(v: &[f32]) -> f32 {
    let mx = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    mx + v.iter().map(|x| (x - mx).exp()).sum::<f32>().ln()
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold(0, |b, (i, &x)| if x > v[b] { i } else { b })
}

impl Classes {
    /// Greedy token by screening: the `top` most likely classes and the always-on rows are evaluated
    /// and the highest raw logit wins — the rows are the model's own, so logits compare across
    /// classes, and the result equals the full head's whenever its token is among those evaluated.
    /// Returns None when the most likely class has probability below `min_confidence`, so the caller
    /// can fall back to the full head.
    fn next(&mut self, m: &Gemma, h: &[f32], act: &mut Act, pool: &Pool, isa: Isa) -> Option<u32> {
        let d = m.d;
        for (c, out) in self.cl.iter_mut().enumerate() {
            *out = dot(&self.a[c * d..(c + 1) * d], h, isa) + self.b[c];
        }
        let norm = lse(&self.cl);
        act.quantize(h, isa);
        let mut ranked: Vec<usize> = (0..self.cl.len()).collect();
        ranked.sort_unstable_by(|&x, &y| self.cl[y].total_cmp(&self.cl[x]));
        if (self.cl[ranked[0]] - norm).exp() < self.min_confidence {
            return None;
        }
        let mut best = (f32::NEG_INFINITY, 0usize);
        for &c in &ranked[..self.top] {
            let first = c * self.per;
            let n = self.per.min(self.order.len() - first);
            matvec_rows(pool, isa, m.head(), first, act, &mut self.inner[..n]);
            let j = argmax(&self.inner[..n]);
            if self.inner[j] > best.0 {
                best = (self.inner[j], first + j);
            }
        }
        for &row in &self.always {
            let mut one = [0f32];
            matvec_rows(pool, isa, m.head(), row, act, &mut one);
            if one[0] > best.0 {
                best = (one[0], row);
            }
        }
        Some(self.order[best.1])
    }
}

fn to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let (sign, mant) = ((x >> 16) & 0x8000, x & 0x7f_ffff);
    let exp = ((x >> 23) & 0xff) as i32 - 127 + 15;
    if exp <= 0 {
        return sign as u16;
    }
    if exp >= 31 {
        return (sign | 0x7c00) as u16;
    }
    (sign | (exp as u32) << 10 | mant >> 13) as u16 + ((mant >> 12) & 1) as u16
}

#[allow(clippy::too_many_arguments)]
fn next_token(classes: &mut Option<Classes>, m: &Gemma, h: &[f32], act: &mut Act, logits: &mut [f32], fallbacks: &mut usize, pool: &Pool, isa: Isa) -> u32 {
    if let Some(Some(t)) = classes.as_mut().map(|c| c.next(m, h, act, pool, isa)) {
        return t;
    }
    act.quantize(h, isa);
    matvec(pool, isa, m.head(), act, logits);
    let row = argmax(logits);
    match classes.as_ref() {
        Some(c) => {
            *fallbacks += 1;
            c.order[row]
        }
        None => row as u32,
    }
}

fn status_mb(key: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status").unwrap().lines().find(|l| l.starts_with(key)).and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok()).unwrap_or(0) / 1024
}

pub fn run(args: &[String]) {
    let root = ruler::results::root();
    let chat = args.get(1).is_some_and(|a| a == "chat");
    let model_path: PathBuf = arg(args, "--model", root.join("data/models/llm/qwen2.5-coder-0.5b-instruct-q4_0.gguf"));
    let tag: String = arg(args, "--tag", "qwencoder".to_string());
    let head2: String = arg(args, "--head2", if chat { "qwenchat-prefix-self".to_string() } else { format!("{tag}-prefix-self") });
    let full = args.iter().any(|a| a == "--full");
    let threads: usize = arg(args, "--threads", 8);
    let ctx: usize = arg(args, "--ctx", if chat { 2048 } else { 1024 });
    let max_tokens: usize = arg(args, "--max-tokens", if chat { 1024 } else { 512 });
    let top: usize = arg(args, "--classes", if chat { 64 } else { 1 });
    let min_confidence: f32 = arg(args, "--fallback", 0.0);
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
        Classes { cl: vec![0.0; b.len()], inner: vec![0.0; per], a, b, order, per, top, min_confidence, always: Vec::new() }
    });
    let pcs = pieces(&root.join(format!("data/retro/{tag}-test.vocab")));
    if let Some(c) = classes.as_mut() {
        if !args.iter().any(|a| a == "--no-always") {
            c.always = (0..m.vocab as u32).filter(|&t| stops.contains(&t) || pcs[t as usize].windows(3).any(|w| w == b"```") || pcs[t as usize].starts_with(b"<|")).map(|t| m.row_of(t)).collect();
        }
    }
    let mut act = Act::new(m.d);
    let mut logits = if full || min_confidence > 0.0 { vec![0.0; m.vocab] } else { Vec::new() };
    let mut fallbacks = 0usize;
    let off = m.bos.is_some() as usize;

    if chat {
        use std::io::{BufRead, Write};
        let system = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n";
        eprintln!("{} | {} | {} MB | /novo recomeça, /sair encerra", model_path.file_name().unwrap().to_string_lossy(), if full { "cabeça cheia".to_string() } else { format!("cabeça por prefixo, {top} classes") }, status_mb("VmRSS"));
        let (mut pos, mut fresh) = (0usize, true);
        let stdin = std::io::stdin();
        loop {
            eprint!("\n> ");
            let mut line = String::new();
            if stdin.lock().read_line(&mut line).unwrap() == 0 || line.trim() == "/sair" {
                break;
            }
            let msg = line.trim();
            if msg.is_empty() {
                continue;
            }
            if msg == "/novo" {
                (pos, fresh) = (0, true);
                continue;
            }
            let turn = format!("{}<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n", if fresh { system } else { "<|im_end|>\n" });
            let toks = tokenize_with(&model_path, &turn, true);
            if pos + toks.len() + 64 > ctx {
                eprintln!("(contexto cheio: comece outra conversa com /novo)");
                continue;
            }
            fresh = false;
            let mut h = Vec::new();
            for &t in &toks {
                h = m.forward(t, pos, &pool, isa).to_vec();
                pos += 1;
            }
            let (t0, mut n) = (Instant::now(), 0usize);
            let mut out = std::io::stdout().lock();
            while n < max_tokens && pos < ctx {
                let t = next_token(&mut classes, &m, &h, &mut act, &mut logits, &mut fallbacks, &pool, isa);
                n += 1;
                if stops.contains(&t) {
                    break;
                }
                out.write_all(&pcs[t as usize]).unwrap();
                out.flush().unwrap();
                h = m.forward(t, pos, &pool, isa).to_vec();
                pos += 1;
            }
            eprintln!("\n({n} tokens, {:.0} tokens/s, {} MB)", n as f64 / t0.elapsed().as_secs_f64(), status_mb("VmRSS"));
        }
        return;
    }

    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|e| e == "txt")).collect();
    files.sort();
    let dump: PathBuf = arg(args, "--dump", PathBuf::new());
    let (mut dump_h, mut dump_y, mut dump_chars) = (Vec::<u8>::new(), Vec::<u8>::new(), 0usize);
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
            let t = next_token(&mut classes, &m, &h, &mut act, &mut logits, &mut fallbacks, &pool, isa);
            gen_tokens += 1;
            if dump.as_os_str().len() > 0 {
                dump_h.extend(h.iter().flat_map(|&v| to_f16(v).to_le_bytes()));
                dump_y.extend(t.to_le_bytes());
                dump_chars += pcs[t as usize].iter().filter(|&&b| b & 0xC0 != 0x80).count().max(1);
            }
            if stops.contains(&t) {
                break;
            }
            out.extend_from_slice(&pcs[t as usize]);
            h = m.forward(t, start + i, &pool, isa).to_vec();
        }
        gen_secs += t1.elapsed().as_secs_f64();
        std::fs::write(file.with_extension("out"), &out).unwrap();
    }
    if dump.as_os_str().len() > 0 {
        std::fs::write(dump.with_extension("h"), &dump_h).unwrap();
        std::fs::write(dump.with_extension("y"), &dump_y).unwrap();
        std::fs::write(dump.with_extension("json"), format!("{{\"model\":\"{}\",\"d\":{},\"vocab\":{},\"rows\":{},\"chars\":{dump_chars},\"ctx\":{ctx},\"every\":1,\"tokens\":{}}}\n", model_path.display(), m.d, m.vocab, dump_y.len() / 4, dump_y.len() / 4)).unwrap();
    }
    println!(
        "{{\"file\":\"{}\",\"head\":\"{}\",\"classes_tried\":{top},\"fallback\":{min_confidence},\"fallback_share\":{:.4},\"threads\":{threads},\"ctx\":{ctx},\"prompts\":{},\"gen_tokens\":{gen_tokens},\"tok_per_s\":{:.1},\"prompt_tok_per_s\":{:.1},\"peak_rss_mb\":{},\"rss_mb\":{},\"rss_anon_mb\":{},\"rss_file_mb\":{}}}",
        model_path.file_name().unwrap().to_string_lossy(),
        if full { "full".to_string() } else { head2 },
        fallbacks as f64 / gen_tokens.max(1) as f64,
        files.len(),
        gen_tokens as f64 / gen_secs,
        prompt_tokens as f64 / prompt_secs,
        status_mb("VmHWM"),
        status_mb("VmRSS"),
        status_mb("RssAnon"),
        status_mb("RssFile")
    );
}

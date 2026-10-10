//! Gemma 3 text decoder, one token at a time, mirroring llama.cpp's graph (src/models/gemma3.cpp):
//! scaled embedding, RMSNorm with the +1 already folded into the GGUF weights, QK-norm, NeoX RoPE
//! (10k on sliding-window layers, the model base on global ones), GELU-tanh gated FFN, post-norms.

use crate::gguf::{F16, Gguf, Q4_0, Q8_0};
use crate::pool::Pool;
use crate::q8::{Act, Isa, Mat, matvec};
use std::arch::x86_64::*;

pub struct Layer {
    q: Mat,
    k: Mat,
    v: Mat,
    o: Mat,
    gate: Mat,
    up: Mat,
    down: Mat,
    attn_norm: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    post_attn: Vec<f32>,
    ffn_norm: Vec<f32>,
    post_ffw: Vec<f32>,
    swa: bool,
}

pub struct Gemma {
    pub d: usize,
    pub vocab: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    eps: f32,
    window: usize,
    pub layers: Vec<Layer>,
    pub emb: Mat,
    out_norm: Vec<f32>,
    pub ctx: usize,
    kc: Vec<Vec<f32>>,
    vc: Vec<Vec<f32>>,
    x: Vec<f32>,
    xn: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att: Vec<f32>,
    tmp: Vec<f32>,
    g: Vec<f32>,
    u: Vec<f32>,
    scores: Vec<f32>,
    ad: Act,
    aq: Act,
    aff: Act,
    pub prof: [f64; 6],
    inv_freq: [Vec<f32>; 2],
    rot: [Vec<(f32, f32)>; 2],
    gt: Vec<f32>,
    acc: Vec<f32>,
}

fn mat(g: &Gguf, name: &str) -> Mat {
    let (t, b) = g.tensor(name);
    match t.kind {
        Q8_0 => Mat::from_q8_0(b, t.dims[1], t.dims[0]),
        Q4_0 => Mat::from_q4_0(b, t.dims[1], t.dims[0]),
        F16 => Mat::from_f16(b, t.dims[1], t.dims[0]),
        k => panic!("{name}: unsupported tensor type {k}"),
    }
}

fn rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1.0 / (ms + eps).sqrt();
    for ((o, v), w) in out.iter_mut().zip(x).zip(w) {
        *o = v * r * w;
    }
}

fn rmsnorm_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1.0 / (ms + eps).sqrt();
    for (v, w) in x.iter_mut().zip(w) {
        *v *= r * w;
    }
}

fn rope(x: &mut [f32], table: &[(f32, f32)]) {
    let half = x.len() / 2;
    for (i, &(s, c)) in table.iter().enumerate() {
        let (a, b) = (x[i], x[i + half]);
        x[i] = a * c - b * s;
        x[i + half] = a * s + b * c;
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn dot512(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let mut acc = _mm512_setzero_ps();
        let n = a.len() / 16;
        for i in 0..n {
            acc = _mm512_fmadd_ps(_mm512_loadu_ps(a.as_ptr().add(i * 16)), _mm512_loadu_ps(b.as_ptr().add(i * 16)), acc);
        }
        let mut s = _mm512_reduce_add_ps(acc);
        for i in n * 16..a.len() {
            s += a[i] * b[i];
        }
        s
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn axpy512(p: f32, x: &[f32], y: &mut [f32]) {
    unsafe {
        let pv = _mm512_set1_ps(p);
        let n = x.len() / 16;
        for i in 0..n {
            let o = y.as_mut_ptr().add(i * 16);
            _mm512_storeu_ps(o, _mm512_fmadd_ps(pv, _mm512_loadu_ps(x.as_ptr().add(i * 16)), _mm512_loadu_ps(o)));
        }
        for i in n * 16..x.len() {
            y[i] += p * x[i];
        }
    }
}

fn axpy(p: f32, x: &[f32], y: &mut [f32], isa: Isa) {
    if isa == Isa::Avx512 {
        unsafe { axpy512(p, x, y) }
    } else {
        for (o, v) in y.iter_mut().zip(x) {
            *o += p * v;
        }
    }
}

pub fn dot(a: &[f32], b: &[f32], isa: Isa) -> f32 {
    if isa == Isa::Avx512 { unsafe { dot512(a, b) } } else { a.iter().zip(b).map(|(x, y)| x * y).sum() }
}

impl Gemma {
    pub fn load(g: &Gguf, ctx: usize) -> Self {
        let a = "gemma3";
        let d = g.u(&format!("{a}.embedding_length"));
        let nl = g.u(&format!("{a}.block_count"));
        let nh = g.u(&format!("{a}.attention.head_count"));
        let nkv = g.u(&format!("{a}.attention.head_count_kv"));
        let hd = g.u(&format!("{a}.attention.key_length"));
        let ff = g.u(&format!("{a}.feed_forward_length"));
        let eps = g.f(&format!("{a}.attention.layer_norm_rms_epsilon"));
        let window = g.u(&format!("{a}.attention.sliding_window"));
        let base = g.f(&format!("{a}.rope.freq_base"));
        let layers = (0..nl)
            .map(|l| {
                let p = |s: &str| format!("blk.{l}.{s}");
                Layer {
                    q: mat(g, &p("attn_q.weight")),
                    k: mat(g, &p("attn_k.weight")),
                    v: mat(g, &p("attn_v.weight")),
                    o: mat(g, &p("attn_output.weight")),
                    gate: mat(g, &p("ffn_gate.weight")),
                    up: mat(g, &p("ffn_up.weight")),
                    down: mat(g, &p("ffn_down.weight")),
                    attn_norm: g.f32s(&p("attn_norm.weight")),
                    q_norm: g.f32s(&p("attn_q_norm.weight")),
                    k_norm: g.f32s(&p("attn_k_norm.weight")),
                    post_attn: g.f32s(&p("post_attention_norm.weight")),
                    ffn_norm: g.f32s(&p("ffn_norm.weight")),
                    post_ffw: g.f32s(&p("post_ffw_norm.weight")),
                    swa: l % 6 < 5,
                }
            })
            .collect();
        let emb = mat(g, "token_embd.weight");
        let vocab = emb.rows;
        Self {
            d,
            vocab,
            nh,
            nkv,
            hd,
            eps,
            window,
            layers,
            emb,
            out_norm: g.f32s("output_norm.weight"),
            ctx,
            kc: (0..nl).map(|_| vec![0.0; ctx * nkv * hd]).collect(),
            vc: (0..nl).map(|_| vec![0.0; ctx * nkv * hd]).collect(),
            x: vec![0.0; d],
            xn: vec![0.0; d],
            q: vec![0.0; nh * hd],
            k: vec![0.0; nkv * hd],
            v: vec![0.0; nkv * hd],
            att: vec![0.0; nh * hd],
            tmp: vec![0.0; d],
            g: vec![0.0; ff],
            u: vec![0.0; ff],
            scores: vec![0.0; nh * ctx],
            ad: Act::new(d),
            aq: Act::new(nh * hd),
            aff: Act::new(ff),
            prof: [0.0; 6],
            inv_freq: [10000.0f32, base].map(|b| (0..hd / 2).map(|i| b.powf(-2.0 * i as f32 / hd as f32)).collect()),
            rot: [vec![(0.0, 1.0); hd / 2], vec![(0.0, 1.0); hd / 2]],
            gt: vec![0.0; ff],
            acc: Vec::new(),
        }
    }

    pub fn requantize_head(&mut self, bits: u8) {
        self.emb = match bits {
            8 => self.emb.to_q8(),
            4 => self.emb.to_q4(),
            _ => return,
        };
    }

    pub fn requantize(&mut self, body4: bool, head4: bool) {
        if body4 {
            for l in &mut self.layers {
                for m in [&mut l.q, &mut l.k, &mut l.v, &mut l.o, &mut l.gate, &mut l.up, &mut l.down] {
                    *m = m.to_q4();
                }
            }
        }
        if head4 {
            self.emb = self.emb.to_q4();
        }
    }

    pub fn body_bytes(&self) -> usize {
        self.layers.iter().map(|l| [&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down].iter().map(|m| m.bytes()).sum::<usize>()).sum()
    }

    /// Runs token `tok` at position `pos`; returns the final-norm hidden state.
    pub fn forward(&mut self, tok: u32, pos: usize, pool: &Pool, isa: Isa) -> &[f32] {
        assert!(pos < self.ctx);
        let (d, hd, nh, nkv, eps) = (self.d, self.hd, self.nh, self.nkv, self.eps);
        let scale = (d as f32).sqrt();
        for (x, e) in self.x.iter_mut().zip(self.emb.row_f32(tok as usize)) {
            *x = e * scale;
        }
        if self.acc.len() != pool.threads() * nh * hd {
            self.acc = vec![0.0; pool.threads() * nh * hd];
        }
        for (rot, inv) in self.rot.iter_mut().zip(&self.inv_freq) {
            for (r, f) in rot.iter_mut().zip(inv) {
                *r = (pos as f32 * f).sin_cos();
            }
        }
        for l in 0..self.layers.len() {
            let ly = &self.layers[l];
            rmsnorm(&self.x, &ly.attn_norm, eps, &mut self.xn);
            self.ad.quantize(&self.xn, isa);
            let t0 = std::time::Instant::now();
            matvec(pool, isa, &ly.q, &self.ad, &mut self.q);
            matvec(pool, isa, &ly.k, &self.ad, &mut self.k);
            matvec(pool, isa, &ly.v, &self.ad, &mut self.v);
            self.prof[0] += t0.elapsed().as_secs_f64();
            let table = &self.rot[if ly.swa { 0 } else { 1 }];
            let qs = 1.0 / (hd as f32).sqrt();
            for h in 0..nh {
                let qh = &mut self.q[h * hd..(h + 1) * hd];
                rmsnorm_inplace(qh, &ly.q_norm, eps);
                rope(qh, table);
                qh.iter_mut().for_each(|v| *v *= qs);
            }
            for h in 0..nkv {
                let kh = &mut self.k[h * hd..(h + 1) * hd];
                rmsnorm_inplace(kh, &ly.k_norm, eps);
                rope(kh, table);
            }
            let row = nkv * hd;
            self.kc[l][pos * row..(pos + 1) * row].copy_from_slice(&self.k);
            self.vc[l][pos * row..(pos + 1) * row].copy_from_slice(&self.v);
            let lo = if ly.swa { (pos + 1).saturating_sub(self.window) } else { 0 };
            let (kc, vc, q, ctx) = (&self.kc[l], &self.vc[l], &self.q, self.ctx);
            let n_pos = pos + 1 - lo;
            let threads = pool.threads();
            let per = n_pos.div_ceil(threads);
            let group = nh / nkv;
            let sc = SendPtr(self.scores.as_mut_ptr());
            let t1 = std::time::Instant::now();
            pool.run(&|t| {
                let sc = &sc;
                for i in (t * per).min(n_pos)..((t + 1) * per).min(n_pos) {
                    let krow = &kc[(lo + i) * row..(lo + i + 1) * row];
                    for h in 0..nh {
                        let kvh = h / group;
                        unsafe { *sc.0.add(h * ctx + i) = dot(&q[h * hd..(h + 1) * hd], &krow[kvh * hd..(kvh + 1) * hd], isa) };
                    }
                }
            });
            for h in 0..nh {
                let s = &mut self.scores[h * ctx..h * ctx + n_pos];
                let mx = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                s.iter_mut().for_each(|v| *v -= mx);
                nn::fast::exp_inplace(s);
                let inv = 1.0 / s.iter().sum::<f32>();
                s.iter_mut().for_each(|v| *v *= inv);
            }
            let width = nh * hd;
            self.acc.iter_mut().for_each(|v| *v = 0.0);
            let acc = SendPtr(self.acc.as_mut_ptr());
            let scores = &self.scores;
            pool.run(&|t| {
                let acc = &acc;
                let out = unsafe { std::slice::from_raw_parts_mut(acc.0.add(t * width), width) };
                for i in (t * per).min(n_pos)..((t + 1) * per).min(n_pos) {
                    let vrow = &vc[(lo + i) * row..(lo + i + 1) * row];
                    for h in 0..nh {
                        let kvh = h / group;
                        axpy(scores[h * ctx + i], &vrow[kvh * hd..(kvh + 1) * hd], &mut out[h * hd..(h + 1) * hd], isa);
                    }
                }
            });
            self.att.iter_mut().for_each(|v| *v = 0.0);
            for t in 0..threads {
                for (o, v) in self.att.iter_mut().zip(&self.acc[t * width..(t + 1) * width]) {
                    *o += v;
                }
            }
            self.prof[1] += t1.elapsed().as_secs_f64();
            self.aq.quantize(&self.att, isa);
            let t2 = std::time::Instant::now();
            matvec(pool, isa, &ly.o, &self.aq, &mut self.tmp);
            self.prof[2] += t2.elapsed().as_secs_f64();
            rmsnorm_inplace(&mut self.tmp, &ly.post_attn, eps);
            for (x, t) in self.x.iter_mut().zip(&self.tmp) {
                *x += t;
            }
            rmsnorm(&self.x, &ly.ffn_norm, eps, &mut self.xn);
            self.ad.quantize(&self.xn, isa);
            let t3 = std::time::Instant::now();
            matvec(pool, isa, &ly.gate, &self.ad, &mut self.g);
            matvec(pool, isa, &ly.up, &self.ad, &mut self.u);
            self.prof[3] += t3.elapsed().as_secs_f64();
            for (t, &g) in self.gt.iter_mut().zip(&self.g) {
                *t = 0.797_884_6 * (g + 0.044715 * g * g * g);
            }
            nn::fast::tanh_inplace(&mut self.gt);
            for ((g, u), t) in self.g.iter_mut().zip(&self.u).zip(&self.gt) {
                *g = 0.5 * *g * (1.0 + t) * u;
            }
            self.aff.quantize(&self.g, isa);
            let t4 = std::time::Instant::now();
            matvec(pool, isa, &ly.down, &self.aff, &mut self.tmp);
            self.prof[4] += t4.elapsed().as_secs_f64();
            rmsnorm_inplace(&mut self.tmp, &ly.post_ffw, eps);
            for (x, t) in self.x.iter_mut().zip(&self.tmp) {
                *x += t;
            }
        }
        rmsnorm(&self.x, &self.out_norm, eps, &mut self.xn);
        &self.xn
    }
}

struct SendPtr(*mut f32);
unsafe impl Sync for SendPtr {}

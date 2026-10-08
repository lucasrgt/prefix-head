use crate::train::{Param, gemm};
use kern::Rng;

/// MLP over tokens made of F factors (token t = seq[t*F..(t+1)*F]).
/// Input: each context slot embeds as the sum of its factor embeddings.
/// Output: factors predicted in order; factor f reads h_f = tanh(W1·x + b1 + Σ_{g<f} C_g[c_g]),
/// so the summed loss is exactly −ln p(token).
/// With `cond_head`, factor q has one output matrix per prefix (c_0..c_{q−1}): parameters grow like a
/// flat softmax, but each token still reads only Σ h·V_q weights.
/// With `full_in`, a context token embeds as one row of a full-vocabulary table instead of a sum of
/// factor rows (a lookup either way, so it costs no MACs).
pub struct FMlp {
    pub radix: Vec<usize>,
    pub cond_head: bool,
    pub full_in: bool,
    pub d: usize,
    pub k: usize,
    pub h: usize,
    ein: Vec<Param>,
    w1: Param,
    b1: Param,
    cond: Vec<Param>,
    w2: Vec<Param>,
    b2: Vec<Param>,
}

pub struct Batch {
    x: Vec<f32>,
    pre: Vec<f32>,
    hs: Vec<Vec<f32>>,
    zs: Vec<Vec<f32>>,
    dpre: Vec<f32>,
    dx: Vec<f32>,
    ctx: Vec<u32>,
}

impl FMlp {
    pub fn new(radix: Vec<usize>, d: usize, k: usize, h: usize, seed: u64, cond_head: bool, full_in: bool) -> Self {
        let mut rng = Rng::new(seed);
        let f = radix.len();
        let prefixes: Vec<usize> = (0..f).map(|q| if cond_head { radix[..q].iter().product() } else { 1 }).collect();
        let ein = if full_in {
            vec![Param::new((radix.iter().product::<usize>() + 1) * d, 1.0, &mut rng)]
        } else {
            radix.iter().map(|&v| Param::new((v + 1) * d, 1.0 / (f as f32).sqrt(), &mut rng)).collect()
        };
        let w1 = Param::new(k * d * h, 1.0 / ((k * d) as f32).sqrt(), &mut rng);
        let b1 = Param::new(h, 0.0, &mut rng);
        let cond = radix[..f - 1].iter().map(|&v| Param::new(v * h, 0.1, &mut rng)).collect();
        let w2 = radix.iter().zip(&prefixes).map(|(&v, &p)| Param::new(p * h * v, 1.0 / (h as f32).sqrt(), &mut rng)).collect();
        let b2 = radix.iter().zip(&prefixes).map(|(&v, &p)| Param::new(p * v, 0.0, &mut rng)).collect();
        Self { radix, cond_head, full_in, d, k, h, ein, w1, b1, cond, w2, b2 }
    }

    /// (table, row) pairs whose sum is the embedding of one context slot.
    fn rows(&self, slot: &[u32]) -> Vec<(usize, usize)> {
        if !self.full_in {
            return slot.iter().enumerate().map(|(q, &s)| (q, s as usize)).collect();
        }
        let pad = slot.iter().zip(&self.radix).any(|(&s, &r)| s as usize == r);
        let id = if pad { self.radix.iter().product() } else { slot.iter().zip(&self.radix).fold(0, |a, (&s, &r)| a * r + s as usize) };
        vec![(0, id)]
    }

    fn prefix(&self, tok: &[u32], q: usize) -> usize {
        if !self.cond_head {
            return 0;
        }
        (0..q).fold(0, |a, g| a * self.radix[g] + tok[g] as usize)
    }

    fn per_prefix(&self, q: usize) -> bool {
        self.cond_head && q > 0
    }

    pub fn f(&self) -> usize {
        self.radix.len()
    }

    pub fn params(&self) -> usize {
        let all = self.ein.iter().chain(&self.cond).chain(&self.w2).chain(&self.b2);
        all.map(|p| p.w.len()).sum::<usize>() + self.w1.w.len() + self.b1.w.len()
    }

    pub fn macs_per_token(&self) -> usize {
        self.k * self.d * self.h + self.radix.iter().map(|v| self.h * v).sum::<usize>()
    }

    pub fn batch(&self, n: usize) -> Batch {
        Batch {
            x: vec![0.0; n * self.k * self.d],
            pre: vec![0.0; n * self.h],
            hs: (0..self.f()).map(|_| vec![0.0; n * self.h]).collect(),
            zs: self.radix.iter().map(|&v| vec![0.0; n * v]).collect(),
            dpre: vec![0.0; n * self.h],
            dx: vec![0.0; n * self.k * self.d],
            ctx: vec![0; n * self.k * self.f()],
        }
    }

    fn context(&self, seq: &[u32], t: usize, out: &mut [u32]) {
        let f = self.f();
        for j in 0..self.k {
            let back = self.k - j;
            for q in 0..f {
                out[j * f + q] = if t >= back { seq[(t - back) * f + q] } else { self.radix[q] as u32 };
            }
        }
    }

    fn forward(&self, seq: &[u32], ts: &[usize], bt: &mut Batch) -> f64 {
        let (n, kd, h, f, d) = (ts.len(), self.k * self.d, self.h, self.f(), self.d);
        for (i, &t) in ts.iter().enumerate() {
            let ctx = &mut bt.ctx[i * self.k * f..(i + 1) * self.k * f];
            self.context(seq, t, ctx);
            for j in 0..self.k {
                let slot = &mut bt.x[i * kd + j * d..i * kd + (j + 1) * d];
                slot.fill(0.0);
                for (tb, s) in self.rows(&ctx[j * f..(j + 1) * f]) {
                    for (x, &e) in slot.iter_mut().zip(&self.ein[tb].w[s * d..(s + 1) * d]) {
                        *x += e;
                    }
                }
            }
            bt.pre[i * h..(i + 1) * h].copy_from_slice(&self.b1.w);
        }
        gemm(n, kd, h, &bt.x[..n * kd], false, &self.w1.w, false, 1.0, &mut bt.pre[..n * h]);

        let mut loss = 0.0f64;
        for q in 0..f {
            let v = self.radix[q];
            for (i, &t) in ts.iter().enumerate() {
                let hq = &mut bt.hs[q][i * h..(i + 1) * h];
                hq.copy_from_slice(&bt.pre[i * h..(i + 1) * h]);
                for g in 0..q {
                    let c = seq[t * f + g] as usize;
                    for (a, &e) in hq.iter_mut().zip(&self.cond[g].w[c * h..(c + 1) * h]) {
                        *a += e;
                    }
                }
                for a in hq.iter_mut() {
                    *a = a.tanh();
                }
                let p = self.prefix(&seq[t * f..(t + 1) * f], q);
                bt.zs[q][i * v..(i + 1) * v].copy_from_slice(&self.b2[q].w[p * v..(p + 1) * v]);
            }
            if self.per_prefix(q) {
                for (i, &t) in ts.iter().enumerate() {
                    let p = self.prefix(&seq[t * f..(t + 1) * f], q);
                    let w = &self.w2[q].w[p * h * v..(p + 1) * h * v];
                    let z = &mut bt.zs[q][i * v..(i + 1) * v];
                    for (r, &hv) in bt.hs[q][i * h..(i + 1) * h].iter().enumerate() {
                        for (zz, &ww) in z.iter_mut().zip(&w[r * v..(r + 1) * v]) {
                            *zz += hv * ww;
                        }
                    }
                }
            } else {
                gemm(n, h, v, &bt.hs[q][..n * h], false, &self.w2[q].w, false, 1.0, &mut bt.zs[q][..n * v]);
            }
            for (i, &t) in ts.iter().enumerate() {
                let row = &mut bt.zs[q][i * v..(i + 1) * v];
                let mx = row.iter().copied().fold(f32::MIN, f32::max);
                let lse = mx + row.iter().map(|&z| (z - mx).exp()).sum::<f32>().ln();
                for z in row.iter_mut() {
                    *z -= lse;
                }
                loss -= row[seq[t * f + q] as usize] as f64;
            }
        }
        loss
    }

    pub fn train_step(&mut self, seq: &[u32], ts: &[usize], bt: &mut Batch, lr: f32, step: i32) -> f64 {
        let loss = self.forward(seq, ts, bt);
        let (n, kd, h, f, d) = (ts.len(), self.k * self.d, self.h, self.f(), self.d);
        let inv = 1.0 / n as f32;
        bt.dpre[..n * h].fill(0.0);
        let mut dh = vec![0.0f32; n * h];
        for q in 0..f {
            let v = self.radix[q];
            for (i, &t) in ts.iter().enumerate() {
                let row = &mut bt.zs[q][i * v..(i + 1) * v];
                for z in row.iter_mut() {
                    *z = z.exp() * inv;
                }
                row[seq[t * f + q] as usize] -= inv;
                let p = self.prefix(&seq[t * f..(t + 1) * f], q);
                for (g, &dz) in self.b2[q].g[p * v..(p + 1) * v].iter_mut().zip(row.iter()) {
                    *g += dz;
                }
            }
            if self.per_prefix(q) {
                let prefixes: Vec<usize> = ts.iter().map(|&t| self.prefix(&seq[t * f..(t + 1) * f], q)).collect();
                let param = &mut self.w2[q];
                for (i, &p) in prefixes.iter().enumerate() {
                    let dz = &bt.zs[q][i * v..(i + 1) * v];
                    let w = &param.w[p * h * v..(p + 1) * h * v];
                    let gw = &mut param.g[p * h * v..(p + 1) * h * v];
                    for r in 0..h {
                        let hv = bt.hs[q][i * h + r];
                        let (wr, gr) = (&w[r * v..(r + 1) * v], &mut gw[r * v..(r + 1) * v]);
                        let mut acc = 0.0f32;
                        for c in 0..v {
                            gr[c] += hv * dz[c];
                            acc += wr[c] * dz[c];
                        }
                        dh[i * h + r] = acc;
                    }
                }
            } else {
                gemm(h, n, v, &bt.hs[q][..n * h], true, &bt.zs[q][..n * v], false, 0.0, &mut self.w2[q].g);
                gemm(n, v, h, &bt.zs[q][..n * v], false, &self.w2[q].w, true, 0.0, &mut dh);
            }
            for (i, &t) in ts.iter().enumerate() {
                for j in 0..h {
                    let a = bt.hs[q][i * h + j];
                    let g = dh[i * h + j] * (1.0 - a * a);
                    dh[i * h + j] = g;
                    bt.dpre[i * h + j] += g;
                }
                for g in 0..q {
                    let c = seq[t * f + g] as usize;
                    for (w, &dg) in self.cond[g].g[c * h..(c + 1) * h].iter_mut().zip(&dh[i * h..(i + 1) * h]) {
                        *w += dg;
                    }
                }
            }
        }
        for i in 0..n {
            for (g, &dp) in self.b1.g.iter_mut().zip(&bt.dpre[i * h..(i + 1) * h]) {
                *g += dp;
            }
        }
        gemm(kd, n, h, &bt.x[..n * kd], true, &bt.dpre[..n * h], false, 0.0, &mut self.w1.g);
        gemm(n, h, kd, &bt.dpre[..n * h], false, &self.w1.w, true, 0.0, &mut bt.dx[..n * kd]);
        for i in 0..n {
            let ctx = &bt.ctx[i * self.k * f..(i + 1) * self.k * f];
            for j in 0..self.k {
                let src = &bt.dx[i * kd + j * d..i * kd + (j + 1) * d];
                for (tb, s) in self.rows(&ctx[j * f..(j + 1) * f]) {
                    for (g, &dv) in self.ein[tb].g[s * d..(s + 1) * d].iter_mut().zip(src) {
                        *g += dv;
                    }
                }
            }
        }
        let params = self.ein.iter_mut().chain(self.cond.iter_mut()).chain(self.w2.iter_mut()).chain(self.b2.iter_mut());
        for p in params.chain([&mut self.w1, &mut self.b1]) {
            p.adam(lr, step);
        }
        loss
    }

    pub fn eval_nats(&self, seq: &[u32]) -> f64 {
        let chunk = 4096;
        let mut bt = self.batch(chunk);
        let ts: Vec<usize> = (0..seq.len() / self.f()).collect();
        ts.chunks(chunk).map(|c| self.forward(seq, c, &mut bt)).sum()
    }

    /// Batch-1 generation step on one core; `pick` chooses factor q from its logits.
    pub fn step_single(&self, ctx: &[u32], mut pick: impl FnMut(usize, &[f32]) -> u32, out: &mut [u32]) {
        let (h, d, f) = (self.h, self.d, self.f());
        let mut pre = self.b1.w.clone();
        let mut x = vec![0.0f32; d];
        for j in 0..self.k {
            x.fill(0.0);
            for (tb, s) in self.rows(&ctx[j * f..(j + 1) * f]) {
                for (a, &e) in x.iter_mut().zip(&self.ein[tb].w[s * d..(s + 1) * d]) {
                    *a += e;
                }
            }
            for (r, &xv) in x.iter().enumerate() {
                let row = &self.w1.w[(j * d + r) * h..(j * d + r + 1) * h];
                for (p, &w) in pre.iter_mut().zip(row) {
                    *p += xv * w;
                }
            }
        }
        let mut hq = vec![0.0f32; h];
        for q in 0..f {
            let v = self.radix[q];
            hq.copy_from_slice(&pre);
            for g in 0..q {
                let c = out[g] as usize;
                for (a, &e) in hq.iter_mut().zip(&self.cond[g].w[c * h..(c + 1) * h]) {
                    *a += e;
                }
            }
            for a in hq.iter_mut() {
                *a = a.tanh();
            }
            let p = self.prefix(out, q);
            let mut logits = self.b2[q].w[p * v..(p + 1) * v].to_vec();
            let w2 = &self.w2[q].w[p * h * v..(p + 1) * h * v];
            for (r, &hv) in hq.iter().enumerate() {
                for (l, &w) in logits.iter_mut().zip(&w2[r * v..(r + 1) * v]) {
                    *l += hv * w;
                }
            }
            out[q] = pick(q, &logits);
        }
    }

    pub fn context_of(&self, seq: &[u32], t: usize) -> Vec<u32> {
        let mut c = vec![0; self.k * self.f()];
        self.context(seq, t, &mut c);
        c
    }
}

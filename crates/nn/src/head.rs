use crate::fast::{gemv_rows, logsumexp, tanh_inplace};
use crate::train::{Param, gemm};
use kern::Rng;

/// Coarse-to-fine output over tokens of F factors. Factor q reads h_q = tanh(x + Σ_{g<q} C_g[c_g])
/// and, past the first, an output matrix chosen by its prefix (c_0..c_{q−1}): parameters grow like a
/// flat softmax while each token reads only Σ h·V_q weights. Summed loss is exactly −ln p(token).
pub struct Head {
    pub radix: Vec<usize>,
    pub h: usize,
    cond: Vec<Param>,
    w2: Vec<Param>,
    b2: Vec<Param>,
}

pub struct HeadBuf {
    /// −ln p of each row's token from the last `forward`.
    pub row_loss: Vec<f32>,
    hs: Vec<Vec<f32>>,
    zs: Vec<Vec<f32>>,
    dh: Vec<f32>,
}

pub struct HeadScratch {
    hq: Vec<f32>,
    logits: Vec<f32>,
}

impl Head {
    pub fn new(radix: Vec<usize>, h: usize, rng: &mut Rng) -> Self {
        let f = radix.len();
        let cond = radix[..f - 1].iter().map(|&v| Param::new(v * h, 0.1, rng)).collect();
        let w2 = (0..f).map(|q| Param::new(Self::prefixes_of(&radix, q) * h * radix[q], 1.0 / (h as f32).sqrt(), rng)).collect();
        let b2 = (0..f).map(|q| Param::new(Self::prefixes_of(&radix, q) * radix[q], 0.0, rng)).collect();
        Self { radix, h, cond, w2, b2 }
    }

    fn prefixes_of(radix: &[usize], q: usize) -> usize {
        radix[..q].iter().product()
    }

    fn f(&self) -> usize {
        self.radix.len()
    }

    fn prefix(&self, tok: &[u32], q: usize) -> usize {
        (0..q).fold(0, |a, g| a * self.radix[g] + tok[g] as usize)
    }

    pub fn params(&self) -> usize {
        self.cond.iter().chain(&self.w2).chain(&self.b2).map(|p| p.w.len()).sum()
    }

    pub fn macs(&self) -> usize {
        self.radix.iter().map(|v| self.h * v).sum()
    }

    pub(crate) fn params_mut(&mut self) -> impl Iterator<Item = &mut Param> {
        self.cond.iter_mut().chain(self.w2.iter_mut()).chain(self.b2.iter_mut())
    }

    pub fn buf(&self, n: usize) -> HeadBuf {
        HeadBuf {
            row_loss: vec![0.0; n],
            hs: (0..self.f()).map(|_| vec![0.0; n * self.h]).collect(),
            zs: self.radix.iter().map(|&v| vec![0.0; n * v]).collect(),
            dh: vec![0.0; n * self.h],
        }
    }

    /// `x` is n×h, `tgt` is n×F; leaves log-probabilities in the buffer; returns Σ −ln p.
    pub fn forward(&self, x: &[f32], tgt: &[u32], n: usize, b: &mut HeadBuf) -> f64 {
        let (h, f) = (self.h, self.f());
        let mut loss = 0.0f64;
        b.row_loss[..n].fill(0.0);
        for q in 0..f {
            let v = self.radix[q];
            for i in 0..n {
                let hq = &mut b.hs[q][i * h..(i + 1) * h];
                hq.copy_from_slice(&x[i * h..(i + 1) * h]);
                for g in 0..q {
                    let c = tgt[i * f + g] as usize;
                    for (a, &e) in hq.iter_mut().zip(&self.cond[g].w[c * h..(c + 1) * h]) {
                        *a += e;
                    }
                }
                for a in hq.iter_mut() {
                    *a = a.tanh();
                }
                let p = self.prefix(&tgt[i * f..(i + 1) * f], q);
                b.zs[q][i * v..(i + 1) * v].copy_from_slice(&self.b2[q].w[p * v..(p + 1) * v]);
            }
            if q == 0 {
                gemm(n, h, v, &b.hs[q][..n * h], false, &self.w2[q].w, false, 1.0, &mut b.zs[q][..n * v]);
            } else {
                for i in 0..n {
                    let p = self.prefix(&tgt[i * f..(i + 1) * f], q);
                    let w = &self.w2[q].w[p * h * v..(p + 1) * h * v];
                    let z = &mut b.zs[q][i * v..(i + 1) * v];
                    for (r, &hv) in b.hs[q][i * h..(i + 1) * h].iter().enumerate() {
                        for (zz, &ww) in z.iter_mut().zip(&w[r * v..(r + 1) * v]) {
                            *zz += hv * ww;
                        }
                    }
                }
            }
            for i in 0..n {
                let row = &mut b.zs[q][i * v..(i + 1) * v];
                let mx = row.iter().copied().fold(f32::MIN, f32::max);
                let lse = mx + row.iter().map(|&z| (z - mx).exp()).sum::<f32>().ln();
                for z in row.iter_mut() {
                    *z -= lse;
                }
                loss -= row[tgt[i * f + q] as usize] as f64;
                b.row_loss[i] -= row[tgt[i * f + q] as usize];
            }
        }
        loss
    }

    /// Gradients of `scale`·loss: accumulates parameter grads and writes dloss/dx into `dx` (n×h).
    pub fn backward(&mut self, tgt: &[u32], n: usize, scale: f32, b: &mut HeadBuf, dx: &mut [f32]) {
        let (h, f) = (self.h, self.f());
        dx[..n * h].fill(0.0);
        for q in 0..f {
            let v = self.radix[q];
            for i in 0..n {
                let row = &mut b.zs[q][i * v..(i + 1) * v];
                for z in row.iter_mut() {
                    *z = z.exp() * scale;
                }
                row[tgt[i * f + q] as usize] -= scale;
                let p = self.prefix(&tgt[i * f..(i + 1) * f], q);
                for (g, &dz) in self.b2[q].g[p * v..(p + 1) * v].iter_mut().zip(row.iter()) {
                    *g += dz;
                }
            }
            if q == 0 {
                gemm(h, n, v, &b.hs[q][..n * h], true, &b.zs[q][..n * v], false, 1.0, &mut self.w2[q].g);
                gemm(n, v, h, &b.zs[q][..n * v], false, &self.w2[q].w, true, 0.0, &mut b.dh[..n * h]);
            } else {
                let prefixes: Vec<usize> = (0..n).map(|i| self.prefix(&tgt[i * f..(i + 1) * f], q)).collect();
                let param = &mut self.w2[q];
                for (i, &p) in prefixes.iter().enumerate() {
                    let dz = &b.zs[q][i * v..(i + 1) * v];
                    let w = &param.w[p * h * v..(p + 1) * h * v];
                    let gw = &mut param.g[p * h * v..(p + 1) * h * v];
                    for r in 0..h {
                        let hv = b.hs[q][i * h + r];
                        let (wr, gr) = (&w[r * v..(r + 1) * v], &mut gw[r * v..(r + 1) * v]);
                        let mut acc = 0.0f32;
                        for c in 0..v {
                            gr[c] += hv * dz[c];
                            acc += wr[c] * dz[c];
                        }
                        b.dh[i * h + r] = acc;
                    }
                }
            }
            for i in 0..n {
                for j in 0..h {
                    let a = b.hs[q][i * h + j];
                    let g = b.dh[i * h + j] * (1.0 - a * a);
                    b.dh[i * h + j] = g;
                    dx[i * h + j] += g;
                }
                for g in 0..q {
                    let c = tgt[i * f + g] as usize;
                    for (w, &dg) in self.cond[g].g[c * h..(c + 1) * h].iter_mut().zip(&b.dh[i * h..(i + 1) * h]) {
                        *w += dg;
                    }
                }
            }
        }
    }

    pub fn scratch(&self) -> HeadScratch {
        HeadScratch { hq: vec![0.0; self.h], logits: vec![0.0; self.radix.iter().copied().max().unwrap()] }
    }

    /// One token on one core: `pick(q, logits, lse)` chooses factor q; returns ln p of the picks.
    pub fn infer(&self, x: &[f32], s: &mut HeadScratch, mut pick: impl FnMut(usize, &[f32], f32) -> u32, out: &mut [u32]) -> f32 {
        let h = self.h;
        let mut logp = 0.0;
        for q in 0..self.f() {
            let v = self.radix[q];
            s.hq.copy_from_slice(x);
            for g in 0..q {
                let c = out[g] as usize;
                for (a, &e) in s.hq.iter_mut().zip(&self.cond[g].w[c * h..(c + 1) * h]) {
                    *a += e;
                }
            }
            tanh_inplace(&mut s.hq);
            let p = self.prefix(out, q);
            let logits = &mut s.logits[..v];
            logits.copy_from_slice(&self.b2[q].w[p * v..(p + 1) * v]);
            gemv_rows(&s.hq, &self.w2[q].w[p * h * v..(p + 1) * h * v], v, logits);
            let lse = logsumexp(logits);
            out[q] = pick(q, logits, lse);
            logp += logits[out[q] as usize] - lse;
        }
        logp
    }
}

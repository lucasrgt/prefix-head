use crate::fast::gemv_rows;
use crate::head::{Head, HeadBuf, HeadScratch};
use crate::train::{Param, gemm};
use kern::Rng;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Stacked minGRU (Feng et al. 2024) with residuals and a coarse-to-fine head.
/// Layer l: z = σ(u·Wz + bz), c = u·Wc + bc, h_t = (1 − z) ⊙ h_{t−1} + z ⊙ c, out = u + tanh(h).
/// Gates read only the input, so the whole state is `layers × h` floats and training is two
/// GEMMs per layer plus an elementwise scan over time.
pub struct MinGru {
    pub vocab: usize,
    pub h: usize,
    pub layers: usize,
    emb: Param,
    wz: Vec<Param>,
    bz: Vec<Param>,
    wc: Vec<Param>,
    bc: Vec<Param>,
    pub head: Head,
}

pub struct Buf {
    n: usize,
    u: Vec<Vec<f32>>,
    z: Vec<Vec<f32>>,
    c: Vec<Vec<f32>>,
    hs: Vec<Vec<f32>>,
    out: Vec<f32>,
    du: Vec<f32>,
    dz: Vec<f32>,
    dc: Vec<f32>,
    tmp: Vec<f32>,
    head: HeadBuf,
}

/// Per-stream inference state: one h vector per layer. This is the whole "thought".
#[derive(Clone)]
pub struct State(pub Vec<Vec<f32>>);

pub struct Scratch {
    u: Vec<f32>,
    z: Vec<f32>,
    c: Vec<f32>,
    head: HeadScratch,
}

impl MinGru {
    /// `vocab` input ids plus one start id (`vocab`) used before the first token.
    pub fn new(vocab: usize, radix: Vec<usize>, h: usize, layers: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let s = 1.0 / (h as f32).sqrt();
        let emb = Param::new((vocab + 1) * h, 1.0, &mut rng);
        let wz = (0..layers).map(|_| Param::new(h * h, s, &mut rng)).collect();
        let bz = (0..layers).map(|_| Param::new(h, 0.0, &mut rng)).collect();
        let wc = (0..layers).map(|_| Param::new(h * h, s, &mut rng)).collect();
        let bc = (0..layers).map(|_| Param::new(h, 0.0, &mut rng)).collect();
        let head = Head::new(radix, h, &mut rng);
        Self { vocab, h, layers, emb, wz, bz, wc, bc, head }
    }

    pub fn params(&self) -> usize {
        let core: usize = self.wz.iter().chain(&self.bz).chain(&self.wc).chain(&self.bc).map(|p| p.w.len()).sum();
        self.emb.w.len() + core + self.head.params()
    }

    pub fn macs_per_token(&self) -> usize {
        self.layers * 2 * self.h * self.h + self.head.macs()
    }

    pub fn state_bytes(&self) -> usize {
        self.layers * self.h * 4
    }

    pub fn buf(&self, n: usize) -> Buf {
        let v = |k: usize| (0..k).map(|_| vec![0.0; n * self.h]).collect::<Vec<_>>();
        Buf {
            n,
            u: v(self.layers),
            z: v(self.layers),
            c: v(self.layers),
            hs: v(self.layers),
            out: vec![0.0; n * self.h],
            du: vec![0.0; n * self.h],
            dz: vec![0.0; n * self.h],
            dc: vec![0.0; n * self.h],
            tmp: vec![0.0; n * self.h],
            head: self.head.buf(n),
        }
    }

    /// `x` is b×t input ids (row r = stream·t + step), `tgt` is b×t×F target factors; every
    /// stream starts from a zero state. Returns Σ −ln p.
    pub fn forward(&self, x: &[u32], tgt: &[u32], b: usize, t: usize, bf: &mut Buf) -> f64 {
        let (h, n) = (self.h, b * t);
        debug_assert_eq!(bf.n, n);
        for (r, &id) in x.iter().enumerate() {
            bf.u[0][r * h..(r + 1) * h].copy_from_slice(&self.emb.w[id as usize * h..(id as usize + 1) * h]);
        }
        for l in 0..self.layers {
            for r in 0..n {
                bf.z[l][r * h..(r + 1) * h].copy_from_slice(&self.bz[l].w);
                bf.c[l][r * h..(r + 1) * h].copy_from_slice(&self.bc[l].w);
            }
            gemm(n, h, h, &bf.u[l], false, &self.wz[l].w, false, 1.0, &mut bf.z[l]);
            gemm(n, h, h, &bf.u[l], false, &self.wc[l].w, false, 1.0, &mut bf.c[l]);
            for zz in bf.z[l].iter_mut() {
                *zz = sigmoid(*zz);
            }
            for s in 0..b {
                for j in 0..h {
                    let mut prev = 0.0f32;
                    for k in 0..t {
                        let i = (s * t + k) * h + j;
                        let z = bf.z[l][i];
                        prev = (1.0 - z) * prev + z * bf.c[l][i];
                        bf.hs[l][i] = prev;
                    }
                }
            }
            if l + 1 < self.layers {
                let (lo, hi) = bf.u.split_at_mut(l + 1);
                for i in 0..n * h {
                    hi[0][i] = lo[l][i] + bf.hs[l][i].tanh();
                }
            } else {
                for i in 0..n * h {
                    bf.out[i] = bf.u[l][i] + bf.hs[l][i].tanh();
                }
            }
        }
        self.head.forward(&bf.out, tgt, n, &mut bf.head)
    }

    /// Gradients of `scale`·loss after `forward` on the same batch.
    pub fn backward(&mut self, x: &[u32], tgt: &[u32], b: usize, t: usize, scale: f32, bf: &mut Buf) {
        let (h, n) = (self.h, b * t);
        self.head.backward(tgt, n, scale, &mut bf.head, &mut bf.du);
        for l in (0..self.layers).rev() {
            for i in 0..n * h {
                let th = bf.hs[l][i].tanh();
                bf.tmp[i] = bf.du[i] * (1.0 - th * th);
            }
            for s in 0..b {
                for j in 0..h {
                    let mut g = 0.0f32;
                    for k in (0..t).rev() {
                        let i = (s * t + k) * h + j;
                        g += bf.tmp[i];
                        let z = bf.z[l][i];
                        let prev = if k > 0 { bf.hs[l][i - h] } else { 0.0 };
                        bf.dc[i] = g * z;
                        bf.dz[i] = g * (bf.c[l][i] - prev) * z * (1.0 - z);
                        g *= 1.0 - z;
                    }
                }
            }
            gemm(h, n, h, &bf.u[l], true, &bf.dz, false, 1.0, &mut self.wz[l].g);
            gemm(h, n, h, &bf.u[l], true, &bf.dc, false, 1.0, &mut self.wc[l].g);
            for r in 0..n {
                for j in 0..h {
                    self.bz[l].g[j] += bf.dz[r * h + j];
                    self.bc[l].g[j] += bf.dc[r * h + j];
                }
            }
            gemm(n, h, h, &bf.dz, false, &self.wz[l].w, true, 1.0, &mut bf.du);
            gemm(n, h, h, &bf.dc, false, &self.wc[l].w, true, 1.0, &mut bf.du);
        }
        for (r, &id) in x.iter().enumerate() {
            let row = &mut self.emb.g[id as usize * h..(id as usize + 1) * h];
            for (g, &d) in row.iter_mut().zip(&bf.du[r * h..(r + 1) * h]) {
                *g += d;
            }
        }
    }

    fn params_mut(&mut self) -> impl Iterator<Item = &mut Param> {
        let core = self.wz.iter_mut().chain(self.bz.iter_mut()).chain(self.wc.iter_mut()).chain(self.bc.iter_mut());
        std::iter::once(&mut self.emb).chain(core).chain(self.head.params_mut())
    }

    /// Clips the global gradient norm to `clip`, then takes an Adam step; returns the norm.
    pub fn step(&mut self, lr: f32, t: i32, clip: f32) -> f32 {
        let norm = self.params_mut().map(|p| p.g.iter().map(|g| g * g).sum::<f32>()).sum::<f32>().sqrt();
        if norm > clip {
            let k = clip / norm;
            for p in self.params_mut() {
                p.g.iter_mut().for_each(|g| *g *= k);
            }
        }
        for p in self.params_mut() {
            p.adam(lr, t);
        }
        norm
    }

    pub fn state(&self) -> State {
        State(vec![vec![0.0; self.h]; self.layers])
    }

    pub fn scratch(&self) -> Scratch {
        Scratch { u: vec![0.0; self.h], z: vec![0.0; self.h], c: vec![0.0; self.h], head: self.head.scratch() }
    }

    /// Reads one input id, advances the state, and predicts the next token's factors.
    pub fn infer_step(&self, id: u32, st: &mut State, s: &mut Scratch, pick: impl FnMut(usize, &[f32], f32) -> u32, out: &mut [u32]) -> f32 {
        let h = self.h;
        s.u.copy_from_slice(&self.emb.w[id as usize * h..(id as usize + 1) * h]);
        for l in 0..self.layers {
            s.z.copy_from_slice(&self.bz[l].w);
            s.c.copy_from_slice(&self.bc[l].w);
            gemv_rows(&s.u, &self.wz[l].w, h, &mut s.z);
            gemv_rows(&s.u, &self.wc[l].w, h, &mut s.c);
            for j in 0..h {
                let z = sigmoid(s.z[j]);
                st.0[l][j] = (1.0 - z) * st.0[l][j] + z * s.c[j];
                s.u[j] += st.0[l][j].tanh();
            }
        }
        self.head.infer(&s.u, &mut s.head, pick, out)
    }

    #[cfg(test)]
    fn all_params(&mut self) -> Vec<&mut Param> {
        self.params_mut().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch(rng: &mut Rng, vocab: usize, radix: &[usize], b: usize, t: usize) -> (Vec<u32>, Vec<u32>) {
        let x = (0..b * t).map(|_| (rng.next() % (vocab as u64 + 1)) as u32).collect();
        let y = (0..b * t * radix.len()).map(|i| (rng.next() % radix[i % radix.len()] as u64) as u32).collect();
        (x, y)
    }

    #[test]
    fn gradients_match_finite_differences() {
        let radix = vec![3, 4];
        let (vocab, h, layers, b, t) = (12, 6, 2, 2, 5);
        let mut m = MinGru::new(vocab, radix.clone(), h, layers, 11);
        let mut rng = Rng::new(5);
        let (x, y) = batch(&mut rng, vocab, &radix, b, t);
        let mut bf = m.buf(b * t);
        m.forward(&x, &y, b, t, &mut bf);
        m.backward(&x, &y, b, t, 1.0, &mut bf);
        let grads: Vec<Vec<f32>> = m.all_params().iter().map(|p| p.g.clone()).collect();
        let eps = 1e-2f32;
        let mut checked = 0;
        for pi in 0..grads.len() {
            let len = grads[pi].len();
            for k in (0..len).step_by((len / 7).max(1)) {
                let orig = m.all_params()[pi].w[k];
                m.all_params()[pi].w[k] = orig + eps;
                let lp = m.forward(&x, &y, b, t, &mut bf);
                m.all_params()[pi].w[k] = orig - eps;
                let lm = m.forward(&x, &y, b, t, &mut bf);
                m.all_params()[pi].w[k] = orig;
                let num = ((lp - lm) / (2.0 * eps as f64)) as f32;
                let ana = grads[pi][k];
                assert!((num - ana).abs() <= 2e-3 + 2e-2 * ana.abs().max(num.abs()), "param {pi}[{k}]: numeric {num} vs analytic {ana}");
                checked += 1;
            }
        }
        assert!(checked > 50);
    }

    #[test]
    fn streaming_inference_matches_batched_forward() {
        let radix = vec![3, 4];
        let (vocab, h, layers, t) = (12, 8, 2, 9);
        let m = MinGru::new(vocab, radix.clone(), h, layers, 3);
        let mut rng = Rng::new(8);
        let (x, y) = batch(&mut rng, vocab, &radix, 1, t);
        let mut bf = m.buf(t);
        let want = m.forward(&x, &y, 1, t, &mut bf);
        let (mut st, mut s) = (m.state(), m.scratch());
        let mut out = vec![0u32; 2];
        let mut got = 0.0f64;
        for k in 0..t {
            got -= m.infer_step(x[k], &mut st, &mut s, |q, _, _| y[k * 2 + q], &mut out) as f64;
        }
        assert!((got - want).abs() < 1e-3 * want.abs(), "{got} vs {want}");
    }
}

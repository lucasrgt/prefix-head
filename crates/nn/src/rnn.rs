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
    /// Starting state per layer (streams × h); zero unless the caller carries state between windows.
    pub h0: Vec<Vec<f32>>,
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

    /// Chrono initialisation (Tallec & Ollivier 2018): channel j of every layer starts with
    /// z ≈ 1/T_j, T_j log-uniform in [2, t_max], so the state starts with a spread of memory
    /// half-lives instead of forgetting half of itself every token.
    pub fn chrono(&mut self, t_max: f32, seed: u64) {
        let mut rng = Rng::new(seed ^ 0xc4);
        for b in self.bz.iter_mut() {
            for w in b.w.iter_mut() {
                let u = (rng.next() >> 40) as f32 / (1u64 << 24) as f32;
                let t = 2.0f32 * (t_max / 2.0).powf(u);
                *w = -(t - 1.0).ln();
            }
        }
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
            h0: Vec::new(),
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

    /// `x` is b×t input ids (row r = stream·t + step), `tgt` is b×t×F target factors; streams start
    /// from `bf.h0` (zero when empty). Returns Σ −ln p.
    pub fn forward(&self, x: &[u32], tgt: &[u32], b: usize, t: usize, bf: &mut Buf) -> f64 {
        let (h, n) = (self.h, b * t);
        debug_assert_eq!(bf.n, n);
        if bf.h0.len() != self.layers || bf.h0[0].len() != b * h {
            bf.h0 = vec![vec![0.0; b * h]; self.layers];
        }
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
                    let mut prev = bf.h0[l][s * h + j];
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

    /// After `forward`, makes each stream's final state the next window's starting state
    /// (truncated BPTT: no gradient flows across windows).
    pub fn row_losses<'b>(&self, bf: &'b Buf) -> &'b [f32] {
        &bf.head.row_loss
    }

    pub fn carry(&self, b: usize, t: usize, bf: &mut Buf) {
        let h = self.h;
        for l in 0..self.layers {
            for s in 0..b {
                let last = (s * t + t - 1) * h;
                let (src, dst) = (&bf.hs[l][last..last + h], &mut bf.h0[l][s * h..(s + 1) * h]);
                dst.copy_from_slice(src);
            }
        }
    }

    pub fn reset_stream(&self, s: usize, bf: &mut Buf) {
        for l in bf.h0.iter_mut() {
            l[s * self.h..(s + 1) * self.h].fill(0.0);
        }
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
                        let prev = if k > 0 { bf.hs[l][i - h] } else { bf.h0[l][s * h + j] };
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

    /// Writes config line `meta` and every weight (same order as training) as little-endian f32.
    pub fn save(&mut self, path: &std::path::Path, meta: &str) -> std::io::Result<()> {
        use std::io::Write;
        let header = format!("mingru {} {} {} {}\n{meta}\n", self.vocab, self.h, self.layers, self.head.radix.iter().map(|r| r.to_string()).collect::<Vec<_>>().join("x"));
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(header.as_bytes())?;
        for p in self.params_mut() {
            for w in &p.w {
                f.write_all(&w.to_le_bytes())?;
            }
        }
        Ok(())
    }

    /// Reads a file from `save`; returns the model and its meta line.
    pub fn load(path: &std::path::Path) -> std::io::Result<(Self, String)> {
        let bytes = std::fs::read(path)?;
        let mut lines = bytes.splitn(3, |&b| b == b'\n');
        let head = String::from_utf8_lossy(lines.next().unwrap()).into_owned();
        let meta = String::from_utf8_lossy(lines.next().unwrap()).into_owned();
        let body = lines.next().unwrap();
        let f: Vec<&str> = head.split(' ').collect();
        assert_eq!(f[0], "mingru", "not a minGRU checkpoint");
        let radix = f[4].split('x').map(|r| r.parse().unwrap()).collect();
        let mut m = Self::new(f[1].parse().unwrap(), radix, f[2].parse().unwrap(), f[3].parse().unwrap(), 0);
        let mut it = body.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()));
        for p in m.params_mut() {
            for w in p.w.iter_mut() {
                *w = it.next().expect("checkpoint too short");
            }
        }
        assert!(it.next().is_none(), "checkpoint too long");
        Ok((m, meta))
    }

    pub fn state(&self) -> State {
        State(vec![vec![0.0; self.h]; self.layers])
    }

    pub fn scratch(&self) -> Scratch {
        Scratch { u: vec![0.0; self.h], z: vec![0.0; self.h], c: vec![0.0; self.h], head: self.head.scratch() }
    }

    /// Reads one input id into the state without predicting: reading costs no head.
    pub fn advance(&self, id: u32, st: &mut State, s: &mut Scratch) {
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
        bf.h0 = (0..layers).map(|_| (0..b * h).map(|_| rng.f32()).collect()).collect();
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
    fn save_and_load_roundtrip() {
        let mut m = MinGru::new(12, vec![3, 4], 8, 2, 4);
        let path = std::env::temp_dir().join(format!("mingru-{}.bin", std::process::id()));
        m.save(&path, "meta line").unwrap();
        let (mut back, meta) = MinGru::load(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(meta, "meta line");
        let a: Vec<f32> = m.params_mut().flat_map(|p| p.w.clone()).collect();
        let b: Vec<f32> = back.params_mut().flat_map(|p| p.w.clone()).collect();
        assert_eq!(a, b);
    }

    #[test]
    fn streaming_inference_matches_windows_with_carried_state() {
        let radix = vec![3, 4];
        let (vocab, h, layers, t) = (12, 8, 2, 9);
        let m = MinGru::new(vocab, radix.clone(), h, layers, 3);
        let mut rng = Rng::new(8);
        let (x, y) = batch(&mut rng, vocab, &radix, 1, t);
        let mut carried = Vec::new();
        let mut want = 0.0f64;
        for (lo, hi) in [(0, t / 2), (t / 2, t)] {
            let mut bf = m.buf(hi - lo);
            bf.h0 = carried;
            want += m.forward(&x[lo..hi], &y[lo * 2..hi * 2], 1, hi - lo, &mut bf);
            m.carry(1, hi - lo, &mut bf);
            carried = bf.h0;
        }
        let (mut st, mut s) = (m.state(), m.scratch());
        let mut out = vec![0u32; 2];
        let mut got = 0.0f64;
        for k in 0..t {
            got -= m.infer_step(x[k], &mut st, &mut s, |q, _, _| y[k * 2 + q], &mut out) as f64;
        }
        assert!((got - want).abs() < 1e-3 * want.abs(), "{got} vs {want}");
        assert_eq!(st.0, carried);
    }
}

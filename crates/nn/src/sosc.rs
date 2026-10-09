use crate::fast::gemv_rows;
use crate::head::{Head, HeadBuf, HeadScratch};
use crate::rnn::State;
use crate::train::{Param, gemm};
use kern::Rng;

/// Selective oscillators: the minGRU gate decides how much to keep, the kept part rotates by a learned
/// per-channel phase. Layer l, channel j:
/// z_t = σ(u_t·Wz + bz), c_t = u_t·B (complex), h_t = (1 − z_t)·e^{iθ_j}·h_{t−1} + z_t·c_t,
/// y_t = Re(h_t·C) + D ⊙ u_t, out = u_t + tanh(y_t). The state is `layers × n` complex numbers.
pub struct SelOsc {
    pub vocab: usize,
    pub h: usize,
    pub n: usize,
    pub layers: usize,
    emb: Param,
    bre: Vec<Param>,
    bim: Vec<Param>,
    cre: Vec<Param>,
    cim: Vec<Param>,
    d: Vec<Param>,
    wz: Vec<Param>,
    bz: Vec<Param>,
    theta: Vec<Param>,
    rotate: bool,
    slow: usize,
    period: usize,
    pub head: Head,
}

pub struct Buf {
    pub h0: Vec<Vec<f32>>,
    u: Vec<Vec<f32>>,
    br: Vec<Vec<f32>>,
    bi: Vec<Vec<f32>>,
    hr: Vec<Vec<f32>>,
    hi: Vec<Vec<f32>>,
    z: Vec<Vec<f32>>,
    th: Vec<Vec<f32>>,
    out: Vec<f32>,
    du: Vec<f32>,
    dy: Vec<f32>,
    dhr: Vec<f32>,
    dhi: Vec<f32>,
    dz: Vec<f32>,
    tmp_h: Vec<f32>,
    tmp_nh: Vec<f32>,
    head: HeadBuf,
}

pub struct Scratch {
    u: Vec<f32>,
    z: Vec<f32>,
    br: Vec<f32>,
    bi: Vec<f32>,
    neg_hi: Vec<f32>,
    y: Vec<f32>,
    pos: usize,
    head: HeadScratch,
}


impl SelOsc {
    pub fn new(vocab: usize, radix: Vec<usize>, h: usize, n: usize, layers: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let emb = Param::new((vocab + 1) * h, 1.0, &mut rng);
        let mk = |k: usize, s: f32, rng: &mut Rng| (0..layers).map(|_| Param::new(k, s, rng)).collect::<Vec<_>>();
        let bre = mk(h * n, 1.0 / (2.0 * h as f32).sqrt(), &mut rng);
        let bim = mk(h * n, 1.0 / (2.0 * h as f32).sqrt(), &mut rng);
        let cre = mk(n * h, 1.0 / (n as f32).sqrt(), &mut rng);
        let cim = mk(n * h, 1.0 / (n as f32).sqrt(), &mut rng);
        let d = mk(h, 0.0, &mut rng);
        let wz = mk(h * n, 1.0 / (h as f32).sqrt(), &mut rng);
        let bz = mk(n, 0.0, &mut rng);
        let theta = mk(n, 0.0, &mut rng);
        let head = Head::new(radix, h, &mut rng);
        let mut m = Self { vocab, h, n, layers, emb, bre, bim, cre, cim, d, wz, bz, theta, rotate: true, slow: 0, period: 1, head };
        m.chrono(1024.0, seed);
        m
    }

    /// Gate biases give z ≈ 1/T with T log-uniform in [2, t_max] (chrono); periods log-uniform too.
    pub fn chrono(&mut self, t_max: f32, seed: u64) {
        let mut rng = Rng::new(seed ^ 0x05c);
        let lu = |rng: &mut Rng| 2.0f32 * (t_max / 2.0).powf((rng.next() >> 40) as f32 / (1u64 << 24) as f32);
        for l in 0..self.layers {
            for j in 0..self.n {
                self.bz[l].w[j] = -(lu(&mut rng) - 1.0).ln();
                self.theta[l].w[j] = std::f32::consts::TAU / lu(&mut rng);
            }
        }
    }

    /// θ = 0 and frozen: the re/im halves become 2n independent real channels in the same state bytes.
    pub fn without_rotation(&mut self) {
        self.rotate = false;
        for t in &mut self.theta {
            t.w.iter_mut().for_each(|w| *w = 0.0);
        }
    }

    /// The last `slow` channels of every layer update only on positions divisible by `period` and hold
    /// their state in between (Clockwork-style); their read-out is then constant and can be cached.
    pub fn two_speed(&mut self, slow: usize, period: usize) {
        assert!(slow <= self.n && period >= 1);
        self.slow = slow;
        self.period = period;
    }

    fn held(&self, j: usize, k: usize) -> bool {
        j >= self.n - self.slow && k % self.period != 0
    }

    /// Input embedding table, (vocab + 1) × h.
    pub fn embedding(&self) -> &[f32] {
        &self.emb.w
    }

    pub fn params(&self) -> usize {
        let layers: usize = [&self.bre, &self.bim, &self.cre, &self.cim, &self.d, &self.wz, &self.bz, &self.theta].iter().flat_map(|v| v.iter()).map(|p| p.w.len()).sum();
        self.emb.w.len() + layers + self.head.params()
    }

    pub fn macs_per_token(&self) -> usize {
        self.layers * 5 * self.h * ((self.n - self.slow) * self.period + self.slow) / self.period + self.head.macs()
    }

    pub fn state_bytes(&self) -> usize {
        self.layers * 2 * self.n * 4
    }

    pub fn buf(&self, rows: usize) -> Buf {
        let (h, n, ls) = (self.h, self.n, self.layers);
        let v = |k: usize| (0..ls).map(|_| vec![0.0; rows * k]).collect::<Vec<_>>();
        Buf {
            h0: Vec::new(),
            u: v(h),
            br: v(n),
            bi: v(n),
            hr: v(n),
            hi: v(n),
            z: v(n),
            th: v(h),
            out: vec![0.0; rows * h],
            du: vec![0.0; rows * h],
            dy: vec![0.0; rows * h],
            dhr: vec![0.0; rows * n],
            dhi: vec![0.0; rows * n],
            dz: vec![0.0; rows * n],
            tmp_h: vec![0.0; rows * h],
            tmp_nh: vec![0.0; n * h],
            head: self.head.buf(rows),
        }
    }

    pub fn forward(&self, x: &[u32], tgt: &[u32], b: usize, t: usize, bf: &mut Buf) -> f64 {
        let (h, n, rows) = (self.h, self.n, b * t);
        if bf.h0.len() != self.layers || bf.h0[0].len() != b * 2 * n {
            bf.h0 = vec![vec![0.0; b * 2 * n]; self.layers];
        }
        for (r, &id) in x.iter().enumerate() {
            bf.u[0][r * h..(r + 1) * h].copy_from_slice(&self.emb.w[id as usize * h..(id as usize + 1) * h]);
        }
        for l in 0..self.layers {
            gemm(rows, h, n, &bf.u[l], false, &self.bre[l].w, false, 0.0, &mut bf.br[l]);
            gemm(rows, h, n, &bf.u[l], false, &self.bim[l].w, false, 0.0, &mut bf.bi[l]);
            for r in 0..rows {
                bf.z[l][r * n..(r + 1) * n].copy_from_slice(&self.bz[l].w);
            }
            gemm(rows, h, n, &bf.u[l], false, &self.wz[l].w, false, 1.0, &mut bf.z[l]);
            for v in bf.z[l].iter_mut() {
                *v = 1.0 / (1.0 + (-*v).exp());
            }
            for j in 0..n {
                let (ct, st) = (self.theta[l].w[j].cos(), self.theta[l].w[j].sin());
                for s in 0..b {
                    let (mut pr, mut pi) = (bf.h0[l][s * 2 * n + j], bf.h0[l][s * 2 * n + n + j]);
                    for k in 0..t {
                        let i = (s * t + k) * n + j;
                        if self.held(j, k) {
                            bf.hr[l][i] = pr;
                            bf.hi[l][i] = pi;
                            continue;
                        }
                        let (z, a) = (bf.z[l][i], 1.0 - bf.z[l][i]);
                        let hr = a * (ct * pr - st * pi) + z * bf.br[l][i];
                        let hi = a * (st * pr + ct * pi) + z * bf.bi[l][i];
                        bf.hr[l][i] = hr;
                        bf.hi[l][i] = hi;
                        (pr, pi) = (hr, hi);
                    }
                }
            }
            gemm(rows, n, h, &bf.hr[l], false, &self.cre[l].w, false, 0.0, &mut bf.th[l]);
            gemm(rows, n, h, &bf.hi[l], false, &self.cim[l].w, false, 0.0, &mut bf.tmp_h);
            for r in 0..rows {
                for c in 0..h {
                    let i = r * h + c;
                    bf.th[l][i] = (bf.th[l][i] - bf.tmp_h[i] + self.d[l].w[c] * bf.u[l][i]).tanh();
                }
            }
            if l + 1 < self.layers {
                let (lo, hi) = bf.u.split_at_mut(l + 1);
                for i in 0..rows * h {
                    hi[0][i] = lo[l][i] + bf.th[l][i];
                }
            } else {
                for i in 0..rows * h {
                    bf.out[i] = bf.u[l][i] + bf.th[l][i];
                }
            }
        }
        self.head.forward(&bf.out, tgt, rows, &mut bf.head)
    }

    pub fn row_losses<'b>(&self, bf: &'b Buf) -> &'b [f32] {
        &bf.head.row_loss
    }

    pub fn carry(&self, b: usize, t: usize, bf: &mut Buf) {
        let n = self.n;
        for l in 0..self.layers {
            for s in 0..b {
                let last = (s * t + t - 1) * n;
                bf.h0[l][s * 2 * n..s * 2 * n + n].copy_from_slice(&bf.hr[l][last..last + n]);
                bf.h0[l][s * 2 * n + n..(s + 1) * 2 * n].copy_from_slice(&bf.hi[l][last..last + n]);
            }
        }
    }

    pub fn reset_stream(&self, s: usize, bf: &mut Buf) {
        let w = 2 * self.n;
        for l in bf.h0.iter_mut() {
            l[s * w..(s + 1) * w].fill(0.0);
        }
    }

    pub fn backward(&mut self, x: &[u32], tgt: &[u32], b: usize, t: usize, scale: f32, bf: &mut Buf) {
        let (h, n, rows) = (self.h, self.n, b * t);
        self.head.backward(tgt, rows, scale, &mut bf.head, &mut bf.du);
        for l in (0..self.layers).rev() {
            for r in 0..rows {
                for c in 0..h {
                    let i = r * h + c;
                    let th = bf.th[l][i];
                    let dy = bf.du[i] * (1.0 - th * th);
                    bf.dy[i] = dy;
                    self.d[l].g[c] += dy * bf.u[l][i];
                    bf.du[i] += dy * self.d[l].w[c];
                }
            }
            gemm(n, rows, h, &bf.hr[l], true, &bf.dy, false, 1.0, &mut self.cre[l].g);
            gemm(n, rows, h, &bf.hi[l], true, &bf.dy, false, 0.0, &mut bf.tmp_nh);
            for (g, &v) in self.cim[l].g.iter_mut().zip(&bf.tmp_nh) {
                *g -= v;
            }
            gemm(rows, h, n, &bf.dy, false, &self.cre[l].w, true, 0.0, &mut bf.dhr);
            gemm(rows, h, n, &bf.dy, false, &self.cim[l].w, true, 0.0, &mut bf.dhi);
            for v in bf.dhi.iter_mut() {
                *v = -*v;
            }
            for j in 0..n {
                let (ct, st) = (self.theta[l].w[j].cos(), self.theta[l].w[j].sin());
                let mut gth = 0.0f32;
                for s in 0..b {
                    let (mut gr, mut gi) = (0.0f32, 0.0f32);
                    for k in (0..t).rev() {
                        let i = (s * t + k) * n + j;
                        gr += bf.dhr[i];
                        gi += bf.dhi[i];
                        if self.held(j, k) {
                            bf.dz[i] = 0.0;
                            bf.dhr[i] = 0.0;
                            bf.dhi[i] = 0.0;
                            continue;
                        }
                        let (pr, pi) = if k > 0 { (bf.hr[l][i - n], bf.hi[l][i - n]) } else { (bf.h0[l][s * 2 * n + j], bf.h0[l][s * 2 * n + n + j]) };
                        let (rpr, rpi) = (ct * pr - st * pi, st * pr + ct * pi);
                        let (z, a) = (bf.z[l][i], 1.0 - bf.z[l][i]);
                        let dz = gr * (bf.br[l][i] - rpr) + gi * (bf.bi[l][i] - rpi);
                        bf.dz[i] = dz * z * a;
                        let (dr, di) = (a * gr, a * gi);
                        gth += -dr * rpi + di * rpr;
                        bf.dhr[i] = z * gr;
                        bf.dhi[i] = z * gi;
                        (gr, gi) = (ct * dr + st * di, -st * dr + ct * di);
                    }
                }
                if self.rotate {
                    self.theta[l].g[j] += gth;
                }
            }
            gemm(h, rows, n, &bf.u[l], true, &bf.dz, false, 1.0, &mut self.wz[l].g);
            for r in 0..rows {
                for j in 0..n {
                    self.bz[l].g[j] += bf.dz[r * n + j];
                }
            }
            gemm(rows, n, h, &bf.dz, false, &self.wz[l].w, true, 1.0, &mut bf.du);
            gemm(h, rows, n, &bf.u[l], true, &bf.dhr, false, 1.0, &mut self.bre[l].g);
            gemm(h, rows, n, &bf.u[l], true, &bf.dhi, false, 1.0, &mut self.bim[l].g);
            gemm(rows, n, h, &bf.dhr, false, &self.bre[l].w, true, 1.0, &mut bf.du);
            gemm(rows, n, h, &bf.dhi, false, &self.bim[l].w, true, 1.0, &mut bf.du);
        }
        for (r, &id) in x.iter().enumerate() {
            let row = &mut self.emb.g[id as usize * h..(id as usize + 1) * h];
            for (g, &dv) in row.iter_mut().zip(&bf.du[r * h..(r + 1) * h]) {
                *g += dv;
            }
        }
    }

    fn params_mut(&mut self) -> impl Iterator<Item = &mut Param> {
        let layers = self.bre.iter_mut().chain(self.bim.iter_mut()).chain(self.cre.iter_mut()).chain(self.cim.iter_mut()).chain(self.d.iter_mut()).chain(self.wz.iter_mut()).chain(self.bz.iter_mut()).chain(self.theta.iter_mut());
        std::iter::once(&mut self.emb).chain(layers).chain(self.head.params_mut())
    }

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

    pub fn save(&mut self, path: &std::path::Path, meta: &str) -> std::io::Result<()> {
        use std::io::Write;
        let radix = self.head.radix.iter().map(|r| r.to_string()).collect::<Vec<_>>().join("x");
        let header = format!("sosc {} {} {} {} {radix}\n{meta}\n", self.vocab, self.h, self.n, self.layers);
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(header.as_bytes())?;
        for p in self.params_mut() {
            for w in &p.w {
                f.write_all(&w.to_le_bytes())?;
            }
        }
        Ok(())
    }

    pub fn load(path: &std::path::Path) -> std::io::Result<(Self, String)> {
        let bytes = std::fs::read(path)?;
        let mut lines = bytes.splitn(3, |&b| b == b'\n');
        let head = String::from_utf8_lossy(lines.next().unwrap()).into_owned();
        let meta = String::from_utf8_lossy(lines.next().unwrap()).into_owned();
        let f: Vec<&str> = head.split(' ').collect();
        assert_eq!(f[0], "sosc", "not a selective-oscillator checkpoint");
        let radix = f[5].split('x').map(|r| r.parse().unwrap()).collect();
        let mut m = Self::new(f[1].parse().unwrap(), radix, f[2].parse().unwrap(), f[3].parse().unwrap(), f[4].parse().unwrap(), 0);
        let mut it = lines.next().unwrap().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()));
        for p in m.params_mut() {
            for w in p.w.iter_mut() {
                *w = it.next().expect("checkpoint too short");
            }
        }
        assert!(it.next().is_none(), "checkpoint too long");
        Ok((m, meta))
    }

    pub fn state(&self) -> State {
        State(vec![vec![0.0; 2 * self.n]; self.layers])
    }

    pub fn scratch(&self) -> Scratch {
        let (h, n) = (self.h, self.n);
        Scratch { u: vec![0.0; h], z: vec![0.0; n], br: vec![0.0; n], bi: vec![0.0; n], neg_hi: vec![0.0; n], y: vec![0.0; h], pos: 0, head: self.head.scratch() }
    }

    pub fn advance(&self, id: u32, st: &mut State, s: &mut Scratch) {
        let (h, n) = (self.h, self.n);
        s.u.copy_from_slice(&self.emb.w[id as usize * h..(id as usize + 1) * h]);
        for l in 0..self.layers {
            s.br.fill(0.0);
            s.bi.fill(0.0);
            s.z.copy_from_slice(&self.bz[l].w);
            gemv_rows(&s.u, &self.bre[l].w, n, &mut s.br);
            gemv_rows(&s.u, &self.bim[l].w, n, &mut s.bi);
            gemv_rows(&s.u, &self.wz[l].w, n, &mut s.z);
            let (re, im) = st.0[l].split_at_mut(n);
            for j in 0..n {
                if self.held(j, s.pos) {
                    s.neg_hi[j] = -im[j];
                    continue;
                }
                let (ct, sn) = (self.theta[l].w[j].cos(), self.theta[l].w[j].sin());
                let z = 1.0 / (1.0 + (-s.z[j]).exp());
                let a = 1.0 - z;
                let hr = a * (ct * re[j] - sn * im[j]) + z * s.br[j];
                let hi = a * (sn * re[j] + ct * im[j]) + z * s.bi[j];
                re[j] = hr;
                im[j] = hi;
                s.neg_hi[j] = -hi;
            }
            for c in 0..h {
                s.y[c] = self.d[l].w[c] * s.u[c];
            }
            gemv_rows(re, &self.cre[l].w, h, &mut s.y);
            gemv_rows(&s.neg_hi, &self.cim[l].w, h, &mut s.y);
            for c in 0..h {
                s.u[c] += s.y[c].tanh();
            }
        }
        s.pos += 1;
    }

    pub fn infer_step(&self, id: u32, st: &mut State, s: &mut Scratch, pick: impl FnMut(usize, &[f32], f32) -> u32, out: &mut [u32]) -> f32 {
        self.advance(id, st, s);
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
        check_gradients(0, 1);
    }

    #[test]
    fn two_speed_gradients_match_finite_differences() {
        check_gradients(2, 2);
    }

    fn check_gradients(slow: usize, period: usize) {
        let radix = vec![3, 4];
        let (vocab, h, n, layers, b, t) = (12, 6, 4, 2, 2, 5);
        let mut m = SelOsc::new(vocab, radix.clone(), h, n, layers, 11);
        m.two_speed(slow, period);
        for d in m.d.iter_mut() {
            d.w.iter_mut().for_each(|w| *w = 0.3);
        }
        let mut rng = Rng::new(5);
        let (x, y) = batch(&mut rng, vocab, &radix, b, t);
        let mut bf = m.buf(b * t);
        bf.h0 = (0..layers).map(|_| (0..b * 2 * n).map(|_| rng.f32()).collect()).collect();
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
                assert!((num - ana).abs() <= 3e-3 + 3e-2 * ana.abs().max(num.abs()), "param {pi}[{k}]: numeric {num} vs analytic {ana}");
                checked += 1;
            }
        }
        assert!(checked > 60);
    }

    #[test]
    fn streaming_inference_matches_windows_with_carried_state() {
        let radix = vec![3, 4];
        let (vocab, h, n, layers, t) = (12, 8, 4, 2, 9);
        let m = SelOsc::new(vocab, radix.clone(), h, n, layers, 3);
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
        for (a, b) in st.0.iter().flatten().zip(carried.iter().flatten()) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}

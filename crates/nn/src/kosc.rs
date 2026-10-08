use crate::head::{Head, HeadBuf, HeadScratch};
use crate::monarch::{MonBuf, Monarch};
use crate::rnn::State;
use crate::train::Param;
use kern::Rng;

const K_IN: usize = 8;
const K_OUT: usize = 16;

/// Selective oscillators whose dense maps are replaced by kaleidoscope (Monarch) maps, so the
/// saved MACs buy a much larger state. Layer l: [z̃ | c_re | c_im] = In(u), z = σ(z̃),
/// h_t = (1 − z)·e^{iθ}·h_{t−1} + z·c, y = Out([h_re | h_im]) + D ⊙ u, out = u + tanh(y).
pub struct KOsc {
    pub vocab: usize,
    pub h: usize,
    pub n: usize,
    pub layers: usize,
    emb: Param,
    inm: Vec<Monarch>,
    outm: Vec<Monarch>,
    d: Vec<Param>,
    theta: Vec<Param>,
    pub head: Head,
}

pub struct Buf {
    pub h0: Vec<Vec<f32>>,
    u: Vec<Vec<f32>>,
    pre: Vec<Vec<f32>>,
    hcat: Vec<Vec<f32>>,
    th: Vec<Vec<f32>>,
    inb: Vec<MonBuf>,
    outb: Vec<MonBuf>,
    out: Vec<f32>,
    du: Vec<f32>,
    dy: Vec<f32>,
    dh: Vec<f32>,
    dpre: Vec<f32>,
    head: HeadBuf,
}

pub struct Scratch {
    u: Vec<f32>,
    pre: Vec<f32>,
    hcat: Vec<f32>,
    y: Vec<f32>,
    zin: Vec<f32>,
    zpin: Vec<f32>,
    zout: Vec<f32>,
    zpout: Vec<f32>,
    head: HeadScratch,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl KOsc {
    pub fn new(vocab: usize, radix: Vec<usize>, h: usize, n: usize, layers: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let emb = Param::new((vocab + 1) * h, 1.0, &mut rng);
        let inm = (0..layers).map(|_| Monarch::new(h, h, 3 * n, K_IN, &mut rng)).collect();
        let outm = (0..layers).map(|_| Monarch::new(2 * n, 2 * h, h, K_OUT, &mut rng)).collect();
        let d = (0..layers).map(|_| Param::new(h, 0.0, &mut rng)).collect();
        let theta = (0..layers).map(|_| Param::new(n, 0.0, &mut rng)).collect();
        let head = Head::new(radix, h, &mut rng);
        let mut m = Self { vocab, h, n, layers, emb, inm, outm, d, theta, head };
        m.chrono(1024.0, seed);
        m
    }

    /// Gate biases give z ≈ 1/T with T log-uniform in [2, t_max]; periods log-uniform too.
    pub fn chrono(&mut self, t_max: f32, seed: u64) {
        let mut rng = Rng::new(seed ^ 0x05c);
        let lu = |rng: &mut Rng| 2.0f32 * (t_max / 2.0).powf((rng.next() >> 40) as f32 / (1u64 << 24) as f32);
        for l in 0..self.layers {
            for j in 0..self.n {
                self.inm[l].bias_mut()[j] = -(lu(&mut rng) - 1.0).ln();
                self.theta[l].w[j] = std::f32::consts::TAU / lu(&mut rng);
            }
        }
    }

    pub fn params(&self) -> usize {
        let maps: usize = self.inm.iter().chain(&self.outm).map(|m| m.params()).sum();
        self.emb.w.len() + maps + self.d.iter().chain(&self.theta).map(|p| p.w.len()).sum::<usize>() + self.head.params()
    }

    pub fn macs_per_token(&self) -> usize {
        self.inm.iter().chain(&self.outm).map(|m| m.macs()).sum::<usize>() + self.head.macs()
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
            pre: v(3 * n),
            hcat: v(2 * n),
            th: v(h),
            inb: self.inm.iter().map(|m| m.buf(rows)).collect(),
            outb: self.outm.iter().map(|m| m.buf(rows)).collect(),
            out: vec![0.0; rows * h],
            du: vec![0.0; rows * h],
            dy: vec![0.0; rows * h],
            dh: vec![0.0; rows * 2 * n],
            dpre: vec![0.0; rows * 3 * n],
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
            self.inm[l].forward(&bf.u[l], rows, &mut bf.inb[l], &mut bf.pre[l]);
            for r in 0..rows {
                for j in 0..n {
                    bf.pre[l][r * 3 * n + j] = sigmoid(bf.pre[l][r * 3 * n + j]);
                }
            }
            for j in 0..n {
                let (ct, st) = (self.theta[l].w[j].cos(), self.theta[l].w[j].sin());
                for s in 0..b {
                    let (mut pr, mut pi) = (bf.h0[l][s * 2 * n + j], bf.h0[l][s * 2 * n + n + j]);
                    for k in 0..t {
                        let r = s * t + k;
                        let p = &bf.pre[l][r * 3 * n..(r + 1) * 3 * n];
                        let (z, a) = (p[j], 1.0 - p[j]);
                        let hr = a * (ct * pr - st * pi) + z * p[n + j];
                        let hi = a * (st * pr + ct * pi) + z * p[2 * n + j];
                        bf.hcat[l][r * 2 * n + j] = hr;
                        bf.hcat[l][r * 2 * n + n + j] = hi;
                        (pr, pi) = (hr, hi);
                    }
                }
            }
            self.outm[l].forward(&bf.hcat[l], rows, &mut bf.outb[l], &mut bf.th[l]);
            for r in 0..rows {
                for c in 0..h {
                    let i = r * h + c;
                    bf.th[l][i] = (bf.th[l][i] + self.d[l].w[c] * bf.u[l][i]).tanh();
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

    pub fn carry(&self, b: usize, t: usize, bf: &mut Buf) {
        let n2 = 2 * self.n;
        for l in 0..self.layers {
            for s in 0..b {
                let last = (s * t + t - 1) * n2;
                let src = bf.hcat[l][last..last + n2].to_vec();
                bf.h0[l][s * n2..(s + 1) * n2].copy_from_slice(&src);
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
            bf.dh.fill(0.0);
            self.outm[l].backward(&bf.hcat[l], rows, &mut bf.outb[l], &bf.dy, &mut bf.dh);
            for j in 0..n {
                let (ct, st) = (self.theta[l].w[j].cos(), self.theta[l].w[j].sin());
                let mut gth = 0.0f32;
                for s in 0..b {
                    let (mut gr, mut gi) = (0.0f32, 0.0f32);
                    for k in (0..t).rev() {
                        let r = s * t + k;
                        gr += bf.dh[r * 2 * n + j];
                        gi += bf.dh[r * 2 * n + n + j];
                        let (pr, pi) = if k > 0 {
                            (bf.hcat[l][(r - 1) * 2 * n + j], bf.hcat[l][(r - 1) * 2 * n + n + j])
                        } else {
                            (bf.h0[l][s * 2 * n + j], bf.h0[l][s * 2 * n + n + j])
                        };
                        let (rpr, rpi) = (ct * pr - st * pi, st * pr + ct * pi);
                        let p = &bf.pre[l][r * 3 * n..(r + 1) * 3 * n];
                        let (z, a) = (p[j], 1.0 - p[j]);
                        let dz = gr * (p[n + j] - rpr) + gi * (p[2 * n + j] - rpi);
                        bf.dpre[r * 3 * n + j] = dz * z * a;
                        bf.dpre[r * 3 * n + n + j] = z * gr;
                        bf.dpre[r * 3 * n + 2 * n + j] = z * gi;
                        let (dr, di) = (a * gr, a * gi);
                        gth += -dr * rpi + di * rpr;
                        (gr, gi) = (ct * dr + st * di, -st * dr + ct * di);
                    }
                }
                self.theta[l].g[j] += gth;
            }
            self.inm[l].backward(&bf.u[l], rows, &mut bf.inb[l], &bf.dpre, &mut bf.du);
        }
        for (r, &id) in x.iter().enumerate() {
            let row = &mut self.emb.g[id as usize * h..(id as usize + 1) * h];
            for (g, &dv) in row.iter_mut().zip(&bf.du[r * h..(r + 1) * h]) {
                *g += dv;
            }
        }
    }

    fn params_mut(&mut self) -> impl Iterator<Item = &mut Param> {
        let maps = self.inm.iter_mut().chain(self.outm.iter_mut()).flat_map(|m| m.params_mut());
        std::iter::once(&mut self.emb).chain(maps).chain(self.d.iter_mut()).chain(self.theta.iter_mut()).chain(self.head.params_mut())
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
        let header = format!("kosc {} {} {} {} {radix}\n{meta}\n", self.vocab, self.h, self.n, self.layers);
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
        assert_eq!(f[0], "kosc", "not a kaleidoscope-oscillator checkpoint");
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
        Scratch {
            u: vec![0.0; h],
            pre: vec![0.0; 3 * n],
            hcat: vec![0.0; 2 * n],
            y: vec![0.0; h],
            zin: vec![0.0; h],
            zpin: vec![0.0; h],
            zout: vec![0.0; 2 * h],
            zpout: vec![0.0; 2 * h],
            head: self.head.scratch(),
        }
    }

    pub fn advance(&self, id: u32, st: &mut State, s: &mut Scratch) {
        let (h, n) = (self.h, self.n);
        s.u.copy_from_slice(&self.emb.w[id as usize * h..(id as usize + 1) * h]);
        for l in 0..self.layers {
            self.inm[l].apply(&s.u, &mut s.zin, &mut s.zpin, &mut s.pre);
            let (re, im) = st.0[l].split_at_mut(n);
            for j in 0..n {
                let (ct, sn) = (self.theta[l].w[j].cos(), self.theta[l].w[j].sin());
                let z = sigmoid(s.pre[j]);
                let a = 1.0 - z;
                let hr = a * (ct * re[j] - sn * im[j]) + z * s.pre[n + j];
                let hi = a * (sn * re[j] + ct * im[j]) + z * s.pre[2 * n + j];
                re[j] = hr;
                im[j] = hi;
            }
            s.hcat[..n].copy_from_slice(re);
            s.hcat[n..].copy_from_slice(im);
            self.outm[l].apply(&s.hcat, &mut s.zout, &mut s.zpout, &mut s.y);
            for c in 0..h {
                s.u[c] += (s.y[c] + self.d[l].w[c] * s.u[c]).tanh();
            }
        }
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
        let radix = vec![3, 4];
        let (vocab, h, n, layers, b, t) = (12, 16, 16, 2, 2, 5);
        let mut m = KOsc::new(vocab, radix.clone(), h, n, layers, 11);
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
        assert!(checked > 80);
    }

    #[test]
    fn streaming_inference_matches_windows_with_carried_state() {
        let radix = vec![3, 4];
        let (vocab, h, n, layers, t) = (12, 16, 16, 2, 9);
        let m = KOsc::new(vocab, radix.clone(), h, n, layers, 3);
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

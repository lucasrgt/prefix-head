use crate::fast::gemv_rows;
use crate::head::{Head, HeadBuf, HeadScratch};
use crate::rnn::State;
use crate::train::{Param, gemm};
use kern::Rng;

const K: usize = 4;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

fn dsilu(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 + x * (1.0 - s))
}

/// Mamba-2-style selective state-space block (Dao & Gu 2024), written out for the CPU:
/// n = LN(u); [x | z | B | C | dt] = n·W_in; [x | B | C] = SiLU(causal depthwise conv(k=4));
/// Δ = softplus(dt + dt_bias), a = exp(Δ·A_h) with A_h = −exp(A_log_h);
/// per head h: S ← a·S + Δ·x_h ⊗ B, y_h = S·C + D_h·x_h; out = u + (y ⊙ SiLU(z))·W_out.
/// State per layer: heads × P × N for S plus the last K−1 conv inputs.
pub struct Mamba {
    pub vocab: usize,
    pub d: usize,
    pub di: usize,
    pub n: usize,
    pub heads: usize,
    pub layers: usize,
    emb: Param,
    lg: Vec<Param>,
    lb: Vec<Param>,
    win: Vec<Param>,
    cw: Vec<Param>,
    cb: Vec<Param>,
    dtb: Vec<Param>,
    alog: Vec<Param>,
    dd: Vec<Param>,
    wout: Vec<Param>,
    pub head: Head,
}

pub struct Buf {
    pub h0: Vec<Vec<f32>>,
    u: Vec<Vec<f32>>,
    nrm: Vec<Vec<f32>>,
    nhat: Vec<Vec<f32>>,
    nr: Vec<Vec<f32>>,
    proj: Vec<Vec<f32>>,
    cpre: Vec<Vec<f32>>,
    cact: Vec<Vec<f32>>,
    delta: Vec<Vec<f32>>,
    s: Vec<Vec<f32>>,
    y: Vec<Vec<f32>>,
    g: Vec<Vec<f32>>,
    out: Vec<f32>,
    du: Vec<f32>,
    dg: Vec<f32>,
    dproj: Vec<f32>,
    dcact: Vec<f32>,
    dn: Vec<f32>,
    head: HeadBuf,
}

pub struct Scratch {
    u: Vec<f32>,
    nrm: Vec<f32>,
    proj: Vec<f32>,
    cact: Vec<f32>,
    g: Vec<f32>,
    head: HeadScratch,
}

impl Mamba {
    pub fn new(vocab: usize, radix: Vec<usize>, d: usize, di: usize, n: usize, heads: usize, layers: usize, seed: u64) -> Self {
        assert_eq!(di % heads, 0);
        let mut rng = Rng::new(seed);
        let emb = Param::new((vocab + 1) * d, 1.0, &mut rng);
        let pw = 2 * di + 2 * n + heads;
        let cc = di + 2 * n;
        let mk = |k: usize, s: f32, rng: &mut Rng| (0..layers).map(|_| Param::new(k, s, rng)).collect::<Vec<_>>();
        let fill = |k: usize, v: f32| {
            (0..layers)
                .map(|_| {
                    let mut p = Param::new(k, 0.0, &mut Rng::new(1));
                    p.w.fill(v);
                    p
                })
                .collect::<Vec<_>>()
        };
        let lg = fill(d, 1.0);
        let lb = mk(d, 0.0, &mut rng);
        let win = mk(d * pw, 1.0 / (d as f32).sqrt(), &mut rng);
        let cw = mk(cc * K, 1.0 / (K as f32).sqrt(), &mut rng);
        let cb = mk(cc, 0.0, &mut rng);
        let mut dtb = mk(heads, 0.0, &mut rng);
        let mut alog = mk(heads, 0.0, &mut rng);
        for l in 0..layers {
            for h in 0..heads {
                let dt0 = (0.001f32.ln() + (0.1f32.ln() - 0.001f32.ln()) * h as f32 / (heads - 1).max(1) as f32).exp();
                dtb[l].w[h] = dt0.exp_m1().ln();
                alog[l].w[h] = (1.0 + 15.0 * h as f32 / (heads - 1).max(1) as f32).ln();
            }
        }
        let dd = fill(heads, 1.0);
        let wout = mk(di * d, 1.0 / (di as f32).sqrt() / (2.0 * layers as f32).sqrt(), &mut rng);
        let head = Head::new(radix, d, &mut rng);
        Self { vocab, d, di, n, heads, layers, emb, lg, lb, win, cw, cb, dtb, alog, dd, wout, head }
    }

    fn pw(&self) -> usize {
        2 * self.di + 2 * self.n + self.heads
    }

    fn cc(&self) -> usize {
        self.di + 2 * self.n
    }

    fn ssm(&self) -> usize {
        self.di * self.n
    }

    pub fn params(&self) -> usize {
        let per: usize = [&self.lg, &self.lb, &self.win, &self.cw, &self.cb, &self.dtb, &self.alog, &self.dd, &self.wout].iter().flat_map(|v| v.iter()).map(|p| p.w.len()).sum();
        self.emb.w.len() + per + self.head.params()
    }

    pub fn macs_per_token(&self) -> usize {
        self.layers * (self.d * self.pw() + self.di * self.d + 2 * self.ssm() + self.cc() * K) + self.head.macs()
    }

    pub fn state_bytes(&self) -> usize {
        self.layers * (self.ssm() + (K - 1) * self.cc()) * 4
    }

    pub fn buf(&self, rows: usize) -> Buf {
        let (d, di, ls) = (self.d, self.di, self.layers);
        let v = |k: usize| (0..ls).map(|_| vec![0.0; rows * k]).collect::<Vec<_>>();
        Buf {
            h0: Vec::new(),
            u: v(d),
            nrm: v(d),
            nhat: v(d),
            nr: v(1),
            proj: v(self.pw()),
            cpre: v(self.cc()),
            cact: v(self.cc()),
            delta: v(self.heads),
            s: v(self.ssm()),
            y: v(di),
            g: v(di),
            out: vec![0.0; rows * d],
            du: vec![0.0; rows * d],
            dg: vec![0.0; rows * di],
            dproj: vec![0.0; rows * self.pw()],
            dcact: vec![0.0; rows * self.cc()],
            dn: vec![0.0; rows * d],
            head: self.head.buf(rows),
        }
    }

    fn state_len(&self) -> usize {
        self.ssm() + (K - 1) * self.cc()
    }

    /// Conv input of channel c at step k of stream s: the projection, or the carried history before the window.
    fn conv_in(&self, l: usize, bf: &Buf, s: usize, t: usize, k: isize, c: usize) -> f32 {
        if k >= 0 {
            let r = s * t + k as usize;
            let pc = if c < self.di { c } else { c + self.di };
            bf.proj[l][r * self.pw() + pc]
        } else {
            let lag = (-k) as usize;
            bf.h0[l][s * self.state_len() + self.ssm() + (K - 1 - lag) * self.cc() + c]
        }
    }

    pub fn forward(&self, x: &[u32], tgt: &[u32], b: usize, t: usize, bf: &mut Buf) -> f64 {
        let (d, di, n, hs, rows) = (self.d, self.di, self.n, self.heads, b * t);
        let (pw, cc, p) = (self.pw(), self.cc(), di / hs);
        if bf.h0.len() != self.layers || bf.h0[0].len() != b * self.state_len() {
            bf.h0 = vec![vec![0.0; b * self.state_len()]; self.layers];
        }
        for (r, &id) in x.iter().enumerate() {
            bf.u[0][r * d..(r + 1) * d].copy_from_slice(&self.emb.w[id as usize * d..(id as usize + 1) * d]);
        }
        for l in 0..self.layers {
            crate::tfm::ln_forward(&bf.u[l], d, &self.lg[l].w, &self.lb[l].w, &mut bf.nrm[l], &mut bf.nhat[l], &mut bf.nr[l]);
            gemm(rows, d, pw, &bf.nrm[l], false, &self.win[l].w, false, 0.0, &mut bf.proj[l]);
            for s in 0..b {
                for k in 0..t {
                    let r = s * t + k;
                    for c in 0..cc {
                        let mut acc = self.cb[l].w[c];
                        for j in 0..K {
                            acc += self.cw[l].w[c * K + j] * self.conv_in(l, bf, s, t, k as isize - (K - 1 - j) as isize, c);
                        }
                        bf.cpre[l][r * cc + c] = acc;
                        bf.cact[l][r * cc + c] = silu(acc);
                    }
                    for h in 0..hs {
                        bf.delta[l][r * hs + h] = softplus(bf.proj[l][r * pw + 2 * di + 2 * n + h] + self.dtb[l].w[h]);
                    }
                }
            }
            for s in 0..b {
                for h in 0..hs {
                    let a_h = -self.alog[l].w[h].exp();
                    let mut st: Vec<f32> = bf.h0[l][s * self.state_len() + h * p * n..s * self.state_len() + (h + 1) * p * n].to_vec();
                    for k in 0..t {
                        let r = s * t + k;
                        let dl = bf.delta[l][r * hs + h];
                        let a = (dl * a_h).exp();
                        let ca = &bf.cact[l][r * cc..(r + 1) * cc];
                        for i in 0..p {
                            let xi = ca[h * p + i];
                            let mut yi = self.dd[l].w[h] * xi;
                            for j in 0..n {
                                let v = a * st[i * n + j] + dl * xi * ca[di + j];
                                st[i * n + j] = v;
                                yi += v * ca[di + n + j];
                            }
                            bf.y[l][r * di + h * p + i] = yi;
                        }
                        bf.s[l][r * self.ssm() + h * p * n..r * self.ssm() + (h + 1) * p * n].copy_from_slice(&st);
                    }
                }
            }
            for r in 0..rows {
                for c in 0..di {
                    bf.g[l][r * di + c] = bf.y[l][r * di + c] * silu(bf.proj[l][r * pw + di + c]);
                }
            }
            let next = bf.u[l].clone();
            let dst = if l + 1 < self.layers { &mut bf.u[l + 1] } else { &mut bf.out };
            dst.copy_from_slice(&next);
            gemm(rows, di, d, &bf.g[l], false, &self.wout[l].w, false, 1.0, dst);
        }
        self.head.forward(&bf.out, tgt, rows, &mut bf.head)
    }

    pub fn carry(&self, b: usize, t: usize, bf: &mut Buf) {
        let (cc, sl, ss) = (self.cc(), self.state_len(), self.ssm());
        for l in 0..self.layers {
            for s in 0..b {
                let last = s * t + t - 1;
                let snap = bf.s[l][last * ss..(last + 1) * ss].to_vec();
                let mut hist = vec![0.0f32; (K - 1) * cc];
                for lag in 1..K {
                    for c in 0..cc {
                        hist[(K - 1 - lag) * cc + c] = self.conv_in(l, bf, s, t, t as isize - lag as isize, c);
                    }
                }
                bf.h0[l][s * sl..s * sl + ss].copy_from_slice(&snap);
                bf.h0[l][s * sl + ss..(s + 1) * sl].copy_from_slice(&hist);
            }
        }
    }

    pub fn reset_stream(&self, s: usize, bf: &mut Buf) {
        let sl = self.state_len();
        for l in bf.h0.iter_mut() {
            l[s * sl..(s + 1) * sl].fill(0.0);
        }
    }

    pub fn row_losses<'b>(&self, bf: &'b Buf) -> &'b [f32] {
        &bf.head.row_loss
    }

    pub fn backward(&mut self, x: &[u32], tgt: &[u32], b: usize, t: usize, scale: f32, bf: &mut Buf) {
        let (d, di, n, hs, rows) = (self.d, self.di, self.n, self.heads, b * t);
        let (pw, cc, p, ss, sl) = (self.pw(), self.cc(), di / hs, self.ssm(), self.state_len());
        self.head.backward(tgt, rows, scale, &mut bf.head, &mut bf.du);
        for l in (0..self.layers).rev() {
            gemm(di, rows, d, &bf.g[l], true, &bf.du, false, 1.0, &mut self.wout[l].g);
            gemm(rows, d, di, &bf.du, false, &self.wout[l].w, true, 0.0, &mut bf.dg);
            bf.dproj.fill(0.0);
            bf.dcact.fill(0.0);
            let mut dy = vec![0.0f32; rows * di];
            for r in 0..rows {
                for c in 0..di {
                    let zc = bf.proj[l][r * pw + di + c];
                    dy[r * di + c] = bf.dg[r * di + c] * silu(zc);
                    bf.dproj[r * pw + di + c] = bf.dg[r * di + c] * bf.y[l][r * di + c] * dsilu(zc);
                }
            }
            let mut ddelta = vec![0.0f32; rows * hs];
            for s in 0..b {
                for h in 0..hs {
                    let a_h = -self.alog[l].w[h].exp();
                    let mut gs = vec![0.0f32; p * n];
                    let (mut ga_h, mut gd) = (0.0f32, 0.0f32);
                    for k in (0..t).rev() {
                        let r = s * t + k;
                        let dl = bf.delta[l][r * hs + h];
                        let a = (dl * a_h).exp();
                        let st = &bf.s[l][r * ss + h * p * n..r * ss + (h + 1) * p * n];
                        let prev: &[f32] = if k > 0 { &bf.s[l][(r - 1) * ss + h * p * n..(r - 1) * ss + (h + 1) * p * n] } else { &bf.h0[l][s * sl + h * p * n..s * sl + (h + 1) * p * n] };
                        let ca = &bf.cact[l][r * cc..(r + 1) * cc];
                        let mut da = 0.0f32;
                        let mut ddl = 0.0f32;
                        for i in 0..p {
                            let xi = ca[h * p + i];
                            let dyi = dy[r * di + h * p + i];
                            gd += dyi * xi;
                            let mut dxi = self.dd[l].w[h] * dyi;
                            for j in 0..n {
                                let idx = i * n + j;
                                bf.dcact[r * cc + di + n + j] += dyi * st[idx];
                                let g = gs[idx] + dyi * ca[di + n + j];
                                da += g * prev[idx];
                                ddl += g * xi * ca[di + j];
                                dxi += g * dl * ca[di + j];
                                bf.dcact[r * cc + di + j] += g * dl * xi;
                                gs[idx] = g * a;
                            }
                            bf.dcact[r * cc + h * p + i] += dxi;
                        }
                        ddl += da * a_h * a;
                        ga_h += da * dl * a;
                        ddelta[r * hs + h] = ddl;
                    }
                    self.dd[l].g[h] += gd;
                    self.alog[l].g[h] += ga_h * a_h;
                }
            }
            for r in 0..rows {
                for h in 0..hs {
                    let pre = bf.proj[l][r * pw + 2 * di + 2 * n + h] + self.dtb[l].w[h];
                    let gpre = ddelta[r * hs + h] * sigmoid(pre);
                    self.dtb[l].g[h] += gpre;
                    bf.dproj[r * pw + 2 * di + 2 * n + h] = gpre;
                }
            }
            for s in 0..b {
                for k in 0..t {
                    let r = s * t + k;
                    for c in 0..cc {
                        let gpre = bf.dcact[r * cc + c] * dsilu(bf.cpre[l][r * cc + c]);
                        self.cb[l].g[c] += gpre;
                        for j in 0..K {
                            let kk = k as isize - (K - 1 - j) as isize;
                            self.cw[l].g[c * K + j] += gpre * self.conv_in(l, bf, s, t, kk, c);
                            if kk >= 0 {
                                let rr = s * t + kk as usize;
                                let pc = if c < di { c } else { c + di };
                                bf.dproj[rr * pw + pc] += gpre * self.cw[l].w[c * K + j];
                            }
                        }
                    }
                }
            }
            gemm(d, rows, pw, &bf.nrm[l], true, &bf.dproj, false, 1.0, &mut self.win[l].g);
            gemm(rows, pw, d, &bf.dproj, false, &self.win[l].w, true, 0.0, &mut bf.dn);
            let (lg, lb) = (&mut self.lg[l], &mut self.lb[l]);
            crate::tfm::ln_backward(&bf.dn, d, &lg.w, &bf.nhat[l], &bf.nr[l], &mut lg.g, &mut lb.g, &mut bf.du);
        }
        for (r, &id) in x.iter().enumerate() {
            let row = &mut self.emb.g[id as usize * d..(id as usize + 1) * d];
            for (g, &dv) in row.iter_mut().zip(&bf.du[r * d..(r + 1) * d]) {
                *g += dv;
            }
        }
    }

    fn params_mut(&mut self) -> impl Iterator<Item = &mut Param> {
        let per = self.lg.iter_mut().chain(self.lb.iter_mut()).chain(self.win.iter_mut()).chain(self.cw.iter_mut()).chain(self.cb.iter_mut()).chain(self.dtb.iter_mut()).chain(self.alog.iter_mut()).chain(self.dd.iter_mut()).chain(self.wout.iter_mut());
        std::iter::once(&mut self.emb).chain(per).chain(self.head.params_mut())
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
        let header = format!("mamba {} {} {} {} {} {} {radix}\n{meta}\n", self.vocab, self.d, self.di, self.n, self.heads, self.layers);
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
        assert_eq!(f[0], "mamba", "not a mamba checkpoint");
        let q = |i: usize| f[i].parse::<usize>().unwrap();
        let radix = f[7].split('x').map(|r| r.parse().unwrap()).collect();
        let mut m = Self::new(q(1), radix, q(2), q(3), q(4), q(5), q(6), 0);
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
        State(vec![vec![0.0; self.state_len()]; self.layers])
    }

    pub fn scratch(&self) -> Scratch {
        Scratch {
            u: vec![0.0; self.d],
            nrm: vec![0.0; self.d],
            proj: vec![0.0; self.pw()],
            cact: vec![0.0; self.cc()],
            g: vec![0.0; self.di],
            head: self.head.scratch(),
        }
    }

    pub fn advance(&self, id: u32, st: &mut State, sc: &mut Scratch) {
        let (d, di, n, hs) = (self.d, self.di, self.n, self.heads);
        let (pw, cc, p, ss) = (self.pw(), self.cc(), di / hs, self.ssm());
        sc.u.copy_from_slice(&self.emb.w[id as usize * d..(id as usize + 1) * d]);
        for l in 0..self.layers {
            let mut hat = vec![0.0f32; d];
            let mut r1 = [0.0f32];
            crate::tfm::ln_forward(&sc.u, d, &self.lg[l].w, &self.lb[l].w, &mut sc.nrm, &mut hat, &mut r1);
            sc.proj.fill(0.0);
            gemv_rows(&sc.nrm, &self.win[l].w, pw, &mut sc.proj);
            let state = &mut st.0[l];
            for c in 0..cc {
                let pc = if c < di { c } else { c + di };
                let cur = sc.proj[pc];
                let mut acc = self.cb[l].w[c] + self.cw[l].w[c * K + K - 1] * cur;
                for j in 0..K - 1 {
                    acc += self.cw[l].w[c * K + j] * state[ss + j * cc + c];
                }
                for j in 0..K - 2 {
                    state[ss + j * cc + c] = state[ss + (j + 1) * cc + c];
                }
                state[ss + (K - 2) * cc + c] = cur;
                sc.cact[c] = silu(acc);
            }
            for h in 0..hs {
                let dl = softplus(sc.proj[2 * di + 2 * n + h] + self.dtb[l].w[h]);
                let a = (dl * -self.alog[l].w[h].exp()).exp();
                for i in 0..p {
                    let xi = sc.cact[h * p + i];
                    let mut yi = self.dd[l].w[h] * xi;
                    for j in 0..n {
                        let idx = h * p * n + i * n + j;
                        let v = a * state[idx] + dl * xi * sc.cact[di + j];
                        state[idx] = v;
                        yi += v * sc.cact[di + n + j];
                    }
                    sc.g[h * p + i] = yi * silu(sc.proj[di + h * p + i]);
                }
            }
            gemv_rows(&sc.g, &self.wout[l].w, d, &mut sc.u);
        }
    }

    pub fn infer_step(&self, id: u32, st: &mut State, sc: &mut Scratch, pick: impl FnMut(usize, &[f32], f32) -> u32, out: &mut [u32]) -> f32 {
        self.advance(id, st, sc);
        self.head.infer(&sc.u, &mut sc.head, pick, out)
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
        let (vocab, d, di, n, hs, layers, b, t) = (12, 8, 8, 3, 2, 2, 2, 6);
        let mut m = Mamba::new(vocab, radix.clone(), d, di, n, hs, layers, 11);
        for p in m.lb.iter_mut().chain(m.cb.iter_mut()) {
            p.w.iter_mut().enumerate().for_each(|(i, w)| *w = 0.05 * i as f32 - 0.1);
        }
        let mut rng = Rng::new(5);
        let (x, y) = batch(&mut rng, vocab, &radix, b, t);
        let mut bf = m.buf(b * t);
        bf.h0 = (0..layers).map(|_| (0..b * m.state_len()).map(|_| rng.f32()).collect()).collect();
        m.forward(&x, &y, b, t, &mut bf);
        m.backward(&x, &y, b, t, 1.0, &mut bf);
        let grads: Vec<Vec<f32>> = m.all_params().iter().map(|p| p.g.clone()).collect();
        let eps = 1e-2f32;
        let mut bad = Vec::new();
        let mut checked = 0;
        for pi in 0..grads.len() {
            let len = grads[pi].len();
            for k in (0..len).step_by((len / 5).max(1)) {
                let orig = m.all_params()[pi].w[k];
                m.all_params()[pi].w[k] = orig + eps;
                let lp = m.forward(&x, &y, b, t, &mut bf);
                m.all_params()[pi].w[k] = orig - eps;
                let lm = m.forward(&x, &y, b, t, &mut bf);
                m.all_params()[pi].w[k] = orig;
                let num = ((lp - lm) / (2.0 * eps as f64)) as f32;
                let ana = grads[pi][k];
                if (num - ana).abs() > 4e-3 + 3e-2 * ana.abs().max(num.abs()) {
                    bad.push(format!("param {pi}[{k}]: numeric {num} vs analytic {ana}"));
                }
                checked += 1;
            }
        }
        assert!(bad.is_empty(), "{}", bad.join("\n"));
        assert!(checked > 80);
    }

    #[test]
    fn streaming_inference_matches_windows_with_carried_state() {
        let radix = vec![3, 4];
        let (vocab, d, di, n, hs, layers, t) = (12, 8, 8, 3, 2, 2, 9);
        let m = Mamba::new(vocab, radix.clone(), d, di, n, hs, layers, 3);
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
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }
}

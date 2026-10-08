use crate::fast::gemv_rows;
use crate::head::{Head, HeadBuf, HeadScratch};
use crate::rnn::State;
use crate::train::{Param, gemm};
use kern::Rng;

/// Pre-LayerNorm causal transformer baseline with ALiBi positions (no position parameters, so
/// inference can slide a window of `ctx` tokens) and the same coarse-to-fine head as the other
/// cores. Block: x += Attn(LN(x))·Wo; x += ReLU(LN(x)·W1)·W2. Final LN, then the head.
pub struct Tfm {
    pub vocab: usize,
    pub d: usize,
    pub heads: usize,
    pub layers: usize,
    pub mlp: usize,
    pub ctx: usize,
    emb: Param,
    g1: Vec<Param>,
    b1: Vec<Param>,
    wq: Vec<Param>,
    wk: Vec<Param>,
    wv: Vec<Param>,
    wo: Vec<Param>,
    g2: Vec<Param>,
    b2: Vec<Param>,
    w1: Vec<Param>,
    w2: Vec<Param>,
    gf: Param,
    bf: Param,
    pub head: Head,
}

pub struct Buf {
    pub h0: Vec<Vec<f32>>,
    x: Vec<Vec<f32>>,
    a: Vec<Vec<f32>>,
    ahat: Vec<Vec<f32>>,
    ar: Vec<Vec<f32>>,
    q: Vec<Vec<f32>>,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    p: Vec<Vec<f32>>,
    o: Vec<Vec<f32>>,
    xm: Vec<Vec<f32>>,
    c: Vec<Vec<f32>>,
    chat: Vec<Vec<f32>>,
    cr: Vec<Vec<f32>>,
    pre: Vec<Vec<f32>>,
    act: Vec<Vec<f32>>,
    xf: Vec<f32>,
    fhat: Vec<f32>,
    fr: Vec<f32>,
    out: Vec<f32>,
    dx: Vec<f32>,
    dt: Vec<f32>,
    dq: Vec<f32>,
    dk: Vec<f32>,
    dv: Vec<f32>,
    dpre: Vec<f32>,
    head: HeadBuf,
}

pub struct Scratch {
    x: Vec<f32>,
    a: Vec<f32>,
    q: Vec<f32>,
    kv: Vec<f32>,
    o: Vec<f32>,
    t: Vec<f32>,
    m: Vec<f32>,
    s: Vec<f32>,
    head: HeadScratch,
}

fn slope(h: usize, heads: usize) -> f32 {
    2f32.powf(-8.0 * (h + 1) as f32 / heads as f32)
}

/// y = (x − μ)/σ·g + b over rows of width d; keeps x̂ and 1/σ for backward.
fn ln_forward(x: &[f32], d: usize, g: &[f32], b: &[f32], y: &mut [f32], xhat: &mut [f32], rstd: &mut [f32]) {
    for r in 0..x.len() / d {
        let row = &x[r * d..(r + 1) * d];
        let mu = row.iter().sum::<f32>() / d as f32;
        let var = row.iter().map(|v| (v - mu) * (v - mu)).sum::<f32>() / d as f32;
        let rs = 1.0 / (var + 1e-5).sqrt();
        rstd[r] = rs;
        for c in 0..d {
            let xh = (row[c] - mu) * rs;
            xhat[r * d + c] = xh;
            y[r * d + c] = xh * g[c] + b[c];
        }
    }
}

/// Adds dL/dx into `dx` given dL/dy, accumulating dg and db.
fn ln_backward(dy: &[f32], d: usize, g: &[f32], xhat: &[f32], rstd: &[f32], gg: &mut [f32], gb: &mut [f32], dx: &mut [f32]) {
    for r in 0..dy.len() / d {
        let (mut s1, mut s2) = (0.0f32, 0.0f32);
        for c in 0..d {
            let i = r * d + c;
            let dxh = dy[i] * g[c];
            gg[c] += dy[i] * xhat[i];
            gb[c] += dy[i];
            s1 += dxh;
            s2 += dxh * xhat[i];
        }
        for c in 0..d {
            let i = r * d + c;
            let dxh = dy[i] * g[c];
            dx[i] += rstd[r] / d as f32 * (d as f32 * dxh - s1 - xhat[i] * s2);
        }
    }
}

impl Tfm {
    pub fn new(vocab: usize, radix: Vec<usize>, d: usize, heads: usize, layers: usize, mlp: usize, ctx: usize, seed: u64) -> Self {
        assert_eq!(d % heads, 0);
        let mut rng = Rng::new(seed);
        let s = 1.0 / (d as f32).sqrt();
        let emb = Param::new((vocab + 1) * d, 1.0, &mut rng);
        let mk = |n: usize, sc: f32, rng: &mut Rng| (0..layers).map(|_| Param::new(n, sc, rng)).collect::<Vec<_>>();
        let ones = |n: usize| {
            let mut p = Param::new(n, 0.0, &mut Rng::new(1));
            p.w.fill(1.0);
            p
        };
        let g1 = (0..layers).map(|_| ones(d)).collect();
        let b1 = mk(d, 0.0, &mut rng);
        let wq = mk(d * d, s, &mut rng);
        let wk = mk(d * d, s, &mut rng);
        let wv = mk(d * d, s, &mut rng);
        let wo = mk(d * d, s / (2.0 * layers as f32).sqrt(), &mut rng);
        let g2 = (0..layers).map(|_| ones(d)).collect();
        let b2 = mk(d, 0.0, &mut rng);
        let w1 = mk(d * mlp, s, &mut rng);
        let w2 = mk(mlp * d, 1.0 / (mlp as f32).sqrt() / (2.0 * layers as f32).sqrt(), &mut rng);
        let gf = ones(d);
        let bf = Param::new(d, 0.0, &mut rng);
        let head = Head::new(radix, d, &mut rng);
        Self { vocab, d, heads, layers, mlp, ctx, emb, g1, b1, wq, wk, wv, wo, g2, b2, w1, w2, gf, bf, head }
    }

    pub fn params(&self) -> usize {
        let per: usize = [&self.g1, &self.b1, &self.wq, &self.wk, &self.wv, &self.wo, &self.g2, &self.b2, &self.w1, &self.w2].iter().flat_map(|v| v.iter()).map(|p| p.w.len()).sum();
        self.emb.w.len() + per + 2 * self.d + self.head.params()
    }

    /// Per generated token with a full window: projections, MLP and attention over `ctx` keys.
    pub fn macs_per_token(&self) -> usize {
        self.layers * (4 * self.d * self.d + 2 * self.d * self.mlp + 2 * self.ctx * self.d) + self.head.macs()
    }

    /// KV cache at a full window.
    pub fn state_bytes(&self) -> usize {
        self.layers * self.ctx * 2 * self.d * 4
    }

    pub fn buf(&self, rows: usize) -> Buf {
        let (d, m, ls) = (self.d, self.mlp, self.layers);
        let v = |k: usize| (0..ls).map(|_| vec![0.0; rows * k]).collect::<Vec<_>>();
        Buf {
            h0: Vec::new(),
            x: v(d),
            a: v(d),
            ahat: v(d),
            ar: v(1),
            q: v(d),
            k: v(d),
            v: v(d),
            p: Vec::new(),
            o: v(d),
            xm: v(d),
            c: v(d),
            chat: v(d),
            cr: v(1),
            pre: v(m),
            act: v(m),
            xf: vec![0.0; rows * d],
            fhat: vec![0.0; rows * d],
            fr: vec![0.0; rows],
            out: vec![0.0; rows * d],
            dx: vec![0.0; rows * d],
            dt: vec![0.0; rows * d.max(m)],
            dq: vec![0.0; rows * d],
            dk: vec![0.0; rows * d],
            dv: vec![0.0; rows * d],
            dpre: vec![0.0; rows * m],
            head: self.head.buf(rows),
        }
    }

    fn attend(&self, l: usize, b: usize, t: usize, bf: &mut Buf) {
        let (d, hs) = (self.d, self.heads);
        let hd = d / hs;
        let scale = 1.0 / (hd as f32).sqrt();
        if bf.p.len() != self.layers || bf.p[0].len() != b * hs * t * t {
            bf.p = vec![vec![0.0; b * hs * t * t]; self.layers];
        }
        for s in 0..b {
            for h in 0..hs {
                let m = slope(h, hs);
                let pb = (s * hs + h) * t * t;
                for i in 0..t {
                    let qi = &bf.q[l][(s * t + i) * d + h * hd..(s * t + i) * d + (h + 1) * hd];
                    let row = &mut bf.p[l][pb + i * t..pb + (i + 1) * t];
                    let lo = (i + 1).saturating_sub(self.ctx);
                    let mut mx = f32::MIN;
                    for j in lo..=i {
                        let kj = &bf.k[l][(s * t + j) * d + h * hd..(s * t + j) * d + (h + 1) * hd];
                        let sc = qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale - m * (i - j) as f32;
                        row[j] = sc;
                        mx = mx.max(sc);
                    }
                    let mut z = 0.0;
                    for j in 0..t {
                        if j < lo || j > i {
                            row[j] = 0.0;
                        } else {
                            row[j] = (row[j] - mx).exp();
                            z += row[j];
                        }
                    }
                    for j in lo..=i {
                        row[j] /= z;
                    }
                    let oi = &mut bf.o[l][(s * t + i) * d + h * hd..(s * t + i) * d + (h + 1) * hd];
                    oi.fill(0.0);
                    for j in lo..=i {
                        let vj = &bf.v[l][(s * t + j) * d + h * hd..(s * t + j) * d + (h + 1) * hd];
                        for (o, &vv) in oi.iter_mut().zip(vj) {
                            *o += row[j] * vv;
                        }
                    }
                }
            }
        }
    }

    pub fn forward(&self, x: &[u32], tgt: &[u32], b: usize, t: usize, bf: &mut Buf) -> f64 {
        let (d, m, rows) = (self.d, self.mlp, b * t);
        for (r, &id) in x.iter().enumerate() {
            bf.x[0][r * d..(r + 1) * d].copy_from_slice(&self.emb.w[id as usize * d..(id as usize + 1) * d]);
        }
        for l in 0..self.layers {
            ln_forward(&bf.x[l], d, &self.g1[l].w, &self.b1[l].w, &mut bf.a[l], &mut bf.ahat[l], &mut bf.ar[l]);
            gemm(rows, d, d, &bf.a[l], false, &self.wq[l].w, false, 0.0, &mut bf.q[l]);
            gemm(rows, d, d, &bf.a[l], false, &self.wk[l].w, false, 0.0, &mut bf.k[l]);
            gemm(rows, d, d, &bf.a[l], false, &self.wv[l].w, false, 0.0, &mut bf.v[l]);
            self.attend(l, b, t, bf);
            bf.xm[l].copy_from_slice(&bf.x[l]);
            gemm(rows, d, d, &bf.o[l], false, &self.wo[l].w, false, 1.0, &mut bf.xm[l]);
            ln_forward(&bf.xm[l], d, &self.g2[l].w, &self.b2[l].w, &mut bf.c[l], &mut bf.chat[l], &mut bf.cr[l]);
            gemm(rows, d, m, &bf.c[l], false, &self.w1[l].w, false, 0.0, &mut bf.pre[l]);
            for (a, &p) in bf.act[l].iter_mut().zip(&bf.pre[l]) {
                *a = p.max(0.0);
            }
            let next = if l + 1 < self.layers { l + 1 } else { usize::MAX };
            let xm = bf.xm[l].clone();
            if next != usize::MAX {
                bf.x[next].copy_from_slice(&xm);
                gemm(rows, m, d, &bf.act[l], false, &self.w2[l].w, false, 1.0, &mut bf.x[next]);
            } else {
                bf.xf.copy_from_slice(&xm);
                gemm(rows, m, d, &bf.act[l], false, &self.w2[l].w, false, 1.0, &mut bf.xf);
            }
        }
        ln_forward(&bf.xf, d, &self.gf.w, &self.bf.w, &mut bf.out, &mut bf.fhat, &mut bf.fr);
        self.head.forward(&bf.out, tgt, rows, &mut bf.head)
    }

    pub fn carry(&self, _b: usize, _t: usize, _bf: &mut Buf) {}

    pub fn reset_stream(&self, _s: usize, _bf: &mut Buf) {}

    pub fn row_losses<'b>(&self, bf: &'b Buf) -> &'b [f32] {
        &bf.head.row_loss
    }

    pub fn backward(&mut self, x: &[u32], tgt: &[u32], b: usize, t: usize, scale: f32, bf: &mut Buf) {
        let (d, m, hs, rows) = (self.d, self.mlp, self.heads, b * t);
        let hd = d / hs;
        let sc = 1.0 / (hd as f32).sqrt();
        self.head.backward(tgt, rows, scale, &mut bf.head, &mut bf.dt);
        bf.dx.fill(0.0);
        ln_backward(&bf.dt[..rows * d], d, &self.gf.w, &bf.fhat, &bf.fr, &mut self.gf.g, &mut self.bf.g, &mut bf.dx);
        for l in (0..self.layers).rev() {
            gemm(m, rows, d, &bf.act[l], true, &bf.dx, false, 1.0, &mut self.w2[l].g);
            gemm(rows, d, m, &bf.dx, false, &self.w2[l].w, true, 0.0, &mut bf.dpre);
            for (g, &p) in bf.dpre.iter_mut().zip(&bf.pre[l]) {
                if p <= 0.0 {
                    *g = 0.0;
                }
            }
            gemm(d, rows, m, &bf.c[l], true, &bf.dpre, false, 1.0, &mut self.w1[l].g);
            gemm(rows, m, d, &bf.dpre, false, &self.w1[l].w, true, 0.0, &mut bf.dt[..rows * d]);
            let dc = bf.dt[..rows * d].to_vec();
            let (g2, b2) = (&mut self.g2[l], &mut self.b2[l]);
            ln_backward(&dc, d, &g2.w, &bf.chat[l], &bf.cr[l], &mut g2.g, &mut b2.g, &mut bf.dx);
            gemm(d, rows, d, &bf.o[l], true, &bf.dx, false, 1.0, &mut self.wo[l].g);
            gemm(rows, d, d, &bf.dx, false, &self.wo[l].w, true, 0.0, &mut bf.dt[..rows * d]);
            bf.dq.fill(0.0);
            bf.dk.fill(0.0);
            bf.dv.fill(0.0);
            let mut dp = vec![0.0f32; t];
            for s in 0..b {
                for h in 0..hs {
                    let pb = (s * hs + h) * t * t;
                    for i in 0..t {
                        let lo = (i + 1).saturating_sub(self.ctx);
                        let doi = &bf.dt[(s * t + i) * d + h * hd..(s * t + i) * d + (h + 1) * hd];
                        let p = &bf.p[l][pb + i * t..pb + (i + 1) * t];
                        let mut dot = 0.0f32;
                        for j in lo..=i {
                            let vj = &bf.v[l][(s * t + j) * d + h * hd..(s * t + j) * d + (h + 1) * hd];
                            dp[j] = doi.iter().zip(vj).map(|(a, b)| a * b).sum();
                            dot += dp[j] * p[j];
                            let dvj = &mut bf.dv[(s * t + j) * d + h * hd..(s * t + j) * d + (h + 1) * hd];
                            for (g, &o) in dvj.iter_mut().zip(doi) {
                                *g += p[j] * o;
                            }
                        }
                        for j in lo..=i {
                            let ds = p[j] * (dp[j] - dot) * sc;
                            for c in 0..hd {
                                let (qi, kj) = ((s * t + i) * d + h * hd + c, (s * t + j) * d + h * hd + c);
                                bf.dq[qi] += ds * bf.k[l][kj];
                                bf.dk[kj] += ds * bf.q[l][qi];
                            }
                        }
                    }
                }
            }
            gemm(d, rows, d, &bf.a[l], true, &bf.dq, false, 1.0, &mut self.wq[l].g);
            gemm(d, rows, d, &bf.a[l], true, &bf.dk, false, 1.0, &mut self.wk[l].g);
            gemm(d, rows, d, &bf.a[l], true, &bf.dv, false, 1.0, &mut self.wv[l].g);
            gemm(rows, d, d, &bf.dq, false, &self.wq[l].w, true, 0.0, &mut bf.dt[..rows * d]);
            gemm(rows, d, d, &bf.dk, false, &self.wk[l].w, true, 1.0, &mut bf.dt[..rows * d]);
            gemm(rows, d, d, &bf.dv, false, &self.wv[l].w, true, 1.0, &mut bf.dt[..rows * d]);
            let da = bf.dt[..rows * d].to_vec();
            let (g1, b1) = (&mut self.g1[l], &mut self.b1[l]);
            ln_backward(&da, d, &g1.w, &bf.ahat[l], &bf.ar[l], &mut g1.g, &mut b1.g, &mut bf.dx);
        }
        for (r, &id) in x.iter().enumerate() {
            let row = &mut self.emb.g[id as usize * d..(id as usize + 1) * d];
            for (g, &dv) in row.iter_mut().zip(&bf.dx[r * d..(r + 1) * d]) {
                *g += dv;
            }
        }
    }

    fn params_mut(&mut self) -> impl Iterator<Item = &mut Param> {
        let per = self.g1.iter_mut().chain(self.b1.iter_mut()).chain(self.wq.iter_mut()).chain(self.wk.iter_mut()).chain(self.wv.iter_mut()).chain(self.wo.iter_mut()).chain(self.g2.iter_mut()).chain(self.b2.iter_mut()).chain(self.w1.iter_mut()).chain(self.w2.iter_mut());
        std::iter::once(&mut self.emb).chain(per).chain([&mut self.gf, &mut self.bf]).chain(self.head.params_mut())
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
        let header = format!("tfm {} {} {} {} {} {} {radix}\n{meta}\n", self.vocab, self.d, self.heads, self.layers, self.mlp, self.ctx);
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
        assert_eq!(f[0], "tfm", "not a transformer checkpoint");
        let p = |i: usize| f[i].parse::<usize>().unwrap();
        let radix = f[7].split('x').map(|r| r.parse().unwrap()).collect();
        let mut m = Self::new(p(1), radix, p(2), p(3), p(4), p(5), p(6), 0);
        let mut it = lines.next().unwrap().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()));
        for p in m.params_mut() {
            for w in p.w.iter_mut() {
                *w = it.next().expect("checkpoint too short");
            }
        }
        assert!(it.next().is_none(), "checkpoint too long");
        Ok((m, meta))
    }

    /// The KV cache: per layer, keys then values of up to `ctx` past tokens (oldest first).
    pub fn state(&self) -> State {
        State(vec![Vec::new(); self.layers])
    }

    pub fn scratch(&self) -> Scratch {
        let d = self.d;
        Scratch {
            x: vec![0.0; d],
            a: vec![0.0; d],
            q: vec![0.0; d],
            kv: vec![0.0; 2 * d],
            o: vec![0.0; d],
            t: vec![0.0; d],
            m: vec![0.0; self.mlp],
            s: vec![0.0; self.ctx],
            head: self.head.scratch(),
        }
    }

    fn ln1(x: &[f32], g: &[f32], b: &[f32], y: &mut [f32]) {
        let d = x.len();
        let mu = x.iter().sum::<f32>() / d as f32;
        let var = x.iter().map(|v| (v - mu) * (v - mu)).sum::<f32>() / d as f32;
        let rs = 1.0 / (var + 1e-5).sqrt();
        for c in 0..d {
            y[c] = (x[c] - mu) * rs * g[c] + b[c];
        }
    }

    pub fn advance(&self, id: u32, st: &mut State, s: &mut Scratch) {
        let (d, hs) = (self.d, self.heads);
        let hd = d / hs;
        let sc = 1.0 / (hd as f32).sqrt();
        s.x.copy_from_slice(&self.emb.w[id as usize * d..(id as usize + 1) * d]);
        for l in 0..self.layers {
            Self::ln1(&s.x, &self.g1[l].w, &self.b1[l].w, &mut s.a);
            s.q.fill(0.0);
            s.kv.fill(0.0);
            gemv_rows(&s.a, &self.wq[l].w, d, &mut s.q);
            gemv_rows(&s.a, &self.wk[l].w, d, &mut s.kv[..d]);
            gemv_rows(&s.a, &self.wv[l].w, d, &mut s.kv[d..]);
            let cache = &mut st.0[l];
            if cache.len() / (2 * d) == self.ctx {
                cache.drain(..2 * d);
            }
            cache.extend_from_slice(&s.kv);
            let n = cache.len() / (2 * d);
            for h in 0..hs {
                let m = slope(h, hs);
                let qh = &s.q[h * hd..(h + 1) * hd];
                let mut mx = f32::MIN;
                for j in 0..n {
                    let kj = &cache[j * 2 * d + h * hd..j * 2 * d + (h + 1) * hd];
                    s.s[j] = qh.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * sc - m * (n - 1 - j) as f32;
                    mx = mx.max(s.s[j]);
                }
                let mut z = 0.0;
                for j in 0..n {
                    s.s[j] = (s.s[j] - mx).exp();
                    z += s.s[j];
                }
                let oh = &mut s.o[h * hd..(h + 1) * hd];
                oh.fill(0.0);
                for j in 0..n {
                    let vj = &cache[j * 2 * d + d + h * hd..j * 2 * d + d + (h + 1) * hd];
                    for (o, &v) in oh.iter_mut().zip(vj) {
                        *o += s.s[j] / z * v;
                    }
                }
            }
            gemv_rows(&s.o, &self.wo[l].w, d, &mut s.x);
            Self::ln1(&s.x, &self.g2[l].w, &self.b2[l].w, &mut s.t);
            s.m.fill(0.0);
            gemv_rows(&s.t, &self.w1[l].w, self.mlp, &mut s.m);
            for v in s.m.iter_mut() {
                *v = v.max(0.0);
            }
            gemv_rows(&s.m, &self.w2[l].w, d, &mut s.x);
        }
        Self::ln1(&s.x, &self.gf.w, &self.bf.w, &mut s.t);
    }

    pub fn infer_step(&self, id: u32, st: &mut State, s: &mut Scratch, pick: impl FnMut(usize, &[f32], f32) -> u32, out: &mut [u32]) -> f32 {
        self.advance(id, st, s);
        self.head.infer(&s.t, &mut s.head, pick, out)
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
        let (vocab, d, hs, layers, mlp, ctx, b, t) = (12, 8, 2, 2, 12, 4, 2, 6);
        let mut m = Tfm::new(vocab, radix.clone(), d, hs, layers, mlp, ctx, 11);
        for p in m.b1.iter_mut().chain(m.b2.iter_mut()) {
            p.w.iter_mut().enumerate().for_each(|(i, w)| *w = 0.05 * i as f32);
        }
        let mut rng = Rng::new(5);
        let (x, y) = batch(&mut rng, vocab, &radix, b, t);
        let mut bf = m.buf(b * t);
        m.forward(&x, &y, b, t, &mut bf);
        m.backward(&x, &y, b, t, 1.0, &mut bf);
        let grads: Vec<Vec<f32>> = m.all_params().iter().map(|p| p.g.clone()).collect();
        let eps = 1e-2f32;
        // A W1 column whose ReLU input passes within 0.01 of zero would be measured across the kink.
        let near_kink: Vec<Vec<bool>> = (0..layers).map(|l| (0..mlp).map(|u| (0..b * t).any(|r| bf.pre[l][r * mlp + u].abs() < 1e-2)).collect()).collect();
        let w1_index = |l: usize| 1 + 8 * layers + l;
        let mut checked = 0;
        let mut bad = Vec::new();
        for pi in 0..grads.len() {
            let len = grads[pi].len();
            for k in (0..len).step_by((len / 5).max(1)) {
                if (0..layers).any(|l| pi == w1_index(l) && near_kink[l][k % mlp]) {
                    continue;
                }
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
    fn streaming_with_kv_cache_matches_batched_forward() {
        let radix = vec![3, 4];
        let (vocab, d, hs, layers, mlp, ctx, t) = (12, 8, 2, 2, 12, 4, 9);
        let m = Tfm::new(vocab, radix.clone(), d, hs, layers, mlp, ctx, 3);
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

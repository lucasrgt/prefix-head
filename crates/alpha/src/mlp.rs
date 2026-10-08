use kern::Rng;

struct Param {
    w: Vec<f32>,
    g: Vec<f32>,
    m: Vec<f32>,
    s: Vec<f32>,
}

impl Param {
    fn new(n: usize, scale: f32, rng: &mut Rng) -> Self {
        let w = (0..n).map(|_| rng.f32() * 2.0 * scale).collect();
        Self { w, g: vec![0.0; n], m: vec![0.0; n], s: vec![0.0; n] }
    }

    fn adam(&mut self, lr: f32, t: i32) {
        let (b1, b2) = (0.9f32, 0.999f32);
        let c1 = 1.0 - b1.powi(t);
        let c2 = 1.0 - b2.powi(t);
        for i in 0..self.w.len() {
            let g = self.g[i];
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g;
            self.s[i] = b2 * self.s[i] + (1.0 - b2) * g * g;
            self.w[i] -= lr * (self.m[i] / c1) / ((self.s[i] / c2).sqrt() + 1e-8);
            self.g[i] = 0.0;
        }
    }
}

/// `a` is m×k (or k×m when `ta`), `b` is k×n (or n×k when `tb`), all row-major; c = a·b + beta·c.
fn gemm(m: usize, k: usize, n: usize, a: &[f32], ta: bool, b: &[f32], tb: bool, beta: f32, c: &mut [f32]) {
    let (rsa, csa) = if ta { (1, m as isize) } else { (k as isize, 1) };
    let (rsb, csb) = if tb { (1, k as isize) } else { (n as isize, 1) };
    unsafe {
        matrixmultiply::sgemm(m, k, n, 1.0, a.as_ptr(), rsa, csa, b.as_ptr(), rsb, csb, beta, c.as_mut_ptr(), n as isize, 1);
    }
}

pub struct Mlp {
    pub v: usize,
    pub d: usize,
    pub k: usize,
    pub h: usize,
    e: Param,
    w1: Param,
    b1: Param,
    w2: Param,
    b2: Param,
}

pub struct Batch {
    x: Vec<f32>,
    a: Vec<f32>,
    z: Vec<f32>,
    dx: Vec<f32>,
    ctx: Vec<u32>,
}

impl Mlp {
    pub fn new(v: usize, d: usize, k: usize, h: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        Self {
            v,
            d,
            k,
            h,
            e: Param::new((v + 1) * d, 1.0, &mut rng),
            w1: Param::new(k * d * h, 1.0 / ((k * d) as f32).sqrt(), &mut rng),
            b1: Param::new(h, 0.0, &mut rng),
            w2: Param::new(h * v, 1.0 / (h as f32).sqrt(), &mut rng),
            b2: Param::new(v, 0.0, &mut rng),
        }
    }

    pub fn params(&self) -> usize {
        self.e.w.len() + self.w1.w.len() + self.b1.w.len() + self.w2.w.len() + self.b2.w.len()
    }

    pub fn macs_per_symbol(&self) -> usize {
        self.k * self.d * self.h + self.h * self.v
    }

    pub fn batch(&self, n: usize) -> Batch {
        Batch {
            x: vec![0.0; n * self.k * self.d],
            a: vec![0.0; n * self.h],
            z: vec![0.0; n * self.v],
            dx: vec![0.0; n * self.k * self.d],
            ctx: vec![0; n * self.k],
        }
    }

    fn context(&self, seq: &[u32], t: usize, out: &mut [u32]) {
        for j in 0..self.k {
            let back = self.k - j;
            out[j] = if t >= back { seq[t - back] } else { self.v as u32 };
        }
    }

    /// Fills `bt.z` with log-probabilities for positions `ts`; returns summed -ln p(target).
    fn forward(&self, seq: &[u32], ts: &[usize], bt: &mut Batch) -> f64 {
        let (n, kd, h, v) = (ts.len(), self.k * self.d, self.h, self.v);
        for (i, &t) in ts.iter().enumerate() {
            self.context(seq, t, &mut bt.ctx[i * self.k..(i + 1) * self.k]);
            for j in 0..self.k {
                let s = bt.ctx[i * self.k + j] as usize;
                bt.x[i * kd + j * self.d..i * kd + (j + 1) * self.d].copy_from_slice(&self.e.w[s * self.d..(s + 1) * self.d]);
            }
        }
        for i in 0..n {
            bt.a[i * h..(i + 1) * h].copy_from_slice(&self.b1.w);
            bt.z[i * v..(i + 1) * v].copy_from_slice(&self.b2.w);
        }
        gemm(n, kd, h, &bt.x[..n * kd], false, &self.w1.w, false, 1.0, &mut bt.a[..n * h]);
        for a in &mut bt.a[..n * h] {
            *a = a.tanh();
        }
        gemm(n, h, v, &bt.a[..n * h], false, &self.w2.w, false, 1.0, &mut bt.z[..n * v]);
        let mut loss = 0.0f64;
        for (i, &t) in ts.iter().enumerate() {
            let row = &mut bt.z[i * v..(i + 1) * v];
            let mx = row.iter().copied().fold(f32::MIN, f32::max);
            let lse = mx + row.iter().map(|&z| (z - mx).exp()).sum::<f32>().ln();
            for z in row.iter_mut() {
                *z -= lse;
            }
            loss -= row[seq[t] as usize] as f64;
        }
        loss
    }

    pub fn train_step(&mut self, seq: &[u32], ts: &[usize], bt: &mut Batch, lr: f32, step: i32) -> f64 {
        let loss = self.forward(seq, ts, bt);
        let (n, kd, h, v) = (ts.len(), self.k * self.d, self.h, self.v);
        let inv = 1.0 / n as f32;
        for (i, &t) in ts.iter().enumerate() {
            let row = &mut bt.z[i * v..(i + 1) * v];
            for z in row.iter_mut() {
                *z = z.exp() * inv;
            }
            row[seq[t] as usize] -= inv;
            for (g, &dz) in self.b2.g.iter_mut().zip(row.iter()) {
                *g += dz;
            }
        }
        gemm(h, n, v, &bt.a[..n * h], true, &bt.z[..n * v], false, 0.0, &mut self.w2.g);
        let mut da = vec![0.0f32; n * h];
        gemm(n, v, h, &bt.z[..n * v], false, &self.w2.w, true, 0.0, &mut da);
        for (d, &a) in da.iter_mut().zip(&bt.a[..n * h]) {
            *d *= 1.0 - a * a;
        }
        for i in 0..n {
            for (g, &d) in self.b1.g.iter_mut().zip(&da[i * h..(i + 1) * h]) {
                *g += d;
            }
        }
        gemm(kd, n, h, &bt.x[..n * kd], true, &da, false, 0.0, &mut self.w1.g);
        gemm(n, h, kd, &da, false, &self.w1.w, true, 0.0, &mut bt.dx[..n * kd]);
        for i in 0..n {
            for j in 0..self.k {
                let s = bt.ctx[i * self.k + j] as usize;
                let src = &bt.dx[i * kd + j * self.d..i * kd + (j + 1) * self.d];
                for (g, &d) in self.e.g[s * self.d..(s + 1) * self.d].iter_mut().zip(src) {
                    *g += d;
                }
            }
        }
        for p in [&mut self.e, &mut self.w1, &mut self.b1, &mut self.w2, &mut self.b2] {
            p.adam(lr, step);
        }
        loss
    }

    pub fn eval_nats(&self, seq: &[u32]) -> f64 {
        let chunk = 4096;
        let mut bt = self.batch(chunk);
        let mut total = 0.0;
        let ts: Vec<usize> = (0..seq.len()).collect();
        for c in ts.chunks(chunk) {
            total += self.forward(seq, c, &mut bt);
        }
        total
    }

    /// Batch-1 forward written as plain axpy loops: the cost of generating one symbol on one core.
    pub fn step_single(&self, ctx: &[u32], hbuf: &mut [f32], logits: &mut [f32]) {
        hbuf.copy_from_slice(&self.b1.w);
        for j in 0..self.k {
            let e = &self.e.w[ctx[j] as usize * self.d..(ctx[j] as usize + 1) * self.d];
            for (q, &x) in e.iter().enumerate() {
                let row = &self.w1.w[(j * self.d + q) * self.h..(j * self.d + q + 1) * self.h];
                for (hv, &w) in hbuf.iter_mut().zip(row) {
                    *hv += x * w;
                }
            }
        }
        for hv in hbuf.iter_mut() {
            *hv = hv.tanh();
        }
        logits.copy_from_slice(&self.b2.w);
        for (r, &x) in hbuf.iter().enumerate() {
            let row = &self.w2.w[r * self.v..(r + 1) * self.v];
            for (l, &w) in logits.iter_mut().zip(row) {
                *l += x * w;
            }
        }
    }

    pub fn sample(&self, prompt: &[u32], n: usize, temp: f32, seed: u64) -> Vec<u32> {
        let mut rng = Rng::new(seed);
        let mut seq = prompt.to_vec();
        let mut ctx = vec![0u32; self.k];
        let mut hbuf = vec![0.0; self.h];
        let mut logits = vec![0.0; self.v];
        for _ in 0..n {
            self.context(&seq, seq.len(), &mut ctx);
            self.step_single(&ctx, &mut hbuf, &mut logits);
            let mx = logits.iter().copied().fold(f32::MIN, f32::max);
            let p: Vec<f32> = logits.iter().map(|&l| ((l - mx) / temp).exp()).collect();
            let mut r = (rng.next() >> 40) as f32 / (1u64 << 24) as f32 * p.iter().sum::<f32>();
            let mut pick = p.len() - 1;
            for (i, &pi) in p.iter().enumerate() {
                if r < pi {
                    pick = i;
                    break;
                }
                r -= pi;
            }
            seq.push(pick as u32);
        }
        seq[prompt.len()..].to_vec()
    }

    pub fn context_of(&self, seq: &[u32], t: usize) -> Vec<u32> {
        let mut c = vec![0; self.k];
        self.context(seq, t, &mut c);
        c
    }
}

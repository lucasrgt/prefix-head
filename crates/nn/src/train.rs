use kern::Rng;

pub(crate) struct Param {
    pub w: Vec<f32>,
    pub g: Vec<f32>,
    m: Vec<f32>,
    s: Vec<f32>,
}

impl Param {
    pub fn new(n: usize, scale: f32, rng: &mut Rng) -> Self {
        let w = (0..n).map(|_| rng.f32() * 2.0 * scale).collect();
        Self { w, g: vec![0.0; n], m: vec![0.0; n], s: vec![0.0; n] }
    }

    pub fn adam(&mut self, lr: f32, t: i32) {
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
pub(crate) fn gemm(m: usize, k: usize, n: usize, a: &[f32], ta: bool, b: &[f32], tb: bool, beta: f32, c: &mut [f32]) {
    let (rsa, csa) = if ta { (1, m as isize) } else { (k as isize, 1) };
    let (rsb, csb) = if tb { (1, k as isize) } else { (n as isize, 1) };
    unsafe {
        matrixmultiply::sgemm(m, k, n, 1.0, a.as_ptr(), rsa, csa, b.as_ptr(), rsb, csb, beta, c.as_mut_ptr(), n as isize, 1);
    }
}

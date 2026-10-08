use crate::fast::gemv_rows;
use crate::train::Param;
use kern::Rng;

/// c = a·b + beta·c on strided row-major blocks: a is m×k (k×m when `ta`) with row stride `rsa`,
/// b is k×n with row stride `rsb`, c is m×n with row stride `rsc`.
fn gemm_s(m: usize, k: usize, n: usize, a: &[f32], rsa: usize, ta: bool, b: &[f32], rsb: usize, beta: f32, c: &mut [f32], rsc: usize) {
    let (ra, ca) = if ta { (1, rsa as isize) } else { (rsa as isize, 1) };
    let (rb, cb) = (rsb as isize, 1);
    unsafe {
        matrixmultiply::sgemm(m, k, n, 1.0, a.as_ptr(), ra, ca, b.as_ptr(), rb, cb, beta, c.as_mut_ptr(), rsc as isize, 1);
    }
}

/// Kaleidoscope-family (Monarch) map din → dout: `k` block-diagonal matrices din/k × mid/k, a
/// stride permutation that makes group j collect element j of every block, then `g = mid/k`
/// block-diagonal matrices k × dout/g. Every output depends on every input; MACs =
/// din·mid/k + mid·dout/g instead of din·dout.
pub struct Monarch {
    pub din: usize,
    pub mid: usize,
    pub dout: usize,
    pub k: usize,
    r: Param,
    l: Param,
    bias: Param,
}

pub struct MonBuf {
    z: Vec<f32>,
    zp: Vec<f32>,
    dz: Vec<f32>,
}

impl Monarch {
    pub fn new(din: usize, mid: usize, dout: usize, k: usize, rng: &mut Rng) -> Self {
        let g = mid / k;
        assert!(din % k == 0 && mid % k == 0 && dout % g == 0, "monarch {din}->{mid}->{dout} with k={k}");
        let r = Param::new(k * (din / k) * (mid / k), 1.0 / ((din / k) as f32).sqrt(), rng);
        let l = Param::new(g * k * (dout / g), 1.0 / (k as f32).sqrt(), rng);
        let bias = Param::new(dout, 0.0, rng);
        Self { din, mid, dout, k, r, l, bias }
    }

    fn g(&self) -> usize {
        self.mid / self.k
    }

    pub fn macs(&self) -> usize {
        self.din * self.mid / self.k + self.mid * self.dout / self.g()
    }

    pub fn params(&self) -> usize {
        self.r.w.len() + self.l.w.len() + self.bias.w.len()
    }

    pub fn bias_mut(&mut self) -> &mut [f32] {
        &mut self.bias.w
    }

    pub(crate) fn params_mut(&mut self) -> impl Iterator<Item = &mut Param> {
        [&mut self.r, &mut self.l, &mut self.bias].into_iter()
    }

    pub fn buf(&self, rows: usize) -> MonBuf {
        MonBuf { z: vec![0.0; rows * self.mid], zp: vec![0.0; rows * self.mid], dz: vec![0.0; rows * self.mid] }
    }

    /// y (rows×dout) = map(x (rows×din)); keeps the permuted middle for `backward`.
    pub fn forward(&self, x: &[f32], rows: usize, b: &mut MonBuf, y: &mut [f32]) {
        let (k, g, bi, bo) = (self.k, self.g(), self.din / self.k, self.dout / self.g());
        for i in 0..k {
            gemm_s(rows, bi, g, &x[i * bi..], self.din, false, &self.r.w[i * bi * g..(i + 1) * bi * g], g, 0.0, &mut b.z[i * g..], self.mid);
        }
        for r in 0..rows {
            let (z, zp) = (&b.z[r * self.mid..(r + 1) * self.mid], &mut b.zp[r * self.mid..(r + 1) * self.mid]);
            for i in 0..k {
                for j in 0..g {
                    zp[j * k + i] = z[i * g + j];
                }
            }
        }
        for r in 0..rows {
            y[r * self.dout..(r + 1) * self.dout].copy_from_slice(&self.bias.w);
        }
        for j in 0..g {
            gemm_s(rows, k, bo, &b.zp[j * k..], self.mid, false, &self.l.w[j * k * bo..(j + 1) * k * bo], bo, 1.0, &mut y[j * bo..], self.dout);
        }
    }

    /// Accumulates parameter grads from dy and adds dloss/dx into `dx`.
    pub fn backward(&mut self, x: &[f32], rows: usize, b: &mut MonBuf, dy: &[f32], dx: &mut [f32]) {
        let (k, g, bi, bo) = (self.k, self.g(), self.din / self.k, self.dout / self.g());
        for r in 0..rows {
            for (gb, &d) in self.bias.g.iter_mut().zip(&dy[r * self.dout..(r + 1) * self.dout]) {
                *gb += d;
            }
        }
        for j in 0..g {
            gemm_s(k, rows, bo, &b.zp[j * k..], self.mid, true, &dy[j * bo..], self.dout, 1.0, &mut self.l.g[j * k * bo..(j + 1) * k * bo], bo);
        }
        for j in 0..g {
            let mut tmp = vec![0.0f32; rows * k];
            let lw = &self.l.w[j * k * bo..(j + 1) * k * bo];
            for r in 0..rows {
                let d = &dy[r * self.dout + j * bo..r * self.dout + (j + 1) * bo];
                for i in 0..k {
                    tmp[r * k + i] = lw[i * bo..(i + 1) * bo].iter().zip(d).map(|(a, b)| a * b).sum();
                }
            }
            for r in 0..rows {
                for i in 0..k {
                    b.dz[r * self.mid + i * g + j] = tmp[r * k + i];
                }
            }
        }
        for i in 0..k {
            gemm_s(bi, rows, g, &x[i * bi..], self.din, true, &b.dz[i * g..], self.mid, 1.0, &mut self.r.g[i * bi * g..(i + 1) * bi * g], g);
            let rw = &self.r.w[i * bi * g..(i + 1) * bi * g];
            for r in 0..rows {
                let dzr = &b.dz[r * self.mid + i * g..r * self.mid + (i + 1) * g];
                for p in 0..bi {
                    dx[r * self.din + i * bi + p] += rw[p * g..(p + 1) * g].iter().zip(dzr).map(|(a, b)| a * b).sum::<f32>();
                }
            }
        }
    }

    /// One row on one core, SIMD per block; `z`/`zp` are mid-sized scratch.
    pub fn apply(&self, x: &[f32], z: &mut [f32], zp: &mut [f32], y: &mut [f32]) {
        let (k, g, bi, bo) = (self.k, self.g(), self.din / self.k, self.dout / self.g());
        z.fill(0.0);
        for i in 0..k {
            gemv_rows(&x[i * bi..(i + 1) * bi], &self.r.w[i * bi * g..(i + 1) * bi * g], g, &mut z[i * g..(i + 1) * g]);
        }
        for i in 0..k {
            for j in 0..g {
                zp[j * k + i] = z[i * g + j];
            }
        }
        y.copy_from_slice(&self.bias.w);
        for j in 0..g {
            gemv_rows(&zp[j * k..(j + 1) * k], &self.l.w[j * k * bo..(j + 1) * k * bo], bo, &mut y[j * bo..(j + 1) * bo]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loss(m: &Monarch, x: &[f32], rows: usize, w: &[f32]) -> f64 {
        let mut b = m.buf(rows);
        let mut y = vec![0.0; rows * m.dout];
        m.forward(x, rows, &mut b, &mut y);
        y.iter().zip(w).map(|(a, b)| (a * b) as f64).sum()
    }

    #[test]
    fn gradients_match_and_apply_matches_forward() {
        let mut rng = Rng::new(4);
        let (din, mid, dout, k, rows) = (12, 8, 6, 4, 3);
        let mut m = Monarch::new(din, mid, dout, k, &mut rng);
        m.bias.w.iter_mut().for_each(|b| *b = rng.f32());
        let x: Vec<f32> = (0..rows * din).map(|_| rng.f32()).collect();
        let w: Vec<f32> = (0..rows * dout).map(|_| rng.f32()).collect();
        let mut b = m.buf(rows);
        let mut y = vec![0.0; rows * dout];
        m.forward(&x, rows, &mut b, &mut y);
        let (mut z, mut zp, mut y1) = (vec![0.0; mid], vec![0.0; mid], vec![0.0; dout]);
        m.apply(&x[din..2 * din], &mut z, &mut zp, &mut y1);
        for (a, b) in y1.iter().zip(&y[dout..2 * dout]) {
            assert!((a - b).abs() < 1e-5);
        }
        let mut dx = vec![0.0; rows * din];
        m.backward(&x, rows, &mut b, &w, &mut dx);
        let eps = 1e-2f32;
        for i in 0..x.len() {
            let mut xp = x.clone();
            xp[i] += eps;
            let mut xm = x.clone();
            xm[i] -= eps;
            let num = ((loss(&m, &xp, rows, &w) - loss(&m, &xm, rows, &w)) / (2.0 * eps as f64)) as f32;
            assert!((num - dx[i]).abs() < 1e-3, "dx[{i}] {num} vs {}", dx[i]);
        }
        let grads: Vec<Vec<f32>> = m.params_mut().map(|p| p.g.clone()).collect();
        for pi in 0..3 {
            for idx in 0..grads[pi].len() {
                let orig = m.params_mut().nth(pi).unwrap().w[idx];
                m.params_mut().nth(pi).unwrap().w[idx] = orig + eps;
                let lp = loss(&m, &x, rows, &w);
                m.params_mut().nth(pi).unwrap().w[idx] = orig - eps;
                let lm = loss(&m, &x, rows, &w);
                m.params_mut().nth(pi).unwrap().w[idx] = orig;
                let num = ((lp - lm) / (2.0 * eps as f64)) as f32;
                assert!((num - grads[pi][idx]).abs() < 1e-3, "param {pi}[{idx}] {num} vs {}", grads[pi][idx]);
            }
        }
        assert_eq!(m.macs(), din * mid / k + mid * dout / (mid / k));
    }
}

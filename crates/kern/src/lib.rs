pub mod dense;
pub mod ternary;

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    pub fn ternary(&mut self) -> i8 {
        (self.next() % 3) as i8 - 1
    }

    pub fn i8(&mut self) -> i8 {
        self.next() as i8
    }

    pub fn f32(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::dense::*;
    use super::ternary::*;
    use super::Rng;

    fn shapes() -> [(usize, usize); 4] {
        [(1, 512), (7, 1024), (33, 2048), (64, 4096)]
    }

    #[test]
    fn ternary_kernels_match_reference() {
        let mut rng = Rng::new(42);
        for (rows, cols) in shapes() {
            let w: Vec<i8> = (0..rows * cols).map(|_| rng.ternary()).collect();
            let x: Vec<i8> = (0..cols).map(|_| rng.i8()).collect();
            let t = Ternary::pack(&w, rows, cols);
            let a = Act::new(&x);
            let mut want = vec![0; rows];
            ternary_ref(&t, &a, &mut want);
            let naive: Vec<i32> = (0..rows)
                .map(|r| (0..cols).map(|c| w[r * cols + c] as i32 * x[c] as i32).sum())
                .collect();
            assert_eq!(want, naive);

            let mut got = vec![0; rows];
            if std::is_x86_feature_detected!("avx2") {
                unsafe { ternary_avx2(&t, &a, &mut got) };
                assert_eq!(got, want, "avx2 {rows}x{cols}");
            }
            if has_avx512_vnni() {
                unsafe { ternary_vnni(&t, &a, &mut got) };
                assert_eq!(got, want, "vnni {rows}x{cols}");
                unsafe { ternary_vnni_r4(&t, &a, &mut got) };
                assert_eq!(got, want, "vnni_r4 {rows}x{cols}");
                unsafe { ternary_sad(&t, &a, &mut got) };
                assert_eq!(got, want, "sad {rows}x{cols}");
            }

            let mut want_sign = vec![0; rows];
            sign_ref(&t, &a, &mut want_sign);
            unsafe { sign_popcnt(&t, &a, &mut got) };
            assert_eq!(got, want_sign, "popcnt {rows}x{cols}");
            if has_avx512_vpopcnt() {
                unsafe { sign_vpopcnt(&t, &a, &mut got) };
                assert_eq!(got, want_sign, "vpopcnt {rows}x{cols}");
            }
        }
    }

    #[test]
    fn dense_kernels_match_reference() {
        let mut rng = Rng::new(7);
        for (rows, cols) in shapes() {
            let w: Vec<i8> = (0..rows * cols).map(|_| rng.i8()).collect();
            let x: Vec<i8> = (0..cols).map(|_| rng.i8()).collect();
            let d = DenseI8::new(w, rows, cols);
            let a = Act::new(&x);
            let mut want = vec![0; rows];
            i8_ref(&d, &a, &mut want);
            if has_avx512_vnni() {
                let mut got = vec![0; rows];
                unsafe { i8_vnni(&d, &a, &mut got) };
                assert_eq!(got, want, "i8_vnni {rows}x{cols}");
            }

            let wf: Vec<f32> = (0..rows * cols).map(|_| rng.f32()).collect();
            let xf: Vec<f32> = (0..cols).map(|_| rng.f32()).collect();
            let mut want = vec![0.0; rows];
            f32_ref(&wf, rows, cols, &xf, &mut want);
            if std::is_x86_feature_detected!("avx512f") {
                let mut got = vec![0.0; rows];
                unsafe { f32_avx512(&wf, rows, cols, &xf, &mut got) };
                for (g, w) in got.iter().zip(&want) {
                    assert!((g - w).abs() <= 1e-3 * (1.0 + w.abs()), "f32 {g} vs {w}");
                }
            }
        }
    }
}

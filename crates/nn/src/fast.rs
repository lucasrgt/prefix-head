use std::arch::x86_64::*;

/// y += xᵀ·W, with W row-major len(x) × n. Runtime dispatch: AVX-512, AVX2+FMA, scalar.
pub fn gemv_rows(x: &[f32], w: &[f32], n: usize, y: &mut [f32]) {
    debug_assert_eq!(w.len(), x.len() * n);
    debug_assert_eq!(y.len(), n);
    if is_x86_feature_detected!("avx512f") {
        unsafe { gemv_rows_avx512(x, w, n, y) }
    } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { gemv_rows_avx2(x, w, n, y) }
    } else {
        gemv_rows_scalar(x, w, n, y)
    }
}

pub fn gemv_rows_scalar(x: &[f32], w: &[f32], n: usize, y: &mut [f32]) {
    for (i, &xv) in x.iter().enumerate() {
        for (yy, &ww) in y.iter_mut().zip(&w[i * n..(i + 1) * n]) {
            *yy += xv * ww;
        }
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_rows_avx2(x: &[f32], w: &[f32], n: usize, y: &mut [f32]) {
    for (i, &xv) in x.iter().enumerate() {
        for (yy, &ww) in y.iter_mut().zip(&w[i * n..(i + 1) * n]) {
            *yy = xv.mul_add(ww, *yy);
        }
    }
}

/// One block of up to 16·R columns kept in R zmm accumulators for the whole pass over x.
#[target_feature(enable = "avx512f")]
unsafe fn block<const R: usize>(x: &[f32], w: *const f32, n: usize, y: *mut f32, width: usize) {
    unsafe {
        let rem = width - (R - 1) * 16;
        let last: __mmask16 = if rem == 16 { 0xffff } else { (1u16 << rem) - 1 };
        let mask = |r: usize| if r == R - 1 { last } else { 0xffff };
        let mut acc = [_mm512_setzero_ps(); R];
        for r in 0..R {
            acc[r] = _mm512_maskz_loadu_ps(mask(r), y.add(r * 16));
        }
        for (i, &xv) in x.iter().enumerate() {
            let b = _mm512_set1_ps(xv);
            let row = w.add(i * n);
            for r in 0..R {
                acc[r] = _mm512_fmadd_ps(b, _mm512_maskz_loadu_ps(mask(r), row.add(r * 16)), acc[r]);
            }
        }
        for r in 0..R {
            _mm512_mask_storeu_ps(y.add(r * 16), mask(r), acc[r]);
        }
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn gemv_rows_avx512(x: &[f32], w: &[f32], n: usize, y: &mut [f32]) {
    let mut c = 0;
    while c < n {
        let width = (n - c).min(256);
        let (wp, yp) = unsafe { (w.as_ptr().add(c), y.as_mut_ptr().add(c)) };
        macro_rules! go {
            ($($r:literal)*) => {
                match width.div_ceil(16) {
                    $($r => unsafe { block::<$r>(x, wp, n, yp, width) },)*
                    _ => unreachable!(),
                }
            };
        }
        go!(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16);
        c += 256;
    }
}

/// tanh = (e^{2x} − 1)/(e^{2x} + 1) with `exp_fast`; absolute error < 2e-6 everywhere.
#[inline(always)]
fn tanh_fast(x: f32) -> f32 {
    let e = exp_fast(2.0 * x.clamp(-10.0, 10.0));
    (e - 1.0) / (e + 1.0)
}

pub fn tanh_inplace(v: &mut [f32]) {
    if is_x86_feature_detected!("avx512f") {
        unsafe { tanh_avx512(v) }
    } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { tanh_avx2(v) }
    } else {
        v.iter_mut().for_each(|a| *a = tanh_fast(*a));
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn tanh_avx512(v: &mut [f32]) {
    v.iter_mut().for_each(|a| *a = tanh_fast(*a));
}

#[target_feature(enable = "avx2,fma")]
unsafe fn tanh_avx2(v: &mut [f32]) {
    v.iter_mut().for_each(|a| *a = tanh_fast(*a));
}

/// v ← e^v, with the same runtime dispatch as `tanh_inplace`.
pub fn exp_inplace(v: &mut [f32]) {
    if is_x86_feature_detected!("avx512f") {
        unsafe { exp_avx512(v) }
    } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { exp_avx2(v) }
    } else {
        v.iter_mut().for_each(|a| *a = exp_fast(*a));
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn exp_avx512(v: &mut [f32]) {
    v.iter_mut().for_each(|a| *a = exp_fast(*a));
}

#[target_feature(enable = "avx2,fma")]
unsafe fn exp_avx2(v: &mut [f32]) {
    v.iter_mut().for_each(|a| *a = exp_fast(*a));
}

/// e^x via 2^n · 2^f, n = round(x·log₂e), f ∈ [−½, ½], degree-6 Taylor for 2^f; relative error < 3e-7.
#[inline(always)]
fn exp_fast(x: f32) -> f32 {
    let t = (x * std::f32::consts::LOG2_E).clamp(-126.0, 126.0);
    let n = (t + 0.5).floor();
    let f = t - n;
    let p = 1.0 + f * (6.931471806e-01 + f * (2.402265070e-01 + f * (5.550410866e-02 + f * (9.618129108e-03 + f * (1.333355815e-03 + f * 1.540353039e-04)))));
    p * f32::from_bits(((n as i32 + 127) << 23) as u32)
}

/// ln Σ e^v — the softmax normaliser every generated factor pays for.
pub fn logsumexp(v: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx512f") {
        unsafe { logsumexp_avx512(v) }
    } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { logsumexp_avx2(v) }
    } else {
        logsumexp_scalar(v)
    }
}

fn logsumexp_scalar(v: &[f32]) -> f32 {
    let mx = v.iter().copied().fold(f32::MIN, f32::max);
    mx + v.iter().map(|&a| exp_fast(a - mx)).sum::<f32>().ln()
}

#[target_feature(enable = "avx512f")]
unsafe fn logsumexp_avx512(v: &[f32]) -> f32 {
    logsumexp_lanes(v)
}

#[target_feature(enable = "avx2,fma")]
unsafe fn logsumexp_avx2(v: &[f32]) -> f32 {
    logsumexp_lanes(v)
}

/// Lane-wise max and sum so the compiler can vectorise without reassociating floats.
#[inline(always)]
fn logsumexp_lanes(v: &[f32]) -> f32 {
    const L: usize = 16;
    let mut mx = [f32::MIN; L];
    let chunks = v.chunks_exact(L);
    let tail = chunks.remainder();
    for c in chunks.clone() {
        for i in 0..L {
            mx[i] = mx[i].max(c[i]);
        }
    }
    let m = tail.iter().copied().fold(mx.iter().copied().fold(f32::MIN, f32::max), f32::max);
    let mut s = [0.0f32; L];
    for c in chunks {
        for i in 0..L {
            s[i] += exp_fast(c[i] - m);
        }
    }
    let total = s.iter().sum::<f32>() + tail.iter().map(|&a| exp_fast(a - m)).sum::<f32>();
    m + total.ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kern::Rng;

    #[test]
    fn gemv_matches_scalar_for_every_width() {
        let mut rng = Rng::new(3);
        for &(m, n) in &[(1, 1), (5, 15), (7, 16), (9, 17), (160, 256), (33, 300), (256, 91), (256, 8192), (3, 513)] {
            let x: Vec<f32> = (0..m).map(|_| rng.f32()).collect();
            let w: Vec<f32> = (0..m * n).map(|_| rng.f32()).collect();
            let y0: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
            let mut want = y0.clone();
            gemv_rows_scalar(&x, &w, n, &mut want);
            let mut got = y0.clone();
            gemv_rows(&x, &w, n, &mut got);
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() <= 1e-4 * (1.0 + w.abs()), "{m}x{n}: {g} vs {w}");
            }
        }
    }

    #[test]
    fn tanh_and_exp_are_accurate() {
        for i in -1000..=1000 {
            let x = i as f32 / 100.0;
            assert!((tanh_fast(x) - x.tanh()).abs() < 2e-6, "tanh {x}");
            let e = exp_fast(x);
            assert!((e - x.exp()).abs() <= 1e-6 * x.exp().max(1.0), "exp {x}: {e} vs {}", x.exp());
        }
        let v: Vec<f32> = (0..8193).map(|i| ((i * 37) % 101) as f32 / 7.0 - 5.0).collect();
        let want = {
            let m = v.iter().copied().fold(f32::MIN, f32::max);
            m + v.iter().map(|&a| (a - m).exp()).sum::<f32>().ln()
        };
        assert!((logsumexp(&v) - want).abs() < 1e-4);
    }
}

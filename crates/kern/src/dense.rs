use crate::ternary::Act;
use std::arch::x86_64::*;

pub struct DenseI8 {
    pub rows: usize,
    pub cols: usize,
    pub w: Vec<i8>,
    pub row_sum: Vec<i32>,
}

impl DenseI8 {
    pub fn new(w: Vec<i8>, rows: usize, cols: usize) -> Self {
        assert_eq!(cols % 64, 0);
        assert_eq!(w.len(), rows * cols);
        let row_sum = w.chunks(cols).map(|r| r.iter().map(|&v| v as i32).sum()).collect();
        Self { rows, cols, w, row_sum }
    }
}

pub fn i8_ref(d: &DenseI8, a: &Act, y: &mut [i32]) {
    for (r, row) in d.w.chunks(d.cols).enumerate() {
        y[r] = row.iter().zip(&a.x).map(|(&w, &x)| w as i32 * x as i32).sum();
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub unsafe fn i8_vnni(d: &DenseI8, a: &Act, y: &mut [i32]) {
    let xp = a.xu.as_ptr();
    for r in 0..d.rows {
        unsafe {
            let wp = d.w.as_ptr().add(r * d.cols);
            let mut acc = [_mm512_setzero_si512(); 2];
            for c in (0..d.cols).step_by(128) {
                for k in 0..2 {
                    let o = c + k * 64;
                    if o < d.cols {
                        let x = _mm512_loadu_si512(xp.add(o).cast());
                        let w = _mm512_loadu_si512(wp.add(o).cast());
                        acc[k] = _mm512_dpbusd_epi32(acc[k], x, w);
                    }
                }
            }
            y[r] = _mm512_reduce_add_epi32(_mm512_add_epi32(acc[0], acc[1])) - 128 * d.row_sum[r];
        }
    }
}

pub fn f32_ref(w: &[f32], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    for r in 0..rows {
        y[r] = w[r * cols..(r + 1) * cols].iter().zip(x).map(|(a, b)| a * b).sum();
    }
}

#[target_feature(enable = "avx512f")]
pub unsafe fn f32_avx512(w: &[f32], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    assert_eq!(cols % 64, 0);
    for r in 0..rows {
        unsafe {
            let wp = w.as_ptr().add(r * cols);
            let mut acc = [_mm512_setzero_ps(); 4];
            for c in (0..cols).step_by(64) {
                for k in 0..4 {
                    let o = c + k * 16;
                    acc[k] = _mm512_fmadd_ps(_mm512_loadu_ps(wp.add(o)), _mm512_loadu_ps(x.as_ptr().add(o)), acc[k]);
                }
            }
            let s = _mm512_add_ps(_mm512_add_ps(acc[0], acc[1]), _mm512_add_ps(acc[2], acc[3]));
            y[r] = _mm512_reduce_add_ps(s);
        }
    }
}

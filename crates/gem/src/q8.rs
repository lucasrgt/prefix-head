//! Q8_0 matrices repacked as contiguous int8 rows plus one f32 scale per 32 weights, and matvec
//! against an activation quantized the same way (as llama.cpp does for Q8_0 on the CPU).

use crate::pool::Pool;
use std::arch::x86_64::*;

pub struct Mat {
    pub q: Vec<i8>,
    pub s: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

pub fn f16(x: u16) -> f32 {
    let sign = if x & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((x >> 10) & 0x1f) as i32;
    let mant = (x & 0x3ff) as f32;
    match exp {
        0 => sign * mant * 2f32.powi(-24),
        31 => sign * f32::INFINITY,
        _ => sign * (1.0 + mant / 1024.0) * 2f32.powi(exp - 15),
    }
}

impl Mat {
    pub fn from_q8_0(raw: &[u8], rows: usize, cols: usize) -> Self {
        let nb = rows * cols / 32;
        assert_eq!(raw.len(), nb * 34);
        let mut q = Vec::with_capacity(rows * cols);
        let mut s = Vec::with_capacity(nb);
        for blk in raw.chunks_exact(34) {
            s.push(f16(u16::from_le_bytes([blk[0], blk[1]])));
            q.extend(blk[2..].iter().map(|&b| b as i8));
        }
        Self { q, s, rows, cols }
    }

    /// Rows `ids` gathered into a new matrix, in that order.
    pub fn gather(&self, ids: &[u32]) -> Self {
        let (c, b) = (self.cols, self.cols / 32);
        let mut q = Vec::with_capacity(ids.len() * c);
        let mut s = Vec::with_capacity(ids.len() * b);
        for &r in ids {
            let r = r as usize;
            q.extend_from_slice(&self.q[r * c..(r + 1) * c]);
            s.extend_from_slice(&self.s[r * b..(r + 1) * b]);
        }
        Self { q, s, rows: ids.len(), cols: c }
    }

    pub fn row_f32(&self, r: usize) -> Vec<f32> {
        let c = self.cols;
        (0..c).map(|j| self.q[r * c + j] as f32 * self.s[(r * c + j) / 32]).collect()
    }

    pub fn bytes(&self) -> usize {
        self.q.len() + self.s.len() * 4
    }
}

pub struct Act {
    pub q: Vec<i8>,
    pub s: Vec<f32>,
}

impl Act {
    pub fn new(n: usize) -> Self {
        Self { q: vec![0; n], s: vec![0.0; n / 32] }
    }

    pub fn quantize(&mut self, x: &[f32]) {
        for (b, chunk) in x.chunks_exact(32).enumerate() {
            let amax = chunk.iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            self.s[b] = d;
            for (q, v) in self.q[b * 32..(b + 1) * 32].iter_mut().zip(chunk) {
                *q = (v * inv).round() as i8;
            }
        }
    }
}

fn dot_scalar(wq: &[i8], ws: &[f32], a: &Act) -> f32 {
    let mut acc = 0f32;
    for (b, (wb, ab)) in wq.chunks_exact(32).zip(a.q.chunks_exact(32)).enumerate() {
        let s: i32 = wb.iter().zip(ab).map(|(&w, &x)| w as i32 * x as i32).sum();
        acc += s as f32 * ws[b] * a.s[b];
    }
    acc
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn dot_avx512(wq: &[i8], ws: &[f32], a: &Act) -> f32 {
    unsafe {
        let mut acc = _mm512_setzero_ps();
        let nb = wq.len() / 32;
        for b in 0..nb {
            let w = _mm512_cvtepi8_epi16(_mm256_loadu_si256(wq.as_ptr().add(b * 32) as *const __m256i));
            let x = _mm512_cvtepi8_epi16(_mm256_loadu_si256(a.q.as_ptr().add(b * 32) as *const __m256i));
            let p = _mm512_cvtepi32_ps(_mm512_madd_epi16(w, x));
            acc = _mm512_fmadd_ps(p, _mm512_set1_ps(*ws.get_unchecked(b) * *a.s.get_unchecked(b)), acc);
        }
        _mm512_reduce_add_ps(acc)
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(wq: &[i8], ws: &[f32], a: &Act) -> f32 {
    unsafe {
        let mut acc = _mm256_setzero_ps();
        let nb = wq.len() / 32;
        for b in 0..nb {
            for h in 0..2 {
                let o = b * 32 + h * 16;
                let w = _mm256_cvtepi8_epi16(_mm_loadu_si128(wq.as_ptr().add(o) as *const __m128i));
                let x = _mm256_cvtepi8_epi16(_mm_loadu_si128(a.q.as_ptr().add(o) as *const __m128i));
                let p = _mm256_cvtepi32_ps(_mm256_madd_epi16(w, x));
                acc = _mm256_fmadd_ps(p, _mm256_set1_ps(*ws.get_unchecked(b) * *a.s.get_unchecked(b)), acc);
            }
        }
        let lo = _mm256_castps256_ps128(acc);
        let hi = _mm256_extractf128_ps(acc, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_hadd_ps(s, s);
        _mm_cvtss_f32(_mm_hadd_ps(s, s))
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum Isa {
    Avx512,
    Avx2,
    Scalar,
}

pub fn detect() -> Isa {
    match std::env::var("GEM_ISA").as_deref() {
        Ok("avx2") => return Isa::Avx2,
        Ok("scalar") => return Isa::Scalar,
        _ => {}
    }
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
        Isa::Avx512
    } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        Isa::Avx2
    } else {
        Isa::Scalar
    }
}

fn rows(m: &Mat, a: &Act, isa: Isa, r0: usize, out: &mut [f32]) {
    let (c, b) = (m.cols, m.cols / 32);
    for (i, y) in out.iter_mut().enumerate() {
        let r = r0 + i;
        let (wq, ws) = (&m.q[r * c..(r + 1) * c], &m.s[r * b..(r + 1) * b]);
        *y = match isa {
            Isa::Avx512 => unsafe { dot_avx512(wq, ws, a) },
            Isa::Avx2 => unsafe { dot_avx2(wq, ws, a) },
            Isa::Scalar => dot_scalar(wq, ws, a),
        };
    }
}

struct SendPtr(*mut f32);
unsafe impl Sync for SendPtr {}

/// y = M·x over the pool; rows split into one contiguous range per thread.
pub fn matvec(pool: &Pool, isa: Isa, m: &Mat, a: &Act, y: &mut [f32]) {
    assert_eq!(y.len(), m.rows);
    let n = pool.threads();
    if n == 1 || m.rows < 64 {
        rows(m, a, isa, 0, y);
        return;
    }
    let per = m.rows.div_ceil(n);
    let out = SendPtr(y.as_mut_ptr());
    pool.run(&|t| {
        let r0 = (t * per).min(m.rows);
        let r1 = ((t + 1) * per).min(m.rows);
        let o = &out;
        let part = unsafe { std::slice::from_raw_parts_mut(o.0.add(r0), r1 - r0) };
        rows(m, a, isa, r0, part);
    });
}

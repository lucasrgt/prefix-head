//! Q8_0 matrices repacked as contiguous int8 rows plus one f32 scale per 32 weights, and matvec
//! against an activation quantized the same way (as llama.cpp does for Q8_0 on the CPU). A matrix can
//! be requantized to 4 bits: per 32 weights, 16 bytes holding w[j] + 8 in the low nibble of byte j
//! and w[j + 16] + 8 in the high nibble.

use crate::pool::Pool;
use std::arch::x86_64::*;

pub struct Mat {
    pub q: Vec<i8>,
    pub s: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub bits: u8,
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
        Self { q, s, rows, cols, bits: 8 }
    }

    /// GGUF Q4_0: per 32 weights an f16 scale and 16 bytes in the layout `to_q4` produces.
    pub fn from_q4_0(raw: &[u8], rows: usize, cols: usize) -> Self {
        let nb = rows * cols / 32;
        assert_eq!(raw.len(), nb * 18);
        let mut q = Vec::with_capacity(rows * cols / 2);
        let mut s = Vec::with_capacity(nb);
        for blk in raw.chunks_exact(18) {
            s.push(f16(u16::from_le_bytes([blk[0], blk[1]])));
            q.extend(blk[2..].iter().map(|&b| b as i8));
        }
        Self { q, s, rows, cols, bits: 4 }
    }

    /// GGUF F16, kept as raw half floats and multiplied against the unquantized activation.
    pub fn from_f16(raw: &[u8], rows: usize, cols: usize) -> Self {
        assert_eq!(raw.len(), rows * cols * 2);
        Self { q: raw.iter().map(|&b| b as i8).collect(), s: Vec::new(), rows, cols, bits: 16 }
    }

    /// Rows `ids` gathered into a new matrix, in that order.
    pub fn gather(&self, ids: &[u32]) -> Self {
        let (c, b) = (self.row_bytes(), self.cols / 32);
        let mut q = Vec::with_capacity(ids.len() * c);
        let mut s = Vec::with_capacity(ids.len() * b);
        for &r in ids {
            let r = r as usize;
            q.extend_from_slice(&self.q[r * c..(r + 1) * c]);
            if !self.s.is_empty() {
                s.extend_from_slice(&self.s[r * b..(r + 1) * b]);
            }
        }
        Self { q, s, rows: ids.len(), cols: self.cols, bits: self.bits }
    }

    fn row_bytes(&self) -> usize {
        self.cols * self.bits as usize / 8
    }

    fn weight(&self, r: usize, j: usize) -> f32 {
        let c = self.cols;
        if self.bits == 16 {
            let i = (r * c + j) * 2;
            return f16(u16::from_le_bytes([self.q[i] as u8, self.q[i + 1] as u8]));
        }
        let scale = self.s[(r * c + j) / 32];
        if self.bits == 8 {
            return self.q[r * c + j] as f32 * scale;
        }
        let (blk, k) = (j / 32, j % 32);
        let byte = self.q[r * c / 2 + blk * 16 + k % 16] as u8;
        let nib = if k < 16 { byte & 15 } else { byte >> 4 };
        (nib as i32 - 8) as f32 * scale
    }

    /// F16 weights at 8 bits: per 32 weights, scale = max |w| / 127.
    pub fn to_q8(&self) -> Self {
        if self.bits == 8 {
            return Self { q: self.q.clone(), s: self.s.clone(), rows: self.rows, cols: self.cols, bits: 8 };
        }
        assert_eq!(self.bits, 16);
        let mut q = Vec::with_capacity(self.rows * self.cols);
        let mut s = Vec::with_capacity(self.rows * self.cols / 32);
        for blk in self.q.chunks_exact(64) {
            let w: Vec<f32> = blk.chunks_exact(2).map(|b| f16(u16::from_le_bytes([b[0] as u8, b[1] as u8]))).collect();
            let amax = w.iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            q.extend(w.iter().map(|v| (v * inv).round() as i8));
            s.push(d);
        }
        Self { q, s, rows: self.rows, cols: self.cols, bits: 8 }
    }

    /// The same weights at 4 bits: per 32 weights, scale = max |w| / 7 and round(w / scale) in [-8, 7].
    pub fn to_q4(&self) -> Self {
        if self.bits == 16 {
            return self.to_q8().to_q4();
        }
        assert_eq!(self.bits, 8);
        let mut q = Vec::with_capacity(self.q.len() / 2);
        let mut s = Vec::with_capacity(self.s.len());
        for (blk, &scale) in self.q.chunks_exact(32).zip(&self.s) {
            let w: Vec<f32> = blk.iter().map(|&x| x as f32 * scale).collect();
            let amax = w.iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = amax / 7.0;
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            let n: Vec<u8> = w.iter().map(|v| ((v * inv).round().clamp(-8.0, 7.0) as i32 + 8) as u8).collect();
            q.extend((0..16).map(|j| (n[j] | (n[j + 16] << 4)) as i8));
            s.push(d);
        }
        Self { q, s, rows: self.rows, cols: self.cols, bits: 4 }
    }

    pub fn row_f32(&self, r: usize) -> Vec<f32> {
        (0..self.cols).map(|j| self.weight(r, j)).collect()
    }

    pub fn bytes(&self) -> usize {
        self.q.len() + self.s.len() * 4
    }
}

pub struct Act {
    pub q: Vec<i8>,
    pub s: Vec<f32>,
    pub x: Vec<f32>,
}

impl Act {
    pub fn new(n: usize) -> Self {
        Self { q: vec![0; n], s: vec![0.0; n / 32], x: vec![0.0; n] }
    }

    pub fn quantize(&mut self, x: &[f32], isa: Isa) {
        self.x.copy_from_slice(x);
        match isa {
            Isa::Avx512 => return unsafe { self.quantize_avx512(x) },
            Isa::Avx2 => return unsafe { self.quantize_avx2(x) },
            Isa::Scalar => {}
        }
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

impl Act {
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn quantize_avx512(&mut self, x: &[f32]) {
        unsafe {
            for b in 0..x.len() / 32 {
                let lo = _mm512_loadu_ps(x.as_ptr().add(b * 32));
                let hi = _mm512_loadu_ps(x.as_ptr().add(b * 32 + 16));
                let amax = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(lo), _mm512_abs_ps(hi)));
                let d = amax / 127.0;
                let inv = _mm512_set1_ps(if d > 0.0 { 1.0 / d } else { 0.0 });
                *self.s.get_unchecked_mut(b) = d;
                let out = self.q.as_mut_ptr().add(b * 32) as *mut __m128i;
                _mm_storeu_si128(out, _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(lo, inv))));
                _mm_storeu_si128(out.add(1), _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(hi, inv))));
            }
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn quantize_avx2(&mut self, x: &[f32]) {
        unsafe {
            let sign = _mm256_set1_ps(-0.0);
            for b in 0..x.len() / 32 {
                let v: [__m256; 4] = std::array::from_fn(|i| _mm256_loadu_ps(x.as_ptr().add(b * 32 + i * 8)));
                let m = v.iter().fold(_mm256_setzero_ps(), |m, &a| _mm256_max_ps(m, _mm256_andnot_ps(sign, a)));
                let mut lanes = [0f32; 8];
                _mm256_storeu_ps(lanes.as_mut_ptr(), m);
                let d = lanes.iter().fold(0f32, |a, &c| a.max(c)) / 127.0;
                let inv = _mm256_set1_ps(if d > 0.0 { 1.0 / d } else { 0.0 });
                *self.s.get_unchecked_mut(b) = d;
                let mut ints = [0i32; 32];
                for (i, &a) in v.iter().enumerate() {
                    _mm256_storeu_si256(ints.as_mut_ptr().add(i * 8) as *mut __m256i, _mm256_cvtps_epi32(_mm256_mul_ps(a, inv)));
                }
                for (q, &i) in self.q[b * 32..(b + 1) * 32].iter_mut().zip(&ints) {
                    *q = i as i8;
                }
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

fn dot4_scalar(wq: &[i8], ws: &[f32], a: &Act) -> f32 {
    let mut acc = 0f32;
    for (b, (wb, ab)) in wq.chunks_exact(16).zip(a.q.chunks_exact(32)).enumerate() {
        let mut s = 0i32;
        for j in 0..16 {
            let byte = wb[j] as u8;
            s += ((byte & 15) as i32 - 8) * ab[j] as i32 + ((byte >> 4) as i32 - 8) * ab[j + 16] as i32;
        }
        acc += s as f32 * ws[b] * a.s[b];
    }
    acc
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn dot4_avx512(wq: &[i8], ws: &[f32], a: &Act) -> f32 {
    unsafe {
        let mut acc = _mm512_setzero_ps();
        let mask = _mm_set1_epi8(15);
        let eight = _mm256_set1_epi8(8);
        let nb = wq.len() / 16;
        for b in 0..nb {
            let raw = _mm_loadu_si128(wq.as_ptr().add(b * 16) as *const __m128i);
            let lo = _mm_and_si128(raw, mask);
            let hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask);
            let w8 = _mm256_sub_epi8(_mm256_set_m128i(hi, lo), eight);
            let w = _mm512_cvtepi8_epi16(w8);
            let x = _mm512_cvtepi8_epi16(_mm256_loadu_si256(a.q.as_ptr().add(b * 32) as *const __m256i));
            let p = _mm512_cvtepi32_ps(_mm512_madd_epi16(w, x));
            acc = _mm512_fmadd_ps(p, _mm512_set1_ps(*ws.get_unchecked(b) * *a.s.get_unchecked(b)), acc);
        }
        _mm512_reduce_add_ps(acc)
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dot4_avx2(wq: &[i8], ws: &[f32], a: &Act) -> f32 {
    unsafe {
        let mut acc = _mm256_setzero_ps();
        let mask = _mm_set1_epi8(15);
        let eight = _mm_set1_epi8(8);
        let nb = wq.len() / 16;
        for b in 0..nb {
            let raw = _mm_loadu_si128(wq.as_ptr().add(b * 16) as *const __m128i);
            let lo = _mm_sub_epi8(_mm_and_si128(raw, mask), eight);
            let hi = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(raw, 4), mask), eight);
            let scale = _mm256_set1_ps(*ws.get_unchecked(b) * *a.s.get_unchecked(b));
            for (h, half) in [lo, hi].into_iter().enumerate() {
                let w = _mm256_cvtepi8_epi16(half);
                let x = _mm256_cvtepi8_epi16(_mm_loadu_si128(a.q.as_ptr().add(b * 32 + h * 16) as *const __m128i));
                acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(w, x)), scale, acc);
            }
        }
        let lo = _mm256_castps256_ps128(acc);
        let hi = _mm256_extractf128_ps(acc, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_hadd_ps(s, s);
        _mm_cvtss_f32(_mm_hadd_ps(s, s))
    }
}

fn dot16_scalar(wq: &[i8], a: &Act) -> f32 {
    wq.chunks_exact(2).zip(&a.x).map(|(b, x)| f16(u16::from_le_bytes([b[0] as u8, b[1] as u8])) * x).sum()
}

#[target_feature(enable = "avx512f,f16c")]
unsafe fn dot16_avx512(wq: &[i8], a: &Act) -> f32 {
    unsafe {
        let mut acc = _mm512_setzero_ps();
        let n = a.x.len() / 16;
        for i in 0..n {
            let w = _mm512_cvtph_ps(_mm256_loadu_si256(wq.as_ptr().add(i * 32) as *const __m256i));
            acc = _mm512_fmadd_ps(w, _mm512_loadu_ps(a.x.as_ptr().add(i * 16)), acc);
        }
        _mm512_reduce_add_ps(acc)
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot16_avx2(wq: &[i8], a: &Act) -> f32 {
    unsafe {
        let mut acc = _mm256_setzero_ps();
        let n = a.x.len() / 8;
        for i in 0..n {
            let w = _mm256_cvtph_ps(_mm_loadu_si128(wq.as_ptr().add(i * 16) as *const __m128i));
            acc = _mm256_fmadd_ps(w, _mm256_loadu_ps(a.x.as_ptr().add(i * 8)), acc);
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
    let (c, b) = (m.row_bytes(), m.cols / 32);
    for (i, y) in out.iter_mut().enumerate() {
        let r = r0 + i;
        let wq = &m.q[r * c..(r + 1) * c];
        if m.bits == 16 {
            *y = match isa {
                Isa::Avx512 => unsafe { dot16_avx512(wq, a) },
                Isa::Avx2 => unsafe { dot16_avx2(wq, a) },
                Isa::Scalar => dot16_scalar(wq, a),
            };
            continue;
        }
        let ws = &m.s[r * b..(r + 1) * b];
        *y = match (m.bits, isa) {
            (8, Isa::Avx512) => unsafe { dot_avx512(wq, ws, a) },
            (8, Isa::Avx2) => unsafe { dot_avx2(wq, ws, a) },
            (8, Isa::Scalar) => dot_scalar(wq, ws, a),
            (_, Isa::Avx512) => unsafe { dot4_avx512(wq, ws, a) },
            (_, Isa::Avx2) => unsafe { dot4_avx2(wq, ws, a) },
            (_, Isa::Scalar) => dot4_scalar(wq, ws, a),
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

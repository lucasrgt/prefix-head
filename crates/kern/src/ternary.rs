use std::arch::x86_64::*;

pub struct Ternary {
    pub rows: usize,
    pub cols: usize,
    pub words: usize,
    pub plus: Vec<u64>,
    pub minus: Vec<u64>,
    pub row_sum: Vec<i32>,
}

impl Ternary {
    pub fn pack(w: &[i8], rows: usize, cols: usize) -> Self {
        assert_eq!(cols % 64, 0);
        assert_eq!(w.len(), rows * cols);
        let words = cols / 64;
        let mut plus = vec![0u64; rows * words];
        let mut minus = vec![0u64; rows * words];
        let mut row_sum = vec![0i32; rows];
        for r in 0..rows {
            for c in 0..cols {
                let v = w[r * cols + c];
                let i = r * words + c / 64;
                match v {
                    1 => plus[i] |= 1 << (c % 64),
                    -1 => minus[i] |= 1 << (c % 64),
                    0 => {}
                    _ => panic!("weight {v} is not ternary"),
                }
                row_sum[r] += v as i32;
            }
        }
        Self { rows, cols, words, plus, minus, row_sum }
    }

    pub fn bytes(&self) -> usize {
        (self.plus.len() + self.minus.len()) * 8
    }
}

/// `xu` is `x + 128` as u8, so u8×s8 instructions can be used; kernels subtract `128 * row_sum`.
/// `pos`/`neg` are the sign bitplanes of `x`, for kernels with ternary activations.
pub struct Act {
    pub x: Vec<i8>,
    pub xu: Vec<u8>,
    pub pos: Vec<u64>,
    pub neg: Vec<u64>,
}

impl Act {
    pub fn new(x: &[i8]) -> Self {
        assert_eq!(x.len() % 64, 0);
        let xu = x.iter().map(|&v| v as u8 ^ 0x80).collect();
        let mut pos = vec![0u64; x.len() / 64];
        let mut neg = vec![0u64; x.len() / 64];
        for (c, &v) in x.iter().enumerate() {
            if v > 0 {
                pos[c / 64] |= 1 << (c % 64);
            } else if v < 0 {
                neg[c / 64] |= 1 << (c % 64);
            }
        }
        Self { x: x.to_vec(), xu, pos, neg }
    }
}

pub fn has_avx512_vnni() -> bool {
    is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vnni")
}

pub fn has_avx512_vpopcnt() -> bool {
    is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vpopcntdq")
}

pub fn ternary_ref(t: &Ternary, a: &Act, y: &mut [i32]) {
    for r in 0..t.rows {
        let mut s = 0i32;
        for c in 0..t.cols {
            let i = r * t.words + c / 64;
            let bit = 1u64 << (c % 64);
            if t.plus[i] & bit != 0 {
                s += a.x[c] as i32;
            } else if t.minus[i] & bit != 0 {
                s -= a.x[c] as i32;
            }
        }
        y[r] = s;
    }
}

pub fn sign_ref(t: &Ternary, a: &Act, y: &mut [i32]) {
    for r in 0..t.rows {
        let mut s = 0i32;
        for c in 0..t.cols {
            let i = r * t.words + c / 64;
            let bit = 1u64 << (c % 64);
            let w = (t.plus[i] & bit != 0) as i32 - (t.minus[i] & bit != 0) as i32;
            s += w * a.x[c].signum() as i32;
        }
        y[r] = s;
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub unsafe fn ternary_vnni(t: &Ternary, a: &Act, y: &mut [i32]) {
    let n = t.words;
    let xp = a.xu.as_ptr();
    for r in 0..t.rows {
        unsafe {
            let p = t.plus.as_ptr().add(r * n);
            let m = t.minus.as_ptr().add(r * n);
            let mut acc = _mm512_setzero_si512();
            for c in 0..n {
                let x = _mm512_loadu_si512(xp.add(c * 64).cast());
                let w = _mm512_sub_epi8(_mm512_movm_epi8(*m.add(c)), _mm512_movm_epi8(*p.add(c)));
                acc = _mm512_dpbusd_epi32(acc, x, w);
            }
            y[r] = _mm512_reduce_add_epi32(acc) - 128 * t.row_sum[r];
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub unsafe fn ternary_vnni_r4(t: &Ternary, a: &Act, y: &mut [i32]) {
    let n = t.words;
    let xp = a.xu.as_ptr();
    let full = t.rows / 4 * 4;
    for r in (0..full).step_by(4) {
        unsafe {
            let p = t.plus.as_ptr().add(r * n);
            let m = t.minus.as_ptr().add(r * n);
            let mut acc = [_mm512_setzero_si512(); 4];
            for c in 0..n {
                let x = _mm512_loadu_si512(xp.add(c * 64).cast());
                for k in 0..4 {
                    let o = k * n + c;
                    let w = _mm512_sub_epi8(_mm512_movm_epi8(*m.add(o)), _mm512_movm_epi8(*p.add(o)));
                    acc[k] = _mm512_dpbusd_epi32(acc[k], x, w);
                }
            }
            for k in 0..4 {
                y[r + k] = _mm512_reduce_add_epi32(acc[k]) - 128 * t.row_sum[r + k];
            }
        }
    }
    for r in full..t.rows {
        unsafe {
            let p = t.plus.as_ptr().add(r * n);
            let m = t.minus.as_ptr().add(r * n);
            let mut acc = _mm512_setzero_si512();
            for c in 0..n {
                let x = _mm512_loadu_si512(xp.add(c * 64).cast());
                let w = _mm512_sub_epi8(_mm512_movm_epi8(*m.add(c)), _mm512_movm_epi8(*p.add(c)));
                acc = _mm512_dpbusd_epi32(acc, x, w);
            }
            y[r] = _mm512_reduce_add_epi32(acc) - 128 * t.row_sum[r];
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn ternary_sad(t: &Ternary, a: &Act, y: &mut [i32]) {
    let n = t.words;
    let xp = a.xu.as_ptr();
    let z = _mm512_setzero_si512();
    for r in 0..t.rows {
        unsafe {
            let p = t.plus.as_ptr().add(r * n);
            let m = t.minus.as_ptr().add(r * n);
            let mut ap = z;
            let mut am = z;
            for c in 0..n {
                let x = _mm512_loadu_si512(xp.add(c * 64).cast());
                ap = _mm512_add_epi64(ap, _mm512_sad_epu8(_mm512_maskz_mov_epi8(*p.add(c), x), z));
                am = _mm512_add_epi64(am, _mm512_sad_epu8(_mm512_maskz_mov_epi8(*m.add(c), x), z));
            }
            let s = _mm512_reduce_add_epi64(_mm512_sub_epi64(ap, am));
            y[r] = s as i32 - 128 * t.row_sum[r];
        }
    }
}

#[target_feature(enable = "avx2")]
fn expand32(bits: u32) -> __m256i {
    let v = _mm256_set1_epi32(bits as i32);
    let shuf = _mm256_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    );
    let sel = _mm256_set1_epi64x(0x8040_2010_0804_0201u64 as i64);
    _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(v, shuf), sel), sel)
}

#[target_feature(enable = "avx2")]
fn hsum256(v: __m256i) -> i32 {
    let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b01_00_11_10>(s));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b10_11_00_01>(s));
    _mm_cvtsi128_si32(s)
}

#[target_feature(enable = "avx2")]
pub unsafe fn ternary_avx2(t: &Ternary, a: &Act, y: &mut [i32]) {
    let n = t.words;
    let xp = a.xu.as_ptr();
    let ones = _mm256_set1_epi16(1);
    for r in 0..t.rows {
        unsafe {
            let p = t.plus.as_ptr().add(r * n);
            let m = t.minus.as_ptr().add(r * n);
            let mut acc = _mm256_setzero_si256();
            for c in 0..n {
                let (pw, mw) = (*p.add(c), *m.add(c));
                for h in 0..2 {
                    let x = _mm256_loadu_si256(xp.add(c * 64 + h * 32).cast());
                    let w = _mm256_sub_epi8(expand32((mw >> (h * 32)) as u32), expand32((pw >> (h * 32)) as u32));
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(x, w), ones));
                }
            }
            y[r] = hsum256(acc) - 128 * t.row_sum[r];
        }
    }
}

#[target_feature(enable = "popcnt")]
pub unsafe fn sign_popcnt(t: &Ternary, a: &Act, y: &mut [i32]) {
    let n = t.words;
    for r in 0..t.rows {
        let p = &t.plus[r * n..(r + 1) * n];
        let m = &t.minus[r * n..(r + 1) * n];
        let mut s = 0i32;
        for c in 0..n {
            let (xp, xm) = (a.pos[c], a.neg[c]);
            s += ((p[c] & xp) | (m[c] & xm)).count_ones() as i32;
            s -= ((p[c] & xm) | (m[c] & xp)).count_ones() as i32;
        }
        y[r] = s;
    }
}

#[target_feature(enable = "avx512f,avx512vpopcntdq")]
pub unsafe fn sign_vpopcnt(t: &Ternary, a: &Act, y: &mut [i32]) {
    let n = t.words;
    let body = n / 8 * 8;
    for r in 0..t.rows {
        unsafe {
            let p = t.plus.as_ptr().add(r * n);
            let m = t.minus.as_ptr().add(r * n);
            let mut acc = _mm512_setzero_si512();
            for c in (0..body).step_by(8) {
                let pw = _mm512_loadu_si512(p.add(c).cast());
                let mw = _mm512_loadu_si512(m.add(c).cast());
                let xp = _mm512_loadu_si512(a.pos.as_ptr().add(c).cast());
                let xm = _mm512_loadu_si512(a.neg.as_ptr().add(c).cast());
                let same = _mm512_or_si512(_mm512_and_si512(pw, xp), _mm512_and_si512(mw, xm));
                let diff = _mm512_or_si512(_mm512_and_si512(pw, xm), _mm512_and_si512(mw, xp));
                acc = _mm512_add_epi64(acc, _mm512_sub_epi64(_mm512_popcnt_epi64(same), _mm512_popcnt_epi64(diff)));
            }
            let mut s = _mm512_reduce_add_epi64(acc) as i32;
            for c in body..n {
                let (pw, mw, xp, xm) = (*p.add(c), *m.add(c), a.pos[c], a.neg[c]);
                s += ((pw & xp) | (mw & xm)).count_ones() as i32;
                s -= ((pw & xm) | (mw & xp)).count_ones() as i32;
            }
            y[r] = s;
        }
    }
}

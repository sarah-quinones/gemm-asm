use core::arch::x86_64::*;
use core::mem::transmute;

#[inline(always)]
pub unsafe fn avx_transpose_32x2(values: [[u32; 2]; 2]) -> [[u32; 2]; 2] {
    let x: [__m128i; 2] = transmute([[values[0], [0, 0]], [values[1], [0, 0]]]);

    transmute([_mm_unpacklo_epi32(x[0], x[1])])
}

#[inline(always)]
pub unsafe fn avx_transpose_32x4(values: [[u32; 4]; 4]) -> [[u32; 4]; 4] {
    let x: [__m128i; 4] = transmute(values);

    let x = [
        _mm_unpacklo_epi32(x[0], x[1]),
        _mm_unpackhi_epi32(x[0], x[1]),
        _mm_unpacklo_epi32(x[2], x[3]),
        _mm_unpackhi_epi32(x[2], x[3]),
    ];

    transmute([
        _mm_unpacklo_epi64(x[0], x[2]),
        _mm_unpackhi_epi64(x[0], x[2]),
        _mm_unpacklo_epi64(x[1], x[3]),
        _mm_unpackhi_epi64(x[1], x[3]),
    ])
}

#[inline(always)]
pub unsafe fn avx_transpose_32x8(values: [[u32; 8]; 8]) -> [[u32; 8]; 8] {
    let x: [__m256i; 8] = transmute(values);

    let x = [
        _mm256_unpacklo_epi32(x[0], x[1]),
        _mm256_unpackhi_epi32(x[0], x[1]),
        _mm256_unpacklo_epi32(x[2], x[3]),
        _mm256_unpackhi_epi32(x[2], x[3]),
        _mm256_unpacklo_epi32(x[4], x[5]),
        _mm256_unpackhi_epi32(x[4], x[5]),
        _mm256_unpacklo_epi32(x[6], x[7]),
        _mm256_unpackhi_epi32(x[6], x[7]),
    ];

    let x = [
        _mm256_unpacklo_epi64(x[0], x[2]),
        _mm256_unpackhi_epi64(x[0], x[2]),
        _mm256_unpacklo_epi64(x[1], x[3]),
        _mm256_unpackhi_epi64(x[1], x[3]),
        _mm256_unpacklo_epi64(x[4], x[6]),
        _mm256_unpackhi_epi64(x[4], x[6]),
        _mm256_unpacklo_epi64(x[5], x[7]),
        _mm256_unpackhi_epi64(x[5], x[7]),
    ];

    transmute([
        _mm256_permute2x128_si256::<0b00100000>(x[0], x[4]),
        _mm256_permute2x128_si256::<0b00100000>(x[1], x[5]),
        _mm256_permute2x128_si256::<0b00100000>(x[2], x[6]),
        _mm256_permute2x128_si256::<0b00100000>(x[3], x[7]),
        _mm256_permute2x128_si256::<0b00110001>(x[0], x[4]),
        _mm256_permute2x128_si256::<0b00110001>(x[1], x[5]),
        _mm256_permute2x128_si256::<0b00110001>(x[2], x[6]),
        _mm256_permute2x128_si256::<0b00110001>(x[3], x[7]),
    ])
}

#[cfg(feature = "nightly")]
#[inline(always)]
pub unsafe fn avx512_transpose_32x16(values: [[u32; 16]; 16]) -> [[u32; 16]; 16] {
    let x: [__m512i; 16] = transmute(values);

    const A: usize = 0xA;
    const B: usize = 0xB;
    const C: usize = 0xC;
    const D: usize = 0xD;
    const E: usize = 0xE;
    const F: usize = 0xF;

    let x = [
        _mm512_unpacklo_epi32(x[0], x[1]),
        _mm512_unpackhi_epi32(x[0], x[1]),
        _mm512_unpacklo_epi32(x[2], x[3]),
        _mm512_unpackhi_epi32(x[2], x[3]),
        _mm512_unpacklo_epi32(x[4], x[5]),
        _mm512_unpackhi_epi32(x[4], x[5]),
        _mm512_unpacklo_epi32(x[6], x[7]),
        _mm512_unpackhi_epi32(x[6], x[7]),
        _mm512_unpacklo_epi32(x[8], x[9]),
        _mm512_unpackhi_epi32(x[8], x[9]),
        _mm512_unpacklo_epi32(x[A], x[B]),
        _mm512_unpackhi_epi32(x[A], x[B]),
        _mm512_unpacklo_epi32(x[C], x[D]),
        _mm512_unpackhi_epi32(x[C], x[D]),
        _mm512_unpacklo_epi32(x[E], x[F]),
        _mm512_unpackhi_epi32(x[E], x[F]),
    ];

    let x = [
        _mm512_unpacklo_epi64(x[0], x[2]),
        _mm512_unpackhi_epi64(x[0], x[2]),
        _mm512_unpacklo_epi64(x[1], x[3]),
        _mm512_unpackhi_epi64(x[1], x[3]),
        _mm512_unpacklo_epi64(x[4], x[6]),
        _mm512_unpackhi_epi64(x[4], x[6]),
        _mm512_unpacklo_epi64(x[5], x[7]),
        _mm512_unpackhi_epi64(x[5], x[7]),
        _mm512_unpacklo_epi64(x[8], x[A]),
        _mm512_unpackhi_epi64(x[8], x[A]),
        _mm512_unpacklo_epi64(x[9], x[B]),
        _mm512_unpackhi_epi64(x[9], x[B]),
        _mm512_unpacklo_epi64(x[C], x[E]),
        _mm512_unpackhi_epi64(x[C], x[E]),
        _mm512_unpacklo_epi64(x[D], x[F]),
        _mm512_unpackhi_epi64(x[D], x[F]),
    ];

    let x = [
        _mm512_shuffle_i64x2::<0b10001000>(x[0], x[4]),
        _mm512_shuffle_i64x2::<0b11011101>(x[0], x[4]),
        _mm512_shuffle_i64x2::<0b10001000>(x[1], x[5]),
        _mm512_shuffle_i64x2::<0b11011101>(x[1], x[5]),
        _mm512_shuffle_i64x2::<0b10001000>(x[2], x[6]),
        _mm512_shuffle_i64x2::<0b11011101>(x[2], x[6]),
        _mm512_shuffle_i64x2::<0b10001000>(x[3], x[7]),
        _mm512_shuffle_i64x2::<0b11011101>(x[3], x[7]),
        _mm512_shuffle_i64x2::<0b10001000>(x[8], x[C]),
        _mm512_shuffle_i64x2::<0b11011101>(x[8], x[C]),
        _mm512_shuffle_i64x2::<0b10001000>(x[9], x[D]),
        _mm512_shuffle_i64x2::<0b11011101>(x[9], x[D]),
        _mm512_shuffle_i64x2::<0b10001000>(x[A], x[E]),
        _mm512_shuffle_i64x2::<0b11011101>(x[A], x[E]),
        _mm512_shuffle_i64x2::<0b10001000>(x[B], x[F]),
        _mm512_shuffle_i64x2::<0b11011101>(x[B], x[F]),
    ];

    transmute([
        _mm512_shuffle_i64x2::<0b10001000>(x[0], x[8]),
        _mm512_shuffle_i64x2::<0b10001000>(x[2], x[A]),
        _mm512_shuffle_i64x2::<0b10001000>(x[4], x[C]),
        _mm512_shuffle_i64x2::<0b10001000>(x[6], x[E]),
        _mm512_shuffle_i64x2::<0b10001000>(x[1], x[9]),
        _mm512_shuffle_i64x2::<0b10001000>(x[3], x[B]),
        _mm512_shuffle_i64x2::<0b10001000>(x[5], x[D]),
        _mm512_shuffle_i64x2::<0b10001000>(x[7], x[F]),
        _mm512_shuffle_i64x2::<0b11011101>(x[0], x[8]),
        _mm512_shuffle_i64x2::<0b11011101>(x[2], x[A]),
        _mm512_shuffle_i64x2::<0b11011101>(x[4], x[C]),
        _mm512_shuffle_i64x2::<0b11011101>(x[6], x[E]),
        _mm512_shuffle_i64x2::<0b11011101>(x[1], x[9]),
        _mm512_shuffle_i64x2::<0b11011101>(x[3], x[B]),
        _mm512_shuffle_i64x2::<0b11011101>(x[5], x[D]),
        _mm512_shuffle_i64x2::<0b11011101>(x[7], x[F]),
    ])
}

#[inline(always)]
pub unsafe fn avx_transpose_64x2(values: [[u64; 2]; 2]) -> [[u64; 2]; 2] {
    let x: [__m128i; 2] = transmute(values);

    transmute([
        _mm_unpacklo_epi64(x[0], x[1]),
        _mm_unpackhi_epi64(x[0], x[1]),
    ])
}

#[inline(always)]
pub unsafe fn avx_transpose_64x4(values: [[u64; 4]; 4]) -> [[u64; 4]; 4] {
    let x: [__m256i; 4] = transmute(values);

    let x = [
        _mm256_unpacklo_epi64(x[0], x[1]),
        _mm256_unpackhi_epi64(x[0], x[1]),
        _mm256_unpacklo_epi64(x[2], x[3]),
        _mm256_unpackhi_epi64(x[2], x[3]),
    ];

    transmute([
        _mm256_permute2x128_si256::<0b00100000>(x[0], x[2]),
        _mm256_permute2x128_si256::<0b00100000>(x[1], x[3]),
        _mm256_permute2x128_si256::<0b00110001>(x[0], x[2]),
        _mm256_permute2x128_si256::<0b00110001>(x[1], x[3]),
    ])
}

#[cfg(feature = "nightly")]
#[inline(always)]
pub unsafe fn avx512_transpose_64x8(values: [[u64; 8]; 8]) -> [[u64; 8]; 8] {
    let x: [__m512i; 8] = transmute(values);

    let x = [
        _mm512_unpacklo_epi64(x[0], x[1]),
        _mm512_unpackhi_epi64(x[0], x[1]),
        _mm512_unpacklo_epi64(x[2], x[3]),
        _mm512_unpackhi_epi64(x[2], x[3]),
        _mm512_unpacklo_epi64(x[4], x[5]),
        _mm512_unpackhi_epi64(x[4], x[5]),
        _mm512_unpacklo_epi64(x[6], x[7]),
        _mm512_unpackhi_epi64(x[6], x[7]),
    ];

    let x = [
        _mm512_shuffle_i64x2::<0b10001000>(x[0], x[2]),
        _mm512_shuffle_i64x2::<0b11011101>(x[0], x[2]),
        _mm512_shuffle_i64x2::<0b10001000>(x[1], x[3]),
        _mm512_shuffle_i64x2::<0b11011101>(x[1], x[3]),
        _mm512_shuffle_i64x2::<0b10001000>(x[4], x[6]),
        _mm512_shuffle_i64x2::<0b11011101>(x[4], x[6]),
        _mm512_shuffle_i64x2::<0b10001000>(x[5], x[7]),
        _mm512_shuffle_i64x2::<0b11011101>(x[5], x[7]),
    ];

    transmute([
        _mm512_shuffle_i64x2::<0b10001000>(x[0], x[4]),
        _mm512_shuffle_i64x2::<0b10001000>(x[2], x[6]),
        _mm512_shuffle_i64x2::<0b10001000>(x[1], x[5]),
        _mm512_shuffle_i64x2::<0b10001000>(x[3], x[7]),
        _mm512_shuffle_i64x2::<0b11011101>(x[0], x[4]),
        _mm512_shuffle_i64x2::<0b11011101>(x[2], x[6]),
        _mm512_shuffle_i64x2::<0b11011101>(x[1], x[5]),
        _mm512_shuffle_i64x2::<0b11011101>(x[3], x[7]),
    ])
}

#[cfg(test)]
mod tests {
    use core::array::from_fn;

    use super::*;
    use rand::random;

    #[test]
    fn test_transpose_32x2() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }

        let x = from_fn(|_| from_fn(|_| random()));
        let xt = unsafe { avx_transpose_32x2(x) };

        let mut target = x;
        for i in 0..2 {
            for j in 0..2 {
                target[i][j] = x[j][i];
            }
        }

        assert_eq!(xt, target);
    }

    #[test]
    fn test_transpose_32x4() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }

        let x = from_fn(|_| from_fn(|_| random()));
        let xt = unsafe { avx_transpose_32x4(x) };

        let mut target = x;
        for i in 0..4 {
            for j in 0..4 {
                target[i][j] = x[j][i];
            }
        }

        assert_eq!(xt, target);
    }

    #[test]
    fn test_transpose_32x8() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }

        let x = from_fn(|_| from_fn(|_| random()));

        let xt = unsafe { avx_transpose_32x8(x) };

        let mut target = x;
        for i in 0..8 {
            for j in 0..8 {
                target[i][j] = x[j][i];
            }
        }

        assert_eq!(xt, target);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_transpose_32x16() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }

        let x = from_fn(|_| from_fn(|_| random()));
        let xt = unsafe { avx512_transpose_32x16(x) };

        let mut target = x;
        for i in 0..16 {
            for j in 0..16 {
                target[i][j] = x[j][i];
            }
        }

        assert_eq!(xt, target);
    }

    #[test]
    fn test_transpose_64x2() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }

        let x = from_fn(|_| from_fn(|_| random()));
        let xt = unsafe { avx_transpose_64x2(x) };

        let mut target = x;
        for i in 0..2 {
            for j in 0..2 {
                target[i][j] = x[j][i];
            }
        }

        assert_eq!(xt, target);
    }

    #[test]
    fn test_transpose_64x4() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }

        let x = from_fn(|_| from_fn(|_| random()));
        let xt = unsafe { avx_transpose_64x4(x) };

        let mut target = x;
        for i in 0..4 {
            for j in 0..4 {
                target[i][j] = x[j][i];
            }
        }

        assert_eq!(xt, target);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_transpose_64x8() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }

        let x = from_fn(|_| from_fn(|_| random()));
        let xt = unsafe { avx512_transpose_64x8(x) };

        let mut target = x;
        for i in 0..8 {
            for j in 0..8 {
                target[i][j] = x[j][i];
            }
        }

        assert_eq!(xt, target);
    }
}

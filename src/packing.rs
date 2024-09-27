use core::arch::x86_64::*;
use core::mem::{transmute, zeroed};

#[cfg(feature = "nightly")]
#[target_feature(enable = "avx512f")]
pub unsafe fn pack_avx512_u32_col_major(
    dst: &mut [u32],
    src: *const u32,
    src_byte_stride: isize,
    dst_byte_stride: isize,
    total_nrows: usize,
    ncols: usize,
    rows_to_skip: usize,
) {
    if dst_byte_stride % 64 != 0 && rows_to_skip < 8 {
        return pack_avx_u32_col_major(
            dst,
            src,
            src_byte_stride,
            dst_byte_stride,
            total_nrows,
            ncols,
            rows_to_skip,
        );
    }
    debug_assert!(rows_to_skip < 16);

    let mut dst = dst.as_mut_ptr() as *mut __m512i;
    let mut src = src as *const __m512i;

    if rows_to_skip == 0 && total_nrows == 32 {
        for _ in 0..ncols {
            dst.write_unaligned(src.read_unaligned());
            dst.add(1).write_unaligned(src.add(1).read_unaligned());
            dst = dst.wrapping_byte_offset(dst_byte_stride);
            src = src.wrapping_byte_offset(src_byte_stride);
        }
    } else {
        let head_mask = crate::millikernel::AVX512_HEAD_MASK_F32[rows_to_skip];
        let tail_mask = crate::millikernel::AVX512_TAIL_MASK_F32[total_nrows % 16];

        if total_nrows <= 16 {
            let mask = head_mask & tail_mask;

            for _ in 0..ncols {
                _mm512_mask_storeu_epi32(
                    dst as *mut i32,
                    mask,
                    _mm512_maskz_loadu_epi32(mask, src as *const i32),
                );

                dst = dst.wrapping_byte_offset(dst_byte_stride);
                src = src.wrapping_byte_offset(src_byte_stride);
            }
        } else if total_nrows <= 32 {
            for _ in 0..ncols {
                _mm512_mask_storeu_epi32(
                    dst as *mut i32,
                    head_mask,
                    _mm512_maskz_loadu_epi32(head_mask, src as *const i32),
                );
                _mm512_mask_storeu_epi32(
                    dst.add(1) as *mut i32,
                    tail_mask,
                    _mm512_maskz_loadu_epi32(tail_mask, src.add(1) as *const i32),
                );

                dst = dst.wrapping_byte_offset(dst_byte_stride);
                src = src.wrapping_byte_offset(src_byte_stride);
            }
        } else {
            debug_assert!(total_nrows <= 48);

            for _ in 0..ncols {
                _mm512_mask_storeu_epi32(
                    dst as *mut i32,
                    head_mask,
                    _mm512_maskz_loadu_epi32(head_mask, src as *const i32),
                );
                dst.add(1).write_unaligned(src.add(1).read_unaligned());
                _mm512_mask_storeu_epi32(
                    dst.add(2) as *mut i32,
                    tail_mask,
                    _mm512_maskz_loadu_epi32(tail_mask, src.add(2) as *const i32),
                );

                dst = dst.wrapping_byte_offset(dst_byte_stride);
                src = src.wrapping_byte_offset(src_byte_stride);
            }
        }
    }
}

#[cfg(feature = "nightly")]
pub unsafe fn pack_avx512_u64_col_major(
    dst: &mut [u64],
    src: *const u64,
    src_byte_stride: isize,
    dst_byte_stride: isize,
    total_nrows: usize,
    ncols: usize,
    rows_to_skip: usize,
) {
    let len = dst.len();
    pack_avx512_u32_col_major(
        core::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u32, 2 * len),
        src as *const u32,
        src_byte_stride,
        dst_byte_stride,
        2 * total_nrows,
        ncols,
        2 * rows_to_skip,
    )
}

#[cfg(feature = "nightly")]
pub unsafe fn pack_avx512_u128_col_major(
    dst: &mut [u128],
    src: *const u128,
    src_byte_stride: isize,
    dst_byte_stride: isize,
    total_nrows: usize,
    ncols: usize,
    rows_to_skip: usize,
) {
    let len = dst.len();
    pack_avx512_u32_col_major(
        core::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u32, 4 * len),
        src as *const u32,
        src_byte_stride,
        dst_byte_stride,
        4 * total_nrows,
        ncols,
        4 * rows_to_skip,
    )
}

#[target_feature(enable = "avx2")]
pub unsafe fn pack_avx_u32_col_major(
    dst: &mut [u32],
    src: *const u32,
    src_byte_stride: isize,
    dst_byte_stride: isize,
    total_nrows: usize,
    ncols: usize,
    rows_to_skip: usize,
) {
    if dst_byte_stride % 32 != 0 && rows_to_skip < 4 {
        return pack_avx_half_u32_col_major(
            dst,
            src,
            src_byte_stride,
            dst_byte_stride,
            total_nrows,
            ncols,
            rows_to_skip,
        );
    }

    debug_assert!(rows_to_skip < 8);
    let mut dst = dst.as_mut_ptr() as *mut __m256i;
    let mut src = src as *const __m256i;
    if rows_to_skip == 0 && total_nrows == 16 {
        for _ in 0..ncols {
            dst.write_unaligned(src.read_unaligned());
            dst.add(1).write_unaligned(src.add(1).read_unaligned());
            dst = dst.wrapping_byte_offset(dst_byte_stride);
            src = src.wrapping_byte_offset(src_byte_stride);
        }
    } else {
        let head_mask = crate::millikernel::AVX_HEAD_MASK_F32[rows_to_skip];
        let tail_mask = crate::millikernel::AVX_TAIL_MASK_F32[total_nrows % 8];

        if total_nrows <= 8 {
            let mask = _mm256_and_si256(head_mask, tail_mask);

            for _ in 0..ncols {
                _mm256_maskstore_epi32(
                    dst as *mut i32,
                    mask,
                    _mm256_maskload_epi32(src as *const i32, mask),
                );

                dst = dst.wrapping_byte_offset(dst_byte_stride);
                src = src.wrapping_byte_offset(src_byte_stride);
            }
        } else if total_nrows <= 16 {
            for _ in 0..ncols {
                _mm256_maskstore_epi32(
                    dst as *mut i32,
                    head_mask,
                    _mm256_maskload_epi32(src as *const i32, head_mask),
                );
                _mm256_maskstore_epi32(
                    dst.add(1) as *mut i32,
                    tail_mask,
                    _mm256_maskload_epi32(src.add(1) as *const i32, tail_mask),
                );

                dst = dst.wrapping_byte_offset(dst_byte_stride);
                src = src.wrapping_byte_offset(src_byte_stride);
            }
        } else {
            debug_assert!(total_nrows <= 24);

            for _ in 0..ncols {
                _mm256_maskstore_epi32(
                    dst as *mut i32,
                    head_mask,
                    _mm256_maskload_epi32(src as *const i32, head_mask),
                );
                dst.add(1).write_unaligned(src.add(1).read_unaligned());
                _mm256_maskstore_epi32(
                    dst.add(2) as *mut i32,
                    tail_mask,
                    _mm256_maskload_epi32(src.add(2) as *const i32, tail_mask),
                );

                dst = dst.wrapping_byte_offset(dst_byte_stride);
                src = src.wrapping_byte_offset(src_byte_stride);
            }
        }
    }
}

#[target_feature(enable = "avx2")]
pub unsafe fn pack_avx_half_u32_col_major(
    dst: &mut [u32],
    src: *const u32,
    src_byte_stride: isize,
    dst_byte_stride: isize,
    total_nrows: usize,
    ncols: usize,
    rows_to_skip: usize,
) {
    let mut dst = dst.as_mut_ptr() as *mut __m128i;
    let mut src = src as *const __m128i;

    if rows_to_skip == 0 && total_nrows == 12 {
        for _ in 0..ncols {
            dst.write_unaligned(src.read_unaligned());
            dst.add(1).write_unaligned(src.add(1).read_unaligned());
            dst.add(2).write_unaligned(src.add(2).read_unaligned());
            dst = dst.wrapping_byte_offset(dst_byte_stride);
            src = src.wrapping_byte_offset(src_byte_stride);
        }
    } else if rows_to_skip == 0 && total_nrows == 6 {
        for _ in 0..ncols {
            dst.write_unaligned(src.read_unaligned());
            (dst.add(1) as *mut u64).write_unaligned((src.add(1) as *const u64).read_unaligned());
            dst = dst.wrapping_byte_offset(dst_byte_stride);
            src = src.wrapping_byte_offset(src_byte_stride);
        }
    } else {
        panic!();
    }
}

pub unsafe fn pack_avx_u64_col_major(
    dst: &mut [u64],
    src: *const u64,
    src_byte_stride: isize,
    dst_byte_stride: isize,
    total_nrows: usize,
    ncols: usize,
    rows_to_skip: usize,
) {
    let len = dst.len();
    pack_avx_u32_col_major(
        core::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u32, 2 * len),
        src as *const u32,
        src_byte_stride,
        dst_byte_stride,
        2 * total_nrows,
        ncols,
        2 * rows_to_skip,
    )
}

pub unsafe fn pack_avx_u128_col_major(
    dst: &mut [u128],
    src: *const u128,
    src_byte_stride: isize,
    dst_byte_stride: isize,
    total_nrows: usize,
    ncols: usize,
    rows_to_skip: usize,
) {
    let len = dst.len();
    pack_avx_u32_col_major(
        core::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u32, 4 * len),
        src as *const u32,
        src_byte_stride,
        dst_byte_stride,
        4 * total_nrows,
        ncols,
        4 * rows_to_skip,
    )
}

#[cfg(feature = "nightly")]
#[target_feature(enable = "avx512f")]
pub unsafe fn pack_avx512_u32_row_major(
    dst: &mut [u32],
    src: *const u32,
    src_byte_stride: isize,
    dst_byte_stride: isize,
    total_nrows: usize,
    ncols: usize,
    rows_to_skip: usize,
) {
    debug_assert!(rows_to_skip == 0);

    #[inline(always)]
    unsafe fn imp<const N: usize>(
        mut total_ncols: usize,
        tail_mask: u16,
        mut dst: *mut u32,
        src_byte_stride: isize,
        dst_byte_stride: isize,
        mut src: *const u32,
        total_nrows: usize,
    ) {
        macro_rules! do_the_thing_16 {
            () => {{
                load!(z0, 0x0);
                load!(z1, 0x1);
                load!(z2, 0x2);
                load!(z3, 0x3);
                load!(z4, 0x4);
                load!(z5, 0x5);
                load!(z6, 0x6);
                load!(z7, 0x7);
                load!(z8, 0x8);
                load!(z9, 0x9);
                load!(za, 0xa);
                load!(zb, 0xb);
                load!(zc, 0xc);
                load!(zd, 0xd);
                load!(ze, 0xe);
                load!(zf, 0xf);

                let [z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, za, zb, zc, zd, ze, zf] =
                    transmute::<_, [__m512i; 16]>(crate::transpose::avx512_transpose_32x16(
                        transmute([
                            z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, za, zb, zc, zd, ze, zf,
                        ]),
                    ));

                store!(z0, 0x0);
                store!(z1, 0x1);
                store!(z2, 0x2);
                store!(z3, 0x3);
                store!(z4, 0x4);
                store!(z5, 0x5);
                store!(z6, 0x6);
                store!(z7, 0x7);
                store!(z8, 0x8);
                store!(z9, 0x9);
                store!(za, 0xa);
                store!(zb, 0xb);
                store!(zc, 0xc);
                store!(zd, 0xd);
                store!(ze, 0xe);
                store!(zf, 0xf);
            }};
        }

        macro_rules! do_the_thing_8 {
            () => {{
                load!(z0, 0x0);
                load!(z1, 0x1);
                load!(z2, 0x2);
                load!(z3, 0x3);
                load!(z4, 0x4);
                load!(z5, 0x5);
                load!(z6, 0x6);
                load!(z7, 0x7);

                let [z0, z1, z2, z3, z4, z5, z6, z7] =
                    transmute::<_, [__m256i; 8]>(crate::transpose::avx_transpose_32x8(transmute(
                        [z0, z1, z2, z3, z4, z5, z6, z7],
                    )));

                store!(z0, 0x0);
                store!(z1, 0x1);
                store!(z2, 0x2);
                store!(z3, 0x3);
                store!(z4, 0x4);
                store!(z5, 0x5);
                store!(z6, 0x6);
                store!(z7, 0x7);
            }};
        }

        {
            while total_ncols >= 16 {
                let old = dst;
                {
                    macro_rules! load {
                        ($name: ident, $i: expr) => {
                            let $name = if $i < total_nrows {
                                (src.wrapping_byte_offset(src_byte_stride * $i as isize)
                                    as *const __m512i)
                                    .read_unaligned()
                            } else {
                                zeroed()
                            };
                        };
                    }

                    macro_rules! store {
                        ($name: expr, $i: expr) => {
                            (dst.wrapping_byte_offset($i * dst_byte_stride) as *mut __m512i)
                                .write_unaligned($name);
                        };
                    }

                    do_the_thing_16!();
                    dst = dst.wrapping_add(16);
                }
                if const { N == 16 } {
                    macro_rules! load {
                        ($name: ident, $i: expr) => {
                            let $name = if ($i + 16) < total_nrows {
                                (src.wrapping_byte_offset(src_byte_stride * ($i + 16) as isize)
                                    as *const __m512i)
                                    .read_unaligned()
                            } else {
                                zeroed()
                            };
                        };
                    }

                    macro_rules! store {
                        ($name: expr, $i: expr) => {
                            (dst.wrapping_byte_offset($i * dst_byte_stride) as *mut __m512i)
                                .write_unaligned($name);
                        };
                    }

                    do_the_thing_16!();
                } else if const { N == 8 } {
                    macro_rules! load {
                        ($name: ident, $i: expr) => {
                            let $name = if ($i + 16) < total_nrows {
                                (src.wrapping_byte_offset(src_byte_stride * ($i + 16) as isize)
                                    as *const __m256i)
                                    .read_unaligned()
                            } else {
                                zeroed()
                            };
                        };
                    }
                    {
                        macro_rules! store {
                            ($name: expr, $i: expr) => {
                                (dst.wrapping_byte_offset($i * dst_byte_stride) as *mut __m256i)
                                    .write_unaligned($name);
                            };
                        }

                        do_the_thing_8!();
                    }
                    src = src.wrapping_add(8);
                    {
                        macro_rules! store {
                            ($name: expr, $i: expr) => {
                                (dst.wrapping_byte_offset(($i + 8) * dst_byte_stride)
                                    as *mut __m256i)
                                    .write_unaligned($name);
                            };
                        }

                        do_the_thing_8!();
                    }
                    src = src.wrapping_sub(8);
                }
                dst = old.wrapping_byte_offset(16 * dst_byte_stride);
                src = src.wrapping_add(16);
                total_ncols -= 16;
            }
        }

        if total_ncols > 0 {
            {
                macro_rules! load {
                    ($name: ident, $i: expr) => {
                        let $name = if $i < total_nrows {
                            _mm512_maskz_loadu_epi32(
                                tail_mask,
                                src.wrapping_byte_offset(src_byte_stride * $i as isize)
                                    as *const i32,
                            )
                        } else {
                            zeroed()
                        };
                    };
                }

                macro_rules! store {
                    ($name: expr, $i: expr) => {
                        if $i < total_ncols {
                            (dst.wrapping_byte_offset($i * dst_byte_stride) as *mut __m512i)
                                .write_unaligned($name)
                        };
                    };
                }

                do_the_thing_16!();
                dst = dst.wrapping_add(16);
            }

            if const { N == 16 } {
                macro_rules! load {
                    ($name: ident, $i: expr) => {
                        let $name = if $i + 16 < total_nrows {
                            _mm512_maskz_loadu_epi32(
                                tail_mask,
                                src.wrapping_byte_offset(src_byte_stride * ($i + 16) as isize)
                                    as *const i32,
                            )
                        } else {
                            zeroed()
                        };
                    };
                }

                macro_rules! store {
                    ($name: expr, $i: expr) => {
                        if $i < total_ncols {
                            (dst.wrapping_byte_offset($i * dst_byte_stride) as *mut __m512i)
                                .write_unaligned($name)
                        };
                    };
                }

                do_the_thing_16!();
            } else if const { N == 8 } {
                {
                    macro_rules! load {
                        ($name: ident, $i: expr) => {
                            let $name = if $i + 16 < total_nrows {
                                _mm256_maskz_loadu_epi32(
                                    tail_mask as u8,
                                    src.wrapping_byte_offset(src_byte_stride * ($i + 16) as isize)
                                        as *const i32,
                                )
                            } else {
                                zeroed()
                            };
                        };
                    }
                    macro_rules! store {
                        ($name: expr, $i: expr) => {
                            if $i < total_ncols {
                                (dst.wrapping_byte_offset($i * dst_byte_stride) as *mut __m256i)
                                    .write_unaligned($name)
                            };
                        };
                    }

                    do_the_thing_8!();
                }
                src = src.wrapping_add(8);
                {
                    macro_rules! load {
                        ($name: ident, $i: expr) => {
                            let $name = if $i + 16 < total_nrows {
                                _mm256_maskz_loadu_epi32(
                                    (tail_mask >> 8) as u8,
                                    src.wrapping_byte_offset(src_byte_stride * ($i + 16) as isize)
                                        as *const i32,
                                )
                            } else {
                                zeroed()
                            };
                        };
                    }
                    macro_rules! store {
                        ($name: expr, $i: expr) => {
                            if $i + 8 < total_ncols {
                                (dst.wrapping_byte_offset(($i + 8) * dst_byte_stride)
                                    as *mut __m256i)
                                    .write_unaligned($name)
                            };
                        };
                    }

                    do_the_thing_8!();
                }
            }
        }
    }

    let total_ncols = ncols;
    let padded_ncols = (total_ncols + 15) / 16 * 16;
    let tail_len = padded_ncols - total_ncols;

    let tail_mask = if tail_len == 16 {
        0
    } else {
        crate::millikernel::AVX512_TAIL_MASK_F32[padded_ncols - total_ncols]
    };

    if dst_byte_stride == 128 {
        let dst = dst.as_mut_ptr();
        imp::<16>(
            total_ncols,
            tail_mask,
            dst,
            src_byte_stride,
            dst_byte_stride,
            src,
            total_nrows,
        );
    } else if dst_byte_stride == 64 {
        let dst = dst.as_mut_ptr();
        imp::<0>(
            total_ncols,
            tail_mask,
            dst,
            src_byte_stride,
            dst_byte_stride,
            src,
            total_nrows,
        );
    } else {
        debug_assert!(dst_byte_stride == 48);
        pack_avx_u32_row_major(
            dst,
            src,
            src_byte_stride,
            dst_byte_stride,
            total_nrows,
            ncols,
            rows_to_skip,
        );
    }
}

#[cfg(feature = "nightly")]
#[target_feature(enable = "avx512f")]
pub unsafe fn pack_avx512_u64_row_major(
    dst: &mut [u64],
    src: *const u64,
    src_byte_stride: isize,
    dst_byte_stride: isize,
    total_nrows: usize,
    ncols: usize,
    rows_to_skip: usize,
) {
    debug_assert!(rows_to_skip == 0);

    #[inline(always)]
    unsafe fn imp<const N: usize>(
        mut total_ncols: usize,
        tail_mask: u8,
        mut dst: *mut u64,
        src_byte_stride: isize,
        dst_byte_stride: isize,
        mut src: *const u64,
        total_nrows: usize,
    ) {
        macro_rules! do_the_thing_8 {
            () => {{
                load!(z0, 0x0);
                load!(z1, 0x1);
                load!(z2, 0x2);
                load!(z3, 0x3);
                load!(z4, 0x4);
                load!(z5, 0x5);
                load!(z6, 0x6);
                load!(z7, 0x7);

                let [z0, z1, z2, z3, z4, z5, z6, z7] =
                    transmute::<_, [__m512i; 8]>(crate::transpose::avx512_transpose_64x8(
                        transmute([z0, z1, z2, z3, z4, z5, z6, z7]),
                    ));

                store!(z0, 0x0);
                store!(z1, 0x1);
                store!(z2, 0x2);
                store!(z3, 0x3);
                store!(z4, 0x4);
                store!(z5, 0x5);
                store!(z6, 0x6);
                store!(z7, 0x7);
            }};
        }

        macro_rules! do_the_thing_4 {
            () => {{
                load!(z0, 0x0);
                load!(z1, 0x1);
                load!(z2, 0x2);
                load!(z3, 0x3);

                let [z0, z1, z2, z3] = transmute::<_, [__m256i; 4]>(
                    crate::transpose::avx_transpose_64x4(transmute([z0, z1, z2, z3])),
                );

                store!(z0, 0x0);
                store!(z1, 0x1);
                store!(z2, 0x2);
                store!(z3, 0x3);
            }};
        }

        {
            while total_ncols >= 8 {
                let old = dst;
                {
                    macro_rules! load {
                        ($name: ident, $i: expr) => {
                            let $name = if $i < total_nrows {
                                (src.wrapping_byte_offset(src_byte_stride * $i as isize)
                                    as *const __m512i)
                                    .read_unaligned()
                            } else {
                                zeroed()
                            };
                        };
                    }

                    macro_rules! store {
                        ($name: expr, $i: expr) => {
                            (dst.wrapping_byte_offset($i * dst_byte_stride) as *mut __m512i)
                                .write_unaligned($name);
                        };
                    }

                    do_the_thing_8!();
                    dst = dst.wrapping_add(8);
                }
                if const { N == 8 } {
                    macro_rules! load {
                        ($name: ident, $i: expr) => {
                            let $name = if ($i + 8) < total_nrows {
                                (src.wrapping_byte_offset(src_byte_stride * ($i + 8) as isize)
                                    as *const __m512i)
                                    .read_unaligned()
                            } else {
                                zeroed()
                            };
                        };
                    }

                    macro_rules! store {
                        ($name: expr, $i: expr) => {
                            (dst.wrapping_byte_offset($i * dst_byte_stride) as *mut __m512i)
                                .write_unaligned($name);
                        };
                    }

                    do_the_thing_8!();
                } else if const { N == 4 } {
                    macro_rules! load {
                        ($name: ident, $i: expr) => {
                            let $name = if ($i + 8) < total_nrows {
                                (src.wrapping_byte_offset(src_byte_stride * ($i + 8) as isize)
                                    as *const __m256i)
                                    .read_unaligned()
                            } else {
                                zeroed()
                            };
                        };
                    }
                    {
                        macro_rules! store {
                            ($name: expr, $i: expr) => {
                                (dst.wrapping_byte_offset($i * dst_byte_stride) as *mut __m256i)
                                    .write_unaligned($name);
                            };
                        }

                        do_the_thing_4!();
                    }
                    src = src.wrapping_add(4);
                    {
                        macro_rules! store {
                            ($name: expr, $i: expr) => {
                                (dst.wrapping_byte_offset(($i + 4) * dst_byte_stride)
                                    as *mut __m256i)
                                    .write_unaligned($name);
                            };
                        }

                        do_the_thing_4!();
                    }
                    src = src.wrapping_sub(4);
                }
                dst = old.wrapping_byte_offset(8 * dst_byte_stride);
                src = src.wrapping_add(8);
                total_ncols -= 8;
            }
        }

        if total_ncols > 0 {
            {
                macro_rules! load {
                    ($name: ident, $i: expr) => {
                        let $name = if $i < total_nrows {
                            _mm512_maskz_loadu_epi64(
                                tail_mask,
                                src.wrapping_byte_offset(src_byte_stride * $i as isize)
                                    as *const i64,
                            )
                        } else {
                            zeroed()
                        };
                    };
                }

                macro_rules! store {
                    ($name: expr, $i: expr) => {
                        if $i < total_ncols {
                            (dst.wrapping_byte_offset($i * dst_byte_stride) as *mut __m512i)
                                .write_unaligned($name)
                        };
                    };
                }

                do_the_thing_8!();
                dst = dst.wrapping_add(8);
            }

            if const { N == 8 } {
                macro_rules! load {
                    ($name: ident, $i: expr) => {
                        let $name = if $i + 8 < total_nrows {
                            _mm512_maskz_loadu_epi64(
                                tail_mask,
                                src.wrapping_byte_offset(src_byte_stride * ($i + 8) as isize)
                                    as *const i64,
                            )
                        } else {
                            zeroed()
                        };
                    };
                }

                macro_rules! store {
                    ($name: expr, $i: expr) => {
                        if $i < total_ncols {
                            (dst.wrapping_byte_offset($i * dst_byte_stride) as *mut __m512i)
                                .write_unaligned($name)
                        };
                    };
                }

                do_the_thing_8!();
            } else if const { N == 4 } {
                {
                    macro_rules! load {
                        ($name: ident, $i: expr) => {
                            let $name = if $i + 8 < total_nrows {
                                _mm256_maskz_loadu_epi64(
                                    tail_mask,
                                    src.wrapping_byte_offset(src_byte_stride * ($i + 8) as isize)
                                        as *const i64,
                                )
                            } else {
                                zeroed()
                            };
                        };
                    }
                    macro_rules! store {
                        ($name: expr, $i: expr) => {
                            if $i < total_ncols {
                                (dst.wrapping_byte_offset($i * dst_byte_stride) as *mut __m256i)
                                    .write_unaligned($name)
                            };
                        };
                    }

                    do_the_thing_4!();
                }
                src = src.wrapping_add(4);
                {
                    macro_rules! load {
                        ($name: ident, $i: expr) => {
                            let $name = if $i + 8 < total_nrows {
                                _mm256_maskz_loadu_epi64(
                                    (tail_mask >> 4),
                                    src.wrapping_byte_offset(src_byte_stride * ($i + 8) as isize)
                                        as *const i64,
                                )
                            } else {
                                zeroed()
                            };
                        };
                    }
                    macro_rules! store {
                        ($name: expr, $i: expr) => {
                            if $i + 4 < total_ncols {
                                (dst.wrapping_byte_offset(($i + 4) * dst_byte_stride)
                                    as *mut __m256i)
                                    .write_unaligned($name)
                            };
                        };
                    }

                    do_the_thing_4!();
                }
            }
        }
    }

    let total_ncols = ncols;
    let padded_ncols = (total_ncols + 7) / 8 * 8;
    let tail_len = padded_ncols - total_ncols;

    let dst = dst.as_mut_ptr();
    let src = src;

    let tail_mask = if tail_len == 8 {
        0
    } else {
        crate::millikernel::AVX512_TAIL_MASK_F64[padded_ncols - total_ncols]
    };

    if dst_byte_stride == 128 {
        imp::<8>(
            total_ncols,
            tail_mask,
            dst,
            src_byte_stride,
            dst_byte_stride,
            src,
            total_nrows,
        );
    } else if dst_byte_stride == 64 {
        imp::<0>(
            total_ncols,
            tail_mask,
            dst,
            src_byte_stride,
            dst_byte_stride,
            src,
            total_nrows,
        );
    } else {
        debug_assert!(dst_byte_stride == 96);
        imp::<4>(
            total_ncols,
            tail_mask,
            dst,
            src_byte_stride,
            dst_byte_stride,
            src,
            total_nrows,
        );
    }
}

#[cfg(feature = "nightly")]
#[target_feature(enable = "avx512f")]
pub unsafe fn pack_avx512_u128_row_major(
    dst: &mut [u128],
    src: *const u128,
    src_byte_stride: isize,
    dst_byte_stride: isize,
    total_nrows: usize,
    ncols: usize,
    rows_to_skip: usize,
) {
    debug_assert!(rows_to_skip == 0);

    #[inline(always)]
    unsafe fn imp(
        n: usize,
        mut total_ncols: usize,
        tail_mask: u8,
        mut dst: *mut u128,
        src_byte_stride: isize,
        dst_byte_stride: isize,
        mut src: *const u128,
        total_nrows: usize,
    ) {
        macro_rules! do_the_thing_4 {
            () => {{
                load!(z0, 0x0);
                load!(z1, 0x1);
                load!(z2, 0x2);
                load!(z3, 0x3);

                let [z0, z1, z2, z3] = transmute::<_, [__m512i; 4]>(
                    crate::transpose::avx512_transpose_128x4(transmute([z0, z1, z2, z3])),
                );

                store!(z0, 0x0);
                store!(z1, 0x1);
                store!(z2, 0x2);
                store!(z3, 0x3);
            }};
        }

        {
            while total_ncols >= 4 {
                let old = dst;

                for i in 0..n {
                    macro_rules! load {
                        ($name: ident, $i: expr) => {
                            let $name = if $i < total_nrows {
                                (src.wrapping_byte_offset(src_byte_stride * ($i + i * 4) as isize)
                                    as *const __m512i)
                                    .read_unaligned()
                            } else {
                                zeroed()
                            };
                        };
                    }

                    macro_rules! store {
                        ($name: expr, $i: expr) => {
                            (dst.wrapping_byte_offset($i * dst_byte_stride) as *mut __m512i)
                                .write_unaligned($name);
                        };
                    }

                    do_the_thing_4!();
                    dst = dst.wrapping_add(4);
                }

                dst = old.wrapping_byte_offset(4 * dst_byte_stride);
                src = src.wrapping_add(4);
                total_ncols -= 4;
            }
        }

        if total_ncols > 0 {
            for i in 0..n {
                macro_rules! load {
                    ($name: ident, $i: expr) => {
                        let $name = if ($i + i * 4) < total_nrows {
                            _mm512_maskz_loadu_epi64(
                                tail_mask,
                                src.wrapping_byte_offset(src_byte_stride * ($i + i * 4) as isize)
                                    as *const i64,
                            )
                        } else {
                            zeroed()
                        };
                    };
                }

                macro_rules! store {
                    ($name: expr, $i: expr) => {
                        if $i < total_ncols {
                            (dst.wrapping_byte_offset($i * dst_byte_stride) as *mut __m512i)
                                .write_unaligned($name)
                        };
                    };
                }

                do_the_thing_4!();
                dst = dst.wrapping_add(4);
            }
        }
    }

    let total_ncols = ncols;
    let padded_ncols = (total_ncols + 3) / 4 * 4;
    let tail_len = padded_ncols - total_ncols;

    let tail_mask = if tail_len == 4 {
        0
    } else {
        crate::millikernel::AVX512_TAIL_MASK_F64[2 * (padded_ncols - total_ncols)]
    };

    let n = if dst_byte_stride == 128 {
        2
    } else if dst_byte_stride == 64 {
        1
    } else {
        debug_assert!(dst_byte_stride == 192);
        3
    };

    let dst = dst.as_mut_ptr();
    imp(
        n,
        total_ncols,
        tail_mask,
        dst,
        src_byte_stride,
        dst_byte_stride,
        src,
        total_nrows,
    );
}

#[target_feature(enable = "avx2")]
pub unsafe fn pack_avx_u32_row_major(
    dst: &mut [u32],
    src: *const u32,
    src_byte_stride: isize,
    dst_byte_stride: isize,
    total_nrows: usize,
    ncols: usize,
    rows_to_skip: usize,
) {
    debug_assert!(rows_to_skip == 0);

    #[inline(always)]
    unsafe fn imp<const N: usize>(
        mut total_ncols: usize,
        tail_mask: __m256i,
        mut dst: *mut u32,
        src_byte_stride: isize,
        dst_byte_stride: isize,
        mut src: *const u32,
        total_nrows: usize,
    ) {
        macro_rules! do_the_thing_8 {
            () => {{
                load!(z0, 0x0);
                load!(z1, 0x1);
                load!(z2, 0x2);
                load!(z3, 0x3);
                load!(z4, 0x4);
                load!(z5, 0x5);
                load!(z6, 0x6);
                load!(z7, 0x7);

                let [z0, z1, z2, z3, z4, z5, z6, z7] =
                    transmute::<_, [__m256i; 8]>(crate::transpose::avx_transpose_32x8(transmute(
                        [z0, z1, z2, z3, z4, z5, z6, z7],
                    )));

                store!(z0, 0x0);
                store!(z1, 0x1);
                store!(z2, 0x2);
                store!(z3, 0x3);
                store!(z4, 0x4);
                store!(z5, 0x5);
                store!(z6, 0x6);
                store!(z7, 0x7);
            }};
        }

        macro_rules! do_the_thing_4 {
            () => {{
                load!(z0, 0x0);
                load!(z1, 0x1);
                load!(z2, 0x2);
                load!(z3, 0x3);

                let [z0, z1, z2, z3] = transmute::<_, [__m128i; 4]>(
                    crate::transpose::avx_transpose_32x4(transmute([z0, z1, z2, z3])),
                );

                store!(z0, 0x0);
                store!(z1, 0x1);
                store!(z2, 0x2);
                store!(z3, 0x3);
            }};
        }

        {
            while total_ncols >= 8 {
                let old = dst;
                {
                    macro_rules! load {
                        ($name: ident, $i: expr) => {
                            let $name = if $i < total_nrows {
                                (src.wrapping_byte_offset(src_byte_stride * $i as isize)
                                    as *const __m256i)
                                    .read_unaligned()
                            } else {
                                zeroed()
                            };
                        };
                    }

                    macro_rules! store {
                        ($name: expr, $i: expr) => {
                            (dst.wrapping_byte_offset($i * dst_byte_stride) as *mut __m256i)
                                .write_unaligned($name);
                        };
                    }

                    do_the_thing_8!();
                    dst = dst.wrapping_add(8);
                }
                if const { N == 8 } {
                    macro_rules! load {
                        ($name: ident, $i: expr) => {
                            let $name = if ($i + 8) < total_nrows {
                                (src.wrapping_byte_offset(src_byte_stride * ($i + 8) as isize)
                                    as *const __m256i)
                                    .read_unaligned()
                            } else {
                                zeroed()
                            };
                        };
                    }

                    macro_rules! store {
                        ($name: expr, $i: expr) => {
                            (dst.wrapping_byte_offset($i * dst_byte_stride) as *mut __m256i)
                                .write_unaligned($name);
                        };
                    }

                    do_the_thing_8!();
                } else if const { N == 4 } {
                    macro_rules! load {
                        ($name: ident, $i: expr) => {
                            let $name = if ($i + 8) < total_nrows {
                                (src.wrapping_byte_offset(src_byte_stride * ($i + 8) as isize)
                                    as *const __m128i)
                                    .read_unaligned()
                            } else {
                                zeroed()
                            };
                        };
                    }
                    {
                        macro_rules! store {
                            ($name: expr, $i: expr) => {
                                (dst.wrapping_byte_offset($i * dst_byte_stride) as *mut __m128i)
                                    .write_unaligned($name);
                            };
                        }

                        do_the_thing_4!();
                    }
                    src = src.wrapping_add(4);
                    {
                        macro_rules! store {
                            ($name: expr, $i: expr) => {
                                (dst.wrapping_byte_offset(($i + 4) * dst_byte_stride)
                                    as *mut __m128i)
                                    .write_unaligned($name);
                            };
                        }

                        do_the_thing_4!();
                    }
                    src = src.wrapping_sub(4);
                }
                dst = old.wrapping_byte_offset(8 * dst_byte_stride);
                src = src.wrapping_add(8);
                total_ncols -= 8;
            }
        }

        if total_ncols > 0 {
            {
                macro_rules! load {
                    ($name: ident, $i: expr) => {
                        let $name = if $i < total_nrows {
                            _mm256_maskload_epi32(
                                src.wrapping_byte_offset(src_byte_stride * $i as isize)
                                    as *const i32,
                                tail_mask,
                            )
                        } else {
                            zeroed()
                        };
                    };
                }

                macro_rules! store {
                    ($name: expr, $i: expr) => {
                        if $i < total_ncols {
                            (dst.wrapping_byte_offset($i * dst_byte_stride) as *mut __m256i)
                                .write_unaligned($name)
                        };
                    };
                }

                do_the_thing_8!();
                dst = dst.wrapping_add(8);
            }

            if const { N == 8 } {
                macro_rules! load {
                    ($name: ident, $i: expr) => {
                        let $name = if $i + 8 < total_nrows {
                            _mm256_maskload_epi32(
                                src.wrapping_byte_offset(src_byte_stride * ($i + 8) as isize)
                                    as *const i32,
                                tail_mask,
                            )
                        } else {
                            zeroed()
                        };
                    };
                }

                macro_rules! store {
                    ($name: expr, $i: expr) => {
                        if $i < total_ncols {
                            (dst.wrapping_byte_offset($i * dst_byte_stride) as *mut __m256i)
                                .write_unaligned($name)
                        };
                    };
                }

                do_the_thing_8!();
            } else if const { N == 4 } {
                let [tail_mask0, tail_mask1] = transmute::<_, [__m128i; 2]>(tail_mask);
                {
                    macro_rules! load {
                        ($name: ident, $i: expr) => {
                            let $name = if $i + 8 < total_nrows {
                                _mm_maskload_epi32(
                                    src.wrapping_byte_offset(src_byte_stride * ($i + 8) as isize)
                                        as *const i32,
                                    tail_mask0,
                                )
                            } else {
                                zeroed()
                            };
                        };
                    }
                    macro_rules! store {
                        ($name: expr, $i: expr) => {
                            if $i < total_ncols {
                                (dst.wrapping_byte_offset($i * dst_byte_stride) as *mut __m128i)
                                    .write_unaligned($name)
                            };
                        };
                    }

                    do_the_thing_4!();
                }
                src = src.wrapping_add(4);
                {
                    macro_rules! load {
                        ($name: ident, $i: expr) => {
                            let $name = if $i + 8 < total_nrows {
                                _mm_maskload_epi32(
                                    src.wrapping_byte_offset(src_byte_stride * ($i + 8) as isize)
                                        as *const i32,
                                    tail_mask1,
                                )
                            } else {
                                zeroed()
                            };
                        };
                    }
                    macro_rules! store {
                        ($name: expr, $i: expr) => {
                            if $i + 4 < total_ncols {
                                (dst.wrapping_byte_offset(($i + 4) * dst_byte_stride)
                                    as *mut __m128i)
                                    .write_unaligned($name)
                            };
                        };
                    }

                    do_the_thing_4!();
                }
            }
        }
    }

    let dst = dst.as_mut_ptr();
    let src = src;

    let total_ncols = ncols;
    let padded_ncols = (total_ncols + 7) / 8 * 8;
    let tail_len = padded_ncols - total_ncols;

    let tail_mask = if tail_len == 8 {
        zeroed()
    } else {
        crate::millikernel::AVX_TAIL_MASK_F32[padded_ncols - total_ncols]
    };

    if dst_byte_stride == 64 {
        imp::<8>(
            total_ncols,
            tail_mask,
            dst,
            src_byte_stride,
            dst_byte_stride,
            src,
            total_nrows,
        );
    } else if dst_byte_stride == 32 {
        imp::<0>(
            total_ncols,
            tail_mask,
            dst,
            src_byte_stride,
            dst_byte_stride,
            src,
            total_nrows,
        );
    } else if dst_byte_stride == 48 {
        imp::<4>(
            total_ncols,
            tail_mask,
            dst,
            src_byte_stride,
            dst_byte_stride,
            src,
            total_nrows,
        );
    } else {
        debug_assert!(dst_byte_stride == 24);

        let total_ncols = ncols;
        let padded_ncols = (total_ncols + 3) / 4 * 4;
        let tail_len = padded_ncols - total_ncols;

        let tail_mask = if tail_len == 4 {
            zeroed()
        } else {
            transmute::<_, [__m128i; 2]>(
                crate::millikernel::AVX_TAIL_MASK_F32[4 + (padded_ncols - total_ncols)],
            )[0]
        };

        imp(
            total_ncols,
            tail_mask,
            dst,
            src_byte_stride,
            dst_byte_stride,
            src,
            total_nrows,
        );

        #[inline(always)]
        unsafe fn imp(
            mut total_ncols: usize,
            tail_mask: __m128i,
            mut dst: *mut u32,
            src_byte_stride: isize,
            dst_byte_stride: isize,
            mut src: *const u32,
            total_nrows: usize,
        ) {
            macro_rules! do_the_thing_4 {
                () => {{
                    load!(z0, 0x0);
                    load!(z1, 0x1);
                    load!(z2, 0x2);
                    load!(z3, 0x3);

                    let [z0, z1, z2, z3] = transmute::<_, [__m128i; 4]>(
                        crate::transpose::avx_transpose_32x4(transmute([z0, z1, z2, z3])),
                    );

                    store!(z0, 0x0);
                    store!(z1, 0x1);
                    store!(z2, 0x2);
                    store!(z3, 0x3);
                }};
            }

            macro_rules! do_the_thing_2 {
                () => {{
                    load!(z0, 0x0);
                    load!(z1, 0x1);

                    let [z0, z1] = transmute::<_, [u64; 2]>(crate::transpose::avx_transpose_32x2(
                        transmute([z0, z1]),
                    ));

                    store!(z0, 0x0);
                    store!(z1, 0x1);
                }};
            }

            {
                while total_ncols >= 4 {
                    let old = dst;
                    {
                        macro_rules! load {
                            ($name: ident, $i: expr) => {
                                let $name = if $i < total_nrows {
                                    (src.wrapping_byte_offset(src_byte_stride * $i as isize)
                                        as *const __m128i)
                                        .read_unaligned()
                                } else {
                                    zeroed()
                                };
                            };
                        }

                        macro_rules! store {
                            ($name: expr, $i: expr) => {
                                (dst.wrapping_byte_offset($i * dst_byte_stride) as *mut __m128i)
                                    .write_unaligned($name);
                            };
                        }

                        do_the_thing_4!();
                        dst = dst.wrapping_add(4);
                    }
                    {
                        macro_rules! load {
                            ($name: ident, $i: expr) => {
                                let $name = if ($i + 4) < total_nrows {
                                    (src.wrapping_byte_offset(src_byte_stride * ($i + 4) as isize)
                                        as *const u64)
                                        .read_unaligned()
                                } else {
                                    zeroed()
                                };
                            };
                        }
                        {
                            macro_rules! store {
                                ($name: expr, $i: expr) => {
                                    (dst.wrapping_byte_offset($i * dst_byte_stride) as *mut u64)
                                        .write_unaligned($name);
                                };
                            }

                            do_the_thing_2!();
                        }
                        src = src.wrapping_add(2);
                        {
                            macro_rules! store {
                                ($name: expr, $i: expr) => {
                                    (dst.wrapping_byte_offset(($i + 2) * dst_byte_stride)
                                        as *mut u64)
                                        .write_unaligned($name);
                                };
                            }

                            do_the_thing_2!();
                        }
                        src = src.wrapping_sub(2);
                    }
                    dst = old.wrapping_byte_offset(4 * dst_byte_stride);
                    src = src.wrapping_add(4);
                    total_ncols -= 4;
                }
            }

            if total_ncols > 0 {
                {
                    macro_rules! load {
                        ($name: ident, $i: expr) => {
                            let $name = if $i < total_nrows {
                                _mm_maskload_epi32(
                                    src.wrapping_byte_offset(src_byte_stride * $i as isize)
                                        as *const i32,
                                    tail_mask,
                                )
                            } else {
                                zeroed()
                            };
                        };
                    }

                    macro_rules! store {
                        ($name: expr, $i: expr) => {
                            if $i < total_ncols {
                                (dst.wrapping_byte_offset($i * dst_byte_stride) as *mut __m128i)
                                    .write_unaligned($name)
                            };
                        };
                    }

                    do_the_thing_4!();
                    dst = dst.wrapping_add(4);
                }

                {
                    let [tail_mask0, tail_mask1] = transmute::<_, [u64; 2]>(tail_mask);
                    let tail_mask0 = transmute([tail_mask0, 0]);
                    let tail_mask1 = transmute([tail_mask1, 0]);
                    {
                        macro_rules! load {
                            ($name: ident, $i: expr) => {
                                let $name = if $i + 4 < total_nrows {
                                    transmute::<_, [u64; 2]>(_mm_maskload_epi32(
                                        src.wrapping_byte_offset(
                                            src_byte_stride * ($i + 4) as isize,
                                        ) as *const i32,
                                        tail_mask0,
                                    ))[0]
                                } else {
                                    zeroed()
                                };
                            };
                        }
                        macro_rules! store {
                            ($name: expr, $i: expr) => {
                                if $i < total_ncols {
                                    (dst.wrapping_byte_offset($i * dst_byte_stride) as *mut u64)
                                        .write_unaligned($name)
                                };
                            };
                        }

                        do_the_thing_2!();
                    }
                    src = src.wrapping_add(2);
                    {
                        macro_rules! load {
                            ($name: ident, $i: expr) => {
                                let $name = if $i + 4 < total_nrows {
                                    transmute::<_, [u64; 2]>(_mm_maskload_epi32(
                                        src.wrapping_byte_offset(
                                            src_byte_stride * ($i + 4) as isize,
                                        ) as *const i32,
                                        tail_mask1,
                                    ))[0]
                                } else {
                                    zeroed()
                                };
                            };
                        }
                        macro_rules! store {
                            ($name: expr, $i: expr) => {
                                if $i + 2 < total_ncols {
                                    (dst.wrapping_byte_offset(($i + 2) * dst_byte_stride)
                                        as *mut u64)
                                        .write_unaligned($name)
                                };
                            };
                        }

                        do_the_thing_2!();
                    }
                }
            }
        }
    }
}

#[target_feature(enable = "avx2")]
pub unsafe fn pack_avx_u64_row_major(
    dst: &mut [u64],
    src: *const u64,
    src_byte_stride: isize,
    dst_byte_stride: isize,
    total_nrows: usize,
    ncols: usize,
    rows_to_skip: usize,
) {
    debug_assert!(rows_to_skip == 0);

    #[inline(always)]
    unsafe fn imp<const N: usize>(
        mut total_ncols: usize,
        tail_mask: __m256i,
        mut dst: *mut u64,
        src_byte_stride: isize,
        dst_byte_stride: isize,
        mut src: *const u64,
        total_nrows: usize,
    ) {
        macro_rules! do_the_thing_4 {
            () => {{
                load!(z0, 0x0);
                load!(z1, 0x1);
                load!(z2, 0x2);
                load!(z3, 0x3);

                let [z0, z1, z2, z3] = transmute::<_, [__m256i; 4]>(
                    crate::transpose::avx_transpose_64x4(transmute([z0, z1, z2, z3])),
                );

                store!(z0, 0x0);
                store!(z1, 0x1);
                store!(z2, 0x2);
                store!(z3, 0x3);
            }};
        }

        macro_rules! do_the_thing_2 {
            () => {{
                load!(z0, 0x0);
                load!(z1, 0x1);

                let [z0, z1] = transmute::<_, [__m128i; 2]>(crate::transpose::avx_transpose_64x2(
                    transmute([z0, z1]),
                ));

                store!(z0, 0x0);
                store!(z1, 0x1);
            }};
        }

        {
            while total_ncols >= 4 {
                let old = dst;
                {
                    macro_rules! load {
                        ($name: ident, $i: expr) => {
                            let $name = if $i < total_nrows {
                                (src.wrapping_byte_offset(src_byte_stride * $i as isize)
                                    as *const __m256i)
                                    .read_unaligned()
                            } else {
                                zeroed()
                            };
                        };
                    }

                    macro_rules! store {
                        ($name: expr, $i: expr) => {
                            (dst.wrapping_byte_offset($i * dst_byte_stride) as *mut __m256i)
                                .write_unaligned($name);
                        };
                    }

                    do_the_thing_4!();
                    dst = dst.wrapping_add(4);
                }
                if const { N == 4 } {
                    macro_rules! load {
                        ($name: ident, $i: expr) => {
                            let $name = if ($i + 4) < total_nrows {
                                (src.wrapping_byte_offset(src_byte_stride * ($i + 4) as isize)
                                    as *const __m256i)
                                    .read_unaligned()
                            } else {
                                zeroed()
                            };
                        };
                    }

                    macro_rules! store {
                        ($name: expr, $i: expr) => {
                            (dst.wrapping_byte_offset($i * dst_byte_stride) as *mut __m256i)
                                .write_unaligned($name);
                        };
                    }

                    do_the_thing_4!();
                } else if const { N == 2 } {
                    macro_rules! load {
                        ($name: ident, $i: expr) => {
                            let $name = if ($i + 4) < total_nrows {
                                (src.wrapping_byte_offset(src_byte_stride * ($i + 4) as isize)
                                    as *const __m128i)
                                    .read_unaligned()
                            } else {
                                zeroed()
                            };
                        };
                    }
                    {
                        macro_rules! store {
                            ($name: expr, $i: expr) => {
                                (dst.wrapping_byte_offset($i * dst_byte_stride) as *mut __m128i)
                                    .write_unaligned($name);
                            };
                        }

                        do_the_thing_2!();
                    }
                    src = src.wrapping_add(2);
                    {
                        macro_rules! store {
                            ($name: expr, $i: expr) => {
                                (dst.wrapping_byte_offset(($i + 2) * dst_byte_stride)
                                    as *mut __m128i)
                                    .write_unaligned($name);
                            };
                        }

                        do_the_thing_2!();
                    }
                    src = src.wrapping_sub(2);
                }
                dst = old.wrapping_byte_offset(4 * dst_byte_stride);
                src = src.wrapping_add(4);
                total_ncols -= 4;
            }
        }

        if total_ncols > 0 {
            {
                macro_rules! load {
                    ($name: ident, $i: expr) => {
                        let $name = if $i < total_nrows {
                            _mm256_maskload_epi64(
                                src.wrapping_byte_offset(src_byte_stride * $i as isize)
                                    as *const i64,
                                tail_mask,
                            )
                        } else {
                            zeroed()
                        };
                    };
                }

                macro_rules! store {
                    ($name: expr, $i: expr) => {
                        if $i < total_ncols {
                            (dst.wrapping_byte_offset($i * dst_byte_stride) as *mut __m256i)
                                .write_unaligned($name)
                        };
                    };
                }

                do_the_thing_4!();
                dst = dst.wrapping_add(4);
            }

            if const { N == 4 } {
                macro_rules! load {
                    ($name: ident, $i: expr) => {
                        let $name = if $i + 4 < total_nrows {
                            _mm256_maskload_epi64(
                                src.wrapping_byte_offset(src_byte_stride * ($i + 4) as isize)
                                    as *const i64,
                                tail_mask,
                            )
                        } else {
                            zeroed()
                        };
                    };
                }

                macro_rules! store {
                    ($name: expr, $i: expr) => {
                        if $i < total_ncols {
                            (dst.wrapping_byte_offset($i * dst_byte_stride) as *mut __m256i)
                                .write_unaligned($name)
                        };
                    };
                }

                do_the_thing_4!();
            } else if const { N == 2 } {
                let [tail_mask0, tail_mask1] = transmute::<_, [__m128i; 2]>(tail_mask);
                {
                    macro_rules! load {
                        ($name: ident, $i: expr) => {
                            let $name = if $i + 4 < total_nrows {
                                _mm_maskload_epi64(
                                    src.wrapping_byte_offset(src_byte_stride * ($i + 4) as isize)
                                        as *const i64,
                                    tail_mask0,
                                )
                            } else {
                                zeroed()
                            };
                        };
                    }
                    macro_rules! store {
                        ($name: expr, $i: expr) => {
                            if $i < total_ncols {
                                (dst.wrapping_byte_offset($i * dst_byte_stride) as *mut __m128i)
                                    .write_unaligned($name)
                            };
                        };
                    }

                    do_the_thing_2!();
                }
                src = src.wrapping_add(2);
                {
                    macro_rules! load {
                        ($name: ident, $i: expr) => {
                            let $name = if $i + 4 < total_nrows {
                                _mm_maskload_epi64(
                                    src.wrapping_byte_offset(src_byte_stride * ($i + 4) as isize)
                                        as *const i64,
                                    tail_mask1,
                                )
                            } else {
                                zeroed()
                            };
                        };
                    }
                    macro_rules! store {
                        ($name: expr, $i: expr) => {
                            if $i + 2 < total_ncols {
                                (dst.wrapping_byte_offset(($i + 2) * dst_byte_stride)
                                    as *mut __m128i)
                                    .write_unaligned($name)
                            };
                        };
                    }

                    do_the_thing_2!();
                }
            }
        }
    }

    let total_ncols = ncols;
    let padded_ncols = (total_ncols + 3) / 4 * 4;
    let tail_len = padded_ncols - total_ncols;

    let dst = dst.as_mut_ptr();
    let src = src;

    let tail_mask = if tail_len == 4 {
        zeroed()
    } else {
        crate::millikernel::AVX_TAIL_MASK_F64[padded_ncols - total_ncols]
    };

    if dst_byte_stride == 64 {
        imp::<4>(
            total_ncols,
            tail_mask,
            dst,
            src_byte_stride,
            dst_byte_stride,
            src,
            total_nrows,
        );
    } else if dst_byte_stride == 32 {
        imp::<0>(
            total_ncols,
            tail_mask,
            dst,
            src_byte_stride,
            dst_byte_stride,
            src,
            total_nrows,
        );
    } else {
        debug_assert!(dst_byte_stride == 48);
        imp::<2>(
            total_ncols,
            tail_mask,
            dst,
            src_byte_stride,
            dst_byte_stride,
            src,
            total_nrows,
        );
    }
}

#[target_feature(enable = "avx2")]
pub unsafe fn pack_avx_u128_row_major(
    dst: &mut [u128],
    src: *const u128,
    src_byte_stride: isize,
    dst_byte_stride: isize,
    total_nrows: usize,
    ncols: usize,
    rows_to_skip: usize,
) {
    debug_assert!(rows_to_skip == 0);

    #[inline(always)]
    unsafe fn imp(
        n: usize,
        mut total_ncols: usize,
        tail_mask: __m256i,
        mut dst: *mut u128,
        src_byte_stride: isize,
        dst_byte_stride: isize,
        mut src: *const u128,
        total_nrows: usize,
    ) {
        macro_rules! do_the_thing_2 {
            () => {{
                load!(z0, 0x0);
                load!(z1, 0x1);

                let [z0, z1] = transmute::<_, [__m256i; 2]>(crate::transpose::avx_transpose_128x2(
                    transmute([z0, z1]),
                ));

                store!(z0, 0x0);
                store!(z1, 0x1);
            }};
        }

        {
            while total_ncols >= 2 {
                let old = dst;

                for i in 0..n {
                    macro_rules! load {
                        ($name: ident, $i: expr) => {
                            let $name = if $i < total_nrows {
                                (src.wrapping_byte_offset(src_byte_stride * ($i + i * 2) as isize)
                                    as *const __m256i)
                                    .read_unaligned()
                            } else {
                                zeroed()
                            };
                        };
                    }

                    macro_rules! store {
                        ($name: expr, $i: expr) => {
                            (dst.wrapping_byte_offset($i * dst_byte_stride) as *mut __m256i)
                                .write_unaligned($name);
                        };
                    }

                    do_the_thing_2!();
                    dst = dst.wrapping_add(2);
                }

                dst = old.wrapping_byte_offset(2 * dst_byte_stride);
                src = src.wrapping_add(2);
                total_ncols -= 2;
            }
        }

        if total_ncols > 0 {
            for i in 0..n {
                macro_rules! load {
                    ($name: ident, $i: expr) => {
                        let $name = if ($i + i * 2) < total_nrows {
                            _mm256_maskload_epi64(
                                src.wrapping_byte_offset(src_byte_stride * ($i + i * 2) as isize)
                                    as *const i64,
                                tail_mask,
                            )
                        } else {
                            zeroed()
                        };
                    };
                }

                macro_rules! store {
                    ($name: expr, $i: expr) => {
                        if $i < total_ncols {
                            (dst.wrapping_byte_offset($i * dst_byte_stride) as *mut __m256i)
                                .write_unaligned($name)
                        };
                    };
                }

                do_the_thing_2!();
                dst = dst.wrapping_add(2);
            }
        }
    }

    let total_ncols = ncols;
    let padded_ncols = (total_ncols + 1) / 2 * 2;
    let tail_len = padded_ncols - total_ncols;

    let tail_mask = if tail_len == 2 {
        zeroed()
    } else {
        crate::millikernel::AVX_TAIL_MASK_F64[2 * (padded_ncols - total_ncols)]
    };

    let n = if dst_byte_stride == 64 {
        2
    } else if dst_byte_stride == 32 {
        1
    } else {
        debug_assert!(dst_byte_stride == 96);
        3
    };

    let dst = dst.as_mut_ptr();
    imp(
        n,
        total_ncols,
        tail_mask,
        dst,
        src_byte_stride,
        dst_byte_stride,
        src,
        total_nrows,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "nightly")]
    #[test]
    fn test_avx512_32_row_major() {
        for m in [12, 16, 32] {
            for n in 0..111 {
                for skip in 0..1usize {
                    let mut src = vec![0u32; m * n + skip.next_multiple_of(16)];
                    let mut dst = vec![0u32; m * n + skip.next_multiple_of(16)];

                    let src = &mut src[skip..][..m * n];
                    let dst = &mut dst[skip..][..m * n];

                    for x in &mut *src {
                        *x = rand::random();
                    }

                    unsafe {
                        pack_avx512_u32_row_major(
                            &mut *dst,
                            src.as_ptr(),
                            (n * size_of::<u32>()) as isize,
                            (m * size_of::<u32>()) as isize,
                            m,
                            n,
                            0,
                        )
                    };

                    for j in 0..n {
                        for i in 0..m {
                            assert_eq!(dst[m * j + i], src[n * i + j]);
                        }
                    }
                }
            }
        }
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_avx512_64_row_major() {
        for m in [12, 8, 16] {
            for n in 0..111 {
                for skip in 0..1usize {
                    let mut src = vec![0u64; m * n + skip.next_multiple_of(16)];
                    let mut dst = vec![0u64; m * n + skip.next_multiple_of(16)];

                    let src = &mut src[skip..][..m * n];
                    let dst = &mut dst[skip..][..m * n];

                    for x in &mut *src {
                        *x = rand::random();
                    }

                    unsafe {
                        pack_avx512_u64_row_major(
                            &mut *dst,
                            src.as_ptr(),
                            (n * size_of::<u64>()) as isize,
                            (m * size_of::<u64>()) as isize,
                            m,
                            n,
                            0,
                        )
                    };

                    for j in 0..n {
                        for i in 0..m {
                            assert_eq!(dst[m * j + i], src[n * i + j]);
                        }
                    }
                }
            }
        }
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_avx512_128_row_major() {
        for m in [12, 4, 8] {
            for n in 0..111 {
                for skip in 0..1usize {
                    let mut src = vec![0u128; m * n + skip.next_multiple_of(16)];
                    let mut dst = vec![0u128; m * n + skip.next_multiple_of(16)];

                    let src = &mut src[skip..][..m * n];
                    let dst = &mut dst[skip..][..m * n];

                    for x in &mut *src {
                        *x = rand::random();
                    }

                    unsafe {
                        pack_avx512_u128_row_major(
                            &mut *dst,
                            src.as_ptr(),
                            (n * size_of::<u128>()) as isize,
                            (m * size_of::<u128>()) as isize,
                            m,
                            n,
                            0,
                        )
                    };

                    for j in 0..n {
                        for i in 0..m {
                            assert_eq!(dst[m * j + i], src[n * i + j]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_avx_32_row_major() {
        for m in [6, 8, 16] {
            for n in 0..111 {
                for skip in 0..1usize {
                    let mut src = vec![0u32; m * n + skip.next_multiple_of(16)];
                    let mut dst = vec![0u32; m * n + skip.next_multiple_of(16)];

                    let src = &mut src[skip..][..m * n];
                    let dst = &mut dst[skip..][..m * n];

                    for x in &mut *src {
                        *x = rand::random();
                    }

                    unsafe {
                        pack_avx_u32_row_major(
                            &mut *dst,
                            src.as_ptr(),
                            (n * size_of::<u32>()) as isize,
                            (m * size_of::<u32>()) as isize,
                            m,
                            n,
                            0,
                        )
                    };

                    for j in 0..n {
                        for i in 0..m {
                            assert_eq!(dst[m * j + i], src[n * i + j]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_avx_64_row_major() {
        for m in [6, 4, 8] {
            for n in 0..111 {
                for skip in 0..1usize {
                    let mut src = vec![0u64; m * n + skip.next_multiple_of(16)];
                    let mut dst = vec![0u64; m * n + skip.next_multiple_of(16)];

                    let src = &mut src[skip..][..m * n];
                    let dst = &mut dst[skip..][..m * n];

                    for x in &mut *src {
                        *x = rand::random();
                    }

                    unsafe {
                        pack_avx_u64_row_major(
                            &mut *dst,
                            src.as_ptr(),
                            (n * size_of::<u64>()) as isize,
                            (m * size_of::<u64>()) as isize,
                            m,
                            n,
                            0,
                        )
                    };

                    for j in 0..n {
                        for i in 0..m {
                            assert_eq!(dst[m * j + i], src[n * i + j]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_avx_128_row_major() {
        for m in [6, 2, 4] {
            for n in 0..111 {
                for skip in 0..1usize {
                    let mut src = vec![0u128; m * n + skip.next_multiple_of(16)];
                    let mut dst = vec![0u128; m * n + skip.next_multiple_of(16)];

                    let src = &mut src[skip..][..m * n];
                    let dst = &mut dst[skip..][..m * n];

                    for x in &mut *src {
                        *x = rand::random();
                    }

                    unsafe {
                        pack_avx_u128_row_major(
                            &mut *dst,
                            src.as_ptr(),
                            (n * size_of::<u128>()) as isize,
                            (m * size_of::<u128>()) as isize,
                            m,
                            n,
                            0,
                        )
                    };

                    for j in 0..n {
                        for i in 0..m {
                            assert_eq!(dst[m * j + i], src[n * i + j]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_avx_128_col_major() {
        for m in [6, 2, 4] {
            for n in 0..111 {
                for skip in 0..1usize {
                    let mut src = vec![0u128; m * n + skip.next_multiple_of(16)];
                    let mut dst = vec![0u128; m * n + skip.next_multiple_of(16)];

                    let src = &mut src[skip..][..m * n];
                    let dst = &mut dst[skip..][..m * n];

                    for x in &mut *src {
                        *x = rand::random();
                    }

                    unsafe {
                        pack_avx_u128_col_major(
                            &mut *dst,
                            src.as_ptr(),
                            (m * size_of::<u128>()) as isize,
                            (m * size_of::<u128>()) as isize,
                            m,
                            n,
                            0,
                        )
                    };

                    for j in 0..n {
                        for i in 0..m {
                            assert_eq!(dst[n * i + j], src[n * i + j]);
                        }
                    }
                }
            }
        }
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_avx512_128_col_major() {
        for m in [12, 4, 8] {
            for n in 0..111 {
                for skip in 0..1usize {
                    let mut src = vec![0u128; m * n + skip.next_multiple_of(16)];
                    let mut dst = vec![0u128; m * n + skip.next_multiple_of(16)];

                    let src = &mut src[skip..][..m * n];
                    let dst = &mut dst[skip..][..m * n];

                    for x in &mut *src {
                        *x = rand::random();
                    }

                    unsafe {
                        pack_avx512_u128_col_major(
                            &mut *dst,
                            src.as_ptr(),
                            (m * size_of::<u128>()) as isize,
                            (m * size_of::<u128>()) as isize,
                            m,
                            n,
                            0,
                        )
                    };

                    for j in 0..n {
                        for i in 0..m {
                            assert_eq!(dst[n * i + j], src[n * i + j]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_avx_32_col_major() {
        for m in [6, 8, 16] {
            for n in 0..111 {
                for skip in 0..1usize {
                    let mut src = vec![0u32; m * n + skip.next_multiple_of(16)];
                    let mut dst = vec![0u32; m * n + skip.next_multiple_of(16)];

                    let src = &mut src[skip..][..m * n];
                    let dst = &mut dst[skip..][..m * n];

                    for x in &mut *src {
                        *x = rand::random();
                    }

                    unsafe {
                        pack_avx_u32_col_major(
                            &mut *dst,
                            src.as_ptr(),
                            (m * size_of::<u32>()) as isize,
                            (m * size_of::<u32>()) as isize,
                            m,
                            n,
                            0,
                        )
                    };

                    for j in 0..n {
                        for i in 0..m {
                            assert_eq!(dst[n * i + j], src[n * i + j]);
                        }
                    }
                }
            }
        }
    }
}

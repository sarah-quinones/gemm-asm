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
        let tail_mask = crate::millikernel::AVX512_TAIL_MASK_F32[(total_nrows - 1) % 16];

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
        } else {
            debug_assert!(total_nrows <= 32);
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
        }
    }
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
        let tail_mask = crate::millikernel::AVX_TAIL_MASK_F32[(total_nrows - 1) % 8];

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
        } else {
            debug_assert!(total_nrows <= 32);
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
        }
    }
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
    #[inline(always)]
    unsafe fn imp<const N: usize>(
        cols_to_skip: usize,
        mut total_ncols: usize,
        mut head_mask: u16,
        tail_mask: u16,
        mut dst: *mut u32,
        src_byte_stride: isize,
        dst_byte_stride: isize,
        mut src: *const u32,
        rows_to_skip: usize,
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

        if cols_to_skip > 0 {
            if total_ncols < 16 {
                head_mask &= tail_mask;
            }

            let old = dst;
            {
                macro_rules! load {
                    ($name: ident, $i: expr) => {
                        let $name = if $i >= rows_to_skip && $i < total_nrows {
                            _mm512_maskz_loadu_epi32(
                                head_mask,
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
                        if $i >= cols_to_skip && $i < total_ncols {
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
                        let $name = if $i + 16 >= rows_to_skip && $i + 16 < total_nrows {
                            _mm512_maskz_loadu_epi32(
                                head_mask,
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
                        if $i >= cols_to_skip && $i < total_ncols {
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
                            let $name = if $i + 16 >= rows_to_skip && $i + 16 < total_nrows {
                                _mm256_maskz_loadu_epi32(
                                    head_mask as u8,
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
                            if $i >= cols_to_skip && $i < total_ncols {
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
                            let $name = if $i + 16 >= rows_to_skip && $i + 16 < total_nrows {
                                _mm256_maskz_loadu_epi32(
                                    (head_mask >> 8) as u8,
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
                            if ($i + 8) >= cols_to_skip && $i + 8 < total_ncols {
                                (dst.wrapping_byte_offset(($i + 8) * dst_byte_stride)
                                    as *mut __m256i)
                                    .write_unaligned($name)
                            };
                        };
                    }

                    do_the_thing_8!();
                }
                src = src.wrapping_sub(8);
            }

            dst = old.wrapping_byte_offset(16 * dst_byte_stride);
            src = src.wrapping_add(16);
            total_ncols = total_ncols.saturating_sub(16);
        }

        if rows_to_skip == 0 && total_nrows == 32 {
            while total_ncols >= 16 {
                let old = dst;
                {
                    macro_rules! load {
                        ($name: ident, $i: expr) => {
                            let $name = (src.wrapping_byte_offset(src_byte_stride * $i as isize)
                                as *const __m512i)
                                .read_unaligned();
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
                            let $name = (src
                                .wrapping_byte_offset(src_byte_stride * ($i + 16) as isize)
                                as *const __m512i)
                                .read_unaligned();
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
                            let $name = (src
                                .wrapping_byte_offset(src_byte_stride * ($i + 16) as isize)
                                as *const __m256i)
                                .read_unaligned();
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
        } else {
            while total_ncols >= 16 {
                let old = dst;
                {
                    macro_rules! load {
                        ($name: ident, $i: expr) => {
                            let $name = if $i >= rows_to_skip && $i < total_nrows {
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
                            let $name = if ($i + 16) >= rows_to_skip && ($i + 16) < total_nrows {
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
                            let $name = if ($i + 16) >= rows_to_skip && ($i + 16) < total_nrows {
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
                        let $name = if $i >= rows_to_skip && $i < total_nrows {
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
                        let $name = if $i + 16 >= rows_to_skip && $i + 16 < total_nrows {
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
                            let $name = if $i + 16 >= rows_to_skip && $i + 16 < total_nrows {
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
                            let $name = if $i + 16 >= rows_to_skip && $i + 16 < total_nrows {
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

    let offset = src.align_offset(64);
    let cols_to_skip = (16 - offset) % 16;

    let total_ncols = ncols + cols_to_skip;
    let padded_ncols = (total_ncols + 15) / 16 * 16;
    let tail_len = padded_ncols - total_ncols;

    let dst = dst
        .as_mut_ptr()
        .wrapping_byte_offset(cols_to_skip as isize * -dst_byte_stride);
    let src = src.wrapping_sub(cols_to_skip);

    let head_mask = crate::millikernel::AVX512_HEAD_MASK_F32[cols_to_skip];
    let tail_mask = if tail_len == 16 {
        0
    } else {
        crate::millikernel::AVX512_TAIL_MASK_F32[padded_ncols - total_ncols]
    };

    if dst_byte_stride == 128 {
        imp::<16>(
            cols_to_skip,
            total_ncols,
            head_mask,
            tail_mask,
            dst,
            src_byte_stride,
            dst_byte_stride,
            src,
            rows_to_skip,
            total_nrows,
        );
    } else if dst_byte_stride == 64 {
        imp::<0>(
            cols_to_skip,
            total_ncols,
            head_mask,
            tail_mask,
            dst,
            src_byte_stride,
            dst_byte_stride,
            src,
            rows_to_skip,
            total_nrows,
        );
    } else {
        debug_assert!(dst_byte_stride == 96);
        imp::<8>(
            cols_to_skip,
            total_ncols,
            head_mask,
            tail_mask,
            dst,
            src_byte_stride,
            dst_byte_stride,
            src,
            rows_to_skip,
            total_nrows,
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
    #[inline(always)]
    unsafe fn imp<const N: usize>(
        cols_to_skip: usize,
        mut total_ncols: usize,
        mut head_mask: u8,
        tail_mask: u8,
        mut dst: *mut u64,
        src_byte_stride: isize,
        dst_byte_stride: isize,
        mut src: *const u64,
        rows_to_skip: usize,
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

        if cols_to_skip > 0 {
            if total_ncols < 8 {
                head_mask &= tail_mask;
            }

            let old = dst;
            {
                macro_rules! load {
                    ($name: ident, $i: expr) => {
                        let $name = if $i >= rows_to_skip && $i < total_nrows {
                            _mm512_maskz_loadu_epi64(
                                head_mask,
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
                        if $i >= cols_to_skip && $i < total_ncols {
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
                        let $name = if $i + 8 >= rows_to_skip && $i + 8 < total_nrows {
                            _mm512_maskz_loadu_epi64(
                                head_mask,
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
                        if $i >= cols_to_skip && $i < total_ncols {
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
                            let $name = if $i + 8 >= rows_to_skip && $i + 8 < total_nrows {
                                _mm256_maskz_loadu_epi64(
                                    head_mask,
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
                            if $i >= cols_to_skip && $i < total_ncols {
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
                            let $name = if $i + 8 >= rows_to_skip && $i + 8 < total_nrows {
                                _mm256_maskz_loadu_epi64(
                                    (head_mask >> 4),
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
                            if ($i + 4) >= cols_to_skip && $i + 4 < total_ncols {
                                (dst.wrapping_byte_offset(($i + 4) * dst_byte_stride)
                                    as *mut __m256i)
                                    .write_unaligned($name)
                            };
                        };
                    }

                    do_the_thing_4!();
                }
                src = src.wrapping_sub(4);
            }

            dst = old.wrapping_byte_offset(8 * dst_byte_stride);
            src = src.wrapping_add(8);
            total_ncols = total_ncols.saturating_sub(8);
        }

        if rows_to_skip == 0 && total_nrows == 64 {
            while total_ncols >= 8 {
                let old = dst;
                {
                    macro_rules! load {
                        ($name: ident, $i: expr) => {
                            let $name = (src.wrapping_byte_offset(src_byte_stride * $i as isize)
                                as *const __m512i)
                                .read_unaligned();
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
                            let $name = (src
                                .wrapping_byte_offset(src_byte_stride * ($i + 8) as isize)
                                as *const __m512i)
                                .read_unaligned();
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
                            let $name = (src
                                .wrapping_byte_offset(src_byte_stride * ($i + 8) as isize)
                                as *const __m256i)
                                .read_unaligned();
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
        } else {
            while total_ncols >= 8 {
                let old = dst;
                {
                    macro_rules! load {
                        ($name: ident, $i: expr) => {
                            let $name = if $i >= rows_to_skip && $i < total_nrows {
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
                            let $name = if ($i + 8) >= rows_to_skip && ($i + 8) < total_nrows {
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
                            let $name = if ($i + 8) >= rows_to_skip && ($i + 8) < total_nrows {
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
                        let $name = if $i >= rows_to_skip && $i < total_nrows {
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
                        let $name = if $i + 8 >= rows_to_skip && $i + 8 < total_nrows {
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
                            let $name = if $i + 8 >= rows_to_skip && $i + 8 < total_nrows {
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
                            let $name = if $i + 8 >= rows_to_skip && $i + 8 < total_nrows {
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

    let offset = src.align_offset(64);
    let cols_to_skip = (8 - offset) % 8;

    let total_ncols = ncols + cols_to_skip;
    let padded_ncols = (total_ncols + 7) / 8 * 8;
    let tail_len = padded_ncols - total_ncols;

    let dst = dst
        .as_mut_ptr()
        .wrapping_byte_offset(cols_to_skip as isize * -dst_byte_stride);
    let src = src.wrapping_sub(cols_to_skip);

    let head_mask = crate::millikernel::AVX512_HEAD_MASK_F64[cols_to_skip];
    let tail_mask = if tail_len == 8 {
        0
    } else {
        crate::millikernel::AVX512_TAIL_MASK_F64[padded_ncols - total_ncols]
    };

    if dst_byte_stride == 128 {
        imp::<8>(
            cols_to_skip,
            total_ncols,
            head_mask,
            tail_mask,
            dst,
            src_byte_stride,
            dst_byte_stride,
            src,
            rows_to_skip,
            total_nrows,
        );
    } else if dst_byte_stride == 64 {
        imp::<0>(
            cols_to_skip,
            total_ncols,
            head_mask,
            tail_mask,
            dst,
            src_byte_stride,
            dst_byte_stride,
            src,
            rows_to_skip,
            total_nrows,
        );
    } else {
        debug_assert!(dst_byte_stride == 96);
        imp::<4>(
            cols_to_skip,
            total_ncols,
            head_mask,
            tail_mask,
            dst,
            src_byte_stride,
            dst_byte_stride,
            src,
            rows_to_skip,
            total_nrows,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "nightly")]
    #[test]
    fn test_avx512_u32_row_major() {
        for m in [16, 24, 32] {
            for n in 0..111 {
                for skip in 0..32usize {
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
        for m in [8, 12, 16] {
            for n in 0..111 {
                for skip in 0..32usize {
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
}

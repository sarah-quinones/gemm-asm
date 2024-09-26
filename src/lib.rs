use core::arch::x86_64::{__m256i, __m512d};

core::arch::global_asm!(include_str!(concat!(env!("OUT_DIR"), "/codegen.s")));
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[no_mangle]
static LIBFAER_GEMM_COMPLEX64_MASK_REAL: __m512d = unsafe {
    core::mem::transmute([
        -0.0, 0.0, //
        -0.0, 0.0, //
        -0.0, 0.0, //
        -0.0, 0.0f64, //
    ])
};
#[no_mangle]
static LIBFAER_GEMM_COMPLEX64_MASK_IMAG: __m512d = unsafe {
    core::mem::transmute([
        0.0, -0.0, //
        0.0, -0.0, //
        0.0, -0.0, //
        0.0, -0.0f64, //
    ])
};

#[no_mangle]
static LIBFAER_GEMM_COMPLEX32_MASK_REAL: __m512d = unsafe {
    core::mem::transmute([
        -0.0, 0.0, //
        -0.0, 0.0, //
        -0.0, 0.0, //
        -0.0, 0.0, //
        -0.0, 0.0, //
        -0.0, 0.0, //
        -0.0, 0.0, //
        -0.0, 0.0f32, //
    ])
};
#[no_mangle]
static LIBFAER_GEMM_COMPLEX32_MASK_IMAG: __m512d = unsafe {
    core::mem::transmute([
        0.0, -0.0, //
        0.0, -0.0, //
        0.0, -0.0, //
        0.0, -0.0, //
        0.0, -0.0, //
        0.0, -0.0, //
        0.0, -0.0, //
        0.0, -0.0f32, //
    ])
};

#[repr(C)]
pub struct Plan {
    // 0
    head_mask: *const (),
    // 8
    tail_mask: *const (),
    // 16
    top_left: unsafe extern "C" fn(),
    // 24
    top_right: unsafe extern "C" fn(),
    // 32
    mid_left: unsafe extern "C" fn(),
    // 40
    mid_right: unsafe extern "C" fn(),
    // 48
    bot_left: unsafe extern "C" fn(),
    // 56
    bot_right: unsafe extern "C" fn(),

    // 64
    mid_height: usize,
    // 72
    left_width: usize,
    // 80
    flags: usize,
    // 88
    do_nothing: bool,
    // 89
    two_or_more: bool,

    pub mr: usize,
    pub nr: usize,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Accum {
    Add,
    Replace,
}

const fn real_plan_impl<const MR: usize, const NR: usize, const N: usize, M>(
    extra_masked_top_rows: usize,
    nrows: usize,
    ncols: usize,
    dst: Accum,
    ukr: &'static [[unsafe extern "C" fn(); MR]; NR],
    head_masks: &'static [M; N],
    tail_masks: &'static [M; N],
) -> Plan {
    assert!(extra_masked_top_rows < N);

    let full_rows = nrows + extra_masked_top_rows;
    let padded_rows = full_rows.next_multiple_of(N);

    let bottom_mr = (padded_rows / N + MR - 1) % MR + 1;
    let right_nr = (ncols + NR - 1) % NR + 1;

    let left_width = ncols.saturating_sub(1) / NR;
    let mid_height = ((padded_rows / N).saturating_sub(1) / MR).saturating_sub(1);

    let two_or_more = padded_rows > MR * N;

    let bit0 = match dst {
        Accum::Add => 1,
        Accum::Replace => 0,
    };
    let bit1 = if extra_masked_top_rows > 0 { 1 } else { 0 };
    let bit2 = if padded_rows - full_rows > 0 { 1 } else { 0 };

    Plan {
        head_mask: core::ptr::from_ref(&head_masks[extra_masked_top_rows]) as *const _,
        tail_mask: core::ptr::from_ref(&tail_masks[padded_rows - full_rows]) as *const _,
        top_left: ukr[NR - 1][MR - 1],
        mid_left: ukr[NR - 1][MR - 1],
        top_right: ukr[right_nr - 1][MR - 1],
        mid_right: ukr[right_nr - 1][MR - 1],
        bot_left: ukr[NR - 1][bottom_mr - 1],
        bot_right: ukr[right_nr - 1][bottom_mr - 1],
        mid_height,
        left_width,
        flags: bit0 | (bit1 << 1) | (bit2 << 2),
        do_nothing: nrows == 0 || ncols == 0,
        two_or_more,
        mr: MR * N,
        nr: NR,
    }
}

core::arch::global_asm! {
    ".globl libfaer_gemm_nop",
    "libfaer_gemm_nop:",
    "ret",
}

extern "C" {
    pub fn libfaer_gemm_nop();
}

const fn real_dot_plan_impl<const MR: usize, const NR: usize, const N: usize, M>(
    extra_masked_top_rows: usize,
    nrows_plus_extra: usize,
    lhs_ncols: usize,
    rhs_ncols: usize,
    dst: Accum,
    ukr: &'static [[unsafe extern "C" fn(); MR]; NR],
    head_masks: &'static [M; N],
    tail_masks: &'static [M; N],
) -> Plan {
    assert!(extra_masked_top_rows < N);
    assert!(nrows_plus_extra >= extra_masked_top_rows);

    let full_rows = nrows_plus_extra;
    let padded_rows = full_rows.next_multiple_of(N);

    let bottom_mr = (lhs_ncols + MR - 1) % MR + 1;
    let right_nr = (rhs_ncols + NR - 1) % NR + 1;

    let left_width = rhs_ncols.saturating_sub(1) / NR;
    let mid_height = lhs_ncols.saturating_sub(1) / MR;

    let two_or_more = mid_height != 0;
    let mid_height = mid_height.saturating_sub(1);

    let bit0 = match dst {
        Accum::Add => 1,
        Accum::Replace => 0,
    };
    let bit1 = 1;
    let bit2 = 1;

    Plan {
        head_mask: core::ptr::from_ref(&head_masks[extra_masked_top_rows]) as *const _,
        tail_mask: core::ptr::from_ref(&tail_masks[padded_rows - full_rows]) as *const _,

        top_left: ukr[NR - 1][MR - 1],
        mid_left: ukr[NR - 1][MR - 1],
        top_right: ukr[right_nr - 1][MR - 1],
        mid_right: ukr[right_nr - 1][MR - 1],
        bot_left: ukr[NR - 1][bottom_mr - 1],
        bot_right: ukr[right_nr - 1][bottom_mr - 1],

        mid_height,
        left_width,
        flags: bit0 | (bit1 << 1) | (bit2 << 2),
        do_nothing: lhs_ncols == 0 || rhs_ncols == 0,
        two_or_more,
        mr: MR,
        nr: NR,
    }
}

const fn cplx_plan_impl<const MR: usize, const NR: usize, const N: usize, M: core::fmt::Debug>(
    extra_masked_top_rows: usize,
    nrows: usize,
    ncols: usize,
    dst: Accum,
    conj_lhs: bool,
    conj_rhs: bool,
    ukr: &'static [[unsafe extern "C" fn(); MR]; NR],
    head_masks: &'static [M; N],
    tail_masks: &'static [M; N],
) -> Plan {
    assert!(extra_masked_top_rows < N);

    let full_rows = nrows + extra_masked_top_rows;
    let padded_rows = full_rows.next_multiple_of(N);

    let bottom_mr = (padded_rows / N + MR - 1) % MR + 1;
    let right_nr = (ncols + NR - 1) % NR + 1;

    let left_width = ncols.saturating_sub(1) / NR;
    let mid_height = ((padded_rows / N).saturating_sub(1) / MR).saturating_sub(1);

    let mid_chunk_only = padded_rows > MR * N;

    let bit0 = match dst {
        Accum::Add => 1,
        Accum::Replace => 0,
    };
    let bit1 = if extra_masked_top_rows > 0 { 1 } else { 0 };
    let bit2 = if padded_rows - full_rows > 0 { 1 } else { 0 };
    let bit3 = (conj_lhs == conj_rhs) as usize;
    let bit4 = conj_rhs as usize;

    Plan {
        head_mask: core::ptr::from_ref(&head_masks[extra_masked_top_rows]) as *const _,
        tail_mask: core::ptr::from_ref(&tail_masks[padded_rows - full_rows]) as *const _,
        top_left: ukr[NR - 1][MR - 1],
        mid_left: ukr[NR - 1][MR - 1],
        top_right: ukr[right_nr - 1][MR - 1],
        mid_right: ukr[right_nr - 1][MR - 1],
        bot_left: ukr[NR - 1][bottom_mr - 1],
        bot_right: ukr[right_nr - 1][bottom_mr - 1],
        mid_height,
        left_width,
        flags: bit0 | (bit1 << 1) | (bit2 << 2) | (bit3 << 3) | (bit4 << 4),
        do_nothing: nrows == 0 || ncols == 0,
        two_or_more: mid_chunk_only,
        mr: MR * N,
        nr: NR,
    }
}

const fn cplx_dot_plan_impl<const MR: usize, const NR: usize, const N: usize, M>(
    extra_masked_top_rows: usize,
    nrows_plus_extra: usize,
    lhs_ncols: usize,
    rhs_ncols: usize,
    dst: Accum,
    conj_lhs: bool,
    conj_rhs: bool,
    ukr: &'static [[unsafe extern "C" fn(); MR]; NR],
    head_masks: &'static [M; N],
    tail_masks: &'static [M; N],
) -> Plan {
    assert!(extra_masked_top_rows < N);
    assert!(nrows_plus_extra >= extra_masked_top_rows);

    let full_rows = nrows_plus_extra;
    let padded_rows = full_rows.next_multiple_of(N);

    let bottom_mr = (lhs_ncols + MR - 1) % MR + 1;
    let right_nr = (rhs_ncols + NR - 1) % NR + 1;

    let left_width = rhs_ncols.saturating_sub(1) / NR;
    let mid_height = lhs_ncols.saturating_sub(1) / MR;

    let two_or_more = mid_height != 0;
    let mid_height = mid_height.saturating_sub(1);

    let bit0 = match dst {
        Accum::Add => 1,
        Accum::Replace => 0,
    };
    let bit1 = 1;
    let bit2 = 1;
    let bit3 = (conj_lhs == conj_rhs) as usize;
    let bit4 = conj_lhs as usize;

    Plan {
        head_mask: core::ptr::from_ref(&head_masks[extra_masked_top_rows]) as *const _,
        tail_mask: core::ptr::from_ref(&tail_masks[padded_rows - full_rows]) as *const _,

        top_left: ukr[NR - 1][MR - 1],
        mid_left: ukr[NR - 1][MR - 1],
        top_right: ukr[right_nr - 1][MR - 1],
        mid_right: ukr[right_nr - 1][MR - 1],
        bot_left: ukr[NR - 1][bottom_mr - 1],
        bot_right: ukr[right_nr - 1][bottom_mr - 1],

        mid_height,
        left_width,
        flags: bit0 | (bit1 << 1) | (bit2 << 2) | (bit3 << 3) | (bit4 << 4),
        do_nothing: lhs_ncols == 0 || rhs_ncols == 0,
        two_or_more,
        mr: MR,
        nr: NR,
    }
}

#[inline]
pub const fn f32_plan_avx(
    extra_masked_top_rows: usize,
    nrows: usize,
    ncols: usize,
    dst: Accum,
) -> Plan {
    const N: usize = 32 / size_of::<f32>();
    const HEAD_MASK: &[__m256i; N] = &unsafe {
        core::mem::transmute([
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [0, -1, -1, -1, -1, -1, -1, -1],
            [0, 0, -1, -1, -1, -1, -1, -1],
            [0, 0, 0, -1, -1, -1, -1, -1],
            [0, 0, 0, 0, -1, -1, -1, -1],
            [0, 0, 0, 0, 0, -1, -1, -1],
            [0, 0, 0, 0, 0, 0, -1, -1],
            [0, 0, 0, 0, 0, 0, 0, -1i32],
        ])
    };
    const TAIL_MASK: &[__m256i; N] = &unsafe {
        core::mem::transmute([
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, 0],
            [-1, -1, -1, -1, -1, -1, 0, 0],
            [-1, -1, -1, -1, -1, 0, 0, 0],
            [-1, -1, -1, -1, 0, 0, 0, 0],
            [-1, -1, -1, 0, 0, 0, 0, 0],
            [-1, -1, 0, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0, 0, 0, 0i32],
        ])
    };

    real_plan_impl(
        extra_masked_top_rows,
        nrows,
        ncols,
        dst,
        UKR_AVX_real_float32,
        HEAD_MASK,
        TAIL_MASK,
    )
}

#[inline]
pub const fn f64_plan_avx(
    extra_masked_top_rows: usize,
    nrows: usize,
    ncols: usize,
    dst: Accum,
) -> Plan {
    const N: usize = 32 / size_of::<f64>();
    const HEAD_MASK: &[__m256i; N] = &unsafe {
        core::mem::transmute([
            [-1, -1, -1, -1],
            [0, -1, -1, -1],
            [0, 0, -1, -1],
            [0, 0, 0, -1i64],
        ])
    };
    const TAIL_MASK: &[__m256i; N] = &unsafe {
        core::mem::transmute([
            [-1, -1, -1, -1],
            [-1, -1, -1, 0],
            [-1, -1, 0, 0],
            [-1, 0, 0, 0i64],
        ])
    };

    real_plan_impl(
        extra_masked_top_rows,
        nrows,
        ncols,
        dst,
        UKR_AVX_real_float64,
        HEAD_MASK,
        TAIL_MASK,
    )
}

#[inline]
pub const fn c32_plan_avx(
    extra_masked_top_rows: usize,
    nrows: usize,
    ncols: usize,
    dst: Accum,
    conj_lhs: bool,
    conj_rhs: bool,
) -> Plan {
    const N: usize = 32 / (2 * size_of::<f32>());
    const HEAD_MASK: &[__m256i; N] = &unsafe {
        core::mem::transmute([
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [0, 0, -1, -1, -1, -1, -1, -1],
            [0, 0, 0, 0, -1, -1, -1, -1],
            [0, 0, 0, 0, 0, 0, -1, -1i32],
        ])
    };
    const TAIL_MASK: &[__m256i; N] = &unsafe {
        core::mem::transmute([
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, 0, 0],
            [-1, -1, -1, -1, 0, 0, 0, 0],
            [-1, -1, 0, 0, 0, 0, 0, 0i32],
        ])
    };

    cplx_plan_impl(
        extra_masked_top_rows,
        nrows,
        ncols,
        dst,
        conj_lhs,
        conj_rhs,
        UKR_AVX_cplx_float32,
        HEAD_MASK,
        TAIL_MASK,
    )
}

#[inline]
pub const fn c64_plan_avx(
    extra_masked_top_rows: usize,
    nrows: usize,
    ncols: usize,
    dst: Accum,
    conj_lhs: bool,
    conj_rhs: bool,
) -> Plan {
    const N: usize = 32 / (2 * size_of::<f64>());
    const HEAD_MASK: &[__m256i; N] =
        &unsafe { core::mem::transmute([[-1, -1, -1, -1], [0, 0, -1, -1i64]]) };
    const TAIL_MASK: &[__m256i; N] =
        &unsafe { core::mem::transmute([[-1, -1, -1, -1], [-1, -1, 0, 0i64]]) };

    cplx_plan_impl(
        extra_masked_top_rows,
        nrows,
        ncols,
        dst,
        conj_lhs,
        conj_rhs,
        UKR_AVX_cplx_float64,
        HEAD_MASK,
        TAIL_MASK,
    )
}

#[inline]
pub const fn f32_plan_avx512(
    extra_masked_top_rows: usize,
    nrows: usize,
    ncols: usize,
    dst: Accum,
) -> Plan {
    const N: usize = 64 / size_of::<f32>();
    const HEAD_MASK: &[u16; N] = &[
        0b1111111111111111, //
        0b1111111111111110, //
        0b1111111111111100, //
        0b1111111111111000, //
        0b1111111111110000, //
        0b1111111111100000, //
        0b1111111111000000, //
        0b1111111110000000, //
        0b1111111100000000, //
        0b1111111000000000, //
        0b1111110000000000, //
        0b1111100000000000, //
        0b1111000000000000, //
        0b1110000000000000, //
        0b1100000000000000, //
        0b1000000000000000, //
    ];
    const TAIL_MASK: &[u16; N] = &[
        0b1111111111111111, //
        0b0111111111111111, //
        0b0011111111111111, //
        0b0001111111111111, //
        0b0000111111111111, //
        0b0000011111111111, //
        0b0000001111111111, //
        0b0000000111111111, //
        0b0000000011111111, //
        0b0000000001111111, //
        0b0000000000111111, //
        0b0000000000011111, //
        0b0000000000001111, //
        0b0000000000000111, //
        0b0000000000000011, //
        0b0000000000000001, //
    ];

    real_plan_impl(
        extra_masked_top_rows,
        nrows,
        ncols,
        dst,
        UKR_AVX512_real_float32,
        HEAD_MASK,
        TAIL_MASK,
    )
}
#[inline]
pub const fn f64_plan_avx512(
    extra_masked_top_rows: usize,
    nrows: usize,
    ncols: usize,
    dst: Accum,
) -> Plan {
    const N: usize = 64 / size_of::<f64>();
    const HEAD_MASK: &[u8; N] = &[
        0b11111111, //
        0b11111110, //
        0b11111100, //
        0b11111000, //
        0b11110000, //
        0b11100000, //
        0b11000000, //
        0b10000000, //
    ];
    const TAIL_MASK: &[u8; N] = &[
        0b11111111, //
        0b01111111, //
        0b00111111, //
        0b00011111, //
        0b00001111, //
        0b00000111, //
        0b00000011, //
        0b00000001, //
    ];

    real_plan_impl(
        extra_masked_top_rows,
        nrows,
        ncols,
        dst,
        UKR_AVX512_real_float64,
        HEAD_MASK,
        TAIL_MASK,
    )
}

#[inline]
pub const fn c32_plan_avx512(
    extra_masked_top_rows: usize,
    nrows: usize,
    ncols: usize,
    dst: Accum,
    conj_lhs: bool,
    conj_rhs: bool,
) -> Plan {
    const N: usize = 64 / (2 * size_of::<f32>());
    const HEAD_MASK: &[u16; N] = &[
        0b1111111111111111, //
        0b1111111111111100, //
        0b1111111111110000, //
        0b1111111111000000, //
        0b1111111100000000, //
        0b1111110000000000, //
        0b1111000000000000, //
        0b1100000000000000, //
    ];
    const TAIL_MASK: &[u16; N] = &[
        0b1111111111111111, //
        0b0011111111111111, //
        0b0000111111111111, //
        0b0000001111111111, //
        0b0000000011111111, //
        0b0000000000111111, //
        0b0000000000001111, //
        0b0000000000000011, //
    ];

    cplx_plan_impl(
        extra_masked_top_rows,
        nrows,
        ncols,
        dst,
        conj_lhs,
        conj_rhs,
        UKR_AVX512_cplx_float32,
        HEAD_MASK,
        TAIL_MASK,
    )
}
#[inline]
pub const fn c64_plan_avx512(
    extra_masked_top_rows: usize,
    nrows: usize,
    ncols: usize,
    dst: Accum,
    conj_lhs: bool,
    conj_rhs: bool,
) -> Plan {
    const N: usize = 64 / (2 * size_of::<f64>());
    const HEAD_MASK: &[u8; N] = &[
        0b11111111, //
        0b11111100, //
        0b11110000, //
        0b11000000, //
    ];
    const TAIL_MASK: &[u8; N] = &[
        0b11111111, //
        0b00111111, //
        0b00001111, //
        0b00000011, //
    ];

    cplx_plan_impl(
        extra_masked_top_rows,
        nrows,
        ncols,
        dst,
        conj_lhs,
        conj_rhs,
        UKR_AVX512_cplx_float64,
        HEAD_MASK,
        TAIL_MASK,
    )
}

#[inline]
pub const fn f32_dot_plan_avx(
    extra_masked_top_rows: usize,
    nrows: usize,
    lhs_ncols: usize,
    rhs_ncols: usize,
    dst: Accum,
) -> Plan {
    const N: usize = 32 / size_of::<f32>();
    const HEAD_MASK: &[__m256i; N] = &unsafe {
        core::mem::transmute([
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [0, -1, -1, -1, -1, -1, -1, -1],
            [0, 0, -1, -1, -1, -1, -1, -1],
            [0, 0, 0, -1, -1, -1, -1, -1],
            [0, 0, 0, 0, -1, -1, -1, -1],
            [0, 0, 0, 0, 0, -1, -1, -1],
            [0, 0, 0, 0, 0, 0, -1, -1],
            [0, 0, 0, 0, 0, 0, 0, -1i32],
        ])
    };
    const TAIL_MASK: &[__m256i; N] = &unsafe {
        core::mem::transmute([
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, 0],
            [-1, -1, -1, -1, -1, -1, 0, 0],
            [-1, -1, -1, -1, -1, 0, 0, 0],
            [-1, -1, -1, -1, 0, 0, 0, 0],
            [-1, -1, -1, 0, 0, 0, 0, 0],
            [-1, -1, 0, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0, 0, 0, 0i32],
        ])
    };

    real_dot_plan_impl(
        extra_masked_top_rows,
        nrows,
        lhs_ncols,
        rhs_ncols,
        dst,
        HKR_AVX_real_float32,
        HEAD_MASK,
        TAIL_MASK,
    )
}
#[inline]
pub const fn f64_dot_plan_avx(
    extra_masked_top_rows: usize,
    nrows: usize,
    lhs_ncols: usize,
    rhs_ncols: usize,
    dst: Accum,
) -> Plan {
    const N: usize = 32 / size_of::<f64>();
    const HEAD_MASK: &[__m256i; N] = &unsafe {
        core::mem::transmute([
            [-1, -1, -1, -1],
            [0, -1, -1, -1],
            [0, 0, -1, -1],
            [0, 0, 0, -1i64],
        ])
    };
    const TAIL_MASK: &[__m256i; N] = &unsafe {
        core::mem::transmute([
            [-1, -1, -1, -1],
            [-1, -1, -1, 0],
            [-1, -1, 0, 0],
            [-1, 0, 0, 0i64],
        ])
    };

    real_dot_plan_impl(
        extra_masked_top_rows,
        nrows,
        lhs_ncols,
        rhs_ncols,
        dst,
        HKR_AVX_real_float64,
        HEAD_MASK,
        TAIL_MASK,
    )
}
#[inline]
pub const fn c32_dot_plan_avx(
    extra_masked_top_rows: usize,
    nrows: usize,
    lhs_ncols: usize,
    rhs_ncols: usize,
    dst: Accum,
    conj_lhs: bool,
    conj_rhs: bool,
) -> Plan {
    const N: usize = 32 / (2 * size_of::<f32>());
    const HEAD_MASK: &[__m256i; N] = &unsafe {
        core::mem::transmute([
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [0, 0, -1, -1, -1, -1, -1, -1],
            [0, 0, 0, 0, -1, -1, -1, -1],
            [0, 0, 0, 0, 0, 0, -1, -1i32],
        ])
    };
    const TAIL_MASK: &[__m256i; N] = &unsafe {
        core::mem::transmute([
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, 0, 0],
            [-1, -1, -1, -1, 0, 0, 0, 0],
            [-1, -1, 0, 0, 0, 0, 0, 0i32],
        ])
    };

    cplx_dot_plan_impl(
        extra_masked_top_rows,
        nrows,
        lhs_ncols,
        rhs_ncols,
        dst,
        conj_lhs,
        conj_rhs,
        HKR_AVX_cplx_float32,
        HEAD_MASK,
        TAIL_MASK,
    )
}
#[inline]
pub const fn c64_dot_plan_avx(
    extra_masked_top_rows: usize,
    nrows: usize,
    lhs_ncols: usize,
    rhs_ncols: usize,
    dst: Accum,
    conj_lhs: bool,
    conj_rhs: bool,
) -> Plan {
    const N: usize = 32 / (2 * size_of::<f64>());
    const HEAD_MASK: &[__m256i; N] =
        &unsafe { core::mem::transmute([[-1, -1, -1, -1], [0, 0, -1, -1i64]]) };
    const TAIL_MASK: &[__m256i; N] =
        &unsafe { core::mem::transmute([[-1, -1, -1, -1], [-1, -1, 0, 0i64]]) };

    cplx_dot_plan_impl(
        extra_masked_top_rows,
        nrows,
        lhs_ncols,
        rhs_ncols,
        dst,
        conj_lhs,
        conj_rhs,
        HKR_AVX_cplx_float64,
        HEAD_MASK,
        TAIL_MASK,
    )
}
#[inline]
pub const fn f32_dot_plan_avx512(
    extra_masked_top_rows: usize,
    nrows: usize,
    lhs_ncols: usize,
    rhs_ncols: usize,
    dst: Accum,
) -> Plan {
    const N: usize = 64 / size_of::<f32>();
    const HEAD_MASK: &[u16; N] = &[
        0b1111111111111111, //
        0b1111111111111110, //
        0b1111111111111100, //
        0b1111111111111000, //
        0b1111111111110000, //
        0b1111111111100000, //
        0b1111111111000000, //
        0b1111111110000000, //
        0b1111111100000000, //
        0b1111111000000000, //
        0b1111110000000000, //
        0b1111100000000000, //
        0b1111000000000000, //
        0b1110000000000000, //
        0b1100000000000000, //
        0b1000000000000000, //
    ];
    const TAIL_MASK: &[u16; N] = &[
        0b1111111111111111, //
        0b0111111111111111, //
        0b0011111111111111, //
        0b0001111111111111, //
        0b0000111111111111, //
        0b0000011111111111, //
        0b0000001111111111, //
        0b0000000111111111, //
        0b0000000011111111, //
        0b0000000001111111, //
        0b0000000000111111, //
        0b0000000000011111, //
        0b0000000000001111, //
        0b0000000000000111, //
        0b0000000000000011, //
        0b0000000000000001, //
    ];

    real_dot_plan_impl(
        extra_masked_top_rows,
        nrows,
        lhs_ncols,
        rhs_ncols,
        dst,
        HKR_AVX512_real_float32,
        HEAD_MASK,
        TAIL_MASK,
    )
}
#[inline]
pub const fn f64_dot_plan_avx512(
    extra_masked_top_rows: usize,
    nrows: usize,
    lhs_ncols: usize,
    rhs_ncols: usize,
    dst: Accum,
) -> Plan {
    const N: usize = 64 / size_of::<f64>();
    const HEAD_MASK: &[u8; N] = &[
        0b11111111, //
        0b11111110, //
        0b11111100, //
        0b11111000, //
        0b11110000, //
        0b11100000, //
        0b11000000, //
        0b10000000, //
    ];
    const TAIL_MASK: &[u8; N] = &[
        0b11111111, //
        0b01111111, //
        0b00111111, //
        0b00011111, //
        0b00001111, //
        0b00000111, //
        0b00000011, //
        0b00000001, //
    ];

    real_dot_plan_impl(
        extra_masked_top_rows,
        nrows,
        lhs_ncols,
        rhs_ncols,
        dst,
        HKR_AVX512_real_float64,
        HEAD_MASK,
        TAIL_MASK,
    )
}

#[inline]
pub const fn c32_dot_plan_avx512(
    extra_masked_top_rows: usize,
    nrows: usize,
    lhs_ncols: usize,
    rhs_ncols: usize,
    dst: Accum,
    conj_lhs: bool,
    conj_rhs: bool,
) -> Plan {
    const N: usize = 64 / (2 * size_of::<f32>());
    const HEAD_MASK: &[u16; N] = &[
        0b1111111111111111, //
        0b1111111111111100, //
        0b1111111111110000, //
        0b1111111111000000, //
        0b1111111100000000, //
        0b1111110000000000, //
        0b1111000000000000, //
        0b1100000000000000, //
    ];
    const TAIL_MASK: &[u16; N] = &[
        0b1111111111111111, //
        0b0011111111111111, //
        0b0000111111111111, //
        0b0000001111111111, //
        0b0000000011111111, //
        0b0000000000111111, //
        0b0000000000001111, //
        0b0000000000000011, //
    ];

    cplx_dot_plan_impl(
        extra_masked_top_rows,
        nrows,
        lhs_ncols,
        rhs_ncols,
        dst,
        conj_lhs,
        conj_rhs,
        HKR_AVX512_cplx_float32,
        HEAD_MASK,
        TAIL_MASK,
    )
}
#[inline]
pub const fn c64_dot_plan_avx512(
    extra_masked_top_rows: usize,
    nrows: usize,
    lhs_ncols: usize,
    rhs_ncols: usize,
    dst: Accum,
    conj_lhs: bool,
    conj_rhs: bool,
) -> Plan {
    const N: usize = 64 / (2 * size_of::<f64>());
    const HEAD_MASK: &[u8; N] = &[
        0b11111111, //
        0b11111100, //
        0b11110000, //
        0b11000000, //
    ];
    const TAIL_MASK: &[u8; N] = &[
        0b11111111, //
        0b00111111, //
        0b00001111, //
        0b00000011, //
    ];

    cplx_dot_plan_impl(
        extra_masked_top_rows,
        nrows,
        lhs_ncols,
        rhs_ncols,
        dst,
        conj_lhs,
        conj_rhs,
        HKR_AVX512_cplx_float64,
        HEAD_MASK,
        TAIL_MASK,
    )
}
pub unsafe fn millikernel(
    data: &Plan,
    depth: usize,
    lhs_cs: isize,
    rhs_rs: isize,
    rhs_cs: isize,
    dst_cs: isize,

    dst_brs: isize,
    dst_bcs: isize,

    lhs_bs: isize,
    rhs_bs: isize,

    lhs_ptr: *const (),
    rhs_ptr: *const (),
    dst_ptr: *mut (),

    alpha: *const (),
) {
    core::arch::asm! {
        "cmp byte ptr [{data} + 88], 0",
        "jnz 2333f",

        "push rbx",
        "push rbp",
        "push r12",
        "push r13",
        "push r14",
        "push r15",

        "mov r14, {data}",

        "mov r13, qword ptr [r14 + 64]",
        "push r13",
        "mov r13, qword ptr [r14 + 72]",
        "push r13",
        "mov rbx, qword ptr [r14]",
        "mov rbp, qword ptr [r14 + 8]",

        "mov r12, qword ptr [r14 + 80]",

        // stack state
        //  0|16        8|24        16|32    24|40    32|48   40|56
        // [left width, mid height, dst_bcs, dst_brs, rhs_bs, lhs_bs]

        "cmp byte ptr [r14 + 89], 0",
        "jnz 23f",
            // bot chunk only
            "cmp qword ptr [rsp], 0",
            "jz 32f",
                "33:",

                "call qword ptr [r14 + 48]",
                "add r10, qword ptr [rsp + 32]",
                "add r11, qword ptr [rsp + 16]",
                "dec qword ptr [rsp]",
                "jnz 33b",
            "32:",

            "call qword ptr [r14 + 56]",
            "jmp 22f",
        "23:",
            // multiple row chunks

            // left col chunk
            "cmp qword ptr [rsp], 0",
            "jz 32f",
                "33:",

                "mov r13, qword ptr [r14 + 64]",
                "mov qword ptr [rsp + 8], r13",

                "push r9",
                "push r11",

                // top
                "mov r12, qword ptr [r14 + 80]",
                "and r12, 27",
                "call qword ptr [r14 + 16]",
                "add r9, qword ptr [rsp + 56]",
                "add r11, qword ptr [rsp + 40]",

                "and r12, 25",
                // mid
                "cmp qword ptr [rsp + 24], 0",
                "jz 42f",
                    "43:",
                    "call qword ptr [r14 + 32]",
                    "add r9, qword ptr [rsp + 56]",
                    "add r11, qword ptr [rsp + 40]",
                    "dec qword ptr [rsp + 24]",
                    "jnz 43b",
                "42:",

                // bot
                "mov r12, qword ptr [r14 + 80]",
                "and r12, 29",
                "call qword ptr [r14 + 48]",

                "pop r11",
                "pop r9",
                "add r10, qword ptr [rsp + 32]",
                "add r11, qword ptr [rsp + 16]",
                "dec qword ptr [rsp]",
                "jnz 33b",
            "32:",

            // right col chunk
            "mov r13, qword ptr [r14 + 64]",
            "mov qword ptr [rsp + 8], r13",

            // top
            "mov r12, qword ptr [r14 + 80]",
            "and r12, 27",
            "call qword ptr [r14 + 24]",
            "add r9, qword ptr [rsp + 40]",
            "add r11, qword ptr [rsp + 24]",

            "and r12, 25",
            // mid
            "cmp qword ptr [rsp + 8], 0",
            "jz 42f",
                "43:",
                "call qword ptr [r14 + 40]",
                "add r9, qword ptr [rsp + 40]",
                "add r11, qword ptr [rsp + 24]",
                "dec qword ptr [rsp + 8]",
                "jnz 43b",
            "42:",

            // bot
            "mov r12, qword ptr [r14 + 80]",
            "and r12, 29",
            "call qword ptr [r14 + 56]",

        "22:",

        "pop r13",
        "pop r13",
        "pop r15",
        "pop r14",
        "pop r13",
        "pop r12",
        "pop rbp",
        "pop rbx",

        "2333:",

        data = in(reg) data,
        in("rax") depth,
        in("rdi") lhs_cs,
        in("rcx") rhs_rs,
        in("rdx") rhs_cs,
        in("rsi") dst_cs,
        in("r8") alpha,
        inout("r9") lhs_ptr => _,
        inout("r10") rhs_ptr => _,
        inout("r11") dst_ptr => _,

        in("r12") lhs_bs,
        in("r13") rhs_bs,
        in("r14") dst_brs,
        in("r15") dst_bcs,
    };
}

pub unsafe fn dot_millikernel(
    data: &Plan,
    depth: usize,
    lhs_cs: isize,
    rhs_cs: isize,
    dst_rs: isize,
    dst_cs: isize,

    dst_brs: isize,
    dst_bcs: isize,

    lhs_bs: isize,
    rhs_bs: isize,

    lhs_ptr: *const (),
    rhs_ptr: *const (),
    dst_ptr: *mut (),

    alpha: *const (),
) {
    millikernel(
        data, depth, lhs_cs, dst_rs, rhs_cs, dst_cs, dst_brs, dst_bcs, lhs_bs, rhs_bs, lhs_ptr,
        rhs_ptr, dst_ptr, alpha,
    );
}

#[cfg(test)]
mod real_tests {
    use aligned_vec::avec;

    use super::*;

    #[test]
    fn test_avx() {
        for m in 0..48 {
            for n in 0..24 {
                let k = 2;
                for skip in 0..Ord::min(4, m) {
                    let target = &mut *avec![0.0; m * n];
                    let dst = &mut *avec![0.0; m * n];
                    let lhs = &mut *avec![0.0; m * k];
                    let rhs = &mut *avec![0.0; n * k];

                    for x in &mut *lhs {
                        *x = rand::random();
                    }
                    for x in &mut *rhs {
                        *x = rand::random();
                    }

                    let plan = f64_plan_avx(skip, m - skip, n, Accum::Replace);

                    unsafe {
                        millikernel(
                            &plan,
                            k,
                            (m * size_of::<f64>()) as isize,
                            (size_of::<f64>()) as isize,
                            (k * size_of::<f64>()) as isize,
                            (m * size_of::<f64>()) as isize,
                            (plan.mr * size_of::<f64>()) as isize,
                            (plan.nr * m * size_of::<f64>()) as isize,
                            (plan.mr * size_of::<f64>()) as isize,
                            (plan.nr * k * size_of::<f64>()) as isize,
                            lhs.as_ptr() as _,
                            rhs.as_ptr() as _,
                            dst.as_mut_ptr() as _,
                            core::ptr::from_ref(&1.0) as _,
                        )
                    };

                    for i in 0..m {
                        for j in 0..n {
                            for depth in 0..k {
                                if i >= skip {
                                    target[i + m * j] = f64::mul_add(
                                        lhs[i + m * depth],
                                        rhs[depth + j * k],
                                        target[i + m * j],
                                    );
                                }
                            }
                        }
                    }
                    assert_eq!(dst, target);
                }
            }
        }
    }

    #[test]
    fn test_avx512() {
        for m in 0..48 {
            for n in 0..24 {
                let k = 2;
                for skip in 0..Ord::min(8, m) {
                    let target = &mut *avec![0.0; m * n];
                    let dst = &mut *avec![0.0; m * n];
                    let lhs = &mut *avec![0.0; m * k];
                    let rhs = &mut *avec![0.0; n * k];

                    for x in &mut *lhs {
                        *x = rand::random();
                    }
                    for x in &mut *rhs {
                        *x = rand::random();
                    }

                    let plan = f64_plan_avx512(skip, m - skip, n, Accum::Replace);

                    unsafe {
                        millikernel(
                            &plan,
                            k,
                            (m * size_of::<f64>()) as isize,
                            (size_of::<f64>()) as isize,
                            (k * size_of::<f64>()) as isize,
                            (m * size_of::<f64>()) as isize,
                            (plan.mr * size_of::<f64>()) as isize,
                            (plan.nr * m * size_of::<f64>()) as isize,
                            (plan.mr * size_of::<f64>()) as isize,
                            (plan.nr * k * size_of::<f64>()) as isize,
                            lhs.as_ptr() as _,
                            rhs.as_ptr() as _,
                            dst.as_mut_ptr() as _,
                            core::ptr::from_ref(&1.0) as _,
                        )
                    };

                    for i in 0..m {
                        for j in 0..n {
                            for depth in 0..k {
                                if i >= skip {
                                    target[i + m * j] = f64::mul_add(
                                        lhs[i + m * depth],
                                        rhs[depth + j * k],
                                        target[i + m * j],
                                    );
                                }
                            }
                        }
                    }
                    assert_eq!(dst, target);
                }
            }
        }
    }

    #[test]
    fn test_avx_f32() {
        for m in 0..48 {
            for n in 0..24 {
                let k = 2;
                for skip in 0..Ord::min(8, m) {
                    let target = &mut *avec![0.0; m * n];
                    let dst = &mut *avec![0.0; m * n];
                    let lhs = &mut *avec![0.0; m * k];
                    let rhs = &mut *avec![0.0; n * k];

                    for x in &mut *lhs {
                        *x = rand::random();
                    }
                    for x in &mut *rhs {
                        *x = rand::random();
                    }

                    let plan = f32_plan_avx(skip, m - skip, n, Accum::Replace);

                    unsafe {
                        millikernel(
                            &plan,
                            k,
                            (m * size_of::<f32>()) as isize,
                            (size_of::<f32>()) as isize,
                            (k * size_of::<f32>()) as isize,
                            (m * size_of::<f32>()) as isize,
                            (plan.mr * size_of::<f32>()) as isize,
                            (plan.nr * m * size_of::<f32>()) as isize,
                            (plan.mr * size_of::<f32>()) as isize,
                            (plan.nr * k * size_of::<f32>()) as isize,
                            lhs.as_ptr() as _,
                            rhs.as_ptr() as _,
                            dst.as_mut_ptr() as _,
                            core::ptr::from_ref(&1.0f32) as _,
                        )
                    };

                    for i in 0..m {
                        for j in 0..n {
                            for depth in 0..k {
                                if i >= skip {
                                    target[i + m * j] = f32::mul_add(
                                        lhs[i + m * depth],
                                        rhs[depth + j * k],
                                        target[i + m * j],
                                    );
                                }
                            }
                        }
                    }
                    assert_eq!(dst, target);
                }
            }
        }
    }

    #[test]
    fn test_avx512_f32() {
        for m in 0..48 {
            for n in 0..24 {
                let k = 2;
                for skip in 0..Ord::min(16, m) {
                    let target = &mut *avec![0.0; m * n];
                    let dst = &mut *avec![0.0; m * n];
                    let lhs = &mut *avec![0.0; m * k];
                    let rhs = &mut *avec![0.0; n * k];

                    for x in &mut *lhs {
                        *x = rand::random();
                    }
                    for x in &mut *rhs {
                        *x = rand::random();
                    }

                    let plan = f32_plan_avx512(skip, m - skip, n, Accum::Replace);

                    unsafe {
                        millikernel(
                            &plan,
                            k,
                            (m * size_of::<f32>()) as isize,
                            (size_of::<f32>()) as isize,
                            (k * size_of::<f32>()) as isize,
                            (m * size_of::<f32>()) as isize,
                            (plan.mr * size_of::<f32>()) as isize,
                            (plan.nr * m * size_of::<f32>()) as isize,
                            (plan.mr * size_of::<f32>()) as isize,
                            (plan.nr * k * size_of::<f32>()) as isize,
                            lhs.as_ptr() as _,
                            rhs.as_ptr() as _,
                            dst.as_mut_ptr() as _,
                            core::ptr::from_ref(&1.0f32) as _,
                        )
                    };

                    for i in 0..m {
                        for j in 0..n {
                            for depth in 0..k {
                                if i >= skip {
                                    target[i + m * j] = f32::mul_add(
                                        lhs[i + m * depth],
                                        rhs[depth + j * k],
                                        target[i + m * j],
                                    );
                                }
                            }
                        }
                    }
                    assert_eq!(dst, target);
                }
            }
        }
    }
}

#[cfg(test)]
mod cplx_tests {
    use super::*;
    use aligned_vec::avec;
    use num_complex::ComplexFloat;
    use rand::random;

    #[allow(non_camel_case_types)]
    type c64 = num_complex::Complex<f64>;
    #[allow(non_camel_case_types)]
    type c32 = num_complex::Complex<f32>;

    #[test]
    fn complex_formulas() {
        let x = c64::new(random(), random());
        let y = c64::new(random(), random());

        let mul = x * y;
        let re = f64::mul_add(x.im, y.im, -f64::mul_add(x.re, y.re, -0.0));
        let im = f64::mul_add(x.re, y.im, f64::mul_add(x.im, y.re, 0.0));
        assert!((c64::new(-re, im) - mul).abs() < 1e-8);

        let mul = x.conj() * y.conj();
        let re = f64::mul_add(x.im, y.im, -f64::mul_add(x.re, y.re, -0.0));
        let im = f64::mul_add(x.re, y.im, f64::mul_add(x.im, y.re, 0.0));
        assert!((c64::new(-re, -im) - mul).abs() < 1e-8);

        let mul = x.conj() * y;
        let re = f64::mul_add(x.im, y.im, f64::mul_add(x.re, y.re, 0.0));
        let im = f64::mul_add(x.re, y.im, -f64::mul_add(x.im, y.re, -0.0));
        assert!((c64::new(re, im) - mul).abs() < 1e-8);

        let mul = x * y.conj();
        let re = f64::mul_add(x.im, y.im, f64::mul_add(x.re, y.re, 0.0));
        let im = f64::mul_add(x.re, y.im, -f64::mul_add(x.im, y.re, -0.0));
        assert!((c64::new(re, -im) - mul).abs() < 1e-8);
    }

    #[test]
    fn test_avx() {
        for m in 1..48 {
            for n in 1..24 {
                let k = 2;
                for skip in 0..Ord::min(2, m) {
                    let target = &mut *avec![c64::ZERO; m * n];
                    let dst = &mut *avec![c64::ZERO; m * n];
                    let lhs = &mut *avec![c64::ZERO; m * k];
                    let rhs = &mut *avec![c64::ZERO; n * k];

                    for x in &mut *lhs {
                        x.re = random();
                        x.im = random();
                    }
                    for x in &mut *rhs {
                        x.re = random();
                        x.im = random();
                    }

                    let plan = c64_plan_avx(skip, m - skip, n, Accum::Replace, false, false);

                    unsafe {
                        millikernel(
                            &plan,
                            k,
                            (m * size_of::<c64>()) as isize,
                            (size_of::<c64>()) as isize,
                            (k * size_of::<c64>()) as isize,
                            (m * size_of::<c64>()) as isize,
                            (plan.mr * size_of::<c64>()) as isize,
                            (plan.nr * m * size_of::<c64>()) as isize,
                            (plan.mr * size_of::<c64>()) as isize,
                            (plan.nr * k * size_of::<c64>()) as isize,
                            lhs.as_ptr() as _,
                            rhs.as_ptr() as _,
                            dst.as_mut_ptr() as _,
                            core::ptr::from_ref(&c64::new(1.0, 0.0)) as _,
                        )
                    };

                    for i in 0..m {
                        for j in 0..n {
                            for depth in 0..k {
                                if i >= skip {
                                    target[i + m * j] += lhs[i + m * depth] * rhs[depth + j * k];
                                }
                            }
                        }
                    }
                    for i in 0..m * n {
                        assert!((dst[i] - target[i]).abs() < 1e-12);
                    }
                }
            }
        }
    }

    #[test]
    fn test_avx512() {
        for m in 1..48 {
            for n in 1..24 {
                let k = 2;
                for skip in 0..Ord::min(4, m) {
                    for conj_lhs in [true, false] {
                        for conj_rhs in [true, false] {
                            let target = &mut *avec![c64::ZERO; m * n];
                            let dst = &mut *avec![c64::ZERO; m * n];
                            let lhs = &mut *avec![c64::ZERO; m * k];
                            let rhs = &mut *avec![c64::ZERO; n * k];

                            for x in &mut *lhs {
                                x.re = random();
                                x.im = random();
                            }
                            for x in &mut *rhs {
                                x.re = random();
                                x.im = random();
                            }

                            let plan = c64_plan_avx512(
                                skip,
                                m - skip,
                                n,
                                Accum::Replace,
                                conj_lhs,
                                conj_rhs,
                            );

                            let alpha = c64::new(1.3, 2.5);

                            unsafe {
                                millikernel(
                                    &plan,
                                    k,
                                    (m * size_of::<c64>()) as isize,
                                    (size_of::<c64>()) as isize,
                                    (k * size_of::<c64>()) as isize,
                                    (m * size_of::<c64>()) as isize,
                                    (plan.mr * size_of::<c64>()) as isize,
                                    (plan.nr * m * size_of::<c64>()) as isize,
                                    (plan.mr * size_of::<c64>()) as isize,
                                    (plan.nr * k * size_of::<c64>()) as isize,
                                    lhs.as_ptr() as _,
                                    rhs.as_ptr() as _,
                                    dst.as_mut_ptr() as _,
                                    core::ptr::from_ref(&alpha) as _,
                                )
                            };

                            for i in 0..m {
                                for j in 0..n {
                                    for depth in 0..k {
                                        if i >= skip {
                                            let mut lhs = lhs[i + m * depth];
                                            let mut rhs = rhs[depth + j * k];
                                            if conj_lhs {
                                                lhs = lhs.conj();
                                            }
                                            if conj_rhs {
                                                rhs = rhs.conj();
                                            }
                                            target[i + m * j] += alpha * lhs * rhs;
                                        }
                                    }
                                }
                            }
                            for i in 0..m * n {
                                assert!((dst[i] - target[i]).abs() < 1e-12);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_avx_f32() {
        for m in 1..48 {
            for n in 1..24 {
                let k = 2;
                for skip in 0..Ord::min(4, m) {
                    for conj_lhs in [true, false] {
                        for conj_rhs in [true, false] {
                            let target = &mut *avec![c32::ZERO; m * n];
                            let dst = &mut *avec![c32::ZERO; m * n];
                            let lhs = &mut *avec![c32::ZERO; m * k];
                            let rhs = &mut *avec![c32::ZERO; n * k];

                            for x in &mut *lhs {
                                x.re = random();
                                x.im = random();
                            }
                            for x in &mut *rhs {
                                x.re = random();
                                x.im = random();
                            }

                            let plan =
                                c32_plan_avx(skip, m - skip, n, Accum::Replace, conj_lhs, conj_rhs);

                            let alpha = c32::new(1.3, 2.5);

                            unsafe {
                                millikernel(
                                    &plan,
                                    k,
                                    (m * size_of::<c32>()) as isize,
                                    (size_of::<c32>()) as isize,
                                    (k * size_of::<c32>()) as isize,
                                    (m * size_of::<c32>()) as isize,
                                    (plan.mr * size_of::<c32>()) as isize,
                                    (plan.nr * m * size_of::<c32>()) as isize,
                                    (plan.mr * size_of::<c32>()) as isize,
                                    (plan.nr * k * size_of::<c32>()) as isize,
                                    lhs.as_ptr() as _,
                                    rhs.as_ptr() as _,
                                    dst.as_mut_ptr() as _,
                                    core::ptr::from_ref(&alpha) as _,
                                )
                            };

                            for i in 0..m {
                                for j in 0..n {
                                    for depth in 0..k {
                                        if i >= skip {
                                            let mut lhs = lhs[i + m * depth];
                                            let mut rhs = rhs[depth + j * k];
                                            if conj_lhs {
                                                lhs = lhs.conj();
                                            }
                                            if conj_rhs {
                                                rhs = rhs.conj();
                                            }
                                            target[i + m * j] += alpha * lhs * rhs;
                                        }
                                    }
                                }
                            }
                            for i in 0..m * n {
                                assert!((dst[i] - target[i]).abs() < 1e-4);
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn test_avx512_f32() {
        for m in 1..48 {
            for n in 1..24 {
                let k = 2;
                for skip in 0..Ord::min(8, m) {
                    for conj_lhs in [true, false] {
                        for conj_rhs in [true, false] {
                            let target = &mut *avec![c32::ZERO; m * n];
                            let dst = &mut *avec![c32::ZERO; m * n];
                            let lhs = &mut *avec![c32::ZERO; m * k];
                            let rhs = &mut *avec![c32::ZERO; n * k];

                            for x in &mut *lhs {
                                x.re = random();
                                x.im = random();
                            }
                            for x in &mut *rhs {
                                x.re = random();
                                x.im = random();
                            }

                            let plan = c32_plan_avx512(
                                skip,
                                m - skip,
                                n,
                                Accum::Replace,
                                conj_lhs,
                                conj_rhs,
                            );

                            let alpha = c32::new(1.3, 2.5);

                            unsafe {
                                millikernel(
                                    &plan,
                                    k,
                                    (m * size_of::<c32>()) as isize,
                                    (size_of::<c32>()) as isize,
                                    (k * size_of::<c32>()) as isize,
                                    (m * size_of::<c32>()) as isize,
                                    (plan.mr * size_of::<c32>()) as isize,
                                    (plan.nr * m * size_of::<c32>()) as isize,
                                    (plan.mr * size_of::<c32>()) as isize,
                                    (plan.nr * k * size_of::<c32>()) as isize,
                                    lhs.as_ptr() as _,
                                    rhs.as_ptr() as _,
                                    dst.as_mut_ptr() as _,
                                    core::ptr::from_ref(&alpha) as _,
                                )
                            };

                            for i in 0..m {
                                for j in 0..n {
                                    for depth in 0..k {
                                        if i >= skip {
                                            let mut lhs = lhs[i + m * depth];
                                            let mut rhs = rhs[depth + j * k];
                                            if conj_lhs {
                                                lhs = lhs.conj();
                                            }
                                            if conj_rhs {
                                                rhs = rhs.conj();
                                            }
                                            target[i + m * j] += alpha * lhs * rhs;
                                        }
                                    }
                                }
                            }
                            for i in 0..m * n {
                                assert!((dst[i] - target[i]).abs() < 1e-4);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod dot_real_tests {
    use super::*;
    use aligned_vec::avec;

    #[test]
    fn test_avx_f32() {
        for m in 0..12 {
            for n in 0..12 {
                for k in 0..20 {
                    for skip in 0..Ord::min(8, k) {
                        let target = &mut *avec![0.0; m * n];
                        let dst = &mut *avec![0.0; m * n];
                        let lhs = &mut *avec![0.0; k * m];
                        let rhs = &mut *avec![0.0; k * n];

                        for x in &mut *lhs {
                            *x = rand::random();
                        }
                        for x in &mut *rhs {
                            *x = rand::random();
                        }

                        let plan = f32_dot_plan_avx(skip, k, m, n, Accum::Replace);

                        unsafe {
                            dot_millikernel(
                                &plan,
                                k,
                                (k * size_of::<f32>()) as isize,
                                (k * size_of::<f32>()) as isize,
                                (size_of::<f32>()) as isize,
                                (m * size_of::<f32>()) as isize,
                                (plan.mr * size_of::<f32>()) as isize,
                                (plan.nr * m * size_of::<f32>()) as isize,
                                (plan.mr * k * size_of::<f32>()) as isize,
                                (plan.nr * k * size_of::<f32>()) as isize,
                                lhs.as_ptr() as _,
                                rhs.as_ptr() as _,
                                dst.as_mut_ptr() as _,
                                core::ptr::from_ref(&1.0_f32) as _,
                            )
                        };

                        for i in 0..m {
                            for j in 0..n {
                                let mut acc = 0.0;
                                for depth in 0..k {
                                    if depth >= skip {
                                        acc = f32::mul_add(
                                            lhs[depth + k * i],
                                            rhs[depth + k * j],
                                            acc,
                                        );
                                    }
                                }
                                target[i + m * j] = acc;
                            }
                        }
                        for i in 0..m * n {
                            assert!((dst[i] - target[i]).abs() < 1e-4);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_avx512_f32() {
        for m in 0..12 {
            for n in 0..12 {
                for k in 0..20 {
                    for skip in 0..Ord::min(16, k) {
                        let target = &mut *avec![0.0; m * n];
                        let dst = &mut *avec![0.0; m * n];
                        let lhs = &mut *avec![0.0; k * m];
                        let rhs = &mut *avec![0.0; k * n];

                        for x in &mut *lhs {
                            *x = rand::random();
                        }
                        for x in &mut *rhs {
                            *x = rand::random();
                        }

                        let plan = f32_dot_plan_avx512(skip, k, m, n, Accum::Replace);

                        unsafe {
                            dot_millikernel(
                                &plan,
                                k,
                                (k * size_of::<f32>()) as isize,
                                (k * size_of::<f32>()) as isize,
                                (size_of::<f32>()) as isize,
                                (m * size_of::<f32>()) as isize,
                                (plan.mr * size_of::<f32>()) as isize,
                                (plan.nr * m * size_of::<f32>()) as isize,
                                (plan.mr * k * size_of::<f32>()) as isize,
                                (plan.nr * k * size_of::<f32>()) as isize,
                                lhs.as_ptr() as _,
                                rhs.as_ptr() as _,
                                dst.as_mut_ptr() as _,
                                core::ptr::from_ref(&1.0_f32) as _,
                            )
                        };

                        for i in 0..m {
                            for j in 0..n {
                                let mut acc = 0.0;
                                for depth in 0..k {
                                    if depth >= skip {
                                        acc = f32::mul_add(
                                            lhs[depth + k * i],
                                            rhs[depth + k * j],
                                            acc,
                                        );
                                    }
                                }
                                target[i + m * j] = acc;
                            }
                        }
                        for i in 0..m * n {
                            assert!((dst[i] - target[i]).abs() < 1e-4);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_avx_f64() {
        for m in 0..12 {
            for n in 0..12 {
                for k in 0..20 {
                    for skip in 0..Ord::min(4, k) {
                        let target = &mut *avec![0.0; m * n];
                        let dst = &mut *avec![0.0; m * n];
                        let lhs = &mut *avec![0.0; k * m];
                        let rhs = &mut *avec![0.0; k * n];

                        for x in &mut *lhs {
                            *x = rand::random();
                        }
                        for x in &mut *rhs {
                            *x = rand::random();
                        }

                        let plan = f64_dot_plan_avx(skip, k, m, n, Accum::Replace);

                        unsafe {
                            dot_millikernel(
                                &plan,
                                k,
                                (k * size_of::<f64>()) as isize,
                                (k * size_of::<f64>()) as isize,
                                (size_of::<f64>()) as isize,
                                (m * size_of::<f64>()) as isize,
                                (plan.mr * size_of::<f64>()) as isize,
                                (plan.nr * m * size_of::<f64>()) as isize,
                                (plan.mr * k * size_of::<f64>()) as isize,
                                (plan.nr * k * size_of::<f64>()) as isize,
                                lhs.as_ptr() as _,
                                rhs.as_ptr() as _,
                                dst.as_mut_ptr() as _,
                                core::ptr::from_ref(&1.0_f64) as _,
                            )
                        };

                        for i in 0..m {
                            for j in 0..n {
                                let mut acc = 0.0;
                                for depth in 0..k {
                                    if depth >= skip {
                                        acc = f64::mul_add(
                                            lhs[depth + k * i],
                                            rhs[depth + k * j],
                                            acc,
                                        );
                                    }
                                }
                                target[i + m * j] = acc;
                            }
                        }
                        for i in 0..m * n {
                            assert!((dst[i] - target[i]).abs() < 1e-4);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_avx512_f64() {
        for m in 0..12 {
            for n in 0..12 {
                for k in 0..20 {
                    for skip in 0..Ord::min(8, k) {
                        let target = &mut *avec![0.0; m * n];
                        let dst = &mut *avec![0.0; m * n];
                        let lhs = &mut *avec![0.0; k * m];
                        let rhs = &mut *avec![0.0; k * n];

                        for x in &mut *lhs {
                            *x = rand::random();
                        }
                        for x in &mut *rhs {
                            *x = rand::random();
                        }

                        let plan = f64_dot_plan_avx512(skip, k, m, n, Accum::Replace);

                        unsafe {
                            dot_millikernel(
                                &plan,
                                k,
                                (k * size_of::<f64>()) as isize,
                                (k * size_of::<f64>()) as isize,
                                (size_of::<f64>()) as isize,
                                (m * size_of::<f64>()) as isize,
                                (plan.mr * size_of::<f64>()) as isize,
                                (plan.nr * m * size_of::<f64>()) as isize,
                                (plan.mr * k * size_of::<f64>()) as isize,
                                (plan.nr * k * size_of::<f64>()) as isize,
                                lhs.as_ptr() as _,
                                rhs.as_ptr() as _,
                                dst.as_mut_ptr() as _,
                                core::ptr::from_ref(&1.0_f64) as _,
                            )
                        };

                        for i in 0..m {
                            for j in 0..n {
                                let mut acc = 0.0;
                                for depth in 0..k {
                                    if depth >= skip {
                                        acc = f64::mul_add(
                                            lhs[depth + k * i],
                                            rhs[depth + k * j],
                                            acc,
                                        );
                                    }
                                }
                                target[i + m * j] = acc;
                            }
                        }
                        for i in 0..m * n {
                            assert!((dst[i] - target[i]).abs() < 1e-4);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod dot_cplx_tests {
    use super::*;
    use aligned_vec::avec;
    use num_complex::ComplexFloat;
    use rand::random;

    #[allow(non_camel_case_types)]
    type c64 = num_complex::Complex<f64>;
    #[allow(non_camel_case_types)]
    type c32 = num_complex::Complex<f32>;

    #[test]
    fn test_avx_f32() {
        for m in 1..12 {
            for n in 1..12 {
                for k in 0..10 {
                    for skip in 0..Ord::min(4, k) {
                        for conj_lhs in [false, true] {
                            for conj_rhs in [false, true] {
                                let target = &mut *avec![c32::ZERO; m * n];
                                let dst = &mut *avec![c32::ZERO; m * n];
                                let lhs = &mut *avec![c32::ZERO; m * k];
                                let rhs = &mut *avec![c32::ZERO; n * k];

                                for x in &mut *lhs {
                                    x.re = random();
                                    x.im = random();
                                }
                                for x in &mut *rhs {
                                    x.re = random();
                                    x.im = random();
                                }

                                let plan = c32_dot_plan_avx(
                                    skip,
                                    k,
                                    m,
                                    n,
                                    Accum::Replace,
                                    conj_lhs,
                                    conj_rhs,
                                );
                                let alpha = c32::new(2.3, -1.5);

                                unsafe {
                                    dot_millikernel(
                                        &plan,
                                        k,
                                        (k * size_of::<c32>()) as isize,
                                        (k * size_of::<c32>()) as isize,
                                        (size_of::<c32>()) as isize,
                                        (m * size_of::<c32>()) as isize,
                                        (plan.mr * size_of::<c32>()) as isize,
                                        (plan.nr * m * size_of::<c32>()) as isize,
                                        (plan.mr * k * size_of::<c32>()) as isize,
                                        (plan.nr * k * size_of::<c32>()) as isize,
                                        lhs.as_ptr() as _,
                                        rhs.as_ptr() as _,
                                        dst.as_mut_ptr() as _,
                                        core::ptr::from_ref(&alpha) as _,
                                    )
                                };

                                for i in 0..m {
                                    for j in 0..n {
                                        let mut acc = c32::ZERO;
                                        for depth in 0..k {
                                            if depth >= skip {
                                                let mut lhs = lhs[depth + i * k];
                                                let mut rhs = rhs[depth + j * k];
                                                if conj_lhs {
                                                    lhs = lhs.conj();
                                                }
                                                if conj_rhs {
                                                    rhs = rhs.conj();
                                                }

                                                acc += lhs * rhs;
                                            }
                                        }
                                        target[i + m * j] = alpha * acc;
                                    }
                                }

                                for i in 0..m * n {
                                    assert!((dst[i] - target[i]).abs() < 1e-4);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_avx_f64() {
        for m in 1..12 {
            for n in 1..12 {
                for k in 0..10 {
                    for skip in 0..Ord::min(2, k) {
                        for conj_lhs in [false, true] {
                            for conj_rhs in [false, true] {
                                let target = &mut *avec![c64::ZERO; m * n];
                                let dst = &mut *avec![c64::ZERO; m * n];
                                let lhs = &mut *avec![c64::ZERO; m * k];
                                let rhs = &mut *avec![c64::ZERO; n * k];

                                for x in &mut *lhs {
                                    x.re = random();
                                    x.im = random();
                                }
                                for x in &mut *rhs {
                                    x.re = random();
                                    x.im = random();
                                }

                                let plan = c64_dot_plan_avx(
                                    skip,
                                    k,
                                    m,
                                    n,
                                    Accum::Replace,
                                    conj_lhs,
                                    conj_rhs,
                                );
                                let alpha = c64::new(2.3, -1.5);

                                unsafe {
                                    dot_millikernel(
                                        &plan,
                                        k,
                                        (k * size_of::<c64>()) as isize,
                                        (k * size_of::<c64>()) as isize,
                                        (size_of::<c64>()) as isize,
                                        (m * size_of::<c64>()) as isize,
                                        (plan.mr * size_of::<c64>()) as isize,
                                        (plan.nr * m * size_of::<c64>()) as isize,
                                        (plan.mr * k * size_of::<c64>()) as isize,
                                        (plan.nr * k * size_of::<c64>()) as isize,
                                        lhs.as_ptr() as _,
                                        rhs.as_ptr() as _,
                                        dst.as_mut_ptr() as _,
                                        core::ptr::from_ref(&alpha) as _,
                                    )
                                };

                                for i in 0..m {
                                    for j in 0..n {
                                        let mut acc = c64::ZERO;
                                        for depth in 0..k {
                                            if depth >= skip {
                                                let mut lhs = lhs[depth + i * k];
                                                let mut rhs = rhs[depth + j * k];
                                                if conj_lhs {
                                                    lhs = lhs.conj();
                                                }
                                                if conj_rhs {
                                                    rhs = rhs.conj();
                                                }

                                                acc += lhs * rhs;
                                            }
                                        }
                                        target[i + m * j] = alpha * acc;
                                    }
                                }

                                for i in 0..m * n {
                                    assert!((dst[i] - target[i]).abs() < 1e-4);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_avx512_f32() {
        for m in 1..12 {
            for n in 1..12 {
                for k in 0..10 {
                    for skip in 0..Ord::min(8, k) {
                        for conj_lhs in [false, true] {
                            for conj_rhs in [false, true] {
                                let target = &mut *avec![c32::ZERO; m * n];
                                let dst = &mut *avec![c32::ZERO; m * n];
                                let lhs = &mut *avec![c32::ZERO; m * k];
                                let rhs = &mut *avec![c32::ZERO; n * k];

                                for x in &mut *lhs {
                                    x.re = random();
                                    x.im = random();
                                }
                                for x in &mut *rhs {
                                    x.re = random();
                                    x.im = random();
                                }

                                let plan = c32_dot_plan_avx512(
                                    skip,
                                    k,
                                    m,
                                    n,
                                    Accum::Replace,
                                    conj_lhs,
                                    conj_rhs,
                                );
                                let alpha = c32::new(2.3, -1.5);

                                unsafe {
                                    dot_millikernel(
                                        &plan,
                                        k,
                                        (k * size_of::<c32>()) as isize,
                                        (k * size_of::<c32>()) as isize,
                                        (size_of::<c32>()) as isize,
                                        (m * size_of::<c32>()) as isize,
                                        (plan.mr * size_of::<c32>()) as isize,
                                        (plan.nr * m * size_of::<c32>()) as isize,
                                        (plan.mr * k * size_of::<c32>()) as isize,
                                        (plan.nr * k * size_of::<c32>()) as isize,
                                        lhs.as_ptr() as _,
                                        rhs.as_ptr() as _,
                                        dst.as_mut_ptr() as _,
                                        core::ptr::from_ref(&alpha) as _,
                                    )
                                };

                                for i in 0..m {
                                    for j in 0..n {
                                        let mut acc = c32::ZERO;
                                        for depth in 0..k {
                                            if depth >= skip {
                                                let mut lhs = lhs[depth + i * k];
                                                let mut rhs = rhs[depth + j * k];
                                                if conj_lhs {
                                                    lhs = lhs.conj();
                                                }
                                                if conj_rhs {
                                                    rhs = rhs.conj();
                                                }

                                                acc += lhs * rhs;
                                            }
                                        }
                                        target[i + m * j] = alpha * acc;
                                    }
                                }

                                for i in 0..m * n {
                                    assert!((dst[i] - target[i]).abs() < 1e-4);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn test_avx512_f64() {
        for m in 1..12 {
            for n in 1..12 {
                for k in 0..10 {
                    for skip in 0..Ord::min(4, k) {
                        for conj_lhs in [false, true] {
                            for conj_rhs in [false, true] {
                                let target = &mut *avec![c64::ZERO; m * n];
                                let dst = &mut *avec![c64::ZERO; m * n];
                                let lhs = &mut *avec![c64::ZERO; m * k];
                                let rhs = &mut *avec![c64::ZERO; n * k];

                                for x in &mut *lhs {
                                    x.re = random();
                                    x.im = random();
                                }
                                for x in &mut *rhs {
                                    x.re = random();
                                    x.im = random();
                                }

                                let plan = c64_dot_plan_avx512(
                                    skip,
                                    k,
                                    m,
                                    n,
                                    Accum::Replace,
                                    conj_lhs,
                                    conj_rhs,
                                );
                                let alpha = c64::new(2.3, -1.5);

                                unsafe {
                                    dot_millikernel(
                                        &plan,
                                        k,
                                        (k * size_of::<c64>()) as isize,
                                        (k * size_of::<c64>()) as isize,
                                        (size_of::<c64>()) as isize,
                                        (m * size_of::<c64>()) as isize,
                                        (plan.mr * size_of::<c64>()) as isize,
                                        (plan.nr * m * size_of::<c64>()) as isize,
                                        (plan.mr * k * size_of::<c64>()) as isize,
                                        (plan.nr * k * size_of::<c64>()) as isize,
                                        lhs.as_ptr() as _,
                                        rhs.as_ptr() as _,
                                        dst.as_mut_ptr() as _,
                                        core::ptr::from_ref(&alpha) as _,
                                    )
                                };

                                for i in 0..m {
                                    for j in 0..n {
                                        let mut acc = c64::ZERO;
                                        for depth in 0..k {
                                            if depth >= skip {
                                                let mut lhs = lhs[depth + i * k];
                                                let mut rhs = rhs[depth + j * k];
                                                if conj_lhs {
                                                    lhs = lhs.conj();
                                                }
                                                if conj_rhs {
                                                    rhs = rhs.conj();
                                                }

                                                acc += lhs * rhs;
                                            }
                                        }
                                        target[i + m * j] = alpha * acc;
                                    }
                                }

                                for i in 0..m * n {
                                    assert!((dst[i] - target[i]).abs() < 1e-4);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

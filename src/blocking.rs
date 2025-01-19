use crate::millikernel::{millikernel, Accum, Plan};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

pub struct BlockingPlan {
    pub top_l: Plan,
    pub mid_l: Plan,
    pub bot_l: Plan,
    pub top_r: Plan,
    pub mid_r: Plan,
    pub bot_r: Plan,
}

pub fn blocking_plan(
    base_plan: fn(extra_masked_top_rows: usize, nrows: usize, ncols: usize, dst: Accum) -> Plan,
    extra_masked_top_rows: usize,
    nrows: usize,
    ncols: usize,
    dst: Accum,

    mc: usize,
    nc: usize,
) -> BlockingPlan {
    let extra = extra_masked_top_rows;
    let nrows = nrows + extra;

    if nrows <= mc {
        let top_l = base_plan(extra, nrows - extra, nc, dst);
        let top_r = base_plan(extra, nrows - extra, ncols % nc, dst);

        BlockingPlan {
            top_l,
            mid_l: top_l,
            bot_l: top_l,
            top_r,
            mid_r: top_r,
            bot_r: top_r,
        }
    } else {
        BlockingPlan {
            top_l: base_plan(extra, mc - extra, nc, dst),
            mid_l: base_plan(0, mc, nc, dst),
            bot_l: base_plan(0, nrows % mc, nc, dst),

            top_r: base_plan(extra, mc - extra, ncols % nc, dst),
            mid_r: base_plan(0, mc, ncols % nc, dst),
            bot_r: base_plan(0, nrows % mc, ncols % nc, dst),
        }
    }
}

pub unsafe fn blocking(
    inner_blocking: &Shape,
    outer_blocking: &Shape,
    dim: &Shape,
    plan: &BlockingPlan,

    dst: *mut (),
    lhs: *const (),
    rhs: *const (),
    alpha: *const (),

    dst_rs: isize,
    dst_cs: isize,

    lhs_cs: isize,
    lhs_inner_rs: isize,
    lhs_outer_cs: isize,

    rhs_rs: isize,
    rhs_cs: isize,
    rhs_inner_cs: isize,
    rhs_outer_rs: isize,
) {
    let mut lhs = lhs;
    let mut rhs = rhs;

    let rhs_stride = rhs_inner_cs * (outer_blocking.n / inner_blocking.n) as isize;

    let BlockingPlan {
        top_l,
        mid_l,
        bot_l,
        top_r,
        mid_r,
        bot_r,
    } = plan;

    let top_l = &mut { *top_l };
    let mid_l = &mut { *mid_l };
    let bot_l = &mut { *bot_l };

    let top_r = &mut { *top_r };
    let mid_r = &mut { *mid_r };
    let bot_r = &mut { *bot_r };
    let mut outer_blocking = *outer_blocking;

    let mut depth = 0;
    while depth < dim.k {
        let kc = Ord::min(outer_blocking.k, dim.k - depth);
        outer_blocking.k = kc;

        blocking_rhs(
            rhs,
            dim,
            &outer_blocking,
            inner_blocking,
            top_l,
            mid_l,
            bot_l,
            dst,
            lhs,
            alpha,
            lhs_cs,
            lhs_inner_rs,
            rhs_rs,
            rhs_cs,
            rhs_inner_cs,
            dst_rs,
            dst_cs,
            rhs_stride,
            top_r,
            mid_r,
            bot_r,
        );

        top_l.flags |= 1;
        mid_l.flags |= 1;
        bot_l.flags |= 1;
        top_r.flags |= 1;
        mid_r.flags |= 1;
        bot_r.flags |= 1;

        rhs = rhs.wrapping_byte_offset(rhs_outer_rs);
        lhs = lhs.wrapping_byte_offset(lhs_outer_cs);
        depth += kc;
    }
}

unsafe fn blocking_rhs(
    rhs: *const (),
    dim: &Shape,
    outer_blocking: &Shape,
    inner_blocking: &Shape,
    top_l: &Plan,
    mid_l: &Plan,
    bot_l: &Plan,
    dst: *mut (),
    lhs: *const (),
    alpha: *const (),
    lhs_cs: isize,
    lhs_inner_rs: isize,
    rhs_rs: isize,
    rhs_cs: isize,
    rhs_inner_cs: isize,
    dst_rs: isize,
    dst_cs: isize,
    rhs_stride: isize,
    top_r: &Plan,
    mid_r: &Plan,
    bot_r: &Plan,
) {
    let mut rhs = rhs;
    let mut dst = dst;
    let mut col = 0;
    while col < dim.n {
        let nc = Ord::min(outer_blocking.n, dim.n - col);
        if nc < outer_blocking.n {
            break;
        }

        blocking_lhs(
            inner_blocking,
            outer_blocking,
            dim,
            top_l,
            mid_l,
            bot_l,
            dst,
            lhs,
            rhs,
            alpha,
            lhs_cs,
            lhs_inner_rs,
            rhs_rs,
            rhs_cs,
            rhs_inner_cs,
            dst_rs,
            dst_cs,
        );
        rhs = rhs.wrapping_byte_offset(rhs_stride);
        dst = dst.wrapping_byte_offset(nc as isize * dst_cs);
        col += nc;
    }
    if col < dim.n {
        blocking_lhs(
            inner_blocking,
            outer_blocking,
            dim,
            top_r,
            mid_r,
            bot_r,
            dst,
            lhs,
            rhs,
            alpha,
            lhs_cs,
            lhs_inner_rs,
            rhs_rs,
            rhs_cs,
            rhs_inner_cs,
            dst_rs,
            dst_cs,
        );
    }
}

unsafe fn blocking_lhs(
    inner_blocking: &Shape,
    outer_blocking: &Shape,
    dim: &Shape,

    top_plan: &Plan,
    mid_plan: &Plan,
    bot_plan: &Plan,

    dst: *mut (),
    lhs: *const (),
    rhs: *const (),
    alpha: *const (),

    lhs_cs: isize,
    lhs_inner_rs: isize,

    rhs_rs: isize,
    rhs_cs: isize,
    rhs_inner_cs: isize,

    dst_rs: isize,
    dst_cs: isize,
) {
    let mut lhs = lhs;
    let mut dst = dst;
    let mut row = 0;

    let stride = lhs_inner_rs * (outer_blocking.m / inner_blocking.m) as isize;
    {
        let mc = Ord::min(outer_blocking.m, dim.m - row);

        millikernel(
            top_plan,
            outer_blocking.k,
            lhs_cs,
            rhs_rs,
            rhs_cs,
            dst_cs,
            dst_rs * inner_blocking.m as isize,
            dst_cs * inner_blocking.n as isize,
            lhs_inner_rs,
            rhs_inner_cs,
            lhs,
            rhs,
            dst,
            alpha,
        );

        lhs = lhs.wrapping_byte_offset(stride);
        dst = dst.wrapping_byte_offset(mc as isize * dst_rs);
        row += mc;
    }

    while row < dim.m {
        let mc = Ord::min(outer_blocking.m, dim.m - row);
        if mc < outer_blocking.m {
            break;
        }

        millikernel(
            mid_plan,
            outer_blocking.k,
            lhs_cs,
            rhs_rs,
            rhs_cs,
            dst_cs,
            dst_rs * inner_blocking.m as isize,
            dst_cs * inner_blocking.n as isize,
            lhs_inner_rs,
            rhs_inner_cs,
            lhs,
            rhs,
            dst,
            alpha,
        );

        lhs = lhs.wrapping_byte_offset(stride);
        dst = dst.wrapping_byte_offset(mc as isize * dst_rs);
        row += mc;
    }

    if row < dim.m {
        millikernel(
            bot_plan,
            outer_blocking.k,
            lhs_cs,
            rhs_rs,
            rhs_cs,
            dst_cs,
            dst_rs * inner_blocking.m as isize,
            dst_cs * inner_blocking.n as isize,
            lhs_inner_rs,
            rhs_inner_cs,
            lhs,
            rhs,
            dst,
            alpha,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cache,
        millikernel::{self, Accum},
        packing,
    };
    use aligned_vec::avec;
    use rand::prelude::*;

    #[test]
    fn test_blocking() {
        let rng = &mut StdRng::seed_from_u64(0);

        for (m, n, k) in [
            (256, 3, 1),
            (256, 3, 17),
            (256, 3, 19),
            (16, 12, 1),
            (5, 3, 4),
            (1024, 3, 4),
            (5, 3, 33),
            (5, 1024, 4),
            (16, 12, 16),
            (16, 24, 16),
            (1024, 1024, 32),
            (1025, 1024, 32),
            (1026, 1024, 32),
            (1027, 1024, 32),
            (1028, 1024, 32),
            (1029, 1024, 32),
            (1030, 1024, 32),
            (1031, 1024, 32),
            (1024, 1027, 32),
            (1025, 1027, 32),
            (1026, 1027, 32),
            (1027, 1027, 32),
            (1028, 1027, 32),
            (1029, 1027, 32),
            (1030, 1027, 32),
            (1031, 1027, 32),
        ] {
            let mr = 16;
            let nr = 12;

            let mut outer_blocking = cache::kernel_params(m, n, k, mr, nr, size_of::<f64>());
            outer_blocking.kc = Ord::min(16, k);
            let kc = outer_blocking.kc;

            let lhs_stride = (mr * outer_blocking.kc) * size_of::<f64>();
            let rhs_stride = (nr * outer_blocking.kc) * size_of::<f64>();

            let plan = blocking_plan(
                millikernel::f64_plan_avx512,
                0,
                m,
                n,
                Accum::Replace,
                outer_blocking.mc,
                outer_blocking.nc,
            );

            let dst = &mut *avec![0.0f64; m * n];
            let unpacked_lhs = &mut *avec![1.0; m * k];
            let unpacked_rhs = &mut *avec![1.0; n * k];
            rng.fill(unpacked_lhs);
            rng.fill(unpacked_rhs);

            let target = &mut *avec![0.0; m * n];
            for j in 0..n {
                for i in 0..m {
                    for depth in 0..k {
                        target[i + m * j] +=
                            unpacked_lhs[i + m * depth] * unpacked_rhs[depth + k * j];
                    }
                }
            }

            for pack_lhs in [false, true] {
                for pack_rhs in [false, true] {
                    let unpacked_lhs = &*unpacked_lhs;
                    let unpacked_rhs = &*unpacked_rhs;

                    let packed_lhs =
                        &mut *avec![0.0; m.next_multiple_of(mr) * k.next_multiple_of(kc)];
                    let packed_rhs =
                        &mut *avec![0.0; n.next_multiple_of(nr) * k.next_multiple_of(kc)];

                    unsafe {
                        if pack_lhs {
                            packing::pack_avx512_u64(
                                packed_lhs.as_mut_ptr() as _,
                                unpacked_lhs.as_ptr() as _,
                                size_of::<f64>() as isize,
                                (m * size_of::<f64>()) as isize,
                                (mr * size_of::<f64>()) as isize,
                                lhs_stride as isize,
                                0,
                                mr,
                                outer_blocking.kc,
                                m,
                                k,
                            );
                        }
                        if pack_rhs {
                            packing::pack_avx512_u64(
                                packed_rhs.as_mut_ptr() as _,
                                unpacked_rhs.as_ptr() as _,
                                (k * size_of::<f64>()) as isize,
                                size_of::<f64>() as isize,
                                (nr * size_of::<f64>()) as isize,
                                rhs_stride as isize,
                                0,
                                nr,
                                outer_blocking.kc,
                                n,
                                k,
                            );
                        }

                        blocking(
                            &Shape { m: mr, n: nr, k: 1 },
                            &Shape {
                                m: outer_blocking.mc,
                                n: outer_blocking.nc,
                                k: outer_blocking.kc,
                            },
                            &Shape { m, n, k },
                            &plan,
                            dst.as_mut_ptr() as _,
                            if pack_lhs {
                                packed_lhs.as_ptr()
                            } else {
                                unpacked_lhs.as_ptr()
                            } as _,
                            if pack_rhs {
                                packed_rhs.as_ptr()
                            } else {
                                unpacked_rhs.as_ptr()
                            } as _,
                            core::ptr::from_ref(&1.0f64) as _,
                            size_of::<f64>() as isize,
                            (m * size_of::<f64>()) as isize,
                            if pack_lhs {
                                (mr * size_of::<f64>()) as isize
                            } else {
                                (m * size_of::<f64>()) as isize
                            },
                            if pack_lhs {
                                lhs_stride as isize
                            } else {
                                (mr * size_of::<f64>()) as isize
                            },
                            if pack_lhs {
                                (lhs_stride * m.div_ceil(mr)) as isize
                            } else {
                                (outer_blocking.kc * m * size_of::<f64>()) as isize
                            },
                            if pack_rhs {
                                (nr * size_of::<f64>()) as isize
                            } else {
                                size_of::<f64>() as isize
                            },
                            if pack_rhs {
                                size_of::<f64>() as isize
                            } else {
                                (k * size_of::<f64>()) as isize
                            },
                            if pack_rhs {
                                rhs_stride as isize
                            } else {
                                (k * nr * size_of::<f64>()) as isize
                            },
                            if pack_rhs {
                                (rhs_stride * n.div_ceil(nr)) as isize
                            } else {
                                (outer_blocking.kc * size_of::<f64>()) as isize
                            },
                        );
                    }

                    for i in 0..m * n {
                        assert!((dst[i] - target[i]).abs() < 1e-10);
                    }
                }
            }
        }
    }
}

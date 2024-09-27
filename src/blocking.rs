use crate::millikernel::{millikernel, Plan};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

pub unsafe fn blocking(
    inner_blocking: &Shape,
    outer_blocking: &Shape,
    dim: &Shape,

    top_l_plan: &Plan,
    mid_l_plan: &Plan,
    bot_l_plan: &Plan,

    top_r_plan: &Plan,
    mid_r_plan: &Plan,
    bot_r_plan: &Plan,

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

    let top_l_plan = &mut { *top_l_plan };
    let mid_l_plan = &mut { *mid_l_plan };
    let bot_l_plan = &mut { *bot_l_plan };

    let top_r_plan = &mut { *top_r_plan };
    let mid_r_plan = &mut { *mid_r_plan };
    let bot_r_plan = &mut { *bot_r_plan };

    let mut depth = 0;
    while depth < dim.k {
        let kc = Ord::min(outer_blocking.k, dim.k - depth);

        blocking_rhs(
            rhs,
            dim,
            outer_blocking,
            inner_blocking,
            top_l_plan,
            mid_l_plan,
            bot_l_plan,
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
            top_r_plan,
            mid_r_plan,
            bot_r_plan,
        );

        top_l_plan.flags |= 1;
        mid_l_plan.flags |= 1;
        bot_l_plan.flags |= 1;
        top_r_plan.flags |= 1;
        mid_r_plan.flags |= 1;
        bot_r_plan.flags |= 1;

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
    top_l_plan: &Plan,
    mid_l_plan: &Plan,
    bot_l_plan: &Plan,
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
    top_r_plan: &Plan,
    mid_r_plan: &Plan,
    bot_r_plan: &Plan,
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
            top_l_plan,
            mid_l_plan,
            bot_l_plan,
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
            top_r_plan,
            mid_r_plan,
            bot_r_plan,
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

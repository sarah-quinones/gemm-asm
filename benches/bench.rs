use aligned_vec::avec;
use blocking::Shape;
use diol::prelude::*;
use gemm_asm::*;
use millikernel::ColMajorDstData;

fn gemm_asm(bencher: Bencher, (m, n): (usize, usize)) {
    let k = 4;
    let dst = &mut *avec![0.0; m * n];
    let lhs = &mut *avec![0.0; m * k];
    let rhs = &mut *avec![0.0; n * k];

    for x in &mut *lhs {
        *x = rand::random();
    }
    for x in &mut *rhs {
        *x = rand::random();
    }
    let plan = millikernel::f64_plan_avx(0, m, n, millikernel::Accum::Replace);

    bencher.bench(move || unsafe {
        millikernel::millikernel(
            &plan,
            k,
            (m * size_of::<f64>()) as isize,
            (size_of::<f64>()) as isize,
            (k * size_of::<f64>()) as isize,
            0,
            0,
            (plan.mr * size_of::<f64>()) as isize,
            (plan.nr * k * size_of::<f64>()) as isize,
            lhs.as_ptr() as _,
            rhs.as_ptr() as _,
            &raw const *&ColMajorDstData {
                ptr: dst.as_mut_ptr() as _,
                col_stride: (m * size_of::<f64>()) as isize,
            } as *mut _,
            core::ptr::from_ref(&1.0) as _,
        )
    });
}

fn pack_row_major(bencher: Bencher, n: usize) {
    let m = 32;
    let dst = &mut *avec![1; m * (n + 1)];
    let src = &*avec![1; m * (n + 1)];
    let src = &src[1..];

    bencher.bench(|| unsafe {
        packing::pack_avx512_u32_row_major(
            dst.as_mut_ptr(),
            src.as_ptr(),
            (n * size_of::<f32>()) as isize,
            (m * size_of::<f32>()) as isize,
            m,
            n,
            0,
        )
    });
}

fn pack_col_major(bencher: Bencher, n: usize) {
    let m = 32;
    let dst = &mut *avec![1; m * n];
    let src = &*avec![1; m * n];

    bencher.bench(|| unsafe {
        packing::pack_avx512_u32_col_major(
            dst.as_mut_ptr(),
            src.as_ptr(),
            (n * size_of::<f32>()) as isize,
            (m * size_of::<f32>()) as isize,
            0,
            m,
            n,
            0,
        )
    });
}

fn pack_naive(bencher: Bencher, n: usize) {
    let m = 32;
    let dst = &mut *avec![1; m * n];
    let src = &*avec![1; m * n];

    bencher.bench(|| unsafe {
        let src_byte_stride = (n * size_of::<f32>()) as isize;
        let dst_byte_stride = (m * size_of::<f32>()) as isize;
        let mut src = src.as_ptr();
        let mut dst = dst.as_mut_ptr();

        for _ in 0..n {
            for i in 0..m {
                *dst.wrapping_add(i) = *src.wrapping_byte_offset(i as isize * src_byte_stride);
            }
            dst = dst.wrapping_byte_offset(dst_byte_stride);
            src = src.wrapping_add(1);
        }
    });
}

fn big_gemm_asm(bencher: Bencher, (m, n, k): (usize, usize, usize)) {
    let mr = 16;
    let nr = 12;

    let mut outer_blocking = cache::kernel_params(m, n, k, mr, nr, size_of::<f64>());
    outer_blocking.mc = outer_blocking.mc.next_multiple_of(mr * 4);

    let lhs_stride = (mr * outer_blocking.kc) * size_of::<f64>();
    let rhs_stride = (nr * outer_blocking.kc) * size_of::<f64>();

    let pack_lhs = true;
    let pack_rhs = false;

    let plan = blocking::blocking_plan(
        millikernel::f64_plan_avx512,
        0,
        m,
        n,
        millikernel::Accum::Replace,
        outer_blocking.mc,
        outer_blocking.nc,
    );

    let dst = &mut *avec![0.0; m * n];
    let unpacked_lhs = &*avec![1.0; m * k];
    let unpacked_rhs = &*avec![1.0; n * k];

    let packed_lhs =
        &mut *avec![1.0; m.next_multiple_of(mr) * k.next_multiple_of(outer_blocking.kc)];
    let packed_rhs =
        &mut *avec![1.0; n.next_multiple_of(nr) * k.next_multiple_of(outer_blocking.kc)];

    let f = move || unsafe {
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

        blocking::blocking(
            &Shape { m: mr, n: nr, k: 1 },
            &Shape {
                m: outer_blocking.mc,
                n: outer_blocking.nc,
                k: outer_blocking.kc,
            },
            &Shape { m, n, k },
            &plan,
            &raw const *&ColMajorDstData {
                ptr: dst.as_mut_ptr() as _,
                col_stride: (m * size_of::<f64>()) as isize,
            } as *mut _,
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
    };
    bencher.bench(f);
}

fn big_gemm_old(bencher: Bencher, (m, n, k): (usize, usize, usize)) {
    let dst = &mut *avec![0.0; m * n];
    let lhs = &*avec![1.0; m * k];
    let rhs = &*avec![1.0; n * k];

    let f = move || unsafe {
        gemm::gemm(
            m,
            n,
            k,
            dst.as_mut_ptr() as _,
            m as isize,
            1,
            false,
            lhs.as_ptr(),
            m as isize,
            1,
            rhs.as_ptr(),
            k as isize,
            1,
            0.0,
            1.0,
            false,
            false,
            false,
            gemm::Parallelism::None,
        );
    };
    bencher.bench(f);
}

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);

    bench.register_many(list![pack_naive, pack_col_major, pack_row_major], [32]);
    bench.register_many(list![gemm_asm], [(8, 4)]);
    bench.register_many(
        list![big_gemm_asm, big_gemm_old],
        [
            (1024, 1024, 1024),
            (128, 128, 128),
            (256, 256, 256),
            (512, 512, 512),
            (4096, 4096, 4096),
            (1024, 1024, 128),
            (4096, 4096, 128),
        ],
    );

    bench.run()?;

    Ok(())
}

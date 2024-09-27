use aligned_vec::avec;
use diol::prelude::*;
use gemm_asm::*;

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

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);

    bench.register_many(list![pack_naive, pack_col_major, pack_row_major], [32]);
    bench.register_many(list![gemm_asm], [(8, 4)]);

    bench.run()?;

    Ok(())
}

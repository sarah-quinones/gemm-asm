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
    let plan = f64_plan_avx(0, m, n, Accum::Replace);

    bencher.bench(move || unsafe {
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
    });
}

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);

    bench.register_many(list![gemm_asm], [(8, 4)]);

    bench.run()?;

    Ok(())
}

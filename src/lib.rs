#![cfg_attr(
    feature = "nightly",
    feature(stdarch_x86_avx512, avx512_target_feature)
)]

pub mod millikernel;
pub mod transpose;

pub mod packing;

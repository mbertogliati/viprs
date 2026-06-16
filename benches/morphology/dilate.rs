#![allow(missing_docs)]
#[path = "../common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use viprs::domain::{format::U8, ops::morphology::dilate::Dilate};

fn bench_dilate(c: &mut Criterion) {
    common::bench_direct_op_with_regions::<Dilate, U8, U8, _, _, _>(
        c,
        "dilate_u8",
        1,
        1,
        Dilate::cross_3x3,
        common::full_image_regions::<Dilate>,
        |samples| {
            (0..samples)
                .map(|i| if i % 7 == 0 { 255 } else { 0 })
                .collect()
        },
    );
}

criterion_group!(benches, bench_dilate);
criterion_main!(benches);

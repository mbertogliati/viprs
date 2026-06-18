#![allow(missing_docs)]
#[path = "../common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use viprs::domain::{format::U8, ops::morphology::erode::Erode};

fn bench_erode(c: &mut Criterion) {
    common::bench_direct_op_with_regions::<Erode, U8, U8, _, _, _>(
        c,
        "erode_u8",
        1,
        1,
        || Erode::rect(3).unwrap(),
        common::full_image_regions::<Erode>,
        |samples| {
            (0..samples)
                .map(|i| if i % 5 == 0 { 0 } else { 255 })
                .collect()
        },
    );
}

criterion_group!(benches, bench_erode);
criterion_main!(benches);

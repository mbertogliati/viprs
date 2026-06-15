#[path = "../common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use viprs::domain::{format::U8, ops::morphology::open::Open};

fn bench_open(c: &mut Criterion) {
    common::bench_direct_op_with_regions::<Open, U8, U8, _, _, _>(
        c,
        "open_u8",
        1,
        1,
        || Open::rect(3).unwrap(),
        common::full_image_regions::<Open>,
        |samples| {
            (0..samples)
                .map(|i| if i % 11 == 0 { 255 } else { 0 })
                .collect()
        },
    );
}

criterion_group!(benches, bench_open);
criterion_main!(benches);

#![allow(missing_docs)]
#[path = "../common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use viprs::domain::{format::U8, ops::morphology::close::Close};

fn bench_close(c: &mut Criterion) {
    common::bench_direct_op_with_regions::<Close, U8, U8, _, _, _>(
        c,
        "close_u8",
        1,
        1,
        || Close::rect(3).unwrap(),
        common::full_image_regions::<Close>,
        |samples| {
            (0..samples)
                .map(|i| if i % 13 == 0 { 0 } else { 255 })
                .collect()
        },
    );
}

criterion_group!(benches, bench_close);
criterion_main!(benches);

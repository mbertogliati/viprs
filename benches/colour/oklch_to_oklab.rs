#[path = "../common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use viprs::domain::{
    colorspace::{Oklab, Oklch},
    format::F32,
    ops::colour::oklab::OklchToOklab,
};

fn bench_oklch_to_oklab(c: &mut Criterion) {
    common::bench_colour_convert_with_regions::<OklchToOklab, Oklch, Oklab, F32, F32, _, _, _>(
        c,
        "oklch_to_oklab",
        3,
        3,
        || OklchToOklab,
        |converter, size| {
            common::colour_convert_tile_regions::<OklchToOklab, Oklch, Oklab>(converter, size)
        },
        |samples| vec![0.25f32; samples],
    );
}

criterion_group!(benches, bench_oklch_to_oklab);
criterion_main!(benches);

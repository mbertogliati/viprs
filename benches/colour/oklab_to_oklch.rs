#[path = "../common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use viprs::domain::{
    colorspace::{Oklab, Oklch},
    format::F32,
    ops::colour::oklab::OklabToOklch,
};

fn bench_oklab_to_oklch(c: &mut Criterion) {
    common::bench_colour_convert_with_regions::<OklabToOklch, Oklab, Oklch, F32, F32, _, _, _>(
        c,
        "oklab_to_oklch",
        3,
        3,
        || OklabToOklch,
        |converter, size| {
            common::colour_convert_tile_regions::<OklabToOklch, Oklab, Oklch>(converter, size)
        },
        |samples| vec![0.25f32; samples],
    );
}

criterion_group!(benches, bench_oklab_to_oklch);
criterion_main!(benches);

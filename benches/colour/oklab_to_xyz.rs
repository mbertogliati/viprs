#[path = "../common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use viprs::domain::{
    colorspace::{Oklab, Xyz},
    format::F32,
    ops::colour::oklab::OklabToXyz,
};

fn bench_oklab_to_xyz(c: &mut Criterion) {
    common::bench_colour_convert_with_regions::<OklabToXyz, Oklab, Xyz, F32, F32, _, _, _>(
        c,
        "oklab_to_xyz",
        3,
        3,
        || OklabToXyz,
        |converter, size| {
            common::colour_convert_tile_regions::<OklabToXyz, Oklab, Xyz>(converter, size)
        },
        |samples| vec![0.25f32; samples],
    );
}

criterion_group!(benches, bench_oklab_to_xyz);
criterion_main!(benches);

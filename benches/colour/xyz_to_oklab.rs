#[path = "../common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use viprs::domain::{
    colorspace::{Oklab, Xyz},
    format::F32,
    ops::colour::oklab::XyzToOklab,
};

fn bench_xyz_to_oklab(c: &mut Criterion) {
    common::bench_colour_convert_with_regions::<XyzToOklab, Xyz, Oklab, F32, F32, _, _, _>(
        c,
        "xyz_to_oklab_f32",
        3,
        3,
        || XyzToOklab,
        |converter, size| {
            common::colour_convert_tile_regions::<XyzToOklab, Xyz, Oklab>(converter, size)
        },
        |samples| vec![0.5f32; samples],
    );
}

criterion_group!(benches, bench_xyz_to_oklab);
criterion_main!(benches);

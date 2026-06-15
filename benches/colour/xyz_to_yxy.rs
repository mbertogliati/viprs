#[path = "../common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use viprs::domain::{
    colorspace::{Xyz, Yxy},
    format::F32,
    ops::colour::xyz_to_yxy::XyzToYxy,
};

fn bench_xyz_to_yxy(c: &mut Criterion) {
    common::bench_colour_convert_with_regions::<XyzToYxy, Xyz, Yxy, F32, F32, _, _, _>(
        c,
        "xyz_to_yxy",
        3,
        3,
        || XyzToYxy,
        |converter, size| {
            common::colour_convert_tile_regions::<XyzToYxy, Xyz, Yxy>(converter, size)
        },
        |samples| vec![0.5f32; samples],
    );
}

criterion_group!(benches, bench_xyz_to_yxy);
criterion_main!(benches);

#![allow(missing_docs)]
#[path = "../common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use viprs::domain::{
    colorspace::{Xyz, Yxy},
    format::F32,
    ops::colour::yxy_to_xyz::YxyToXyz,
};

fn bench_yxy_to_xyz(c: &mut Criterion) {
    common::bench_colour_convert_with_regions::<YxyToXyz, Yxy, Xyz, F32, F32, _, _, _>(
        c,
        "yxy_to_xyz",
        3,
        3,
        || YxyToXyz,
        |converter, size| {
            common::colour_convert_tile_regions::<YxyToXyz, Yxy, Xyz>(converter, size)
        },
        |samples| vec![0.5f32; samples],
    );
}

criterion_group!(benches, bench_yxy_to_xyz);
criterion_main!(benches);

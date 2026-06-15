#[path = "../common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use viprs::domain::{
    colorspace::{Lch, Ucs},
    format::F32,
    ops::colour::lch_to_ucs::LchToUcs,
};

fn bench_lch_to_ucs(c: &mut Criterion) {
    common::bench_colour_convert_with_regions::<LchToUcs, Lch, Ucs, F32, F32, _, _, _>(
        c,
        "lch_to_ucs",
        3,
        3,
        || LchToUcs,
        |converter, size| {
            common::colour_convert_tile_regions::<LchToUcs, Lch, Ucs>(converter, size)
        },
        |samples| vec![0.5f32; samples],
    );
}

criterion_group!(benches, bench_lch_to_ucs);
criterion_main!(benches);

#[path = "../common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use viprs::domain::{
    colorspace::{Lch, Ucs},
    format::F32,
    ops::colour::ucs_to_lch::UcsToLch,
};

fn bench_ucs_to_lch(c: &mut Criterion) {
    common::bench_colour_convert_with_regions::<UcsToLch, Ucs, Lch, F32, F32, _, _, _>(
        c,
        "ucs_to_lch",
        3,
        3,
        || UcsToLch,
        |converter, size| {
            common::colour_convert_tile_regions::<UcsToLch, Ucs, Lch>(converter, size)
        },
        |samples| vec![0.75f32; samples],
    );
}

criterion_group!(benches, bench_ucs_to_lch);
criterion_main!(benches);

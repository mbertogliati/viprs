#[path = "../common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use viprs::domain::{
    colorspace::{Greyscale, SRgb},
    format::U8,
    ops::colour::bw_to_srgb::BwToSRgb,
};

fn bench_bw_to_srgb(c: &mut Criterion) {
    common::bench_colour_convert_with_regions::<BwToSRgb, Greyscale, SRgb, U8, U8, _, _, _>(
        c,
        "bw_to_srgb",
        1,
        3,
        || BwToSRgb,
        common::colour_convert_tile_regions::<BwToSRgb, Greyscale, SRgb>,
        |samples| {
            (0..samples)
                .map(|idx| ((idx * 29 + idx / 7) % 256) as u8)
                .collect()
        },
    );
}

criterion_group!(benches, bench_bw_to_srgb);
criterion_main!(benches);

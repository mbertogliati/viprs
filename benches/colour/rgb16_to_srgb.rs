#[path = "../common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use viprs::domain::{
    colorspace::{Rgb16, SRgb},
    format::{U8, U16},
    ops::colour::rgb16_to_srgb::Rgb16ToSRgb,
};

fn bench_rgb16_to_srgb(c: &mut Criterion) {
    common::bench_colour_convert_with_regions::<Rgb16ToSRgb, Rgb16, SRgb, U16, U8, _, _, _>(
        c,
        "rgb16_to_srgb",
        3,
        3,
        || Rgb16ToSRgb,
        common::colour_convert_tile_regions::<Rgb16ToSRgb, Rgb16, SRgb>,
        |samples| {
            (0..samples)
                .map(|idx| ((idx * 977 + idx / 5 * 131) % 65_536) as u16)
                .collect()
        },
    );
}

criterion_group!(benches, bench_rgb16_to_srgb);
criterion_main!(benches);

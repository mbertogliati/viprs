#![allow(missing_docs)]
#[cfg(feature = "gif")]
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
#[cfg(feature = "gif")]
use viprs::{
    adapters::codecs::GifCodec,
    domain::{codec_options::SaveOptions, format::U8, image::InMemoryImage},
    ports::codec::ImageEncoder,
};

#[cfg(feature = "gif")]
const DIMENSIONS: [u32; 3] = [512, 2048, 8192];

#[cfg(feature = "gif")]
fn raw_rgba_bytes(dimension: u32) -> u64 {
    u64::from(dimension) * u64::from(dimension) * 4
}

#[cfg(feature = "gif")]
fn rgba_pixels(dimension: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(raw_rgba_bytes(dimension) as usize);
    for y in 0..dimension {
        for x in 0..dimension {
            pixels.push((x & 0xFF) as u8);
            pixels.push((y & 0xFF) as u8);
            pixels.push(((x * 3 + y * 5) & 0xFF) as u8);
            pixels.push(if (x + y) % 7 == 0 { 0 } else { 255 });
        }
    }
    pixels
}

#[cfg(feature = "gif")]
fn make_image(dimension: u32) -> InMemoryImage<U8> {
    match InMemoryImage::<U8>::from_buffer(dimension, dimension, 4, rgba_pixels(dimension)) {
        Ok(image) => image,
        Err(err) => panic!("gif encode bench fixture image must be valid: {err}"),
    }
}

#[cfg(feature = "gif")]
fn bench_gif_encode(c: &mut Criterion) {
    let codec = GifCodec::default();
    let images: Vec<(u32, InMemoryImage<U8>)> = DIMENSIONS
        .iter()
        .copied()
        .map(|dimension| (dimension, make_image(dimension)))
        .collect();
    let opts = SaveOptions::default().with_colors(256).with_dither(true);
    let mut group = c.benchmark_group("codec_gif_encode_rgba_u8");

    for (dimension, image) in &images {
        group.throughput(Throughput::Bytes(raw_rgba_bytes(*dimension)));
        group.bench_with_input(BenchmarkId::from_parameter(dimension), image, |b, image| {
            b.iter(|| {
                let encoded = match codec.encode_with_options::<U8>(image, &opts) {
                    Ok(encoded) => encoded,
                    Err(err) => panic!("gif encode benchmark must succeed: {err}"),
                };
                black_box(encoded);
            });
        });
    }

    group.finish();
}

#[cfg(feature = "gif")]
criterion_group!(benches, bench_gif_encode);
#[cfg(feature = "gif")]
criterion_main!(benches);

#[cfg(not(feature = "gif"))]
fn main() {}

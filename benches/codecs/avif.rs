#![allow(missing_docs)]
#[cfg(feature = "avif")]
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
#[cfg(feature = "avif")]
use viprs::{
    adapters::codecs::AvifCodec,
    domain::{format::U8, image::Image},
    ports::codec::{ImageDecoder, ImageEncoder},
};

#[cfg(feature = "avif")]
const DIMENSIONS: [u32; 3] = [512, 2048, 8192];

#[cfg(feature = "avif")]
struct AvifFixture {
    dimension: u32,
    image: Image<U8>,
    encoded: Vec<u8>,
}

#[cfg(feature = "avif")]
fn raw_rgb_bytes(dimension: u32) -> u64 {
    u64::from(dimension) * u64::from(dimension) * 3
}

#[cfg(feature = "avif")]
fn rgb_pixels(dimension: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(raw_rgb_bytes(dimension) as usize);
    for y in 0..dimension {
        for x in 0..dimension {
            pixels.push((x & 0xFF) as u8);
            pixels.push((y & 0xFF) as u8);
            pixels.push((((x * 3) + (y * 5)) & 0xFF) as u8);
        }
    }
    pixels
}

#[cfg(feature = "avif")]
fn make_image(dimension: u32) -> Image<U8> {
    match Image::<U8>::from_buffer(dimension, dimension, 3, rgb_pixels(dimension)) {
        Ok(image) => image,
        Err(err) => panic!("avif bench fixture image must be valid: {err}"),
    }
}

#[cfg(feature = "avif")]
fn build_fixtures() -> Vec<AvifFixture> {
    let codec = AvifCodec;
    DIMENSIONS
        .iter()
        .map(|&dimension| {
            let image = make_image(dimension);
            let encoded = match codec.encode::<U8>(&image) {
                Ok(encoded) => encoded,
                Err(err) => panic!("avif bench fixture encode must succeed: {err}"),
            };
            AvifFixture {
                dimension,
                image,
                encoded,
            }
        })
        .collect()
}

#[cfg(feature = "avif")]
fn bench_avif_decode(c: &mut Criterion) {
    let codec = AvifCodec;
    let fixtures = build_fixtures();
    let mut group = c.benchmark_group("codec_avif_decode_u8");

    for fixture in &fixtures {
        group.throughput(Throughput::Bytes(fixture.encoded.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(fixture.dimension),
            fixture,
            |b, fixture| {
                b.iter(|| {
                    let decoded = match codec.decode::<U8>(&fixture.encoded) {
                        Ok(decoded) => decoded,
                        Err(err) => panic!("avif decode benchmark must succeed: {err}"),
                    };
                    black_box(decoded);
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "avif")]
fn bench_avif_encode(c: &mut Criterion) {
    let codec = AvifCodec;
    let fixtures = build_fixtures();
    let mut group = c.benchmark_group("codec_avif_encode_u8");

    for fixture in &fixtures {
        group.throughput(Throughput::Bytes(raw_rgb_bytes(fixture.dimension)));
        group.bench_with_input(
            BenchmarkId::from_parameter(fixture.dimension),
            fixture,
            |b, fixture| {
                b.iter(|| {
                    let encoded = match codec.encode::<U8>(&fixture.image) {
                        Ok(encoded) => encoded,
                        Err(err) => panic!("avif encode benchmark must succeed: {err}"),
                    };
                    black_box(encoded);
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "avif")]
criterion_group!(benches, bench_avif_decode, bench_avif_encode);
#[cfg(feature = "avif")]
criterion_main!(benches);

#[cfg(not(feature = "avif"))]
fn main() {}

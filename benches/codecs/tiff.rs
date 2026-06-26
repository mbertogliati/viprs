#![allow(missing_docs)]
#[cfg(feature = "tiff")]
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
#[cfg(feature = "tiff")]
use viprs::{
    adapters::codecs::TiffCodec,
    domain::{format::U8, image::InMemoryImage},
    ports::codec::{ImageDecoder, ImageEncoder},
};

#[cfg(feature = "tiff")]
const DIMENSIONS: [u32; 3] = [512, 2048, 8192];

#[cfg(feature = "tiff")]
struct TiffFixture {
    dimension: u32,
    image: InMemoryImage<U8>,
    encoded: Vec<u8>,
}

#[cfg(feature = "tiff")]
fn raw_rgb_bytes(dimension: u32) -> u64 {
    u64::from(dimension) * u64::from(dimension) * 3
}

#[cfg(feature = "tiff")]
fn rgb_pixels(dimension: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(raw_rgb_bytes(dimension) as usize);
    for y in 0..dimension {
        for x in 0..dimension {
            pixels.push((x & 0xFF) as u8);
            pixels.push((y & 0xFF) as u8);
            pixels.push(((x ^ y) & 0xFF) as u8);
        }
    }
    pixels
}

#[cfg(feature = "tiff")]
fn make_image(dimension: u32) -> InMemoryImage<U8> {
    match InMemoryImage::<U8>::from_buffer(dimension, dimension, 3, rgb_pixels(dimension)) {
        Ok(image) => image,
        Err(err) => panic!("tiff bench fixture image must be valid: {err}"),
    }
}

#[cfg(feature = "tiff")]
fn build_fixtures() -> Vec<TiffFixture> {
    let codec = TiffCodec::default();
    DIMENSIONS
        .iter()
        .map(|&dimension| {
            let image = make_image(dimension);
            let encoded = match codec.encode::<U8>(&image) {
                Ok(encoded) => encoded,
                Err(err) => panic!("tiff bench fixture encode must succeed: {err}"),
            };
            TiffFixture {
                dimension,
                image,
                encoded,
            }
        })
        .collect()
}

#[cfg(feature = "tiff")]
fn bench_tiff_decode(c: &mut Criterion) {
    let codec = TiffCodec::default();
    let fixtures = build_fixtures();
    let mut group = c.benchmark_group("codec_tiff_decode_u8");

    for fixture in &fixtures {
        group.throughput(Throughput::Bytes(fixture.encoded.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(fixture.dimension),
            fixture,
            |b, fixture| {
                b.iter(|| {
                    let decoded = match codec.decode::<U8>(&fixture.encoded) {
                        Ok(decoded) => decoded,
                        Err(err) => panic!("tiff decode benchmark must succeed: {err}"),
                    };
                    black_box(decoded);
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "tiff")]
fn bench_tiff_encode(c: &mut Criterion) {
    let codec = TiffCodec::default();
    let fixtures = build_fixtures();
    let mut group = c.benchmark_group("codec_tiff_encode_u8");

    for fixture in &fixtures {
        group.throughput(Throughput::Bytes(raw_rgb_bytes(fixture.dimension)));
        group.bench_with_input(
            BenchmarkId::from_parameter(fixture.dimension),
            fixture,
            |b, fixture| {
                b.iter(|| {
                    let encoded = match codec.encode::<U8>(&fixture.image) {
                        Ok(encoded) => encoded,
                        Err(err) => panic!("tiff encode benchmark must succeed: {err}"),
                    };
                    black_box(encoded);
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "tiff")]
criterion_group!(benches, bench_tiff_decode, bench_tiff_encode);
#[cfg(feature = "tiff")]
criterion_main!(benches);

#[cfg(not(feature = "tiff"))]
fn main() {}

#[cfg(feature = "heif")]
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
#[cfg(feature = "heif")]
use libheif_rs::{
    Channel, ColorSpace, CompressionFormat, EncoderQuality, HeifContext, Image as HeifImage,
    LibHeif, RgbChroma,
};
#[cfg(feature = "heif")]
use viprs::{adapters::codecs::HeifCodec, domain::format::U8, ports::codec::ImageDecoder};

#[cfg(feature = "heif")]
const DIMENSIONS: [u32; 3] = [512, 2048, 8192];

#[cfg(feature = "heif")]
struct HeifFixture {
    dimension: u32,
    encoded: Vec<u8>,
}

#[cfg(feature = "heif")]
fn raw_rgb_bytes(dimension: u32) -> u64 {
    u64::from(dimension) * u64::from(dimension) * 3
}

#[cfg(feature = "heif")]
fn rgb_pixels(dimension: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(raw_rgb_bytes(dimension) as usize);
    for y in 0..dimension {
        for x in 0..dimension {
            pixels.push((x & 0xFF) as u8);
            pixels.push((y & 0xFF) as u8);
            pixels.push((((x * 7) + (y * 11)) & 0xFF) as u8);
        }
    }
    pixels
}

#[cfg(feature = "heif")]
fn encode_fixture_heif(dimension: u32) -> Vec<u8> {
    let pixels = rgb_pixels(dimension);
    let mut image = match HeifImage::new(dimension, dimension, ColorSpace::Rgb(RgbChroma::Rgb)) {
        Ok(image) => image,
        Err(err) => panic!("heif bench fixture image allocation must succeed: {err}"),
    };
    if let Err(err) = image.create_plane(Channel::Interleaved, dimension, dimension, 8) {
        panic!("heif bench fixture plane allocation must succeed: {err}");
    }

    let mut planes = image.planes_mut();
    let plane = match planes.interleaved.as_mut() {
        Some(plane) => plane,
        None => panic!("heif bench fixture interleaved plane must exist"),
    };
    let row_bytes = dimension as usize * 3;
    for row in 0..dimension as usize {
        let src_start = row * row_bytes;
        let src_end = src_start + row_bytes;
        let dst_start = row * plane.stride;
        let dst_end = dst_start + row_bytes;
        plane.data[dst_start..dst_end].copy_from_slice(&pixels[src_start..src_end]);
    }

    let lib_heif = LibHeif::new();
    let mut context = match HeifContext::new() {
        Ok(context) => context,
        Err(err) => panic!("heif bench fixture context creation must succeed: {err}"),
    };
    let mut encoder = match lib_heif.encoder_for_format(CompressionFormat::Hevc) {
        Ok(encoder) => encoder,
        Err(_) => match lib_heif.encoder_for_format(CompressionFormat::Av1) {
            Ok(encoder) => encoder,
            Err(err) => panic!("heif bench fixture encoder lookup must succeed: {err}"),
        },
    };
    if let Err(err) = encoder.set_quality(EncoderQuality::LossLess) {
        panic!("heif bench fixture encoder quality must be settable: {err}");
    }
    if let Err(err) = context.encode_image(&image, &mut encoder, None) {
        panic!("heif bench fixture encode must succeed: {err}");
    }
    match context.write_to_bytes() {
        Ok(encoded) => encoded,
        Err(err) => panic!("heif bench fixture write_to_bytes must succeed: {err}"),
    }
}

#[cfg(feature = "heif")]
fn build_fixtures() -> Vec<HeifFixture> {
    DIMENSIONS
        .iter()
        .map(|&dimension| HeifFixture {
            dimension,
            encoded: encode_fixture_heif(dimension),
        })
        .collect()
}

#[cfg(feature = "heif")]
fn bench_heif_decode(c: &mut Criterion) {
    let codec = HeifCodec;
    let fixtures = build_fixtures();
    let mut group = c.benchmark_group("codec_heif_decode_u8");

    for fixture in &fixtures {
        group.throughput(Throughput::Bytes(fixture.encoded.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(fixture.dimension),
            fixture,
            |b, fixture| {
                b.iter(|| {
                    let decoded = match codec.decode::<U8>(&fixture.encoded) {
                        Ok(decoded) => decoded,
                        Err(err) => panic!("heif decode benchmark must succeed: {err}"),
                    };
                    black_box(decoded);
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "heif")]
criterion_group!(benches, bench_heif_decode);
#[cfg(feature = "heif")]
criterion_main!(benches);

#[cfg(not(feature = "heif"))]
fn main() {}

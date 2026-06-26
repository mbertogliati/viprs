#![allow(missing_docs)]
/// Benchmark: streaming `DecoderSource<D>` tile reads.
///
/// The fake decoder exposes 512, 2048, and 8192 square images but only decodes
/// the requested tile into the caller-provided buffer. The setup assertion on
/// `resident_decoded_bytes()` makes the benchmark fail if the streaming path
/// regresses to retaining a full decoded frame.
use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::sources::decoder_source::DecoderSource,
    domain::{
        codec_options::LoadOptions,
        error::ViprsError,
        format::{BandFormat, U8},
        image::{InMemoryImage, Region},
    },
    ports::{
        codec::{ImageDecoder, ImageMetadataProbe, TileImageDecoder},
        source::ImageSource,
    },
};

#[cfg(feature = "png")]
use viprs::{
    adapters::codecs::{PngCodec, PngEncoder},
    ports::codec::ImageEncoder,
};

const TILE: u32 = 128;

fn must<T>(result: Result<T, ViprsError>, context: &str) -> T {
    match result {
        Ok(value) => value,
        Err(err) => panic!("{context}: {err}"),
    }
}

struct BenchmarkTileDecoder {
    size: u32,
}

impl ImageDecoder for BenchmarkTileDecoder {
    fn format_name(&self) -> &'static str {
        "benchmark-tile"
    }

    fn sniff(&self, _: &[u8]) -> bool {
        true
    }

    fn decode<F: BandFormat>(&self, _: &[u8]) -> Result<InMemoryImage<F>, ViprsError> {
        Err(ViprsError::Codec(
            "benchmark tile decoder must not full-decode".into(),
        ))
    }

    fn decode_with_options<F: BandFormat>(
        &self,
        src: &[u8],
        _: &LoadOptions,
    ) -> Result<InMemoryImage<F>, ViprsError> {
        self.decode(src)
    }

    fn probe(&self, _: &[u8]) -> Result<(u32, u32, u32), ViprsError> {
        Ok((self.size, self.size, 1))
    }
}

impl TileImageDecoder for BenchmarkTileDecoder {
    fn probe_with_options(
        &self,
        src: &[u8],
        opts: &LoadOptions,
    ) -> Result<ImageMetadataProbe, ViprsError> {
        let (width, height, bands) = self.probe(src)?;
        let factor = opts.shrink_factor.map_or(1, std::num::NonZeroU8::get);
        Ok(ImageMetadataProbe::new(
            (width / u32::from(factor)).max(1),
            (height / u32::from(factor)).max(1),
            bands,
        ))
    }

    fn decode_region_into<F: BandFormat>(
        &self,
        _: &[u8],
        _: &LoadOptions,
        region: Region,
        output: &mut [u8],
    ) -> Result<(), ViprsError> {
        if F::ID != U8::ID {
            return Err(ViprsError::Codec(
                "benchmark tile decoder only decodes U8".into(),
            ));
        }

        let expected = region.pixel_count();
        if output.len() != expected {
            return Err(ViprsError::Codec(format!(
                "benchmark tile output mismatch: got {}, expected {expected}",
                output.len()
            )));
        }

        output.fill(127);
        Ok(())
    }
}

fn bench_streaming_tile_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("decoder_source_streaming_tile_read");

    for &size in &[512u32, 2048, 8192] {
        let encoded: Arc<[u8]> = Arc::from(&b"encoded"[..]);
        let source = must(
            DecoderSource::<_, U8>::streaming_shared(
                BenchmarkTileDecoder { size },
                encoded,
                LoadOptions::default(),
            ),
            "create streaming decoder source",
        );
        assert_eq!(source.resident_decoded_bytes(), 0);

        let region = Region::new(0, 0, TILE, TILE);
        let mut output = vec![0u8; region.pixel_count()];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                must(
                    source.read_region(region, &mut output),
                    "read streaming tile",
                );
                black_box(&output);
            });
        });
    }

    group.finish();
}

#[cfg(feature = "png")]
fn bench_png_eager_vs_streaming_tile_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("decoder_source_png_eager_vs_streaming_tile_read");

    for &size in &[512u32, 2048, 8192] {
        let pixel_count = size as usize * size as usize;
        let image = must(
            InMemoryImage::<U8>::from_buffer(size, size, 1, vec![17u8; pixel_count])
                .map_err(|err| ViprsError::Codec(err.to_string())),
            "build PNG benchmark image",
        );
        let encoded = must(
            PngEncoder {
                compression: 0,
                ..PngEncoder::default()
            }
            .encode(&image),
            "encode PNG benchmark image",
        );

        let eager = must(
            DecoderSource::<_, U8>::new(PngCodec::default(), &encoded),
            "create eager PNG decoder source",
        );
        let streaming = must(
            DecoderSource::<_, U8>::streaming_shared(
                PngCodec::default(),
                Arc::from(encoded.into_boxed_slice()),
                LoadOptions::default(),
            ),
            "create streaming PNG decoder source",
        );
        assert_eq!(eager.resident_decoded_bytes(), pixel_count);
        assert_eq!(streaming.resident_decoded_bytes(), 0);

        let region = Region::new(0, 0, TILE, TILE);
        let mut eager_output = vec![0u8; region.pixel_count()];
        let mut streaming_output = vec![0u8; region.pixel_count()];

        group.bench_with_input(BenchmarkId::new("eager", size), &size, |b, _| {
            b.iter(|| {
                must(
                    eager.read_region(region, &mut eager_output),
                    "read eager PNG tile",
                );
                black_box(&eager_output);
            });
        });
        group.bench_with_input(BenchmarkId::new("streaming", size), &size, |b, _| {
            b.iter(|| {
                must(
                    streaming.read_region(region, &mut streaming_output),
                    "read streaming PNG tile",
                );
                black_box(&streaming_output);
            });
        });
    }

    group.finish();
}

#[cfg(not(feature = "png"))]
fn bench_png_eager_vs_streaming_tile_read(_: &mut Criterion) {}

criterion_group!(
    benches,
    bench_streaming_tile_read,
    bench_png_eager_vs_streaming_tile_read
);
criterion_main!(benches);

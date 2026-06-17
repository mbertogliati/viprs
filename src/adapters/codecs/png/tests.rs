use std::io::Cursor;
use std::num::NonZeroU8;
use std::path::PathBuf;
use std::sync::{Arc, Barrier, Mutex};
use std::time::Duration;

/// Tests that mutate the global `PNG_ROW_DECODE_PROBE` must hold this lock to
/// avoid racing with each other when `cargo test` runs threads in parallel.
static PROBE_MUTEX: Mutex<()> = Mutex::new(());

use png::{ColorType as RawColorType, Decoder as RawDecoder, Filter, Unit as RawUnit};

#[cfg(feature = "libspng")]
use super::decode_full::decode_png_with_libspng;
use super::region_decode::clamp_region_coordinate;
use super::state::PNG_ROW_DECODE_PROBE;
use super::{PngCodec, PngEncoder};
use crate::adapters::sources::decoder_source::DecoderSource;
use crate::domain::codec_options::{LoadOptions, PngFilterStrategy, SaveOptions};
use crate::domain::error::ViprsError;
use crate::domain::format::{U8, U16};
use crate::domain::image::{Image, ImageMetadata, Interpretation, Region};
use crate::ports::codec::{ImageDecoder, ImageEncoder, TileImageDecoder};
use crate::ports::source::ImageSource;

fn clamped_region_pixels_u8(image: &Image<U8>, region: Region) -> Vec<u8> {
    let bands = image.bands() as usize;
    let mut output = vec![0u8; region.pixel_count() * bands];

    for out_y in 0..region.height {
        let src_y = clamp_region_coordinate(region.y, out_y, image.height()) as usize;
        for out_x in 0..region.width {
            let src_x = clamp_region_coordinate(region.x, out_x, image.width()) as usize;
            let src = (src_y * image.width() as usize + src_x) * bands;
            let dst = (out_y as usize * region.width as usize + out_x as usize) * bands;
            output[dst..dst + bands].copy_from_slice(&image.pixels()[src..src + bands]);
        }
    }

    output
}

fn assert_u8_pixels_equal_with_index(expected: &[u8], actual: &[u8], context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{context}: decoded sample count mismatch"
    );
    if let Some(idx) = actual
        .iter()
        .zip(expected.iter())
        .position(|(actual_sample, expected_sample)| actual_sample != expected_sample)
    {
        panic!(
            "{context}: first mismatching sample index {idx} (expected {}, got {})",
            expected[idx], actual[idx]
        );
    }
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

fn png_test_output_path(name: &str) -> PathBuf {
    let dir = PathBuf::from("target/test-output/png");
    std::fs::create_dir_all(&dir).unwrap();
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    dir.join(format!("{name}-{unique}.png"))
}

struct PngRowDecodeProbeReset;

impl PngRowDecodeProbeReset {
    fn enable(row_delay: Duration) -> Self {
        PNG_ROW_DECODE_PROBE.enable(row_delay);
        Self
    }
}

impl Drop for PngRowDecodeProbeReset {
    fn drop(&mut self) {
        PNG_ROW_DECODE_PROBE.disable();
    }
}

fn box_shrink_expected_u8(
    pixels: &[u8],
    src_w: usize,
    src_h: usize,
    bands: usize,
    factor: usize,
) -> (u32, u32, Vec<u8>) {
    let dst_w = src_w.div_ceil(factor);
    let dst_h = src_h.div_ceil(factor);
    let mut output = vec![0u8; dst_w * dst_h * bands];

    for dy in 0..dst_h {
        let sy0 = dy * factor;
        let sy1 = (sy0 + factor).min(src_h);
        for dx in 0..dst_w {
            let sx0 = dx * factor;
            let sx1 = (sx0 + factor).min(src_w);
            let total = ((sy1 - sy0) * (sx1 - sx0)) as u32;

            for b in 0..bands {
                let mut sum = 0u32;
                for sy in sy0..sy1 {
                    for sx in sx0..sx1 {
                        let src = (sy * src_w + sx) * bands + b;
                        sum += u32::from(pixels[src]);
                    }
                }

                let dst = (dy * dst_w + dx) * bands + b;
                output[dst] = ((sum + total / 2) / total) as u8;
            }
        }
    }

    (dst_w as u32, dst_h as u32, output)
}

// ── sniff ─────────────────────────────────────────────────────────────────

#[test]
fn sniff_recognises_png_magic() {
    let codec = PngCodec::default();
    // Exact 8-byte PNG magic followed by arbitrary bytes.
    let header: &[u8] = &[137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 0];
    assert!(codec.sniff(header));
}

#[test]
fn sniff_rejects_jpeg() {
    let codec = PngCodec::default();
    let header: &[u8] = &[0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46];
    assert!(!codec.sniff(header));
}

#[test]
fn sniff_rejects_short_header() {
    let codec = PngCodec::default();
    assert!(!codec.sniff(&[137, 80, 78]));
}

// ── round-trip U8 RGB ─────────────────────────────────────────────────────

#[test]
fn round_trip_u8_rgb() {
    let codec = PngCodec::default();
    // 4×4 RGB image with a recognisable gradient pattern.
    let pixels: Vec<u8> = (0u8..48).collect();
    let original = Image::<U8>::from_buffer(4, 4, 3, pixels.clone()).unwrap();

    let encoded = codec.encode::<U8>(&original).unwrap();
    let decoded = codec.decode::<U8>(&encoded).unwrap();

    assert_eq!(decoded.width(), 4);
    assert_eq!(decoded.height(), 4);
    assert_eq!(decoded.bands(), 3);
    assert_eq!(decoded.pixels(), original.pixels());
}

// ── round-trip U8 grayscale ───────────────────────────────────────────────

#[test]
fn round_trip_u8_grayscale() {
    let codec = PngCodec::default();
    let pixels: Vec<u8> = (0u8..16).collect();
    let original = Image::<U8>::from_buffer(4, 4, 1, pixels).unwrap();

    let encoded = codec.encode::<U8>(&original).unwrap();
    let decoded = codec.decode::<U8>(&encoded).unwrap();

    assert_eq!(decoded.width(), 4);
    assert_eq!(decoded.height(), 4);
    assert_eq!(decoded.bands(), 1);
    assert_eq!(decoded.pixels(), original.pixels());
}

// ── round-trip U16 RGB ────────────────────────────────────────────────────

#[test]
fn round_trip_u16_rgb() {
    let codec = PngCodec::default();
    // 3×3 U16 RGB image.
    let pixels: Vec<u16> = (0u16..27).map(|v| v * 1000).collect();
    let original = Image::<U16>::from_buffer(3, 3, 3, pixels).unwrap();

    let encoded = codec.encode::<U16>(&original).unwrap();
    let decoded = codec.decode::<U16>(&encoded).unwrap();

    assert_eq!(decoded.width(), 3);
    assert_eq!(decoded.height(), 3);
    assert_eq!(decoded.bands(), 3);
    assert_eq!(decoded.pixels(), original.pixels());
}

#[cfg(feature = "libspng")]
#[test]
fn libspng_round_trip_u8_rgba() {
    let codec = PngCodec::default();
    let pixels: Vec<u8> = (0u8..128).cycle().take(4 * 8 * 4).collect();
    let original = Image::<U8>::from_buffer(4, 8, 4, pixels).unwrap();

    let encoded = codec.encode::<U8>(&original).unwrap();
    let decoded = decode_png_with_libspng::<U8>(&encoded).unwrap();

    assert_eq!(decoded.width(), original.width());
    assert_eq!(decoded.height(), original.height());
    assert_eq!(decoded.bands(), original.bands());
    assert_eq!(decoded.pixels(), original.pixels());
}

#[cfg(feature = "libspng")]
#[test]
fn libspng_decodes_sample_fixture_used_by_xtask_bench() {
    let fixture = std::fs::read("tests/fixtures/images/sample.png").unwrap();

    let direct = decode_png_with_libspng::<U16>(&fixture).unwrap();
    let eager = PngCodec::default().decode::<U16>(&fixture).unwrap();

    assert_eq!(direct.width(), eager.width());
    assert_eq!(direct.height(), eager.height());
    assert_eq!(direct.bands(), eager.bands());
    assert_eq!(direct.pixels(), eager.pixels());
}

#[cfg(feature = "libspng")]
#[test]
fn libspng_decodes_png_bench_fixture() {
    let fixture = std::fs::read("tests/fixtures/images/bench_2048x2048.png").unwrap();

    let direct = decode_png_with_libspng::<U8>(&fixture).unwrap();
    let eager = PngCodec::default().decode::<U8>(&fixture).unwrap();

    assert_eq!(direct.width(), eager.width());
    assert_eq!(direct.height(), eager.height());
    assert_eq!(direct.bands(), eager.bands());
    assert_eq!(direct.pixels(), eager.pixels());
}

// ── probe ─────────────────────────────────────────────────────────────────

#[test]
fn probe_returns_correct_dimensions() {
    let codec = PngCodec::default();
    // Encode a 7×5 RGBA image and probe it.
    let pixels: Vec<u8> = vec![128u8; 7 * 5 * 4];
    let image = Image::<U8>::from_buffer(7, 5, 4, pixels).unwrap();
    let encoded = codec.encode::<U8>(&image).unwrap();

    let (w, h, bands) = codec.probe(&encoded).unwrap();
    assert_eq!(w, 7);
    assert_eq!(h, 5);
    assert_eq!(bands, 4);
}

// ── format_name ───────────────────────────────────────────────────────────

#[test]
fn format_name_is_png_for_both_traits() {
    let codec = PngCodec::default();
    assert_eq!(<PngCodec as ImageDecoder>::format_name(&codec), "png");
    assert_eq!(<PngCodec as ImageEncoder>::format_name(&codec), "png");
}

// ── encode_with_options ───────────────────────────────────────────────────

#[test]
fn png_encoder_default_matches_libvips_filter_none() {
    assert_eq!(PngEncoder::default().filter, Filter::NoFilter);
}

#[test]
fn png_encoder_default_uses_fast_compression() {
    assert_eq!(PngEncoder::default().compression, 1);
}

#[test]
fn encode_with_options_compression_level_round_trip_preserves_pixels() {
    let codec = PngCodec::default();
    let pixels: Vec<u8> = (0..(8 * 8))
        .flat_map(|pixel_index| {
            let x = (pixel_index % 8) as u8;
            let y = (pixel_index / 8) as u8;
            [
                x.wrapping_mul(17).wrapping_add(y),
                y.wrapping_mul(29).wrapping_add(3),
                (x ^ y).wrapping_mul(11),
            ]
        })
        .collect();
    let image = Image::<U8>::from_buffer(8, 8, 3, pixels).unwrap();

    let opts = SaveOptions::default().with_compression_level(9);
    let encoded = codec.encode_with_options::<U8>(&image, &opts).unwrap();

    let decoded = codec.decode::<U8>(&encoded).unwrap();
    assert_eq!(decoded.width(), 8);
    assert_eq!(decoded.height(), 8);
    assert_eq!(decoded.bands(), 3);
    assert_u8_pixels_equal_with_index(
        image.pixels(),
        decoded.pixels(),
        "compression-level PNG round-trip must preserve pixels",
    );
}

#[test]
fn png_encoder_compression_zero_is_larger_than_nine() {
    let pixels: Vec<u8> = vec![7u8; 256 * 256 * 3];
    let image = Image::<U8>::from_buffer(256, 256, 3, pixels).unwrap();

    let uncompressed = PngEncoder {
        compression: 0,
        interlace: false,
        filter: Filter::Adaptive,
    }
    .encode(&image)
    .unwrap();
    let compressed = PngEncoder {
        compression: 9,
        interlace: false,
        filter: Filter::Adaptive,
    }
    .encode(&image)
    .unwrap();

    assert!(uncompressed.len() > compressed.len());
}

#[test]
fn png_encoder_interlace_round_trip_preserves_pixels() {
    let pixels: Vec<u8> = (0u8..192).cycle().take(16 * 16 * 3).collect();
    let image = Image::<U8>::from_buffer(16, 16, 3, pixels).unwrap();

    let encoded = PngEncoder {
        compression: 6,
        interlace: true,
        filter: Filter::Adaptive,
    }
    .encode(&image)
    .unwrap();

    let reader = RawDecoder::new(Cursor::new(&encoded)).read_info().unwrap();
    assert!(reader.info().interlaced);

    let decoded = PngCodec::default().decode::<U8>(&encoded).unwrap();
    assert_eq!(decoded.pixels(), image.pixels());
}

#[test]
fn png_round_trip_preserves_resolution_metadata() {
    let image = Image::<U8>::from_buffer(4, 4, 3, vec![64u8; 4 * 4 * 3])
        .unwrap()
        .with_metadata(ImageMetadata {
            xres: Some(12.0),
            yres: Some(8.0),
            interpretation: Some(Interpretation::Srgb),
            ..ImageMetadata::default()
        });

    let encoded = PngEncoder::default().encode(&image).unwrap();

    let reader = RawDecoder::new(Cursor::new(&encoded)).read_info().unwrap();
    let pixel_dims = reader.info().pixel_dims.expect("pHYs chunk must exist");
    assert_eq!(pixel_dims.xppu, 12_000);
    assert_eq!(pixel_dims.yppu, 8_000);
    assert_eq!(pixel_dims.unit, RawUnit::Meter);

    let decoded = PngCodec::default().decode::<U8>(&encoded).unwrap();
    assert_eq!(decoded.metadata().xres, Some(12.0));
    assert_eq!(decoded.metadata().yres, Some(8.0));
    assert_eq!(
        decoded.metadata().interpretation,
        Some(Interpretation::Srgb)
    );
}

#[test]
fn encode_with_options_interlace_sets_adam7() {
    let image = Image::<U8>::from_buffer(8, 8, 3, vec![9u8; 8 * 8 * 3]).unwrap();
    let encoded = PngCodec::default()
        .encode_with_options::<U8>(&image, &SaveOptions::default().with_interlace(true))
        .unwrap();

    let reader = RawDecoder::new(Cursor::new(&encoded)).read_info().unwrap();
    assert!(reader.info().interlaced);
}

#[test]
fn encode_with_options_png_filter_changes_encoded_stream() {
    let pixels: Vec<u8> = (0..16)
        .flat_map(|row| (0..16).flat_map(move |col| [row as u8, col as u8, (row * col) as u8]))
        .collect();
    let image = Image::<U8>::from_buffer(16, 16, 3, pixels).unwrap();

    let no_filter = PngCodec::default()
        .encode_with_options::<U8>(
            &image,
            &SaveOptions::default().with_png_filter(PngFilterStrategy::None),
        )
        .unwrap();
    let paeth = PngCodec::default()
        .encode_with_options::<U8>(
            &image,
            &SaveOptions::default().with_png_filter(PngFilterStrategy::Paeth),
        )
        .unwrap();

    assert_ne!(no_filter, paeth);
    assert_eq!(
        PngCodec::default()
            .decode::<U8>(&no_filter)
            .unwrap()
            .pixels(),
        image.pixels()
    );
    assert_eq!(
        PngCodec::default().decode::<U8>(&paeth).unwrap().pixels(),
        image.pixels()
    );
}

#[test]
fn encode_to_writer_matches_encode_with_options() {
    let pixels: Vec<u8> = (0u8..96).cycle().take(8 * 4 * 3).collect();
    let image = Image::<U8>::from_buffer(8, 4, 3, pixels).unwrap();
    let opts = SaveOptions::default()
        .with_compression_level(6)
        .with_png_filter(PngFilterStrategy::Paeth);

    let expected = PngCodec::default()
        .encode_with_options::<U8>(&image, &opts)
        .unwrap();
    let mut streamed = Vec::new();

    PngCodec::default()
        .encode_to_writer::<U8>(&image, &opts, &mut streamed)
        .unwrap();

    assert_eq!(streamed, expected);
}

#[test]
fn png_metadata_round_trip_preserves_color_type_mapping() {
    let cases = [
        (
            Image::<U8>::from_buffer(3, 2, 1, vec![5u8; 3 * 2])
                .unwrap()
                .with_metadata(ImageMetadata {
                    interpretation: Some(Interpretation::BW),
                    ..ImageMetadata::default()
                }),
            RawColorType::Grayscale,
            Interpretation::BW,
        ),
        (
            Image::<U8>::from_buffer(3, 2, 3, vec![9u8; 3 * 2 * 3])
                .unwrap()
                .with_metadata(ImageMetadata {
                    interpretation: Some(Interpretation::Srgb),
                    ..ImageMetadata::default()
                }),
            RawColorType::Rgb,
            Interpretation::Srgb,
        ),
    ];

    for (image, expected_color_type, expected_interpretation) in cases {
        let encoded = PngEncoder::default().encode(&image).unwrap();
        let reader = RawDecoder::new(Cursor::new(&encoded)).read_info().unwrap();
        assert_eq!(reader.info().color_type, expected_color_type);

        let decoded = PngCodec::default().decode::<U8>(&encoded).unwrap();
        assert_eq!(
            decoded.metadata().interpretation,
            Some(expected_interpretation)
        );
    }
}

#[test]
fn png_metadata_round_trip_preserves_icc_exif_and_xmp() {
    let image = Image::<U8>::from_buffer(2, 2, 3, vec![32u8; 12])
        .unwrap()
        .with_metadata(ImageMetadata {
            icc_profile: Some((0u8..32).collect()),
            exif: Some(vec![b'I', b'I', 42, 0, 8, 0, 0, 0]),
            xmp: Some(br#"<x:xmpmeta><rdf:RDF/></x:xmpmeta>"#.to_vec()),
            ..ImageMetadata::default()
        });

    let encoded = PngCodec::default().encode::<U8>(&image).unwrap();
    let decoded = PngCodec::default().decode::<U8>(&encoded).unwrap();

    assert_eq!(decoded.metadata().icc_profile, image.metadata().icc_profile);
    assert_eq!(decoded.metadata().exif, image.metadata().exif);
    assert_eq!(decoded.metadata().xmp, image.metadata().xmp);
}

#[test]
fn strip_metadata_removes_png_icc_exif_and_xmp() {
    let image = Image::<U8>::from_buffer(2, 2, 3, vec![48u8; 12])
        .unwrap()
        .with_metadata(ImageMetadata {
            icc_profile: Some((0u8..24).collect()),
            exif: Some(vec![b'M', b'M', 0, 42, 0, 0, 0, 8]),
            xmp: Some(br#"<x:xmpmeta>strip-me</x:xmpmeta>"#.to_vec()),
            ..ImageMetadata::default()
        });

    let encoded = PngCodec::default()
        .encode_with_options::<U8>(&image, &SaveOptions::default().strip_metadata())
        .unwrap();
    let decoded = PngCodec::default().decode::<U8>(&encoded).unwrap();

    assert!(decoded.metadata().icc_profile.is_none());
    assert!(decoded.metadata().exif.is_none());
    assert!(decoded.metadata().xmp.is_none());
}

// ── decode_with_options delegates ────────────────────────────────────────

#[test]
fn decode_with_options_delegates_to_decode() {
    let codec = PngCodec::default();
    let pixels: Vec<u8> = vec![0u8; 4 * 4];
    let image = Image::<U8>::from_buffer(4, 4, 1, pixels).unwrap();
    let encoded = codec.encode::<U8>(&image).unwrap();

    let opts = LoadOptions::default();
    let decoded = codec.decode_with_options::<U8>(&encoded, &opts).unwrap();
    assert_eq!(decoded.pixels(), image.pixels());
}

#[test]
fn decode_path_with_options_matches_in_memory_decode() {
    let codec = PngCodec::default();
    let pixels: Vec<u8> = (0u8..96).cycle().take(8 * 4 * 3).collect();
    let image = Image::<U8>::from_buffer(8, 4, 3, pixels).unwrap();
    let encoded = codec.encode::<U8>(&image).unwrap();
    let path = png_test_output_path("decode-path");
    std::fs::write(&path, &encoded).unwrap();

    let from_memory = codec.decode::<U8>(&encoded).unwrap();
    let from_path = codec
        .decode_path_with_options::<U8>(&path, &LoadOptions::default())
        .unwrap();

    assert_eq!(from_path.width(), from_memory.width());
    assert_eq!(from_path.height(), from_memory.height());
    assert_eq!(from_path.bands(), from_memory.bands());
    assert_eq!(from_path.metadata(), from_memory.metadata());
    assert_eq!(from_path.pixels(), from_memory.pixels());
}

#[test]
fn decode_path_with_shrink_keeps_partial_tail_blocks() {
    let codec = PngCodec::default();
    let width = 17usize;
    let height = 18usize;
    let factor = 16usize;
    let pixels: Vec<u8> = (0..(width * height))
        .flat_map(|index| {
            let x = (index % width) as u8;
            let y = (index / width) as u8;
            [
                x.wrapping_mul(13).wrapping_add(y),
                y.wrapping_mul(7).wrapping_add(x.wrapping_mul(3)),
                x ^ y.wrapping_mul(11),
            ]
        })
        .collect();
    let image = Image::<U8>::from_buffer(width as u32, height as u32, 3, pixels.clone()).unwrap();
    let encoded = codec.encode::<U8>(&image).unwrap();
    let path = png_test_output_path("decode-path-shrink-tail");
    std::fs::write(&path, &encoded).unwrap();

    let decoded = codec
        .decode_path_with_options::<U8>(
            &path,
            &LoadOptions::default().with_shrink(NonZeroU8::new(factor as u8).unwrap()),
        )
        .unwrap();
    let (expected_w, expected_h, expected_pixels) =
        box_shrink_expected_u8(&pixels, width, height, 3, factor);

    assert_eq!(decoded.width(), expected_w);
    assert_eq!(decoded.height(), expected_h);
    assert_eq!(decoded.bands(), 3);
    assert_eq!(decoded.pixels(), expected_pixels);
}

#[test]
fn decode_path_with_shrink_preserves_grayscale_alpha_pairs() {
    let codec = PngCodec::default();
    let width = 4usize;
    let height = 4usize;
    let factor = 2usize;
    let pixels: Vec<u8> = (0..(width * height))
        .flat_map(|index| {
            let x = (index % width) as u8;
            let y = (index / width) as u8;
            [
                x.wrapping_mul(17).wrapping_add(y.wrapping_mul(9)),
                255u8.wrapping_sub(x.wrapping_mul(21).wrapping_add(y.wrapping_mul(5))),
            ]
        })
        .collect();
    let image = Image::<U8>::from_buffer(width as u32, height as u32, 2, pixels.clone()).unwrap();
    let encoded = codec.encode::<U8>(&image).unwrap();
    let path = png_test_output_path("decode-path-shrink-ga");
    std::fs::write(&path, &encoded).unwrap();

    let decoded = codec
        .decode_path_with_options::<U8>(
            &path,
            &LoadOptions::default().with_shrink(NonZeroU8::new(factor as u8).unwrap()),
        )
        .unwrap();
    let (expected_w, expected_h, expected_pixels) =
        box_shrink_expected_u8(&pixels, width, height, 2, factor);

    assert_eq!(decoded.width(), expected_w);
    assert_eq!(decoded.height(), expected_h);
    assert_eq!(decoded.bands(), 2);
    assert_eq!(decoded.pixels(), expected_pixels);
}

#[test]
fn decode_region_from_path_matches_in_memory_decode() {
    let codec = PngCodec::default();
    let pixels: Vec<u8> = (0u8..90).cycle().take(6 * 5 * 3).collect();
    let image = Image::<U8>::from_buffer(6, 5, 3, pixels).unwrap();
    let encoded = codec.encode::<U8>(&image).unwrap();
    let path = png_test_output_path("decode-region-path");
    std::fs::write(&path, &encoded).unwrap();
    let region = Region::new(1, 2, 3, 2);
    let mut from_memory = vec![0u8; region.pixel_count() * 3];
    let mut from_path = vec![0u8; region.pixel_count() * 3];

    codec
        .decode_region_into::<U8>(&encoded, &LoadOptions::default(), region, &mut from_memory)
        .unwrap();
    codec
        .decode_region_from_path::<U8>(&path, &LoadOptions::default(), region, &mut from_path)
        .unwrap();

    assert_eq!(from_path, from_memory);
}

#[test]
fn decode_region_from_path_streams_sequential_full_width_strips() {
    let codec = PngCodec::default();
    let pixels: Vec<u8> = (0u8..144).cycle().take(6 * 8 * 3).collect();
    let image = Image::<U8>::from_buffer(6, 8, 3, pixels).unwrap();
    let encoded = codec.encode::<U8>(&image).unwrap();
    let eager = codec.decode::<U8>(&encoded).unwrap();
    let path = png_test_output_path("decode-region-path-sequential");
    std::fs::write(&path, &encoded).unwrap();

    let top = Region::new(0, 0, 6, 3);
    let mut top_output = vec![0u8; top.pixel_count() * 3];
    codec
        .decode_region_from_path::<U8>(&path, &LoadOptions::default(), top, &mut top_output)
        .unwrap();
    assert_eq!(top_output, clamped_region_pixels_u8(&eager, top));

    let bottom = Region::new(0, 3, 6, 5);
    let mut bottom_output = vec![0u8; bottom.pixel_count() * 3];
    codec
        .decode_region_from_path::<U8>(&path, &LoadOptions::default(), bottom, &mut bottom_output)
        .unwrap();
    assert_eq!(bottom_output, clamped_region_pixels_u8(&eager, bottom));
}

#[test]
fn decode_region_from_path_does_not_hold_session_mutex_across_row_decode() {
    let _guard = PROBE_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let codec = Arc::new(PngCodec::default());
    // Use a larger image (256×256) so row decode takes long enough for two threads
    // to overlap, even on slow CI runners with limited parallelism.
    let pixels: Vec<u8> = (0u8..=255).cycle().take(256 * 256 * 3).collect();
    let image = Image::<U8>::from_buffer(256, 256, 3, pixels).unwrap();
    let encoded = codec.encode::<U8>(&image).unwrap();
    let eager = codec.decode::<U8>(&encoded).unwrap();
    let path = png_test_output_path("decode-region-path-concurrent-session");
    std::fs::write(&path, &encoded).unwrap();

    let top = Region::new(0, 0, 256, 64);
    let mut top_output = vec![0u8; top.pixel_count() * 3];
    codec
        .decode_region_from_path::<U8>(&path, &LoadOptions::default(), top, &mut top_output)
        .unwrap();
    assert_eq!(top_output, clamped_region_pixels_u8(&eager, top));

    let sequential = Region::new(0, 64, 256, 64);
    let partial = Region::new(32, 32, 128, 96);
    let expected_sequential = clamped_region_pixels_u8(&eager, sequential);
    let expected_partial = clamped_region_pixels_u8(&eager, partial);
    // 10ms per-row delay ensures each decode takes ~640ms (64 rows × 10ms),
    // giving ample time for both threads to be active simultaneously.
    let _probe = PngRowDecodeProbeReset::enable(Duration::from_millis(10));
    let start = Arc::new(Barrier::new(3));

    let sequential_thread = {
        let codec = Arc::clone(&codec);
        let path = path.clone();
        let start = Arc::clone(&start);
        std::thread::spawn(move || {
            let mut output = vec![0u8; sequential.pixel_count() * 3];
            start.wait();
            codec
                .decode_region_from_path::<U8>(
                    &path,
                    &LoadOptions::default(),
                    sequential,
                    &mut output,
                )
                .map(|()| output)
        })
    };
    let partial_thread = {
        let codec = Arc::clone(&codec);
        let path = path.clone();
        let start = Arc::clone(&start);
        std::thread::spawn(move || {
            let mut output = vec![0u8; partial.pixel_count() * 3];
            start.wait();
            codec
                .decode_region_from_path::<U8>(&path, &LoadOptions::default(), partial, &mut output)
                .map(|()| output)
        })
    };

    start.wait();

    let sequential_output = sequential_thread.join().unwrap().unwrap();
    let partial_output = partial_thread.join().unwrap().unwrap();

    assert_eq!(sequential_output, expected_sequential);
    assert_eq!(partial_output, expected_partial);
    assert!(
        PNG_ROW_DECODE_PROBE.max_active() >= 2,
        "expected concurrent path decode rows without session-mutex serialization, saw max concurrency {}",
        PNG_ROW_DECODE_PROBE.max_active()
    );
}

#[test]
fn tile_decoder_streams_u8_regions_out_of_order_without_resident_frame() {
    let codec = PngCodec::default();
    let pixels: Vec<u8> = (0u8..60).collect();
    let image = Image::<U8>::from_buffer(5, 4, 3, pixels.clone()).unwrap();
    let encoded = codec.encode::<U8>(&image).unwrap();
    let source =
        DecoderSource::<_, U8>::streaming(PngCodec::default(), &encoded, LoadOptions::default())
            .unwrap();

    assert!(source.is_streaming());
    assert_eq!(source.resident_decoded_bytes(), 0);
    assert!(source.image().is_none());

    let lower = Region::new(2, 2, 2, 2);
    let mut lower_output = vec![0u8; lower.pixel_count() * 3];
    source.read_region(lower, &mut lower_output).unwrap();
    assert_eq!(
        lower_output,
        vec![36, 37, 38, 39, 40, 41, 51, 52, 53, 54, 55, 56]
    );

    let upper = Region::new(0, 0, 3, 1);
    let mut upper_output = vec![0u8; upper.pixel_count() * 3];
    source.read_region(upper, &mut upper_output).unwrap();
    assert_eq!(upper_output, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
}

#[test]
fn tile_decoder_clamps_region_edges() {
    let codec = PngCodec::default();
    let pixels: Vec<u8> = (0u8..16).collect();
    let image = Image::<U8>::from_buffer(4, 4, 1, pixels).unwrap();
    let encoded = codec.encode::<U8>(&image).unwrap();
    let source =
        DecoderSource::<_, U8>::streaming(PngCodec::default(), &encoded, LoadOptions::default())
            .unwrap();

    let region = Region::new(-1, -1, 3, 3);
    let mut output = vec![0u8; region.pixel_count()];
    source.read_region(region, &mut output).unwrap();

    assert_eq!(output, vec![0, 0, 1, 0, 0, 1, 4, 4, 5]);
}

#[test]
fn tile_decoder_streams_u16_region_without_full_frame() {
    let codec = PngCodec::default();
    let pixels: Vec<u16> = (0u16..27).map(|sample| sample * 257).collect();
    let image = Image::<U16>::from_buffer(3, 3, 3, pixels).unwrap();
    let encoded = codec.encode::<U16>(&image).unwrap();
    let source =
        DecoderSource::<_, U16>::streaming(PngCodec::default(), &encoded, LoadOptions::default())
            .unwrap();

    let region = Region::new(1, 1, 2, 1);
    let mut output = vec![0u8; region.pixel_count() * 3 * std::mem::size_of::<u16>()];
    source.read_region(region, &mut output).unwrap();
    let samples: &[u16] = bytemuck::try_cast_slice(&output).unwrap();

    assert_eq!(samples, &[3084, 3341, 3598, 3855, 4112, 4369]);
    assert_eq!(source.resident_decoded_bytes(), 0);
}

#[test]
fn decode_region_into_interlaced_png_matches_eager_decode() {
    let pixels: Vec<u8> = (0u8..192).cycle().take(8 * 8 * 3).collect();
    let image = Image::<U8>::from_buffer(8, 8, 3, pixels).unwrap();
    let encoded = PngEncoder {
        compression: 6,
        interlace: true,
        filter: Filter::Adaptive,
    }
    .encode(&image)
    .unwrap();
    let eager = PngCodec::default().decode::<U8>(&encoded).unwrap();
    let region = Region::new(1, 2, 4, 3);
    let mut output = vec![0u8; region.pixel_count() * 3];

    PngCodec::default()
        .decode_region_into::<U8>(&encoded, &LoadOptions::default(), region, &mut output)
        .unwrap();

    assert_eq!(output, clamped_region_pixels_u8(&eager, region));
}

#[test]
fn decode_region_from_path_interlaced_png_reuses_eager_backing() {
    let _guard = PROBE_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let codec = PngCodec::default();
    let pixels: Vec<u8> = (0u8..=255).cycle().take(256 * 256 * 3).collect();
    let image = Image::<U8>::from_buffer(256, 256, 3, pixels).unwrap();
    let encoded = PngEncoder {
        compression: 6,
        interlace: true,
        filter: Filter::Adaptive,
    }
    .encode(&image)
    .unwrap();
    let path = png_test_output_path("decode-region-path-interlaced-cache");
    std::fs::write(&path, &encoded).unwrap();
    let eager = codec.decode::<U8>(&encoded).unwrap();
    let top = Region::new(3, 4, 5, 6);
    let bottom = Region::new(200, 210, 7, 8);
    let mut top_output = vec![0u8; top.pixel_count() * 3];
    let mut bottom_output = vec![0u8; bottom.pixel_count() * 3];
    let _probe = PngRowDecodeProbeReset::enable(Duration::ZERO);

    codec
        .decode_region_from_path::<U8>(&path, &LoadOptions::default(), top, &mut top_output)
        .unwrap();
    let full_decodes_after_first = PNG_ROW_DECODE_PROBE.full_raster_decodes();
    let rows_after_first = PNG_ROW_DECODE_PROBE.total_rows();

    codec
        .decode_region_from_path::<U8>(&path, &LoadOptions::default(), bottom, &mut bottom_output)
        .unwrap();
    let full_decodes_after_second = PNG_ROW_DECODE_PROBE.full_raster_decodes();
    let rows_after_second = PNG_ROW_DECODE_PROBE.total_rows();

    assert_eq!(top_output, clamped_region_pixels_u8(&eager, top));
    assert_eq!(bottom_output, clamped_region_pixels_u8(&eager, bottom));
    assert_eq!(
        rows_after_first, 0,
        "interlaced path decode should materialize eager backing instead of Adam7 row iteration"
    );
    assert_eq!(
        rows_after_second, 0,
        "subsequent interlaced tile reads should reuse the eager backing"
    );
    assert_eq!(
        full_decodes_after_first, 1,
        "first interlaced tile should materialize exactly one eager backing"
    );
    assert_eq!(
        full_decodes_after_second, 1,
        "subsequent interlaced tiles should reuse the cached eager backing"
    );
}

#[test]
fn decode_region_into_allows_parallel_tile_reads_on_same_codec() {
    let _guard = PROBE_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let codec = Arc::new(PngCodec::default());
    let pixels: Vec<u8> = (0u8..=255).cycle().take(64 * 64 * 3).collect();
    let image = Image::<U8>::from_buffer(64, 64, 3, pixels).unwrap();
    let encoded = Arc::new(codec.encode::<U8>(&image).unwrap());
    let left = Region::new(0, 0, 32, 64);
    let right = Region::new(32, 0, 32, 64);
    let expected_left = clamped_region_pixels_u8(&image, left);
    let expected_right = clamped_region_pixels_u8(&image, right);
    let _probe = PngRowDecodeProbeReset::enable(Duration::from_millis(1));
    let start = Arc::new(Barrier::new(3));

    let left_thread = {
        let codec = Arc::clone(&codec);
        let encoded = Arc::clone(&encoded);
        let start = Arc::clone(&start);
        std::thread::spawn(move || {
            let mut output = vec![0u8; left.pixel_count() * 3];
            start.wait();
            codec
                .decode_region_into::<U8>(&encoded, &LoadOptions::default(), left, &mut output)
                .map(|()| output)
        })
    };
    let right_thread = {
        let codec = Arc::clone(&codec);
        let encoded = Arc::clone(&encoded);
        let start = Arc::clone(&start);
        std::thread::spawn(move || {
            let mut output = vec![0u8; right.pixel_count() * 3];
            start.wait();
            codec
                .decode_region_into::<U8>(&encoded, &LoadOptions::default(), right, &mut output)
                .map(|()| output)
        })
    };

    start.wait();

    let left_output = left_thread.join().unwrap().unwrap();
    let right_output = right_thread.join().unwrap().unwrap();

    assert_eq!(left_output, expected_left);
    assert_eq!(right_output, expected_right);
    assert!(
        PNG_ROW_DECODE_PROBE.max_active() >= 2,
        "expected parallel row decode loops, saw max concurrency {}",
        PNG_ROW_DECODE_PROBE.max_active()
    );
}

#[test]
fn decode_region_into_returns_image_too_large_for_overflowing_region() {
    let image = Image::<U8>::from_buffer(1, 1, 3, vec![1, 2, 3]).unwrap();
    let encoded = PngCodec::default().encode::<U8>(&image).unwrap();
    let region = Region::new(0, 0, u32::MAX, u32::MAX);
    let mut output = Vec::new();

    let result = PngCodec::default().decode_region_into::<U8>(
        &encoded,
        &LoadOptions::default(),
        region,
        &mut output,
    );

    assert!(matches!(
        result,
        Err(ViprsError::ImageTooLarge {
            width,
            height,
            bands,
            ..
        }) if width == u32::MAX && height == u32::MAX && bands == 3
    ));
}

#[test]
fn tile_decoder_streams_interlaced_png_region_matches_eager_decode() {
    let pixels: Vec<u8> = (0u8..192).cycle().take(8 * 8 * 3).collect();
    let image = Image::<U8>::from_buffer(8, 8, 3, pixels).unwrap();
    let encoded = PngEncoder {
        compression: 6,
        interlace: true,
        filter: Filter::Adaptive,
    }
    .encode(&image)
    .unwrap();
    let eager = PngCodec::default().decode::<U8>(&encoded).unwrap();
    let source =
        DecoderSource::<_, U8>::streaming(PngCodec::default(), &encoded, LoadOptions::default())
            .unwrap();

    let lower = Region::new(2, 3, 3, 2);
    let mut lower_output = vec![0u8; lower.pixel_count() * 3];
    source.read_region(lower, &mut lower_output).unwrap();
    assert_eq!(lower_output, clamped_region_pixels_u8(&eager, lower));

    let edge = Region::new(-1, 6, 4, 3);
    let mut edge_output = vec![0u8; edge.pixel_count() * 3];
    source.read_region(edge, &mut edge_output).unwrap();
    assert_eq!(edge_output, clamped_region_pixels_u8(&eager, edge));
}

// ── unsupported format error ──────────────────────────────────────────────

#[test]
fn encode_unsupported_format_returns_error() {
    use crate::domain::format::F32;

    let codec = PngCodec::default();
    let pixels: Vec<f32> = vec![0.5f32; 4 * 4 * 3];
    let image = Image::<F32>::from_buffer(4, 4, 3, pixels).unwrap();
    let result = codec.encode::<F32>(&image);
    assert!(result.is_err());
    if let Err(ViprsError::Codec(msg)) = result {
        assert!(
            msg.contains("unsupported format"),
            "unexpected error message: {msg}"
        );
    }
}

// ── edge cases ────────────────────────────────────────────────────────────

#[test]
fn decode_empty_slice_returns_codec_error() {
    let codec = PngCodec::default();
    let result = codec.decode::<U8>(&[]);
    assert!(
        matches!(result, Err(ViprsError::Codec(_))),
        "empty input must return ViprsError::Codec, got: {result:?}"
    );
}

#[test]
fn round_trip_u8_rgba_4x4() {
    let codec = PngCodec::default();
    // 4×4 RGBA image — PNG is lossless so pixels must be identical after round-trip.
    let pixels: Vec<u8> = (0u8..64).collect(); // 4*4*4
    let original = Image::<U8>::from_buffer(4, 4, 4, pixels).unwrap();

    let encoded = codec.encode::<U8>(&original).unwrap();
    let decoded = codec.decode::<U8>(&encoded).unwrap();

    assert_eq!(decoded.width(), 4);
    assert_eq!(decoded.height(), 4);
    assert_eq!(decoded.bands(), 4);
    assert_eq!(
        decoded.pixels(),
        original.pixels(),
        "PNG lossless round-trip must be pixel-perfect for RGBA"
    );
}

#[test]
fn round_trip_u16_rgb_4x4() {
    let codec = PngCodec::default();
    // 4×4 U16 RGB image — PNG is lossless so pixels must be identical.
    let pixels: Vec<u16> = (0u16..48).map(|v| v * 1000).collect(); // 4*4*3
    let original = Image::<U16>::from_buffer(4, 4, 3, pixels).unwrap();

    let encoded = codec.encode::<U16>(&original).unwrap();
    let decoded = codec.decode::<U16>(&encoded).unwrap();

    assert_eq!(decoded.width(), 4);
    assert_eq!(decoded.height(), 4);
    assert_eq!(decoded.bands(), 3);
    assert_eq!(
        decoded.pixels(),
        original.pixels(),
        "PNG lossless round-trip must be pixel-perfect for U16 RGB"
    );
}

#[test]
fn audit_reference_sample_fixture_decode_matches_imagemagick() {
    let fixture = std::fs::read("tests/fixtures/images/sample.png").unwrap();
    let decoded = PngCodec::default().decode::<U16>(&fixture).unwrap();

    assert_eq!(
        (decoded.width(), decoded.height(), decoded.bands()),
        (290, 442, 3)
    );
    assert_eq!(
        decoded.metadata().interpretation,
        Some(Interpretation::Rgb16)
    );
    let samples = [
        ((0u32, 0u32), [49210u16, 41207, 29445]),
        ((1, 0), [49172, 41415, 29447]),
        ((10, 10), [38671, 33914, 26762]),
        ((145, 221), [16382, 19308, 28635]),
        ((289, 441), [28156, 23698, 15306]),
        ((50, 300), [40653, 35027, 24704]),
        ((200, 100), [16566, 20310, 30412]),
    ];
    for ((x, y), expected) in samples {
        let offset = ((y * decoded.width() + x) * decoded.bands()) as usize;
        assert_eq!(&decoded.pixels()[offset..offset + 3], expected.as_slice());
    }
}

#[test]
fn audit_reference_rgba_fixture_decode_matches_imagemagick() {
    let fixture = std::fs::read("tests/fixtures/images/rgba.png").unwrap();
    let decoded = PngCodec::default().decode::<U8>(&fixture).unwrap();

    assert_eq!(
        (decoded.width(), decoded.height(), decoded.bands()),
        (128, 128, 4)
    );
    assert_eq!(
        decoded.metadata().interpretation,
        Some(Interpretation::Srgb)
    );
    assert_eq!(crc32(decoded.pixels()), 0x0935_AB1F);
}

#[test]
fn audit_reference_grayscale_fixture_decode_matches_imagemagick() {
    let fixture = std::fs::read("tests/fixtures/images/bench_512x512_gray.png").unwrap();
    let decoded = PngCodec::default().decode::<U8>(&fixture).unwrap();

    assert_eq!(
        (decoded.width(), decoded.height(), decoded.bands()),
        (512, 512, 1)
    );
    assert_eq!(decoded.metadata().interpretation, Some(Interpretation::BW));
    assert_eq!(crc32(decoded.pixels()), 0x2A63_CBBF);
}

#[test]
fn audit_reference_large_fixture_decode_matches_imagemagick() {
    let fixture = std::fs::read("tests/fixtures/images/bench_8192x8192.png").unwrap();
    let decoded = PngCodec::default().decode::<U8>(&fixture).unwrap();

    assert_eq!(
        (decoded.width(), decoded.height(), decoded.bands()),
        (8192, 8192, 3)
    );
    assert_eq!(
        decoded.metadata().interpretation,
        Some(Interpretation::Srgb)
    );
    assert_eq!(crc32(decoded.pixels()), 0x1950_EA2D);
}

#[test]
fn audit_roundtrip_tiny_2x2_rgba_is_exact() {
    let original = Image::<U8>::from_buffer(
        2,
        2,
        4,
        vec![1, 2, 3, 255, 4, 5, 6, 128, 7, 8, 9, 64, 10, 11, 12, 0],
    )
    .unwrap();
    let codec = PngCodec::default();
    let encoded = codec.encode(&original).unwrap();
    let decoded = codec.decode::<U8>(&encoded).unwrap();

    assert_eq!(
        (decoded.width(), decoded.height(), decoded.bands()),
        (2, 2, 4)
    );
    assert_eq!(decoded.pixels(), original.pixels());
}

#[test]
fn encode_f32_returns_codec_error() {
    use crate::domain::format::F32;

    let codec = PngCodec::default();
    // F32 is not supported by the PNG codec.
    let pixels: Vec<f32> = vec![0.0f32; 4 * 4 * 3];
    let image = Image::<F32>::from_buffer(4, 4, 3, pixels).unwrap();
    let result = codec.encode::<F32>(&image);
    assert!(
        matches!(result, Err(ViprsError::Codec(_))),
        "F32 encode must return ViprsError::Codec, got: {result:?}"
    );
}

#[test]
fn sniff_with_less_than_8_bytes_returns_false() {
    let codec = PngCodec::default();
    // PNG magic is exactly 8 bytes; shorter slices must not panic.
    assert!(!codec.sniff(&[]), "empty slice must return false");
    assert!(
        !codec.sniff(&[137, 80, 78]),
        "3-byte slice must return false"
    );
    assert!(
        !codec.sniff(&[137, 80, 78, 71, 13, 10, 26]),
        "7-byte slice must return false"
    );
}

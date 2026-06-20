use super::super::shrink_on_load::ShrinkOnLoadBackend;
use super::common::{
    EXIF_SIGNATURE, ICC_PROFILE_SIGNATURE, insert_segment_after_soi, jpeg_shrink_on_load_plan,
    normalize_xmp_app1_payload, visit_jpeg_segments,
};
use super::{JpegCodec, apply_exif_orientation};
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::Command;

use viprs_core::codec_options::{JpegSubsampling, LoadOptions, SaveOptions};
use viprs_core::error::ViprsError;
use viprs_core::format::U8;
use viprs_core::image::{Image, ImageMetadata, Interpretation};
#[cfg(feature = "icc")]
use viprs_ops_colour::colour::profile_load;
use viprs_ports::codec::{ImageDecoder, ImageEncoder};

const SAMPLE_JPEG: &[u8] = include_bytes!("../../../../tests/fixtures/images/sample.jpg");
const BENCH_8192_JPEG: &[u8] =
    include_bytes!("../../../../tests/fixtures/images/bench_8192x8192.jpg");
const BENCH_2048_JPEG: &[u8] =
    include_bytes!("../../../../tests/fixtures/images/bench_2048x2048.jpg");
const BENCH_8192_JPEG_RGB_CRC32: u32 = 0x9FDD_3181;
const METRICS_PREFIX: &str = "VIPRS_JPEG_SHRINK_METRICS";
const METRICS_FACTOR_ENV: &str = "VIPRS_JPEG_SHRINK_METRICS_FACTOR";
const METRICS_CHILD_TEST: &str = "jpeg::tests::shrink_on_load_decode_metrics_child";
const RSS_PREFIX: &str = "VIPRS_JPEG_ERROR_RSS";
const RSS_CHILD_ENV: &str = "VIPRS_JPEG_ERROR_RSS_CHILD";
const RSS_CHILD_TEST: &str =
    "jpeg::tests::repeated_truncated_decode_errors_keep_rss_stable_child";
const RSS_STABILITY_THRESHOLD_KB: usize = 4 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DecodeMetrics {
    factor: u8,
    width: u32,
    height: u32,
    pixels: usize,
    alloc_count: u64,
    alloc_bytes: u64,
    peak_live_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RssMetrics {
    warm_rss_kb: usize,
    final_rss_kb: usize,
    delta_kb: usize,
}

fn run_decode_metrics_child(factor: u8) -> DecodeMetrics {
    let output = Command::new(std::env::current_exe().unwrap())
        .env(METRICS_FACTOR_ENV, factor.to_string())
        .arg("--exact")
        .arg(METRICS_CHILD_TEST)
        .arg("--nocapture")
        .arg("--test-threads=1")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "child metrics run failed for factor {factor}: stdout=\n{}\nstderr=\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let combined_output = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let metrics_line = combined_output
        .lines()
        .find_map(|line| line.split_once(METRICS_PREFIX).map(|(_, metrics)| metrics))
        .unwrap_or_else(|| panic!("missing metrics line for factor {factor}: {combined_output}"));
    parse_metrics_line(metrics_line.trim())
}

fn parse_metrics_line(line: &str) -> DecodeMetrics {
    let mut metrics = DecodeMetrics {
        factor: 0,
        width: 0,
        height: 0,
        pixels: 0,
        alloc_count: 0,
        alloc_bytes: 0,
        peak_live_bytes: 0,
    };

    for field in line.split_whitespace() {
        let (key, value) = field
            .split_once('=')
            .unwrap_or_else(|| panic!("invalid metrics field: {field}"));
        match key {
            "factor" => metrics.factor = value.parse().unwrap(),
            "width" => metrics.width = value.parse().unwrap(),
            "height" => metrics.height = value.parse().unwrap(),
            "pixels" => metrics.pixels = value.parse().unwrap(),
            "alloc_count" => metrics.alloc_count = value.parse().unwrap(),
            "alloc_bytes" => metrics.alloc_bytes = value.parse().unwrap(),
            "peak_live_bytes" => metrics.peak_live_bytes = value.parse().unwrap(),
            other => panic!("unexpected metrics key: {other}"),
        }
    }

    metrics
}

fn run_rss_child() -> RssMetrics {
    let output = Command::new(std::env::current_exe().unwrap())
        .env(RSS_CHILD_ENV, "1")
        .arg("--exact")
        .arg(RSS_CHILD_TEST)
        .arg("--nocapture")
        .arg("--test-threads=1")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "RSS child run failed: stdout=\n{}\nstderr=\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let combined_output = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let metrics_line = combined_output
        .lines()
        .find_map(|line| line.split_once(RSS_PREFIX).map(|(_, metrics)| metrics))
        .unwrap_or_else(|| panic!("missing RSS metrics line: {combined_output}"));
    parse_rss_metrics_line(metrics_line.trim())
}

fn parse_rss_metrics_line(line: &str) -> RssMetrics {
    let mut metrics = RssMetrics {
        warm_rss_kb: 0,
        final_rss_kb: 0,
        delta_kb: 0,
    };

    for field in line.split_whitespace() {
        let (key, value) = field
            .split_once('=')
            .unwrap_or_else(|| panic!("invalid RSS metrics field: {field}"));
        match key {
            "warm_rss_kb" => metrics.warm_rss_kb = value.parse().unwrap(),
            "final_rss_kb" => metrics.final_rss_kb = value.parse().unwrap(),
            "delta_kb" => metrics.delta_kb = value.parse().unwrap(),
            other => panic!("unexpected RSS metrics key: {other}"),
        }
    }

    metrics
}

fn current_rss_kb() -> usize {
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "ps failed while sampling RSS: stdout=\n{}\nstderr=\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
        .stdout
        .iter()
        .map(|byte| *byte as char)
        .collect::<String>()
        .trim()
        .parse()
        .unwrap()
}

/// Helper: build a flat 8×8 RGB image with all pixels set to `[r, g, b]`.
fn solid_rgb_image(r: u8, g: u8, b: u8) -> Image<U8> {
    let width = 8u32;
    let height = 8u32;
    let bands = 3u32;
    let data: Vec<u8> = (0..width * height * bands)
        .map(|i| match i % 3 {
            0 => r,
            1 => g,
            _ => b,
        })
        .collect();
    Image::<U8>::from_buffer(width, height, bands, data).unwrap()
}

fn truncated_encoded_jpeg_bytes() -> Vec<u8> {
    let image = patterned_rgb_image(1024, 1024);
    let encoded = JpegCodec.encode(&image).unwrap();
    let truncate_at = encoded.len() * 3 / 4;
    encoded[..truncate_at].to_vec()
}

fn icc_app2_chunk(sequence_number: u8, chunk_count: u8, payload: &[u8]) -> Vec<u8> {
    let mut segment_payload = Vec::with_capacity(ICC_PROFILE_SIGNATURE.len() + 2 + payload.len());
    segment_payload.extend_from_slice(ICC_PROFILE_SIGNATURE);
    segment_payload.push(sequence_number);
    segment_payload.push(chunk_count);
    segment_payload.extend_from_slice(payload);

    let segment_len = (segment_payload.len() + 2) as u16;
    let mut segment = Vec::with_capacity(segment_payload.len() + 4);
    segment.extend_from_slice(&[0xFF, 0xE2]);
    segment.extend_from_slice(&segment_len.to_be_bytes());
    segment.extend_from_slice(&segment_payload);
    segment
}

fn with_embedded_icc_profile(jpeg: &[u8], profile: &[u8]) -> Vec<u8> {
    let split_at = profile.len() / 2;
    let first = &profile[..split_at];
    let second = &profile[split_at..];

    let mut with_profile = Vec::with_capacity(jpeg.len() + first.len() + second.len() + 32);
    with_profile.extend_from_slice(&jpeg[..2]);
    with_profile.extend_from_slice(&icc_app2_chunk(1, 2, first));
    with_profile.extend_from_slice(&icc_app2_chunk(2, 2, second));
    with_profile.extend_from_slice(&jpeg[2..]);
    with_profile
}

fn exif_orientation_segment(orientation: u16) -> Vec<u8> {
    let mut segment_payload = Vec::with_capacity(32);
    segment_payload.extend_from_slice(EXIF_SIGNATURE);
    segment_payload.extend_from_slice(b"II");
    segment_payload.extend_from_slice(&42u16.to_le_bytes());
    segment_payload.extend_from_slice(&8u32.to_le_bytes());
    segment_payload.extend_from_slice(&1u16.to_le_bytes());
    segment_payload.extend_from_slice(&0x0112u16.to_le_bytes());
    segment_payload.extend_from_slice(&3u16.to_le_bytes());
    segment_payload.extend_from_slice(&1u32.to_le_bytes());
    segment_payload.extend_from_slice(&orientation.to_le_bytes());
    segment_payload.extend_from_slice(&0u16.to_le_bytes());
    segment_payload.extend_from_slice(&0u32.to_le_bytes());

    let segment_len = (segment_payload.len() + 2) as u16;
    let mut segment = Vec::with_capacity(segment_payload.len() + 4);
    segment.extend_from_slice(&[0xFF, 0xE1]);
    segment.extend_from_slice(&segment_len.to_be_bytes());
    segment.extend_from_slice(&segment_payload);
    segment
}

fn with_exif_orientation(jpeg: &[u8], orientation: u16) -> Vec<u8> {
    let mut with_orientation = Vec::with_capacity(jpeg.len() + 40);
    with_orientation.extend_from_slice(&jpeg[..2]);
    with_orientation.extend_from_slice(&exif_orientation_segment(orientation));
    with_orientation.extend_from_slice(&jpeg[2..]);
    with_orientation
}

fn with_embedded_xmp(jpeg: &[u8], xmp: &[u8]) -> Vec<u8> {
    let payload = normalize_xmp_app1_payload(xmp);
    let segment_len = u16::try_from(payload.len() + 2).unwrap();
    let mut with_xmp = Vec::with_capacity(jpeg.len() + payload.len() + 4);
    with_xmp.extend_from_slice(&jpeg[..2]);
    with_xmp.extend_from_slice(&[0xFF, 0xE1]);
    with_xmp.extend_from_slice(&segment_len.to_be_bytes());
    with_xmp.extend_from_slice(&payload);
    with_xmp.extend_from_slice(&jpeg[2..]);
    with_xmp
}

fn jpeg_segment(marker: u8, payload: &[u8]) -> Vec<u8> {
    let segment_len = u16::try_from(payload.len() + 2).unwrap();
    let mut segment = Vec::with_capacity(payload.len() + 4);
    segment.extend_from_slice(&[0xFF, marker]);
    segment.extend_from_slice(&segment_len.to_be_bytes());
    segment.extend_from_slice(payload);
    segment
}

fn structural_jpeg_with_segments(segments: &[Vec<u8>]) -> Vec<u8> {
    let mut jpeg = Vec::with_capacity(2 + segments.iter().map(Vec::len).sum::<usize>() + 2);
    jpeg.extend_from_slice(&[0xFF, 0xD8]);
    for segment in segments {
        jpeg.extend_from_slice(segment);
    }
    jpeg.extend_from_slice(&[0xFF, 0xD9]);
    jpeg
}

fn assert_probe_and_decode_share_codec_error(src: &[u8], expected_message: &str) {
    let decode_error = JpegCodec.decode::<U8>(src).unwrap_err();
    let probe_error = JpegCodec.probe(src).unwrap_err();

    assert!(
        matches!(decode_error, ViprsError::Codec(ref message) if message.contains(expected_message)),
        "decode must return typed codec error containing {expected_message:?}, got: {decode_error:?}"
    );
    assert!(
        matches!(probe_error, ViprsError::Codec(ref message) if message.contains(expected_message)),
        "probe must return typed codec error containing {expected_message:?}, got: {probe_error:?}"
    );
}

fn find_marker_payload(jpeg: &[u8], marker: u8) -> Option<Vec<u8>> {
    let mut found = None;
    visit_jpeg_segments(jpeg, |current_marker, payload| {
        if current_marker == marker {
            found = Some(payload.to_vec());
            false
        } else {
            true
        }
    });
    found
}

fn start_of_frame_sampling_factor(jpeg: &[u8]) -> Option<u8> {
    let payload = find_marker_payload(jpeg, 0xC0).or_else(|| find_marker_payload(jpeg, 0xC2))?;
    let components = *payload.get(5)? as usize;
    let mut offset = 6usize;
    for component_index in 0..components {
        let component = payload.get(offset..offset + 3)?;
        if component_index == 0 {
            return Some(component[1]);
        }
        offset += 3;
    }
    None
}

fn patterned_rgb_image(width: u32, height: u32) -> Image<U8> {
    let mut data = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            data.push((x * 40 + y * 7) as u8);
            data.push((x * 11 + y * 50) as u8);
            data.push((x * 19 + y * 23) as u8);
        }
    }

    Image::<U8>::from_buffer(width, height, 3, data).unwrap()
}

fn quality_probe_rgb_image(width: u32, height: u32) -> Image<U8> {
    let mut data = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let checker = ((x / 4) + (y / 4)) % 2;
            let edge = if checker == 0 { 32 } else { 223 };
            let red = ((x * 29 + y * 7) % 256) as u8;
            let green = edge ^ ((x * 11 + y * 17) % 256) as u8;
            let blue = edge.wrapping_add(((x * 37 + y * 19) % 256) as u8);
            data.extend_from_slice(&[red, green, blue]);
        }
    }

    Image::<U8>::from_buffer(width, height, 3, data).unwrap()
}

fn psnr_db(original: &[u8], decoded: &[u8]) -> f64 {
    assert_eq!(original.len(), decoded.len());

    let mse = original
        .iter()
        .zip(decoded.iter())
        .map(|(&lhs, &rhs)| {
            let delta = f64::from(lhs) - f64::from(rhs);
            delta * delta
        })
        .sum::<f64>()
        / original.len() as f64;

    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * ((255.0_f64 * 255.0_f64) / mse).log10()
    }
}

fn rgb_histogram(image: &Image<U8>) -> [[u32; 256]; 3] {
    assert_eq!(image.bands(), 3);

    let mut histogram = [[0u32; 256]; 3];
    for pixel in image.pixels().chunks_exact(3) {
        histogram[0][usize::from(pixel[0])] += 1;
        histogram[1][usize::from(pixel[1])] += 1;
        histogram[2][usize::from(pixel[2])] += 1;
    }

    histogram
}

fn histogram_distance(lhs: &[[u32; 256]; 3], rhs: &[[u32; 256]; 3]) -> u32 {
    lhs.iter()
        .zip(rhs.iter())
        .flat_map(|(lhs_band, rhs_band)| lhs_band.iter().zip(rhs_band.iter()))
        .map(|(&lhs_count, &rhs_count)| lhs_count.abs_diff(rhs_count))
        .sum()
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

#[test]
fn sniff_recognises_jpeg_magic() {
    let codec = JpegCodec;
    // SOI marker + APP0 marker
    let header = [0xFF_u8, 0xD8, 0xFF, 0xE0, 0x00, 0x10];
    assert!(codec.sniff(&header));
}

#[test]
fn sniff_rejects_png() {
    let codec = JpegCodec;
    // PNG magic bytes
    let header = [0x89_u8, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    assert!(!codec.sniff(&header));
}

#[test]
fn round_trip_rgb() {
    let codec = JpegCodec;
    let original = solid_rgb_image(128, 64, 32);

    let opts = SaveOptions::default().with_quality(100);
    let encoded = codec.encode_with_options(&original, &opts).unwrap();

    let decoded = codec.decode::<U8>(&encoded).unwrap();

    assert_eq!(decoded.width(), 8);
    assert_eq!(decoded.height(), 8);
    assert_eq!(decoded.bands(), 3);

    // JPEG is lossy — allow ±10 per channel.
    let tolerance: i16 = 10;
    let orig_pixels = original.pixels();
    let dec_pixels = decoded.pixels();
    assert_eq!(orig_pixels.len(), dec_pixels.len());

    for (i, (&orig, &dec)) in orig_pixels.iter().zip(dec_pixels.iter()).enumerate() {
        let diff = (orig as i16 - dec as i16).abs();
        assert!(
            diff <= tolerance,
            "pixel sample {i}: original={orig}, decoded={dec}, diff={diff} > tolerance={tolerance}"
        );
    }
}

#[test]
fn encode_quality_affects_size() {
    let codec = JpegCodec;
    let image = solid_rgb_image(128, 64, 32);

    let low_quality = SaveOptions::default().with_quality(10);
    let high_quality = SaveOptions::default().with_quality(95);

    let low_bytes = codec.encode_with_options(&image, &low_quality).unwrap();
    let high_bytes = codec.encode_with_options(&image, &high_quality).unwrap();

    assert!(
        high_bytes.len() > low_bytes.len(),
        "quality=95 ({}) should produce more bytes than quality=10 ({})",
        high_bytes.len(),
        low_bytes.len()
    );
}

#[test]
fn jpeg_quality_100_round_trip_is_near_lossless() {
    let codec = JpegCodec;
    let image = quality_probe_rgb_image(64, 64);

    let encoded = codec
        .encode_with_options(&image, &SaveOptions::default().with_quality(100))
        .unwrap();
    let decoded = codec.decode::<U8>(&encoded).unwrap();
    let psnr = psnr_db(image.pixels(), decoded.pixels());

    assert_eq!(
        (decoded.width(), decoded.height(), decoded.bands()),
        (64, 64, 3)
    );
    assert!(
        psnr >= 40.0,
        "quality=100 round-trip PSNR should be >= 40dB, got {psnr:.2}dB"
    );
}

#[test]
fn jpeg_quality_extremes_change_size_and_pixel_distribution() {
    let codec = JpegCodec;
    let image = quality_probe_rgb_image(64, 64);

    let low_encoded = codec
        .encode_with_options(&image, &SaveOptions::default().with_quality(0))
        .unwrap();
    let high_encoded = codec
        .encode_with_options(&image, &SaveOptions::default().with_quality(100))
        .unwrap();
    let low_decoded = codec.decode::<U8>(&low_encoded).unwrap();
    let high_decoded = codec.decode::<U8>(&high_encoded).unwrap();
    let low_psnr = psnr_db(image.pixels(), low_decoded.pixels());
    let high_psnr = psnr_db(image.pixels(), high_decoded.pixels());
    let low_histogram = rgb_histogram(&low_decoded);
    let high_histogram = rgb_histogram(&high_decoded);
    let histogram_delta = histogram_distance(&low_histogram, &high_histogram);

    assert!(
        low_encoded.len() < high_encoded.len(),
        "quality=0 ({}) should produce fewer bytes than quality=100 ({})",
        low_encoded.len(),
        high_encoded.len()
    );
    assert_eq!(
        (
            low_decoded.width(),
            low_decoded.height(),
            low_decoded.bands()
        ),
        (64, 64, 3)
    );
    assert_eq!(
        (
            high_decoded.width(),
            high_decoded.height(),
            high_decoded.bands()
        ),
        (64, 64, 3)
    );
    assert_ne!(
        low_decoded.pixels(),
        high_decoded.pixels(),
        "quality extremes must not decode to identical pixels"
    );
    assert!(
        histogram_delta > 0,
        "quality extremes must change the decoded RGB histogram"
    );
    assert!(
        high_psnr > low_psnr,
        "quality=100 PSNR ({high_psnr:.2}dB) should exceed quality=0 ({low_psnr:.2}dB)"
    );
}

#[test]
fn audit_reference_sample_fixture_decode_matches_imagemagick() {
    let decoded = JpegCodec.decode::<U8>(SAMPLE_JPEG).unwrap();

    assert_eq!(
        (decoded.width(), decoded.height(), decoded.bands()),
        (290, 442, 3)
    );
    assert_eq!(
        decoded.metadata().interpretation,
        Some(Interpretation::Srgb)
    );
    let samples = [
        ((0u32, 0u32), [176u8, 160, 127]),
        ((1, 0), [171, 155, 122]),
        ((10, 10), [141, 127, 90]),
        ((145, 221), [73, 76, 111]),
        ((289, 441), [93, 88, 68]),
        ((50, 300), [145, 134, 112]),
        ((200, 100), [75, 80, 118]),
    ];
    for ((x, y), expected) in samples {
        let offset = ((y * decoded.width() + x) * decoded.bands()) as usize;
        let actual = &decoded.pixels()[offset..offset + 3];
        for (channel, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = actual.abs_diff(expected);
            assert!(
                diff <= 2,
                "sample ({x},{y}) channel {channel}: expected {expected}, got {actual}, diff={diff}"
            );
        }
    }
}

#[test]
fn audit_reference_large_fixture_decode_matches_imagemagick() {
    let decoded = JpegCodec.decode::<U8>(BENCH_8192_JPEG).unwrap();

    assert_eq!(
        (decoded.width(), decoded.height(), decoded.bands()),
        (8192, 8192, 3)
    );
    assert_eq!(
        decoded.metadata().interpretation,
        Some(Interpretation::Srgb)
    );
    assert_eq!(crc32(decoded.pixels()), BENCH_8192_JPEG_RGB_CRC32);
}

#[test]
fn audit_roundtrip_tiny_2x2_rgba_quality_95_preserves_rgb_detail() {
    let original = Image::<U8>::from_buffer(
        2,
        2,
        4,
        vec![
            20, 40, 60, 255, 80, 100, 120, 255, 140, 160, 180, 255, 200, 220, 240, 255,
        ],
    )
    .unwrap();
    let encoded = JpegCodec
        .encode_with_options(&original, &SaveOptions::default().with_quality(95))
        .unwrap();
    let decoded = JpegCodec.decode::<U8>(&encoded).unwrap();
    let original_rgb: Vec<u8> = original
        .pixels()
        .chunks_exact(4)
        .flat_map(|rgba| rgba[..3].iter().copied())
        .collect();
    let psnr = psnr_db(&original_rgb, decoded.pixels());

    assert_eq!(
        (decoded.width(), decoded.height(), decoded.bands()),
        (2, 2, 3)
    );
    assert!(
        psnr >= 40.0,
        "quality=95 tiny RGBA JPEG round-trip PSNR should be >= 40dB, got {psnr:.2}dB"
    );
}

#[test]
fn jpeg_quality_101_returns_error() {
    let codec = JpegCodec;
    let image = patterned_rgb_image(17, 11);

    let error = codec
        .encode_with_options(&image, &SaveOptions::default().with_quality(101))
        .unwrap_err();

    assert!(
        matches!(error, ViprsError::Codec(ref message) if message.contains("quality must be in range 0..=100")),
        "expected out-of-range quality error, got: {error:?}"
    );
}

#[test]
fn encode_default_quality_matches_libvips_default() {
    let codec = JpegCodec;
    let image = patterned_rgb_image(16, 16);

    let default_encoded = codec.encode(&image).unwrap();
    let explicit_default = codec
        .encode_with_options(&image, &SaveOptions::default().with_quality(75))
        .unwrap();

    assert_eq!(
        default_encoded, explicit_default,
        "JPEG encode() must default to libvips Q=75"
    );
}

#[test]
fn encode_progressive_option_emits_progressive_sof_marker() {
    let codec = JpegCodec;
    let image = patterned_rgb_image(16, 16);

    let baseline = codec
        .encode_with_options(&image, &SaveOptions::default())
        .unwrap();
    let progressive = codec
        .encode_with_options(&image, &SaveOptions::default().with_interlace(true))
        .unwrap();

    assert!(find_marker_payload(&baseline, 0xC0).is_some());
    assert!(find_marker_payload(&progressive, 0xC2).is_some());
    assert_ne!(baseline, progressive);
}

#[test]
fn encode_restart_interval_writes_dri_segment() {
    let codec = JpegCodec;
    let image = patterned_rgb_image(16, 16);

    let encoded = codec
        .encode_with_options(&image, &SaveOptions::default().with_restart_interval(7))
        .unwrap();
    let dri = find_marker_payload(&encoded, 0xDD).expect("DRI segment must exist");

    assert_eq!(dri, 7u16.to_be_bytes());
}

#[test]
fn encode_subsampling_option_changes_sof_sampling_factor() {
    let codec = JpegCodec;
    let image = patterned_rgb_image(16, 16);

    let subsampled = codec
        .encode_with_options(
            &image,
            &SaveOptions::default().with_jpeg_subsampling(JpegSubsampling::Subsample420),
        )
        .unwrap();
    let full_res = codec
        .encode_with_options(
            &image,
            &SaveOptions::default().with_jpeg_subsampling(JpegSubsampling::Off),
        )
        .unwrap();

    assert_eq!(start_of_frame_sampling_factor(&subsampled), Some(0x22));
    assert_eq!(start_of_frame_sampling_factor(&full_res), Some(0x11));
    assert_ne!(subsampled, full_res);
}

#[test]
fn decode_embedded_icc_profile_stores_metadata() {
    let codec = JpegCodec;
    let original = solid_rgb_image(128, 64, 32);
    let encoded = codec
        .encode_with_options(&original, &SaveOptions::default().with_quality(100))
        .unwrap();
    let expected_profile = (0u8..32).collect::<Vec<_>>();
    let encoded_with_profile = with_embedded_icc_profile(&encoded, &expected_profile);

    let decoded = codec.decode::<U8>(&encoded_with_profile).unwrap();

    assert_eq!(
        decoded.metadata().icc_profile.as_deref(),
        Some(expected_profile.as_slice())
    );
}

#[test]
fn decode_embedded_xmp_stores_metadata() {
    let codec = JpegCodec;
    let original = solid_rgb_image(16, 16, 48);
    let encoded = codec
        .encode_with_options(&original, &SaveOptions::default().with_quality(100))
        .unwrap();
    let expected_xmp = br#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF/></x:xmpmeta>"#;
    let encoded_with_xmp = with_embedded_xmp(&encoded, expected_xmp);

    let decoded = codec.decode::<U8>(&encoded_with_xmp).unwrap();

    assert_eq!(
        decoded.metadata().xmp.as_deref(),
        Some(expected_xmp.as_slice())
    );
}

#[test]
fn encode_round_trips_embedded_icc_exif_and_xmp_metadata() {
    let codec = JpegCodec;
    let image = patterned_rgb_image(8, 8).with_metadata(ImageMetadata {
        icc_profile: Some((0u8..48).collect()),
        exif: Some(exif_orientation_segment(6)[4..].to_vec()),
        xmp: Some(br#"<x:xmpmeta><rdf:RDF/></x:xmpmeta>"#.to_vec()),
        ..ImageMetadata::default()
    });

    let encoded = codec
        .encode_with_options(&image, &SaveOptions::default())
        .unwrap();
    let decoded = codec.decode::<U8>(&encoded).unwrap();

    assert_eq!(decoded.metadata().icc_profile, image.metadata().icc_profile);
    assert_eq!(decoded.metadata().exif, image.metadata().exif);
    assert_eq!(decoded.metadata().xmp, image.metadata().xmp);
}

#[cfg(feature = "icc")]
#[test]
fn encode_gray_icc_input_normalizes_to_srgb_web_output() {
    let codec = JpegCodec;
    let gray_profile = profile_load("gray").expect("load gray profile");
    let srgb_profile = profile_load("srgb").expect("load srgb profile");
    let image = Image::<U8>::from_buffer(2, 2, 1, vec![32, 96, 160, 224])
        .unwrap()
        .with_metadata(ImageMetadata {
            interpretation: Some(Interpretation::BW),
            icc_profile: Some(gray_profile),
            ..ImageMetadata::default()
        });

    let encoded = codec
        .encode_with_options(&image, &SaveOptions::default().with_quality(100))
        .expect("jpeg encode should succeed");
    let decoded = codec
        .decode::<U8>(&encoded)
        .expect("jpeg decode should succeed");

    assert_eq!(decoded.bands(), 3, "web JPEG output must be sRGB RGB");
    assert_eq!(
        decoded.metadata().interpretation,
        Some(Interpretation::Srgb)
    );
    assert_eq!(
        decoded.metadata().icc_profile.as_deref(),
        Some(srgb_profile.as_slice())
    );
}

#[test]
fn streaming_encode_to_writer_round_trips_embedded_icc_exif_and_xmp_metadata() {
    let codec = JpegCodec;
    let image = patterned_rgb_image(8, 8).with_metadata(ImageMetadata {
        icc_profile: Some((0u8..48).collect()),
        exif: Some(exif_orientation_segment(6)[4..].to_vec()),
        xmp: Some(br#"<x:xmpmeta><rdf:RDF/></x:xmpmeta>"#.to_vec()),
        ..ImageMetadata::default()
    });
    let mut encoded = Vec::new();

    codec
        .encode_to_writer(&image, &SaveOptions::default(), &mut encoded)
        .unwrap();
    let decoded = codec.decode::<U8>(&encoded).unwrap();

    assert_eq!(decoded.metadata().icc_profile, image.metadata().icc_profile);
    assert_eq!(decoded.metadata().exif, image.metadata().exif);
    assert_eq!(decoded.metadata().xmp, image.metadata().xmp);
}

#[test]
fn streaming_encode_to_writer_uses_multiple_write_calls() {
    #[derive(Default)]
    struct CountingWriter {
        bytes: Vec<u8>,
        writes: usize,
    }

    impl Write for CountingWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.writes += 1;
            self.bytes.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    let codec = JpegCodec;
    let image = patterned_rgb_image(256, 256);
    let mut writer = CountingWriter::default();

    codec
        .encode_to_writer(&image, &SaveOptions::default(), &mut writer)
        .unwrap();

    assert!(writer.writes > 1);
    let decoded = codec.decode::<U8>(&writer.bytes).unwrap();
    assert_eq!(decoded.width(), 256);
    assert_eq!(decoded.height(), 256);
}

#[test]
fn strip_metadata_omits_embedded_icc_exif_and_xmp() {
    let codec = JpegCodec;
    let image = patterned_rgb_image(8, 8).with_metadata(ImageMetadata {
        icc_profile: Some((0u8..16).collect()),
        exif: Some(exif_orientation_segment(6)[4..].to_vec()),
        xmp: Some(br#"<x:xmpmeta>strip-me</x:xmpmeta>"#.to_vec()),
        ..ImageMetadata::default()
    });

    let encoded = codec
        .encode_with_options(&image, &SaveOptions::default().strip_metadata())
        .unwrap();
    let decoded = codec.decode::<U8>(&encoded).unwrap();

    assert!(decoded.metadata().icc_profile.is_none());
    assert!(decoded.metadata().exif.is_none());
    assert!(decoded.metadata().xmp.is_none());
}

#[test]
fn decode_sets_srgb_interpretation_for_rgb_jpeg() {
    let codec = JpegCodec;
    let original = Image::<U8>::from_buffer(2, 2, 3, vec![32u8; 12]).unwrap();
    let encoded = codec.encode(&original).unwrap();
    let decoded = codec.decode::<viprs_core::format::U8>(&encoded).unwrap();

    assert_eq!(
        decoded.metadata().interpretation,
        Some(Interpretation::Srgb)
    );
}

#[test]
fn decode_with_shrink_factor_two_uses_native_reduced_idct_pixels_and_dimensions() {
    use std::num::NonZeroU8;

    let codec = JpegCodec;
    let original = patterned_rgb_image(16, 8);
    let encoded = codec
        .encode_with_options(&original, &SaveOptions::default().with_quality(100))
        .unwrap();

    let shrunk = codec
        .decode_with_options::<U8>(
            &encoded,
            &LoadOptions::default().with_shrink(NonZeroU8::new(2).unwrap()),
        )
        .unwrap();

    assert_eq!((shrunk.width(), shrunk.height()), (8, 4));
    assert_eq!(shrunk.bands(), 3);
    assert_eq!(shrunk.pixels().len(), 8 * 4 * 3);

    let pixel = |x: usize, y: usize| -> [u8; 3] {
        let base = (y * shrunk.width() as usize + x) * shrunk.bands() as usize;
        [
            shrunk.pixels()[base],
            shrunk.pixels()[base + 1],
            shrunk.pixels()[base + 2],
        ]
    };

    // Expected samples from libjpeg-turbo reduced-IDCT shrink-on-load, matching
    // `vips copy input.jpg[shrink=2]`.
    assert_eq!(pixel(0, 0), [24, 32, 21]);
    assert_eq!(pixel(3, 1), [85, 197, 181]);
    assert_eq!(pixel(7, 3), [114, 165, 169]);
}

#[cfg(feature = "_integration")]
#[test]
fn decode_with_shrink_factor_eight_uses_less_peak_memory_than_full_decode() {
    let full_decode = run_decode_metrics_child(0);
    let shrunk_decode = run_decode_metrics_child(8);

    assert_eq!((full_decode.width, full_decode.height), (8192, 8192));
    assert_eq!((shrunk_decode.width, shrunk_decode.height), (1024, 1024));
    assert!(
        shrunk_decode.peak_live_bytes < full_decode.peak_live_bytes / 4,
        "factor=8 shrink-on-load should cut peak live bytes well below a full decode: full={full_decode:?}, shrunk={shrunk_decode:?}"
    );
    assert!(
        shrunk_decode.alloc_bytes < full_decode.alloc_bytes / 2,
        "factor=8 shrink-on-load should allocate substantially fewer bytes than a full decode: full={full_decode:?}, shrunk={shrunk_decode:?}"
    );
    // NOTE: alloc_count may be higher for shrunk decode (multiple small buffers
    // vs one large buffer for full decode). What matters is total bytes and peak.
}

#[cfg(feature = "_integration")]
#[test]
fn decode_with_shrink_factor_eight_still_materializes_full_shrunk_frame() {
    let shrunk_decode = run_decode_metrics_child(8);
    let resident_frame_bytes = u64::from(shrunk_decode.width) * u64::from(shrunk_decode.height) * 3;

    assert_eq!((shrunk_decode.width, shrunk_decode.height), (1024, 1024));
    assert!(
        shrunk_decode.peak_live_bytes >= resident_frame_bytes,
        "factor=8 shrink-on-load still uses one full resident shrunken frame in ImageDecoder::decode_with_options: shrunk={shrunk_decode:?}, resident_frame_bytes={resident_frame_bytes}"
    );
}

#[test]
fn decode_path_with_options_matches_in_memory_decode() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("jpeg-decode-path-options.jpg");
    fs::create_dir_all(path.parent().expect("target dir")).unwrap();
    fs::write(&path, BENCH_2048_JPEG).unwrap();

    let from_memory = JpegCodec
        .decode_with_options::<U8>(BENCH_2048_JPEG, &LoadOptions::default())
        .unwrap();
    let from_path = JpegCodec
        .decode_path_with_options::<U8>(&path, &LoadOptions::default())
        .unwrap();

    assert_eq!(
        (from_path.width(), from_path.height(), from_path.bands()),
        (
            from_memory.width(),
            from_memory.height(),
            from_memory.bands()
        )
    );
    assert_eq!(from_path.pixels(), from_memory.pixels());
    assert_eq!(from_path.metadata(), from_memory.metadata());
}

#[cfg(feature = "_integration")]
#[test]
fn shrink_on_load_decode_metrics_child() {
    let Some(factor) = std::env::var_os(METRICS_FACTOR_ENV) else {
        return;
    };
    let factor = factor.to_string_lossy().parse::<u8>().unwrap();
    let load_options = std::num::NonZeroU8::new(factor)
        .map_or_else(LoadOptions::default, |value| {
            LoadOptions::default().with_shrink(value)
        });

    crate::test_support::reset_alloc_stats();
    let decoded = JpegCodec
        .decode_with_options::<U8>(BENCH_8192_JPEG, &load_options)
        .unwrap();
    let stats = crate::test_support::alloc_stats();

    println!(
        "{METRICS_PREFIX} factor={} width={} height={} pixels={} alloc_count={} alloc_bytes={} peak_live_bytes={}",
        factor,
        decoded.width(),
        decoded.height(),
        decoded.pixels().len(),
        stats.alloc_count,
        stats.alloc_bytes,
        stats.peak_live_bytes
    );
}

#[test]
fn shrink_on_load_plan_uses_native_backend() {
    for factor in [2, 4, 8] {
        let plan = jpeg_shrink_on_load_plan(factor);
        assert_eq!(plan.factor(), factor);
        assert_eq!(plan.backend(), ShrinkOnLoadBackend::JpegTurboScaledIdct);
    }
}

#[test]
fn unsupported_shrink_factor_is_not_sent_to_fallback() {
    let plan = jpeg_shrink_on_load_plan(3);

    assert_eq!(plan.factor(), 1);
    assert_eq!(plan.backend(), ShrinkOnLoadBackend::JpegTurboScaledIdct);
}

#[test]
fn decode_with_max_dimension_uses_closest_supported_shrink_factor() {
    let codec = JpegCodec;
    let original = patterned_rgb_image(16, 8);
    let encoded = codec
        .encode_with_options(&original, &SaveOptions::default().with_quality(100))
        .unwrap();

    let shrunk = codec
        .decode_with_options::<U8>(&encoded, &LoadOptions::default().with_max_dimension(6))
        .unwrap();

    assert_eq!((shrunk.width(), shrunk.height()), (4, 2));
    assert!(shrunk.width() <= 6);
    assert!(shrunk.height() <= 6);
}

#[test]
fn decode_with_no_rotate_preserves_stored_orientation_pixels() {
    let codec = JpegCodec;
    let original = patterned_rgb_image(2, 3);
    let encoded = codec
        .encode_with_options(&original, &SaveOptions::default().with_quality(100))
        .unwrap();
    let encoded_with_orientation = with_exif_orientation(&encoded, 6);

    let unrotated = codec
        .decode_with_options::<U8>(
            &encoded_with_orientation,
            &LoadOptions::default().no_rotate(),
        )
        .unwrap();
    let rotated = codec
        .decode_with_options::<U8>(&encoded_with_orientation, &LoadOptions::default())
        .unwrap();

    assert_eq!((unrotated.width(), unrotated.height()), (2, 3));
    assert_eq!(unrotated.metadata().orientation, Some(6));
    assert_eq!(rotated.metadata().orientation, Some(1));
    assert_eq!((rotated.width(), rotated.height()), (3, 2));

    let (expected_width, expected_height, expected_pixels) = apply_exif_orientation(
        unrotated.pixels().to_vec(),
        unrotated.width(),
        unrotated.height(),
        unrotated.bands(),
        6,
        "jpeg-test",
    )
    .unwrap();

    assert_eq!(
        (expected_width, expected_height),
        (rotated.width(), rotated.height())
    );
    assert_eq!(expected_pixels, rotated.pixels());
}

// ── edge cases ────────────────────────────────────────────────────────────

#[test]
fn decode_empty_slice_returns_codec_error() {
    let codec = JpegCodec;
    let result = codec.decode::<U8>(&[]);
    assert!(
        matches!(result, Err(ViprsError::Codec(_))),
        "empty input must return ViprsError::Codec, got: {result:?}"
    );
}

#[test]
fn decode_truncated_valid_jpeg_returns_codec_error() {
    let result = JpegCodec.decode::<U8>(&truncated_encoded_jpeg_bytes());
    assert!(
        matches!(result, Err(ViprsError::Codec(_))),
        "truncated encoded JPEG must return ViprsError::Codec, got: {result:?}"
    );
}

#[cfg(feature = "_integration")]
#[test]
fn repeated_truncated_decode_errors_keep_rss_stable() {
    let metrics = run_rss_child();
    assert!(
        metrics.delta_kb <= RSS_STABILITY_THRESHOLD_KB,
        "repeated JPEG decode failures grew RSS too much: {metrics:?}"
    );
}

#[test]
fn repeated_truncated_decode_errors_keep_rss_stable_child() {
    if std::env::var_os(RSS_CHILD_ENV).is_none() {
        return;
    }

    let truncated = truncated_encoded_jpeg_bytes();
    for _ in 0..20 {
        let result = JpegCodec.decode::<U8>(&truncated);
        assert!(
            matches!(result, Err(ViprsError::Codec(_))),
            "warm-up decode failures must return ViprsError::Codec, got: {result:?}"
        );
    }
    let warm_rss_kb = current_rss_kb();

    for _ in 0..120 {
        let result = JpegCodec.decode::<U8>(&truncated);
        assert!(
            matches!(result, Err(ViprsError::Codec(_))),
            "repeated decode failures must return ViprsError::Codec, got: {result:?}"
        );
    }
    let final_rss_kb = current_rss_kb();

    println!(
        "{RSS_PREFIX} warm_rss_kb={} final_rss_kb={} delta_kb={}",
        warm_rss_kb,
        final_rss_kb,
        final_rss_kb.saturating_sub(warm_rss_kb)
    );
}

#[test]
fn encode_1x1_grayscale_round_trip_within_tolerance() {
    let codec = JpegCodec;
    // 1×1 grayscale image (1 band) with a mid-range pixel value.
    // libjpeg-turbo preserves grayscale output as 1 band.
    let data: Vec<u8> = vec![128u8];
    let original = Image::<U8>::from_buffer(1, 1, 1, data).unwrap();

    let encoded = codec.encode(&original).unwrap();
    let decoded = codec.decode::<U8>(&encoded).unwrap();

    assert_eq!(decoded.width(), 1);
    assert_eq!(decoded.height(), 1);
    assert_eq!(decoded.bands(), 1);

    // The first decoded sample corresponds to the luma channel. Regardless
    // of whether the decoder outputs Luma or RGB, the first sample must be
    // within ±10 of the original 128.
    let orig_val = original.pixels()[0] as i16;
    let dec_val = decoded.pixels()[0] as i16;
    let diff = (orig_val - dec_val).abs();
    assert!(
        diff <= 10,
        "1×1 grayscale pixel: original={orig_val}, decoded={dec_val}, diff={diff} > tolerance=10"
    );
}

#[test]
fn encode_2_band_image_returns_codec_error() {
    let codec = JpegCodec;
    // 2 bands is not a valid JPEG colour model (only 1, 3, 4 are supported).
    let data: Vec<u8> = vec![0u8; 4 * 4 * 2];
    let image = Image::<U8>::from_buffer(4, 4, 2, data).unwrap();
    let result = codec.encode(&image);
    assert!(
        matches!(result, Err(ViprsError::Codec(_))),
        "2-band image must return ViprsError::Codec, got: {result:?}"
    );
}

#[test]
fn sniff_with_less_than_3_bytes_returns_false() {
    let codec = JpegCodec;
    assert!(!codec.sniff(&[]), "empty slice must return false");
    assert!(!codec.sniff(&[0xFF]), "1-byte slice must return false");
    assert!(
        !codec.sniff(&[0xFF, 0xD8]),
        "2-byte slice must return false"
    );
}

#[test]
fn probe_empty_slice_returns_codec_error() {
    let codec = JpegCodec;
    let result = codec.probe(&[]);
    assert!(
        matches!(result, Err(ViprsError::Codec(_))),
        "probe on empty input must return ViprsError::Codec, got: {result:?}"
    );
}

#[test]
fn decode_and_probe_reject_sof_less_jpegs_consistently() {
    let malformed = structural_jpeg_with_segments(&[jpeg_segment(0xE0, b"JFIF\0")]);

    assert_probe_and_decode_share_codec_error(&malformed, "missing SOF");
}

#[test]
fn decode_and_probe_reject_multiple_sof_markers_consistently() {
    let sof_payload = [8, 0, 1, 0, 1, 1];
    let malformed = structural_jpeg_with_segments(&[
        jpeg_segment(0xC0, &sof_payload),
        jpeg_segment(0xC2, &sof_payload),
    ]);

    assert_probe_and_decode_share_codec_error(&malformed, "multiple SOF");
}

#[test]
fn decode_and_probe_reject_truncated_sof_segments_consistently() {
    let malformed = structural_jpeg_with_segments(&[jpeg_segment(0xC0, &[8, 0, 1, 0, 1])]);

    assert_probe_and_decode_share_codec_error(&malformed, "truncated SOF");
}

#[test]
fn decode_and_probe_reject_invalid_icc_chunk_headers_consistently() {
    let encoded = JpegCodec
        .encode_with_options(&patterned_rgb_image(8, 8), &SaveOptions::default())
        .unwrap();
    let malformed = {
        let mut bytes = encoded.clone();
        insert_segment_after_soi(&mut bytes, 0xE2, ICC_PROFILE_SIGNATURE).unwrap();
        bytes
    };

    assert_probe_and_decode_share_codec_error(&malformed, "truncated ICC profile chunk header");
}

#[test]
fn decode_and_probe_reject_invalid_exif_payloads_consistently() {
    let encoded = JpegCodec
        .encode_with_options(&patterned_rgb_image(8, 8), &SaveOptions::default())
        .unwrap();
    let malformed = {
        let mut bytes = encoded.clone();
        insert_segment_after_soi(&mut bytes, 0xE1, EXIF_SIGNATURE).unwrap();
        bytes
    };

    assert_probe_and_decode_share_codec_error(&malformed, "truncated EXIF TIFF header");
}

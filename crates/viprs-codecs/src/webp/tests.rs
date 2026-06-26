#![cfg(test)]

use super::*;
use crate::shrink_on_load::ShrinkOnLoadBackend;
use libwebp_sys::{
    WEBP_MUX_ABI_VERSION, WebPChunkId, WebPData, WebPFree, WebPMuxAnimBlend, WebPMuxAnimDispose,
    WebPMuxAnimParams, WebPMuxAssemble, WebPMuxDelete, WebPMuxError, WebPMuxFrameInfo,
    WebPMuxPushFrame, WebPMuxSetAnimationParams, WebPMuxSetCanvasSize, WebPNewInternal,
};
use std::ffi::c_void;
use std::num::NonZeroU8;
use viprs_core::codec_options::LoadOptions;
use viprs_core::codec_options::SaveOptions;
use viprs_core::error::ViprsError;
use viprs_core::format::U8;
use viprs_core::image::{ImageMetadata, InMemoryImage, Interpretation, Region};
#[cfg(all(feature = "icc", feature = "_integration"))]
use viprs_ops_colour::colour::profile_load;
use viprs_ports::codec::{ImageDecoder, ImageEncoder, TileImageDecoder};
use webp::{AnimEncoder, AnimFrame, BitstreamFeatures, WebPConfig};

struct WebpScratchAllocationLimitGuard {
    previous: Option<u64>,
    previous_total: Option<u64>,
}

impl WebpScratchAllocationLimitGuard {
    fn new(limit: u64) -> Self {
        let previous = test_webp_max_scratch_allocation_bytes_override();
        let previous_total = test_webp_max_total_animation_bytes_override();
        set_test_webp_max_scratch_allocation_bytes(Some(limit));
        set_test_webp_max_total_animation_bytes(Some(limit));
        Self {
            previous,
            previous_total,
        }
    }
}

impl Drop for WebpScratchAllocationLimitGuard {
    fn drop(&mut self) {
        set_test_webp_max_scratch_allocation_bytes(self.previous);
        set_test_webp_max_total_animation_bytes(self.previous_total);
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

// ── sniff ─────────────────────────────────────────────────────────────────

#[test]
fn sniff_recognises_webp_magic() {
    let codec = WebpCodec;
    // Build a 12-byte header with the correct RIFF/WEBP magic.
    // Bytes [4..8] are the RIFF file size — any value is accepted.
    let mut header = [0u8; 12];
    header[0..4].copy_from_slice(b"RIFF");
    header[4..8].copy_from_slice(&1234u32.to_le_bytes());
    header[8..12].copy_from_slice(b"WEBP");
    assert!(codec.sniff(&header), "must recognise RIFF....WEBP magic");
}

#[test]
fn sniff_rejects_jpeg() {
    let codec = WebpCodec;
    // JPEG starts with the SOI marker FF D8 FF.
    assert!(!codec.sniff(&[
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01
    ]));
}

#[test]
fn test_webp_static_region_scratch_allocation_rejects_huge_dimensions() {
    let err = checked_webp_scratch_allocation_len(u32::MAX, u32::MAX, 4, "test scratch")
        .expect_err("huge scratch allocation must be rejected");

    match err {
        ViprsError::ImageTooLarge {
            width,
            height,
            bands,
            bytes,
            limit_bytes,
            details,
        } => {
            assert_eq!(width, u32::MAX);
            assert_eq!(height, u32::MAX);
            assert_eq!(bands, 4);
            assert!(bytes > limit_bytes);
            assert_eq!(details, "test scratch");
        }
        other => panic!("expected ImageTooLarge, got {other:?}"),
    }
}

fn clamped_region_pixels_u8(image: &InMemoryImage<U8>, region: Region) -> Vec<u8> {
    let bands = image.bands() as usize;
    let mut output = vec![0u8; region.pixel_count() * bands];
    for out_y in 0..region.height {
        let src_y = (region.y + out_y as i32).clamp(0, image.height() as i32 - 1) as usize;
        for out_x in 0..region.width {
            let src_x = (region.x + out_x as i32).clamp(0, image.width() as i32 - 1) as usize;
            let src = (src_y * image.width() as usize + src_x) * bands;
            let dst = (out_y as usize * region.width as usize + out_x as usize) * bands;
            output[dst..dst + bands].copy_from_slice(&image.pixels()[src..src + bands]);
        }
    }
    output
}

fn assert_static_region_decode_matches_eager(
    encoded: &[u8],
    opts: &LoadOptions,
    regions: &[Region],
) {
    let eager = WebpCodec.decode_with_options::<U8>(encoded, opts).unwrap();

    for &region in regions {
        let mut output = vec![0u8; region.pixel_count() * eager.bands() as usize];
        WebpCodec
            .decode_region_into::<U8>(encoded, opts, region, &mut output)
            .unwrap();
        assert_eq!(output, clamped_region_pixels_u8(&eager, region));
    }
}

#[test]
fn tile_decoder_matches_eager_decode_region_after_shrink() {
    let codec = WebpCodec;
    let pixels: Vec<u8> = (0..8 * 6 * 3).map(|value| (value % 251) as u8).collect();
    let image = InMemoryImage::<U8>::from_buffer(8, 6, 3, pixels).unwrap();
    let encoded = codec.encode(&image).unwrap();
    let opts = LoadOptions::default().with_shrink(NonZeroU8::new(2).unwrap());
    let eager = codec.decode_with_options::<U8>(&encoded, &opts).unwrap();
    let region = Region::new(-1, 1, 4, 2);
    let mut actual = vec![0u8; region.pixel_count() * eager.bands() as usize];

    codec
        .decode_region_into::<U8>(&encoded, &opts, region, &mut actual)
        .unwrap();

    assert_eq!(actual, clamped_region_pixels_u8(&eager, region));
}

#[test]
fn tile_decoder_matches_eager_decode_with_odd_origin_and_clamped_edges() {
    let image = patterned_rgb(10, 8);
    let encoded = WebpCodec.encode(&image).unwrap();
    let opts = LoadOptions::default().with_shrink(NonZeroU8::new(2).unwrap());
    let eager = WebpCodec
        .decode_with_options::<U8>(&encoded, &opts)
        .unwrap();
    let region = Region::new(-1, 2, 4, 2);
    let mut output = vec![0u8; region.pixel_count() * 3];

    WebpCodec
        .decode_region_into::<U8>(&encoded, &opts, region, &mut output)
        .unwrap();

    assert_eq!(output, clamped_region_pixels_u8(&eager, region));
}

#[test]
fn tile_decoder_full_resolution_matches_eager_for_center_odd_and_clamped_tiles() {
    let image = patterned_rgb(19, 17);
    let encoded = WebpCodec.encode(&image).unwrap();
    let opts = LoadOptions::default();

    assert_static_region_decode_matches_eager(
        &encoded,
        &opts,
        &[
            Region::new(6, 4, 5, 4),
            Region::new(7, 13, 5, 3),
            Region::new(15, 14, 6, 5),
        ],
    );
}

#[test]
fn sniff_rejects_short_header() {
    let codec = WebpCodec;
    assert!(!codec.sniff(b"RIFF"));
}

// ── encode / decode round-trips ───────────────────────────────────────────

/// Create a solid-colour 8x8 RGB U8 image where every pixel is the given
/// `[r, g, b]` triple.
fn solid_rgb_8x8(r: u8, g: u8, b: u8) -> InMemoryImage<U8> {
    let mut data = Vec::with_capacity(8 * 8 * 3);
    for _ in 0..(8 * 8) {
        data.push(r);
        data.push(g);
        data.push(b);
    }
    InMemoryImage::from_buffer(8, 8, 3, data).unwrap()
}

fn patterned_rgb(width: u32, height: u32) -> InMemoryImage<U8> {
    let mut data = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            data.push(((x * 17 + y * 3) % 256) as u8);
            data.push(((x * 7 + y * 29) % 256) as u8);
            data.push((((x ^ y) * 11) % 256) as u8);
        }
    }
    InMemoryImage::from_buffer(width, height, 3, data).unwrap()
}

fn chroma_stress_rgb(width: u32, height: u32) -> InMemoryImage<U8> {
    let mut data = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            let pixel = match (x / 4 + y / 4) % 4 {
                0 => [255, 0, 0],
                1 => [0, 255, 0],
                2 => [0, 0, 255],
                _ => [255, 255, 0],
            };
            data.extend_from_slice(&pixel);
        }
    }
    InMemoryImage::from_buffer(width, height, 3, data).unwrap()
}

fn alternating_rgb(width: u32, height: u32) -> InMemoryImage<U8> {
    let mut data = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            data.push(if x % 2 == 0 { 0 } else { 255 });
            data.push(if y % 2 == 0 { 0 } else { 255 });
            data.push(if (x + y) % 2 == 0 { 0 } else { 255 });
        }
    }
    InMemoryImage::from_buffer(width, height, 3, data).unwrap()
}

fn rgb_nearest_downsample_2x(image: &InMemoryImage<U8>) -> Vec<u8> {
    let width = image.width() / 2;
    let height = image.height() / 2;
    let mut out = Vec::with_capacity(width as usize * height as usize * 3);
    let in_width = image.width() as usize;
    let pixels = image.pixels();

    for y in 0..height as usize {
        for x in 0..width as usize {
            let src_x = x * 2;
            let src_y = y * 2;
            let src = (src_y * in_width + src_x) * 3;
            out.extend_from_slice(&pixels[src..src + 3]);
        }
    }

    out
}

fn rgb_box_downsample_2x(image: &InMemoryImage<U8>) -> Vec<u8> {
    let width = image.width() / 2;
    let height = image.height() / 2;
    let mut out = Vec::with_capacity(width as usize * height as usize * 3);
    let in_width = image.width() as usize;
    let pixels = image.pixels();

    for y in 0..height as usize {
        for x in 0..width as usize {
            let src_x = x * 2;
            let src_y = y * 2;
            for channel in 0..3usize {
                let p00 = u16::from(pixels[(src_y * in_width + src_x) * 3 + channel]);
                let p10 = u16::from(pixels[(src_y * in_width + (src_x + 1)) * 3 + channel]);
                let p01 = u16::from(pixels[((src_y + 1) * in_width + src_x) * 3 + channel]);
                let p11 = u16::from(pixels[((src_y + 1) * in_width + (src_x + 1)) * 3 + channel]);
                out.push(((p00 + p10 + p01 + p11 + 2) / 4) as u8);
            }
        }
    }

    out
}

fn transparent_payload_rgba(width: u32, height: u32) -> InMemoryImage<U8> {
    let mut data = Vec::with_capacity(width as usize * height as usize * 4);
    for y in 0..height {
        for x in 0..width {
            data.push(((x * 67 + y * 13) % 256) as u8);
            data.push(((x * 23 + y * 71) % 256) as u8);
            data.push(((x * 97 + y * 19) % 256) as u8);
            data.push(if (x + y) % 3 == 0 { 255 } else { 0 });
        }
    }
    InMemoryImage::from_buffer(width, height, 4, data).unwrap()
}

fn rgb_total_abs_diff(lhs: &InMemoryImage<U8>, rhs: &InMemoryImage<U8>) -> u64 {
    lhs.pixels()
        .iter()
        .zip(rhs.pixels())
        .map(|(&left, &right)| (i16::from(left) - i16::from(right)).unsigned_abs() as u64)
        .sum()
}

fn transparent_rgb_samples(image: &InMemoryImage<U8>) -> Vec<[u8; 3]> {
    image
        .pixels()
        .chunks_exact(4)
        .filter(|pixel| pixel[3] == 0)
        .map(|pixel| [pixel[0], pixel[1], pixel[2]])
        .collect()
}

#[test]
fn round_trip_rgb() {
    let codec = WebpCodec;
    let original = solid_rgb_8x8(128, 64, 200);

    // Encode at quality 100 to minimise lossy artefacts.
    let opts = SaveOptions::default().with_quality(100);
    let encoded = codec.encode_with_options(&original, &opts).unwrap();

    // Verify the encoded bytes start with the WebP magic.
    assert!(
        codec.sniff(&encoded[..12.min(encoded.len())]),
        "encoded output must have WebP magic"
    );

    let decoded: InMemoryImage<U8> = codec.decode(&encoded).unwrap();
    assert_eq!(decoded.width(), 8);
    assert_eq!(decoded.height(), 8);
    assert_eq!(decoded.bands(), 3);

    // WebP lossy at quality=100 can still introduce minor rounding errors.
    // Allow up to +-15 per channel (conservative bound for a solid-colour image).
    let orig_pixels = original.pixels();
    let dec_pixels = decoded.pixels();
    for (i, (&o, &d)) in orig_pixels.iter().zip(dec_pixels.iter()).enumerate() {
        let diff = (o as i32 - d as i32).abs();
        assert!(
            diff <= 15,
            "pixel {i}: original={o}, decoded={d}, diff={diff} > 15"
        );
    }
}

#[test]
fn decode_with_shrink_factor_two_uses_native_scaled_dimensions() {
    use std::num::NonZeroU8;

    let codec = WebpCodec;
    let original = alternating_rgb(8, 8);
    let encoded = codec
        .encode_with_options(&original, &SaveOptions::default().lossless())
        .unwrap();

    let shrunk = codec
        .decode_with_options::<U8>(
            &encoded,
            &LoadOptions::default().with_shrink(NonZeroU8::new(2).unwrap()),
        )
        .unwrap();

    assert_eq!((shrunk.width(), shrunk.height()), (4, 4));
    assert_eq!(shrunk.bands(), 3);
    assert_eq!(shrunk.pixels().len(), 4 * 4 * 3);

    let nearest = rgb_nearest_downsample_2x(&original);
    assert_ne!(
        shrunk.pixels(),
        nearest.as_slice(),
        "native shrink must not collapse to nearest-neighbour sampling"
    );

    let box_filtered = rgb_box_downsample_2x(&original);
    for (index, (&actual, &expected)) in shrunk.pixels().iter().zip(box_filtered.iter()).enumerate()
    {
        let diff = (i16::from(actual) - i16::from(expected)).unsigned_abs();
        assert!(
            diff <= 1,
            "scaled sample {index}: actual={actual}, expected≈{expected}, diff={diff}"
        );
    }
}

#[test]
fn shrink_on_load_plan_uses_native_backend_for_static_images() {
    for factor in [2, 4, 8] {
        let plan = webp_shrink_on_load_plan(factor);
        assert_eq!(plan.factor(), factor);
        assert_eq!(
            plan.backend(),
            ShrinkOnLoadBackend::WebpDecoderConfigScaling
        );
    }
}

#[test]
fn animated_shrink_plan_uses_native_backend() {
    let plan = webp_anim_shrink_on_load_plan(2);

    assert_eq!(plan.factor(), 2);
    assert_eq!(
        plan.backend(),
        ShrinkOnLoadBackend::WebpDemuxFragmentScaling
    );
}

#[test]
fn unsupported_shrink_factor_is_not_sent_to_native_decoder() {
    let plan = webp_shrink_on_load_plan(3);

    assert_eq!(plan.factor(), 1);
    assert_eq!(
        plan.backend(),
        ShrinkOnLoadBackend::WebpDecoderConfigScaling
    );
}

#[test]
fn lossless_round_trip() {
    let codec = WebpCodec;
    let original = solid_rgb_8x8(42, 137, 255);

    let opts = SaveOptions::default().lossless();
    let encoded = codec.encode_with_options(&original, &opts).unwrap();
    let decoded: InMemoryImage<U8> = codec.decode(&encoded).unwrap();

    assert_eq!(decoded.width(), 8);
    assert_eq!(decoded.height(), 8);
    assert_eq!(decoded.bands(), 3);

    // Lossless encoding must preserve every pixel exactly.
    assert_eq!(
        original.pixels(),
        decoded.pixels(),
        "lossless round-trip must be pixel-perfect"
    );
    assert_eq!(
        decoded.metadata().interpretation,
        Some(Interpretation::Srgb)
    );
}

#[test]
fn audit_reference_sample_fixture_decode_matches_imagemagick() {
    let fixture = std::fs::read("tests/fixtures/images/sample.webp").unwrap();
    let decoded: InMemoryImage<U8> = WebpCodec.decode(&fixture).unwrap();

    assert_eq!(
        (decoded.width(), decoded.height(), decoded.bands()),
        (290, 442, 3)
    );
    assert_eq!(
        decoded.metadata().interpretation,
        Some(Interpretation::Srgb)
    );
    assert_eq!(crc32(decoded.pixels()), 0x8451_1B12);
}

#[test]
fn audit_reference_large_fixture_decode_matches_imagemagick() {
    let fixture = std::fs::read("tests/fixtures/images/bench_8192x8192.webp").unwrap();
    let decoded: InMemoryImage<U8> = WebpCodec.decode(&fixture).unwrap();

    assert_eq!(
        (decoded.width(), decoded.height(), decoded.bands()),
        (8192, 8192, 3)
    );
    assert_eq!(
        decoded.metadata().interpretation,
        Some(Interpretation::Srgb)
    );
    assert_eq!(crc32(decoded.pixels()), 0x8FB8_B6CE);
}

#[test]
fn audit_roundtrip_tiny_2x2_lossless_rgba_is_exact() {
    let original = InMemoryImage::<U8>::from_buffer(
        2,
        2,
        4,
        vec![
            12, 34, 56, 200, 78, 90, 123, 180, 145, 167, 189, 160, 210, 222, 234, 140,
        ],
    )
    .unwrap();
    let encoded = WebpCodec
        .encode_with_options(&original, &SaveOptions::default().lossless())
        .unwrap();
    let decoded: InMemoryImage<U8> = WebpCodec.decode(&encoded).unwrap();

    assert_eq!(
        (decoded.width(), decoded.height(), decoded.bands()),
        (2, 2, 4)
    );
    assert_eq!(decoded.pixels(), original.pixels());
}

#[test]
fn encode_default_quality_matches_libvips_default() {
    let codec = WebpCodec;
    let original = patterned_rgb(16, 16);

    let default_encoded = codec.encode(&original).unwrap();
    let explicit_webp_default = codec
        .encode_with_webp_options(&original, &WebpEncodeOptions::default())
        .unwrap();
    let explicit_default = codec
        .encode_with_options(
            &original,
            &SaveOptions::default().with_quality(75).with_method(4),
        )
        .unwrap();

    assert_eq!(
        default_encoded, explicit_webp_default,
        "WebP encode() must use WebpEncodeOptions defaults"
    );
    assert_eq!(
        default_encoded, explicit_default,
        "WebP encode() must default to libvips parity: quality=75, method=4"
    );
}

#[test]
fn webp_encode_options_quality_90_round_trip_preserves_dimensions() {
    let codec = WebpCodec;
    let original = patterned_rgb(37, 19);
    let opts = WebpEncodeOptions {
        quality: 90,
        ..WebpEncodeOptions::default()
    };

    let encoded = codec.encode_with_webp_options(&original, &opts).unwrap();
    let decoded: InMemoryImage<U8> = codec.decode(&encoded).unwrap();

    assert_eq!(decoded.width(), original.width());
    assert_eq!(decoded.height(), original.height());
    assert_eq!(decoded.bands(), original.bands());
}

#[test]
fn webp_encode_options_bridge_matches_save_options() {
    let codec = WebpCodec;
    let original = patterned_rgb(24, 24);
    let webp_opts = WebpEncodeOptions::new(90, 2, false);

    let via_webp_options = codec
        .encode_with_webp_options(&original, &webp_opts)
        .unwrap();
    let via_trait_options = codec
        .encode_with_options(&original, &SaveOptions::from(webp_opts))
        .unwrap();

    assert_eq!(via_webp_options, via_trait_options);
}

#[test]
fn lossless_round_trip_preserves_xmp_metadata() {
    let codec = WebpCodec;
    let original = solid_rgb_8x8(10, 20, 30).with_metadata(ImageMetadata {
        xmp: Some(br#"<x:xmpmeta><rdf:RDF/></x:xmpmeta>"#.to_vec()),
        ..ImageMetadata::default()
    });

    let encoded = codec
        .encode_with_options(&original, &SaveOptions::default().lossless())
        .unwrap();
    let decoded: InMemoryImage<U8> = codec.decode(&encoded).unwrap();

    assert_eq!(decoded.metadata().xmp, original.metadata().xmp);
}

#[cfg(all(feature = "icc", feature = "_integration"))]
#[test]
fn lossless_round_trip_preserves_icc_and_xmp_metadata() {
    let codec = WebpCodec;
    let original = solid_rgb_8x8(10, 20, 30).with_metadata(ImageMetadata {
        icc_profile: Some(profile_load("srgb").expect("load srgb profile")),
        xmp: Some(br#"<x:xmpmeta><rdf:RDF/></x:xmpmeta>"#.to_vec()),
        ..ImageMetadata::default()
    });

    let encoded = codec
        .encode_with_options(&original, &SaveOptions::default().lossless())
        .unwrap();
    let decoded: InMemoryImage<U8> = codec.decode(&encoded).unwrap();

    assert_eq!(
        decoded.metadata().icc_profile,
        original.metadata().icc_profile
    );
    assert_eq!(decoded.metadata().xmp, original.metadata().xmp);
}

#[test]
fn lossless_mode_is_distinct_from_quality_100() {
    let codec = WebpCodec;
    let original = patterned_rgb(32, 32);

    let lossy = codec
        .encode_with_options(&original, &SaveOptions::default().with_quality(100))
        .unwrap();
    let lossless = codec
        .encode_with_options(&original, &SaveOptions::default().lossless())
        .unwrap();

    let lossy_decoded: InMemoryImage<U8> = codec.decode(&lossy).unwrap();
    let lossless_decoded: InMemoryImage<U8> = codec.decode(&lossless).unwrap();

    assert_ne!(lossy, lossless, "lossless must not alias lossy quality=100");
    assert_eq!(lossless_decoded.pixels(), original.pixels());
    assert_ne!(
        lossy_decoded.pixels(),
        original.pixels(),
        "quality=100 should remain lossy for patterned content"
    );
}

#[test]
fn exact_alpha_preserves_transparent_rgb_payload() {
    let codec = WebpCodec;
    let original = transparent_payload_rgba(8, 8);

    let default_encoded = codec
        .encode_with_options(&original, &SaveOptions::default().lossless())
        .unwrap();
    let exact_encoded = codec
        .encode_with_options(
            &original,
            &SaveOptions::default().lossless().with_exact_alpha(true),
        )
        .unwrap();

    let default_decoded: InMemoryImage<U8> = codec.decode(&default_encoded).unwrap();
    let exact_decoded: InMemoryImage<U8> = codec.decode(&exact_encoded).unwrap();

    assert_eq!(exact_decoded.pixels(), original.pixels());
    assert_ne!(
        transparent_rgb_samples(&default_decoded),
        transparent_rgb_samples(&original),
        "default lossless encode may rewrite transparent RGB payload"
    );
}

#[test]
fn decode_region_into_preserves_alpha_for_rgba_webp() {
    let codec = WebpCodec;
    let original = transparent_payload_rgba(8, 8);
    let encoded = codec
        .encode_with_options(
            &original,
            &SaveOptions::default().lossless().with_exact_alpha(true),
        )
        .unwrap();
    let region = Region::new(0, 0, 4, 4);
    let decoded = codec.decode::<U8>(&encoded).unwrap();
    let expected = clamped_region_pixels_u8(&decoded, region);
    let mut output = vec![0u8; region.pixel_count() * 4];

    codec
        .decode_region_into::<U8>(&encoded, &LoadOptions::default(), region, &mut output)
        .unwrap();

    assert!(
        expected.chunks_exact(4).any(|px| px[3] == 0),
        "fixture must include transparent pixels in the decoded region"
    );
    assert_eq!(output, expected);
}

#[test]
fn decode_region_into_rejects_overflowing_region_for_static_webp() {
    let image = patterned_rgb(4, 4);
    let encoded = WebpCodec.encode(&image).unwrap();
    let err = WebpCodec
        .decode_region_into::<U8>(
            &encoded,
            &LoadOptions::default(),
            Region::new(0, 0, u32::MAX, u32::MAX),
            &mut [],
        )
        .expect_err("overflowing region must return an error");

    assert!(matches!(
        err,
        ViprsError::ImageTooLarge {
            width: u32::MAX,
            height: u32::MAX,
            bands: 3,
            ..
        }
    ));
}

#[test]
fn webp_method_effort_changes_output() {
    let codec = WebpCodec;
    let original = patterned_rgb(96, 96);

    let fast = codec
        .encode_with_options(
            &original,
            &SaveOptions::default().with_quality(70).with_method(0),
        )
        .unwrap();
    let slow = codec
        .encode_with_options(
            &original,
            &SaveOptions::default().with_quality(70).with_method(6),
        )
        .unwrap();

    assert_ne!(
        fast, slow,
        "different WebP methods must change the bitstream"
    );
    assert!(
        slow.len() <= fast.len(),
        "higher WebP effort should not produce a larger file for patterned content"
    );
}

#[test]
fn near_lossless_level_changes_fidelity() {
    let codec = WebpCodec;
    let original = patterned_rgb(48, 48);

    let strong = codec
        .encode_with_options(&original, &SaveOptions::default().with_near_lossless(20))
        .unwrap();
    let gentle = codec
        .encode_with_options(&original, &SaveOptions::default().with_near_lossless(80))
        .unwrap();

    let strong_decoded: InMemoryImage<U8> = codec.decode(&strong).unwrap();
    let gentle_decoded: InMemoryImage<U8> = codec.decode(&gentle).unwrap();
    let strong_error = rgb_total_abs_diff(&original, &strong_decoded);
    let gentle_error = rgb_total_abs_diff(&original, &gentle_decoded);

    assert_ne!(
        strong, gentle,
        "near-lossless level must affect output bytes"
    );
    assert!(
        gentle_error <= strong_error,
        "higher near-lossless level should preserve more RGB detail"
    );
}

#[test]
fn smart_subsample_reduces_chroma_error() {
    let codec = WebpCodec;
    let original = chroma_stress_rgb(64, 64);

    let plain = codec
        .encode_with_options(&original, &SaveOptions::default().with_quality(25))
        .unwrap();
    let smart = codec
        .encode_with_options(
            &original,
            &SaveOptions::default()
                .with_quality(25)
                .with_smart_subsample(true),
        )
        .unwrap();

    let plain_decoded: InMemoryImage<U8> = codec.decode(&plain).unwrap();
    let smart_decoded: InMemoryImage<U8> = codec.decode(&smart).unwrap();
    let plain_error = rgb_total_abs_diff(&original, &plain_decoded);
    let smart_error = rgb_total_abs_diff(&original, &smart_decoded);

    assert_ne!(plain, smart, "smart subsampling must affect the bitstream");
    assert!(
        smart_error < plain_error,
        "smart subsampling should improve chroma fidelity on saturated edges"
    );
}

// ── error cases ───────────────────────────────────────────────────────────

#[test]
fn encode_rejects_non_u8_format() {
    use viprs_core::format::F32;
    let codec = WebpCodec;
    let image = InMemoryImage::<F32>::from_buffer(2, 2, 3, vec![0.0f32; 12]).unwrap();
    let result = codec.encode(&image);
    assert!(
        matches!(result, Err(ViprsError::Codec(_))),
        "must error on non-U8 format"
    );
}

#[test]
fn decode_rejects_non_u8_format() {
    use viprs_core::format::U16;
    let codec = WebpCodec;
    // Any bytes; the format check happens before any actual decode.
    let result = codec.decode::<U16>(b"RIFF\x00\x00\x00\x00WEBP");
    assert!(
        matches!(result, Err(ViprsError::Codec(_))),
        "must error on non-U8 format"
    );
}

#[test]
fn encode_rejects_unsupported_band_count() {
    let codec = WebpCodec;
    // 2-band image (unsupported).
    let image = InMemoryImage::<U8>::from_buffer(2, 2, 2, vec![0u8; 8]).unwrap();
    let result = codec.encode(&image);
    assert!(
        matches!(result, Err(ViprsError::Codec(_))),
        "must error on 2-band image"
    );
}

#[test]
fn probe_returns_correct_dimensions() {
    let codec = WebpCodec;
    let original = solid_rgb_8x8(100, 150, 200);
    let encoded = codec.encode(&original).unwrap();
    let (w, h, bands) = codec.probe(&encoded).unwrap();
    assert_eq!(w, 8);
    assert_eq!(h, 8);
    assert_eq!(bands, 3);
}

#[test]
fn probe_returns_4_bands_for_rgba() {
    let codec = WebpCodec;
    let pixels: Vec<u8> = (0u8..=255).cycle().take(8 * 8 * 4).collect();
    let original = InMemoryImage::<U8>::from_buffer(8, 8, 4, pixels).unwrap();
    let opts = SaveOptions::default().lossless();
    let encoded = codec.encode_with_options(&original, &opts).unwrap();
    let (w, h, bands) = codec.probe(&encoded).unwrap();
    assert_eq!(w, 8);
    assert_eq!(h, 8);
    assert_eq!(bands, 4, "probe must return 4 bands for RGBA WebP");
}

#[test]
fn probe_matches_bitstream_features() {
    let codec = WebpCodec;
    let encoded = codec
        .encode_with_options(
            &solid_rgb_8x8(10, 20, 30),
            &SaveOptions::default().lossless(),
        )
        .unwrap();
    let features = BitstreamFeatures::new(&encoded).unwrap();
    let (width, height, bands) = codec.probe(&encoded).unwrap();

    assert_eq!(width, features.width());
    assert_eq!(height, features.height());
    assert_eq!(bands, 3);
}

#[test]
fn format_name_is_webp() {
    let codec = WebpCodec;
    assert_eq!(<WebpCodec as ImageDecoder>::format_name(&codec), "webp");
    assert_eq!(<WebpCodec as ImageEncoder>::format_name(&codec), "webp");
}

// ── edge cases ────────────────────────────────────────────────────────────

#[test]
fn decode_empty_slice_returns_codec_error() {
    let codec = WebpCodec;
    // An empty byte slice contains no valid WebP stream; decode must return
    // ViprsError::Codec rather than panicking.
    let result = codec.decode::<U8>(&[]);
    assert!(
        matches!(result, Err(ViprsError::Codec(_))),
        "empty input must return ViprsError::Codec, got: {result:?}"
    );
}

#[test]
fn lossless_round_trip_rgb_8x8() {
    let codec = WebpCodec;
    // Lossless encode of a gradient RGB image must decode pixel-perfectly.
    let pixels: Vec<u8> = (0u8..192).collect(); // 8*8*3
    let original = InMemoryImage::<U8>::from_buffer(8, 8, 3, pixels).unwrap();

    let opts = SaveOptions::default().lossless();
    let encoded = codec.encode_with_options(&original, &opts).unwrap();
    let decoded: InMemoryImage<U8> = codec.decode(&encoded).unwrap();

    assert_eq!(decoded.width(), 8);
    assert_eq!(decoded.height(), 8);
    assert_eq!(decoded.bands(), 3);
    assert_eq!(
        decoded.pixels(),
        original.pixels(),
        "lossless RGB round-trip must be pixel-perfect"
    );
}

#[test]
fn lossless_round_trip_rgba_8x8_encodes_without_error() {
    let codec = WebpCodec;
    let mut pixels = Vec::with_capacity(8 * 8 * 4);
    for i in 0u8..64 {
        pixels.push(i.wrapping_mul(3)); // R
        pixels.push(i.wrapping_mul(5)); // G
        pixels.push(i.wrapping_mul(7)); // B
        pixels.push(200u8); // A
    }
    let original = InMemoryImage::<U8>::from_buffer(8, 8, 4, pixels).unwrap();

    let opts = SaveOptions::default().lossless();
    let encoded = codec.encode_with_options(&original, &opts).unwrap();

    assert!(
        codec.sniff(&encoded[..12.min(encoded.len())]),
        "encoded RGBA output must have WebP magic"
    );

    let decoded: InMemoryImage<U8> = codec.decode(&encoded).unwrap();
    assert_eq!(decoded.width(), 8);
    assert_eq!(decoded.height(), 8);
    assert_eq!(decoded.bands(), 4, "decoded RGBA image must have 4 bands");

    // Lossless RGBA round-trip must preserve every sample exactly,
    // except for fully-transparent pixels where libwebp may alter the RGB
    // channels (see note for WebPEncodeLossless at
    // https://developers.google.com/speed/webp/docs/api#simple_encoding_api).
    let orig = original.pixels();
    let dec = decoded.pixels();
    for i in (0..orig.len()).step_by(4) {
        let alpha = orig[i + 3];
        if alpha == 255 {
            assert_eq!(
                &orig[i..i + 4],
                &dec[i..i + 4],
                "pixel {}: lossless RGBA round-trip must be pixel-perfect for opaque pixels",
                i / 4
            );
        }
    }
}

#[test]
fn encode_u16_returns_codec_error() {
    use viprs_core::format::U16;

    let codec = WebpCodec;
    // WebP is an 8-bit format; U16 must be rejected.
    let data: Vec<u16> = vec![0u16; 8 * 8 * 3];
    let image = InMemoryImage::<U16>::from_buffer(8, 8, 3, data).unwrap();
    let result = codec.encode(&image);
    assert!(
        matches!(result, Err(ViprsError::Codec(_))),
        "U16 encode must return ViprsError::Codec, got: {result:?}"
    );
}

#[test]
fn encode_2_band_u8_returns_codec_error() {
    let codec = WebpCodec;
    // 2 bands is not supported by WebP (only 1, 3, 4).
    let data: Vec<u8> = vec![0u8; 4 * 4 * 2];
    let image = InMemoryImage::<U8>::from_buffer(4, 4, 2, data).unwrap();
    let result = codec.encode(&image);
    assert!(
        matches!(result, Err(ViprsError::Codec(_))),
        "2-band U8 encode must return ViprsError::Codec, got: {result:?}"
    );
}

#[test]
fn sniff_with_less_than_12_bytes_returns_false() {
    let codec = WebpCodec;
    // WebP magic requires at least 12 bytes (RIFF + 4-byte size + WEBP).
    assert!(!codec.sniff(&[]), "empty slice must return false");
    assert!(!codec.sniff(b"RIFF"), "4-byte slice must return false");
    assert!(
        !codec.sniff(b"RIFF\x00\x00\x00\x00WEB"),
        "11-byte slice must return false"
    );
}

#[test]
fn probe_empty_slice_returns_codec_error() {
    let codec = WebpCodec;
    // An empty byte slice contains no valid WebP stream.
    let result = codec.probe(&[]);
    assert!(
        matches!(result, Err(ViprsError::Codec(_))),
        "probe on empty input must return ViprsError::Codec, got: {result:?}"
    );
}

fn solid_rgba(width: u32, height: u32, color: [u8; 4]) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 4);
    for _ in 0..(width * height) {
        pixels.extend_from_slice(&color);
    }
    pixels
}

fn encode_rgba_webp_lossless(pixels: &[u8], width: u32, height: u32) -> Vec<u8> {
    let image = InMemoryImage::<U8>::from_buffer(width, height, 4, pixels.to_vec()).unwrap();
    let opts = SaveOptions::default().lossless();
    WebpCodec.encode_with_options(&image, &opts).unwrap()
}

fn patch_vp8l_dimensions(encoded: &mut [u8], width: u32, height: u32) {
    let chunk = encoded
        .windows(4)
        .position(|window| window == b"VP8L")
        .expect("VP8L chunk must exist in encoded WebP");
    let payload = chunk + 8;
    assert_eq!(
        encoded[payload], 0x2f,
        "VP8L payload must start with signature"
    );

    let packed = (width - 1) | ((height - 1) << 14) | (1 << 28);
    encoded[payload + 1..payload + 5].copy_from_slice(&packed.to_le_bytes());
}

fn build_animated_webp_with_offsets_and_dispose(
    canvas_width: u32,
    canvas_height: u32,
    frames: &[(Vec<u8>, i32, i32, i32, WebPMuxAnimDispose, WebPMuxAnimBlend)],
) -> Vec<u8> {
    // SAFETY: the mux is created via the libwebp constructor, each frame buffer stays alive for the duration of `WebPMuxPushFrame`, and the assembled bytes are copied out before being freed with the matching libwebp allocator.
    unsafe {
        let mux = WebPNewInternal(WEBP_MUX_ABI_VERSION as i32);
        assert!(!mux.is_null(), "WebPNewInternal must create a mux");

        assert_eq!(
            WebPMuxSetCanvasSize(mux, canvas_width as i32, canvas_height as i32),
            WebPMuxError::WEBP_MUX_OK
        );
        let anim_params = WebPMuxAnimParams {
            bgcolor: 0,
            loop_count: 0,
        };
        assert_eq!(
            WebPMuxSetAnimationParams(mux, &anim_params),
            WebPMuxError::WEBP_MUX_OK
        );

        for (frame_bytes, x_offset, y_offset, duration, dispose_method, blend_method) in frames {
            let bitstream = WebPData {
                bytes: frame_bytes.as_ptr(),
                size: frame_bytes.len(),
            };
            let frame = WebPMuxFrameInfo {
                bitstream,
                x_offset: *x_offset,
                y_offset: *y_offset,
                duration: *duration,
                id: WebPChunkId::WEBP_CHUNK_ANMF,
                dispose_method: *dispose_method,
                blend_method: *blend_method,
                pad: [0],
            };
            assert_eq!(WebPMuxPushFrame(mux, &frame, 1), WebPMuxError::WEBP_MUX_OK);
        }

        let mut assembled = std::mem::MaybeUninit::<WebPData>::uninit();
        assert_eq!(
            WebPMuxAssemble(mux, assembled.as_mut_ptr()),
            WebPMuxError::WEBP_MUX_OK
        );
        WebPMuxDelete(mux);

        let assembled = assembled.assume_init();
        let bytes = std::slice::from_raw_parts(assembled.bytes, assembled.size).to_vec();
        // SAFETY: `WebPMuxAssemble` allocates `assembled.bytes` with libwebp's
        // allocator; `WebPFree` is the matching deallocator.
        WebPFree(assembled.bytes as *mut c_void);
        bytes
    }
}

#[test]
fn decode_animated_webp_exposes_all_frames() {
    let width = 2;
    let height = 1;
    let mut config = WebPConfig::new().unwrap();
    config.lossless = 1;
    config.alpha_compression = 0;

    let frame0 = solid_rgba(width, height, [255, 0, 0, 255]);
    let frame1 = solid_rgba(width, height, [0, 0, 255, 255]);

    let mut encoder = AnimEncoder::new(width, height, &config);
    encoder.add_frame(AnimFrame::from_rgba(&frame0, width, height, 0));
    encoder.add_frame(AnimFrame::from_rgba(&frame1, width, height, 100));
    let encoded = encoder.try_encode().unwrap();

    let decoded = WebpCodec.decode::<U8>(&encoded).unwrap();
    assert_eq!(decoded.metadata().n_pages, Some(2));
    assert_eq!(decoded.metadata().page_height, Some(1));

    let frames = decoded.frames().expect("animated WebP must expose frames");
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].pixels(), frame0.as_slice());
    assert_eq!(frames[1].pixels(), frame1.as_slice());
}

#[test]
fn eager_static_decode_rejects_guarded_allocation_before_vec_reserve() {
    let _guard = WebpScratchAllocationLimitGuard::new(128);
    let mut encoded = encode_rgba_webp_lossless(&solid_rgba(2, 2, [1, 2, 3, 255]), 2, 2);
    patch_vp8l_dimensions(&mut encoded, 6, 6);

    let err = decode_static_webp_pixels(&encoded, &LoadOptions::default())
        .expect_err("guarded eager static decode must reject oversized buffers");

    assert!(matches!(
        err,
        ViprsError::ImageTooLarge {
            width: 6,
            height: 6,
            bands: 4,
            ..
        }
    ));
}

#[test]
fn decode_animated_webp_with_shrink_factor_two_preserves_shrunk_frame_dimensions() {
    use std::num::NonZeroU8;

    let width = 4;
    let height = 2;
    let mut config = WebPConfig::new().unwrap();
    config.lossless = 1;
    config.alpha_compression = 0;

    let frame0 = solid_rgba(width, height, [255, 0, 0, 255]);
    let frame1 = solid_rgba(width, height, [0, 0, 255, 255]);

    let mut encoder = AnimEncoder::new(width, height, &config);
    encoder.add_frame(AnimFrame::from_rgba(&frame0, width, height, 0));
    encoder.add_frame(AnimFrame::from_rgba(&frame1, width, height, 100));
    let encoded = encoder.try_encode().unwrap();

    let decoded = WebpCodec
        .decode_with_options::<U8>(
            &encoded,
            &LoadOptions::default().with_shrink(NonZeroU8::new(2).unwrap()),
        )
        .unwrap();
    assert_eq!(decoded.width(), 2);
    assert_eq!(decoded.height(), 1);
    assert_eq!(decoded.metadata().n_pages, Some(2));
    assert_eq!(decoded.metadata().page_height, Some(1));

    let frames = decoded.frames().expect("animated WebP must expose frames");
    assert_eq!(frames.len(), 2);
    assert_eq!((frames[0].width(), frames[0].height()), (2, 1));
    assert_eq!((frames[1].width(), frames[1].height()), (2, 1));
    assert_eq!(frames[0].pixels(), solid_rgba(2, 1, [255, 0, 0, 255]));
    assert_eq!(frames[1].pixels(), solid_rgba(2, 1, [0, 0, 255, 255]));
}

#[test]
fn eager_animated_decode_rejects_guarded_canvas_allocation_before_vec_reserve() {
    let _guard = WebpScratchAllocationLimitGuard::new(128);
    let frame = encode_rgba_webp_lossless(&solid_rgba(2, 2, [255, 0, 0, 255]), 2, 2);
    let encoded = build_animated_webp_with_offsets_and_dispose(
        6,
        6,
        &[(
            frame,
            0,
            0,
            100,
            WebPMuxAnimDispose::WEBP_MUX_DISPOSE_NONE,
            WebPMuxAnimBlend::WEBP_MUX_NO_BLEND,
        )],
    );

    let err = WebpCodec
        .decode_with_options::<U8>(&encoded, &LoadOptions::default())
        .expect_err("guarded eager animated decode must reject oversized canvases");

    assert!(matches!(
        err,
        ViprsError::ImageTooLarge {
            width: 6,
            height: 6,
            bands: 4,
            ..
        }
    ));
}

#[test]
fn eager_animated_decode_rejects_total_frame_accumulation_over_limit() {
    let _guard = WebpScratchAllocationLimitGuard::new(128);
    let red = encode_rgba_webp_lossless(&solid_rgba(4, 4, [255, 0, 0, 255]), 4, 4);
    let green = encode_rgba_webp_lossless(&solid_rgba(4, 4, [0, 255, 0, 255]), 4, 4);
    let blue = encode_rgba_webp_lossless(&solid_rgba(4, 4, [0, 0, 255, 255]), 4, 4);
    let encoded = build_animated_webp_with_offsets_and_dispose(
        4,
        4,
        &[
            (
                red,
                0,
                0,
                100,
                WebPMuxAnimDispose::WEBP_MUX_DISPOSE_NONE,
                WebPMuxAnimBlend::WEBP_MUX_NO_BLEND,
            ),
            (
                green,
                0,
                0,
                100,
                WebPMuxAnimDispose::WEBP_MUX_DISPOSE_NONE,
                WebPMuxAnimBlend::WEBP_MUX_NO_BLEND,
            ),
            (
                blue,
                0,
                0,
                100,
                WebPMuxAnimDispose::WEBP_MUX_DISPOSE_NONE,
                WebPMuxAnimBlend::WEBP_MUX_NO_BLEND,
            ),
        ],
    );

    let err = WebpCodec
        .decode_with_options::<U8>(&encoded, &LoadOptions::default())
        .expect_err("guarded eager animated decode must reject excessive accumulated frames");

    assert!(matches!(
        err,
        ViprsError::ImageTooLarge {
            width: 4,
            height: 4,
            bands: 4,
            ..
        }
    ));
}

#[test]
fn animated_tile_decoder_matches_eager_final_composited_region_after_shrink() {
    let width = 6;
    let height = 4;
    let mut config = WebPConfig::new().unwrap();
    config.lossless = 1;
    config.alpha_compression = 0;

    let frame0 = solid_rgba(width, height, [255, 0, 0, 255]);
    let frame1 = solid_rgba(width, height, [0, 0, 255, 255]);

    let mut encoder = AnimEncoder::new(width, height, &config);
    encoder.add_frame(AnimFrame::from_rgba(&frame0, width, height, 0));
    encoder.add_frame(AnimFrame::from_rgba(&frame1, width, height, 100));
    let encoded = encoder.try_encode().unwrap();

    let opts = LoadOptions::default().with_shrink(NonZeroU8::new(2).unwrap());
    let eager = WebpCodec
        .decode_with_options::<U8>(&encoded, &opts)
        .unwrap();
    let final_frame = eager
        .frames()
        .and_then(|frames| frames.last())
        .expect("animated WebP must expose the final composed frame");
    let region = Region::new(1, 0, 2, 2);
    let mut output = vec![0u8; region.pixel_count() * 4];

    WebpCodec
        .decode_region_into::<U8>(&encoded, &opts, region, &mut output)
        .unwrap();

    assert_eq!(output, clamped_region_pixels_u8(final_frame, region));
}

#[test]
fn decode_region_into_rejects_overflowing_region_for_animated_webp() {
    let width = 2;
    let height = 2;
    let mut config = WebPConfig::new().unwrap();
    config.lossless = 1;
    let frame0 = solid_rgba(width, height, [255, 0, 0, 255]);
    let frame1 = solid_rgba(width, height, [0, 0, 255, 255]);
    let mut encoder = AnimEncoder::new(width, height, &config);
    encoder.add_frame(AnimFrame::from_rgba(&frame0, width, height, 0));
    encoder.add_frame(AnimFrame::from_rgba(&frame1, width, height, 100));
    let encoded = encoder.try_encode().unwrap();

    let err = WebpCodec
        .decode_region_into::<U8>(
            &encoded,
            &LoadOptions::default(),
            Region::new(0, 0, u32::MAX, u32::MAX),
            &mut [],
        )
        .expect_err("overflowing region must return an error");

    assert!(matches!(
        err,
        ViprsError::ImageTooLarge {
            width: u32::MAX,
            height: u32::MAX,
            bands: 4,
            ..
        }
    ));
}

#[test]
fn animated_tile_decoder_matches_eager_final_composited_region() {
    let red = encode_rgba_webp_lossless(&solid_rgba(4, 4, [255, 0, 0, 255]), 4, 4);
    let blue = encode_rgba_webp_lossless(&solid_rgba(2, 2, [0, 0, 255, 255]), 2, 2);
    let encoded = build_animated_webp_with_offsets_and_dispose(
        4,
        4,
        &[
            (
                red,
                0,
                0,
                100,
                WebPMuxAnimDispose::WEBP_MUX_DISPOSE_NONE,
                WebPMuxAnimBlend::WEBP_MUX_NO_BLEND,
            ),
            (
                blue,
                2,
                0,
                100,
                WebPMuxAnimDispose::WEBP_MUX_DISPOSE_NONE,
                WebPMuxAnimBlend::WEBP_MUX_NO_BLEND,
            ),
        ],
    );

    let eager = WebpCodec.decode::<U8>(&encoded).unwrap();
    let final_frame = eager
        .frames()
        .and_then(|frames| frames.last())
        .expect("animated WebP must expose the final composed frame");
    let region = Region::new(2, 0, 2, 2);
    let mut output = vec![0u8; region.pixel_count() * 4];

    WebpCodec
        .decode_region_into::<U8>(&encoded, &LoadOptions::default(), region, &mut output)
        .unwrap();

    assert_eq!(output, clamped_region_pixels_u8(final_frame, region));
}

#[test]
fn decode_animated_webp_with_offsets_and_dispose_preserves_canvas_composition() {
    let red = encode_rgba_webp_lossless(&solid_rgba(2, 2, [255, 0, 0, 255]), 2, 2);
    let transparent = encode_rgba_webp_lossless(&solid_rgba(2, 2, [0, 0, 0, 0]), 2, 2);
    let blue = encode_rgba_webp_lossless(&solid_rgba(2, 2, [0, 0, 255, 255]), 2, 2);

    let encoded = build_animated_webp_with_offsets_and_dispose(
        4,
        4,
        &[
            (
                red,
                2,
                0,
                100,
                WebPMuxAnimDispose::WEBP_MUX_DISPOSE_BACKGROUND,
                WebPMuxAnimBlend::WEBP_MUX_NO_BLEND,
            ),
            (
                transparent,
                2,
                0,
                100,
                WebPMuxAnimDispose::WEBP_MUX_DISPOSE_NONE,
                WebPMuxAnimBlend::WEBP_MUX_BLEND,
            ),
            (
                blue,
                0,
                2,
                100,
                WebPMuxAnimDispose::WEBP_MUX_DISPOSE_NONE,
                WebPMuxAnimBlend::WEBP_MUX_NO_BLEND,
            ),
        ],
    );

    let decoded = WebpCodec.decode::<U8>(&encoded).unwrap();
    assert_eq!(decoded.width(), 4);
    assert_eq!(decoded.height(), 4);
    assert_eq!(decoded.metadata().n_pages, Some(3));
    assert_eq!(decoded.metadata().page_height, Some(4));

    let frames = decoded.frames().expect("animated WebP must expose frames");
    assert_eq!(frames.len(), 3);
    assert!(
        frames[0].pixels()[0..8]
            .chunks_exact(4)
            .all(|px| px[3] == 0)
    );
    assert_eq!(
        &frames[0].pixels()[8..16],
        &[255, 0, 0, 255, 255, 0, 0, 255]
    );
    assert!(
        frames[0].pixels()[16..24]
            .chunks_exact(4)
            .all(|px| px[3] == 0)
    );
    assert_eq!(
        &frames[0].pixels()[24..32],
        &[255, 0, 0, 255, 255, 0, 0, 255]
    );
    assert!(frames[1].pixels().chunks_exact(4).all(|px| px[3] == 0));
    assert_eq!(
        &frames[2].pixels()[32..40],
        &[0, 0, 255, 255, 0, 0, 255, 255]
    );
    assert!(
        frames[2].pixels()[40..48]
            .chunks_exact(4)
            .all(|px| px[3] == 0)
    );
    assert_eq!(
        &frames[2].pixels()[48..56],
        &[0, 0, 255, 255, 0, 0, 255, 255]
    );
    assert!(
        frames[2].pixels()[56..64]
            .chunks_exact(4)
            .all(|px| px[3] == 0)
    );
}

#[test]
fn decode_animated_webp_with_partial_frames_disables_shrink_to_match_libvips() {
    use std::num::NonZeroU8;

    let red = encode_rgba_webp_lossless(&solid_rgba(2, 2, [255, 0, 0, 255]), 2, 2);
    let transparent = encode_rgba_webp_lossless(&solid_rgba(2, 2, [0, 0, 0, 0]), 2, 2);
    let blue = encode_rgba_webp_lossless(&solid_rgba(2, 2, [0, 0, 255, 255]), 2, 2);

    let encoded = build_animated_webp_with_offsets_and_dispose(
        4,
        4,
        &[
            (
                red,
                2,
                0,
                100,
                WebPMuxAnimDispose::WEBP_MUX_DISPOSE_BACKGROUND,
                WebPMuxAnimBlend::WEBP_MUX_NO_BLEND,
            ),
            (
                transparent,
                2,
                0,
                100,
                WebPMuxAnimDispose::WEBP_MUX_DISPOSE_NONE,
                WebPMuxAnimBlend::WEBP_MUX_BLEND,
            ),
            (
                blue,
                0,
                2,
                100,
                WebPMuxAnimDispose::WEBP_MUX_DISPOSE_NONE,
                WebPMuxAnimBlend::WEBP_MUX_NO_BLEND,
            ),
        ],
    );

    let baseline = WebpCodec.decode::<U8>(&encoded).unwrap();
    let decoded = WebpCodec
        .decode_with_options::<U8>(
            &encoded,
            &LoadOptions::default().with_shrink(NonZeroU8::new(2).unwrap()),
        )
        .unwrap();

    assert_eq!((decoded.width(), decoded.height()), (4, 4));
    assert_eq!(decoded.metadata(), baseline.metadata());

    let baseline_frames = baseline
        .frames()
        .expect("baseline animation must expose frames");
    let frames = decoded.frames().expect("animated WebP must expose frames");
    assert_eq!(frames.len(), baseline_frames.len());
    for (baseline_frame, frame) in baseline_frames.iter().zip(frames.iter()) {
        assert_eq!((frame.width(), frame.height()), (4, 4));
        assert_eq!(frame.pixels(), baseline_frame.pixels());
    }
}

#[test]
fn animated_tile_decoder_partial_frames_avoids_eager_full_decode_when_shrink_falls_back_to_one() {
    let red = encode_rgba_webp_lossless(&solid_rgba(2, 2, [255, 0, 0, 255]), 2, 2);
    let blue = encode_rgba_webp_lossless(&solid_rgba(2, 2, [0, 0, 255, 255]), 2, 2);
    let encoded = build_animated_webp_with_offsets_and_dispose(
        4,
        4,
        &[
            (
                red,
                2,
                0,
                100,
                WebPMuxAnimDispose::WEBP_MUX_DISPOSE_BACKGROUND,
                WebPMuxAnimBlend::WEBP_MUX_NO_BLEND,
            ),
            (
                blue,
                0,
                2,
                100,
                WebPMuxAnimDispose::WEBP_MUX_DISPOSE_NONE,
                WebPMuxAnimBlend::WEBP_MUX_NO_BLEND,
            ),
        ],
    );

    let opts = LoadOptions::default().with_shrink(NonZeroU8::new(2).unwrap());
    let baseline = WebpCodec
        .decode_with_options::<U8>(&encoded, &opts)
        .unwrap();
    let final_frame = baseline
        .frames()
        .and_then(|frames| frames.last())
        .expect("animated WebP must expose the final composed frame");
    let region = Region::new(2, 0, 2, 1);
    let mut output = vec![0u8; region.pixel_count() * 4];
    let _guard = WebpScratchAllocationLimitGuard::new(96);

    let eager_err = WebpCodec
        .decode_with_options::<U8>(&encoded, &opts)
        .expect_err("guarded eager animated decode must exceed the bounded-memory limit");
    assert!(matches!(eager_err, ViprsError::ImageTooLarge { .. }));

    WebpCodec
        .decode_region_into::<U8>(&encoded, &opts, region, &mut output)
        .unwrap();

    assert_eq!(output, clamped_region_pixels_u8(final_frame, region));
}

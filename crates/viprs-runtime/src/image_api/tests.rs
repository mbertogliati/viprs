use super::*;
#[cfg(feature = "png")]
use std::io::Cursor;
#[cfg(any(feature = "jpeg", feature = "png", feature = "webp"))]
use std::{fs, path::PathBuf};
use std::{
    io::{self, ErrorKind},
    panic::{AssertUnwindSafe, catch_unwind},
};

#[cfg(feature = "jpeg")]
use crate::adapters::codecs::JpegCodec;
#[cfg(feature = "png")]
use crate::adapters::codecs::PngCodec;
#[cfg(feature = "webp")]
use crate::adapters::codecs::WebpCodec;
use crate::adapters::{
    pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
    sources::memory::MemorySource,
};
use crate::domain::colorspace::ColorspaceId;
#[cfg(feature = "png")]
use crate::domain::limits::ResourceLimits;
#[cfg(any(feature = "jpeg", feature = "png"))]
use crate::domain::ops::point;
#[cfg(any(feature = "jpeg", feature = "png", feature = "webp"))]
use crate::ports::codec::ImageDecoder;

#[cfg(feature = "png")]
#[test]
fn image_api_png_round_trips_losslessly() {
    let input = Image::<U8>::from_buffer(2, 1, 1, vec![0, 255]).unwrap();
    let encoded = PngCodec::default().encode(&input).unwrap();

    let output = ImageApi::from_bytes(&encoded)
        .unwrap()
        .apply(point::Invert)
        .unwrap()
        .encode_png()
        .unwrap();

    let decoded = PngCodec::default().decode::<U8>(&output).unwrap();
    assert_eq!(decoded.pixels(), &[255, 0]);
}

#[cfg(feature = "jpeg")]
#[test]
fn image_api_jpeg_smoke_round_trip() {
    let input = Image::<U8>::from_buffer(2, 1, 3, vec![10, 20, 30, 200, 150, 100]).unwrap();
    let encoded = JpegCodec
        .encode_with_options(
            &input,
            &SaveOptions {
                quality: Some(90),
                ..SaveOptions::default()
            },
        )
        .unwrap();

    let output = ImageApi::from_bytes(&encoded)
        .unwrap()
        .linear(1.0, 0.0)
        .unwrap()
        .encode_jpeg(80)
        .unwrap();

    let decoded = JpegCodec.decode::<U8>(&output).unwrap();
    assert_eq!(decoded.width(), 2);
    assert_eq!(decoded.height(), 1);
    assert_eq!(decoded.bands(), 3);
}

#[cfg(feature = "jpeg")]
#[test]
fn image_api_jpeg_streaming_encode_to_writer_produces_decodable_output() {
    let input = Image::<U8>::from_buffer(8, 4, 3, (0..96).collect()).unwrap();
    let encoded = JpegCodec
        .encode_with_options(
            &input,
            &SaveOptions {
                quality: Some(90),
                ..SaveOptions::default()
            },
        )
        .unwrap();

    let mut streamed = Vec::new();
    ImageApi::from_bytes(&encoded)
        .unwrap()
        .linear(1.0, 0.0)
        .unwrap()
        .encode_jpeg_to(&mut streamed, 80)
        .unwrap();

    let decoded = JpegCodec.decode::<U8>(&streamed).unwrap();
    assert_eq!(decoded.width(), 8);
    assert_eq!(decoded.height(), 4);
    assert_eq!(decoded.bands(), 3);
}

#[cfg(feature = "png")]
#[test]
fn image_api_png_streaming_encode_to_writer_round_trips_losslessly() {
    let input = Image::<U8>::from_buffer(3, 1, 1, vec![0, 64, 255]).unwrap();
    let encoded = PngCodec::default().encode(&input).unwrap();
    let mut streamed = Vec::new();

    ImageApi::from_bytes(&encoded)
        .unwrap()
        .invert()
        .unwrap()
        .encode_png_to(&mut streamed)
        .unwrap();

    let decoded = PngCodec::default().decode::<U8>(&streamed).unwrap();
    assert_eq!(decoded.pixels(), &[255, 191, 0]);
}

#[cfg(feature = "jpeg")]
#[test]
fn image_api_from_bytes_defers_jpeg_decode_until_execution() {
    let input = Image::<U8>::from_buffer(8, 8, 3, vec![96; 8 * 8 * 3]).unwrap();
    let encoded = JpegCodec
        .encode_with_options(
            &input,
            &SaveOptions {
                quality: Some(90),
                ..SaveOptions::default()
            },
        )
        .unwrap();
    let truncated = truncate_jpeg_scan_data(&encoded);

    let api = ImageApi::from_bytes(&truncated)
        .unwrap()
        .thumbnail(2)
        .unwrap();

    assert!(api.encode_jpeg(80).is_err());
}

#[cfg(feature = "jpeg")]
#[test]
fn image_api_open_defers_jpeg_decode_until_execution() {
    let input = Image::<U8>::from_buffer(8, 8, 3, vec![64; 8 * 8 * 3]).unwrap();
    let encoded = JpegCodec
        .encode_with_options(
            &input,
            &SaveOptions {
                quality: Some(90),
                ..SaveOptions::default()
            },
        )
        .unwrap();
    let truncated = truncate_jpeg_scan_data(&encoded);
    let path = write_test_image("image-api-open-lazy.jpg", &truncated);

    let api = ImageApi::open(&path).unwrap().thumbnail(2).unwrap();
    assert!(api.encode_jpeg(80).is_err());

    fs::remove_file(path).unwrap();
}

#[test]
fn image_api_rejects_unknown_headers() {
    let err = match ImageApi::from_bytes(b"not-an-image") {
        Ok(_) => panic!("expected invalid image header to be rejected"),
        Err(err) => err,
    };
    assert!(
        matches!(err, ViprsError::Codec(message) if message.contains("unsupported input format"))
    );
}

#[cfg(feature = "png")]
#[test]
fn image_api_apply_accepts_generic_pipeline_ops() {
    let input = Image::<U8>::from_buffer(1, 1, 1, vec![32]).unwrap();
    let encoded = PngCodec::default().encode(&input).unwrap();

    let output = ImageApi::from_bytes(&encoded)
        .unwrap()
        .apply(point::Linear::new(2.0, 10.0))
        .unwrap()
        .encode_png()
        .unwrap();

    let decoded = PngCodec::default().decode::<U8>(&output).unwrap();
    assert_eq!(decoded.pixels(), &[74]);
}

#[cfg(feature = "png")]
#[test]
fn image_api_from_reader_decodes_png_stream() {
    let input = Image::<U8>::from_buffer(2, 1, 1, vec![10, 20]).unwrap();
    let encoded = PngCodec::default().encode(&input).unwrap();

    let output = ImageApi::from_reader(Cursor::new(encoded))
        .unwrap()
        .invert()
        .unwrap()
        .encode_png()
        .unwrap();

    let decoded = PngCodec::default().decode::<U8>(&output).unwrap();
    assert_eq!(decoded.pixels(), &[245, 235]);
}

#[cfg(feature = "png")]
#[test]
fn image_api_open_loads_from_path() {
    let input = Image::<U8>::from_buffer(2, 1, 1, vec![0, 255]).unwrap();
    let encoded = PngCodec::default().encode(&input).unwrap();
    let path = write_test_image("image-api-open.png", &encoded);

    let output = ImageApi::open(&path)
        .unwrap()
        .invert()
        .unwrap()
        .encode_png()
        .unwrap();

    let decoded = PngCodec::default().decode::<U8>(&output).unwrap();
    assert_eq!(decoded.pixels(), &[255, 0]);

    fs::remove_file(path).unwrap();
}

#[cfg(feature = "png")]
#[test]
fn image_api_from_bytes_with_limits_rejects_oversized_decode() {
    let input = Image::<U8>::from_buffer(2, 1, 1, vec![0, 255]).unwrap();
    let encoded = PngCodec::default().encode(&input).unwrap();
    let limits = DecodeLimits {
        max_width: 1,
        ..DecodeLimits::default()
    };

    let err = ImageApi::from_bytes_with_limits(&encoded, limits).unwrap_err();
    assert!(matches!(
        err,
        ViprsError::ImageTooLarge {
            width: 2,
            height: 1,
            bands: 1,
            ..
        }
    ));
}

#[cfg(feature = "png")]
#[test]
fn image_api_with_limits_rejects_decode_pixels() {
    let input = Image::<U8>::from_buffer(2, 1, 1, vec![0, 255]).unwrap();
    let encoded = PngCodec::default().encode(&input).unwrap();
    let limits = ResourceLimits::new(1, 1024, 1);

    let err = ImageApi::with_limits(limits)
        .from_bytes(&encoded)
        .unwrap_err();
    assert!(matches!(
        err,
        ViprsError::ImageTooLarge {
            width: 2,
            height: 1,
            bands: 1,
            details: "pixel count exceeds decode limits",
            ..
        }
    ));
}

#[cfg(feature = "png")]
#[test]
fn image_api_with_limits_rejects_output_bytes() {
    let input = Image::<U8>::from_buffer(2, 2, 1, vec![255; 4]).unwrap();
    let encoded = PngCodec::default().encode(&input).unwrap();
    // Keep decode permissive enough for the 2×2 source so the failure is reported by
    // output validation instead of the loader.
    let limits = ResourceLimits::new(16, 3, 1).with_max_decode_bytes(4);

    let err = ImageApi::with_limits(limits)
        .from_bytes(&encoded)
        .unwrap()
        .encode_png()
        .unwrap_err();
    assert!(matches!(
        err,
        ViprsError::ImageTooLarge {
            width: 2,
            height: 2,
            bands: 1,
            details: "output byte size exceeds resource limits",
            ..
        }
    ));
}

#[cfg(feature = "png")]
#[test]
fn image_api_thumbnail_encodes_png_without_manual_pipeline_plumbing() {
    let input = Image::<U8>::from_buffer(4, 2, 1, vec![0, 64, 128, 255, 255, 128, 64, 0]).unwrap();
    let encoded = PngCodec::default().encode(&input).unwrap();

    let output = ImageApi::from_bytes(&encoded)
        .unwrap()
        .thumbnail(2)
        .unwrap()
        .encode_png()
        .unwrap();

    let decoded = PngCodec::default().decode::<U8>(&output).unwrap();
    assert_eq!(
        (decoded.width(), decoded.height(), decoded.bands()),
        (2, 1, 1)
    );
}

#[cfg(all(feature = "png", feature = "icc"))]
#[test]
fn image_api_normalize_to_srgb_matches_encode_time_normalization() {
    use crate::domain::{
        image::{ImageMetadata, Interpretation},
        ops::colour::profile_load,
    };

    let source = Image::<U8>::from_buffer(2, 1, 2, vec![32, 7, 160, 9])
        .unwrap()
        .with_metadata(ImageMetadata {
            interpretation: Some(Interpretation::BW),
            icc_profile: Some(profile_load("gray").expect("load gray ICC profile")),
            ..ImageMetadata::default()
        });
    let encoded = PngCodec::default().encode(&source).unwrap();

    let explicit = ImageApi::from_bytes(&encoded)
        .unwrap()
        .normalize_to_srgb()
        .unwrap()
        .encode_png()
        .unwrap();
    let implicit = ImageApi::from_bytes(&encoded)
        .unwrap()
        .encode_png()
        .unwrap();

    assert_eq!(explicit, implicit);
}

#[cfg(all(feature = "png", feature = "icc"))]
#[test]
fn image_api_thumbnail_can_auto_normalize_to_srgb() {
    use crate::domain::{
        image::{ImageMetadata, Interpretation},
        ops::colour::profile_load,
    };

    let source = Image::<U8>::from_buffer(
        4,
        2,
        2,
        vec![
            0, 9, 64, 10, 128, 11, 255, 12, 255, 13, 128, 14, 64, 15, 0, 16,
        ],
    )
    .unwrap()
    .with_metadata(ImageMetadata {
        interpretation: Some(Interpretation::BW),
        icc_profile: Some(profile_load("gray").expect("load gray ICC profile")),
        ..ImageMetadata::default()
    });
    let encoded = PngCodec::default().encode(&source).unwrap();

    let explicit = ImageApi::from_bytes(&encoded)
        .unwrap()
        .thumbnail_with_options(
            2,
            ImageApiThumbnailOptions::default().with_auto_normalize_to_srgb(true),
        )
        .unwrap()
        .encode_png()
        .unwrap();
    let manual = ImageApi::from_bytes(&encoded)
        .unwrap()
        .thumbnail(2)
        .unwrap()
        .normalize_to_srgb()
        .unwrap()
        .encode_png()
        .unwrap();

    assert_eq!(explicit, manual);
}

#[test]
fn image_api_sharpen_matches_pipeline_builder_defaults() {
    let pixels = vec![
        12, 30, 48, 64, 82, 100, 118, 136, 154, 45, 63, 81, 97, 115, 133, 149, 167, 185, 78, 96,
        114, 130, 148, 166, 182, 200, 218,
    ];

    let actual = ImageApi {
        builder: PipelineBuilder::from_source(
            MemorySource::<U8>::new(3, 3, 3, pixels.clone()).unwrap(),
        )
        .with_colorspace(ColorspaceId::SRgb),
        resource_limits: None,
    }
    .sharpen()
    .unwrap()
    .builder
    .build()
    .unwrap()
    .run_to_image::<U8, _>(&RayonScheduler::new(1).unwrap())
    .unwrap();

    let expected = PipelineBuilder::from_source(MemorySource::<U8>::new(3, 3, 3, pixels).unwrap())
        .with_colorspace(ColorspaceId::SRgb)
        .sharpen(0.5, 2.0, 10.0, 20.0, 0.0, 3.0)
        .unwrap()
        .build()
        .unwrap()
        .run_to_image::<U8, _>(&RayonScheduler::new(1).unwrap())
        .unwrap();

    assert_eq!(
        (actual.width(), actual.height()),
        (expected.width(), expected.height())
    );
    assert_eq!(actual.bands(), expected.bands());
    assert_eq!(actual.pixels(), expected.pixels());
}

#[test]
fn image_api_sharpen_with_forwards_custom_parameters() {
    let pixels = vec![
        10, 20, 30, 40, 50, 60, 70, 80, 90, 15, 25, 35, 45, 55, 65, 75, 85, 95, 20, 30, 40, 50, 60,
        70, 80, 90, 100,
    ];
    let sigma = 1.25;
    let x1 = 1.5;
    let y2 = 12.0;
    let y3 = 24.0;
    let m1 = 0.5;
    let m2 = 2.5;

    let actual = ImageApi {
        builder: PipelineBuilder::from_source(
            MemorySource::<U8>::new(3, 3, 3, pixels.clone()).unwrap(),
        )
        .with_colorspace(ColorspaceId::SRgb),
        resource_limits: None,
    }
    .sharpen_with(sigma, x1, y2, y3, m1, m2)
    .unwrap()
    .builder
    .build()
    .unwrap()
    .run_to_image::<U8, _>(&RayonScheduler::new(1).unwrap())
    .unwrap();

    let expected = PipelineBuilder::from_source(MemorySource::<U8>::new(3, 3, 3, pixels).unwrap())
        .with_colorspace(ColorspaceId::SRgb)
        .sharpen(sigma, x1, y2, y3, m1, m2)
        .unwrap()
        .build()
        .unwrap()
        .run_to_image::<U8, _>(&RayonScheduler::new(1).unwrap())
        .unwrap();

    assert_eq!(
        (actual.width(), actual.height()),
        (expected.width(), expected.height())
    );
    assert_eq!(actual.bands(), expected.bands());
    assert_eq!(actual.pixels(), expected.pixels());
}

#[cfg(feature = "png")]
#[test]
fn image_api_smartcrop_crops_to_attention_region() {
    let width = 96u32;
    let height = 64u32;
    let crop_width = 24u32;
    let crop_height = 24u32;
    let mut pixels = vec![0u8; width as usize * height as usize * 3];

    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            pixels[idx] = 14 + ((x * 3 + y * 2) % 5) as u8;
            pixels[idx + 1] = 12 + ((x * 5 + y) % 5) as u8;
            pixels[idx + 2] = 10 + ((x + y * 7) % 5) as u8;
        }
    }

    for y in 18..42 {
        for x in 56..82 {
            let idx = (y * width as usize + x) * 3;
            let border = x == 56 || x == 81 || y == 18 || y == 41 || x == 69 || y == 30;
            if border {
                pixels[idx..idx + 3].copy_from_slice(&[8, 8, 8]);
            } else if (x + y) % 3 == 0 {
                pixels[idx..idx + 3].copy_from_slice(&[230, 186, 150]);
            } else {
                pixels[idx..idx + 3].copy_from_slice(&[36, 232, 242]);
            }
        }
    }

    let input = Image::<U8>::from_buffer(width, height, 3, pixels).unwrap();
    let encoded = PngCodec::default().encode(&input).unwrap();

    let output = ImageApi::from_bytes(&encoded)
        .unwrap()
        .smartcrop(crop_width, crop_height)
        .unwrap()
        .encode_png()
        .unwrap();

    let decoded = PngCodec::default().decode::<U8>(&output).unwrap();
    assert_eq!(
        (decoded.width(), decoded.height(), decoded.bands()),
        (crop_width, crop_height, 3)
    );

    let crop = SmartcropOp::<U8>::analyze(&input, crop_width, crop_height);
    assert!(crop.crop_left() <= 82);
    assert!(crop.crop_left() + crop_width > 56);
    assert!(crop.crop_top() <= 42);
    assert!(crop.crop_top() + crop_height > 18);
    let crop_base = ((crop.crop_top() * width + crop.crop_left()) * 3) as usize;
    assert_eq!(
        &decoded.pixels()[0..3],
        &input.pixels()[crop_base..crop_base + 3]
    );
}

#[cfg(feature = "png")]
#[test]
fn image_api_flatten_defaults_to_white_background() {
    let input = Image::<U8>::from_buffer(1, 1, 4, vec![100, 150, 200, 128]).unwrap();
    let encoded = PngCodec::default().encode(&input).unwrap();

    let output = ImageApi::from_bytes(&encoded)
        .unwrap()
        .flatten()
        .unwrap()
        .encode_png()
        .unwrap();

    let decoded = PngCodec::default().decode::<U8>(&output).unwrap();
    assert_eq!(decoded.bands(), 3);
    assert_eq!(decoded.pixels(), &[177, 202, 227]);
}

#[cfg(feature = "png")]
#[test]
fn image_api_flatten_with_uses_custom_background() {
    let input = Image::<U8>::from_buffer(1, 1, 4, vec![100, 150, 200, 128]).unwrap();
    let encoded = PngCodec::default().encode(&input).unwrap();

    let output = ImageApi::from_bytes(&encoded)
        .unwrap()
        .flatten_with(10, 20, 30)
        .unwrap()
        .encode_png()
        .unwrap();

    let decoded = PngCodec::default().decode::<U8>(&output).unwrap();
    assert_eq!(decoded.bands(), 3);
    assert_eq!(decoded.pixels(), &[55, 85, 115]);
}

#[cfg(feature = "png")]
#[test]
fn image_api_premultiply_scales_colour_by_alpha() {
    let input = Image::<U8>::from_buffer(1, 1, 4, vec![128, 64, 32, 128]).unwrap();
    let encoded = PngCodec::default().encode(&input).unwrap();

    let output = ImageApi::from_bytes(&encoded)
        .unwrap()
        .premultiply()
        .unwrap()
        .encode_png()
        .unwrap();

    let decoded = PngCodec::default().decode::<U8>(&output).unwrap();
    assert_eq!(decoded.pixels(), &[64, 32, 16, 128]);
}

#[cfg(feature = "png")]
#[test]
fn image_api_unpremultiply_restores_expected_values() {
    let input = Image::<U8>::from_buffer(1, 1, 4, vec![64, 32, 16, 128]).unwrap();
    let encoded = PngCodec::default().encode(&input).unwrap();

    let output = ImageApi::from_bytes(&encoded)
        .unwrap()
        .unpremultiply()
        .unwrap()
        .encode_png()
        .unwrap();

    let decoded = PngCodec::default().decode::<U8>(&output).unwrap();
    assert_eq!(decoded.pixels(), &[128, 64, 32, 128]);
}

#[cfg(feature = "png")]
#[test]
fn image_api_save_uses_output_path_extension() {
    let input = Image::<U8>::from_buffer(2, 1, 1, vec![10, 20]).unwrap();
    let encoded = PngCodec::default().encode(&input).unwrap();
    let path = write_test_image("image-api-save-source.png", &encoded);
    let output_path = write_test_image("image-api-save-output.png", &[]);

    ImageApi::open(&path)
        .unwrap()
        .invert()
        .unwrap()
        .save(&output_path)
        .unwrap();

    let output = fs::read(&output_path).unwrap();
    let decoded = PngCodec::default().decode::<U8>(&output).unwrap();
    assert_eq!(decoded.pixels(), &[245, 235]);

    fs::remove_file(path).unwrap();
    fs::remove_file(output_path).unwrap();
}

#[cfg(feature = "webp")]
#[test]
fn image_api_open_supports_webp_encode_flow() {
    let input = Image::<U8>::from_buffer(2, 1, 3, vec![16, 32, 64, 128, 144, 160]).unwrap();
    let encoded = WebpCodec
        .encode_with_options(
            &input,
            &SaveOptions {
                quality: Some(90),
                ..SaveOptions::default()
            },
        )
        .unwrap();
    let path = write_test_image("image-api-open.webp", &encoded);

    let output = ImageApi::open(&path)
        .unwrap()
        .invert()
        .unwrap()
        .encode_webp(85)
        .unwrap();

    let decoded = WebpCodec.decode::<U8>(&output).unwrap();
    assert_eq!(decoded.width(), 2);
    assert_eq!(decoded.height(), 1);
    assert_eq!(decoded.bands(), 3);

    fs::remove_file(path).unwrap();
}

#[cfg(any(feature = "jpeg", feature = "png", feature = "webp"))]
fn write_test_image(name: &str, bytes: &[u8]) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("image-api-tests");
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{}-{name}", std::process::id()));
    fs::write(&path, bytes).unwrap();
    path
}

struct ZeroWriter;

impl Write for ZeroWriter {
    fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
        Ok(0)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct BrokenWriter;

impl Write for BrokenWriter {
    fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
        Err(io::Error::new(
            ErrorKind::BrokenPipe,
            "chaos writer exploded",
        ))
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn image_api_chaos_sharpen_accepts_zero_negative_nan_and_large_sigma_without_panicking() {
    let pixels = vec![
        10, 20, 30, 40, 50, 60, 70, 80, 90, 15, 25, 35, 45, 55, 65, 75, 85, 95, 20, 30, 40, 50, 60,
        70, 80, 90, 100,
    ];

    for sigma in [0.0, -1.0, f32::NAN, 64.0] {
        let result = catch_unwind(AssertUnwindSafe(|| {
            ImageApi {
                builder: PipelineBuilder::from_source(
                    MemorySource::<U8>::new(3, 3, 3, pixels.clone()).unwrap(),
                )
                .with_colorspace(ColorspaceId::SRgb),
                resource_limits: None,
            }
            .sharpen_with(sigma, 2.0, 10.0, 20.0, 0.0, 3.0)
            .unwrap()
            .builder
            .build()
            .unwrap()
            .run_to_image::<U8, _>(&RayonScheduler::new(1).unwrap())
            .unwrap()
        }));

        assert!(result.is_ok(), "sigma={sigma:?} should not panic");
    }
}

#[cfg(feature = "png")]
#[test]
fn image_api_chaos_streaming_png_writer_failures_surface_as_io_errors() {
    let input = Image::<U8>::from_buffer(2, 1, 1, vec![10, 20]).unwrap();
    let encoded = PngCodec::default().encode(&input).unwrap();

    let zero_err = ImageApi::from_bytes(&encoded)
        .unwrap()
        .encode_png_to(&mut ZeroWriter)
        .unwrap_err();
    assert!(matches!(zero_err, ViprsError::Io(_)));

    let broken_err = ImageApi::from_bytes(&encoded)
        .unwrap()
        .encode_png_to(&mut BrokenWriter)
        .unwrap_err();
    assert!(matches!(broken_err, ViprsError::Io(_)));
}

#[cfg(feature = "jpeg")]
#[test]
fn image_api_chaos_streaming_jpeg_writer_failures_surface_as_io_errors() {
    let input = Image::<U8>::from_buffer(2, 1, 3, vec![10, 20, 30, 40, 50, 60]).unwrap();
    let encoded = JpegCodec
        .encode_with_options(
            &input,
            &SaveOptions {
                quality: Some(90),
                ..SaveOptions::default()
            },
        )
        .unwrap();

    let zero_err = ImageApi::from_bytes(&encoded)
        .unwrap()
        .encode_jpeg_to(&mut ZeroWriter, 80)
        .unwrap_err();
    assert!(matches!(zero_err, ViprsError::Io(_)));

    let broken_err = ImageApi::from_bytes(&encoded)
        .unwrap()
        .encode_jpeg_to(&mut BrokenWriter, 80)
        .unwrap_err();
    assert!(matches!(broken_err, ViprsError::Io(_)));
}

#[cfg(feature = "webp")]
#[test]
fn image_api_chaos_streaming_webp_writer_failures_surface_as_io_errors() {
    let input = Image::<U8>::from_buffer(2, 1, 3, vec![16, 32, 64, 128, 144, 160]).unwrap();
    let encoded = WebpCodec
        .encode_with_options(
            &input,
            &SaveOptions {
                quality: Some(90),
                ..SaveOptions::default()
            },
        )
        .unwrap();

    let zero_err = ImageApi::from_bytes(&encoded)
        .unwrap()
        .encode_webp_to(&mut ZeroWriter, 80)
        .unwrap_err();
    assert!(matches!(zero_err, ViprsError::Io(_)));

    let broken_err = ImageApi::from_bytes(&encoded)
        .unwrap()
        .encode_webp_to(&mut BrokenWriter, 80)
        .unwrap_err();
    assert!(matches!(broken_err, ViprsError::Io(_)));
}

#[cfg(feature = "png")]
#[test]
fn image_api_chaos_streaming_png_encode_bypasses_decode_rejection_and_preserves_output_resource_limits()
 {
    let input = Image::<U8>::from_buffer(2, 2, 1, vec![255; 4]).unwrap();
    let encoded = PngCodec::default().encode(&input).unwrap();
    // Keep decode permissive enough for the 2×2 source so both buffered and
    // streaming encoders report the output limit instead of failing during decode.
    let limits = ResourceLimits::new(16, 3, 1).with_max_decode_bytes(4);

    let api = ImageApi::with_limits(limits.clone())
        .from_bytes(&encoded)
        .unwrap();

    let err = ImageApi::with_limits(limits)
        .from_bytes(&encoded)
        .unwrap()
        .encode_png()
        .unwrap_err();
    assert!(matches!(
        err,
        ViprsError::ImageTooLarge {
            width: 2,
            height: 2,
            bands: 1,
            details: "output byte size exceeds resource limits",
            ..
        }
    ));

    let mut streamed = Vec::new();
    let stream_err = api.encode_png_to(&mut streamed).unwrap_err();
    assert!(matches!(
        stream_err,
        ViprsError::ImageTooLarge {
            width: 2,
            height: 2,
            bands: 1,
            details: "output byte size exceeds resource limits",
            ..
        }
    ));
}

#[cfg(feature = "png")]
#[test]
fn image_api_chaos_smartcrop_zero_dimensions_clamp_to_one_pixel() {
    let input =
        Image::<U8>::from_buffer(2, 2, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).unwrap();
    let encoded = PngCodec::default().encode(&input).unwrap();

    let output = ImageApi::from_bytes(&encoded)
        .unwrap()
        .smartcrop(0, 0)
        .unwrap()
        .encode_png()
        .unwrap();
    let decoded = PngCodec::default().decode::<U8>(&output).unwrap();

    assert_eq!((decoded.width(), decoded.height()), (1, 1));
}

#[cfg(feature = "png")]
#[test]
fn image_api_chaos_smartcrop_u32_max_clamps_to_source_bounds() {
    let input =
        Image::<U8>::from_buffer(2, 2, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).unwrap();
    let encoded = PngCodec::default().encode(&input).unwrap();

    let output = ImageApi::from_bytes(&encoded)
        .unwrap()
        .smartcrop(u32::MAX, u32::MAX)
        .unwrap()
        .encode_png()
        .unwrap();
    let decoded = PngCodec::default().decode::<U8>(&output).unwrap();

    assert_eq!((decoded.width(), decoded.height()), (2, 2));
}

#[cfg(feature = "png")]
#[test]
fn image_api_chaos_premultiply_panics_on_single_band_grayscale() {
    let input = Image::<U8>::from_buffer(1, 1, 1, vec![128]).unwrap();
    let encoded = PngCodec::default().encode(&input).unwrap();

    let result = catch_unwind(AssertUnwindSafe(|| {
        ImageApi::from_bytes(&encoded)
            .unwrap()
            .premultiply()
            .unwrap()
            .encode_png()
            .unwrap();
    }));

    assert!(
        result.is_err(),
        "single-band premultiply unexpectedly avoided panic"
    );
}

#[cfg(feature = "png")]
#[test]
fn image_api_chaos_unpremultiply_panics_on_single_band_grayscale() {
    let input = Image::<U8>::from_buffer(1, 1, 1, vec![64]).unwrap();
    let encoded = PngCodec::default().encode(&input).unwrap();

    let result = catch_unwind(AssertUnwindSafe(|| {
        ImageApi::from_bytes(&encoded)
            .unwrap()
            .unpremultiply()
            .unwrap()
            .encode_png()
            .unwrap();
    }));

    assert!(
        result.is_err(),
        "single-band unpremultiply unexpectedly avoided panic"
    );
}

#[test]
fn image_api_chaos_flatten_with_rejects_u16_rgba() {
    let err = ImageApi {
        builder: PipelineBuilder::from_source(
            MemorySource::<U16>::new(1, 1, 4, vec![1000, 2000, 3000, 40000]).unwrap(),
        )
        .with_colorspace(ColorspaceId::SRgb),
        resource_limits: None,
    }
    .flatten_with(10, 20, 30)
    .unwrap_err();

    assert!(matches!(
        err,
        BuildError::UnsupportedFormat {
            op: "flatten",
            format: BandFormatId::U16,
        }
    ));
}

#[cfg(feature = "jpeg")]
fn truncate_jpeg_scan_data(encoded: &[u8]) -> Vec<u8> {
    for index in 0..encoded.len().saturating_sub(1) {
        if encoded[index] == 0xFF && encoded[index + 1] == 0xDA {
            let scan_header_len = encoded
                .get(index + 2..index + 4)
                .map_or(0, |bytes| u16::from_be_bytes([bytes[0], bytes[1]]) as usize);
            let cutoff = index + 2 + scan_header_len + 8;
            if cutoff < encoded.len() {
                return encoded[..cutoff].to_vec();
            }
        }
    }

    encoded[..encoded.len().saturating_sub(16).max(1)].to_vec()
}

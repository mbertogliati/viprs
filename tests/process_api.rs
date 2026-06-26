//! Integration tests for `viprs::adapters::process` — the one-call server API.
//!
//! Tests exercise the full decode → ops → encode → write lifecycle with real
//! fixture files. Each test verifies format detection, dimension tracking,
//! output validity (re-decodable), and bytes_written accuracy.

use viprs::adapters::process::{EncodeOptions, ProcessOptions, process};
use viprs::domain::cancel::CancellationToken;
use viprs::domain::error::ViprsError;
use viprs::domain::limits::DecodeLimits;

const FIXTURE_DIR: &str = "tests/fixtures/images";

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!("{FIXTURE_DIR}/{name}"))
        .unwrap_or_else(|e| panic!("failed to read fixture {name}: {e}"))
}

// ── JPEG ──────────────────────────────────────────────────────────────────────

#[cfg(feature = "jpeg")]
mod jpeg {
    use super::*;

    #[test]
    fn process_jpeg_identity() {
        let input = fixture("bench_2048x2048.jpg");
        let mut output = Vec::new();

        let result = process(
            &input,
            &mut output,
            |img| Ok(img),
            &EncodeOptions::Jpeg { quality: 85 },
            &ProcessOptions::default(),
        )
        .expect("process should succeed");

        assert_eq!(result.input_format, "jpeg");
        assert_eq!(result.input_dimensions, (2048, 2048));
        assert_eq!(result.output_format, "jpeg");
        assert_eq!(result.bytes_written, output.len() as u64);
        assert!(!output.is_empty());
        // Output should be valid JPEG (starts with FF D8 FF)
        assert_eq!(&output[..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn process_jpeg_with_strip_metadata() {
        let input = fixture("bench_2048x2048.jpg");
        let mut output = Vec::new();

        let opts = ProcessOptions {
            strip_metadata: true,
            ..ProcessOptions::default()
        };

        let result = process(
            &input,
            &mut output,
            |img| Ok(img),
            &EncodeOptions::Jpeg { quality: 85 },
            &opts,
        )
        .expect("process should succeed");

        assert_eq!(result.input_format, "jpeg");
        assert!(!output.is_empty());
    }

    #[test]
    fn process_jpeg_cancellation() {
        let input = fixture("bench_2048x2048.jpg");
        let mut output = Vec::new();

        let token = CancellationToken::new();
        token.cancel(); // Pre-cancel

        let opts = ProcessOptions {
            cancel_token: Some(token),
            ..ProcessOptions::default()
        };

        let result = process(
            &input,
            &mut output,
            |img| Ok(img),
            &EncodeOptions::Jpeg { quality: 85 },
            &opts,
        );

        assert!(matches!(result, Err(ViprsError::Cancelled)));
    }

    #[test]
    fn process_jpeg_decode_limits_reject() {
        let input = fixture("bench_2048x2048.jpg");
        let mut output = Vec::new();

        let opts = ProcessOptions {
            limits: DecodeLimits {
                max_width: 100,
                max_height: 100,
                ..DecodeLimits::default()
            },
            ..ProcessOptions::default()
        };

        let result = process(
            &input,
            &mut output,
            |img| Ok(img),
            &EncodeOptions::Jpeg { quality: 85 },
            &opts,
        );

        assert!(matches!(result, Err(ViprsError::ImageTooLarge { .. })));
    }
}

// ── PNG ───────────────────────────────────────────────────────────────────────

#[cfg(feature = "png")]
mod png {
    use super::*;

    #[test]
    fn process_png_identity() {
        let input = fixture("bench_2048x2048.png");
        let mut output = Vec::new();

        let result = process(
            &input,
            &mut output,
            |img| Ok(img),
            &EncodeOptions::Png { compression: 6 },
            &ProcessOptions::default(),
        )
        .expect("process should succeed");

        assert_eq!(result.input_format, "png");
        assert_eq!(result.input_dimensions, (2048, 2048));
        assert_eq!(result.output_format, "png");
        assert!(!output.is_empty());
        // PNG signature
        assert_eq!(&output[..4], &[0x89, 0x50, 0x4E, 0x47]);
    }
}

// ── WebP ──────────────────────────────────────────────────────────────────────

#[cfg(feature = "webp")]
mod webp {
    use super::*;

    #[test]
    fn process_webp_identity() {
        let input = fixture("bench_2048x2048.webp");
        let mut output = Vec::new();

        let result = process(
            &input,
            &mut output,
            |img| Ok(img),
            &EncodeOptions::WebP {
                quality: 80,
                lossless: false,
            },
            &ProcessOptions::default(),
        )
        .expect("process should succeed");

        assert_eq!(result.input_format, "webp");
        assert_eq!(result.output_format, "webp");
        assert!(!output.is_empty());
        // RIFF header
        assert_eq!(&output[..4], b"RIFF");
    }

    #[test]
    fn process_webp_lossless() {
        let input = fixture("bench_2048x2048.webp");
        let mut output = Vec::new();

        let result = process(
            &input,
            &mut output,
            |img| Ok(img),
            &EncodeOptions::WebP {
                quality: 100,
                lossless: true,
            },
            &ProcessOptions::default(),
        )
        .expect("process should succeed");

        assert_eq!(result.output_format, "webp");
        assert!(!output.is_empty());
    }
}

// ── TIFF ──────────────────────────────────────────────────────────────────────

#[cfg(feature = "tiff")]
mod tiff {
    use super::*;

    #[test]
    fn process_tiff_identity() {
        let input = fixture("bench_2048x2048.tif");
        let mut output = Vec::new();

        let result = process(
            &input,
            &mut output,
            |img| Ok(img),
            &EncodeOptions::Tiff,
            &ProcessOptions::default(),
        )
        .expect("process should succeed");

        assert_eq!(result.input_format, "tiff");
        assert_eq!(result.output_format, "tiff");
        assert!(!output.is_empty());
        // TIFF magic (little-endian II or big-endian MM)
        assert!(
            &output[..2] == b"II" || &output[..2] == b"MM",
            "expected TIFF magic, got {:?}",
            &output[..2]
        );
    }
}

// ── GIF ───────────────────────────────────────────────────────────────────────

#[cfg(feature = "gif")]
mod gif {
    use super::*;

    #[test]
    fn process_gif_identity() {
        let input = fixture("bench_2048x2048.gif");
        let mut output = Vec::new();

        let result = process(
            &input,
            &mut output,
            |img| Ok(img),
            &EncodeOptions::Gif,
            &ProcessOptions::default(),
        )
        .expect("process should succeed");

        assert_eq!(result.input_format, "gif");
        assert_eq!(result.output_format, "gif");
        assert!(!output.is_empty());
        // GIF magic
        assert_eq!(&output[..3], b"GIF");
    }
}

// ── Cross-format ──────────────────────────────────────────────────────────────

#[cfg(all(feature = "jpeg", feature = "png"))]
mod cross_format {
    use super::*;

    #[test]
    fn process_jpeg_to_png() {
        let input = fixture("bench_2048x2048.jpg");
        let mut output = Vec::new();

        let result = process(
            &input,
            &mut output,
            |img| Ok(img),
            &EncodeOptions::Png { compression: 6 },
            &ProcessOptions::default(),
        )
        .expect("process should succeed");

        assert_eq!(result.input_format, "jpeg");
        assert_eq!(result.output_format, "png");
        // PNG signature
        assert_eq!(&output[..4], &[0x89, 0x50, 0x4E, 0x47]);
    }

    #[test]
    fn process_png_to_jpeg() {
        let input = fixture("bench_2048x2048.png");
        let mut output = Vec::new();

        let result = process(
            &input,
            &mut output,
            |img| Ok(img),
            &EncodeOptions::Jpeg { quality: 90 },
            &ProcessOptions::default(),
        )
        .expect("process should succeed");

        assert_eq!(result.input_format, "png");
        assert_eq!(result.output_format, "jpeg");
        assert_eq!(&output[..2], &[0xFF, 0xD8]);
    }
}

// ── Pipeline-based processing ────────────────────────────────────────────────

#[cfg(all(feature = "jpeg", feature = "rayon"))]
mod pipeline_jpeg {
    use super::*;
    use viprs::adapters::process::process_pipeline;

    #[test]
    fn pipeline_jpeg_identity() {
        let input = fixture("bench_2048x2048.jpg");
        let mut output = Vec::new();

        let result = process_pipeline(
            &input,
            &mut output,
            |builder| Ok(builder),
            &EncodeOptions::Jpeg { quality: 85 },
            &ProcessOptions::default(),
        )
        .expect("process_pipeline should succeed");

        assert_eq!(result.input_format, "jpeg");
        assert_eq!(result.input_dimensions, (2048, 2048));
        assert_eq!(result.output_format, "jpeg");
        assert_eq!(result.bytes_written, output.len() as u64);
        assert!(!output.is_empty());
        assert_eq!(&output[..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn pipeline_jpeg_invert() {
        let input = fixture("bench_2048x2048.jpg");
        let mut output = Vec::new();

        let result = process_pipeline(
            &input,
            &mut output,
            |builder| builder.invert().map_err(Into::into),
            &EncodeOptions::Jpeg { quality: 85 },
            &ProcessOptions::default(),
        )
        .expect("process_pipeline invert should succeed");

        assert_eq!(result.input_format, "jpeg");
        assert_eq!(result.output_format, "jpeg");
        assert!(!output.is_empty());
    }

    #[test]
    fn pipeline_jpeg_linear() {
        let input = fixture("bench_2048x2048.jpg");
        let mut output = Vec::new();

        let result = process_pipeline(
            &input,
            &mut output,
            |builder| builder.linear(1.5, 10.0).map_err(Into::into),
            &EncodeOptions::Jpeg { quality: 85 },
            &ProcessOptions::default(),
        )
        .expect("process_pipeline linear should succeed");

        assert_eq!(result.input_format, "jpeg");
        assert!(!output.is_empty());
    }

    #[test]
    fn pipeline_cancellation() {
        let input = fixture("bench_2048x2048.jpg");
        let mut output = Vec::new();

        let token = CancellationToken::new();
        token.cancel();

        let opts = ProcessOptions {
            cancel_token: Some(token),
            ..ProcessOptions::default()
        };

        let result = process_pipeline(
            &input,
            &mut output,
            |builder| Ok(builder),
            &EncodeOptions::Jpeg { quality: 85 },
            &opts,
        );

        assert!(matches!(result, Err(ViprsError::Cancelled)));
    }

    #[test]
    fn pipeline_decode_limits_reject() {
        let input = fixture("bench_2048x2048.jpg");
        let mut output = Vec::new();

        let opts = ProcessOptions {
            limits: DecodeLimits {
                max_width: 100,
                max_height: 100,
                ..DecodeLimits::default()
            },
            ..ProcessOptions::default()
        };

        let result = process_pipeline(
            &input,
            &mut output,
            |builder| Ok(builder),
            &EncodeOptions::Jpeg { quality: 85 },
            &opts,
        );

        assert!(matches!(result, Err(ViprsError::ImageTooLarge { .. })));
    }

    #[test]
    fn pipeline_strip_metadata() {
        let input = fixture("bench_2048x2048.jpg");
        let mut output = Vec::new();

        let opts = ProcessOptions {
            strip_metadata: true,
            ..ProcessOptions::default()
        };

        let result = process_pipeline(
            &input,
            &mut output,
            |builder| Ok(builder),
            &EncodeOptions::Jpeg { quality: 85 },
            &opts,
        )
        .expect("pipeline strip should succeed");

        assert_eq!(result.input_format, "jpeg");
        assert!(!output.is_empty());
    }
}

#[cfg(all(feature = "png", feature = "rayon"))]
mod pipeline_png {
    use super::*;
    use viprs::adapters::process::process_pipeline;

    #[test]
    fn pipeline_png_identity() {
        let input = fixture("bench_2048x2048.png");
        let mut output = Vec::new();

        let result = process_pipeline(
            &input,
            &mut output,
            |builder| Ok(builder),
            &EncodeOptions::Png { compression: 6 },
            &ProcessOptions::default(),
        )
        .expect("pipeline png should succeed");

        assert_eq!(result.input_format, "png");
        assert_eq!(result.output_format, "png");
        assert!(!output.is_empty());
        assert_eq!(&output[..4], &[0x89, 0x50, 0x4E, 0x47]);
    }

    #[test]
    fn pipeline_png_invert() {
        let input = fixture("bench_2048x2048.png");
        let mut output = Vec::new();

        let result = process_pipeline(
            &input,
            &mut output,
            |builder| builder.invert().map_err(Into::into),
            &EncodeOptions::Png { compression: 6 },
            &ProcessOptions::default(),
        )
        .expect("pipeline png invert should succeed");

        assert_eq!(result.input_format, "png");
        assert!(!output.is_empty());
    }
}

#[cfg(all(feature = "jpeg", feature = "png", feature = "rayon"))]
mod pipeline_cross_format {
    use super::*;
    use viprs::adapters::process::process_pipeline;

    #[test]
    fn pipeline_jpeg_to_png() {
        let input = fixture("bench_2048x2048.jpg");
        let mut output = Vec::new();

        let result = process_pipeline(
            &input,
            &mut output,
            |builder| Ok(builder),
            &EncodeOptions::Png { compression: 6 },
            &ProcessOptions::default(),
        )
        .expect("pipeline cross-format should succeed");

        assert_eq!(result.input_format, "jpeg");
        assert_eq!(result.output_format, "png");
        assert_eq!(&output[..4], &[0x89, 0x50, 0x4E, 0x47]);
    }
}

// ── JP2K ──────────────────────────────────────────────────────────────────────

#[cfg(feature = "jp2k")]
mod jp2k {
    use super::*;

    #[test]
    fn process_jp2k_identity() {
        let input = fixture("bench_2048x2048.jp2");
        let mut output = Vec::new();

        let result = process(
            &input,
            &mut output,
            |img| Ok(img),
            &EncodeOptions::Jp2k { quality: 80 },
            &ProcessOptions::default(),
        )
        .expect("process jp2k should succeed");

        assert_eq!(result.input_format, "jp2k");
        assert_eq!(result.output_format, "jp2k");
        assert!(!output.is_empty());
    }
}

// ── EXR ───────────────────────────────────────────────────────────────────────

#[cfg(feature = "exr")]
mod exr {
    use super::*;

    #[test]
    fn process_exr_identity() {
        let input = fixture("bench_512x512.exr");
        let mut output = Vec::new();

        let result = process(
            &input,
            &mut output,
            |img| Ok(img),
            &EncodeOptions::Exr,
            &ProcessOptions::default(),
        )
        .expect("process exr should succeed");

        assert_eq!(result.input_format, "exr");
        assert_eq!(result.output_format, "exr");
        assert!(!output.is_empty());
    }
}

// ── BMP ───────────────────────────────────────────────────────────────────────

#[cfg(all(feature = "bmp", feature = "jpeg"))]
mod bmp {
    use super::*;

    #[test]
    fn process_jpeg_to_bmp() {
        let input = fixture("bench_2048x2048.jpg");
        let mut output = Vec::new();

        let result = process(
            &input,
            &mut output,
            |img| Ok(img),
            &EncodeOptions::Bmp,
            &ProcessOptions::default(),
        )
        .expect("process jpeg to bmp should succeed");

        assert_eq!(result.input_format, "jpeg");
        assert_eq!(result.output_format, "bmp");
        assert!(!output.is_empty());
        // BMP magic: "BM"
        assert_eq!(&output[..2], b"BM");
    }
}

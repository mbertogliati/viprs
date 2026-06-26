mod robustez_corrupt {
    use std::{
        fs,
        panic::{AssertUnwindSafe, catch_unwind},
        path::{Path, PathBuf},
    };

    use viprs::{InMemoryImage, U8, ViprsError};

    fn project_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).to_owned()
    }

    fn fixture_path(name: &str) -> PathBuf {
        project_root()
            .join("tests")
            .join("fixtures")
            .join("images")
            .join(name)
    }

    fn read_fixture(name: &str) -> Vec<u8> {
        let path = fixture_path(name);
        fs::read(&path)
            .unwrap_or_else(|error| panic!("failed to read fixture {}: {error}", path.display()))
    }

    fn deterministic_bytes(len: usize) -> Vec<u8> {
        (0..len)
            .map(|index| ((index.wrapping_mul(73).wrapping_add(19)) % 251) as u8)
            .collect()
    }

    fn assert_decode_fails_gracefully<T>(
        label: &str,
        decode: impl FnOnce() -> Result<T, ViprsError>,
        error_predicate: impl FnOnce(&ViprsError) -> bool,
    ) {
        let outcome = catch_unwind(AssertUnwindSafe(decode));
        match outcome {
            Ok(Err(error)) => {
                assert!(
                    error_predicate(&error),
                    "{label} returned the wrong typed error: {error:?}"
                );
            }
            Ok(Ok(_)) => panic!("{label} unexpectedly decoded successfully"),
            Err(_) => panic!("{label} panicked instead of returning Err(...)"),
        }
    }

    #[cfg(feature = "jpeg")]
    fn tiny_valid_jpeg() -> Vec<u8> {
        use viprs::{adapters::codecs::JpegCodec, ports::codec::ImageEncoder};

        let image =
            InMemoryImage::<U8>::from_buffer(1, 1, 3, vec![12, 34, 56]).expect("valid RGB test image");
        JpegCodec.encode(&image).expect("tiny JPEG fixture")
    }

    #[cfg(feature = "jpeg")]
    fn patch_jpeg_sof_dimensions(jpeg: &[u8], width: u16, height: u16) -> Vec<u8> {
        let mut patched = jpeg.to_vec();
        let mut offset = 2usize;

        while offset + 1 < patched.len() {
            while offset < patched.len() && patched[offset] == 0xFF {
                offset += 1;
            }
            if offset >= patched.len() {
                break;
            }

            let marker = patched[offset];
            offset += 1;

            if marker == 0xD9 || marker == 0xDA {
                break;
            }
            if matches!(marker, 0x01 | 0xD0..=0xD7) {
                continue;
            }
            if offset + 2 > patched.len() {
                break;
            }

            let segment_len = u16::from_be_bytes([patched[offset], patched[offset + 1]]) as usize;
            let payload_start = offset + 2;
            let payload_end = offset + segment_len;
            if payload_end > patched.len() || segment_len < 7 {
                break;
            }

            if marker == 0xC0 {
                patched[payload_start + 1..payload_start + 3]
                    .copy_from_slice(&height.to_be_bytes());
                patched[payload_start + 3..payload_start + 5].copy_from_slice(&width.to_be_bytes());
                return patched;
            }

            offset += segment_len;
        }

        panic!("baseline JPEG fixture did not contain an SOF0 segment");
    }

    #[test]
    #[cfg(feature = "jpeg")]
    fn truncated_jpeg_returns_typed_error_without_panic() {
        use viprs::{adapters::codecs::JpegCodec, ports::codec::ImageDecoder};

        let valid = read_fixture("sample.jpg");
        let truncated = valid[..valid.len() / 2].to_vec();

        assert_decode_fails_gracefully(
            "truncated JPEG",
            || JpegCodec.decode::<U8>(&truncated),
            |err| matches!(err, ViprsError::Codec(_)),
        );
    }

    #[test]
    #[cfg(feature = "png")]
    fn truncated_png_returns_typed_error_without_panic() {
        use viprs::{adapters::codecs::PngCodec, ports::codec::ImageDecoder};

        let valid = read_fixture("sample.png");
        let truncated = valid[..valid.len() / 2].to_vec();

        assert_decode_fails_gracefully(
            "truncated PNG",
            || PngCodec::default().decode::<U8>(&truncated),
            |err| matches!(err, ViprsError::Codec(_)),
        );
    }

    #[test]
    #[cfg(feature = "jpeg")]
    fn random_bytes_as_jpeg_return_codec_error_without_panic() {
        use viprs::{adapters::codecs::JpegCodec, ports::codec::ImageDecoder};

        let random = deterministic_bytes(512);

        assert_decode_fails_gracefully(
            "random bytes as JPEG",
            || JpegCodec.decode::<U8>(&random),
            |err| matches!(err, ViprsError::Codec(_)),
        );
    }

    #[test]
    #[cfg(feature = "jpeg")]
    fn empty_input_returns_typed_error_without_panic() {
        use viprs::{adapters::codecs::JpegCodec, ports::codec::ImageDecoder};

        assert_decode_fails_gracefully(
            "empty JPEG input",
            || JpegCodec.decode::<U8>(&[]),
            |err| matches!(err, ViprsError::Codec(_)),
        );
    }

    #[test]
    #[cfg(feature = "jpeg")]
    fn jpeg_header_with_corrupt_body_returns_typed_error_without_panic() {
        use viprs::{adapters::codecs::JpegCodec, ports::codec::ImageDecoder};

        let valid = read_fixture("sample.jpg");
        let mut corrupted = valid[..20].to_vec();
        corrupted.extend_from_slice(&deterministic_bytes(256));

        assert_decode_fails_gracefully(
            "JPEG valid header with corrupt body",
            || JpegCodec.decode::<U8>(&corrupted),
            |err| matches!(err, ViprsError::Codec(_)),
        );
    }

    #[test]
    #[cfg(feature = "jpeg")]
    fn jpeg_with_wrong_width_in_header_returns_typed_error_without_panic() {
        use viprs::{adapters::codecs::JpegCodec, ports::codec::ImageDecoder};

        let corrupted = patch_jpeg_sof_dimensions(&tiny_valid_jpeg(), 40_000, 1);

        assert_decode_fails_gracefully(
            "JPEG wrong width in SOF header",
            || JpegCodec.decode::<U8>(&corrupted),
            |err| matches!(err, ViprsError::Codec(_)),
        );
    }

    #[test]
    #[cfg(feature = "jpeg")]
    fn jpeg_oversized_dimensions_in_header_return_typed_error_without_panic() {
        use viprs::{adapters::codecs::JpegCodec, ports::codec::ImageDecoder};

        // JPEG SOF dimensions are 16-bit, so use a near-maximum representable size to
        // exercise the oversized-header path without risking an allocation blow-up.
        let oversized = patch_jpeg_sof_dimensions(&tiny_valid_jpeg(), 65_000, 65_000);

        assert_decode_fails_gracefully(
            "JPEG oversized dimensions in header",
            || JpegCodec.decode::<U8>(&oversized),
            |err| matches!(err, ViprsError::Codec(message) if message.contains("safety limit")),
        );
    }

    #[test]
    #[cfg(feature = "webp")]
    fn truncated_webp_returns_typed_error_without_panic() {
        use viprs::{adapters::codecs::WebpCodec, ports::codec::ImageDecoder};

        let valid = read_fixture("sample.webp");
        let truncated = valid[..100].to_vec();

        assert_decode_fails_gracefully(
            "truncated WebP",
            || WebpCodec.decode::<U8>(&truncated),
            |err| matches!(err, ViprsError::Codec(_)),
        );
    }

    #[test]
    #[cfg(feature = "jpeg")]
    fn zero_length_codec_round_trip_corruption_returns_typed_error_without_panic() {
        use viprs::{
            adapters::codecs::JpegCodec,
            ports::codec::{ImageDecoder, ImageEncoder},
        };

        let image =
            InMemoryImage::<U8>::from_buffer(1, 1, 3, vec![90, 40, 10]).expect("valid RGB test image");
        let mut encoded = JpegCodec
            .encode(&image)
            .expect("tiny JPEG round-trip fixture");
        encoded.clear();

        assert_decode_fails_gracefully(
            "zero-length JPEG round-trip corruption",
            || JpegCodec.decode::<U8>(&encoded),
            |err| matches!(err, ViprsError::Codec(_)),
        );
    }
}

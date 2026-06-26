mod robustez_adversarial {
    #[cfg(any(feature = "jpeg", feature = "png", feature = "webp"))]
    use std::panic::{AssertUnwindSafe, catch_unwind};

    #[cfg(feature = "png")]
    use std::path::{Path, PathBuf};

    #[cfg(any(feature = "jpeg", feature = "png", feature = "webp"))]
    use viprs::ViprsError;

    #[cfg(feature = "png")]
    fn project_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).to_owned()
    }

    #[cfg(any(feature = "jpeg", feature = "png", feature = "webp"))]
    fn assert_rejects_adversarial_input<T>(
        label: &str,
        decode: impl FnOnce() -> Result<T, ViprsError>,
        error_predicate: impl FnOnce(&ViprsError) -> bool,
    ) {
        let outcome = catch_unwind(AssertUnwindSafe(decode));
        match outcome {
            Ok(result) => {
                assert!(result.is_err(), "{label}: should reject adversarial input");
                let error = match result {
                    Ok(_) => unreachable!("checked above"),
                    Err(error) => error,
                };
                assert!(
                    error_predicate(&error),
                    "{label}: returned the wrong typed error: {error:?}"
                );
            }
            Err(_) => panic!("{label}: decoder panicked instead of returning Err(...)"),
        }
    }

    #[cfg(feature = "jpeg")]
    fn tiny_valid_jpeg() -> Vec<u8> {
        use viprs::{InMemoryImage, U8, adapters::codecs::JpegCodec, ports::codec::ImageEncoder};

        let image =
            InMemoryImage::<U8>::from_buffer(1, 1, 3, vec![0, 0, 0]).expect("valid RGB test image");
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

    #[cfg(feature = "jpeg")]
    fn insert_jpeg_app_segment_after_soi(jpeg: &[u8], marker: u8, payload: &[u8]) -> Vec<u8> {
        let segment_len = u16::try_from(payload.len() + 2).expect("APP payload length fits in u16");
        let mut with_segment = Vec::with_capacity(jpeg.len() + payload.len() + 4);
        with_segment.extend_from_slice(&jpeg[..2]);
        with_segment.extend_from_slice(&[0xFF, marker]);
        with_segment.extend_from_slice(&segment_len.to_be_bytes());
        with_segment.extend_from_slice(payload);
        with_segment.extend_from_slice(&jpeg[2..]);
        with_segment
    }

    #[cfg(feature = "jpeg")]
    fn malformed_exif_payload() -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(b"Exif\0\0");
        payload.extend_from_slice(b"ZZ");
        payload.extend_from_slice(&42u16.to_le_bytes());
        payload.extend_from_slice(&8u32.to_le_bytes());
        payload.extend_from_slice(&1u16.to_le_bytes());
        payload
    }

    #[cfg(feature = "jpeg")]
    fn oversized_exif_string_payload() -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(b"Exif\0\0");
        payload.extend_from_slice(b"II");
        payload.extend_from_slice(&42u16.to_le_bytes());
        payload.extend_from_slice(&8u32.to_le_bytes());
        payload.extend_from_slice(&1u16.to_le_bytes());
        payload.extend_from_slice(&0x010Eu16.to_le_bytes());
        payload.extend_from_slice(&2u16.to_le_bytes());
        payload.extend_from_slice(&1_048_576u32.to_le_bytes());
        payload.extend_from_slice(&0x20u32.to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());
        payload
    }

    #[cfg(feature = "jpeg")]
    fn extract_jpeg_segment(jpeg: &[u8], target_marker: u8) -> Vec<u8> {
        let mut offset = 2usize;

        while offset + 1 < jpeg.len() {
            while offset < jpeg.len() && jpeg[offset] == 0xFF {
                offset += 1;
            }
            if offset >= jpeg.len() {
                break;
            }

            let marker = jpeg[offset];
            let marker_offset = offset - 1;
            offset += 1;

            if marker == 0xD9 || marker == 0xDA {
                break;
            }
            if matches!(marker, 0x01 | 0xD0..=0xD7) {
                continue;
            }
            if offset + 2 > jpeg.len() {
                break;
            }

            let segment_len = u16::from_be_bytes([jpeg[offset], jpeg[offset + 1]]) as usize;
            let segment_end = offset + segment_len;
            if segment_end > jpeg.len() {
                break;
            }

            if marker == target_marker {
                return jpeg[marker_offset..segment_end].to_vec();
            }

            offset += segment_len;
        }

        panic!("baseline JPEG fixture did not contain marker 0x{target_marker:02X}");
    }

    #[cfg(feature = "jpeg")]
    fn insert_jpeg_segment_before_sos(jpeg: &[u8], segment: &[u8]) -> Vec<u8> {
        let mut offset = 2usize;

        while offset + 1 < jpeg.len() {
            while offset < jpeg.len() && jpeg[offset] == 0xFF {
                offset += 1;
            }
            if offset >= jpeg.len() {
                break;
            }

            let marker = jpeg[offset];
            let marker_offset = offset - 1;
            offset += 1;
            if marker == 0xDA {
                let mut duplicated = Vec::with_capacity(jpeg.len() + segment.len());
                duplicated.extend_from_slice(&jpeg[..marker_offset]);
                duplicated.extend_from_slice(segment);
                duplicated.extend_from_slice(&jpeg[marker_offset..]);
                return duplicated;
            }
            if marker == 0xD9 {
                break;
            }
            if matches!(marker, 0x01 | 0xD0..=0xD7) {
                continue;
            }
            if offset + 2 > jpeg.len() {
                break;
            }

            let segment_len = u16::from_be_bytes([jpeg[offset], jpeg[offset + 1]]) as usize;
            offset += segment_len;
        }

        panic!("baseline JPEG fixture did not contain an SOS segment");
    }

    #[cfg(feature = "jpeg")]
    fn insert_jpeg_icc_profile(jpeg: &[u8], profile_len: usize) -> Vec<u8> {
        const ICC_SIGNATURE: &[u8] = b"ICC_PROFILE\0";
        const MAX_APP_SEGMENT_PAYLOAD: usize = u16::MAX as usize - 2;
        const MAX_ICC_CHUNK_PAYLOAD: usize = MAX_APP_SEGMENT_PAYLOAD - ICC_SIGNATURE.len() - 2;

        let mut adversarial = jpeg.to_vec();
        let profile = vec![0xAA; profile_len];
        let chunk_count = profile.len().div_ceil(MAX_ICC_CHUNK_PAYLOAD);
        let chunk_count_u8 =
            u8::try_from(chunk_count).expect("test ICC profile fits in JPEG chunk count");

        for (index, chunk) in profile.chunks(MAX_ICC_CHUNK_PAYLOAD).enumerate().rev() {
            let sequence_number = u8::try_from(index + 1).expect("test ICC chunk index fits in u8");
            let mut payload = Vec::with_capacity(ICC_SIGNATURE.len() + 2 + chunk.len());
            payload.extend_from_slice(ICC_SIGNATURE);
            payload.push(sequence_number);
            payload.push(chunk_count_u8);
            payload.extend_from_slice(chunk);
            adversarial = insert_jpeg_app_segment_after_soi(&adversarial, 0xE2, &payload);
        }

        adversarial
    }

    #[cfg(feature = "png")]
    fn tiny_valid_png() -> Vec<u8> {
        use viprs::{InMemoryImage, U8, adapters::codecs::PngCodec, ports::codec::ImageEncoder};

        let image = InMemoryImage::<U8>::from_buffer(1, 1, 3, vec![200, 100, 50])
            .expect("valid RGB test image");
        PngCodec::default()
            .encode(&image)
            .expect("tiny PNG fixture")
    }

    #[cfg(feature = "png")]
    fn patch_png_chunk_crc(mut png: Vec<u8>, chunk_type: [u8; 4], crc: [u8; 4]) -> Vec<u8> {
        let mut offset = 8usize;

        while offset + 12 <= png.len() {
            let chunk_len = u32::from_be_bytes([
                png[offset],
                png[offset + 1],
                png[offset + 2],
                png[offset + 3],
            ]) as usize;
            let chunk_type_start = offset + 4;
            let data_end = chunk_type_start + 4 + chunk_len;
            let crc_start = data_end;
            let crc_end = crc_start + 4;
            if crc_end > png.len() {
                break;
            }

            if png[chunk_type_start..chunk_type_start + 4] == chunk_type {
                png[crc_start..crc_end].copy_from_slice(&crc);
                return png;
            }

            offset = crc_end;
        }

        panic!(
            "baseline PNG fixture did not contain chunk {}",
            std::str::from_utf8(&chunk_type).expect("ASCII chunk type")
        );
    }

    #[cfg(feature = "png")]
    fn patch_png_ihdr_dimensions(mut png: Vec<u8>, width: u32, height: u32) -> Vec<u8> {
        png[16..20].copy_from_slice(&width.to_be_bytes());
        png[20..24].copy_from_slice(&height.to_be_bytes());
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&png[12..29]);
        png[29..33].copy_from_slice(&hasher.finalize().to_be_bytes());
        png
    }

    #[cfg(feature = "webp")]
    fn tiny_valid_webp() -> Vec<u8> {
        use viprs::{InMemoryImage, U8, adapters::codecs::WebpCodec, ports::codec::ImageEncoder};

        let image = InMemoryImage::<U8>::from_buffer(1, 1, 3, vec![17, 34, 51])
            .expect("valid RGB test image");
        WebpCodec.encode(&image).expect("tiny WebP fixture")
    }

    #[cfg(feature = "webp")]
    fn patch_webp_riff_size(mut webp: Vec<u8>, declared_size: u32) -> Vec<u8> {
        webp[4..8].copy_from_slice(&declared_size.to_le_bytes());
        webp
    }

    #[test]
    #[cfg(feature = "jpeg")]
    fn zip_bomb_jpeg_returns_typed_error_without_panic() {
        use viprs::{U8, adapters::codecs::JpegCodec, ports::codec::ImageDecoder};

        let adversarial = patch_jpeg_sof_dimensions(&tiny_valid_jpeg(), 65_000, 65_000);

        assert_rejects_adversarial_input(
            "ZIP bomb JPEG",
            || JpegCodec.decode::<U8>(&adversarial),
            |err| matches!(err, ViprsError::Codec(message) if message.contains("safety limit")),
        );
    }

    #[test]
    #[cfg(feature = "jpeg")]
    fn malformed_exif_jpeg_returns_typed_error_without_panic() {
        use viprs::{U8, adapters::codecs::JpegCodec, ports::codec::ImageDecoder};

        let adversarial =
            insert_jpeg_app_segment_after_soi(&tiny_valid_jpeg(), 0xE1, &malformed_exif_payload());

        assert_rejects_adversarial_input(
            "Malformed EXIF JPEG",
            || JpegCodec.decode::<U8>(&adversarial),
            |err| matches!(err, ViprsError::Codec(message) if message.contains("EXIF")),
        );
    }

    #[test]
    #[cfg(feature = "jpeg")]
    fn jfif_with_wrong_dimensions_returns_typed_error_without_panic() {
        use viprs::{U8, adapters::codecs::JpegCodec, ports::codec::ImageDecoder};

        let adversarial = patch_jpeg_sof_dimensions(&tiny_valid_jpeg(), u16::MAX, u16::MAX);

        assert_rejects_adversarial_input(
            "JFIF wrong dimensions",
            || JpegCodec.decode::<U8>(&adversarial),
            |err| matches!(err, ViprsError::Codec(message) if message.contains("safety limit") || message.contains("Maximum supported image dimension")),
        );
    }

    #[test]
    #[cfg(feature = "jpeg")]
    fn jpeg_with_nested_icc_profile_returns_typed_error_without_panic() {
        use viprs::{U8, adapters::codecs::JpegCodec, ports::codec::ImageDecoder};

        let adversarial = insert_jpeg_icc_profile(&tiny_valid_jpeg(), 1_100_000);

        assert_rejects_adversarial_input(
            "JPEG nested ICC profile",
            || JpegCodec.decode::<U8>(&adversarial),
            |err| matches!(err, ViprsError::Codec(message) if message.contains("ICC profile")),
        );
    }

    #[test]
    #[cfg(feature = "png")]
    fn png_with_invalid_chunk_crc_returns_typed_error_without_panic() {
        use viprs::{U8, adapters::codecs::PngCodec, ports::codec::ImageDecoder};

        let adversarial = patch_png_chunk_crc(tiny_valid_png(), *b"IDAT", [0, 0, 0, 0]);

        assert_rejects_adversarial_input(
            "PNG invalid chunk CRC",
            || PngCodec::default().decode::<U8>(&adversarial),
            |err| matches!(err, ViprsError::Codec(_)),
        );
    }

    #[test]
    #[cfg(feature = "webp")]
    fn webp_with_corrupt_riff_header_returns_typed_error_without_panic() {
        use viprs::{U8, adapters::codecs::WebpCodec, ports::codec::ImageDecoder};

        let adversarial = patch_webp_riff_size(tiny_valid_webp(), 2_147_483_648u32 - 8);

        assert_rejects_adversarial_input(
            "WebP corrupt RIFF header",
            || WebpCodec.decode::<U8>(&adversarial),
            |err| matches!(err, ViprsError::Codec(message) if message.contains("RIFF size")),
        );
    }

    #[test]
    #[cfg(feature = "jpeg")]
    fn jpeg_with_multiple_sof_markers_returns_typed_error_without_panic() {
        use viprs::{U8, adapters::codecs::JpegCodec, ports::codec::ImageDecoder};

        let base = tiny_valid_jpeg();
        let sof_segment = extract_jpeg_segment(&base, 0xC0);
        let adversarial = insert_jpeg_segment_before_sos(&base, &sof_segment);

        assert_rejects_adversarial_input(
            "JPEG multiple SOF markers",
            || JpegCodec.decode::<U8>(&adversarial),
            |err| matches!(err, ViprsError::Codec(message) if message.contains("multiple SOF")),
        );
    }

    #[test]
    #[cfg(feature = "jpeg")]
    fn jpeg_with_very_long_exif_string_returns_typed_error_without_panic() {
        use viprs::{U8, adapters::codecs::JpegCodec, ports::codec::ImageDecoder};

        let adversarial = insert_jpeg_app_segment_after_soi(
            &tiny_valid_jpeg(),
            0xE1,
            &oversized_exif_string_payload(),
        );

        assert_rejects_adversarial_input(
            "JPEG very long EXIF string",
            || JpegCodec.decode::<U8>(&adversarial),
            |err| matches!(err, ViprsError::Codec(message) if message.contains("EXIF field")),
        );
    }

    #[test]
    #[cfg(feature = "png")]
    fn png_with_zlib_bomb_idat_returns_typed_error_without_panic() {
        use viprs::{U8, adapters::codecs::PngCodec, ports::codec::ImageDecoder};

        let adversarial = patch_png_ihdr_dimensions(tiny_valid_png(), 20_000, 20_000);

        assert_rejects_adversarial_input(
            "PNG zlib bomb IDAT",
            || PngCodec::default().decode::<U8>(&adversarial),
            |err| matches!(err, ViprsError::Codec(message) if message.contains("safety limit")),
        );
    }

    #[test]
    #[cfg(feature = "png")]
    fn loading_directory_as_image_returns_error_without_panic() {
        use viprs::{U8, adapters::codecs::PngCodec, ports::codec::ImageDecoder};

        let directory = project_root().join("tests");

        assert_rejects_adversarial_input(
            "directory as image",
            || PngCodec::default().decode_path::<U8>(&directory),
            |err| matches!(err, ViprsError::Io(_) | ViprsError::Codec(_)),
        );
    }
}

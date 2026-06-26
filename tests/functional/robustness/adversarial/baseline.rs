mod robustness_adversarial {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use viprs::ViprsError;

    fn assert_decode_fails_gracefully<T>(
        label: &str,
        decode: impl FnOnce() -> Result<T, ViprsError>,
        error_predicate: impl FnOnce(&ViprsError) -> bool,
    ) {
        let result = catch_unwind(AssertUnwindSafe(decode));
        match result {
            Ok(Err(err)) => {
                assert!(
                    error_predicate(&err),
                    "{label} returned the wrong typed error: {err:?}"
                );
            }
            Ok(Ok(_)) => panic!("{label} unexpectedly decoded successfully"),
            Err(_) => panic!("{label} panicked instead of returning Err(...)"),
        }
    }

    #[cfg(feature = "jpeg")]
    fn tiny_valid_jpeg() -> Vec<u8> {
        use viprs::{InMemoryImage, U8, adapters::codecs::JpegCodec, ports::codec::ImageEncoder};

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
        payload.extend_from_slice(b"II");
        payload.extend_from_slice(&42u16.to_le_bytes());
        payload.extend_from_slice(&8u32.to_le_bytes());
        payload.extend_from_slice(&1u16.to_le_bytes());
        payload
    }

    #[cfg(feature = "png")]
    fn tiny_valid_png() -> Vec<u8> {
        use viprs::{InMemoryImage, U8, adapters::codecs::PngCodec, ports::codec::ImageEncoder};

        let image =
            InMemoryImage::<U8>::from_buffer(1, 1, 3, vec![200, 100, 50]).expect("valid RGB test image");
        PngCodec::default()
            .encode(&image)
            .expect("tiny PNG fixture")
    }

    #[cfg(feature = "png")]
    fn insert_png_chunk_after_ihdr(
        png: &[u8],
        chunk_type: [u8; 4],
        declared_len: u32,
        payload: &[u8],
    ) -> Vec<u8> {
        let ihdr_len = u32::from_be_bytes([png[8], png[9], png[10], png[11]]) as usize;
        let ihdr_end = 8 + 4 + 4 + ihdr_len + 4;

        let mut with_chunk = Vec::with_capacity(png.len() + payload.len() + 8);
        with_chunk.extend_from_slice(&png[..ihdr_end]);
        with_chunk.extend_from_slice(&declared_len.to_be_bytes());
        with_chunk.extend_from_slice(&chunk_type);
        with_chunk.extend_from_slice(payload);
        with_chunk.extend_from_slice(&png[ihdr_end..]);
        with_chunk
    }

    #[test]
    #[cfg(feature = "jpeg")]
    fn jpeg_decompression_bomb_header_returns_typed_error_without_panic() {
        use viprs::{U8, adapters::codecs::JpegCodec, ports::codec::ImageDecoder};

        // JPEG SOF dimensions are 16-bit, so use a near-maximum representable size
        // to exercise the same decompression-bomb path as the task's
        // 100000×100000 scenario without tripping the codec's hard format ceiling.
        let adversarial = patch_jpeg_sof_dimensions(&tiny_valid_jpeg(), 65_000, 65_000);

        assert_decode_fails_gracefully(
            "JPEG decompression bomb",
            || JpegCodec.decode::<U8>(&adversarial),
            |err| matches!(err, ViprsError::Codec(message) if message.contains("safety limit")),
        );
    }

    #[test]
    #[cfg(feature = "jpeg")]
    fn jpeg_with_malformed_exif_returns_typed_error_without_panic() {
        use viprs::{U8, adapters::codecs::JpegCodec, ports::codec::ImageDecoder};

        let adversarial =
            insert_jpeg_app_segment_after_soi(&tiny_valid_jpeg(), 0xE1, &malformed_exif_payload());

        assert_decode_fails_gracefully(
            "JPEG malformed EXIF",
            || JpegCodec.decode::<U8>(&adversarial),
            |err| matches!(err, ViprsError::Codec(message) if message.contains("EXIF")),
        );
    }

    #[test]
    #[cfg(feature = "png")]
    fn png_with_unreasonable_iccp_claim_returns_typed_error_without_panic() {
        use viprs::{U8, adapters::codecs::PngCodec, ports::codec::ImageDecoder};

        let adversarial =
            insert_png_chunk_after_ihdr(&tiny_valid_png(), *b"iCCP", 0x7FFF_FFF0, b"icc\0\0x");

        assert_decode_fails_gracefully(
            "PNG unreasonable iCCP payload",
            || PngCodec::default().decode::<U8>(&adversarial),
            |err| matches!(err, ViprsError::Codec(_)),
        );
    }
}

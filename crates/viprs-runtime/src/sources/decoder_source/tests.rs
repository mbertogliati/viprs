use super::*;
use crate::domain::codec_options::LoadOptions;
use crate::domain::format::U8;
use crate::domain::image::{ImageMetadata, Interpretation};
use crate::ports::codec::{ImageDecoder, ImageMetadataProbe, TileImageDecoder};
use std::fs;
use std::num::NonZeroU8;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

fn test_input_path(name: &str, extension: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("decoder-source-unit-tests");
    fs::create_dir_all(&dir).unwrap();
    dir.join(format!("{name}-{}.{}", std::process::id(), extension))
}

/// Minimal no-op decoder that always fails — used to test error propagation.
struct AlwaysFailDecoder;

impl ImageDecoder for AlwaysFailDecoder {
    fn format_name(&self) -> &'static str {
        "fail"
    }
    fn sniff(&self, _: &[u8]) -> bool {
        false
    }
    fn decode<F: BandFormat>(&self, _: &[u8]) -> Result<InMemoryImage<F>, ViprsError> {
        Err(ViprsError::Codec("always fails".into()))
    }
    fn decode_with_options<F: BandFormat>(
        &self,
        src: &[u8],
        _opts: &LoadOptions,
    ) -> Result<InMemoryImage<F>, ViprsError> {
        self.decode(src)
    }
    fn probe(&self, _: &[u8]) -> Result<(u32, u32, u32), ViprsError> {
        Err(ViprsError::Codec("always fails".into()))
    }
}

#[test]
fn new_propagates_decoder_error() {
    let result = DecoderSource::<_, U8>::new(AlwaysFailDecoder, b"fake");
    assert!(result.is_err());
}

#[test]
fn with_options_propagates_decoder_error() {
    let opts = LoadOptions::default().with_shrink(NonZeroU8::new(2).unwrap());
    let result = DecoderSource::<_, U8>::with_options(AlwaysFailDecoder, b"fake", opts);
    assert!(result.is_err());
}

struct MetadataDecoder;

impl ImageDecoder for MetadataDecoder {
    fn format_name(&self) -> &'static str {
        "metadata"
    }

    fn sniff(&self, _: &[u8]) -> bool {
        true
    }

    fn decode<F: BandFormat>(&self, _: &[u8]) -> Result<InMemoryImage<F>, ViprsError> {
        self.decode_with_options::<F>(&[], &LoadOptions::default())
    }

    fn decode_with_options<F: BandFormat>(
        &self,
        _: &[u8],
        _: &LoadOptions,
    ) -> Result<InMemoryImage<F>, ViprsError> {
        if F::ID != U8::ID {
            return Err(ViprsError::Codec(
                "metadata decoder only supports U8".into(),
            ));
        }

        let mut metadata = ImageMetadata::default();
        metadata.interpretation = Some(Interpretation::Srgb);
        let image = InMemoryImage::from_buffer(1, 1, 1, vec![1u8])
            .map_err(|e| ViprsError::Codec(e.to_string()))?
            .with_metadata(metadata);

        let cast = {
            // SAFETY: `BandFormat` is sealed, so `F::ID == U8` implies
            // `F::Sample == u8` and `Image<U8>` has the same layout as `Image<F>`.
            unsafe { std::mem::transmute::<InMemoryImage<U8>, InMemoryImage<F>>(image) }
        };
        Ok(cast)
    }

    fn probe(&self, _: &[u8]) -> Result<(u32, u32, u32), ViprsError> {
        Ok((1, 1, 1))
    }
}

#[test]
fn metadata_delegates_to_decoded_image() {
    let src = DecoderSource::<_, U8>::new(MetadataDecoder, b"fake").unwrap();
    assert_eq!(src.metadata().interpretation, Some(Interpretation::Srgb));
}

struct PathOnlyDecoder;

impl ImageDecoder for PathOnlyDecoder {
    fn format_name(&self) -> &'static str {
        "path-only"
    }

    fn sniff(&self, _: &[u8]) -> bool {
        false
    }

    fn decode<F: BandFormat>(&self, _: &[u8]) -> Result<InMemoryImage<F>, ViprsError> {
        Err(ViprsError::Codec(
            "path-only: byte decode unavailable".into(),
        ))
    }

    fn decode_with_options<F: BandFormat>(
        &self,
        _src: &[u8],
        _opts: &LoadOptions,
    ) -> Result<InMemoryImage<F>, ViprsError> {
        Err(ViprsError::Codec(
            "path-only: byte decode unavailable".into(),
        ))
    }

    fn decode_path_with_options<F: BandFormat>(
        &self,
        _path: &Path,
        _opts: &LoadOptions,
    ) -> Result<InMemoryImage<F>, ViprsError> {
        if F::ID != U8::ID {
            return Err(ViprsError::Codec("path-only: only U8 is supported".into()));
        }

        let image = InMemoryImage::<U8>::from_buffer(2, 1, 1, vec![7, 9])?;
        Ok({
            // SAFETY: `BandFormat` is sealed, so `F::ID == U8` implies
            // `F::Sample == u8` and `Image<U8>` has the same layout as `Image<F>`.
            unsafe { std::mem::transmute::<InMemoryImage<U8>, InMemoryImage<F>>(image) }
        })
    }

    fn probe(&self, _src: &[u8]) -> Result<(u32, u32, u32), ViprsError> {
        Err(ViprsError::Codec(
            "path-only: byte probe unavailable".into(),
        ))
    }

    fn probe_path(&self, _path: &Path) -> Result<(u32, u32, u32), ViprsError> {
        Ok((2, 1, 1))
    }
}

#[test]
fn from_path_uses_decoder_path_api() {
    let input_path = test_input_path("path-only", "bin");
    fs::write(&input_path, []).unwrap();

    let source = DecoderSource::<_, U8>::from_path(PathOnlyDecoder, &input_path).unwrap();

    assert_eq!(source.width(), 2);
    assert_eq!(source.height(), 1);
    assert_eq!(source.image().unwrap().pixels(), &[7, 9]);

    fs::remove_file(input_path).unwrap();
}

#[test]
fn with_options_records_requested_shrink_factor() {
    let opts = LoadOptions::default().with_shrink(NonZeroU8::new(2).unwrap());
    let source = DecoderSource::<_, U8>::with_options(Fixed4x4Decoder, b"", opts).unwrap();

    assert_eq!(source.shrink_factor(), 2);
    assert_eq!(source.load_options().shrink_factor, NonZeroU8::new(2));
    assert_eq!(source.width(), 2);
    assert_eq!(source.height(), 2);
}

struct TrackingDecoder {
    seen_factors: Arc<Mutex<Vec<u8>>>,
}

impl ImageDecoder for TrackingDecoder {
    fn format_name(&self) -> &'static str {
        "tracking"
    }

    fn sniff(&self, _: &[u8]) -> bool {
        true
    }

    fn decode<F: BandFormat>(&self, src: &[u8]) -> Result<InMemoryImage<F>, ViprsError> {
        self.decode_with_options(src, &LoadOptions::default())
    }

    fn decode_with_options<F: BandFormat>(
        &self,
        _: &[u8],
        opts: &LoadOptions,
    ) -> Result<InMemoryImage<F>, ViprsError> {
        if F::ID != U8::ID {
            return Err(ViprsError::Codec(
                "tracking decoder only supports U8".into(),
            ));
        }

        let factor = opts.shrink_factor.map_or(1, NonZeroU8::get);
        self.seen_factors.lock().unwrap().push(factor);
        let width = (8 / u32::from(factor)).max(1);
        let height = (8 / u32::from(factor)).max(1);
        let image = InMemoryImage::from_buffer(width, height, 1, vec![0u8; (width * height) as usize])
            .map_err(|e| ViprsError::Codec(e.to_string()))?;

        let cast = {
            // SAFETY: `BandFormat` is sealed, so `F::ID == U8` implies
            // `F::Sample == u8` and `Image<U8>` has the same layout as `Image<F>`.
            unsafe { std::mem::transmute::<InMemoryImage<U8>, InMemoryImage<F>>(image) }
        };
        Ok(cast)
    }

    fn probe(&self, _: &[u8]) -> Result<(u32, u32, u32), ViprsError> {
        Ok((8, 8, 1))
    }
}

#[test]
fn set_shrink_on_load_reports_view_only_fallback_after_eager_materialization() {
    let seen_factors = Arc::new(Mutex::new(Vec::new()));
    let decoder = TrackingDecoder {
        seen_factors: Arc::clone(&seen_factors),
    };
    let mut source = DecoderSource::<_, U8>::new(decoder, b"encoded").unwrap();

    assert_eq!(source.width(), 8);
    assert_eq!(source.height(), 8);
    assert_eq!(source.shrink_factor(), 1);
    assert_eq!(source.load_options().shrink_factor, None);
    assert_eq!(&*seen_factors.lock().unwrap(), &[1]);

    let applied = source
        .set_shrink_on_load(NonZeroU8::new(4).unwrap())
        .unwrap();

    assert!(!applied);
    assert_eq!(source.width(), 2);
    assert_eq!(source.height(), 2);
    assert_eq!(source.shrink_factor(), 4);
    assert_eq!(source.load_options().shrink_factor, NonZeroU8::new(4));
    assert_eq!(&*seen_factors.lock().unwrap(), &[1]);
}

#[test]
fn set_thumbnail_shrink_on_load_reopens_jpeg_decoder_natively() {
    struct JpegTrackingDecoder {
        seen_factors: Arc<Mutex<Vec<u8>>>,
    }

    impl ImageDecoder for JpegTrackingDecoder {
        fn format_name(&self) -> &'static str {
            "jpeg"
        }

        fn sniff(&self, _: &[u8]) -> bool {
            true
        }

        fn decode<F: BandFormat>(&self, src: &[u8]) -> Result<InMemoryImage<F>, ViprsError> {
            self.decode_with_options(src, &LoadOptions::default())
        }

        fn decode_with_options<F: BandFormat>(
            &self,
            _: &[u8],
            opts: &LoadOptions,
        ) -> Result<InMemoryImage<F>, ViprsError> {
            if F::ID != U8::ID {
                return Err(ViprsError::Codec(
                    "jpeg tracking decoder only supports U8".into(),
                ));
            }

            let factor = opts.shrink_factor.map_or(1, NonZeroU8::get);
            self.seen_factors.lock().unwrap().push(factor);
            let width = (8 / u32::from(factor)).max(1);
            let height = (8 / u32::from(factor)).max(1);
            let image = InMemoryImage::from_buffer(width, height, 1, vec![0u8; (width * height) as usize])
                .map_err(|e| ViprsError::Codec(e.to_string()))?;

            let cast = {
                // SAFETY: `BandFormat` is sealed, so `F::ID == U8` implies
                // `F::Sample == u8` and `Image<U8>` has the same layout as `Image<F>`.
                unsafe { std::mem::transmute::<InMemoryImage<U8>, InMemoryImage<F>>(image) }
            };
            Ok(cast)
        }

        fn probe(&self, _: &[u8]) -> Result<(u32, u32, u32), ViprsError> {
            Ok((8, 8, 1))
        }
    }

    let seen_factors = Arc::new(Mutex::new(Vec::new()));
    let mut source = DecoderSource::<_, U8>::new(
        JpegTrackingDecoder {
            seen_factors: Arc::clone(&seen_factors),
        },
        b"encoded",
    )
    .unwrap();

    assert!(
        source
            .set_thumbnail_shrink_on_load(NonZeroU8::new(4).unwrap())
            .unwrap()
    );
    assert_eq!(source.width(), 2);
    assert_eq!(source.height(), 2);
    assert_eq!(source.image().unwrap().width(), 2);
    assert_eq!(source.image().unwrap().height(), 2);
    assert_eq!(&*seen_factors.lock().unwrap(), &[1, 4]);
}

#[cfg(feature = "jpeg")]
#[test]
fn set_thumbnail_shrink_on_load_materializes_real_jpeg_backing_at_dct_scale() {
    use crate::adapters::codecs::JpegCodec;

    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("images")
        .join("bench_8192x8192.jpg");
    let mut source = DecoderSource::<_, U8>::probed_path(JpegCodec, &path).unwrap();

    assert_eq!((source.width(), source.height()), (8192, 8192));
    assert!(source.image().is_none());

    assert!(
        source
            .set_thumbnail_shrink_on_load(NonZeroU8::new(8).unwrap())
            .unwrap()
    );

    assert_eq!(source.shrink_factor(), 8);
    assert_eq!((source.width(), source.height()), (1024, 1024));

    let backing = source.image().unwrap();
    assert_eq!((backing.width(), backing.height()), (1024, 1024));
    assert_eq!(source.resident_decoded_bytes(), 1024 * 1024 * 3);
}

#[cfg(feature = "jpeg")]
#[test]
fn set_thumbnail_shrink_on_load_materializes_residual_box_shrink_after_jpeg_dct_scale() {
    use crate::adapters::codecs::JpegCodec;

    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("images")
        .join("bench_8192x8192.jpg");
    let mut source = DecoderSource::<_, U8>::probed_path(JpegCodec, &path).unwrap();

    assert!(
        source
            .set_thumbnail_shrink_on_load(NonZeroU8::new(16).unwrap())
            .unwrap()
    );

    assert_eq!(source.shrink_factor(), 16);
    assert_eq!((source.width(), source.height()), (512, 512));

    let backing = source.image().unwrap();
    assert_eq!((backing.width(), backing.height()), (512, 512));
    assert_eq!(source.resident_decoded_bytes(), 512 * 512 * 3);
}

#[test]
fn set_thumbnail_shrink_on_load_applies_software_box_shrink_for_png() {
    struct PngTrackingDecoder {
        seen_factors: Arc<Mutex<Vec<u8>>>,
    }

    impl ImageDecoder for PngTrackingDecoder {
        fn format_name(&self) -> &'static str {
            "png"
        }

        fn sniff(&self, _: &[u8]) -> bool {
            true
        }

        fn decode<F: BandFormat>(&self, src: &[u8]) -> Result<InMemoryImage<F>, ViprsError> {
            self.decode_with_options(src, &LoadOptions::default())
        }

        fn decode_with_options<F: BandFormat>(
            &self,
            _: &[u8],
            opts: &LoadOptions,
        ) -> Result<InMemoryImage<F>, ViprsError> {
            if F::ID != U8::ID {
                return Err(ViprsError::Codec(
                    "png tracking decoder only supports U8".into(),
                ));
            }

            let factor = opts.shrink_factor.map_or(1, NonZeroU8::get);
            self.seen_factors.lock().unwrap().push(factor);
            let image = InMemoryImage::from_buffer(8, 8, 1, vec![128u8; 64])
                .map_err(|e| ViprsError::Codec(e.to_string()))?;

            let cast = {
                // SAFETY: `BandFormat` is sealed, so `F::ID == U8` implies
                // `F::Sample == u8` and `Image<U8>` has the same layout as `Image<F>`.
                unsafe { std::mem::transmute::<InMemoryImage<U8>, InMemoryImage<F>>(image) }
            };
            Ok(cast)
        }

        fn probe(&self, _: &[u8]) -> Result<(u32, u32, u32), ViprsError> {
            Ok((8, 8, 1))
        }
    }

    let seen_factors = Arc::new(Mutex::new(Vec::new()));
    let mut source = DecoderSource::<_, U8>::new(
        PngTrackingDecoder {
            seen_factors: Arc::clone(&seen_factors),
        },
        b"encoded",
    )
    .unwrap();

    // PNG now supports software box shrink: set_thumbnail_shrink_on_load returns true.
    assert!(
        source
            .set_thumbnail_shrink_on_load(NonZeroU8::new(4).unwrap())
            .unwrap()
    );
    // The backing image is shrunk in-place: 8×8 → 2×2 at factor 4.
    assert_eq!(source.width(), 2);
    assert_eq!(source.height(), 2);
    // Only the initial eager decode (factor=1) happens; no re-decode for shrink.
    assert_eq!(&*seen_factors.lock().unwrap(), &[1]);
}

#[test]
fn with_options_forwards_shrink_to_eager_decoder_and_keeps_resident_image_shrunk() {
    let seen_factors = Arc::new(Mutex::new(Vec::new()));
    let encoded = b"encoded";
    let full = DecoderSource::<_, U8>::new(
        TrackingDecoder {
            seen_factors: Arc::clone(&seen_factors),
        },
        encoded,
    )
    .unwrap();
    let shrunk = DecoderSource::<_, U8>::with_options(
        TrackingDecoder {
            seen_factors: Arc::clone(&seen_factors),
        },
        encoded,
        LoadOptions::default().with_shrink(NonZeroU8::new(4).unwrap()),
    )
    .unwrap();

    assert_eq!(&*seen_factors.lock().unwrap(), &[1, 4]);
    assert_eq!(shrunk.shrink_factor(), 4);
    assert_eq!(shrunk.load_options().shrink_factor, NonZeroU8::new(4));
    assert_eq!(shrunk.width(), 2);
    assert_eq!(shrunk.height(), 2);
    assert_eq!(shrunk.image().unwrap().width(), 2);
    assert_eq!(shrunk.image().unwrap().height(), 2);
    assert!(shrunk.resident_decoded_bytes() < full.resident_decoded_bytes());
}

#[test]
fn eager_source_reports_resident_decoded_bytes() {
    let source = DecoderSource::<_, U8>::new(MetadataDecoder, b"fake").unwrap();

    assert!(!source.is_streaming());
    assert_eq!(source.resident_decoded_bytes(), 1);
    assert!(source.image().is_some());
}

#[test]
fn format_name_delegates_to_decoder() {
    // Instantiate a real DecoderSource to exercise the adapter forwarding path.
    let src = make_4x4_source();
    assert_eq!(src.format_name(), "fixed4x4");
}

#[test]
fn access_mode_markers_are_zero_sized() {
    assert_eq!(std::mem::size_of::<RandomAccess>(), 0);
    assert_eq!(std::mem::size_of::<Sequential>(), 0);
}

/// Decoder that produces a fixed 4×4 single-band U8 image regardless of
/// the input buffer.  Used to exercise `read_region` without a real codec.
struct Fixed4x4Decoder;

impl ImageDecoder for Fixed4x4Decoder {
    fn format_name(&self) -> &'static str {
        "fixed4x4"
    }
    fn sniff(&self, _: &[u8]) -> bool {
        true
    }
    fn decode<F: BandFormat>(&self, _: &[u8]) -> Result<InMemoryImage<F>, ViprsError> {
        // Only valid for U8; other formats are unsupported in this stub.
        Err(ViprsError::Codec(
            "Fixed4x4Decoder only supports U8 via decode_with_options".into(),
        ))
    }
    fn decode_with_options<F: BandFormat>(
        &self,
        _src: &[u8],
        _opts: &LoadOptions,
    ) -> Result<InMemoryImage<F>, ViprsError> {
        // pixel at (x, y) = y * 4 + x  (values 0..15), single band
        // We produce a U8 image; for non-U8 F this will fail the buffer cast,
        // but the tests below always use U8.
        let data: Vec<u8> = (0u8..16).collect();
        // Transmute the Vec<u8> into Vec<F::Sample> — only valid when F = U8.
        // SAFETY: This is a test-only stub; callers always pass F = U8 so
        // `size_of::<F::Sample>() == 1` and the reinterpretation is sound.
        // Production codecs would branch on `F::ID` instead.
        let sample_data = if std::mem::size_of::<F::Sample>() == 1 {
            // SAFETY: F::Sample is 1 byte (U8), and u8 has no invalid bit patterns.
            let ptr = data.as_ptr().cast::<F::Sample>();
            let len = data.len();
            std::mem::forget(data);
            // SAFETY: ownership of the forgotten `Vec<u8>` allocation is transferred into a layout-identical `Vec<F::Sample>`.
            unsafe { Vec::from_raw_parts(ptr.cast_mut(), len, len) }
        } else {
            return Err(ViprsError::Codec("Fixed4x4Decoder only supports U8".into()));
        };
        InMemoryImage::from_buffer(4, 4, 1, sample_data).map_err(|e| ViprsError::Codec(e.to_string()))
    }
    fn probe(&self, _: &[u8]) -> Result<(u32, u32, u32), ViprsError> {
        Ok((4, 4, 1))
    }
}

/// Build a `DecoderSource<'static, Fixed4x4Decoder, U8>` backed by the 4×4 test image.
fn make_4x4_source() -> DecoderSource<'static, Fixed4x4Decoder, U8> {
    DecoderSource::new(Fixed4x4Decoder, b"").unwrap()
}

#[test]
fn read_region_returns_correct_pixels() {
    let src = make_4x4_source();
    // Central 2×2 region starting at (1, 1).
    // pixel (x=1, y=1) = 1*4+1 = 5
    // pixel (x=2, y=1) = 1*4+2 = 6
    // pixel (x=1, y=2) = 2*4+1 = 9
    // pixel (x=2, y=2) = 2*4+2 = 10
    let region = Region::new(1, 1, 2, 2);
    let mut output = vec![0u8; 4];
    src.read_region(region, &mut output).unwrap();
    assert_eq!(output, vec![5, 6, 9, 10]);
}

#[test]
fn set_shrink_on_load_updates_view_without_redecoding() {
    struct BlockDecoder {
        seen_factors: Arc<Mutex<Vec<u8>>>,
    }

    impl ImageDecoder for BlockDecoder {
        fn format_name(&self) -> &'static str {
            "block"
        }

        fn sniff(&self, _: &[u8]) -> bool {
            true
        }

        fn decode<F: BandFormat>(&self, src: &[u8]) -> Result<InMemoryImage<F>, ViprsError> {
            self.decode_with_options(src, &LoadOptions::default())
        }

        fn decode_with_options<F: BandFormat>(
            &self,
            _: &[u8],
            opts: &LoadOptions,
        ) -> Result<InMemoryImage<F>, ViprsError> {
            if F::ID != U8::ID {
                return Err(ViprsError::Codec("block decoder only supports U8".into()));
            }

            let factor = opts.shrink_factor.map_or(1, NonZeroU8::get);
            self.seen_factors.lock().unwrap().push(factor);
            let mut pixels = vec![0u8; 8 * 8];
            for y in 0..8usize {
                for x in 0..8usize {
                    let value = match (x / 4, y / 4) {
                        (0, 0) => 10,
                        (1, 0) => 20,
                        (0, 1) => 30,
                        _ => 40,
                    };
                    pixels[y * 8 + x] = value;
                }
            }
            let image = InMemoryImage::from_buffer(8, 8, 1, pixels)
                .map_err(|e| ViprsError::Codec(e.to_string()))?;

            let cast = {
                // SAFETY: `BandFormat` is sealed, so `F::ID == U8` implies
                // `F::Sample == u8` and `Image<U8>` has the same layout as `Image<F>`.
                unsafe { std::mem::transmute::<InMemoryImage<U8>, InMemoryImage<F>>(image) }
            };
            Ok(cast)
        }

        fn probe(&self, _: &[u8]) -> Result<(u32, u32, u32), ViprsError> {
            Ok((8, 8, 1))
        }
    }

    let seen_factors = Arc::new(Mutex::new(Vec::new()));
    let mut source = DecoderSource::<_, U8>::new(
        BlockDecoder {
            seen_factors: Arc::clone(&seen_factors),
        },
        b"encoded",
    )
    .unwrap();
    let resident_bytes_before = source.resident_decoded_bytes();
    assert_eq!(resident_bytes_before, 64);

    assert!(
        !source
            .set_shrink_on_load(NonZeroU8::new(2).unwrap())
            .unwrap()
    );
    assert_eq!(source.width(), 4);
    assert_eq!(source.height(), 4);
    assert!(
        !source
            .set_shrink_on_load(NonZeroU8::new(4).unwrap())
            .unwrap()
    );
    assert_eq!(source.width(), 2);
    assert_eq!(source.height(), 2);
    assert_eq!(source.resident_decoded_bytes(), resident_bytes_before);

    let mut output = vec![0u8; 4];
    source
        .read_region(Region::new(0, 0, 2, 2), &mut output)
        .unwrap();

    assert_eq!(output, vec![10, 20, 30, 40]);
    assert_eq!(&*seen_factors.lock().unwrap(), &[1]);
}

#[test]
fn read_region_clamps_negative_coordinates() {
    let src = make_4x4_source();
    // Region at (-2, -2) size 2×2 — all coordinates clamp to (0, 0) => value 0.
    let region = Region::new(-2, -2, 2, 2);
    let mut output = vec![0u8; 4];
    src.read_region(region, &mut output).unwrap();
    assert_eq!(output, vec![0, 0, 0, 0]);
}

#[test]
fn read_region_clamps_out_of_bounds_right_bottom() {
    let src = make_4x4_source();
    // Region at (3, 3) size 2×2 — (4,3), (3,4), (4,4) clamp to edge.
    // pixel (3,3)=15, (4→3,3)=15, (3,4→3)=15, (4→3,4→3)=15
    let region = Region::new(3, 3, 2, 2);
    let mut output = vec![0u8; 4];
    src.read_region(region, &mut output).unwrap();
    assert_eq!(output, vec![15, 15, 15, 15]);
}

#[test]
fn read_region_rejects_regions_whose_x_end_overflows_i32() {
    let src = make_4x4_source();
    let mut output = vec![0u8; 1];

    let err = src
        .read_region(Region::new(i32::MAX, 0, 1, 1), &mut output)
        .unwrap_err();

    assert!(matches!(err, ViprsError::Codec(message) if message.contains("out of bounds")));
}

#[test]
fn width_height_bands_delegate_to_image() {
    let src = make_4x4_source();
    assert_eq!(src.width(), 4);
    assert_eq!(src.height(), 4);
    assert_eq!(src.bands(), 1);
}

#[test]
fn demand_hint_is_small_tile() {
    let src = make_4x4_source();
    assert_eq!(src.demand_hint(), DemandHint::SmallTile);
}

struct StreamingGridDecoder {
    full_decodes: Arc<Mutex<usize>>,
    region_decodes: Arc<Mutex<Vec<Region>>>,
    seen_probe_factors: Arc<Mutex<Vec<u8>>>,
    seen_decode_factors: Arc<Mutex<Vec<u8>>>,
}

impl StreamingGridDecoder {
    fn new() -> Self {
        Self {
            full_decodes: Arc::new(Mutex::new(0)),
            region_decodes: Arc::new(Mutex::new(Vec::new())),
            seen_probe_factors: Arc::new(Mutex::new(Vec::new())),
            seen_decode_factors: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl ImageDecoder for StreamingGridDecoder {
    fn format_name(&self) -> &'static str {
        "webp"
    }

    fn sniff(&self, _: &[u8]) -> bool {
        true
    }

    fn decode<F: BandFormat>(&self, _: &[u8]) -> Result<InMemoryImage<F>, ViprsError> {
        *self.full_decodes.lock().unwrap() += 1;
        Err(ViprsError::Codec(
            "streaming-grid must not full-decode".into(),
        ))
    }

    fn decode_with_options<F: BandFormat>(
        &self,
        _src: &[u8],
        opts: &LoadOptions,
    ) -> Result<InMemoryImage<F>, ViprsError> {
        *self.full_decodes.lock().unwrap() += 1;
        if F::ID != U8::ID {
            return Err(ViprsError::Codec("streaming-grid only decodes U8".into()));
        }
        let factor = opts.shrink_factor.map_or(1, NonZeroU8::get);
        let dim = (8 / u32::from(factor)).max(1);
        let mut pixels = vec![0u8; (dim * dim) as usize];
        for row in 0..dim {
            for col in 0..dim {
                pixels[(row * dim + col) as usize] = (row as u8) * 10 + col as u8;
            }
        }
        // SAFETY: U8::Sample is u8, and we verified F::ID == U8::ID above.
        let typed_pixels: Vec<F::Sample> = bytemuck::cast_vec(pixels);
        InMemoryImage::from_buffer(dim, dim, 1, typed_pixels)
    }

    fn probe(&self, _: &[u8]) -> Result<(u32, u32, u32), ViprsError> {
        Ok((8, 8, 1))
    }
}

impl TileImageDecoder for StreamingGridDecoder {
    fn probe_with_options(
        &self,
        src: &[u8],
        opts: &LoadOptions,
    ) -> Result<ImageMetadataProbe, ViprsError> {
        assert_eq!(src, b"encoded-grid");
        let factor = opts.shrink_factor.map_or(1, NonZeroU8::get);
        self.seen_probe_factors.lock().unwrap().push(factor);

        let mut metadata = ImageMetadata::default();
        metadata.interpretation = Some(Interpretation::Srgb);
        Ok(
            ImageMetadataProbe::new(8 / u32::from(factor), 8 / u32::from(factor), 1)
                .with_metadata(metadata),
        )
    }

    fn decode_region_into<F: BandFormat>(
        &self,
        src: &[u8],
        opts: &LoadOptions,
        region: Region,
        output: &mut [u8],
    ) -> Result<(), ViprsError> {
        assert_eq!(src, b"encoded-grid");
        if F::ID != U8::ID {
            return Err(ViprsError::Codec("streaming-grid only decodes U8".into()));
        }

        let factor = opts.shrink_factor.map_or(1, NonZeroU8::get);
        self.seen_decode_factors.lock().unwrap().push(factor);
        self.region_decodes.lock().unwrap().push(region);

        let width = i64::from((8 / u32::from(factor)).max(1));
        let height = i64::from((8 / u32::from(factor)).max(1));
        let expected = region.pixel_count();
        if output.len() != expected {
            return Err(ViprsError::Codec(format!(
                "streaming-grid output mismatch: got {}, expected {expected}",
                output.len()
            )));
        }

        for row in 0..region.height as usize {
            for col in 0..region.width as usize {
                let x = (i64::from(region.x) + col as i64).clamp(0, width - 1) as u8;
                let y = (i64::from(region.y) + row as i64).clamp(0, height - 1) as u8;
                output[row * region.width as usize + col] = y * 10 + x;
            }
        }

        Ok(())
    }
}

#[test]
fn streaming_constructor_probes_without_full_decode() {
    let decoder = StreamingGridDecoder::new();
    let full_decodes = Arc::clone(&decoder.full_decodes);
    let seen_probe_factors = Arc::clone(&decoder.seen_probe_factors);
    let source =
        DecoderSource::<_, U8>::streaming(decoder, b"encoded-grid", LoadOptions::default())
            .unwrap();

    assert!(source.is_streaming());
    assert_eq!(source.resident_decoded_bytes(), 0);
    assert!(source.image().is_none());
    assert_eq!(source.width(), 8);
    assert_eq!(source.height(), 8);
    assert_eq!(source.bands(), 1);
    assert_eq!(source.metadata().interpretation, Some(Interpretation::Srgb));
    assert_eq!(*full_decodes.lock().unwrap(), 0);
    assert_eq!(&*seen_probe_factors.lock().unwrap(), &[1]);
}

#[test]
fn streaming_read_region_delegates_exact_tile_to_decoder() {
    let decoder = StreamingGridDecoder::new();
    let full_decodes = Arc::clone(&decoder.full_decodes);
    let region_decodes = Arc::clone(&decoder.region_decodes);
    let seen_decode_factors = Arc::clone(&decoder.seen_decode_factors);
    let source =
        DecoderSource::<_, U8>::streaming(decoder, b"encoded-grid", LoadOptions::default())
            .unwrap();

    let region = Region::new(1, 2, 2, 2);
    let mut output = vec![0u8; region.pixel_count()];
    source.read_region(region, &mut output).unwrap();

    assert_eq!(output, vec![21, 22, 31, 32]);
    assert_eq!(*full_decodes.lock().unwrap(), 0);
    assert_eq!(&*region_decodes.lock().unwrap(), &[region]);
    assert_eq!(&*seen_decode_factors.lock().unwrap(), &[1]);
}

#[test]
fn streaming_source_forwards_normalized_shrink_to_tile_decoder() {
    let decoder = StreamingGridDecoder::new();
    let seen_probe_factors = Arc::clone(&decoder.seen_probe_factors);
    let seen_decode_factors = Arc::clone(&decoder.seen_decode_factors);
    let opts = LoadOptions::default().with_shrink(NonZeroU8::new(2).unwrap());
    let source = DecoderSource::<_, U8>::streaming(decoder, b"encoded-grid", opts).unwrap();

    assert_eq!(source.shrink_factor(), 2);
    assert_eq!(source.width(), 4);
    assert_eq!(source.height(), 4);

    let mut output = vec![0u8; 4];
    source
        .read_region(Region::new(0, 0, 2, 2), &mut output)
        .unwrap();

    assert_eq!(output, vec![0, 1, 10, 11]);
    assert_eq!(&*seen_probe_factors.lock().unwrap(), &[2]);
    assert_eq!(&*seen_decode_factors.lock().unwrap(), &[2]);
}

#[test]
fn streaming_thumbnail_hint_eagerly_decodes_with_shrink() {
    let decoder = StreamingGridDecoder::new();
    let full_decodes = Arc::clone(&decoder.full_decodes);
    let seen_probe_factors = Arc::clone(&decoder.seen_probe_factors);
    let seen_decode_factors = Arc::clone(&decoder.seen_decode_factors);
    let mut source =
        DecoderSource::<_, U8>::streaming(decoder, b"encoded-grid", LoadOptions::default())
            .unwrap();

    assert!(
        source
            .set_thumbnail_shrink_on_load(NonZeroU8::new(4).unwrap())
            .unwrap()
    );
    assert_eq!(source.width(), 2);
    assert_eq!(source.height(), 2);

    // The thumbnail hint triggers a single eager decode (no per-tile decode).
    assert_eq!(*full_decodes.lock().unwrap(), 1);

    let mut output = vec![0u8; 4];
    source
        .read_region(Region::new(0, 0, 2, 2), &mut output)
        .unwrap();

    assert_eq!(output, vec![0, 1, 10, 11]);
    // No additional decodes after the eager materialization.
    assert_eq!(*full_decodes.lock().unwrap(), 1);
    // Probe was called at construction time only (factor 1).
    assert_eq!(&*seen_probe_factors.lock().unwrap(), &[1]);
    // No region decodes — tiles served from eager backing.
    assert_eq!(&*seen_decode_factors.lock().unwrap(), &[] as &[u8]);
}

struct PathOnlyStreamingDecoder;

impl ImageDecoder for PathOnlyStreamingDecoder {
    fn format_name(&self) -> &'static str {
        "path-streaming"
    }

    fn sniff(&self, _: &[u8]) -> bool {
        false
    }

    fn decode<F: BandFormat>(&self, _: &[u8]) -> Result<InMemoryImage<F>, ViprsError> {
        Err(ViprsError::Codec(
            "path-streaming: full byte decode unavailable".into(),
        ))
    }

    fn decode_with_options<F: BandFormat>(
        &self,
        _src: &[u8],
        _opts: &LoadOptions,
    ) -> Result<InMemoryImage<F>, ViprsError> {
        Err(ViprsError::Codec(
            "path-streaming: full byte decode unavailable".into(),
        ))
    }

    fn probe(&self, _src: &[u8]) -> Result<(u32, u32, u32), ViprsError> {
        Err(ViprsError::Codec(
            "path-streaming: byte probe unavailable".into(),
        ))
    }
}

impl TileImageDecoder for PathOnlyStreamingDecoder {
    fn probe_with_options(
        &self,
        _src: &[u8],
        _opts: &LoadOptions,
    ) -> Result<ImageMetadataProbe, ViprsError>
    where
        Self: Sized,
    {
        Err(ViprsError::Codec(
            "path-streaming: byte probe unavailable".into(),
        ))
    }

    fn probe_path_with_options(
        &self,
        _path: &Path,
        _opts: &LoadOptions,
    ) -> Result<ImageMetadataProbe, ViprsError>
    where
        Self: Sized,
    {
        Ok(ImageMetadataProbe::new(3, 2, 1))
    }

    fn decode_region_into<F: BandFormat>(
        &self,
        _src: &[u8],
        _opts: &LoadOptions,
        _region: Region,
        _output: &mut [u8],
    ) -> Result<(), ViprsError>
    where
        Self: Sized,
    {
        Err(ViprsError::Codec(
            "path-streaming: byte tile decode unavailable".into(),
        ))
    }

    fn decode_region_from_path<F: BandFormat>(
        &self,
        _path: &Path,
        _opts: &LoadOptions,
        region: Region,
        output: &mut [u8],
    ) -> Result<(), ViprsError>
    where
        Self: Sized,
    {
        if F::ID != U8::ID {
            return Err(ViprsError::Codec(
                "path-streaming: only U8 is supported".into(),
            ));
        }

        for y in 0..region.height {
            for x in 0..region.width {
                output[(y * region.width + x) as usize] =
                    ((region.y + y as i32) * 3 + region.x + x as i32) as u8;
            }
        }
        Ok(())
    }
}

#[test]
fn streaming_path_uses_decoder_path_api() {
    let input_path = test_input_path("path-streaming", "bin");
    fs::write(&input_path, []).unwrap();

    let source = DecoderSource::<_, U8>::streaming_path(
        PathOnlyStreamingDecoder,
        &input_path,
        LoadOptions::default(),
    )
    .unwrap();
    let mut output = vec![0u8; 4];
    source
        .read_region(Region::new(1, 0, 2, 2), &mut output)
        .unwrap();

    assert_eq!(source.width(), 3);
    assert_eq!(source.height(), 2);
    assert_eq!(output, vec![1, 2, 4, 5]);

    fs::remove_file(input_path).unwrap();
}

#[test]
fn streaming_shared_source_runs_through_pipeline_scheduler() {
    use crate::pipeline::ImagePipeline;
    use crate::scheduler::rayon_scheduler::RayonScheduler;

    let decoder = StreamingGridDecoder::new();
    let full_decodes = Arc::clone(&decoder.full_decodes);
    let encoded: Arc<[u8]> = Arc::from(&b"encoded-grid"[..]);
    let source =
        DecoderSource::<_, U8>::streaming_shared(decoder, encoded, LoadOptions::default()).unwrap();

    let pipeline = ImagePipeline::from_source(source)
        .invert()
        .unwrap()
        .build()
        .unwrap();
    let image = pipeline
        .run_to_image::<U8, _>(&RayonScheduler::new(1).unwrap())
        .unwrap();

    assert_eq!(image.width(), 8);
    assert_eq!(image.height(), 8);
    assert_eq!(image.bands(), 1);
    assert_eq!(image.pixels()[0], 255);
    assert_eq!(image.pixels()[1], 254);
    assert_eq!(image.pixels()[8], 245);
    assert_eq!(*full_decodes.lock().unwrap(), 0);
}

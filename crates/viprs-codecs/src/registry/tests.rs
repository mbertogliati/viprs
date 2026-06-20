use std::{
    any::Any,
    fs,
    path::{Path, PathBuf},
};

use crate::{ImageCodecExt, RawCodec};
use viprs_core::{
    codec_options::{LoadOptions, SaveOptions},
    error::ViprsError,
    format::{BandFormat, BandFormatId, F32, F64, I16, I32, U8, U16, U32},
    image::Image,
};
use viprs_ports::codec::{ImageCodec, ImageDecoder, ImageEncoder};

use super::*;

struct ExtensionFallbackCodec;
struct HeaderSniffCodec;
struct PathExtensionCodec;
struct DirectoryPathCodec;

impl ImageCodec for ExtensionFallbackCodec {
    fn format_name(&self) -> &'static str {
        "ext-fallback"
    }

    fn file_extensions(&self) -> &'static [&'static str] {
        &["fallback"]
    }

    fn can_encode(&self) -> bool {
        false
    }

    fn supports_extension_decode_fallback(&self) -> bool {
        true
    }

    fn sniff(&self, _header: &[u8]) -> bool {
        false
    }

    fn decode_boxed(
        &self,
        src: &[u8],
        band_format: BandFormatId,
        _opts: &LoadOptions,
    ) -> Result<Box<dyn Any + Send>, ViprsError> {
        if band_format != BandFormatId::U8 {
            return Err(ViprsError::Codec("ext-fallback: only U8 supported".into()));
        }
        let image = Image::<U8>::from_buffer(1, 1, 1, vec![src.first().copied().unwrap_or(0)])?;
        Ok(Box::new(image))
    }

    fn encode_boxed(
        &self,
        _image: &(dyn Any + Send + Sync),
        _band_format: BandFormatId,
        _opts: &SaveOptions,
    ) -> Result<Vec<u8>, ViprsError> {
        Err(ViprsError::Codec("ext-fallback: decode-only".into()))
    }
}

impl ImageCodec for DirectoryPathCodec {
    fn format_name(&self) -> &'static str {
        "directory-path"
    }

    fn file_extensions(&self) -> &'static [&'static str] {
        &["mrxs"]
    }

    fn can_encode(&self) -> bool {
        false
    }

    fn sniff(&self, _header: &[u8]) -> bool {
        false
    }

    fn can_decode_path(&self, path: &Path) -> bool {
        path.is_dir()
            && path
                .extension()
                .and_then(std::ffi::OsStr::to_str)
                .is_some_and(|extension| extension.eq_ignore_ascii_case("mrxs"))
    }

    fn decode_boxed(
        &self,
        _src: &[u8],
        _band_format: BandFormatId,
        _opts: &LoadOptions,
    ) -> Result<Box<dyn Any + Send>, ViprsError> {
        Err(ViprsError::Codec(
            "directory-path: byte decode should not be used".into(),
        ))
    }

    fn decode_boxed_path(
        &self,
        path: &Path,
        band_format: BandFormatId,
        _opts: &LoadOptions,
    ) -> Result<Box<dyn Any + Send>, ViprsError> {
        if band_format != BandFormatId::U8 {
            return Err(ViprsError::Codec(
                "directory-path: only U8 is supported".into(),
            ));
        }
        assert!(path.is_dir());
        Ok(Box::new(Image::<U8>::from_buffer(1, 1, 1, vec![9])?))
    }

    fn encode_boxed(
        &self,
        _image: &(dyn Any + Send + Sync),
        _band_format: BandFormatId,
        _opts: &SaveOptions,
    ) -> Result<Vec<u8>, ViprsError> {
        Err(ViprsError::Codec("directory-path: decode-only".into()))
    }
}

impl ImageCodec for HeaderSniffCodec {
    fn format_name(&self) -> &'static str {
        "header-sniff"
    }

    fn file_extensions(&self) -> &'static [&'static str] {
        &["jpg", "jpeg"]
    }

    fn can_encode(&self) -> bool {
        false
    }

    fn sniff(&self, header: &[u8]) -> bool {
        header.starts_with(&[0xFF, 0xD8, 0xFF])
    }

    fn decode_boxed(
        &self,
        _src: &[u8],
        band_format: BandFormatId,
        _opts: &LoadOptions,
    ) -> Result<Box<dyn Any + Send>, ViprsError> {
        if band_format != BandFormatId::U8 {
            return Err(ViprsError::Codec("header-sniff: only U8 supported".into()));
        }
        Ok(Box::new(Image::<U8>::from_buffer(1, 1, 1, vec![2])?))
    }

    fn encode_boxed(
        &self,
        _image: &(dyn Any + Send + Sync),
        _band_format: BandFormatId,
        _opts: &SaveOptions,
    ) -> Result<Vec<u8>, ViprsError> {
        Err(ViprsError::Codec("header-sniff: decode-only".into()))
    }
}

impl ImageCodec for PathExtensionCodec {
    fn format_name(&self) -> &'static str {
        "path-extension"
    }

    fn file_extensions(&self) -> &'static [&'static str] {
        &["png"]
    }

    fn can_encode(&self) -> bool {
        false
    }

    fn sniff(&self, _header: &[u8]) -> bool {
        false
    }

    fn can_decode_path(&self, path: &Path) -> bool {
        path.extension()
            .and_then(std::ffi::OsStr::to_str)
            .is_some_and(|extension| extension.eq_ignore_ascii_case("png"))
    }

    fn decode_boxed(
        &self,
        _src: &[u8],
        band_format: BandFormatId,
        _opts: &LoadOptions,
    ) -> Result<Box<dyn Any + Send>, ViprsError> {
        if band_format != BandFormatId::U8 {
            return Err(ViprsError::Codec(
                "path-extension: only U8 supported".into(),
            ));
        }
        Ok(Box::new(Image::<U8>::from_buffer(1, 1, 1, vec![1])?))
    }

    fn encode_boxed(
        &self,
        _image: &(dyn Any + Send + Sync),
        _band_format: BandFormatId,
        _opts: &SaveOptions,
    ) -> Result<Vec<u8>, ViprsError> {
        Err(ViprsError::Codec("path-extension: decode-only".into()))
    }
}

fn test_output_path(name: &str, extension: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("foreign-registry-unit-tests");
    fs::create_dir_all(&dir).unwrap();
    dir.join(format!("{name}-{}.{}", std::process::id(), extension))
}

fn test_directory_path(name: &str, extension: &str) -> PathBuf {
    let dir = test_output_path(name, extension);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn raw_load_options<F: BandFormat>() -> LoadOptions {
    LoadOptions::default()
        .with_raw_layout(1, 1, 1)
        .with_raw_format(F::ID)
}

fn sample_bytes<T: bytemuck::Pod>(sample: T) -> Vec<u8> {
    bytemuck::bytes_of(&sample).to_vec()
}

#[derive(Clone, Copy)]
struct ExtensionOnlyDecoder;

impl ImageDecoder for ExtensionOnlyDecoder {
    fn format_name(&self) -> &'static str {
        "extension-only"
    }

    fn sniff(&self, _header: &[u8]) -> bool
    where
        Self: Sized,
    {
        false
    }

    fn decode<F: BandFormat>(&self, src: &[u8]) -> Result<Image<F>, ViprsError> {
        self.decode_with_options(src, &LoadOptions::default())
    }

    fn decode_with_options<F: BandFormat>(
        &self,
        src: &[u8],
        _opts: &LoadOptions,
    ) -> Result<Image<F>, ViprsError>
    where
        Self: Sized,
    {
        let mut bytes = vec![0u8; std::mem::size_of::<F::Sample>()];
        let copied = src.len().min(bytes.len());
        bytes[..copied].copy_from_slice(&src[..copied]);
        let pixels = bytemuck::cast_slice::<u8, F::Sample>(&bytes).to_vec();
        Image::from_buffer(1, 1, 1, pixels)
    }

    fn probe(&self, _src: &[u8]) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized,
    {
        Ok((1, 1, 1))
    }
}

#[test]
fn pdf_header_sniff_handles_short_and_offset_headers() {
    assert!(!pdf_header_sniff(b"%PD"));
    assert!(pdf_header_sniff(b" \n\t%PDF-1.7\n"));
}

#[test]
fn detect_format_matches_extension_case_insensitively() {
    let mut registry = ForeignRegistry::new();
    registry.register(boxed_codec(RawCodec, &["raw"]));

    let codec = registry.detect_format(Path::new("image.RAW")).unwrap();
    assert_eq!(codec.format_name(), "raw");
}

#[test]
fn registry_loads_raw_with_explicit_options_and_saves_by_extension() {
    let input_path = test_output_path("input", "raw");
    let output_path = test_output_path("output", "raw");

    let mut registry = ForeignRegistry::new();
    registry.register(boxed_codec(RawCodec, &["raw"]));

    let bytes: Vec<u8> = vec![1, 2, 3, 4];
    fs::write(&input_path, &bytes).unwrap();

    let image = registry
        .load_with_options(
            &input_path,
            &LoadOptions::default().with_raw_layout(2, 2, 1),
        )
        .unwrap();
    assert_eq!(image.width(), 2);
    assert_eq!(image.height(), 2);
    assert_eq!(image.bands(), 1);
    assert_eq!(image.pixels(), &[1, 2, 3, 4]);

    registry.save(&image, &output_path).unwrap();
    assert_eq!(fs::read(&output_path).unwrap(), bytes);

    fs::remove_file(input_path).unwrap();
    fs::remove_file(output_path).unwrap();
}

#[test]
fn registry_loads_directory_backed_codec_via_path_api() {
    let input_path = test_directory_path("slide", "mrxs");
    let mut registry = ForeignRegistry::new();
    registry.register(Box::new(DirectoryPathCodec));

    let image = registry.load(&input_path).unwrap();

    assert_eq!(image.width(), 1);
    assert_eq!(image.height(), 1);
    assert_eq!(image.pixels(), &[9]);

    fs::remove_dir_all(input_path).unwrap();
}

#[test]
fn registry_uses_extension_decode_fallback_only_after_sniff_miss() {
    let path = test_output_path("ext-fallback", "fallback");
    fs::write(&path, b"\x2Abytes").unwrap();

    let mut registry = ForeignRegistry::new();
    registry.register(Box::new(ExtensionFallbackCodec));
    let image = registry.load(&path).unwrap();
    assert_eq!(image.pixels(), &[42]);

    fs::remove_file(path).unwrap();
}

#[test]
fn helper_codecs_expose_expected_capabilities_and_errors() {
    let ext = ExtensionFallbackCodec;
    assert_eq!(ext.format_name(), "ext-fallback");
    assert_eq!(ext.file_extensions(), &["fallback"]);
    assert!(!ext.can_encode());
    assert!(ext.supports_extension_decode_fallback());
    assert!(!ext.sniff(b"header"));
    let ext_image = ext
        .decode_boxed(b"*", BandFormatId::U8, &LoadOptions::default())
        .unwrap()
        .downcast::<Image<U8>>()
        .unwrap();
    assert_eq!(ext_image.pixels(), &[b'*']);
    let ext_decode_err = ext
        .decode_boxed(b"*", BandFormatId::U16, &LoadOptions::default())
        .unwrap_err();
    assert!(ext_decode_err.to_string().contains("only U8 supported"));
    let ext_encode_err = ext
        .encode_boxed(
            &Image::<U8>::from_buffer(1, 1, 1, vec![1]).unwrap(),
            BandFormatId::U8,
            &SaveOptions::default(),
        )
        .unwrap_err();
    assert!(ext_encode_err.to_string().contains("decode-only"));

    let directory = DirectoryPathCodec;
    let directory_path = test_directory_path("helper-directory", "mrxs");
    assert_eq!(directory.format_name(), "directory-path");
    assert_eq!(directory.file_extensions(), &["mrxs"]);
    assert!(!directory.can_encode());
    assert!(!directory.sniff(b"header"));
    assert!(directory.can_decode_path(&directory_path));
    let directory_image = directory
        .decode_boxed_path(&directory_path, BandFormatId::U8, &LoadOptions::default())
        .unwrap()
        .downcast::<Image<U8>>()
        .unwrap();
    assert_eq!(directory_image.pixels(), &[9]);
    let directory_decode_err = directory
        .decode_boxed(b"", BandFormatId::U8, &LoadOptions::default())
        .unwrap_err();
    assert!(
        directory_decode_err
            .to_string()
            .contains("byte decode should not be used")
    );
    let directory_path_err = directory
        .decode_boxed_path(&directory_path, BandFormatId::U16, &LoadOptions::default())
        .unwrap_err();
    assert!(
        directory_path_err
            .to_string()
            .contains("only U8 is supported")
    );
    let directory_encode_err = directory
        .encode_boxed(
            &Image::<U8>::from_buffer(1, 1, 1, vec![1]).unwrap(),
            BandFormatId::U8,
            &SaveOptions::default(),
        )
        .unwrap_err();
    assert!(directory_encode_err.to_string().contains("decode-only"));

    let sniff = HeaderSniffCodec;
    assert_eq!(sniff.format_name(), "header-sniff");
    assert_eq!(sniff.file_extensions(), &["jpg", "jpeg"]);
    assert!(!sniff.can_encode());
    assert!(sniff.sniff(&[0xFF, 0xD8, 0xFF, 0x00]));
    let sniff_image = sniff
        .decode_boxed(b"", BandFormatId::U8, &LoadOptions::default())
        .unwrap()
        .downcast::<Image<U8>>()
        .unwrap();
    assert_eq!(sniff_image.pixels(), &[2]);
    let sniff_err = sniff
        .decode_boxed(b"", BandFormatId::U16, &LoadOptions::default())
        .unwrap_err();
    assert!(sniff_err.to_string().contains("only U8 supported"));
    let sniff_encode_err = sniff
        .encode_boxed(
            &Image::<U8>::from_buffer(1, 1, 1, vec![1]).unwrap(),
            BandFormatId::U8,
            &SaveOptions::default(),
        )
        .unwrap_err();
    assert!(sniff_encode_err.to_string().contains("decode-only"));

    let path = PathExtensionCodec;
    assert_eq!(path.format_name(), "path-extension");
    assert_eq!(path.file_extensions(), &["png"]);
    assert!(!path.can_encode());
    assert!(!path.sniff(b"header"));
    assert!(path.can_decode_path(Path::new("direct.PNG")));
    let path_image = path
        .decode_boxed(b"", BandFormatId::U8, &LoadOptions::default())
        .unwrap()
        .downcast::<Image<U8>>()
        .unwrap();
    assert_eq!(path_image.pixels(), &[1]);
    let path_err = path
        .decode_boxed(b"", BandFormatId::U16, &LoadOptions::default())
        .unwrap_err();
    assert!(path_err.to_string().contains("only U8 supported"));
    let path_encode_err = path
        .encode_boxed(
            &Image::<U8>::from_buffer(1, 1, 1, vec![1]).unwrap(),
            BandFormatId::U8,
            &SaveOptions::default(),
        )
        .unwrap_err();
    assert!(path_encode_err.to_string().contains("decode-only"));

    fs::remove_dir_all(directory_path).unwrap();
}

#[test]
fn codec_bridge_dispatches_all_band_formats_and_type_mismatches() {
    let codec = boxed_codec(RawCodec, &["raw"]);
    assert_eq!(codec.format_name(), "raw");
    assert_eq!(codec.file_extensions(), &["raw"]);
    assert!(!codec.sniff(b"header"));
    assert!(codec.can_decode_path(Path::new("pixels.raw")));

    macro_rules! exercise_codec_bridge {
        ($format:ty, $band:expr, $sample:expr, $name:literal) => {{
            let bytes = sample_bytes($sample);
            let opts = raw_load_options::<$format>();
            let decoded = codec
                .decode_boxed(&bytes, $band, &opts)
                .unwrap()
                .downcast::<Image<$format>>()
                .unwrap();
            assert_eq!(decoded.pixels(), &[$sample]);

            let path = test_output_path($name, "raw");
            fs::write(&path, &bytes).unwrap();
            let decoded_path = codec
                .decode_boxed_path(&path, $band, &opts)
                .unwrap()
                .downcast::<Image<$format>>()
                .unwrap();
            assert_eq!(decoded_path.pixels(), &[$sample]);

            let image = Image::<$format>::from_buffer(1, 1, 1, vec![$sample]).unwrap();
            let encoded = codec
                .encode_boxed(&image, $band, &SaveOptions::default())
                .unwrap();
            assert_eq!(encoded, bytes);
            fs::remove_file(path).unwrap();
        }};
    }

    exercise_codec_bridge!(U8, BandFormatId::U8, 7u8, "codec-bridge-u8");
    exercise_codec_bridge!(U16, BandFormatId::U16, 257u16, "codec-bridge-u16");
    exercise_codec_bridge!(I16, BandFormatId::I16, -7i16, "codec-bridge-i16");
    exercise_codec_bridge!(U32, BandFormatId::U32, 65_793u32, "codec-bridge-u32");
    exercise_codec_bridge!(I32, BandFormatId::I32, -65_793i32, "codec-bridge-i32");
    exercise_codec_bridge!(F32, BandFormatId::F32, 3.5f32, "codec-bridge-f32");
    exercise_codec_bridge!(F64, BandFormatId::F64, 7.25f64, "codec-bridge-f64");

    let mismatch = codec
        .encode_boxed(
            &Image::<U8>::from_buffer(1, 1, 1, vec![1]).unwrap(),
            BandFormatId::F32,
            &SaveOptions::default(),
        )
        .unwrap_err();
    assert!(mismatch.to_string().contains("mismatched image type"));
}

#[test]
fn decoder_bridges_cover_decode_only_and_extension_fallback_paths() {
    let decoder = boxed_decoder(RawCodec, &["raw"]);
    assert_eq!(decoder.format_name(), "raw");
    assert_eq!(decoder.file_extensions(), &["raw"]);
    assert!(!decoder.supports_extension_decode_fallback());
    assert!(!decoder.can_encode());
    assert!(!decoder.sniff(b"header"));
    assert!(decoder.can_decode_path(Path::new("pixels.raw")));

    macro_rules! exercise_decoder_bridge {
        ($format:ty, $band:expr, $sample:expr, $name:literal) => {{
            let bytes = sample_bytes($sample);
            let opts = raw_load_options::<$format>();
            let decoded = decoder
                .decode_boxed(&bytes, $band, &opts)
                .unwrap()
                .downcast::<Image<$format>>()
                .unwrap();
            assert_eq!(decoded.pixels(), &[$sample]);

            let path = test_output_path($name, "raw");
            fs::write(&path, &bytes).unwrap();
            let decoded_path = decoder
                .decode_boxed_path(&path, $band, &opts)
                .unwrap()
                .downcast::<Image<$format>>()
                .unwrap();
            assert_eq!(decoded_path.pixels(), &[$sample]);
            fs::remove_file(path).unwrap();
        }};
    }

    exercise_decoder_bridge!(U8, BandFormatId::U8, 3u8, "decoder-bridge-u8");
    exercise_decoder_bridge!(U16, BandFormatId::U16, 513u16, "decoder-bridge-u16");
    exercise_decoder_bridge!(I16, BandFormatId::I16, -3i16, "decoder-bridge-i16");
    exercise_decoder_bridge!(U32, BandFormatId::U32, 131_329u32, "decoder-bridge-u32");
    exercise_decoder_bridge!(I32, BandFormatId::I32, -131_329i32, "decoder-bridge-i32");
    exercise_decoder_bridge!(F32, BandFormatId::F32, 1.5f32, "decoder-bridge-f32");
    exercise_decoder_bridge!(F64, BandFormatId::F64, 2.5f64, "decoder-bridge-f64");

    let decode_only_err = decoder
        .encode_boxed(
            &Image::<U8>::from_buffer(1, 1, 1, vec![1]).unwrap(),
            BandFormatId::U8,
            &SaveOptions::default(),
        )
        .unwrap_err();
    assert!(decode_only_err.to_string().contains("decode-only"));

    let extension_decoder = boxed_extension_decoder(ExtensionOnlyDecoder, &["ext"]);
    assert_eq!(extension_decoder.format_name(), "extension-only");
    assert_eq!(extension_decoder.file_extensions(), &["ext"]);
    assert!(extension_decoder.supports_extension_decode_fallback());
    assert!(!extension_decoder.can_encode());
    assert!(!extension_decoder.sniff(b"header"));
    assert!(extension_decoder.can_decode_path(Path::new("fallback.ext")));
    assert!(!extension_decoder.can_decode_path(Path::new("fallback.bin")));

    let path = test_output_path("extension-only", "ext");
    fs::write(&path, [11]).unwrap();
    let decoded = extension_decoder
        .decode_boxed_path(&path, BandFormatId::U8, &LoadOptions::default())
        .unwrap()
        .downcast::<Image<U8>>()
        .unwrap();
    assert_eq!(decoded.pixels(), &[11]);
    let decode_only_err = extension_decoder
        .encode_boxed(
            &Image::<U8>::from_buffer(1, 1, 1, vec![1]).unwrap(),
            BandFormatId::U8,
            &SaveOptions::default(),
        )
        .unwrap_err();
    assert!(decode_only_err.to_string().contains("decode-only"));
    fs::remove_file(path).unwrap();
}

#[test]
fn registry_and_image_convenience_apis_cover_path_and_error_edges() {
    let registry = ForeignRegistry::shared();

    let u8_input = test_output_path("shared-u8-input", "raw");
    let u8_output = test_output_path("shared-u8-output", "raw");
    fs::write(&u8_input, [5]).unwrap();
    let u8_image = Image::<U8>::load_with_options(&u8_input, &raw_load_options::<U8>()).unwrap();
    assert_eq!(u8_image.pixels(), &[5]);
    registry
        .save_with_options(&u8_image, &u8_output, &SaveOptions::default())
        .unwrap();
    assert_eq!(fs::read(&u8_output).unwrap(), [5]);
    u8_image.save(&u8_output).unwrap();
    u8_image
        .save_with_options(&u8_output, &SaveOptions::default())
        .unwrap();

    let u16_input = test_output_path("shared-u16-input", "raw");
    let u16_output = test_output_path("shared-u16-output", "raw");
    fs::write(&u16_input, sample_bytes(1025u16)).unwrap();
    let opts_u16 = raw_load_options::<U16>();
    let u16_image = registry.load_as::<U16>(&u16_input).unwrap_err();
    assert!(u16_image.to_string().contains("required"));
    let u16_image = Image::<U16>::load_with_options(&u16_input, &opts_u16).unwrap();
    assert_eq!(u16_image.pixels(), &[1025]);
    registry.save_as(&u16_image, &u16_output).unwrap();
    assert_eq!(fs::read(&u16_output).unwrap(), sample_bytes(1025u16));
    u16_image
        .save_with_options(&u16_output, &SaveOptions::default())
        .unwrap();

    let no_decoder_dir = test_directory_path("no-decoder", "unknown");
    let err = registry.load(&no_decoder_dir).unwrap_err();
    assert!(err.to_string().contains("no decoder matched"));
    assert!(registry.detect_format(Path::new("no-extension")).is_none());
    assert!(
        registry
            .find_decoder_by_extension(Path::new("no-extension"))
            .is_none()
    );
    assert_eq!(read_header(&u8_input).unwrap(), vec![5]);

    fs::remove_file(u8_input).unwrap();
    fs::remove_file(u8_output).unwrap();
    fs::remove_file(u16_input).unwrap();
    fs::remove_file(u16_output).unwrap();
    fs::remove_dir_all(no_decoder_dir).unwrap();
}

#[test]
fn registry_prefers_sniffed_decoder_before_path_extension_match() {
    let path = test_output_path("sniff-before-path", "png");
    fs::write(&path, [0xFF, 0xD8, 0xFF, 0x00]).unwrap();

    let mut registry = ForeignRegistry::new();
    registry.register(Box::new(PathExtensionCodec));
    registry.register(Box::new(HeaderSniffCodec));

    let image = registry.load(&path).unwrap();
    assert_eq!(image.pixels(), &[2]);

    fs::remove_file(path).unwrap();
}

#[test]
fn registry_errors_when_no_decoder_matches() {
    let path = test_output_path("unknown", "raw");
    fs::write(&path, b"not-an-image").unwrap();

    let registry = ForeignRegistry::new();
    let err = registry.load(&path).unwrap_err();

    assert!(
        err.to_string().contains("no decoder matched"),
        "unexpected error: {err}"
    );

    fs::remove_file(path).unwrap();
}

#[test]
fn registry_errors_when_extension_is_missing() {
    let registry = ForeignRegistry::new();
    let image = Image::<U8>::from_buffer(1, 1, 1, vec![7]).unwrap();
    let err = registry
        .save(
            &image,
            Path::new("target/foreign-registry-unit-tests/no-extension"),
        )
        .unwrap_err();

    assert!(
        err.to_string().contains("no encoder registered"),
        "unexpected error: {err}"
    );
}

#[test]
#[cfg(not(feature = "pdf-poppler"))]
fn registry_returns_typed_deferred_decode_error() {
    let path = test_output_path("deferred-decode", "pdf");
    fs::write(&path, b"%PDF-1.7\n").unwrap();

    let registry = ForeignRegistry::new();
    let err = registry.load(&path).unwrap_err();
    assert!(
        matches!(err, ViprsError::Unimplemented { .. }),
        "expected typed unimplemented error, got {err}"
    );
    assert!(
        err.to_string().contains("pdf-poppler"),
        "unexpected error message: {err}"
    );

    fs::remove_file(path).unwrap();
}

#[test]
#[cfg(not(feature = "pdf-poppler"))]
fn registry_returns_deferred_pdf_error_by_header_even_without_extension() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("foreign-registry-unit-tests");
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("deferred-pdf-header-{}", std::process::id()));
    fs::write(&path, b" \n\t%PDF-1.5\n").unwrap();

    let err = ForeignRegistry::new().load(&path).unwrap_err();
    assert!(matches!(err, ViprsError::Unimplemented { .. }));
    assert!(
        err.to_string().contains("pdf-poppler"),
        "unexpected error: {err}"
    );

    fs::remove_file(path).unwrap();
}

#[test]
#[cfg(feature = "pdf-poppler")]
fn registry_loads_pdf_when_poppler_backend_enabled() {
    use std::process::{Command, Stdio};

    fn poppler_available() -> bool {
        Command::new("pdfinfo")
            .arg("-v")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
            && Command::new("pdftoppm")
                .arg("-v")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .is_ok_and(|status| status.success())
    }

    if !poppler_available() {
        eprintln!("skipping registry pdf test: poppler tools not available");
        return;
    }

    let fixture_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("images")
        .join("pdf-two-pages-72pt-144pt.pdf");
    let image = ForeignRegistry::default()
        .load_with_options(
            &fixture_path,
            &LoadOptions::default().with_page(0).with_n(2),
        )
        .unwrap();

    assert_eq!(image.width(), 144);
    assert_eq!(image.height(), 144);
    assert_eq!(image.metadata().n_pages, Some(2));
}

#[test]
fn registry_returns_typed_deferred_encode_error() {
    let registry = ForeignRegistry::new();
    let image = Image::<U8>::from_buffer(1, 1, 1, vec![7]).unwrap();
    let err = registry
        .save(
            &image,
            Path::new("target/foreign-registry-unit-tests/deferred-save.pdf"),
        )
        .unwrap_err();

    assert!(
        matches!(err, ViprsError::Unimplemented { .. }),
        "expected typed unimplemented error, got {err}"
    );
    assert!(
        err.to_string()
            .contains("PDF encode is not yet implemented"),
        "unexpected error message: {err}"
    );
}

#[test]
fn registry_returns_family_specific_deferred_errors_for_document_slide_and_fallbacks() {
    let registry = ForeignRegistry::new();
    let image = Image::<U8>::from_buffer(1, 1, 1, vec![3]).unwrap();

    #[cfg(all(not(feature = "magick"), not(feature = "dcraw")))]
    let decode_cases = [
        (
            "openslide",
            "svs",
            b"slide-bytes".as_slice(),
            "feature `openslide`",
        ),
        (
            "dcraw",
            "nef",
            b"raw-bytes".as_slice(),
            "Camera RAW support is not yet implemented",
        ),
        (
            "magick",
            "psd",
            b"8BPS\0\x01\0\0".as_slice(),
            "ImageMagick fallback support is not yet implemented",
        ),
    ];
    #[cfg(all(feature = "magick", not(feature = "dcraw")))]
    let decode_cases = [
        (
            "openslide",
            "svs",
            b"slide-bytes".as_slice(),
            "feature `openslide`",
        ),
        (
            "dcraw",
            "nef",
            b"raw-bytes".as_slice(),
            "Camera RAW support is not yet implemented",
        ),
    ];
    #[cfg(all(not(feature = "magick"), feature = "dcraw"))]
    let decode_cases = [
        (
            "openslide",
            "svs",
            b"slide-bytes".as_slice(),
            "feature `openslide`",
        ),
        (
            "magick",
            "psd",
            b"8BPS\0\x01\0\0".as_slice(),
            "ImageMagick fallback support is not yet implemented",
        ),
    ];
    #[cfg(all(feature = "magick", feature = "dcraw"))]
    let decode_cases = [(
        "openslide",
        "svs",
        b"slide-bytes".as_slice(),
        "feature `openslide`",
    )];
    for (name, extension, bytes, expected_task) in decode_cases {
        let path = test_output_path(name, extension);
        fs::write(&path, bytes).unwrap();
        let err = registry.load(&path).unwrap_err();
        #[cfg(feature = "openslide")]
        if name == "openslide" {
            assert!(
                matches!(err, ViprsError::Codec(_)),
                "expected codec error for {name}, got {err}"
            );
            assert!(
                err.to_string().contains("openslide: open"),
                "unexpected {name} codec error: {err}"
            );
            fs::remove_file(path).unwrap();
            continue;
        }
        assert!(
            matches!(err, ViprsError::Unimplemented { .. }),
            "expected deferred error for {name}, got {err}"
        );
        assert!(
            err.to_string().contains(expected_task),
            "unexpected {name} deferred message: {err}"
        );
        fs::remove_file(path).unwrap();
    }

    #[cfg(all(not(feature = "deepzoom"), not(feature = "magick")))]
    let encode_cases = [
        ("deepzoom", "dzi", "DeepZoom export is not yet implemented"),
        (
            "magick-save",
            "ico",
            "ImageMagick fallback support is not yet implemented",
        ),
    ];
    #[cfg(all(feature = "deepzoom", not(feature = "magick")))]
    let encode_cases = [(
        "magick-save",
        "ico",
        "ImageMagick fallback support is not yet implemented",
    )];
    #[cfg(all(not(feature = "deepzoom"), feature = "magick"))]
    let encode_cases = [("deepzoom", "dzi", "DeepZoom export is not yet implemented")];
    #[cfg(all(feature = "deepzoom", feature = "magick"))]
    let encode_cases: [(&str, &str, &str); 0] = [];
    for (name, extension, expected_task) in encode_cases {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("foreign-registry-unit-tests")
            .join(format!("{name}-{}.{}", std::process::id(), extension));
        let err = registry.save(&image, &path).unwrap_err();
        assert!(
            matches!(err, ViprsError::Unimplemented { .. }),
            "expected deferred encode error for {name}, got {err}"
        );
        assert!(
            err.to_string().contains(expected_task),
            "unexpected {name} deferred message: {err}"
        );
    }
}

#[cfg(feature = "deepzoom")]
#[test]
fn registry_saves_deepzoom_layout_for_dzi_path() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("foreign-registry-unit-tests");
    fs::create_dir_all(&root).unwrap();
    let descriptor = root.join(format!("deepzoom-export-{}.dzi", std::process::id()));
    let tile_root = root.join(format!("deepzoom-export-{}_files", std::process::id()));
    let image = Image::<U8>::from_buffer(4, 4, 3, (0u8..48).collect()).unwrap();

    ForeignRegistry::default()
        .save_as(&image, &descriptor)
        .unwrap();

    let descriptor_xml = fs::read_to_string(&descriptor).unwrap();
    assert!(descriptor_xml.contains("TileSize=\"254\""));
    assert!(descriptor_xml.contains("Width=\"4\""));
    assert!(descriptor_xml.contains("Height=\"4\""));

    let top_tile = tile_root.join("2/0_0.ppm");
    let tile_bytes = fs::read(top_tile).unwrap();
    assert!(tile_bytes.starts_with(b"P6\n"));

    let _ = fs::remove_file(descriptor);
    let _ = fs::remove_dir_all(tile_root);
}

#[cfg(feature = "deepzoom")]
#[test]
fn registry_quantizes_u16_deepzoom_tiles_to_u8() {
    let descriptor = test_output_path("deepzoom-u16-export", "dzi");
    let tile_root = descriptor.with_extension("").with_file_name(format!(
        "{}_files",
        descriptor
            .file_stem()
            .and_then(std::ffi::OsStr::to_str)
            .unwrap()
    ));
    let image = Image::<U16>::from_buffer(2, 1, 1, vec![0, 257]).unwrap();

    ForeignRegistry::default()
        .save_as(&image, &descriptor)
        .unwrap();

    let tile_bytes = fs::read(tile_root.join("1/0_0.ppm")).unwrap();
    assert_eq!(tile_bytes, b"P6\n2 1\n255\n\0\0\0\x01\x01\x01");

    let _ = fs::remove_file(descriptor);
    let _ = fs::remove_dir_all(tile_root);
}

#[cfg(feature = "deepzoom")]
#[test]
fn to_u8_image_scales_unit_range_f32_pixels() {
    let image = Image::<F32>::from_buffer(3, 1, 1, vec![0.0, 0.5, 1.0]).unwrap();

    let quantized = to_u8_image(&image).unwrap();

    assert_eq!(quantized.pixels(), &[0, 127, 255]);
}

#[cfg(feature = "deepzoom")]
#[test]
fn to_u8_image_clamps_byte_range_f32_pixels() {
    let image = Image::<F32>::from_buffer(3, 1, 1, vec![-5.0, 12.5, 300.0]).unwrap();

    let quantized = to_u8_image(&image).unwrap();

    assert_eq!(quantized.pixels(), &[0, 12, 255]);
}

#[test]
fn registry_registers_mat_decoder_for_matlab_signature() {
    let path = test_output_path("mat-signature", "mat");
    let mut header = vec![0u8; 128];
    let magic = b"MATLAB 5.0 MAT-file";
    header[..magic.len()].copy_from_slice(magic);
    header[126] = b'M';
    header[127] = b'I';
    fs::write(&path, &header).unwrap();

    let err = ForeignRegistry::default().load(&path).unwrap_err();
    assert!(
        !err.to_string().contains("no decoder matched"),
        "expected MAT decoder path, got {err}"
    );

    fs::remove_file(path).unwrap();
}

#[cfg(feature = "jpeg")]
#[test]
fn registry_loads_jpeg_by_path() {
    use crate::JpegCodec;

    let path = test_output_path("sample-jpeg", "jpg");
    let codec = JpegCodec;
    let image = Image::<U8>::from_buffer(5, 3, 3, vec![120; 5 * 3 * 3]).unwrap();
    fs::write(&path, codec.encode(&image).unwrap()).unwrap();

    let decoded = ForeignRegistry::default().load(&path).unwrap();
    assert_eq!(decoded.width(), 5);
    assert_eq!(decoded.height(), 3);
    assert_eq!(decoded.bands(), 3);

    fs::remove_file(path).unwrap();
}

#[cfg(all(feature = "jpeg", feature = "png"))]
#[test]
fn registry_loads_jpeg_and_png_when_extensions_are_swapped() {
    use crate::{JpegCodec, PngCodec};

    let jpeg_path = test_output_path("swapped-jpeg-correct", "jpg");
    let jpeg_renamed_path = test_output_path("swapped-jpeg-wrong", "png");
    let png_path = test_output_path("swapped-png-correct", "png");
    let png_renamed_path = test_output_path("swapped-png-wrong", "jpg");

    let jpeg_image = Image::<U8>::from_buffer(5, 3, 3, (0u8..45).collect()).unwrap();
    let png_image = Image::<U8>::from_buffer(4, 4, 3, (0u8..48).collect()).unwrap();
    let jpeg_bytes = JpegCodec.encode(&jpeg_image).unwrap();
    let png_bytes = PngCodec::default().encode(&png_image).unwrap();

    fs::write(&jpeg_path, &jpeg_bytes).unwrap();
    fs::write(&jpeg_renamed_path, &jpeg_bytes).unwrap();
    fs::write(&png_path, &png_bytes).unwrap();
    fs::write(&png_renamed_path, &png_bytes).unwrap();

    let registry = ForeignRegistry::default();
    let jpeg_expected = registry.load(&jpeg_path).unwrap();
    let jpeg_renamed = registry.load(&jpeg_renamed_path).unwrap();
    let png_expected = registry.load(&png_path).unwrap();
    let png_renamed = registry.load(&png_renamed_path).unwrap();

    assert_eq!(jpeg_renamed.width(), jpeg_expected.width());
    assert_eq!(jpeg_renamed.height(), jpeg_expected.height());
    assert_eq!(jpeg_renamed.bands(), jpeg_expected.bands());
    assert_eq!(jpeg_renamed.pixels(), jpeg_expected.pixels());

    assert_eq!(png_renamed.width(), png_expected.width());
    assert_eq!(png_renamed.height(), png_expected.height());
    assert_eq!(png_renamed.bands(), png_expected.bands());
    assert_eq!(png_renamed.pixels(), png_expected.pixels());

    fs::remove_file(jpeg_path).unwrap();
    fs::remove_file(jpeg_renamed_path).unwrap();
    fs::remove_file(png_path).unwrap();
    fs::remove_file(png_renamed_path).unwrap();
}

#[cfg(feature = "png")]
#[test]
fn registry_loads_png_and_saves_png_round_trip() {
    use crate::PngCodec;

    let input_path = test_output_path("sample-png-input", "png");
    let output_path = test_output_path("sample-png-output", "png");
    let codec = PngCodec::default();
    let image = Image::<U8>::from_buffer(4, 4, 3, (0u8..48).collect()).unwrap();
    fs::write(&input_path, codec.encode(&image).unwrap()).unwrap();

    let registry = ForeignRegistry::default();
    let decoded = registry.load(&input_path).unwrap();
    registry.save(&decoded, &output_path).unwrap();

    let round_trip = codec
        .decode::<U8>(&fs::read(&output_path).unwrap())
        .unwrap();
    assert_eq!(round_trip.width(), 4);
    assert_eq!(round_trip.height(), 4);
    assert_eq!(round_trip.bands(), 3);
    assert_eq!(round_trip.pixels(), image.pixels());

    fs::remove_file(input_path).unwrap();
    fs::remove_file(output_path).unwrap();
}

#[cfg(feature = "tiff")]
#[test]
fn registry_loads_tiff_and_saves_tiff_round_trip() {
    use crate::TiffCodec;

    let input_path = test_output_path("sample-tiff-input", "tiff");
    let output_path = test_output_path("sample-tiff-output", "tif");
    let codec = TiffCodec::default();
    let image = Image::<U8>::from_buffer(4, 3, 3, (0u8..36).collect()).unwrap();
    fs::write(&input_path, codec.encode(&image).unwrap()).unwrap();

    let registry = ForeignRegistry::default();
    let decoded = registry.load(&input_path).unwrap();
    registry.save(&decoded, &output_path).unwrap();

    let round_trip = codec
        .decode::<U8>(&fs::read(&output_path).unwrap())
        .unwrap();
    assert_eq!(round_trip.width(), 4);
    assert_eq!(round_trip.height(), 3);
    assert_eq!(round_trip.bands(), 3);
    assert_eq!(round_trip.pixels(), image.pixels());

    fs::remove_file(input_path).unwrap();
    fs::remove_file(output_path).unwrap();
}

#[cfg(feature = "exr")]
#[test]
fn registry_loads_exr_and_saves_exr_round_trip() {
    use crate::ExrCodec;

    let input_path = test_output_path("sample-exr-input", "exr");
    let output_path = test_output_path("sample-exr-output", "exr");
    let codec = ExrCodec;
    let image = Image::<F32>::from_buffer(
        2,
        2,
        4,
        vec![
            0.0, 0.5, 1.0, 1.0, //
            1.5, 2.0, 2.5, 0.75, //
            3.0, 3.5, 4.0, 0.5, //
            4.5, 5.0, 5.5, 0.25,
        ],
    )
    .unwrap();
    fs::write(&input_path, codec.encode(&image).unwrap()).unwrap();

    let registry = ForeignRegistry::default();
    let decoded = registry.load_as::<F32>(&input_path).unwrap();
    registry.save_as(&decoded, &output_path).unwrap();

    let round_trip = codec
        .decode::<F32>(&fs::read(&output_path).unwrap())
        .unwrap();
    assert_eq!(round_trip.width(), 2);
    assert_eq!(round_trip.height(), 2);
    assert_eq!(round_trip.bands(), 4);
    for (actual, expected) in round_trip.pixels().iter().zip(image.pixels().iter()) {
        assert!((actual - expected).abs() <= f32::EPSILON);
    }

    fs::remove_file(input_path).unwrap();
    fs::remove_file(output_path).unwrap();
}

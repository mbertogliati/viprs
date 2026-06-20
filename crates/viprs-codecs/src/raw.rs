//! RAW codec — headerless flat interleaved pixel buffers.
//!
//! Parity target: libvips `rawload` / `rawsave` in
//! `libvips/foreign/rawload.c` and `libvips/foreign/rawsave.c`.
#![allow(clippy::unnecessary_wraps)]
// REASON: raw codec entry points keep the same `Result`-based surface as fallible codecs.
//!
//! RAW carries no header, so callers must supply width, height, band count,
//! and sample format explicitly. In the typed Rust API the sample format comes
//! from `F`; callers can also declare it in [`RawLoadOptions`] or
//! [`LoadOptions::with_raw_format`] for extra validation.

use std::{borrow::Cow, fs, path::Path};

use viprs_core::{
    codec_options::{LoadOptions, RawEndianness, SaveOptions},
    error::ViprsError,
    format::{BandFormat, BandFormatId},
    image::{Image, ImageMetadata, Interpretation},
};
use viprs_ports::codec::{ImageDecoder, ImageEncoder};

/// Explicit parameters for RAW decoding.
///
/// RAW payloads carry no header, so callers must provide the image layout that a
/// structured codec would normally probe from the container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawLoadOptions {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Number of interleaved bands per pixel.
    pub bands: u32,
    /// Number of bytes to skip before the first sample.
    pub offset: u64,
    /// Expected sample format for validation against the requested decode type.
    pub format: Option<BandFormatId>,
    /// Byte order used by multi-byte samples in the stream.
    pub endianness: RawEndianness,
    /// Optional interpretation metadata to attach to the decoded image.
    pub interpretation: Option<Interpretation>,
}

impl RawLoadOptions {
    /// Creates the minimum RAW layout required to decode a headerless buffer.
    #[must_use]
    pub const fn new(width: u32, height: u32, bands: u32) -> Self {
        Self {
            width,
            height,
            bands,
            offset: 0,
            format: None,
            endianness: RawEndianness::native(),
            interpretation: None,
        }
    }

    /// Sets the byte offset of the first sample within the encoded payload.
    #[must_use]
    pub const fn with_offset(mut self, offset: u64) -> Self {
        self.offset = offset;
        self
    }

    /// Declares the on-disk sample format for an extra runtime compatibility check.
    #[must_use]
    pub const fn with_format(mut self, format: BandFormatId) -> Self {
        self.format = Some(format);
        self
    }

    /// Overrides the byte order used to read multi-byte samples.
    #[must_use]
    pub const fn with_endianness(mut self, endianness: RawEndianness) -> Self {
        self.endianness = endianness;
        self
    }

    /// Marks the payload as little-endian.
    #[must_use]
    pub const fn lsb_first(mut self) -> Self {
        self.endianness = RawEndianness::Little;
        self
    }

    /// Marks the payload as big-endian.
    #[must_use]
    pub const fn msb_first(mut self) -> Self {
        self.endianness = RawEndianness::Big;
        self
    }

    /// Attaches the interpretation metadata that should be exposed on decode.
    #[must_use]
    pub const fn with_interpretation(mut self, interpretation: Interpretation) -> Self {
        self.interpretation = Some(interpretation);
        self
    }
}

impl From<RawLoadOptions> for LoadOptions {
    fn from(value: RawLoadOptions) -> Self {
        let opts = Self::default()
            .with_raw_layout(value.width, value.height, value.bands)
            .with_raw_offset(value.offset)
            .with_raw_endianness(value.endianness);
        let opts = if let Some(format) = value.format {
            opts.with_raw_format(format)
        } else {
            opts
        };
        if let Some(interpretation) = value.interpretation {
            opts.with_raw_interpretation(interpretation)
        } else {
            opts
        }
    }
}

/// Explicit parameters for RAW encoding.
///
/// RAW save behavior is intentionally narrow: only byte order needs to be chosen
/// because the layout is implied by the typed [`Image`] being written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawSaveOptions {
    /// Byte order to use when serializing multi-byte samples.
    pub endianness: RawEndianness,
}

impl RawSaveOptions {
    /// Creates RAW save options using the platform-native endianness.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            endianness: RawEndianness::native(),
        }
    }

    /// Overrides the byte order used when writing multi-byte samples.
    #[must_use]
    pub const fn with_endianness(mut self, endianness: RawEndianness) -> Self {
        self.endianness = endianness;
        self
    }
}

impl Default for RawSaveOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl From<RawSaveOptions> for SaveOptions {
    fn from(value: RawSaveOptions) -> Self {
        Self::default().with_raw_endianness(value.endianness)
    }
}

/// RAW codec implementing both `rawload` and `rawsave` style behavior.
///
/// This adapter reads and writes headerless interleaved buffers, mirroring the
/// libvips RAW loaders that rely entirely on caller-supplied layout metadata.
#[derive(Debug, Clone, Copy, Default)]
pub struct RawCodec;

/// Convenience alias for using [`RawCodec`] in decode-only call sites.
pub type RawDecoder = RawCodec;
/// Convenience alias for using [`RawCodec`] in encode-only call sites.
pub type RawEncoder = RawCodec;

impl RawCodec {
    /// `decode_raw` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs_codecs::raw::decode_raw;
    /// ```
    pub fn decode_raw<F: BandFormat>(
        &self,
        src: &[u8],
        opts: &RawLoadOptions,
    ) -> Result<Image<F>, ViprsError> {
        validate_raw_layout(opts.width, opts.height, opts.bands)?;
        validate_raw_format::<F>(opts.format)?;

        let offset = usize::try_from(opts.offset)
            .map_err(|_| ViprsError::Codec("raw: offset overflows usize".into()))?;
        if offset > src.len() {
            return Err(ViprsError::Codec(format!(
                "raw: offset {} exceeds buffer length {}",
                offset,
                src.len()
            )));
        }

        let sample_size = std::mem::size_of::<F::Sample>();
        let expected_bytes = expected_byte_count(opts.width, opts.height, opts.bands, sample_size)?;
        let available = &src[offset..];
        if available.len() < expected_bytes {
            return Err(ViprsError::Codec(format!(
                "raw: buffer has {} bytes after offset; expected {}",
                available.len(),
                expected_bytes
            )));
        }

        let bytes = &available[..expected_bytes];
        let samples = decode_samples::<F>(bytes, opts.endianness)?;
        let metadata = ImageMetadata {
            interpretation: opts.interpretation,
            ..ImageMetadata::default()
        };

        Image::from_buffer(opts.width, opts.height, opts.bands, samples)
            .map(|image| image.with_metadata(metadata))
    }

    /// `encode_raw` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs_codecs::raw::encode_raw;
    /// ```
    pub fn encode_raw<F: BandFormat>(
        &self,
        image: &Image<F>,
        opts: &RawSaveOptions,
    ) -> Result<Vec<u8>, ViprsError> {
        encode_bytes(image.pixels(), opts.endianness)
    }
}

impl ImageDecoder for RawCodec {
    fn format_name(&self) -> &'static str {
        "raw"
    }

    fn can_decode_path(&self, path: &Path) -> bool {
        path.extension()
            .and_then(std::ffi::OsStr::to_str)
            .is_some_and(|ext| ext.eq_ignore_ascii_case("raw"))
    }

    fn sniff(&self, _header: &[u8]) -> bool
    where
        Self: Sized,
    {
        false
    }

    fn decode<F: BandFormat>(&self, _src: &[u8]) -> Result<Image<F>, ViprsError> {
        Err(ViprsError::Codec(
            "raw: width/height/bands are required; use decode_with_options() or decode_raw()"
                .into(),
        ))
    }

    fn decode_with_options<F: BandFormat>(
        &self,
        src: &[u8],
        opts: &LoadOptions,
    ) -> Result<Image<F>, ViprsError>
    where
        Self: Sized,
    {
        let raw_opts = RawLoadOptions::try_from_load_options::<F>(opts)?;
        self.decode_raw(src, &raw_opts)
    }

    fn decode_path_with_options<F: BandFormat>(
        &self,
        path: &Path,
        opts: &LoadOptions,
    ) -> Result<Image<F>, ViprsError>
    where
        Self: Sized,
    {
        let src = fs::read(path)?;
        self.decode_with_options(&src, opts)
    }

    fn probe(&self, _src: &[u8]) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized,
    {
        Err(ViprsError::Codec(
            "raw: cannot probe dimensions from a headerless buffer".into(),
        ))
    }
}

impl ImageEncoder for RawCodec {
    fn format_name(&self) -> &'static str {
        "raw"
    }

    fn encode<F: BandFormat>(&self, image: &Image<F>) -> Result<Vec<u8>, ViprsError> {
        self.encode_raw(image, &RawSaveOptions::default())
    }

    fn encode_with_options<F: BandFormat>(
        &self,
        image: &Image<F>,
        opts: &SaveOptions,
    ) -> Result<Vec<u8>, ViprsError>
    where
        Self: Sized,
    {
        let raw_opts = RawSaveOptions {
            endianness: opts.raw_endianness.unwrap_or(RawEndianness::native()),
        };
        self.encode_raw(image, &raw_opts)
    }
}

impl RawLoadOptions {
    fn try_from_load_options<F: BandFormat>(opts: &LoadOptions) -> Result<Self, ViprsError> {
        let width = opts
            .raw_width
            .ok_or_else(|| ViprsError::Codec("raw: width is required in LoadOptions".into()))?;
        let height = opts
            .raw_height
            .ok_or_else(|| ViprsError::Codec("raw: height is required in LoadOptions".into()))?;
        let bands = opts
            .raw_bands
            .ok_or_else(|| ViprsError::Codec("raw: bands is required in LoadOptions".into()))?;

        Ok(Self {
            width,
            height,
            bands,
            offset: opts.raw_offset.unwrap_or(0),
            format: opts.raw_format.or(Some(F::ID)),
            endianness: opts.raw_endianness.unwrap_or(RawEndianness::native()),
            interpretation: opts.raw_interpretation,
        })
    }
}

fn validate_raw_layout(width: u32, height: u32, bands: u32) -> Result<(), ViprsError> {
    if width == 0 {
        return Err(ViprsError::Codec("raw: width must be > 0".into()));
    }
    if height == 0 {
        return Err(ViprsError::Codec("raw: height must be > 0".into()));
    }
    if bands == 0 {
        return Err(ViprsError::Codec("raw: bands must be > 0".into()));
    }
    Ok(())
}

fn validate_raw_format<F: BandFormat>(declared: Option<BandFormatId>) -> Result<(), ViprsError> {
    if let Some(format) = declared
        && format != F::ID
    {
        return Err(ViprsError::Codec(format!(
            "raw: declared format {:?} does not match requested {:?}",
            format,
            F::ID
        )));
    }
    Ok(())
}

fn expected_byte_count(
    width: u32,
    height: u32,
    bands: u32,
    sample_size: usize,
) -> Result<usize, ViprsError> {
    let pixels = (width as usize)
        .checked_mul(height as usize)
        .and_then(|value| value.checked_mul(bands as usize))
        .ok_or_else(|| ViprsError::Codec("raw: image dimensions overflow usize".into()))?;
    pixels
        .checked_mul(sample_size)
        .ok_or_else(|| ViprsError::Codec("raw: total byte count overflows usize".into()))
}

fn decode_samples<F: BandFormat>(
    bytes: &[u8],
    source_endianness: RawEndianness,
) -> Result<Vec<F::Sample>, ViprsError> {
    let sample_size = std::mem::size_of::<F::Sample>();
    let storage = if needs_byteswap(source_endianness, sample_size) {
        let mut owned = bytes.to_vec();
        swap_bytes_in_place(&mut owned, sample_size);
        Cow::Owned(owned)
    } else {
        Cow::Borrowed(bytes)
    };

    let sample_count = storage.len() / sample_size;
    let mut samples = Vec::with_capacity(sample_count);
    for chunk in storage.as_ref().chunks_exact(sample_size) {
        samples.push(bytemuck::pod_read_unaligned::<F::Sample>(chunk));
    }
    Ok(samples)
}

fn encode_bytes<T: bytemuck::Pod>(
    samples: &[T],
    target_endianness: RawEndianness,
) -> Result<Vec<u8>, ViprsError> {
    let sample_size = std::mem::size_of::<T>();
    let mut bytes = bytemuck::cast_slice(samples).to_vec();
    if needs_byteswap(target_endianness, sample_size) {
        swap_bytes_in_place(&mut bytes, sample_size);
    }
    Ok(bytes)
}

fn needs_byteswap(endianness: RawEndianness, sample_size: usize) -> bool {
    sample_size > 1 && endianness != RawEndianness::native()
}

fn swap_bytes_in_place(bytes: &mut [u8], sample_size: usize) {
    for chunk in bytes.chunks_exact_mut(sample_size) {
        chunk.reverse();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::{
        fs,
        path::{Path, PathBuf},
    };
    use viprs_core::{
        codec_options::LoadOptions,
        format::{F32, U8, U16},
    };

    fn opposite_endianness() -> RawEndianness {
        match RawEndianness::native() {
            RawEndianness::Little => RawEndianness::Big,
            RawEndianness::Big => RawEndianness::Little,
        }
    }

    fn test_output_path(name: &str) -> PathBuf {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("raw-codec-tests");
        fs::create_dir_all(&dir).unwrap();
        dir.join(format!("{name}-{}.raw", std::process::id()))
    }

    #[test]
    fn round_trip_u8_grayscale() {
        let codec = RawCodec;
        let pixels: Vec<u8> = (0..64).collect();
        let image = Image::<U8>::from_buffer(8, 8, 1, pixels.clone()).unwrap();

        let encoded = codec.encode(&image).unwrap();
        assert_eq!(encoded, pixels);

        let decoded = codec
            .decode_raw::<U8>(&encoded, &RawLoadOptions::new(8, 8, 1))
            .unwrap();
        assert_eq!(decoded.pixels(), image.pixels());
    }

    #[test]
    fn raw_option_builders_convert_to_generic_codec_options() {
        let load_opts: LoadOptions = RawLoadOptions::new(7, 5, 3)
            .with_offset(12)
            .with_format(BandFormatId::U16)
            .with_endianness(RawEndianness::Big)
            .with_interpretation(Interpretation::Srgb)
            .into();
        assert_eq!(load_opts.raw_width, Some(7));
        assert_eq!(load_opts.raw_height, Some(5));
        assert_eq!(load_opts.raw_bands, Some(3));
        assert_eq!(load_opts.raw_offset, Some(12));
        assert_eq!(load_opts.raw_format, Some(BandFormatId::U16));
        assert_eq!(load_opts.raw_endianness, Some(RawEndianness::Big));
        assert_eq!(load_opts.raw_interpretation, Some(Interpretation::Srgb));

        let lsb_opts: LoadOptions = RawLoadOptions::new(1, 1, 1).lsb_first().into();
        assert_eq!(lsb_opts.raw_endianness, Some(RawEndianness::Little));

        let msb_opts: LoadOptions = RawLoadOptions::new(1, 1, 1).msb_first().into();
        assert_eq!(msb_opts.raw_endianness, Some(RawEndianness::Big));

        let save_opts: SaveOptions = RawSaveOptions::new()
            .with_endianness(RawEndianness::Little)
            .into();
        assert_eq!(save_opts.raw_endianness, Some(RawEndianness::Little));
    }

    #[test]
    fn round_trip_u16_via_generic_options() {
        let codec = RawCodec;
        let pixels: Vec<u16> = (0u16..16).collect();
        let image = Image::<U16>::from_buffer(4, 4, 1, pixels).unwrap();
        let encoded = codec.encode(&image).unwrap();

        let decoded = codec
            .decode_with_options::<U16>(
                &encoded,
                &LoadOptions::default()
                    .with_raw_layout(4, 4, 1)
                    .with_raw_format(BandFormatId::U16),
            )
            .unwrap();

        assert_eq!(decoded.pixels(), image.pixels());
    }

    #[test]
    fn round_trip_f32_rgba() {
        let codec = RawCodec;
        let pixels: Vec<f32> = (0..32).map(|i| i as f32 / 31.0).collect();
        let image = Image::<F32>::from_buffer(2, 4, 4, pixels).unwrap();
        let encoded = codec.encode(&image).unwrap();
        let decoded = codec
            .decode_raw::<F32>(
                &encoded,
                &RawLoadOptions::new(2, 4, 4).with_format(BandFormatId::F32),
            )
            .unwrap();
        assert_eq!(decoded.pixels(), image.pixels());
    }

    #[test]
    fn decode_with_offset_skips_header_bytes() {
        let codec = RawCodec;
        let mut buf = vec![0xffu8; 16];
        buf.extend_from_slice(&[10, 20, 30, 40]);
        let decoded = codec
            .decode_raw::<U8>(&buf, &RawLoadOptions::new(2, 2, 1).with_offset(16))
            .unwrap();
        assert_eq!(decoded.pixels(), &[10, 20, 30, 40]);
    }

    #[test]
    fn decode_path_with_options_uses_explicit_layout() {
        let codec = RawCodec;
        let path = test_output_path("decode-path");
        fs::write(&path, [1u8, 2, 3, 4]).unwrap();

        let decoded = codec
            .decode_path_with_options::<U8>(&path, &LoadOptions::default().with_raw_layout(2, 2, 1))
            .unwrap();

        assert_eq!(decoded.width(), 2);
        assert_eq!(decoded.height(), 2);
        assert_eq!(decoded.pixels(), &[1, 2, 3, 4]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn encode_with_requested_endianness_swaps_multibyte_samples() {
        let codec = RawCodec;
        let image = Image::<U16>::from_buffer(2, 1, 1, vec![0x0102, 0x0304]).unwrap();
        let native = codec.encode(&image).unwrap();
        let swapped = codec
            .encode_with_options(
                &image,
                &SaveOptions::default().with_raw_endianness(opposite_endianness()),
            )
            .unwrap();

        let mut expected = native.clone();
        swap_bytes_in_place(&mut expected, std::mem::size_of::<u16>());
        assert_eq!(swapped, expected);
    }

    #[test]
    fn decode_with_requested_endianness_restores_original_values() {
        let codec = RawCodec;
        let image = Image::<U16>::from_buffer(2, 1, 1, vec![0x0102, 0x0304]).unwrap();
        let encoded = codec
            .encode_with_options(
                &image,
                &SaveOptions::default().with_raw_endianness(opposite_endianness()),
            )
            .unwrap();

        let decoded = codec
            .decode_with_options::<U16>(
                &encoded,
                &LoadOptions::default()
                    .with_raw_layout(2, 1, 1)
                    .with_raw_format(BandFormatId::U16)
                    .with_raw_endianness(opposite_endianness()),
            )
            .unwrap();

        assert_eq!(decoded.pixels(), image.pixels());
    }

    #[test]
    fn interpretation_is_attached_to_decoded_image() {
        let codec = RawCodec;
        let decoded = codec
            .decode_raw::<U8>(
                &[42u8; 9],
                &RawLoadOptions::new(3, 1, 3).with_interpretation(Interpretation::Srgb),
            )
            .unwrap();
        assert_eq!(
            decoded.metadata().interpretation,
            Some(Interpretation::Srgb)
        );
    }

    #[test]
    fn extra_trailing_bytes_are_ignored() {
        let codec = RawCodec;
        let mut buf = vec![1u8, 2, 3, 4];
        buf.extend_from_slice(&[0xffu8; 8]);
        let decoded = codec
            .decode_raw::<U8>(&buf, &RawLoadOptions::new(2, 2, 1))
            .unwrap();
        assert_eq!(decoded.pixels(), &[1, 2, 3, 4]);
    }

    #[test]
    fn decode_without_layout_errors() {
        let codec = RawCodec;
        let err = codec.decode::<U8>(&[1, 2, 3, 4]).unwrap_err();
        assert!(err.to_string().contains("width/height/bands"));
    }

    #[test]
    fn decode_with_missing_raw_options_errors() {
        let codec = RawCodec;
        let err = codec
            .decode_with_options::<U8>(&[1, 2, 3, 4], &LoadOptions::default())
            .unwrap_err();
        assert!(err.to_string().contains("width"));
    }

    #[test]
    fn decode_with_format_mismatch_errors() {
        let codec = RawCodec;
        let err = codec
            .decode_raw::<U8>(
                &[1, 2, 3, 4],
                &RawLoadOptions::new(2, 2, 1).with_format(BandFormatId::U16),
            )
            .unwrap_err();
        assert!(err.to_string().contains("declared format"));
    }

    #[test]
    fn decode_rejects_offset_beyond_buffer() {
        let codec = RawCodec;
        let err = codec
            .decode_raw::<U8>(&[1, 2, 3, 4], &RawLoadOptions::new(1, 1, 1).with_offset(10))
            .unwrap_err();
        assert!(err.to_string().contains("offset"));
    }

    #[test]
    fn decode_rejects_zero_layout_components() {
        let codec = RawCodec;
        for opts in [
            RawLoadOptions::new(0, 1, 1),
            RawLoadOptions::new(1, 0, 1),
            RawLoadOptions::new(1, 1, 0),
        ] {
            let err = codec.decode_raw::<U8>(&[1], &opts).unwrap_err();
            assert!(err.to_string().contains("must be > 0"));
        }
    }

    #[test]
    fn decode_rejects_short_buffers() {
        let codec = RawCodec;
        let err = codec
            .decode_raw::<U8>(&[0u8; 3], &RawLoadOptions::new(2, 2, 1))
            .unwrap_err();
        assert!(err.to_string().contains("expected"));
    }

    #[test]
    fn probe_always_errors() {
        let codec = RawCodec;
        let err = codec.probe(&[0u8; 4]).unwrap_err();
        assert!(err.to_string().contains("cannot probe"));
    }

    #[test]
    fn sniff_is_always_false_and_names_match() {
        let codec = RawCodec;
        assert!(!codec.sniff(b"raw data"));
        assert_eq!(ImageDecoder::format_name(&codec), "raw");
        assert_eq!(ImageEncoder::format_name(&codec), "raw");
    }

    #[test]
    fn can_decode_raw_paths_by_extension() {
        let codec = RawCodec;
        assert!(codec.can_decode_path(Path::new("image.RAW")));
        assert!(!codec.can_decode_path(Path::new("image.png")));
    }

    proptest! {
        #[test]
        fn prop_round_trip_u8(
            width in 1u32..=32,
            height in 1u32..=32,
            bands in 1u32..=4,
        ) {
            let count = (width * height * bands) as usize;
            let pixels: Vec<u8> = (0..count).map(|index| (index % 255) as u8).collect();
            let image = Image::<U8>::from_buffer(width, height, bands, pixels).unwrap();
            let codec = RawCodec;
            let encoded = codec.encode(&image).unwrap();
            let decoded = codec
                .decode_with_options::<U8>(
                    &encoded,
                    &LoadOptions::default().with_raw_layout(width, height, bands),
                )
                .unwrap();
            prop_assert_eq!(decoded.pixels(), image.pixels());
        }

        #[test]
        fn prop_round_trip_f32(
            width in 1u32..=16,
            height in 1u32..=16,
            bands in 1u32..=4,
        ) {
            let count = (width * height * bands) as usize;
            let pixels: Vec<f32> = (0..count).map(|index| index as f32 / 7.0).collect();
            let image = Image::<F32>::from_buffer(width, height, bands, pixels).unwrap();
            let codec = RawCodec;
            let encoded = codec
                .encode_with_options(
                    &image,
                    &SaveOptions::default().with_raw_endianness(opposite_endianness()),
                )
                .unwrap();
            let decoded = codec
                .decode_with_options::<F32>(
                    &encoded,
                    &LoadOptions::default()
                        .with_raw_layout(width, height, bands)
                        .with_raw_format(BandFormatId::F32)
                        .with_raw_endianness(opposite_endianness()),
                )
                .unwrap();
            prop_assert_eq!(decoded.pixels(), image.pixels());
        }
    }
}

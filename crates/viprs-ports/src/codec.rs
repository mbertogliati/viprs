//! Codec port traits: [`crate::codec::ImageDecoder`] and
//! [`crate::codec::ImageEncoder`].
//!
//! Concrete implementations live in the `viprs-codecs` crate and are gated by
//! Cargo feature flags (e.g., `feature = "jpeg"`). The traits here have no
//! feature dependencies — they express only the capability contract.
//!
//! # Dynamic registries
//!
//! These traits stay generic over `F: BandFormat`; the runtime
//! [`crate::adapters::foreign::ForeignRegistry`] adapts concrete codecs behind
//! an object-safe bridge (`ImageCodec`) when dynamic registration is needed.
//! The `probe` helper methods and `decode_with_options` /
//! `encode_with_options` remain `where Self: Sized` per GUIDELINES.md § 4.
//!
//! # Codec API v2
//!
//! [`crate::codec::ImageDecoder::decode_with_options`] and
//! [`crate::codec::ImageEncoder::encode_with_options`]
//! accept [`LoadOptions`] / [`SaveOptions`] for shrink-on-load, quality control,
//! and other hints.  The base `decode` / `encode` methods remain unchanged for
//! backward compatibility — they delegate to the `*_with_options` variant with
//! default options.

use std::{any::Any, fs, io::Write, path::Path};
use viprs_core::codec_options::{LoadOptions, SaveOptions};
use viprs_core::error::ViprsError;

use viprs_core::format::{BandFormat, BandFormatId};
use viprs_core::image::{Image, ImageMetadata, Region};

/// Header information needed to expose a decoded byte stream as an [`ImageSource`].
///
/// This is intentionally smaller than [`Image`]: it carries dimensions, band count,
/// and metadata without owning decoded pixels. Tile-streaming decoders return this
/// from [`TileImageDecoder::probe_with_options`] before any tile is requested.
///
/// [`ImageSource`]: crate::source::ImageSource
///
/// # Examples
///
/// ```rust
/// use viprs::domain::image::ImageMetadata;
/// use viprs::ports::codec::ImageMetadataProbe;
///
/// let probe = ImageMetadataProbe::new(640, 480, 3)
///     .with_metadata(ImageMetadata::default());
///
/// assert_eq!(probe.width, 640);
/// assert_eq!(probe.height, 480);
/// assert_eq!(probe.bands, 3);
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImageMetadataProbe {
    /// Decoded pixel width in samples.
    pub width: u32,
    /// Decoded pixel height in samples.
    pub height: u32,
    /// Number of bands each decoded pixel exposes.
    pub bands: u32,
    /// Metadata discovered while probing the encoded payload.
    pub metadata: ImageMetadata,
}

impl ImageMetadataProbe {
    /// Creates a new lightweight metadata probe from decoded dimensions and bands.
    ///
    /// This constructor solves early image planning when a decoder needs to
    /// expose shape information before full pixel materialization.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use viprs::ports::codec::ImageMetadataProbe;
    ///
    /// let probe = ImageMetadataProbe::new(320, 200, 4);
    /// assert_eq!(probe.width, 320);
    /// assert_eq!(probe.height, 200);
    /// assert_eq!(probe.bands, 4);
    /// ```
    #[must_use]
    pub fn new(width: u32, height: u32, bands: u32) -> Self {
        Self {
            width,
            height,
            bands,
            metadata: ImageMetadata::default(),
        }
    }

    /// Attaches decoded metadata to a probe without changing its dimensions.
    ///
    /// This builder-style helper keeps probe construction compact when a decoder
    /// discovers metadata such as ICC, EXIF, or orientation alongside the header.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use viprs::domain::image::ImageMetadata;
    /// use viprs::ports::codec::ImageMetadataProbe;
    ///
    /// let probe = ImageMetadataProbe::new(1, 1, 3)
    ///     .with_metadata(ImageMetadata::default());
    ///
    /// assert_eq!(probe.metadata, ImageMetadata::default());
    /// ```
    #[must_use]
    pub fn with_metadata(mut self, metadata: ImageMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Capability to decode a byte stream into an [`Image`].
///
/// # Thread safety
///
/// Implementors must be `Send + Sync`: the codec registry is shared across
/// threads; decoding individual files may run on rayon workers.
///
/// # Contract
///
/// - `decode` must not hold global locks longer than a single tile read.
/// - Metadata returned by `width`, `height`, and `bands` must be consistent
///   with the image produced by `decode`.
/// - `decode_with_options` must not error on unrecognised options; it must
///   silently ignore them (forward-compatibility guarantee).
///
/// # Examples
///
/// ```rust
/// use viprs::domain::{
///     codec_options::LoadOptions,
///     error::ViprsError,
///     format::U8,
///     image::Image,
/// };
/// use viprs::ports::codec::ImageDecoder;
///
/// struct StubDecoder;
///
/// impl ImageDecoder for StubDecoder {
///     fn format_name(&self) -> &'static str { "stub" }
///     fn sniff(&self, header: &[u8]) -> bool where Self: Sized { header.starts_with(b"STUB") }
///     fn decode<F: viprs::domain::format::BandFormat>(&self, _src: &[u8]) -> Result<Image<F>, ViprsError> {
///         Err(ViprsError::Codec("stub decoder".into()))
///     }
///     fn decode_with_options<F: viprs::domain::format::BandFormat>(
///         &self,
///         src: &[u8],
///         _opts: &LoadOptions,
///     ) -> Result<Image<F>, ViprsError>
///     where
///         Self: Sized,
///     {
///         self.decode(src)
///     }
///     fn probe(&self, _src: &[u8]) -> Result<(u32, u32, u32), ViprsError> where Self: Sized {
///         Ok((1, 1, 1))
///     }
/// }
///
/// let decoder = StubDecoder;
/// assert_eq!(decoder.format_name(), "stub");
/// let _ = decoder.decode_with_options::<U8>(b"STUB", &LoadOptions::default());
/// ```
pub trait ImageDecoder: Send + Sync {
    /// Human-readable format name (e.g., `"jpeg"`, `"png"`, `"webp"`).
    fn format_name(&self) -> &'static str;

    /// Return `true` when this decoder can consume `path` directly.
    ///
    /// Path-aware decoders use this to opt into registry dispatch without
    /// forcing the caller to materialise the input as a byte buffer first.
    fn can_decode_path(&self, _path: &Path) -> bool {
        false
    }

    /// Return `true` if this decoder recognises `header`.
    ///
    /// `header` is a small prefix of the file (at least 16 bytes when available).
    /// This method is called on every registered decoder in O(n) order; keep it
    /// fast (a few byte comparisons at most).
    ///
    /// Excluded from the vtable so the registry can call it via `where Self: Sized`
    /// when the concrete type is known, or via a thin wrapper when it is not.
    fn sniff(&self, header: &[u8]) -> bool
    where
        Self: Sized;

    /// Decode the entire byte stream `src` into an in-memory image.
    ///
    /// Equivalent to `decode_with_options(src, &LoadOptions::default())`.
    ///
    /// # Errors
    ///
    /// Returns [`ViprsError::Codec`] on malformed input, unsupported
    /// subformats, or I/O failures.
    fn decode<F: BandFormat>(&self, src: &[u8]) -> Result<Image<F>, ViprsError>;

    /// Decode the stable on-disk image at `path` into an in-memory image.
    ///
    /// Equivalent to `decode_path_with_options(path, &LoadOptions::default())`.
    ///
    /// # Errors
    ///
    /// Returns [`ViprsError::Codec`] on malformed input, unsupported
    /// subformats, or I/O failures.
    fn decode_path<F: BandFormat>(&self, path: &Path) -> Result<Image<F>, ViprsError>
    where
        Self: Sized,
    {
        self.decode_path_with_options(path, &LoadOptions::default())
    }

    /// Decode `src` applying the hints in `opts`.
    ///
    /// The most important hint for performance is [`LoadOptions::shrink_factor`]:
    /// JPEG and WebP can decode at 1/2, 1/4 or 1/8 native resolution with
    /// 4–8× throughput improvement, which is critical for thumbnail pipelines.
    ///
    /// Codecs that do not support a particular option must ignore it silently
    /// and fall back to their default behaviour.
    ///
    /// Excluded from the vtable (`where Self: Sized`) because it accepts a
    /// generic `opts` reference that would otherwise break object safety.
    ///
    /// # Errors
    ///
    /// Returns [`ViprsError::Codec`] on malformed input or I/O failures.
    fn decode_with_options<F: BandFormat>(
        &self,
        src: &[u8],
        opts: &LoadOptions,
    ) -> Result<Image<F>, ViprsError>
    where
        Self: Sized;

    /// Decode `path` applying the hints in `opts`.
    ///
    /// The default implementation reads the file into memory and delegates to
    /// [`Self::decode_with_options`]. Directory-backed codecs should override
    /// this method to open the stable path directly.
    ///
    /// # Errors
    ///
    /// Returns [`ViprsError::Codec`] on malformed input or I/O failures.
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

    /// Probe `src` for image dimensions without fully decoding.
    ///
    /// Returns `(width, height, bands)`. May do a partial decode internally;
    /// must not allocate a full pixel buffer.
    ///
    /// Excluded from the vtable (`where Self: Sized`) because it is an
    /// optimisation hint, not part of the core decode contract.
    fn probe(&self, src: &[u8]) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized;

    /// Probe `path` for image dimensions without fully decoding.
    ///
    /// The default implementation reads the file and delegates to
    /// [`Self::probe`]. Path-aware codecs can override this to avoid loading
    /// the full payload when the source already exists on disk.
    fn probe_path(&self, path: &Path) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized,
    {
        let src = fs::read(path)?;
        self.probe(&src)
    }
}

/// Capability to decode individual regions into caller-owned tile buffers.
///
/// This is the streaming counterpart to [`crate::codec::ImageDecoder`]. Implementors must be
/// able to satisfy any requested [`Region`] without allocating a full decoded
/// frame. The scheduler may call this through `DecoderSource::read_region` from
/// multiple threads and in arbitrary tile order, so implementations must keep
/// shared state immutable or internally synchronized at a granularity coarser
/// than the pixel loop.
///
/// The trait is intentionally not object-safe: the output sample type remains
/// generic over `F: BandFormat`, and `DecoderSource<D, F>` monomorphizes the
/// concrete decoder/type pair. Runtime codec registries continue to use
/// [`ImageCodec`] until a separate object-safe tile bridge is justified.
///
/// # Examples
///
/// ```rust
/// use viprs::domain::{
///     codec_options::LoadOptions,
///     error::ViprsError,
///     format::U8,
///     image::{Image, Region},
/// };
/// use viprs::ports::codec::{ImageDecoder, TileImageDecoder};
///
/// struct StubDecoder;
///
/// impl ImageDecoder for StubDecoder {
///     fn format_name(&self) -> &'static str { "stub" }
///     fn sniff(&self, _header: &[u8]) -> bool where Self: Sized { true }
///     fn decode<F: viprs::domain::format::BandFormat>(&self, _src: &[u8]) -> Result<Image<F>, ViprsError> {
///         Err(ViprsError::Codec("stub decoder".into()))
///     }
///     fn decode_with_options<F: viprs::domain::format::BandFormat>(
///         &self,
///         src: &[u8],
///         _opts: &LoadOptions,
///     ) -> Result<Image<F>, ViprsError>
///     where
///         Self: Sized,
///     {
///         self.decode(src)
///     }
///     fn probe(&self, _src: &[u8]) -> Result<(u32, u32, u32), ViprsError> where Self: Sized {
///         Ok((1, 1, 1))
///     }
/// }
///
/// impl TileImageDecoder for StubDecoder {
///     fn decode_region_into<F: viprs::domain::format::BandFormat>(
///         &self,
///         _src: &[u8],
///         _opts: &LoadOptions,
///         _region: Region,
///         output: &mut [u8],
///     ) -> Result<(), ViprsError>
///     where
///         Self: Sized,
///     {
///         output.fill(0);
///         Ok(())
///     }
/// }
///
/// let decoder = StubDecoder;
/// let mut tile = [255u8; 1];
/// decoder.decode_region_into::<U8>(
///     b"stub",
///     &LoadOptions::default(),
///     Region::new(0, 0, 1, 1),
///     &mut tile,
/// )?;
/// assert_eq!(tile, [0]);
/// # Ok::<(), ViprsError>(())
/// ```
pub trait TileImageDecoder: ImageDecoder {
    /// Probe dimensions and metadata after applying load options, without
    /// allocating a full decoded frame.
    ///
    /// The default implementation delegates to [`ImageDecoder::probe`] and
    /// returns empty metadata. Codecs that can read metadata without decoding
    /// pixels should override this method.
    ///
    /// # Errors
    ///
    /// Returns [`ViprsError::Codec`] on malformed input or unsupported headers.
    fn probe_with_options(
        &self,
        src: &[u8],
        _opts: &LoadOptions,
    ) -> Result<ImageMetadataProbe, ViprsError>
    where
        Self: Sized,
    {
        let (width, height, bands) = self.probe(src)?;
        Ok(ImageMetadataProbe::new(width, height, bands))
    }

    /// Probe `path` after applying load options, without fully decoding.
    ///
    /// The default implementation reads the source bytes and delegates to
    /// [`Self::probe_with_options`]. Directory-backed codecs should override
    /// this to reuse the stable path directly.
    fn probe_path_with_options(
        &self,
        path: &Path,
        opts: &LoadOptions,
    ) -> Result<ImageMetadataProbe, ViprsError>
    where
        Self: Sized,
    {
        let src = fs::read(path)?;
        self.probe_with_options(&src, opts)
    }

    /// Decode `region` directly into `output`.
    ///
    /// `output.len()` must be exactly `region.pixel_count() * bands *
    /// size_of::<F::Sample>()`, where `bands` is the value returned by
    /// [`Self::probe_with_options`]. Coordinates outside image bounds must use
    /// the same clamp-to-edge extension as [`crate::source::ImageSource`].
    ///
    /// # Errors
    ///
    /// Returns [`ViprsError::Codec`] on malformed input, unsupported output
    /// format, I/O failure, or output buffer length mismatch.
    fn decode_region_into<F: BandFormat>(
        &self,
        src: &[u8],
        opts: &LoadOptions,
        region: Region,
        output: &mut [u8],
    ) -> Result<(), ViprsError>
    where
        Self: Sized;

    /// Decode `region` from a stable path directly into `output`.
    ///
    /// The default implementation reads the source bytes and delegates to
    /// [`Self::decode_region_into`]. Path-aware codecs should override this to
    /// avoid re-materialising the encoded payload for every tile read.
    fn decode_region_from_path<F: BandFormat>(
        &self,
        path: &Path,
        opts: &LoadOptions,
        region: Region,
        output: &mut [u8],
    ) -> Result<(), ViprsError>
    where
        Self: Sized,
    {
        let src = fs::read(path)?;
        self.decode_region_into::<F>(&src, opts, region, output)
    }
}

/// Capability to encode an [`Image`] into a byte buffer.
///
/// # Thread safety
///
/// Implementors must be `Send + Sync`.
///
/// # Contract
///
/// - `encode` must not modify `image`; it receives an immutable reference.
/// - The returned `Vec<u8>` is a complete, self-contained byte stream for the
///   target format — not a partial write.
/// - `encode_with_options` must not error on unrecognised options; it must
///   silently ignore them (forward-compatibility guarantee).
///
/// # Examples
///
/// ```rust
/// use viprs::domain::{
///     codec_options::SaveOptions,
///     error::ViprsError,
///     format::U8,
///     image::Image,
/// };
/// use viprs::ports::codec::ImageEncoder;
///
/// struct StubEncoder;
///
/// impl ImageEncoder for StubEncoder {
///     fn format_name(&self) -> &'static str { "stub" }
///     fn encode<F: viprs::domain::format::BandFormat>(&self, _image: &Image<F>) -> Result<Vec<u8>, ViprsError> {
///         Ok(b"stub".to_vec())
///     }
///     fn encode_with_options<F: viprs::domain::format::BandFormat>(
///         &self,
///         image: &Image<F>,
///         _opts: &SaveOptions,
///     ) -> Result<Vec<u8>, ViprsError>
///     where
///         Self: Sized,
///     {
///         self.encode(image)
///     }
/// }
///
/// let encoder = StubEncoder;
/// let image = Image::<U8>::from_buffer(1, 1, 1, vec![0])?;
/// assert_eq!(encoder.encode_with_options(&image, &SaveOptions::default())?, b"stub");
/// # Ok::<(), ViprsError>(())
/// ```
pub trait ImageEncoder: Send + Sync {
    /// Human-readable format name.
    fn format_name(&self) -> &'static str;

    /// Encode `image` and return the resulting byte stream.
    ///
    /// Equivalent to `encode_with_options(image, &SaveOptions::default())`.
    ///
    /// # Errors
    ///
    /// Returns [`ViprsError::Codec`] on encoding failure (e.g., a pixel
    /// value that cannot be represented in the target format).
    fn encode<F: BandFormat>(&self, image: &Image<F>) -> Result<Vec<u8>, ViprsError>;

    /// Encode `image` applying the hints in `opts`.
    ///
    /// Key options:
    /// - [`SaveOptions::quality`]: JPEG/WebP lossy quality 0–100.
    /// - [`SaveOptions::lossless`]: WebP lossless mode.
    /// - [`SaveOptions::method`]: WebP encode effort 0–6.
    /// - [`SaveOptions::compression_level`]: PNG/TIFF compression effort.
    /// - [`SaveOptions::interlace`]: JPEG progressive or PNG Adam7 output.
    /// - [`SaveOptions::restart_interval`]: JPEG restart-marker cadence.
    /// - [`SaveOptions::jpeg_subsampling`]: JPEG chroma subsampling mode.
    /// - [`SaveOptions::png_filter`]: PNG row filter strategy.
    /// - [`SaveOptions::tile_width`] / [`SaveOptions::tile_height`]: TIFF tile layout.
    /// - [`SaveOptions::tiff_compression`]: TIFF compression strategy.
    /// - [`SaveOptions::tiff_predictor`]: TIFF horizontal predictor toggle.
    /// - [`SaveOptions::pyramid`]: TIFF pyramid output toggle.
    /// - [`SaveOptions::effort`]: AVIF/HEIF encoder effort tradeoff.
    /// - [`SaveOptions::dither`]: GIF Floyd-Steinberg remap toggle.
    /// - [`SaveOptions::colors`]: GIF palette size limit.
    ///
    /// Codecs that do not support a particular option must ignore it silently.
    ///
    /// Excluded from the vtable (`where Self: Sized`) — same rationale as
    /// `decode_with_options`.
    ///
    /// # Errors
    ///
    /// Returns [`ViprsError::Codec`] on encoding failure.
    fn encode_with_options<F: BandFormat>(
        &self,
        image: &Image<F>,
        opts: &SaveOptions,
    ) -> Result<Vec<u8>, ViprsError>
    where
        Self: Sized;

    /// Encode `image` and write the resulting byte stream to `writer`.
    ///
    /// This enables streaming: a server can begin chunked transfer before the
    /// entire encode completes, provided the format supports sequential
    /// writing.
    ///
    /// The default implementation encodes to a temporary `Vec<u8>` via
    /// [`Self::encode_with_options`] and forwards it with [`Write::write_all`].
    /// Codecs that support incremental output should override this method for
    /// true streaming.
    ///
    /// # Errors
    ///
    /// Returns [`ViprsError::Codec`] on encoding failure or propagates I/O
    /// errors returned by `writer`.
    fn encode_to_writer<F: BandFormat>(
        &self,
        image: &Image<F>,
        opts: &SaveOptions,
        writer: &mut dyn Write,
    ) -> Result<(), ViprsError>
    where
        Self: Sized,
    {
        let buf = self.encode_with_options(image, opts)?;
        writer.write_all(&buf)?;
        Ok(())
    }
}

/// Object-safe bridge used by [`crate::adapters::codecs::registry::ForeignRegistry`].
///
/// Concrete codecs stay monomorphised over `F: BandFormat`; this trait erases
/// the format only at the runtime registry boundary where `dyn Trait` is
/// explicitly allowed by project rules.
///
/// # Examples
///
/// ```rust
/// use std::any::Any;
///
/// use viprs::domain::{
///     codec_options::{LoadOptions, SaveOptions},
///     error::ViprsError,
///     format::{BandFormatId, U8},
///     image::Image,
/// };
/// use viprs::ports::codec::ImageCodec;
///
/// struct StubCodec;
///
/// impl ImageCodec for StubCodec {
///     fn format_name(&self) -> &'static str { "stub" }
///     fn file_extensions(&self) -> &'static [&'static str] { &["stub"] }
///     fn sniff(&self, header: &[u8]) -> bool { header.starts_with(b"STUB") }
///     fn decode_boxed(
///         &self,
///         _src: &[u8],
///         _band_format: BandFormatId,
///         _opts: &LoadOptions,
///     ) -> Result<Box<dyn Any + Send>, ViprsError> {
///         Ok(Box::new(Image::<U8>::from_buffer(1, 1, 1, vec![0])?))
///     }
///     fn encode_boxed(
///         &self,
///         _image: &(dyn Any + Send + Sync),
///         _band_format: BandFormatId,
///         _opts: &SaveOptions,
///     ) -> Result<Vec<u8>, ViprsError> {
///         Ok(b"stub".to_vec())
///     }
/// }
///
/// let codec = StubCodec;
/// assert!(codec.supports_format("stub"));
/// assert!(codec.sniff(b"STUB"));
/// ```
pub trait ImageCodec: Send + Sync {
    /// Human-readable format name.
    fn format_name(&self) -> &'static str;

    /// Lowercase file extensions recognised by this codec, without the dot.
    fn file_extensions(&self) -> &'static [&'static str];

    /// Return `true` if this codec can encode images for save operations.
    fn can_encode(&self) -> bool {
        true
    }

    /// Return `true` when decode can be attempted by file extension if no
    /// header-sniff decoder matched.
    ///
    /// This is reserved for low-priority fallback loaders (for example,
    /// ImageMagick-backed adapters) so they do not interfere with native
    /// sniff-first decoders.
    fn supports_extension_decode_fallback(&self) -> bool {
        false
    }

    /// Return `true` if `suffix` names a format this codec can save.
    fn supports_format(&self, suffix: &str) -> bool {
        self.file_extensions()
            .iter()
            .any(|candidate| candidate.eq_ignore_ascii_case(suffix))
    }

    /// Sniff an input header to decide whether this codec can decode it.
    fn sniff(&self, header: &[u8]) -> bool;

    /// Decode `src` into a boxed typed [`Image`], applying `opts`.
    fn decode_boxed(
        &self,
        src: &[u8],
        band_format: BandFormatId,
        opts: &LoadOptions,
    ) -> Result<Box<dyn Any + Send>, ViprsError>;

    /// Return `true` when this registry codec can decode `path` directly.
    fn can_decode_path(&self, _path: &Path) -> bool {
        false
    }

    /// Decode `path` into a boxed typed [`Image`], applying `opts`.
    fn decode_boxed_path(
        &self,
        path: &Path,
        band_format: BandFormatId,
        opts: &LoadOptions,
    ) -> Result<Box<dyn Any + Send>, ViprsError> {
        let src = fs::read(path)?;
        self.decode_boxed(&src, band_format, opts)
    }

    /// Encode a boxed typed [`Image`], applying `opts`.
    fn encode_boxed(
        &self,
        image: &(dyn Any + Send + Sync),
        band_format: BandFormatId,
        opts: &SaveOptions,
    ) -> Result<Vec<u8>, ViprsError>;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use viprs_core::format::U8;

    /// Minimal round-trip codec used to verify the trait contract compiles and
    /// behaves correctly with a trivial raw-pixel format.
    struct RawU8Codec;

    /// Raw pixel format: width (4 bytes LE) + height (4 bytes LE) +
    /// bands (4 bytes LE) + pixel data.
    impl ImageDecoder for RawU8Codec {
        fn format_name(&self) -> &'static str {
            "raw_u8"
        }

        fn sniff(&self, header: &[u8]) -> bool {
            // Magic: first 4 bytes spell b"RAWU".
            header.len() >= 4 && &header[..4] == b"RAWU"
        }

        fn decode<F: BandFormat>(&self, _src: &[u8]) -> Result<Image<F>, ViprsError> {
            // Intentionally not implemented for non-U8 formats in this test stub.
            Err(ViprsError::Codec("RawU8Codec only decodes U8".into()))
        }

        fn decode_with_options<F: BandFormat>(
            &self,
            src: &[u8],
            _opts: &LoadOptions,
        ) -> Result<Image<F>, ViprsError> {
            // Test stub ignores all options and delegates to the base decode.
            self.decode(src)
        }

        fn probe(&self, src: &[u8]) -> Result<(u32, u32, u32), ViprsError> {
            if src.len() < 16 {
                return Err(ViprsError::Codec("header too short".into()));
            }
            let w = u32::from_le_bytes(src[4..8].try_into().unwrap());
            let h = u32::from_le_bytes(src[8..12].try_into().unwrap());
            let b = u32::from_le_bytes(src[12..16].try_into().unwrap());
            Ok((w, h, b))
        }
    }

    impl ImageEncoder for RawU8Codec {
        fn format_name(&self) -> &'static str {
            "raw_u8"
        }

        fn encode<F: BandFormat>(&self, image: &Image<F>) -> Result<Vec<u8>, ViprsError> {
            // Encode as header-only for test purposes (no actual pixel data).
            let mut buf = Vec::with_capacity(16);
            buf.extend_from_slice(b"RAWU");
            buf.extend_from_slice(&image.width().to_le_bytes());
            buf.extend_from_slice(&image.height().to_le_bytes());
            buf.extend_from_slice(&image.bands().to_le_bytes());
            Ok(buf)
        }

        fn encode_with_options<F: BandFormat>(
            &self,
            image: &Image<F>,
            _opts: &SaveOptions,
        ) -> Result<Vec<u8>, ViprsError> {
            // Test stub ignores all options and delegates to the base encode.
            self.encode(image)
        }
    }

    #[test]
    fn sniff_recognises_magic() {
        let codec = RawU8Codec;
        assert!(codec.sniff(b"RAWU\x00\x00\x00\x00"));
        assert!(!codec.sniff(b"PNG\x00\x00\x00\x00\x00"));
    }

    #[test]
    fn probe_extracts_dimensions() {
        let codec = RawU8Codec;
        let mut header = Vec::new();
        header.extend_from_slice(b"RAWU");
        header.extend_from_slice(&32u32.to_le_bytes()); // width
        header.extend_from_slice(&16u32.to_le_bytes()); // height
        header.extend_from_slice(&3u32.to_le_bytes()); // bands
        assert_eq!(codec.probe(&header).unwrap(), (32, 16, 3));
    }

    #[test]
    fn encode_writes_header() {
        let codec = RawU8Codec;
        let image = Image::<U8>::from_buffer(4, 4, 1, vec![0u8; 16]).unwrap();
        let encoded = codec.encode(&image).unwrap();
        assert_eq!(&encoded[..4], b"RAWU");
        assert_eq!(u32::from_le_bytes(encoded[4..8].try_into().unwrap()), 4);
        assert_eq!(u32::from_le_bytes(encoded[8..12].try_into().unwrap()), 4);
        assert_eq!(u32::from_le_bytes(encoded[12..16].try_into().unwrap()), 1);
    }

    #[test]
    fn format_name_is_consistent() {
        let codec = RawU8Codec;
        assert_eq!(
            <RawU8Codec as ImageDecoder>::format_name(&codec),
            <RawU8Codec as ImageEncoder>::format_name(&codec)
        );
    }

    #[test]
    fn decode_with_options_compiles_and_delegates() {
        let codec = RawU8Codec;
        let opts = LoadOptions::default().with_max_dimension(256);
        // decode_with_options on this stub just delegates to decode — the
        // important thing is that the trait method compiles and is callable.
        let result = codec.decode_with_options::<U8>(b"", &opts);
        assert!(result.is_err()); // stub always errors
    }

    #[test]
    fn encode_with_options_compiles_and_delegates() {
        let codec = RawU8Codec;
        let image = Image::<U8>::from_buffer(2, 2, 1, vec![0u8; 4]).unwrap();
        let opts = SaveOptions::default().with_quality(80);
        let encoded = codec.encode_with_options(&image, &opts).unwrap();
        assert_eq!(&encoded[..4], b"RAWU");
    }

    #[test]
    fn encode_to_writer_default_impl_writes_encoded_bytes() {
        let codec = RawU8Codec;
        let image = Image::<U8>::from_buffer(2, 2, 1, vec![0u8; 4]).unwrap();
        let opts = SaveOptions::default().with_quality(80);
        let mut encoded = Vec::new();

        codec.encode_to_writer(&image, &opts, &mut encoded).unwrap();

        assert_eq!(&encoded[..4], b"RAWU");
        assert_eq!(u32::from_le_bytes(encoded[4..8].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(encoded[8..12].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(encoded[12..16].try_into().unwrap()), 1);
    }
}

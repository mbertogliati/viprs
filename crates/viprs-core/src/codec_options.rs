//! Load and save options for codec operations.
//!
//! # Design rationale
//!
//! Options are expressed as a *generic* struct rather than per-codec types.
//! This keeps the codec traits uniform and lets the foreign-registry bridge
//! dispatch on runtime format ids without an explosion of `JpegLoadOptions`,
//! `WebpSaveOptions`, … types. The alternative (per-codec types) breaks the
//! unified trait interface and forces callers to know the concrete codec type
//! at compile time — incompatible with a runtime-extensible registry.
//!
//! Fields that are codec-specific carry a doc comment listing which codecs
//! honour them.  Codecs that receive an unsupported option MUST ignore it, not
//! error — forward-compatibility is more important than strict validation here.

use std::num::{NonZeroU8, NonZeroUsize};

use crate::{format::BandFormatId, image::Interpretation, limits::DecodeLimits};

/// Byte order for headerless RAW pixel streams.
///
/// RAW codecs use this when no container metadata exists to describe sample endianness.
///
/// # Examples
/// ```rust
/// # use viprs::domain::codec_options::RawEndianness;
/// let _native = RawEndianness::native();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawEndianness {
    /// Uses the `Little` variant of `RawEndianness`.
    Little,
    /// Uses the `Big` variant of `RawEndianness`.
    Big,
}

impl RawEndianness {
    #[must_use]
    /// Returns or performs native.
    pub const fn native() -> Self {
        if cfg!(target_endian = "little") {
            Self::Little
        } else {
            Self::Big
        }
    }
}

// ── Load options ──────────────────────────────────────────────────────────────

/// Options passed to `ImageDecoder::decode_with_options`.
///
/// All fields are `Option`; `None` means "use the codec's default".
/// Codecs that do not support a given option must ignore it silently.
/// Load-time codec options shared across decoders.
///
/// This collects common decode knobs so callers can configure codecs without depending on
/// format-specific adapter types.
///
/// # Examples
/// ```rust
/// # use std::num::NonZeroU8;
/// # use viprs::domain::codec_options::LoadOptions;
/// let options = LoadOptions::default().with_shrink(NonZeroU8::new(2).unwrap());
/// assert_eq!(options.shrink_factor.unwrap().get(), 2);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct LoadOptions {
    /// Integer shrink factor applied *during* decoding (not as a post-decode
    /// resize).  Only honoured by codecs that support native downscaling:
    ///
    /// | Codec | Support | Notes |
    /// |-------|---------|-------|
    /// | JPEG  | yes     | libjpeg DCT-domain shrink: factors 2, 4, 8 |
    /// | WebP  | yes     | libwebp has `WebPDecodeRGBScaled` |
    /// | PNG   | no      | no native shrink; adapter must ignore |
    /// | TIFF  | partial | depends on sub-format |
    ///
    /// A factor of 2 means the decoded image will be approximately half the
    /// original size in each dimension.  The exact output size is
    /// codec-dependent (JPEG may round to MCU boundaries).  `None` = no shrink.
    pub shrink_factor: Option<NonZeroU8>,

    /// If set, the decoder *may* limit the decoded image so that neither
    /// dimension exceeds `max_dimension` pixels.  The aspect ratio is preserved.
    /// This is a hint: the codec may produce a larger image if its native
    /// shrink factors do not align perfectly.
    ///
    /// When both `shrink_factor` and `max_dimension` are set, `shrink_factor`
    /// takes precedence.
    pub max_dimension: Option<u32>,

    /// Decode safety limits validated before allocating the decoded buffer.
    ///
    /// `None` preserves legacy behaviour and leaves validation to the caller or
    /// codec adapter. Set this to [`DecodeLimits::default_safe`] for untrusted
    /// input.
    pub limits: Option<DecodeLimits>,

    /// If `true`, request that the codec populate orientation/EXIF metadata
    /// without applying the rotation.  If `false` (default), auto-rotate on
    /// load when EXIF orientation is present.
    ///
    /// Currently only relevant for JPEG and HEIF.
    pub no_rotate: bool,

    /// Rasterization DPI for vector decoders.
    ///
    /// Honoured by SVG. `None` = codec default (72 DPI for libvips parity).
    pub dpi: Option<f64>,

    /// Additional render scale factor for vector decoders.
    ///
    /// Honoured by SVG. `None` = no additional scaling.
    pub scale: Option<f64>,

    /// Zero-based page to decode from multi-page containers.
    ///
    /// Honoured by TIFF, AVIF, and HEIF. `None` = codec default page
    /// (primary image for AVIF/HEIF, first page for TIFF).
    pub page: Option<u32>,

    /// Number of pages to decode from multi-page containers.
    ///
    /// Honoured by TIFF, AVIF, and HEIF. `Some(-1)` means "all remaining pages".
    /// `None` = one page.
    pub n: Option<i32>,

    /// Explicit width for headerless RAW input.
    pub raw_width: Option<u32>,

    /// Explicit height for headerless RAW input.
    pub raw_height: Option<u32>,

    /// Explicit interleaved band count for headerless RAW input.
    pub raw_bands: Option<u32>,

    /// Byte offset to skip before RAW pixel data.
    pub raw_offset: Option<u64>,

    /// Explicit sample format for RAW input.
    pub raw_format: Option<BandFormatId>,

    /// Source byte order for RAW input.
    pub raw_endianness: Option<RawEndianness>,

    /// Interpretation metadata to attach to RAW input.
    pub raw_interpretation: Option<Interpretation>,

    /// Decoder worker count hint for codecs with an internal thread pool.
    ///
    /// Honoured by JP2K/OpenJPEG. `None` = codec default.
    pub decoder_threads: Option<NonZeroUsize>,
}

impl LoadOptions {
    /// Construct a `LoadOptions` with all fields set to their defaults
    /// (no shrink, no dimension limit, auto-rotate enabled).
    #[must_use]
    pub const fn default_options() -> Self {
        Self {
            shrink_factor: None,
            max_dimension: None,
            limits: None,
            no_rotate: false,
            dpi: None,
            scale: None,
            page: None,
            n: None,
            raw_width: None,
            raw_height: None,
            raw_bands: None,
            raw_offset: None,
            raw_format: None,
            raw_endianness: None,
            raw_interpretation: None,
            decoder_threads: None,
        }
    }

    /// Builder: set shrink factor.
    #[must_use]
    pub const fn with_shrink(mut self, factor: NonZeroU8) -> Self {
        self.shrink_factor = Some(factor);
        self
    }

    /// Builder: set maximum dimension.
    #[must_use]
    pub const fn with_max_dimension(mut self, dim: u32) -> Self {
        self.max_dimension = Some(dim);
        self
    }

    /// Builder: disable auto-rotate on load.
    #[must_use]
    pub const fn no_rotate(mut self) -> Self {
        self.no_rotate = true;
        self
    }

    /// Builder: set rasterization DPI for vector decoders.
    #[must_use]
    pub const fn with_dpi(mut self, dpi: f64) -> Self {
        self.dpi = Some(dpi);
        self
    }

    /// Builder: set additional rasterization scale for vector decoders.
    #[must_use]
    pub const fn with_scale(mut self, scale: f64) -> Self {
        self.scale = Some(scale);
        self
    }

    /// Builder: decode from a specific zero-based page.
    #[must_use]
    pub const fn with_page(mut self, page: u32) -> Self {
        self.page = Some(page);
        self
    }

    /// Builder: decode `n` pages from the selected page.
    #[must_use]
    pub const fn with_n(mut self, n: i32) -> Self {
        self.n = Some(n);
        self
    }

    /// Builder: set required RAW dimensions and band count.
    #[must_use]
    pub const fn with_raw_layout(mut self, width: u32, height: u32, bands: u32) -> Self {
        self.raw_width = Some(width);
        self.raw_height = Some(height);
        self.raw_bands = Some(bands);
        self
    }

    /// Builder: set RAW byte offset.
    #[must_use]
    pub const fn with_raw_offset(mut self, offset: u64) -> Self {
        self.raw_offset = Some(offset);
        self
    }

    /// Builder: set RAW sample format.
    #[must_use]
    pub const fn with_raw_format(mut self, format: BandFormatId) -> Self {
        self.raw_format = Some(format);
        self
    }

    /// Builder: set RAW source byte order.
    #[must_use]
    pub const fn with_raw_endianness(mut self, endianness: RawEndianness) -> Self {
        self.raw_endianness = Some(endianness);
        self
    }

    /// Builder: set RAW interpretation metadata.
    #[must_use]
    pub const fn with_raw_interpretation(mut self, interpretation: Interpretation) -> Self {
        self.raw_interpretation = Some(interpretation);
        self
    }

    /// Builder: set decoder worker count for codecs that expose internal threading.
    #[must_use]
    pub const fn with_decoder_threads(mut self, threads: NonZeroUsize) -> Self {
        self.decoder_threads = Some(threads);
        self
    }
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self::default_options()
    }
}

// ── Save options ──────────────────────────────────────────────────────────────

/// Options passed to `ImageEncoder::encode_with_options`.
///
/// All fields are `Option`; `None` means "use the codec's default".
/// Codecs that do not support a given option must ignore it silently.
/// Save-time codec options shared across encoders.
///
/// This keeps encoder configuration uniform across formats while still allowing codec-specific
/// knobs to be ignored when unsupported.
///
/// # Examples
/// ```rust
/// # use viprs::domain::codec_options::SaveOptions;
/// let options = SaveOptions::default().with_quality(90).lossless();
/// assert_eq!(options.quality, Some(90));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SaveOptions {
    /// Lossy quality, 0–100 (higher = better quality / larger file).
    ///
    /// Honoured by JPEG (maps to libjpeg `quality`) and WebP (lossy mode).
    /// Ignored by PNG (lossless), GIF, TIFF (lossless).
    pub quality: Option<u8>,

    /// If `true`, request lossless encoding.
    ///
    /// Honoured by WebP.  For JPEG (inherently lossy) this option is ignored.
    /// For PNG this is always lossless; setting `lossless = false` has no effect.
    pub lossless: Option<bool>,

    /// Compression level for lossless formats, 0–9 (higher = smaller file but
    /// slower encoding).  Honoured by PNG (zlib level), TIFF (LZW predictor).
    pub compression_level: Option<u8>,

    /// Progressive / interlaced output toggle.
    ///
    /// Honoured by JPEG (`true` = progressive scans) and PNG (`true` = Adam7).
    pub interlace: Option<bool>,

    /// JPEG restart interval in MCUs.
    ///
    /// Honoured by JPEG encoders that can emit DRI/restart markers.
    pub restart_interval: Option<u16>,

    /// JPEG chroma subsampling mode.
    pub jpeg_subsampling: Option<JpegSubsampling>,

    /// PNG row filter strategy.
    pub png_filter: Option<PngFilterStrategy>,

    /// Enable or disable palette dithering during indexed-color encoding.
    ///
    /// Honoured by GIF. `true` applies Floyd-Steinberg dithering; `false`
    /// remaps each pixel directly to the nearest palette entry.
    pub dither: Option<bool>,

    /// Maximum number of palette colors for indexed output, 2–256.
    ///
    /// Honoured by GIF. If transparency is present, one entry is reserved for
    /// the transparent index.
    pub colors: Option<u16>,

    /// Strip all embedded metadata (EXIF, XMP, ICC profile) from the output.
    /// `None` = preserve metadata (codec default).
    pub strip_metadata: Option<bool>,

    /// Tile width for tiled TIFF output.  Ignored by non-tiled formats.
    pub tile_width: Option<u32>,

    /// Tile height for tiled TIFF output.  Ignored by non-tiled formats.
    pub tile_height: Option<u32>,

    /// TIFF compression strategy.
    pub tiff_compression: Option<TiffCompression>,

    /// TIFF predictor strategy for LZW / Deflate compression.
    pub tiff_predictor: Option<TiffPredictor>,

    /// Request pyramid output when the codec supports it.
    ///
    /// Honoured by TIFF.
    pub pyramid: Option<bool>,

    /// Encoder effort / speed tradeoff, 0–9 (higher = more CPU, smaller file).
    ///
    /// Honoured by AVIF/HEIF now, and by other codecs with explicit effort knobs.
    pub effort: Option<u8>,

    /// WebP encoder method / effort, 0–6 (higher = smaller file, slower encode).
    ///
    /// Honoured by WebP. `None` = codec default.
    pub method: Option<u8>,

    /// HEIF-family compression format.
    ///
    /// Honoured by HEIF. `Auto` preserves the codec default choice.
    pub heif_compression: Option<HeifCompression>,

    /// HEIF/AVIF chroma subsampling mode.
    ///
    /// Honoured by HEIF and AVIF when encoded through libheif.
    pub heif_subsampling: Option<HeifSubsampling>,

    /// HEIF/AVIF output bit depth.
    ///
    /// Honoured by HEIF and AVIF. HEIF can request 16-bit interleaved RGBA output;
    /// unsupported encoders return a codec error.
    pub heif_bit_depth: Option<HeifBitDepth>,

    /// WebP near-lossless preprocessing level, 0–100.
    ///
    /// Honoured by WebP in lossless mode. Lower values give smaller files but
    /// introduce slight preprocessing artifacts. `None` = disabled.
    pub near_lossless: Option<u8>,

    /// Preserve exact RGB values under transparent areas in WebP.
    ///
    /// Honoured by WebP. `true` prevents the encoder from modifying RGB channels
    /// where alpha == 0 (useful for round-trip fidelity).
    pub exact_alpha: Option<bool>,

    /// Enable WebP smart subsampling (adaptive chroma downsampling).
    ///
    /// Honoured by WebP lossy. When `true`, the encoder uses perceptual
    /// heuristics to improve chroma subsampling quality.
    pub smart_subsample: Option<bool>,

    /// Target byte order for RAW output.
    pub raw_endianness: Option<RawEndianness>,
}

/// Compression modes for TIFF output.
///
/// Encoders use this to choose size-versus-compatibility tradeoffs for tiled or scanline TIFFs.
///
/// # Examples
/// ```rust
/// # use viprs::domain::codec_options::TiffCompression;
/// assert!(matches!(TiffCompression::Lzw, TiffCompression::Lzw));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TiffCompression {
    /// Uses the `None` variant of `TiffCompression`.
    None,
    /// Uses the `Lzw` variant of `TiffCompression`.
    Lzw,
    /// Uses the `Deflate` variant of `TiffCompression`.
    Deflate,
    /// Uses the `PackBits` variant of `TiffCompression`.
    PackBits,
    /// Uses the `Jpeg` variant of `TiffCompression`.
    Jpeg,
}

/// Predictor modes for TIFF compression.
///
/// Predictors improve compressibility for suitable pixel data before TIFF encoders write blocks.
///
/// # Examples
/// ```rust
/// # use viprs::domain::codec_options::TiffPredictor;
/// assert!(matches!(TiffPredictor::Horizontal, TiffPredictor::Horizontal));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TiffPredictor {
    /// Uses the `None` variant of `TiffPredictor`.
    None,
    /// Uses the `Horizontal` variant of `TiffPredictor`.
    Horizontal,
}

/// JPEG chroma subsampling controls.
///
/// This chooses how aggressively chroma channels are downsampled during JPEG encoding.
///
/// # Examples
/// ```rust
/// # use viprs::domain::codec_options::JpegSubsampling;
/// assert!(matches!(JpegSubsampling::Off, JpegSubsampling::Off));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JpegSubsampling {
    /// Match encoder defaults for the selected quality.
    Auto,
    /// Disable chroma subsampling (4:4:4).
    Off,
    /// 4:2:0 chroma subsampling.
    Subsample420,
    /// 4:2:2 chroma subsampling.
    Subsample422,
    /// 4:4:0 chroma subsampling.
    Subsample440,
}

/// PNG row filter strategy.
///
/// PNG encoders use row filters to balance compression ratio against CPU cost.
///
/// # Examples
/// ```rust
/// # use viprs::domain::codec_options::PngFilterStrategy;
/// assert!(matches!(PngFilterStrategy::Adaptive, PngFilterStrategy::Adaptive));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PngFilterStrategy {
    /// Uses the `Adaptive` variant of `PngFilterStrategy`.
    Adaptive,
    /// Uses the `None` variant of `PngFilterStrategy`.
    None,
    /// Uses the `Sub` variant of `PngFilterStrategy`.
    Sub,
    /// Uses the `Up` variant of `PngFilterStrategy`.
    Up,
    /// Uses the `Avg` variant of `PngFilterStrategy`.
    Avg,
    /// Uses the `Paeth` variant of `PngFilterStrategy`.
    Paeth,
}

/// Compression families available to HEIF-compatible encoders.
///
/// This lets callers request a concrete bitstream family such as HEVC or AV1 when supported.
///
/// # Examples
/// ```rust
/// # use viprs::domain::codec_options::HeifCompression;
/// assert!(matches!(HeifCompression::Auto, HeifCompression::Auto));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeifCompression {
    /// Uses the `Auto` variant of `HeifCompression`.
    Auto,
    /// Uses the `Hevc` variant of `HeifCompression`.
    Hevc,
    /// Uses the `Avc` variant of `HeifCompression`.
    Avc,
    /// Uses the `Jpeg` variant of `HeifCompression`.
    Jpeg,
    /// Uses the `Av1` variant of `HeifCompression`.
    Av1,
}

/// Chroma subsampling controls for HEIF and AVIF output.
///
/// Encoders use this to trade file size against chroma fidelity.
///
/// # Examples
/// ```rust
/// # use viprs::domain::codec_options::HeifSubsampling;
/// assert!(matches!(HeifSubsampling::Subsample444, HeifSubsampling::Subsample444));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeifSubsampling {
    /// Uses the `Auto` variant of `HeifSubsampling`.
    Auto,
    /// Uses the `Subsample420` variant of `HeifSubsampling`.
    Subsample420,
    /// Uses the `Subsample422` variant of `HeifSubsampling`.
    Subsample422,
    /// Uses the `Subsample444` variant of `HeifSubsampling`.
    Subsample444,
}

/// Output bit depths for HEIF and AVIF encoders.
///
/// This keeps bit-depth selection explicit for HDR and wide-gamut workflows.
///
/// # Examples
/// ```rust
/// # use viprs::domain::codec_options::HeifBitDepth;
/// assert_eq!(HeifBitDepth::Ten.as_u8(), 10);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeifBitDepth {
    /// Uses the `Eight` variant of `HeifBitDepth`.
    Eight,
    /// Uses the `Ten` variant of `HeifBitDepth`.
    Ten,
    /// Uses the `Twelve` variant of `HeifBitDepth`.
    Twelve,
    /// Uses the `Sixteen` variant of `HeifBitDepth`.
    Sixteen,
}

impl HeifBitDepth {
    #[must_use]
    /// Returns this value as u8.
    pub const fn as_u8(self) -> u8 {
        match self {
            Self::Eight => 8,
            Self::Ten => 10,
            Self::Twelve => 12,
            Self::Sixteen => 16,
        }
    }
}

impl SaveOptions {
    /// Construct a `SaveOptions` with all fields set to `None` (codec defaults).
    #[must_use]
    pub const fn default_options() -> Self {
        Self {
            quality: None,
            lossless: None,
            compression_level: None,
            interlace: None,
            restart_interval: None,
            jpeg_subsampling: None,
            png_filter: None,
            dither: None,
            colors: None,
            strip_metadata: None,
            tile_width: None,
            tile_height: None,
            tiff_compression: None,
            tiff_predictor: None,
            pyramid: None,
            effort: None,
            method: None,
            heif_compression: None,
            heif_subsampling: None,
            heif_bit_depth: None,
            near_lossless: None,
            exact_alpha: None,
            smart_subsample: None,
            raw_endianness: None,
        }
    }

    /// Builder: set quality.
    #[must_use]
    pub const fn with_quality(mut self, q: u8) -> Self {
        self.quality = Some(q);
        self
    }

    /// Builder: enable lossless encoding.
    #[must_use]
    pub const fn lossless(mut self) -> Self {
        self.lossless = Some(true);
        self
    }

    /// Builder: set compression level.
    #[must_use]
    pub const fn with_compression_level(mut self, level: u8) -> Self {
        self.compression_level = Some(level);
        self
    }

    /// Builder: enable or disable progressive / interlaced output.
    #[must_use]
    pub const fn with_interlace(mut self, enabled: bool) -> Self {
        self.interlace = Some(enabled);
        self
    }

    /// Builder: set JPEG restart interval.
    #[must_use]
    pub const fn with_restart_interval(mut self, interval: u16) -> Self {
        self.restart_interval = Some(interval);
        self
    }

    /// Builder: set JPEG chroma subsampling.
    #[must_use]
    pub const fn with_jpeg_subsampling(mut self, subsampling: JpegSubsampling) -> Self {
        self.jpeg_subsampling = Some(subsampling);
        self
    }

    /// Builder: set the PNG row filter.
    #[must_use]
    pub const fn with_png_filter(mut self, filter: PngFilterStrategy) -> Self {
        self.png_filter = Some(filter);
        self
    }

    /// Builder: enable or disable dithering.
    #[must_use]
    pub const fn with_dither(mut self, enabled: bool) -> Self {
        self.dither = Some(enabled);
        self
    }

    /// Builder: set the maximum number of palette colors.
    #[must_use]
    pub const fn with_colors(mut self, colors: u16) -> Self {
        self.colors = Some(colors);
        self
    }

    /// Builder: strip all embedded metadata.
    #[must_use]
    pub const fn strip_metadata(mut self) -> Self {
        self.strip_metadata = Some(true);
        self
    }

    /// Builder: set TIFF tile width.
    #[must_use]
    pub const fn with_tile_width(mut self, tile_width: u32) -> Self {
        self.tile_width = Some(tile_width);
        self
    }

    /// Builder: set TIFF tile height.
    #[must_use]
    pub const fn with_tile_height(mut self, tile_height: u32) -> Self {
        self.tile_height = Some(tile_height);
        self
    }

    /// Builder: set TIFF compression strategy.
    #[must_use]
    pub const fn with_tiff_compression(mut self, compression: TiffCompression) -> Self {
        self.tiff_compression = Some(compression);
        self
    }

    /// Builder: set TIFF predictor.
    #[must_use]
    pub const fn with_tiff_predictor(mut self, predictor: TiffPredictor) -> Self {
        self.tiff_predictor = Some(predictor);
        self
    }

    /// Builder: enable or disable TIFF pyramid output.
    #[must_use]
    pub const fn with_pyramid(mut self, pyramid: bool) -> Self {
        self.pyramid = Some(pyramid);
        self
    }

    /// Builder: set encoder effort / speed tradeoff.
    #[must_use]
    pub const fn with_effort(mut self, effort: u8) -> Self {
        self.effort = Some(effort);
        self
    }

    /// Builder: set WebP method / effort.
    #[must_use]
    pub const fn with_method(mut self, method: u8) -> Self {
        self.method = Some(method);
        self
    }

    /// Builder: set HEIF-family compression.
    #[must_use]
    pub const fn with_heif_compression(mut self, compression: HeifCompression) -> Self {
        self.heif_compression = Some(compression);
        self
    }

    /// Builder: set HEIF/AVIF chroma subsampling.
    #[must_use]
    pub const fn with_heif_subsampling(mut self, subsampling: HeifSubsampling) -> Self {
        self.heif_subsampling = Some(subsampling);
        self
    }

    /// Builder: set HEIF/AVIF bit depth.
    #[must_use]
    pub const fn with_heif_bit_depth(mut self, bit_depth: HeifBitDepth) -> Self {
        self.heif_bit_depth = Some(bit_depth);
        self
    }

    /// Builder: set WebP near-lossless preprocessing level (0–100).
    #[must_use]
    pub const fn with_near_lossless(mut self, level: u8) -> Self {
        self.near_lossless = Some(level);
        self
    }

    /// Builder: preserve exact RGB under transparent areas in WebP.
    #[must_use]
    pub const fn with_exact_alpha(mut self, enabled: bool) -> Self {
        self.exact_alpha = Some(enabled);
        self
    }

    /// Builder: enable WebP smart subsampling.
    #[must_use]
    pub const fn with_smart_subsample(mut self, enabled: bool) -> Self {
        self.smart_subsample = Some(enabled);
        self
    }

    /// Builder: set RAW output byte order.
    #[must_use]
    pub const fn with_raw_endianness(mut self, endianness: RawEndianness) -> Self {
        self.raw_endianness = Some(endianness);
        self
    }
}

impl Default for SaveOptions {
    fn default() -> Self {
        Self::default_options()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::*;

    #[test]
    fn load_options_default_is_no_shrink() {
        let opts = LoadOptions::default();
        assert!(opts.shrink_factor.is_none());
        assert!(opts.max_dimension.is_none());
        assert!(!opts.no_rotate);
        assert!(opts.decoder_threads.is_none());
        assert!(opts.page.is_none());
        assert!(opts.n.is_none());
        assert!(opts.raw_width.is_none());
        assert!(opts.raw_endianness.is_none());
    }

    #[test]
    fn load_options_builder_chain() {
        let opts = LoadOptions::default()
            .with_shrink(NonZeroU8::new(4).unwrap())
            .with_decoder_threads(NonZeroUsize::new(4).unwrap())
            .with_max_dimension(1920)
            .no_rotate()
            .with_page(2)
            .with_n(-1)
            .with_raw_layout(640, 480, 3)
            .with_raw_format(BandFormatId::U16)
            .with_raw_endianness(RawEndianness::Big);
        assert_eq!(opts.shrink_factor, NonZeroU8::new(4));
        assert_eq!(opts.decoder_threads, NonZeroUsize::new(4));
        assert_eq!(opts.max_dimension, Some(1920));
        assert!(opts.no_rotate);
        assert_eq!(opts.page, Some(2));
        assert_eq!(opts.n, Some(-1));
        assert_eq!(opts.raw_width, Some(640));
        assert_eq!(opts.raw_height, Some(480));
        assert_eq!(opts.raw_bands, Some(3));
        assert_eq!(opts.raw_format, Some(BandFormatId::U16));
        assert_eq!(opts.raw_endianness, Some(RawEndianness::Big));
    }

    #[test]
    fn save_options_default_all_none() {
        let opts = SaveOptions::default();
        assert!(opts.quality.is_none());
        assert!(opts.lossless.is_none());
        assert!(opts.compression_level.is_none());
        assert!(opts.interlace.is_none());
        assert!(opts.restart_interval.is_none());
        assert!(opts.jpeg_subsampling.is_none());
        assert!(opts.png_filter.is_none());
        assert!(opts.dither.is_none());
        assert!(opts.colors.is_none());
        assert!(opts.strip_metadata.is_none());
        assert!(opts.tile_width.is_none());
        assert!(opts.tile_height.is_none());
        assert!(opts.tiff_compression.is_none());
        assert!(opts.tiff_predictor.is_none());
        assert!(opts.pyramid.is_none());
        assert!(opts.effort.is_none());
        assert!(opts.method.is_none());
        assert!(opts.heif_compression.is_none());
        assert!(opts.heif_subsampling.is_none());
        assert!(opts.heif_bit_depth.is_none());
        assert!(opts.raw_endianness.is_none());
    }

    #[test]
    fn save_options_builder_chain() {
        let opts = SaveOptions::default()
            .with_quality(85)
            .with_compression_level(6)
            .with_interlace(true)
            .with_restart_interval(32)
            .with_jpeg_subsampling(JpegSubsampling::Off)
            .with_png_filter(PngFilterStrategy::Paeth)
            .with_dither(false)
            .with_colors(32)
            .strip_metadata()
            .with_tile_width(128)
            .with_tile_height(64)
            .with_tiff_compression(TiffCompression::Deflate)
            .with_tiff_predictor(TiffPredictor::Horizontal)
            .with_pyramid(true)
            .with_effort(8)
            .with_method(4)
            .with_heif_compression(HeifCompression::Av1)
            .with_heif_subsampling(HeifSubsampling::Subsample422)
            .with_heif_bit_depth(HeifBitDepth::Twelve)
            .with_raw_endianness(RawEndianness::Little);
        assert_eq!(opts.quality, Some(85));
        assert_eq!(opts.compression_level, Some(6));
        assert_eq!(opts.interlace, Some(true));
        assert_eq!(opts.restart_interval, Some(32));
        assert_eq!(opts.jpeg_subsampling, Some(JpegSubsampling::Off));
        assert_eq!(opts.png_filter, Some(PngFilterStrategy::Paeth));
        assert_eq!(opts.dither, Some(false));
        assert_eq!(opts.colors, Some(32));
        assert_eq!(opts.strip_metadata, Some(true));
        assert_eq!(opts.tile_width, Some(128));
        assert_eq!(opts.tile_height, Some(64));
        assert_eq!(opts.tiff_compression, Some(TiffCompression::Deflate));
        assert_eq!(opts.tiff_predictor, Some(TiffPredictor::Horizontal));
        assert_eq!(opts.pyramid, Some(true));
        assert_eq!(opts.effort, Some(8));
        assert_eq!(opts.method, Some(4));
        assert_eq!(opts.heif_compression, Some(HeifCompression::Av1));
        assert_eq!(opts.heif_subsampling, Some(HeifSubsampling::Subsample422));
        assert_eq!(opts.heif_bit_depth, Some(HeifBitDepth::Twelve));
        assert_eq!(opts.raw_endianness, Some(RawEndianness::Little));
    }
}

use crate::{colorspace::ColorspaceId, format::BandFormatId, kernel::InterpolationKernel};
use thiserror::Error;

/// Top-level error type for viprs domain and pipeline operations.
///
/// This consolidates typed module errors so callers can handle image failures without losing the
/// original domain context.
///
/// # Examples
/// ```rust
/// # use viprs_core::error::{BooleanError, ViprsError};
/// let err: ViprsError = BooleanError::ConstLengthMismatch { len: 2, bands: 3 }.into();
/// assert!(matches!(err, ViprsError::Boolean(_)));
/// ```
#[derive(Debug, Error)]
pub enum ViprsError {
    #[error("pipeline execution cancelled")]
    /// Returned when pipeline execution is cancelled.
    Cancelled,
    #[error("pipeline build error: {0}")]
    /// Wraps `BuildError` when that error bubbles up through `ViprsError`.
    Build(BuildError),
    #[error("boolean error: {0}")]
    /// Returned when `Boolean` applies.
    Boolean(#[from] BooleanError),
    #[error("I/O error: {0}")]
    /// Returned when `Io` applies.
    Io(#[from] std::io::Error),
    #[error("draw error: {0}")]
    /// Returned when `Draw` applies.
    Draw(#[from] DrawError),
    #[error("frequency filter error: {0}")]
    /// Returned when `Freqfilt` applies.
    Freqfilt(#[from] FreqfiltError),
    #[error("convolution error: {0}")]
    /// Returned when `Convolution` applies.
    Convolution(#[from] ConvolutionError),
    #[error("composite error: {0}")]
    /// Returned when `Composite` applies.
    Composite(#[from] CompositeError),
    #[error("hough error: {0}")]
    /// Returned when `Hough` applies.
    Hough(#[from] HoughError),
    #[error("mosaicing error: {0}")]
    /// Returned when `Mosaicing` applies.
    Mosaicing(#[from] MosaicingError),
    #[error("source error: {0}")]
    /// Returned when `Source` applies.
    Source(#[from] SourceError),
    #[error("text error: {0}")]
    /// Returned when `Text` applies.
    Text(#[from] TextError),
    #[error("openexr error: {0}")]
    /// Returned when `Exr` applies.
    Exr(#[from] ExrCodecError),
    #[error("openslide error: {0}")]
    /// Returned when `OpenSlide` applies.
    OpenSlide(#[from] OpenSlideCodecError),
    #[error("codec error: {0}")]
    /// Wraps `String` when that error bubbles up through `ViprsError`.
    Codec(String),
    #[error("scheduler error: {0}")]
    /// Wraps `String` when that error bubbles up through `ViprsError`.
    Scheduler(String),
    #[error("scheduler contract error: {0}")]
    /// Returned when `SchedulerContract` applies.
    SchedulerContract(#[from] SchedulerContractError),
    #[error(
        "image too large: {width}x{height}x{bands} would require {bytes} bytes, exceeding the {limit_bytes}-byte limit ({details})"
    )]
    /// Returned when `ImageTooLarge` applies.
    ImageTooLarge {
        /// Width associated with this item.
        width: u32,
        /// Height associated with this item.
        height: u32,
        /// Number of bands associated with this item.
        bands: u32,
        /// Total byte count associated with this condition.
        bytes: u128,
        /// Maximum allowed byte count for this condition.
        limit_bytes: u128,
        /// Additional details describing this condition.
        details: &'static str,
    },
    #[error("{feature} is not implemented: {details}")]
    /// Returned when `Unimplemented` applies.
    Unimplemented {
        /// Feature name associated with this condition.
        feature: &'static str,
        /// Additional details describing this condition.
        details: &'static str,
    },
    #[error("region out of bounds: requested {requested:?}, image is {width}x{height}")]
    /// Returned when `RegionOutOfBounds` applies.
    RegionOutOfBounds {
        /// Requested value associated with this condition.
        requested: String,
        /// Width associated with this item.
        width: u32,
        /// Height associated with this item.
        height: u32,
    },
    #[error(
        "degenerate affine transform {matrix:?} produces {output_width}x{output_height}: {reason}"
    )]
    /// Returned when `DegenerateAffineTransform` applies.
    DegenerateAffineTransform {
        /// Matrix associated with this condition.
        matrix: [f64; 4],
        /// Output width associated with this condition.
        output_width: u32,
        /// Output height associated with this condition.
        output_height: u32,
        /// Explanation of why this condition occurred.
        reason: &'static str,
    },
    #[error("invalid recomb matrix {rows}x{cols} for input_bands {input_bands:?}: {reason}")]
    /// Returned when an `InvalidRecombMatrix` condition is detected.
    InvalidRecombMatrix {
        /// Number of rows associated with this configuration.
        rows: usize,
        /// Number of columns associated with this configuration.
        cols: usize,
        /// Input band count associated with this condition.
        input_bands: Option<u32>,
        /// Explanation of why this condition occurred.
        reason: &'static str,
    },
    #[error("invalid image: {0}")]
    /// Returned when an `InvalidImage` condition is detected.
    InvalidImage(&'static str),
}

impl From<BuildError> for ViprsError {
    fn from(error: BuildError) -> Self {
        match error {
            BuildError::InvalidImage { reason } => Self::InvalidImage(reason),
            other => Self::Build(other),
        }
    }
}

/// Errors produced while decoding or encoding `OpenEXR` content.
///
/// These errors describe EXR-specific structural mismatches that cannot be expressed as generic
/// codec failures.
///
/// # Examples
/// ```rust
/// # use viprs_core::error::ExrCodecError;
/// let err = ExrCodecError::NoLayers;
/// assert_eq!(err.to_string(), "file contains no layers");
/// ```
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ExrCodecError {
    #[error("{0}")]
    /// Wraps `String` when that error bubbles up through `ExrCodecError`.
    Backend(String),
    #[error("file contains no layers")]
    /// Returned when `NoLayers` applies.
    NoLayers,
    #[error(
        "multipart block references layer {requested_layer}, but selected range starts at {selected_start} and spans {selected_count} layer(s) across {header_count} header(s)"
    )]
    /// Returned when an `InvalidMultipartBlockReference` condition is detected.
    InvalidMultipartBlockReference {
        /// Requested layer index associated with this condition.
        requested_layer: usize,
        /// First selected layer index for the multipart range.
        selected_start: usize,
        /// Number of selected layers in the multipart range.
        selected_count: usize,
        /// Number of headers available in the source file.
        header_count: usize,
    },
    #[error("requested layer {requested}, but file only has {total_layers} layer(s)")]
    /// Returned when `RequestedLayerOutOfRange` applies.
    RequestedLayerOutOfRange {
        /// Requested value associated with this condition.
        requested: usize,
        /// Total number of layers available in the source file.
        total_layers: usize,
    },
    #[error("n must be positive or -1, got {value}")]
    /// Returned when an `InvalidLayerCount` condition is detected.
    InvalidLayerCount {
        /// Requested layer count.
        value: i32,
    },
    #[error("unsupported format {format:?} — only F32 is supported")]
    /// Returned when an `UnsupportedFormat` condition is detected.
    UnsupportedFormat {
        /// Encountered sample format.
        format: BandFormatId,
    },
    #[error("layer width exceeds u32")]
    /// Returned when `LayerWidthExceedsU32` applies.
    LayerWidthExceedsU32,
    #[error("layer height exceeds u32")]
    /// Returned when `LayerHeightExceedsU32` applies.
    LayerHeightExceedsU32,
    #[error("band count exceeds u32")]
    /// Returned when `BandCountExceedsU32` applies.
    BandCountExceedsU32,
    #[error("image dimensions exceed usize")]
    /// Returned when `DimensionsExceedUsize` applies.
    DimensionsExceedUsize,
    #[error("channel '{channel}' uses subsampling {x},{y} which is not supported")]
    /// Returned when an `UnsupportedSubsampling` condition is detected.
    UnsupportedSubsampling {
        /// Name of the subsampled channel.
        channel: String,
        /// Horizontal subsampling factor.
        x: usize,
        /// Vertical subsampling factor.
        y: usize,
    },
    #[error("channel '{channel}' length {len} does not match layer pixel count {pixel_count}")]
    /// Returned when `ChannelLengthMismatch` applies.
    ChannelLengthMismatch {
        /// Channel name associated with this condition.
        channel: String,
        /// Observed length associated with this condition.
        len: usize,
        /// Expected pixel count associated with this condition.
        pixel_count: usize,
    },
    #[error("cast error: {details}")]
    /// Returned when `CastError` applies.
    CastError {
        /// Description of the cast failure.
        details: String,
    },
    #[error("no decodable layers selected")]
    /// Returned when `NoDecodableLayersSelected` applies.
    NoDecodableLayersSelected,
}

/// Errors produced while validating `OpenSlide` pyramids and metadata.
///
/// These errors help slide decoders reject malformed level geometry before allocating buffers.
///
/// # Examples
/// ```rust
/// # use viprs_core::error::OpenSlideCodecError;
/// let err = OpenSlideCodecError::ZeroSizedLevel { level: 0, width: 0, height: 10 };
/// assert!(err.to_string().contains("invalid dimensions"));
/// ```
#[derive(Debug, Error, PartialEq)]
pub enum OpenSlideCodecError {
    #[error("level {level} has invalid dimensions {width}x{height}")]
    /// Returned when `ZeroSizedLevel` applies.
    ZeroSizedLevel {
        /// Pyramid level index.
        level: u32,
        /// Reported level width.
        width: u32,
        /// Reported level height.
        height: u32,
    },
    #[error("level {level} has invalid downsample factor {downsample}")]
    /// Returned when an `InvalidLevelDownsample` condition is detected.
    InvalidLevelDownsample {
        /// Pyramid level index.
        level: u32,
        /// Reported level downsample factor.
        downsample: f64,
    },
    #[error("{axis} has invalid microns-per-pixel value {microns_per_pixel}")]
    /// Returned when an `InvalidMicronsPerPixel` condition is detected.
    InvalidMicronsPerPixel {
        /// Axis name associated with this condition.
        axis: &'static str,
        /// Microns-per-pixel value associated with this condition.
        microns_per_pixel: f64,
    },
}

/// Errors raised when a scheduler violates internal compiled-pipeline invariants.
///
/// These errors turn scheduler bookkeeping bugs into typed failures instead of silent corruption.
///
/// # Examples
/// ```rust
/// # use viprs_core::error::SchedulerContractError;
/// let err = SchedulerContractError::MissingTransformState { node: 3 };
/// assert!(err.to_string().contains("missing operation state"));
/// ```
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SchedulerContractError {
    #[error("transform node {node} missing operation state in scheduler pool")]
    /// Returned when `MissingTransformState` applies.
    MissingTransformState {
        /// Transform node index missing its pooled state.
        node: usize,
    },
}

/// Errors raised by in-place draw operations.
///
/// Draw ops use this enum to report invalid overlay shapes and band mismatches before mutating
/// target tiles.
///
/// # Examples
/// ```rust
/// # use viprs_core::error::DrawError;
/// let err = DrawError::EmptyColor;
/// assert_eq!(err.to_string(), "draw colour must contain at least one band");
/// ```
#[derive(Debug, Error, PartialEq, Eq)]
pub enum DrawError {
    #[error("draw colour must contain at least one band")]
    /// Returned when `EmptyColor` applies.
    EmptyColor,
    #[error("draw band count must be greater than zero, got {bands}")]
    /// Returned when an `InvalidBandCount` condition is detected.
    InvalidBandCount {
        /// Requested draw band count.
        bands: u32,
    },
    #[error(
        "draw overlay bands must be 1 or match image bands, got overlay {overlay_bands}, image {image_bands}"
    )]
    /// Returned when `BandCountMismatch` applies.
    BandCountMismatch {
        /// Number of bands present in the overlay input.
        overlay_bands: u32,
        /// Number of bands present in the destination image.
        image_bands: u32,
    },
    #[error("draw buffer length {len} does not match {width}x{height}x{bands}={expected}")]
    /// Returned when `BufferLengthMismatch` applies.
    BufferLengthMismatch {
        /// Observed length associated with this condition.
        len: usize,
        /// Expected value associated with this condition.
        expected: usize,
        /// Width associated with this item.
        width: u32,
        /// Height associated with this item.
        height: u32,
        /// Number of bands associated with this item.
        bands: usize,
    },
    #[error("draw buffer dimensions overflow usize: {width}x{height}x{bands}")]
    /// Returned when `BufferDimensionsOverflow` applies.
    BufferDimensionsOverflow {
        /// Width associated with this item.
        width: u32,
        /// Height associated with this item.
        height: u32,
        /// Number of bands associated with this item.
        bands: usize,
    },
}

/// Errors raised by multi-image compositing operations.
///
/// This keeps Porter-Duff style validation separate from generic pipeline construction errors.
///
/// # Examples
/// ```rust
/// # use viprs_core::error::CompositeError;
/// let err = CompositeError::NonSeparableModeRequiresRgba { mode: "overlay", bands: 3 };
/// assert!(err.to_string().contains("requires exactly 4-band RGBA input"));
/// ```
#[derive(Debug, Error, PartialEq, Eq)]
pub enum CompositeError {
    #[error(
        "non-separable composite mode '{mode}' requires exactly 4-band RGBA input, got {bands} band(s)"
    )]
    /// Returned when `NonSeparableModeRequiresRgba` applies.
    NonSeparableModeRequiresRgba {
        /// Composite mode name.
        mode: &'static str,
        /// Input band count.
        bands: u32,
    },
}

/// Errors raised by boolean point operations.
///
/// These errors capture invalid constant-array shapes before a boolean kernel runs.
///
/// # Examples
/// ```rust
/// # use viprs_core::error::BooleanError;
/// let err = BooleanError::ConstLengthMismatch { len: 2, bands: 3 };
/// assert!(err.to_string().contains("must have 1 or 3 elements"));
/// ```
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BooleanError {
    #[error(
        "boolean const array must have 1 or {bands} elements for a {bands}-band image, got {len}"
    )]
    /// Returned when `ConstLengthMismatch` applies.
    ConstLengthMismatch {
        /// Number of provided constants.
        len: usize,
        /// Expected image band count.
        bands: u32,
    },
}

/// Errors raised by FFT-based frequency filtering operations.
///
/// Frequency-domain ops use this enum to report band-layout and buffer-contract violations.
///
/// # Examples
/// ```rust
/// # use viprs_core::error::FreqfiltError;
/// let err = FreqfiltError::FwfftBands { bands: 2 };
/// assert!(err.to_string().contains("single-band real image"));
/// ```
#[derive(Debug, Error, PartialEq, Eq)]
pub enum FreqfiltError {
    #[error("fwfft expects a single-band real image, got {bands} bands")]
    /// Returned when `FwfftBands` applies.
    FwfftBands {
        /// Observed input band count.
        bands: u32,
    },
    #[error("invfft expects a 2-band complex image laid out as [real, imag], got {bands} bands")]
    /// Returned when `InvfftBands` applies.
    InvfftBands {
        /// Observed input band count.
        bands: u32,
    },
    #[error(
        "fwfft requires a full-image 1-band input and a full-image 2-band output, got input {input_bands} bands at {input_region:?} and output {output_bands} bands at {output_region:?}"
    )]
    /// Returned when `FwfftContract` applies.
    FwfftContract {
        /// Input band count associated with this condition.
        input_bands: u32,
        /// Output bands associated with this error condition.
        output_bands: u32,
        /// Input region associated with this error condition.
        input_region: crate::image::Region,
        /// Output region associated with this error condition.
        output_region: crate::image::Region,
    },
    #[error(
        "invfft requires a full-image 2-band input and a full-image 1-band output, got input {input_bands} bands at {input_region:?} and output {output_bands} bands at {output_region:?}"
    )]
    /// Returned when `InvfftContract` applies.
    InvfftContract {
        /// Input band count associated with this condition.
        input_bands: u32,
        /// Output bands associated with this error condition.
        output_bands: u32,
        /// Input region associated with this error condition.
        input_region: crate::image::Region,
        /// Output region associated with this error condition.
        output_region: crate::image::Region,
    },
    #[error("fft dimensions overflow usize: {width}x{height}")]
    /// Returned when `DimensionsOverflow` applies.
    DimensionsOverflow {
        /// Requested FFT width.
        width: u32,
        /// Requested FFT height.
        height: u32,
    },
    #[error("{op}: expected exactly {expected} input slices, got {actual}")]
    /// Returned when `MultiInputArity` applies.
    MultiInputArity {
        /// Op associated with this error condition.
        op: &'static str,
        /// Expected value associated with this condition.
        expected: usize,
        /// Actual associated with this error condition.
        actual: usize,
    },
    #[error("{op}: {buffer} byte length {actual} does not match expected {expected}")]
    /// Returned when `MultiInputBufferLength` applies.
    MultiInputBufferLength {
        /// Op associated with this error condition.
        op: &'static str,
        /// Buffer associated with this error condition.
        buffer: &'static str,
        /// Expected value associated with this condition.
        expected: usize,
        /// Actual associated with this error condition.
        actual: usize,
    },
    #[error("{op}: {buffer} bytes are not a valid sample cast")]
    /// Returned when `MultiInputBufferCast` applies.
    MultiInputBufferCast {
        /// Op associated with this error condition.
        op: &'static str,
        /// Buffer associated with this error condition.
        buffer: &'static str,
    },
}

/// Errors raised while validating convolution kernels and parameters.
///
/// Convolution builders use this enum to reject malformed kernels before entering the pixel path.
///
/// # Examples
/// ```rust
/// # use viprs_core::error::ConvolutionError;
/// let err = ConvolutionError::EmptyKernel { op: "conv" };
/// assert!(err.to_string().contains("kernel must not be empty"));
/// ```
#[derive(Debug, Error, PartialEq)]
pub enum ConvolutionError {
    #[error("{op}: kernel must not be empty")]
    /// Returned when `EmptyKernel` applies.
    EmptyKernel {
        /// Operation reporting the empty kernel.
        op: &'static str,
    },
    #[error("{op}: kernel rows must not be empty")]
    /// Returned when `EmptyKernelRow` applies.
    EmptyKernelRow {
        /// Operation reporting the empty kernel row.
        op: &'static str,
    },
    #[error("{op}: kernel dimensions must be odd, got {width}x{height}")]
    /// Returned when `EvenKernelDimensions` applies.
    EvenKernelDimensions {
        /// Op associated with this error condition.
        op: &'static str,
        /// Width associated with this item.
        width: usize,
        /// Height associated with this item.
        height: usize,
    },
    #[error("{op}: kernel must be rectangular, row {row} has width {width}, expected {expected}")]
    /// Returned when `RaggedKernel` applies.
    RaggedKernel {
        /// Op associated with this error condition.
        op: &'static str,
        /// Row associated with this error condition.
        row: usize,
        /// Width associated with this item.
        width: usize,
        /// Expected value associated with this condition.
        expected: usize,
    },
    #[error("{op}: kernel length must be odd, got {len}")]
    /// Returned when `EvenKernelLength` applies.
    EvenKernelLength {
        /// Operation reporting the invalid kernel length.
        op: &'static str,
        /// Observed kernel length.
        len: usize,
    },
    #[error("{op}: kernel coefficient at ({x}, {y}) must be finite, got {value}")]
    /// Returned when `NonFiniteCoefficient` applies.
    NonFiniteCoefficient {
        /// Op associated with this error condition.
        op: &'static str,
        /// Horizontal factor associated with this condition.
        x: usize,
        /// Vertical factor associated with this condition.
        y: usize,
        /// Value associated with this item.
        value: f64,
    },
    #[error("{op}: scale must be finite and non-zero, got {scale}")]
    /// Returned when an `InvalidScale` condition is detected.
    InvalidScale {
        /// Operation reporting the invalid scale.
        op: &'static str,
        /// Non-finite or zero scale value.
        scale: f64,
    },
    #[error("{op}: offset must be finite, got {offset}")]
    /// Returned when an `InvalidOffset` condition is detected.
    InvalidOffset {
        /// Operation reporting the invalid offset.
        op: &'static str,
        /// Non-finite offset value.
        offset: f64,
    },
}

/// Errors raised by Hough-transform operations.
///
/// This keeps parameter validation for circle searches typed and domain specific.
///
/// # Examples
/// ```rust
/// # use viprs_core::error::HoughError;
/// let err = HoughError::ZeroScale;
/// assert!(err.to_string().contains("greater than zero"));
/// ```
#[derive(Debug, Error, PartialEq, Eq)]
pub enum HoughError {
    #[error("hough circle scale must be greater than zero")]
    /// Returned when `ZeroScale` applies.
    ZeroScale,
    #[error(
        "hough circle max radius must be greater than min radius, got min {min_radius}, max {max_radius}"
    )]
    /// Returned when an `InvalidRadiusRange` condition is detected.
    InvalidRadiusRange {
        /// Minimum requested circle radius.
        min_radius: u32,
        /// Maximum requested circle radius.
        max_radius: u32,
    },
}

/// Errors raised while aligning multiple images into a mosaic.
///
/// Mosaicing code uses these variants for tie-point validation and affine-solve failures.
///
/// # Examples
/// ```rust
/// # use viprs_core::error::MosaicingError;
/// let err = MosaicingError::InvalidHypothesisCount;
/// assert!(err.to_string().contains("at least 1"));
/// ```
#[derive(Debug, Error)]
pub enum MosaicingError {
    #[error("need at least {minimum} tie-point pairs, got {actual}")]
    /// Returned when `NotEnoughTiePoints` applies.
    NotEnoughTiePoints {
        /// Minimum required tie-point count.
        minimum: usize,
        /// Observed tie-point count.
        actual: usize,
    },
    #[error("tie-point configuration is degenerate")]
    /// Returned when `DegenerateTiePointConfiguration` applies.
    DegenerateTiePointConfiguration,
    #[error("affine transform is singular")]
    /// Returned when `SingularAffineTransform` applies.
    SingularAffineTransform,
    #[error(
        "tie-point search requires matching band counts, got reference {reference_bands} and secondary {secondary_bands}"
    )]
    /// Returned when `BandCountMismatch` applies.
    BandCountMismatch {
        /// Reference bands associated with this error condition.
        reference_bands: u32,
        /// Secondary bands associated with this error condition.
        secondary_bands: u32,
    },
    #[error(
        "overlap too small for tie-point search: got {width}x{height}, need at least {minimum_width}x{minimum_height}"
    )]
    /// Returned when `OverlapTooSmall` applies.
    OverlapTooSmall {
        /// Width associated with this item.
        width: u32,
        /// Height associated with this item.
        height: u32,
        /// Minimum width associated with this error condition.
        minimum_width: u32,
        /// Minimum height associated with this error condition.
        minimum_height: u32,
    },
    #[error("tie-point search could not find a valid overlap window")]
    /// Returned when `NoValidOverlapWindow` applies.
    NoValidOverlapWindow,
    #[error("residual threshold must be finite and > 0, got {threshold}")]
    /// Returned when an `InvalidResidualThreshold` condition is detected.
    InvalidResidualThreshold {
        /// Requested residual threshold.
        threshold: f64,
    },
    #[error("hypothesis count must be at least 1")]
    /// Returned when an `InvalidHypothesisCount` condition is detected.
    InvalidHypothesisCount,
}

/// Errors raised by source adapters and tile caches.
///
/// These variants surface infrastructure failures that happen while materializing source tiles.
///
/// # Examples
/// ```rust
/// # use viprs_core::error::SourceError;
/// let err = SourceError::TileCacheMutexPoisoned { phase: "read", x: 0, y: 0, width: 1, height: 1 };
/// assert!(err.to_string().contains("tile cache mutex poisoned"));
/// ```
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SourceError {
    #[error("tile cache mutex poisoned during {phase} for region ({x}, {y}, {width}x{height})")]
    /// Returned when `TileCacheMutexPoisoned` applies.
    TileCacheMutexPoisoned {
        /// Phase associated with this error condition.
        phase: &'static str,
        /// Horizontal factor associated with this condition.
        x: i32,
        /// Vertical factor associated with this condition.
        y: i32,
        /// Width associated with this item.
        width: u32,
        /// Height associated with this item.
        height: u32,
    },
}

/// Errors raised while rasterizing synthetic text sources.
#[derive(Debug, Error, PartialEq)]
pub enum TextError {
    #[error("text source requires a non-empty string")]
    /// Returned when `EmptyText` applies.
    EmptyText,
    #[error("text source currently supports single-line input only")]
    /// Returned when `MultilineUnsupported` applies.
    MultilineUnsupported,
    #[error("text source font size must be finite and > 0, got {font_size}")]
    /// Returned when an `InvalidFontSize` condition is detected.
    InvalidFontSize {
        /// Requested font size.
        font_size: f32,
    },
    #[error("text source could not load font '{path}': {reason}")]
    /// Returned when `FontLoad` applies.
    FontLoad {
        /// Font path that failed to load.
        path: String,
        /// Loader-provided failure reason.
        reason: String,
    },
    #[error("text source could not locate a default font; set font_path explicitly")]
    /// Returned when `DefaultFontUnavailable` applies.
    DefaultFontUnavailable,
}

/// Errors raised while assembling or validating a pipeline graph.
///
/// Builders use this enum to reject invalid graph topology, format mismatches, and unsupported
/// operation parameters before execution starts.
///
/// # Examples
/// ```rust
/// # use viprs_core::{error::BuildError, format::BandFormatId};
/// let err = BuildError::FormatMismatch {
///     produced: BandFormatId::U8,
///     expected: BandFormatId::F32,
///     hint: "insert a Cast operation",
/// };
/// assert!(err.to_string().contains("format mismatch"));
/// ```
#[derive(Debug, Error)]
pub enum BuildError {
    #[error(
        "format mismatch: upstream produces {produced:?} but downstream expects {expected:?}; hint: {hint}"
    )]
    /// Returned when `FormatMismatch` applies.
    FormatMismatch {
        /// Produced associated with this error condition.
        produced: BandFormatId,
        /// Expected value associated with this condition.
        expected: BandFormatId,
        /// Hint associated with this error condition.
        hint: &'static str,
    },
    #[error("pipeline has no source node")]
    /// Returned when `NoSource` applies.
    NoSource,
    #[error("source hint error while configuring {context}: {message}")]
    /// Returned when `SourceHint` applies.
    SourceHint {
        /// Context associated with this error condition.
        context: &'static str,
        /// Message associated with this error condition.
        message: String,
    },
    #[error("pipeline has no sink node \u{2014} add at least one operation")]
    /// Returned when `NoNodes` applies.
    NoNodes,
    #[error("pipeline contains a cycle")]
    /// Returned when `Cycle` applies.
    Cycle,
    #[error("invalid image: {reason}")]
    /// Returned when an `InvalidImage` condition is detected.
    InvalidImage {
        /// Explanation of why the image is invalid.
        reason: &'static str,
    },
    #[error(
        "pipeline sizing overflow: {width}x{height}x{bands} would require {bytes} bytes, exceeding the {limit_bytes}-byte limit ({details})"
    )]
    /// Returned when `ImageTooLarge` applies.
    ImageTooLarge {
        /// Width associated with this item.
        width: u32,
        /// Height associated with this item.
        height: u32,
        /// Number of bands associated with this item.
        bands: u32,
        /// Total byte count associated with this condition.
        bytes: u128,
        /// Maximum allowed byte count for this condition.
        limit_bytes: u128,
        /// Additional details describing this condition.
        details: &'static str,
    },
    #[error("node index {0} does not exist in the arena")]
    /// Wraps `usize` when that error bubbles up through `BuildError`.
    InvalidNodeIndex(usize),
    #[error("node {node} is a view node and cannot own an operation tile cache")]
    /// Returned when `CacheRequiresTransform` applies.
    CacheRequiresTransform {
        /// View node index that requested a tile cache.
        node: usize,
    },
    #[error("fused op chain must contain at least 2 ops, got {op_count}")]
    /// Returned when an `InvalidFusionChain` condition is detected.
    InvalidFusionChain {
        /// Number of fused operations provided.
        op_count: usize,
    },
    #[error("node {node} has no input slot {slot}; slot count is {slot_count}")]
    /// Returned when an `InvalidInputSlot` condition is detected.
    InvalidInputSlot {
        /// Pipeline node index associated with this condition.
        node: usize,
        /// Slot associated with this error condition.
        slot: u8,
        /// Slot count associated with this error condition.
        slot_count: usize,
    },
    #[error(
        "topological sort visited node {0} more than once — this is a bug in the sort algorithm"
    )]
    /// Wraps `usize` when that error bubbles up through `BuildError`.
    DuplicateNodeInTopoOrder(usize),
    #[error("operation tile cache was enabled but no max-byte budget was configured")]
    /// Returned when `CacheMissingCapacity` applies.
    CacheMissingCapacity,
    #[error("operation '{op}' does not support format {format:?}")]
    /// Returned when an `UnsupportedFormat` condition is detected.
    UnsupportedFormat {
        /// Op associated with this error condition.
        op: &'static str,
        /// Band format associated with this condition.
        format: BandFormatId,
    },
    #[error("operation '{op}' does not support kernel {kernel:?}: {reason}")]
    /// Returned when an `UnsupportedKernel` condition is detected.
    UnsupportedKernel {
        /// Op associated with this error condition.
        op: &'static str,
        /// Kernel associated with this error condition.
        kernel: InterpolationKernel,
        /// Explanation of why this condition occurred.
        reason: &'static str,
    },
    #[error("invalid resample kernel for {op}: {kernel:?} ({reason})")]
    /// Returned when an `InvalidKernel` condition is detected.
    InvalidKernel {
        /// Op associated with this error condition.
        op: &'static str,
        /// Kernel associated with this error condition.
        kernel: InterpolationKernel,
        /// Explanation of why this condition occurred.
        reason: &'static str,
    },
    #[error("invalid affine matrix {matrix:?}: {reason}")]
    /// Returned when an `InvalidAffineMatrix` condition is detected.
    InvalidAffineMatrix {
        /// Matrix associated with this condition.
        matrix: [f64; 4],
        /// Explanation of why this condition occurred.
        reason: &'static str,
    },
    #[error("invalid recomb matrix {rows}x{cols} for input_bands {input_bands}: {reason}")]
    /// Returned when an `InvalidRecombMatrix` condition is detected.
    InvalidRecombMatrix {
        /// Number of rows associated with this configuration.
        rows: usize,
        /// Number of columns associated with this configuration.
        cols: usize,
        /// Input band count associated with this condition.
        input_bands: u32,
        /// Explanation of why this condition occurred.
        reason: &'static str,
    },
    #[error(
        "degenerate affine transform {matrix:?} produces {output_width}x{output_height}: {reason}"
    )]
    /// Returned when `DegenerateAffineTransform` applies.
    DegenerateAffineTransform {
        /// Matrix associated with this condition.
        matrix: [f64; 4],
        /// Output width associated with this condition.
        output_width: u32,
        /// Output height associated with this condition.
        output_height: u32,
        /// Explanation of why this condition occurred.
        reason: &'static str,
    },
    #[error("invalid embed parameters: {message}")]
    /// Returned when an `InvalidEmbedParameters` condition is detected.
    InvalidEmbedParameters {
        /// Explanation of the invalid embed configuration.
        message: &'static str,
    },
    #[error("invalid thumbnail parameters: {message}")]
    /// Returned when an `InvalidThumbnailParameters` condition is detected.
    InvalidThumbnailParameters {
        /// Explanation of the invalid thumbnail configuration.
        message: &'static str,
    },
    #[error("invalid linear parameters: scale={scale}, offset={offset}")]
    /// Returned when an `InvalidLinearParameters` condition is detected.
    InvalidLinearParameters {
        /// Invalid linear scale factor.
        scale: f64,
        /// Invalid linear offset.
        offset: f64,
    },
    #[error(
        "operation '{op}' requires {expected} as input and {expected_output} as output, got input {input_bands} band(s) and output {output_bands} band(s)"
    )]
    /// Returned when an `InvalidOperationBands` condition is detected.
    InvalidOperationBands {
        /// Op associated with this error condition.
        op: &'static str,
        /// Input band count associated with this condition.
        input_bands: u32,
        /// Output bands associated with this error condition.
        output_bands: u32,
        /// Expected value associated with this condition.
        expected: &'static str,
        /// Expected output associated with this error condition.
        expected_output: &'static str,
    },
    #[error("invalid reduce parameters: h_factor={h_factor}, v_factor={v_factor} ({reason})")]
    /// Returned when an `InvalidReduceParameters` condition is detected.
    InvalidReduceParameters {
        /// H factor associated with this error condition.
        h_factor: f64,
        /// V factor associated with this error condition.
        v_factor: f64,
        /// Explanation of why this condition occurred.
        reason: &'static str,
    },
    #[error(
        "invalid extract_area parameters: crop ({x}, {y}, {width}x{height}) exceeds input {image_width}x{image_height}"
    )]
    /// Returned when an `InvalidExtractAreaParameters` condition is detected.
    InvalidExtractAreaParameters {
        /// Horizontal factor associated with this condition.
        x: u32,
        /// Vertical factor associated with this condition.
        y: u32,
        /// Width associated with this item.
        width: u32,
        /// Height associated with this item.
        height: u32,
        /// Image width associated with this error condition.
        image_width: u32,
        /// Image height associated with this error condition.
        image_height: u32,
    },
    /// `PipelinePlan::colourspace` was called when the current colorspace is
    /// `Unknown`. The caller must resolve the colorspace (e.g. with
    /// `with_colorspace`) before inserting a colour conversion.
    #[error(
        "colourspace conversion requires a known source colorspace; \
         current colorspace is Unknown — call with_colorspace() first"
    )]
    UnknownColorspace,
    /// `PipelinePlan::colourspace` was called with a `(from, to)` pair that has
    /// no registered `ColourConvert` implementation.
    #[error("no colour conversion registered for {from:?} → {to:?}")]
    UnsupportedColourConversion {
        /// From associated with this error condition.
        from: ColorspaceId,
        /// To associated with this error condition.
        to: ColorspaceId,
    },
    #[error(
        "invalid input for colour conversion {from:?} → {to:?}: expected {expected}, got {bands} band(s)"
    )]
    /// Returned when an `InvalidColourConversionInput` condition is detected.
    InvalidColourConversionInput {
        /// From associated with this error condition.
        from: ColorspaceId,
        /// To associated with this error condition.
        to: ColorspaceId,
        /// Number of bands associated with this item.
        bands: u32,
        /// Expected value associated with this condition.
        expected: &'static str,
    },
    /// A `CompiledNode` was constructed with an input buffer index that violates the
    /// monotone-assignment invariant required by the scheduler's split-borrow pattern.
    ///
    /// For Transform nodes: every `input_buf` must be strictly less than `output_buf`.
    /// For View nodes: every `input_buf` must equal `output_buf`.
    #[error(
        "CompiledNode buffer invariant violated: input_buf={input_buf} is not valid \
         relative to output_buf={output_buf}"
    )]
    InvalidBufferOrder {
        /// Input buffer index assigned to the node.
        input_buf: usize,
        /// Output buffer index assigned to the node.
        output_buf: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn build_error_format_mismatch_message() {
        let err = BuildError::FormatMismatch {
            produced: BandFormatId::U8,
            expected: BandFormatId::F32,
            hint: "insert a Cast operation",
        };
        let msg = err.to_string();
        assert!(msg.contains("U8"), "expected U8 in: {msg}");
        assert!(msg.contains("F32"), "expected F32 in: {msg}");
        assert!(
            msg.contains("insert a Cast operation"),
            "expected hint in: {msg}"
        );
    }

    #[test]
    fn build_error_invalid_extract_area_message() {
        let err = BuildError::InvalidExtractAreaParameters {
            x: 0,
            y: 0,
            width: 11,
            height: 8,
            image_width: 7,
            image_height: 5,
        };
        let msg = err.to_string();
        assert!(msg.contains("11x8"), "expected crop size in: {msg}");
        assert!(msg.contains("7x5"), "expected image size in: {msg}");
    }

    #[test]
    fn build_error_invalid_thumbnail_message() {
        let err = BuildError::InvalidThumbnailParameters {
            message: "band count must be greater than zero",
        };
        let msg = err.to_string();
        assert!(msg.contains("band count must be greater than zero"));
    }

    #[test]
    fn build_error_invalid_image_message() {
        let err = BuildError::InvalidImage {
            reason: "zero-band image",
        };
        let msg = err.to_string();
        assert!(msg.contains("invalid image"));
        assert!(msg.contains("zero-band image"));
    }

    #[test]
    fn viprs_error_from_build_error() {
        let build_err = BuildError::NoSource;
        let viprs_err: ViprsError = build_err.into();
        assert!(matches!(viprs_err, ViprsError::Build(_)));
    }

    #[test]
    fn viprs_error_from_invalid_image_build_error() {
        let build_err = BuildError::InvalidImage {
            reason: "zero-band image",
        };
        let viprs_err: ViprsError = build_err.into();
        assert!(matches!(
            viprs_err,
            ViprsError::InvalidImage("zero-band image")
        ));
    }

    #[test]
    fn viprs_error_from_io_error() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let viprs_err: ViprsError = io_err.into();
        assert!(matches!(viprs_err, ViprsError::Io(_)));
    }

    #[test]
    fn viprs_error_from_boolean_error() {
        let boolean_err = BooleanError::ConstLengthMismatch { len: 2, bands: 3 };
        let viprs_err: ViprsError = boolean_err.into();
        assert!(matches!(
            viprs_err,
            ViprsError::Boolean(BooleanError::ConstLengthMismatch { len: 2, bands: 3 })
        ));
    }

    #[test]
    fn viprs_error_from_draw_error() {
        let draw_err = DrawError::EmptyColor;
        let viprs_err: ViprsError = draw_err.into();
        assert!(matches!(viprs_err, ViprsError::Draw(DrawError::EmptyColor)));
    }

    #[test]
    fn viprs_error_from_freqfilt_error() {
        let freqfilt_err = FreqfiltError::FwfftBands { bands: 3 };
        let viprs_err: ViprsError = freqfilt_err.into();
        assert!(matches!(
            viprs_err,
            ViprsError::Freqfilt(FreqfiltError::FwfftBands { bands: 3 })
        ));
    }

    #[test]
    fn viprs_error_from_mosaicing_error() {
        let mosaicing_err = MosaicingError::SingularAffineTransform;
        let viprs_err: ViprsError = mosaicing_err.into();
        assert!(matches!(
            viprs_err,
            ViprsError::Mosaicing(MosaicingError::SingularAffineTransform)
        ));
    }

    #[test]
    fn viprs_error_from_source_error() {
        let source_err = SourceError::TileCacheMutexPoisoned {
            phase: "lookup",
            x: 1,
            y: 2,
            width: 16,
            height: 32,
        };
        let viprs_err: ViprsError = source_err.into();
        assert!(matches!(
            viprs_err,
            ViprsError::Source(SourceError::TileCacheMutexPoisoned {
                phase: "lookup",
                x: 1,
                y: 2,
                width: 16,
                height: 32
            })
        ));
    }

    #[test]
    fn viprs_error_scheduler_message() {
        let err = ViprsError::Scheduler("thread pool build failed".into());
        let msg = err.to_string();
        assert!(
            msg.contains("scheduler error"),
            "expected 'scheduler error' in: {msg}"
        );
        assert!(
            msg.contains("thread pool build failed"),
            "expected detail in: {msg}"
        );
    }

    #[test]
    fn viprs_error_unimplemented_message() {
        let err = ViprsError::Unimplemented {
            feature: "icc_transform",
            details: "requires Little CMS adapter wiring",
        };
        let msg = err.to_string();
        assert!(msg.contains("icc_transform"), "expected feature in: {msg}");
        assert!(
            msg.contains("requires Little CMS adapter wiring"),
            "expected detail in: {msg}"
        );
    }

    #[test]
    fn viprs_error_degenerate_affine_message() {
        let err = ViprsError::DegenerateAffineTransform {
            matrix: [0.0, 0.0, 0.0, 0.0],
            output_width: 7,
            output_height: 5,
            reason: "matrix determinant is singular",
        };
        let msg = err.to_string();
        assert!(msg.contains("degenerate affine transform"));
        assert!(msg.contains("7x5"));
        assert!(msg.contains("singular"));
    }
}

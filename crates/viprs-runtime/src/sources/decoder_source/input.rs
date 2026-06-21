use super::{
    Arc, BandFormat, DecoderBacking, Image, ImageDecoder, ImageMetadataProbe, LoadOptions, Path,
    PathBuf, Region, TileImageDecoder, ViprsError, eager_backing_shrink_factor,
    eager_backing_shrink_factor_from_path, fmt,
};

pub(super) type DecodeRegionFn<D> = for<'input> fn(
    &D,
    &DecoderInput<'input>,
    &LoadOptions,
    Region,
    &mut [u8],
) -> Result<(), ViprsError>;

pub(super) type ProbeInputFn<D> = for<'input> fn(
    &D,
    &DecoderInput<'input>,
    &LoadOptions,
) -> Result<ImageMetadataProbe, ViprsError>;

pub(super) fn probe_input_with<D>(
    decoder: &D,
    input: &DecoderInput<'_>,
    opts: &LoadOptions,
) -> Result<ImageMetadataProbe, ViprsError>
where
    D: TileImageDecoder,
{
    match input {
        DecoderInput::Borrowed(src) => decoder.probe_with_options(src, opts),
        DecoderInput::Shared(src) => decoder.probe_with_options(src, opts),
        DecoderInput::StablePath(path) => decoder.probe_path_with_options(path, opts),
    }
}

pub(super) fn decode_region_with<D, F>(
    decoder: &D,
    input: &DecoderInput<'_>,
    opts: &LoadOptions,
    region: Region,
    output: &mut [u8],
) -> Result<(), ViprsError>
where
    D: TileImageDecoder,
    F: BandFormat,
{
    match input {
        DecoderInput::Borrowed(src) => decoder.decode_region_into::<F>(src, opts, region, output),
        DecoderInput::Shared(src) => decoder.decode_region_into::<F>(src, opts, region, output),
        DecoderInput::StablePath(path) => {
            decoder.decode_region_from_path::<F>(path, opts, region, output)
        }
    }
}

/// Eagerly decode the full image from a streaming backing using the codec's
/// standard `decode_with_options` / `decode_path_with_options` path. This
/// produces a single in-memory raster with shrink-on-load applied, suitable
/// for promoting a streaming source to eager mode.
pub(super) fn streaming_eager_decode<D: ImageDecoder, F: BandFormat>(
    backing: &DecoderBacking<'_, D, F>,
    decoder: &D,
    opts: &LoadOptions,
) -> Result<Image<F>, ViprsError> {
    let DecoderBacking::Streaming { input, .. } = backing else {
        return Err(ViprsError::Codec(
            "streaming_eager_decode called on non-streaming backing".into(),
        ));
    };
    match input {
        DecoderInput::Borrowed(src) => decoder.decode_with_options::<F>(src, opts),
        DecoderInput::Shared(src) => decoder.decode_with_options::<F>(src, opts),
        DecoderInput::StablePath(path) => decoder.decode_path_with_options::<F>(path, opts),
    }
}

/// Compute the effective backing shrink factor after an eager decode from a
/// streaming input. Mirrors `eager_backing_shrink_factor` /
/// `eager_backing_shrink_factor_from_path` but dispatches on `DecoderInput`.
pub(super) fn streaming_backing_shrink_factor<D: ImageDecoder, F: BandFormat>(
    backing: &DecoderBacking<'_, D, F>,
    decoder: &D,
    requested_factor: u8,
    image: &Image<F>,
) -> u8 {
    if requested_factor <= 1 {
        return 1;
    }
    let DecoderBacking::Streaming { input, .. } = backing else {
        return 1;
    };
    match input {
        DecoderInput::Borrowed(src) => {
            eager_backing_shrink_factor(decoder, src, requested_factor, image)
        }
        DecoderInput::Shared(src) => {
            eager_backing_shrink_factor(decoder, src, requested_factor, image)
        }
        DecoderInput::StablePath(path) => {
            eager_backing_shrink_factor_from_path(decoder, path, requested_factor, image)
        }
    }
}

/// Compressed input retained by a streaming [`DecoderSource`](super::DecoderSource).
///
/// Borrowed input is the strictest memory path: the source keeps no encoded copy
/// and decodes each tile from the caller-owned bytes. Shared input is for sources
/// that must be `'static` (for example when inserted into a dynamic pipeline).
#[derive(Clone)]
pub enum DecoderInput<'a> {
    /// Caller-owned bytes borrowed for the lifetime of the source.
    Borrowed(&'a [u8]),
    /// Reference-counted encoded bytes owned by the source.
    Shared(Arc<[u8]>),
    /// Stable filesystem path that can be reopened for region decodes.
    StablePath(PathBuf),
}

impl<'a> DecoderInput<'a> {
    /// Wraps caller-owned encoded bytes without taking ownership.
    #[must_use]
    pub const fn borrowed(src: &'a [u8]) -> Self {
        Self::Borrowed(src)
    }

    /// Wraps reference-counted encoded bytes for `'static` pipeline storage.
    #[must_use]
    pub const fn shared(src: Arc<[u8]>) -> Self {
        Self::Shared(src)
    }

    /// Stores a stable path that region decoders may reopen on demand.
    #[must_use]
    pub const fn stable_path(path: PathBuf) -> Self {
        Self::StablePath(path)
    }

    /// Returns the encoded bytes when this input is memory-backed.
    #[must_use]
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Borrowed(src) => Some(src),
            Self::Shared(src) => Some(src),
            Self::StablePath(_) => None,
        }
    }

    /// Returns the stable path when this input is path-backed.
    #[must_use]
    pub fn stable_path_ref(&self) -> Option<&Path> {
        match self {
            Self::StablePath(path) => Some(path.as_path()),
            Self::Borrowed(_) | Self::Shared(_) => None,
        }
    }

    /// Returns the encoded byte length when it is known without I/O.
    #[must_use]
    pub fn len(&self) -> Option<usize> {
        self.as_bytes().map(<[u8]>::len)
    }

    /// Returns `true` when the encoded input is known to contain zero bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.as_bytes().is_some_and(<[u8]>::is_empty)
    }
}

impl fmt::Debug for DecoderInput<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecoderInput")
            .field(
                "kind",
                &match self {
                    Self::Borrowed(_) => "borrowed",
                    Self::Shared(_) => "shared",
                    Self::StablePath(_) => "stable-path",
                },
            )
            .field("len", &self.len())
            .field("path", &self.stable_path_ref())
            .finish()
    }
}

#[derive(Clone)]
pub(super) enum StableDecoderInput {
    Shared(Arc<[u8]>),
    Path(PathBuf),
}

impl StableDecoderInput {
    pub(super) fn decode_with_options<D: ImageDecoder, F: BandFormat>(
        &self,
        decoder: &D,
        opts: &LoadOptions,
    ) -> Result<Image<F>, ViprsError> {
        match self {
            Self::Shared(src) => decoder.decode_with_options::<F>(src, opts),
            Self::Path(path) => decoder.decode_path_with_options::<F>(path, opts),
        }
    }

    pub(super) fn backing_shrink_factor<D: ImageDecoder, F: BandFormat>(
        &self,
        decoder: &D,
        requested_factor: u8,
        image: &Image<F>,
    ) -> u8 {
        match self {
            Self::Shared(src) => eager_backing_shrink_factor(decoder, src, requested_factor, image),
            Self::Path(path) => {
                eager_backing_shrink_factor_from_path(decoder, path, requested_factor, image)
            }
        }
    }
}

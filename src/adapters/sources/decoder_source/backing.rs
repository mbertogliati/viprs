use super::{
    Arc, BandFormat, DecodeRegionFn, DecoderInput, Image, ImageDecoder, ImageMetadata, LoadOptions,
    OnceLock, Path, PhantomData, ProbeInputFn, StableDecoderInput, TileImageDecoder, ViprsError,
    decode_region_with, eager_backing_shrink_factor, eager_backing_shrink_factor_from_path,
    normalize_shrink_factor, normalize_streaming_options, probe_input_with,
    retains_stable_input_for_thumbnail,
};

pub(super) enum DecoderBacking<'a, D: ImageDecoder, F: BandFormat> {
    Eager(Image<F>),
    Deferred {
        width: u32,
        height: u32,
        bands: u32,
        metadata: ImageMetadata,
        image: OnceLock<Result<Arc<Image<F>>, String>>,
    },
    Streaming {
        input: DecoderInput<'a>,
        width: u32,
        height: u32,
        bands: u32,
        metadata: ImageMetadata,
        probe_input: ProbeInputFn<D>,
        decode_region: DecodeRegionFn<D>,
    },
}

// ── Access mode markers ────────────────────────────────────────────────────────

/// Marker: the decoder produces a fully-buffered image that supports random tile access.
///
/// This is the default.  All decoders that fully materialise the image in memory
/// (JPEG, PNG, WebP, …) are random-access after construction.
pub struct RandomAccess;

/// Marker: the decoder streams data sequentially and cannot seek backwards.
///
/// Reserved for streaming formats (e.g., progressive JPEG consumed as a network
/// stream). Sequential-source specialisation will land here.
pub struct Sequential;

// ── DecoderSource ─────────────────────────────────────────────────────────────

/// Adapter that exposes decoded image bytes as an [`ImageSource`](crate::ports::source::ImageSource).
///
/// ## Type parameters
///
/// - `D`: the concrete [`ImageDecoder`] type.
/// - `F`: the [`BandFormat`] of the decoded pixels (chosen by the caller).
/// - `M`: access mode marker — [`RandomAccess`] (default) or [`Sequential`].
///
/// ## Construction
///
/// Use [`DecoderSource::new`] / [`DecoderSource::from_path`] for default eager
/// decode, [`DecoderSource::with_options`] / [`DecoderSource::with_path_options`]
/// to pass [`LoadOptions`], or [`DecoderSource::streaming`] /
/// [`DecoderSource::streaming_path`] when the decoder implements
/// [`TileImageDecoder`].
///
// Memory contract:
// Eager mode keeps at most one fully decoded backing image resident and drops
// the compressed payload unless thumbnail parity needs a later native reopen
// (JPEG DCT shrink / TIFF pyramid level selection). Streaming mode keeps no
// decoded backing image; it retains only the compressed input handle and
// decodes each requested tile into the caller-owned output buffer.
#[allow(dead_code)]
/// The `DecoderSource` type provides concrete adapter functionality in the `sources` module.
/// Use this type when you need the runtime behavior implemented by this adapter.
///
/// # Examples
///
/// ```ignore
/// let _ = core::mem::size_of::<viprs::adapters::sources::decoder_source::DecoderSource>();
/// ```
pub struct DecoderSource<'a, D: ImageDecoder, F: BandFormat, M = RandomAccess> {
    pub(super) backing: DecoderBacking<'a, D, F>,
    pub(super) stable_input: Option<StableDecoderInput>,
    /// Effective shrink factor requested by the caller.
    pub(super) shrink_factor: u8,
    /// Shrink factor already materialized in the eager backing image.
    pub(super) backing_shrink_factor: u8,
    /// Requested load options exposed to callers for diagnostics/reprobe.
    pub(super) opts: LoadOptions,
    /// The decoder instance (retained for diagnostics/reprobe).
    pub(super) decoder: D,
    /// Whether thumbnail shrink hints may eagerly materialize a deferred backing.
    pub(super) materialize_deferred_thumbnail_hints: bool,
    /// Access mode is a compile-time property only; zero runtime cost.
    _mode: PhantomData<M>,
}

impl<D: ImageDecoder, F: BandFormat> DecoderSource<'static, D, F, RandomAccess> {
    /// Decode `src` with default [`LoadOptions`] and wrap the result.
    ///
    /// # Errors
    ///
    /// Propagates any [`ViprsError::Codec`] from the decoder.
    pub fn new(decoder: D, src: &[u8]) -> Result<Self, ViprsError>
    where
        D: Sized,
    {
        Self::with_options(decoder, src, LoadOptions::default())
    }

    /// Decode `src` applying `opts` (e.g., shrink-on-load) and wrap the result.
    ///
    /// # Errors
    ///
    /// Propagates any [`ViprsError::Codec`] from the decoder.
    pub fn with_options(decoder: D, src: &[u8], opts: LoadOptions) -> Result<Self, ViprsError>
    where
        D: Sized,
    {
        let stable_input = retains_stable_input_for_thumbnail(decoder.format_name())
            .then(|| StableDecoderInput::Shared(Arc::<[u8]>::from(src)));
        let shrink_factor = opts
            .shrink_factor
            .map_or(1, |factor| normalize_shrink_factor(factor.get()));
        let decode_src = stable_input
            .as_ref()
            .and_then(|input| match input {
                StableDecoderInput::Shared(bytes) => Some(bytes.as_ref()),
                StableDecoderInput::Path(_) => None,
            })
            .map_or(src, <[u8]>::as_ref);
        let image = decoder.decode_with_options::<F>(decode_src, &opts)?;
        let backing_shrink_factor =
            eager_backing_shrink_factor(&decoder, decode_src, shrink_factor, &image);
        Ok(Self {
            backing: DecoderBacking::Eager(image),
            stable_input,
            shrink_factor,
            backing_shrink_factor,
            opts,
            decoder,
            materialize_deferred_thumbnail_hints: true,
            _mode: PhantomData,
        })
    }

    /// Decode the stable on-disk image at `path` with default [`LoadOptions`].
    ///
    /// # Errors
    ///
    /// Propagates any [`ViprsError::Codec`] from the decoder.
    pub fn from_path(decoder: D, path: impl AsRef<Path>) -> Result<Self, ViprsError>
    where
        D: Sized,
    {
        Self::with_path_options(decoder, path, LoadOptions::default())
    }

    /// Decode the stable on-disk image at `path` applying `opts`.
    ///
    /// # Errors
    ///
    /// Propagates any [`ViprsError::Codec`] from the decoder.
    pub fn with_path_options(
        decoder: D,
        path: impl AsRef<Path>,
        opts: LoadOptions,
    ) -> Result<Self, ViprsError>
    where
        D: Sized,
    {
        let path = path.as_ref().to_path_buf();
        let stable_input = retains_stable_input_for_thumbnail(decoder.format_name())
            .then(|| StableDecoderInput::Path(path.clone()));
        let shrink_factor = opts
            .shrink_factor
            .map_or(1, |factor| normalize_shrink_factor(factor.get()));
        let image = decoder.decode_path_with_options::<F>(&path, &opts)?;
        let backing_shrink_factor =
            eager_backing_shrink_factor_from_path(&decoder, &path, shrink_factor, &image);
        Ok(Self {
            backing: DecoderBacking::Eager(image),
            stable_input,
            shrink_factor,
            backing_shrink_factor,
            opts,
            decoder,
            materialize_deferred_thumbnail_hints: true,
            _mode: PhantomData,
        })
    }

    /// Probe a stable on-disk image and defer full decode until a tile is read or
    /// a thumbnail shrink hint materializes an eager backing.
    ///
    /// This lets thumbnail pipelines compute loader shrink hints from the original
    /// dimensions before any full decode occurs.
    pub fn probed_path(decoder: D, path: impl AsRef<Path>) -> Result<Self, ViprsError>
    where
        D: Sized,
    {
        Self::probed_path_with_options(decoder, path, LoadOptions::default())
    }

    /// Probe shared compressed input and defer full decode until a tile is read.
    ///
    /// This retains the encoded bytes so loader-side thumbnail hints can still
    /// reopen/redecode later without forcing eager full-raster materialization at
    /// source construction time.
    pub fn probed_shared(decoder: D, src: Arc<[u8]>) -> Result<Self, ViprsError>
    where
        D: Sized,
    {
        Self::probed_shared_with_options(decoder, src, LoadOptions::default())
    }

    /// Probe shared compressed input with explicit load options and defer decode.
    pub fn probed_shared_with_options(
        decoder: D,
        src: Arc<[u8]>,
        opts: LoadOptions,
    ) -> Result<Self, ViprsError>
    where
        D: Sized,
    {
        let (width, height, bands) = decoder.probe(src.as_ref())?;
        Ok(Self {
            backing: DecoderBacking::Deferred {
                width,
                height,
                bands,
                metadata: ImageMetadata::default(),
                image: OnceLock::new(),
            },
            stable_input: Some(StableDecoderInput::Shared(src)),
            shrink_factor: 1,
            backing_shrink_factor: 1,
            opts,
            decoder,
            materialize_deferred_thumbnail_hints: true,
            _mode: PhantomData,
        })
    }

    /// Probe a stable on-disk image with explicit load options and defer decode.
    pub fn probed_path_with_options(
        decoder: D,
        path: impl AsRef<Path>,
        opts: LoadOptions,
    ) -> Result<Self, ViprsError>
    where
        D: Sized,
    {
        let path = path.as_ref().to_path_buf();
        let (width, height, bands) = decoder.probe_path(&path)?;
        Ok(Self {
            backing: DecoderBacking::Deferred {
                width,
                height,
                bands,
                metadata: ImageMetadata::default(),
                image: OnceLock::new(),
            },
            stable_input: Some(StableDecoderInput::Path(path)),
            shrink_factor: 1,
            backing_shrink_factor: 1,
            opts,
            decoder,
            materialize_deferred_thumbnail_hints: true,
            _mode: PhantomData,
        })
    }

    /// Create a streaming source from shared compressed input.
    ///
    /// The returned source is `'static`, so it can be inserted into
    /// [`PipelineBuilder::from_source`](crate::adapters::pipeline::PipelineBuilder::from_source).
    ///
    /// # Errors
    ///
    /// Propagates decoder probe errors from [`TileImageDecoder::probe_with_options`].
    pub fn streaming_shared(
        decoder: D,
        src: Arc<[u8]>,
        opts: LoadOptions,
    ) -> Result<Self, ViprsError>
    where
        D: TileImageDecoder,
    {
        Self::streaming_with_input(decoder, DecoderInput::shared(src), opts)
    }

    /// Create a streaming source from a stable on-disk path.
    ///
    /// # Errors
    ///
    /// Propagates decoder probe errors from
    /// [`TileImageDecoder::probe_path_with_options`].
    pub fn streaming_path(
        decoder: D,
        path: impl AsRef<Path>,
        opts: LoadOptions,
    ) -> Result<Self, ViprsError>
    where
        D: TileImageDecoder,
    {
        Self::streaming_with_input(
            decoder,
            DecoderInput::stable_path(path.as_ref().to_path_buf()),
            opts,
        )
    }
}

impl<'a, D: ImageDecoder, F: BandFormat> DecoderSource<'a, D, F, RandomAccess> {
    #[cfg(feature = "jpeg")]
    pub(in crate::adapters) fn without_deferred_thumbnail_materialization(mut self) -> Self {
        self.materialize_deferred_thumbnail_hints = false;
        self
    }

    /// Create a streaming source over borrowed compressed input.
    ///
    /// This is the lowest resident-memory path: no encoded copy and no decoded
    /// full-frame backing image are retained. The caller must keep `src` alive
    /// for the lifetime of the source.
    ///
    /// # Errors
    ///
    /// Propagates decoder probe errors from [`TileImageDecoder::probe_with_options`].
    pub fn streaming(decoder: D, src: &'a [u8], opts: LoadOptions) -> Result<Self, ViprsError>
    where
        D: TileImageDecoder,
    {
        Self::streaming_with_input(decoder, DecoderInput::borrowed(src), opts)
    }

    fn streaming_with_input(
        decoder: D,
        input: DecoderInput<'a>,
        opts: LoadOptions,
    ) -> Result<Self, ViprsError>
    where
        D: TileImageDecoder,
    {
        let shrink_factor = opts
            .shrink_factor
            .map_or(1, |factor| normalize_shrink_factor(factor.get()));
        let decode_opts = normalize_streaming_options(&opts, shrink_factor);
        let info = match &input {
            DecoderInput::Borrowed(src) => decoder.probe_with_options(src, &decode_opts)?,
            DecoderInput::Shared(src) => decoder.probe_with_options(src, &decode_opts)?,
            DecoderInput::StablePath(path) => {
                decoder.probe_path_with_options(path, &decode_opts)?
            }
        };
        Ok(Self {
            backing: DecoderBacking::Streaming {
                input,
                width: info.width,
                height: info.height,
                bands: info.bands,
                metadata: info.metadata,
                probe_input: probe_input_with::<D>,
                decode_region: decode_region_with::<D, F>,
            },
            stable_input: None,
            shrink_factor,
            backing_shrink_factor: 1,
            opts,
            decoder,
            materialize_deferred_thumbnail_hints: true,
            _mode: PhantomData,
        })
    }
}

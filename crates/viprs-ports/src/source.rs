//! Source port traits for feeding pixels into the pipeline.
//!
//! Sources abstract over where image data comes from, such as memory buffers,
//! mapped files, decoders, or generated images. The scheduler reads tiles
//! through these traits without learning any infrastructure-specific details.

use viprs_core::{
    error::ViprsError,
    format::{BandFormat, BandFormatId},
    image::{DemandHint, ImageMetadata, Region},
};
use std::num::NonZeroU8;

/// Pixel source for a pipeline input.
///
/// The associated `Format` keeps the source sample type known at compile time so
/// the compiler can validate compatibility with the first pipeline operation.
///
/// # Object safety
///
/// `ImageSource` is not object-safe because of its associated `Format` type. The
/// dynamic pipeline uses [`DynImageSource`], which is object-safe and is
/// implemented automatically for every `T: ImageSource`.
///
/// # Examples
///
/// ```rust
/// use viprs::domain::{
///     error::ViprsError,
///     format::U8,
///     image::{DemandHint, Region},
/// };
/// use viprs::ports::source::ImageSource;
///
/// struct SolidWhite;
///
/// impl ImageSource for SolidWhite {
///     type Format = U8;
///
///     fn width(&self) -> u32 { 1 }
///     fn height(&self) -> u32 { 1 }
///     fn bands(&self) -> u32 { 1 }
///     fn demand_hint(&self) -> DemandHint { DemandHint::ThinStrip }
///
///     fn read_region(&self, _region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
///         output.fill(255);
///         Ok(())
///     }
/// }
/// ```
pub trait ImageSource: Send + Sync {
    /// Pixel format produced by this source.
    ///
    /// This associated type solves compile-time format compatibility with the
    /// first operation in a pipeline.
    type Format: BandFormat;

    /// Returns the source width in pixels.
    ///
    /// This method gives schedulers the image bounds needed to plan tile reads.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use viprs::domain::{error::ViprsError, format::U8, image::{DemandHint, Region}};
    /// # use viprs::ports::source::ImageSource;
    /// # struct SolidWhite;
    /// # impl ImageSource for SolidWhite {
    /// #     type Format = U8;
    /// #     fn width(&self) -> u32 { 1 }
    /// #     fn height(&self) -> u32 { 1 }
    /// #     fn bands(&self) -> u32 { 1 }
    /// #     fn demand_hint(&self) -> DemandHint { DemandHint::ThinStrip }
    /// #     fn read_region(&self, _region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
    /// #         output.fill(255);
    /// #         Ok(())
    /// #     }
    /// # }
    /// let source = SolidWhite;
    /// assert_eq!(source.width(), 1);
    /// ```
    fn width(&self) -> u32;

    /// Returns the source height in pixels.
    ///
    /// This method gives schedulers the image bounds needed to plan tile reads.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use viprs::domain::{error::ViprsError, format::U8, image::{DemandHint, Region}};
    /// # use viprs::ports::source::ImageSource;
    /// # struct SolidWhite;
    /// # impl ImageSource for SolidWhite {
    /// #     type Format = U8;
    /// #     fn width(&self) -> u32 { 1 }
    /// #     fn height(&self) -> u32 { 1 }
    /// #     fn bands(&self) -> u32 { 1 }
    /// #     fn demand_hint(&self) -> DemandHint { DemandHint::ThinStrip }
    /// #     fn read_region(&self, _region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
    /// #         output.fill(255);
    /// #         Ok(())
    /// #     }
    /// # }
    /// let source = SolidWhite;
    /// assert_eq!(source.height(), 1);
    /// ```
    fn height(&self) -> u32;

    /// Returns the number of bands produced for each pixel.
    ///
    /// This method lets buffers and downstream operations size their tile reads correctly.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use viprs::domain::{error::ViprsError, format::U8, image::{DemandHint, Region}};
    /// # use viprs::ports::source::ImageSource;
    /// # struct SolidWhite;
    /// # impl ImageSource for SolidWhite {
    /// #     type Format = U8;
    /// #     fn width(&self) -> u32 { 1 }
    /// #     fn height(&self) -> u32 { 1 }
    /// #     fn bands(&self) -> u32 { 1 }
    /// #     fn demand_hint(&self) -> DemandHint { DemandHint::ThinStrip }
    /// #     fn read_region(&self, _region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
    /// #         output.fill(255);
    /// #         Ok(())
    /// #     }
    /// # }
    /// let source = SolidWhite;
    /// assert_eq!(source.bands(), 1);
    /// ```
    fn bands(&self) -> u32;

    /// Returns the preferred read pattern for efficient tile scheduling.
    ///
    /// This method helps the scheduler choose an access order that matches the
    /// source's natural layout or decoder characteristics.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use viprs::domain::{error::ViprsError, format::U8, image::{DemandHint, Region}};
    /// # use viprs::ports::source::ImageSource;
    /// # struct SolidWhite;
    /// # impl ImageSource for SolidWhite {
    /// #     type Format = U8;
    /// #     fn width(&self) -> u32 { 1 }
    /// #     fn height(&self) -> u32 { 1 }
    /// #     fn bands(&self) -> u32 { 1 }
    /// #     fn demand_hint(&self) -> DemandHint { DemandHint::ThinStrip }
    /// #     fn read_region(&self, _region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
    /// #         output.fill(255);
    /// #         Ok(())
    /// #     }
    /// # }
    /// let source = SolidWhite;
    /// assert_eq!(source.demand_hint(), DemandHint::ThinStrip);
    /// ```
    fn demand_hint(&self) -> DemandHint;

    /// Returns optional image metadata discovered by the source.
    ///
    /// This method solves metadata propagation for loaders that can surface
    /// orientation, ICC, or related information alongside pixels.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use viprs::domain::{error::ViprsError, format::U8, image::{DemandHint, ImageMetadata, Region}};
    /// # use viprs::ports::source::ImageSource;
    /// # struct SolidWhite;
    /// # impl ImageSource for SolidWhite {
    /// #     type Format = U8;
    /// #     fn width(&self) -> u32 { 1 }
    /// #     fn height(&self) -> u32 { 1 }
    /// #     fn bands(&self) -> u32 { 1 }
    /// #     fn demand_hint(&self) -> DemandHint { DemandHint::ThinStrip }
    /// #     fn metadata(&self) -> ImageMetadata { ImageMetadata::default() }
    /// #     fn read_region(&self, _region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
    /// #         output.fill(255);
    /// #         Ok(())
    /// #     }
    /// # }
    /// let source = SolidWhite;
    /// let _metadata = source.metadata();
    /// ```
    fn metadata(&self) -> ImageMetadata {
        ImageMetadata::default()
    }

    /// Apply a decoder shrink-on-load hint, if this source can still honour it.
    ///
    /// Returns `Ok(true)` when the source updated its decode plan, `Ok(false)` for
    /// sources that are already materialised, cannot reopen natively, or do not
    /// support decoder hints. Implementations may still emulate shrink as a
    /// post-decode view while returning `Ok(false)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use std::num::NonZeroU8;
    /// # use viprs::domain::{error::ViprsError, format::U8, image::{DemandHint, Region}};
    /// # use viprs::ports::source::ImageSource;
    /// # struct SolidWhite;
    /// # impl ImageSource for SolidWhite {
    /// #     type Format = U8;
    /// #     fn width(&self) -> u32 { 1 }
    /// #     fn height(&self) -> u32 { 1 }
    /// #     fn bands(&self) -> u32 { 1 }
    /// #     fn demand_hint(&self) -> DemandHint { DemandHint::ThinStrip }
    /// #     fn read_region(&self, _region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
    /// #         output.fill(255);
    /// #         Ok(())
    /// #     }
    /// # }
    /// let mut source = SolidWhite;
    /// assert!(!source.set_shrink_on_load(NonZeroU8::new(2).unwrap())?);
    /// # Ok::<(), ViprsError>(())
    /// ```
    fn set_shrink_on_load(&mut self, _factor: NonZeroU8) -> Result<bool, ViprsError> {
        Ok(false)
    }

    /// Apply a thumbnail-specific loader pre-shrink hint, if this source can
    /// map it to a native decoder strategy.
    ///
    /// Unlike [`Self::set_shrink_on_load`], this must return `Ok(false)` for
    /// software fallbacks that merely emulate shrink after full decode. The
    /// thumbnail planner uses this to preserve libvips loader-specific parity:
    /// JPEG can re-open with DCT scaling, TIFF can select a pyramid level, PNG
    /// must leave the shrink to downstream resize stages.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use std::num::NonZeroU8;
    /// # use viprs::domain::{error::ViprsError, format::U8, image::{DemandHint, Region}};
    /// # use viprs::ports::source::ImageSource;
    /// # struct SolidWhite;
    /// # impl ImageSource for SolidWhite {
    /// #     type Format = U8;
    /// #     fn width(&self) -> u32 { 1 }
    /// #     fn height(&self) -> u32 { 1 }
    /// #     fn bands(&self) -> u32 { 1 }
    /// #     fn demand_hint(&self) -> DemandHint { DemandHint::ThinStrip }
    /// #     fn read_region(&self, _region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
    /// #         output.fill(255);
    /// #         Ok(())
    /// #     }
    /// # }
    /// let mut source = SolidWhite;
    /// assert!(!source.set_thumbnail_shrink_on_load(NonZeroU8::new(2).unwrap())?);
    /// # Ok::<(), ViprsError>(())
    /// ```
    fn set_thumbnail_shrink_on_load(&mut self, _factor: NonZeroU8) -> Result<bool, ViprsError> {
        Ok(false)
    }

    /// Fills `output` with the pixels that belong to `region`.
    ///
    /// The scheduler may call this concurrently and in arbitrary tile order.
    /// Streaming sources must therefore treat each call as an independent tile
    /// request unless their concrete type is explicitly wrapped in a sequential
    /// adapter/cache contract. Implementations that decode compressed input on
    /// demand must write into `output` directly and must not materialize a full
    /// decoded frame as hidden resident state.
    ///
    /// Coordinates outside image bounds must use clamp-to-edge extension.
    /// `output.len()` must be exactly
    /// `region.pixel_count() * self.bands() * size_of::<Self::Format::Sample>()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use viprs::domain::{error::ViprsError, format::U8, image::{DemandHint, Region}};
    /// # use viprs::ports::source::ImageSource;
    /// # struct SolidWhite;
    /// # impl ImageSource for SolidWhite {
    /// #     type Format = U8;
    /// #     fn width(&self) -> u32 { 1 }
    /// #     fn height(&self) -> u32 { 1 }
    /// #     fn bands(&self) -> u32 { 1 }
    /// #     fn demand_hint(&self) -> DemandHint { DemandHint::ThinStrip }
    /// #     fn read_region(&self, _region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
    /// #         output.fill(255);
    /// #         Ok(())
    /// #     }
    /// # }
    /// let source = SolidWhite;
    /// let mut pixel = [0u8; 1];
    /// source.read_region(Region::new(0, 0, 1, 1), &mut pixel)?;
    /// assert_eq!(pixel, [255]);
    /// # Ok::<(), ViprsError>(())
    /// ```
    fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError>;

    /// Return a direct borrow of `region` when it is already contiguous in backing storage.
    ///
    /// This is an optional zero-copy fast path for in-memory or mmap-backed sources. The
    /// returned slice must have the same layout and length contract as [`Self::read_region`].
    /// Sources that cannot expose a stable contiguous slice should return `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use viprs::domain::{error::ViprsError, format::U8, image::{DemandHint, Region}};
    /// # use viprs::ports::source::ImageSource;
    /// # struct SolidWhite;
    /// # impl ImageSource for SolidWhite {
    /// #     type Format = U8;
    /// #     fn width(&self) -> u32 { 1 }
    /// #     fn height(&self) -> u32 { 1 }
    /// #     fn bands(&self) -> u32 { 1 }
    /// #     fn demand_hint(&self) -> DemandHint { DemandHint::ThinStrip }
    /// #     fn read_region(&self, _region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
    /// #         output.fill(255);
    /// #         Ok(())
    /// #     }
    /// # }
    /// let source = SolidWhite;
    /// assert!(source.borrow_region(Region::new(0, 0, 1, 1)).is_none());
    /// ```
    fn borrow_region(&self, _region: Region) -> Option<&[u8]> {
        None
    }
}

/// Marker trait for sources that can only be consumed in forward sequential order.
///
/// This trait solves the need to distinguish decoders that cannot safely service
/// arbitrary tile reads without a cache or reordering layer.
///
/// Typical implementors include progressive JPEG decoders and network streams.
/// Operations that require random access should instead bound on
/// [`RandomAccessSource`].
///
/// # Examples
///
/// ```rust
/// # use viprs::domain::{error::ViprsError, format::U8, image::{DemandHint, Region}};
/// # use viprs::ports::source::{ImageSource, SequentialSource};
/// # struct SolidWhite;
/// # impl ImageSource for SolidWhite {
/// #     type Format = U8;
/// #     fn width(&self) -> u32 { 1 }
/// #     fn height(&self) -> u32 { 1 }
/// #     fn bands(&self) -> u32 { 1 }
/// #     fn demand_hint(&self) -> DemandHint { DemandHint::ThinStrip }
/// #     fn read_region(&self, _region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
/// #         output.fill(255);
/// #         Ok(())
/// #     }
/// # }
/// impl SequentialSource for SolidWhite {}
/// ```
pub trait SequentialSource: ImageSource {}

/// Marker trait for sources that can satisfy any `read_region` request in any order.
///
/// This trait solves compile-time enforcement for operations that require true
/// random access to their input tiles.
///
/// Typical implementors include memory-backed sources, mmap-backed sources, and
/// cached wrappers around sequential decoders.
///
/// # Examples
///
/// ```rust
/// # use viprs::domain::{error::ViprsError, format::U8, image::{DemandHint, Region}};
/// # use viprs::ports::source::{ImageSource, RandomAccessSource};
/// # struct SolidWhite;
/// # impl ImageSource for SolidWhite {
/// #     type Format = U8;
/// #     fn width(&self) -> u32 { 1 }
/// #     fn height(&self) -> u32 { 1 }
/// #     fn bands(&self) -> u32 { 1 }
/// #     fn demand_hint(&self) -> DemandHint { DemandHint::ThinStrip }
/// #     fn read_region(&self, _region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
/// #         output.fill(255);
/// #         Ok(())
/// #     }
/// # }
/// impl RandomAccessSource for SolidWhite {}
/// ```
pub trait RandomAccessSource: ImageSource {}

/// Object-safe version of `ImageSource` for use in dynamic pipeline graphs.
///
/// The dynamic pipeline cannot store `Box<dyn ImageSource>` because
/// [`ImageSource`] has an associated type. This trait preserves the scheduler
/// surface while erasing the concrete format at the runtime graph boundary.
///
/// It is not meant to be implemented manually: the blanket implementation below
/// covers every `T: ImageSource`.
///
/// # Examples
///
/// ```rust
/// use viprs::domain::{
///     error::ViprsError,
///     format::U8,
///     image::{DemandHint, Region},
/// };
/// use viprs::ports::source::{DynImageSource, ImageSource};
///
/// struct SolidWhite;
///
/// impl ImageSource for SolidWhite {
///     type Format = U8;
///
///     fn width(&self) -> u32 { 1 }
///     fn height(&self) -> u32 { 1 }
///     fn bands(&self) -> u32 { 1 }
///     fn demand_hint(&self) -> DemandHint { DemandHint::ThinStrip }
///
///     fn read_region(&self, _region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
///         output.fill(255);
///         Ok(())
///     }
/// }
///
/// let source: Box<dyn DynImageSource> = Box::new(SolidWhite);
/// assert_eq!(source.width(), 1);
/// ```
pub trait DynImageSource: Send + Sync {
    /// Returns the dynamic source width in pixels.
    ///
    /// This method lets runtime-typed pipelines inspect image bounds without
    /// knowing the concrete source type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use viprs::domain::{error::ViprsError, format::U8, image::{DemandHint, Region}};
    /// # use viprs::ports::source::{DynImageSource, ImageSource};
    /// # struct SolidWhite;
    /// # impl ImageSource for SolidWhite {
    /// #     type Format = U8;
    /// #     fn width(&self) -> u32 { 1 }
    /// #     fn height(&self) -> u32 { 1 }
    /// #     fn bands(&self) -> u32 { 1 }
    /// #     fn demand_hint(&self) -> DemandHint { DemandHint::ThinStrip }
    /// #     fn read_region(&self, _region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
    /// #         output.fill(255);
    /// #         Ok(())
    /// #     }
    /// # }
    /// let source: Box<dyn DynImageSource> = Box::new(SolidWhite);
    /// assert_eq!(source.width(), 1);
    /// ```
    fn width(&self) -> u32;
    /// Returns the dynamic source height in pixels.
    ///
    /// This method lets runtime-typed pipelines inspect image bounds without
    /// knowing the concrete source type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use viprs::domain::{error::ViprsError, format::U8, image::{DemandHint, Region}};
    /// # use viprs::ports::source::{DynImageSource, ImageSource};
    /// # struct SolidWhite;
    /// # impl ImageSource for SolidWhite {
    /// #     type Format = U8;
    /// #     fn width(&self) -> u32 { 1 }
    /// #     fn height(&self) -> u32 { 1 }
    /// #     fn bands(&self) -> u32 { 1 }
    /// #     fn demand_hint(&self) -> DemandHint { DemandHint::ThinStrip }
    /// #     fn read_region(&self, _region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
    /// #         output.fill(255);
    /// #         Ok(())
    /// #     }
    /// # }
    /// let source: Box<dyn DynImageSource> = Box::new(SolidWhite);
    /// assert_eq!(source.height(), 1);
    /// ```
    fn height(&self) -> u32;
    /// Returns the number of bands for the dynamic source.
    ///
    /// This method helps dynamic pipelines size buffers without the static source type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use viprs::domain::{error::ViprsError, format::U8, image::{DemandHint, Region}};
    /// # use viprs::ports::source::{DynImageSource, ImageSource};
    /// # struct SolidWhite;
    /// # impl ImageSource for SolidWhite {
    /// #     type Format = U8;
    /// #     fn width(&self) -> u32 { 1 }
    /// #     fn height(&self) -> u32 { 1 }
    /// #     fn bands(&self) -> u32 { 1 }
    /// #     fn demand_hint(&self) -> DemandHint { DemandHint::ThinStrip }
    /// #     fn read_region(&self, _region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
    /// #         output.fill(255);
    /// #         Ok(())
    /// #     }
    /// # }
    /// let source: Box<dyn DynImageSource> = Box::new(SolidWhite);
    /// assert_eq!(source.bands(), 1);
    /// ```
    fn bands(&self) -> u32;
    /// Returns the erased runtime band format identifier.
    ///
    /// This method solves format inspection after crossing the object-safe source boundary.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use viprs::domain::{error::ViprsError, format::{BandFormatId, U8}, image::{DemandHint, Region}};
    /// # use viprs::ports::source::{DynImageSource, ImageSource};
    /// # struct SolidWhite;
    /// # impl ImageSource for SolidWhite {
    /// #     type Format = U8;
    /// #     fn width(&self) -> u32 { 1 }
    /// #     fn height(&self) -> u32 { 1 }
    /// #     fn bands(&self) -> u32 { 1 }
    /// #     fn demand_hint(&self) -> DemandHint { DemandHint::ThinStrip }
    /// #     fn read_region(&self, _region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
    /// #         output.fill(255);
    /// #         Ok(())
    /// #     }
    /// # }
    /// let source: Box<dyn DynImageSource> = Box::new(SolidWhite);
    /// assert_eq!(source.format(), BandFormatId::U8);
    /// ```
    fn format(&self) -> BandFormatId;
    /// Returns the preferred access pattern for the dynamic source.
    ///
    /// This method preserves scheduler planning hints after erasing the concrete type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use viprs::domain::{error::ViprsError, format::U8, image::{DemandHint, Region}};
    /// # use viprs::ports::source::{DynImageSource, ImageSource};
    /// # struct SolidWhite;
    /// # impl ImageSource for SolidWhite {
    /// #     type Format = U8;
    /// #     fn width(&self) -> u32 { 1 }
    /// #     fn height(&self) -> u32 { 1 }
    /// #     fn bands(&self) -> u32 { 1 }
    /// #     fn demand_hint(&self) -> DemandHint { DemandHint::ThinStrip }
    /// #     fn read_region(&self, _region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
    /// #         output.fill(255);
    /// #         Ok(())
    /// #     }
    /// # }
    /// let source: Box<dyn DynImageSource> = Box::new(SolidWhite);
    /// assert_eq!(source.demand_hint(), DemandHint::ThinStrip);
    /// ```
    fn demand_hint(&self) -> DemandHint;
    /// Returns metadata discovered by the dynamic source.
    ///
    /// This method keeps metadata available even when the source travels through
    /// a runtime-typed pipeline graph.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use viprs::domain::{error::ViprsError, format::U8, image::{DemandHint, ImageMetadata, Region}};
    /// # use viprs::ports::source::{DynImageSource, ImageSource};
    /// # struct SolidWhite;
    /// # impl ImageSource for SolidWhite {
    /// #     type Format = U8;
    /// #     fn width(&self) -> u32 { 1 }
    /// #     fn height(&self) -> u32 { 1 }
    /// #     fn bands(&self) -> u32 { 1 }
    /// #     fn demand_hint(&self) -> DemandHint { DemandHint::ThinStrip }
    /// #     fn metadata(&self) -> ImageMetadata { ImageMetadata::default() }
    /// #     fn read_region(&self, _region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
    /// #         output.fill(255);
    /// #         Ok(())
    /// #     }
    /// # }
    /// let source: Box<dyn DynImageSource> = Box::new(SolidWhite);
    /// let _metadata = source.metadata();
    /// ```
    fn metadata(&self) -> ImageMetadata;
    /// Applies a generic shrink-on-load hint to a dynamic source.
    ///
    /// This method preserves decoder planning hooks after erasing the concrete type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use std::num::NonZeroU8;
    /// # use viprs::domain::{error::ViprsError, format::U8, image::{DemandHint, Region}};
    /// # use viprs::ports::source::{DynImageSource, ImageSource};
    /// # struct SolidWhite;
    /// # impl ImageSource for SolidWhite {
    /// #     type Format = U8;
    /// #     fn width(&self) -> u32 { 1 }
    /// #     fn height(&self) -> u32 { 1 }
    /// #     fn bands(&self) -> u32 { 1 }
    /// #     fn demand_hint(&self) -> DemandHint { DemandHint::ThinStrip }
    /// #     fn read_region(&self, _region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
    /// #         output.fill(255);
    /// #         Ok(())
    /// #     }
    /// # }
    /// let mut source: Box<dyn DynImageSource> = Box::new(SolidWhite);
    /// assert!(!source.set_shrink_on_load(NonZeroU8::new(2).unwrap())?);
    /// # Ok::<(), ViprsError>(())
    /// ```
    fn set_shrink_on_load(&mut self, factor: NonZeroU8) -> Result<bool, ViprsError>;
    /// Applies a thumbnail-specific shrink-on-load hint to a dynamic source.
    ///
    /// This method preserves loader-specific thumbnail planning after erasing the concrete type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use std::num::NonZeroU8;
    /// # use viprs::domain::{error::ViprsError, format::U8, image::{DemandHint, Region}};
    /// # use viprs::ports::source::{DynImageSource, ImageSource};
    /// # struct SolidWhite;
    /// # impl ImageSource for SolidWhite {
    /// #     type Format = U8;
    /// #     fn width(&self) -> u32 { 1 }
    /// #     fn height(&self) -> u32 { 1 }
    /// #     fn bands(&self) -> u32 { 1 }
    /// #     fn demand_hint(&self) -> DemandHint { DemandHint::ThinStrip }
    /// #     fn read_region(&self, _region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
    /// #         output.fill(255);
    /// #         Ok(())
    /// #     }
    /// # }
    /// let mut source: Box<dyn DynImageSource> = Box::new(SolidWhite);
    /// assert!(!source.set_thumbnail_shrink_on_load(NonZeroU8::new(2).unwrap())?);
    /// # Ok::<(), ViprsError>(())
    /// ```
    fn set_thumbnail_shrink_on_load(&mut self, factor: NonZeroU8) -> Result<bool, ViprsError>;
    /// Reads one requested region into a caller-provided output buffer.
    ///
    /// This method preserves tile-based reads across the object-safe source boundary.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use viprs::domain::{error::ViprsError, format::U8, image::{DemandHint, Region}};
    /// # use viprs::ports::source::{DynImageSource, ImageSource};
    /// # struct SolidWhite;
    /// # impl ImageSource for SolidWhite {
    /// #     type Format = U8;
    /// #     fn width(&self) -> u32 { 1 }
    /// #     fn height(&self) -> u32 { 1 }
    /// #     fn bands(&self) -> u32 { 1 }
    /// #     fn demand_hint(&self) -> DemandHint { DemandHint::ThinStrip }
    /// #     fn read_region(&self, _region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
    /// #         output.fill(255);
    /// #         Ok(())
    /// #     }
    /// # }
    /// let source: Box<dyn DynImageSource> = Box::new(SolidWhite);
    /// let mut pixel = [0u8; 1];
    /// source.read_region(Region::new(0, 0, 1, 1), &mut pixel)?;
    /// assert_eq!(pixel, [255]);
    /// # Ok::<(), ViprsError>(())
    /// ```
    fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError>;
    /// Returns a borrowed contiguous region when the dynamic source can expose one.
    ///
    /// This method preserves optional zero-copy access across the object-safe source boundary.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use viprs::domain::{error::ViprsError, format::U8, image::{DemandHint, Region}};
    /// # use viprs::ports::source::{DynImageSource, ImageSource};
    /// # struct SolidWhite;
    /// # impl ImageSource for SolidWhite {
    /// #     type Format = U8;
    /// #     fn width(&self) -> u32 { 1 }
    /// #     fn height(&self) -> u32 { 1 }
    /// #     fn bands(&self) -> u32 { 1 }
    /// #     fn demand_hint(&self) -> DemandHint { DemandHint::ThinStrip }
    /// #     fn read_region(&self, _region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
    /// #         output.fill(255);
    /// #         Ok(())
    /// #     }
    /// # }
    /// let source: Box<dyn DynImageSource> = Box::new(SolidWhite);
    /// assert!(source.borrow_region(Region::new(0, 0, 1, 1)).is_none());
    /// ```
    fn borrow_region(&self, _region: Region) -> Option<&[u8]> {
        None
    }
}

/// Blanket implementation that makes every static source usable in a dynamic pipeline.
///
/// `format()` is derived from `<T as ImageSource>::Format::ID`, lowering static
/// type information into a runtime value for erased pipeline graphs.
impl<T: ImageSource> DynImageSource for T {
    fn width(&self) -> u32 {
        ImageSource::width(self)
    }
    fn height(&self) -> u32 {
        ImageSource::height(self)
    }
    fn bands(&self) -> u32 {
        ImageSource::bands(self)
    }
    fn format(&self) -> BandFormatId {
        <T::Format as BandFormat>::ID
    }
    fn demand_hint(&self) -> DemandHint {
        ImageSource::demand_hint(self)
    }
    fn metadata(&self) -> ImageMetadata {
        ImageSource::metadata(self)
    }
    fn set_shrink_on_load(&mut self, factor: NonZeroU8) -> Result<bool, ViprsError> {
        ImageSource::set_shrink_on_load(self, factor)
    }
    fn set_thumbnail_shrink_on_load(&mut self, factor: NonZeroU8) -> Result<bool, ViprsError> {
        ImageSource::set_thumbnail_shrink_on_load(self, factor)
    }
    fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        ImageSource::read_region(self, region, output)
    }
    fn borrow_region(&self, region: Region) -> Option<&[u8]> {
        ImageSource::borrow_region(self, region)
    }
}

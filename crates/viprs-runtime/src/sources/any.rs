//! Any image source adapter.
//!
//! This module exposes concrete source implementations or helpers that feed
//! pixels into compiled pipelines.

use crate::{
    domain::{
        error::ViprsError,
        format::{BandFormatId, F32, F64, I16, I32, U8, U16, U32},
        image::{DemandHint, ImageMetadata, Region},
    },
    ports::source::{DynImageSource, ImageSource},
    sources::memory::MemorySource,
};
use std::num::NonZeroU8;

/// Type-erased enum over the concrete `MemorySource<F>` variants.
///
/// Allows callers to construct an in-memory source without committing to a specific
/// `BandFormat` type parameter at compile time. The format is chosen at runtime based
/// on the variant. `AnySource` implements `DynImageSource` via a match on each method,
/// so it can be passed directly to `PipelineBuilder::from_source` without `Box::new`.
///
/// This is the bridge between the `ImageSource` static-dispatch world and the dynamic
/// pipeline builder. It is not intended as a general-purpose type — it covers the
/// `MemorySource` family. For other source types (e.g., `MmapSource<F>`), the blanket
/// `impl<T: ImageSource> DynImageSource for T` applies automatically — pass them
/// directly to the pipeline without wrapping in `AnySource`.
pub enum AnySource {
    /// In-memory source backed by `u8` samples.
    U8(MemorySource<U8>),
    /// In-memory source backed by `u16` samples.
    U16(MemorySource<U16>),
    /// In-memory source backed by `i16` samples.
    I16(MemorySource<I16>),
    /// In-memory source backed by `u32` samples.
    U32(MemorySource<U32>),
    /// In-memory source backed by `i32` samples.
    I32(MemorySource<I32>),
    /// In-memory source backed by `f32` samples.
    F32(MemorySource<F32>),
    /// In-memory source backed by `f64` samples.
    F64(MemorySource<F64>),
}

impl DynImageSource for AnySource {
    fn width(&self) -> u32 {
        match self {
            Self::U8(s) => ImageSource::width(s),
            Self::U16(s) => ImageSource::width(s),
            Self::I16(s) => ImageSource::width(s),
            Self::U32(s) => ImageSource::width(s),
            Self::I32(s) => ImageSource::width(s),
            Self::F32(s) => ImageSource::width(s),
            Self::F64(s) => ImageSource::width(s),
        }
    }

    fn height(&self) -> u32 {
        match self {
            Self::U8(s) => ImageSource::height(s),
            Self::U16(s) => ImageSource::height(s),
            Self::I16(s) => ImageSource::height(s),
            Self::U32(s) => ImageSource::height(s),
            Self::I32(s) => ImageSource::height(s),
            Self::F32(s) => ImageSource::height(s),
            Self::F64(s) => ImageSource::height(s),
        }
    }

    fn bands(&self) -> u32 {
        match self {
            Self::U8(s) => ImageSource::bands(s),
            Self::U16(s) => ImageSource::bands(s),
            Self::I16(s) => ImageSource::bands(s),
            Self::U32(s) => ImageSource::bands(s),
            Self::I32(s) => ImageSource::bands(s),
            Self::F32(s) => ImageSource::bands(s),
            Self::F64(s) => ImageSource::bands(s),
        }
    }

    fn format(&self) -> BandFormatId {
        match self {
            Self::U8(_) => BandFormatId::U8,
            Self::U16(_) => BandFormatId::U16,
            Self::I16(_) => BandFormatId::I16,
            Self::U32(_) => BandFormatId::U32,
            Self::I32(_) => BandFormatId::I32,
            Self::F32(_) => BandFormatId::F32,
            Self::F64(_) => BandFormatId::F64,
        }
    }

    fn demand_hint(&self) -> DemandHint {
        match self {
            Self::U8(s) => ImageSource::demand_hint(s),
            Self::U16(s) => ImageSource::demand_hint(s),
            Self::I16(s) => ImageSource::demand_hint(s),
            Self::U32(s) => ImageSource::demand_hint(s),
            Self::I32(s) => ImageSource::demand_hint(s),
            Self::F32(s) => ImageSource::demand_hint(s),
            Self::F64(s) => ImageSource::demand_hint(s),
        }
    }

    fn metadata(&self) -> ImageMetadata {
        match self {
            Self::U8(s) => ImageSource::metadata(s),
            Self::U16(s) => ImageSource::metadata(s),
            Self::I16(s) => ImageSource::metadata(s),
            Self::U32(s) => ImageSource::metadata(s),
            Self::I32(s) => ImageSource::metadata(s),
            Self::F32(s) => ImageSource::metadata(s),
            Self::F64(s) => ImageSource::metadata(s),
        }
    }

    fn set_shrink_on_load(&mut self, factor: NonZeroU8) -> Result<bool, ViprsError> {
        match self {
            Self::U8(s) => ImageSource::set_shrink_on_load(s, factor),
            Self::U16(s) => ImageSource::set_shrink_on_load(s, factor),
            Self::I16(s) => ImageSource::set_shrink_on_load(s, factor),
            Self::U32(s) => ImageSource::set_shrink_on_load(s, factor),
            Self::I32(s) => ImageSource::set_shrink_on_load(s, factor),
            Self::F32(s) => ImageSource::set_shrink_on_load(s, factor),
            Self::F64(s) => ImageSource::set_shrink_on_load(s, factor),
        }
    }

    fn set_thumbnail_shrink_on_load(&mut self, factor: NonZeroU8) -> Result<bool, ViprsError> {
        match self {
            Self::U8(s) => ImageSource::set_thumbnail_shrink_on_load(s, factor),
            Self::U16(s) => ImageSource::set_thumbnail_shrink_on_load(s, factor),
            Self::I16(s) => ImageSource::set_thumbnail_shrink_on_load(s, factor),
            Self::U32(s) => ImageSource::set_thumbnail_shrink_on_load(s, factor),
            Self::I32(s) => ImageSource::set_thumbnail_shrink_on_load(s, factor),
            Self::F32(s) => ImageSource::set_thumbnail_shrink_on_load(s, factor),
            Self::F64(s) => ImageSource::set_thumbnail_shrink_on_load(s, factor),
        }
    }

    fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        match self {
            Self::U8(s) => ImageSource::read_region(s, region, output),
            Self::U16(s) => ImageSource::read_region(s, region, output),
            Self::I16(s) => ImageSource::read_region(s, region, output),
            Self::U32(s) => ImageSource::read_region(s, region, output),
            Self::I32(s) => ImageSource::read_region(s, region, output),
            Self::F32(s) => ImageSource::read_region(s, region, output),
            Self::F64(s) => ImageSource::read_region(s, region, output),
        }
    }

    fn borrow_region(&self, region: Region) -> Option<&[u8]> {
        match self {
            Self::U8(s) => ImageSource::borrow_region(s, region),
            Self::U16(s) => ImageSource::borrow_region(s, region),
            Self::I16(s) => ImageSource::borrow_region(s, region),
            Self::U32(s) => ImageSource::borrow_region(s, region),
            Self::I32(s) => ImageSource::borrow_region(s, region),
            Self::F32(s) => ImageSource::borrow_region(s, region),
            Self::F64(s) => ImageSource::borrow_region(s, region),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_u8_source() -> AnySource {
        AnySource::U8(MemorySource::<U8>::new(4, 4, 1, vec![0u8; 16]).unwrap())
    }

    #[test]
    fn any_source_u8_reports_correct_format() {
        let src = make_u8_source();
        assert_eq!(DynImageSource::format(&src), BandFormatId::U8);
    }

    #[test]
    fn any_source_dimensions_pass_through() {
        let src = AnySource::F32(MemorySource::<F32>::new(10, 20, 3, vec![0.0f32; 600]).unwrap());
        assert_eq!(DynImageSource::width(&src), 10);
        assert_eq!(DynImageSource::height(&src), 20);
        assert_eq!(DynImageSource::bands(&src), 3);
        assert_eq!(DynImageSource::format(&src), BandFormatId::F32);
    }

    #[test]
    fn any_source_read_region_fills_zeros() {
        let src = make_u8_source();
        let mut buf = vec![0xffu8; 4];
        DynImageSource::read_region(&src, Region::new(0, 0, 2, 2), &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0));
    }
}

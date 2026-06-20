//! Zero image source adapter.
//!
//! This module exposes concrete source implementations or helpers that feed
//! pixels into compiled pipelines.

use std::marker::PhantomData;

use crate::{
    domain::{
        error::ViprsError,
        format::BandFormat,
        image::{DemandHint, Region},
    },
    ports::source::{ImageSource, RandomAccessSource},
};

/// Synthetic source that always produces zero samples.
///
/// Used in tests that do not require real pixel data, and as the backing store for
/// `PipelineBuilder::new`. Generic over `F: BandFormat` so that `ImageSource::Format`
/// is known at compile time.
///
/// The `dyn DynImageSource` path used by `PipelineArena` erases the `F` parameter
/// via the blanket `impl<T: ImageSource> DynImageSource for T`.
pub struct ZeroSource<F: BandFormat> {
    width: u32,
    height: u32,
    bands: u32,
    hint: DemandHint,
    _format: PhantomData<F>,
}

impl<F: BandFormat> ZeroSource<F> {
    /// `new` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs_runtime::sources::zero::new;
    /// ```
    #[must_use]
    pub const fn new(width: u32, height: u32, bands: u32) -> Self {
        Self {
            width,
            height,
            bands,
            hint: DemandHint::ThinStrip,
            _format: PhantomData,
        }
    }
}

impl<F: BandFormat> ImageSource for ZeroSource<F> {
    type Format = F;

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn bands(&self) -> u32 {
        self.bands
    }

    fn demand_hint(&self) -> DemandHint {
        self.hint
    }

    #[inline]
    fn read_region(&self, _region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        output.fill(0);
        Ok(())
    }
}

/// `ZeroSource` generates pixel data on the fly and can respond to any region.
impl<F: BandFormat> RandomAccessSource for ZeroSource<F> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::format::{BandFormatId, F32, U8};
    use crate::ports::source::{DynImageSource, ImageSource};

    #[test]
    fn zero_source_fills_output_with_zeros() {
        let src = ZeroSource::<U8>::new(4, 4, 1);
        let mut output = vec![0xffu8; 16];
        ImageSource::read_region(&src, Region::new(0, 0, 4, 4), &mut output).unwrap();
        assert!(output.iter().all(|&b| b == 0));
    }

    #[test]
    fn zero_source_dimensions_match_constructor() {
        let src = ZeroSource::<F32>::new(10, 20, 3);
        assert_eq!(ImageSource::width(&src), 10);
        assert_eq!(ImageSource::height(&src), 20);
        assert_eq!(ImageSource::bands(&src), 3);
        // Via DynImageSource blanket impl, format() returns F32::ID
        let dyn_src: &dyn DynImageSource = &src;
        assert_eq!(dyn_src.format(), BandFormatId::F32);
    }

    #[test]
    fn zero_source_demand_hint_is_thin_strip() {
        let src = ZeroSource::<U8>::new(4, 4, 1);
        assert_eq!(ImageSource::demand_hint(&src), DemandHint::ThinStrip);
    }
}

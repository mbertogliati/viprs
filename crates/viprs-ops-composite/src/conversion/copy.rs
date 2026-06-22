use std::marker::PhantomData;

use viprs_core::{
    format::BandFormat,
    image::{DemandHint, ImageMetadata, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

pub use viprs_core::image::MetadataOverrides;

/// Identity copy operation.
///
/// Pixels are copied tile-for-tile with no transformation. Metadata lives on
/// `Image`, not on `Tile`, so metadata propagation must happen when the caller
/// materializes an output image from the pipeline.
///
/// # Examples
/// ```ignore
/// use viprs_ops_composite::conversion::copy::CopyOp;
///
/// let op = CopyOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone)]
pub struct CopyOp<F: BandFormat> {
    /// Sparse metadata overrides to apply on top of the source image metadata.
    /// The pixel path does not see these fields directly.
    pub overrides: MetadataOverrides,
    _phantom: PhantomData<F>,
}

impl<F: BandFormat> CopyOp<F> {
    #[must_use]
    /// Creates a new `CopyOp`.
    pub const fn new(overrides: MetadataOverrides) -> Self {
        Self {
            overrides,
            _phantom: PhantomData,
        }
    }

    #[must_use]
    /// Returns or performs apply metadata.
    pub fn apply_metadata(&self, source: &ImageMetadata) -> ImageMetadata {
        source.merge_overrides(&self.overrides)
    }
}

impl<F: BandFormat> Default for CopyOp<F> {
    fn default() -> Self {
        Self::new(MetadataOverrides::default())
    }
}

impl<F: BandFormat> Op for CopyOp<F> {
    type Input = F;
    type Output = F;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    fn transform_metadata(&self, source: &ImageMetadata) -> ImageMetadata {
        self.apply_metadata(source)
    }

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        debug_assert_eq!(
            input.bands, output.bands,
            "CopyOp requires matching band counts"
        );
        output.data.copy_from_slice(input.data);
    }
}

impl<F: BandFormat> PixelLocalOp for CopyOp<F> {}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::U8,
        image::{ImageMetadata, Interpretation},
    };

    prop_compose! {
        fn identity_case()(width in 1u32..=4, height in 1u32..=4, bands in 1u32..=4)
                         (width in Just(width), height in Just(height), bands in Just(bands),
                          samples in prop::collection::vec(any::<u8>(), (width * height * bands) as usize))
                         -> (u32, u32, u32, Vec<u8>) {
            (width, height, bands, samples)
        }
    }

    proptest! {
        #[test]
        fn copy_op_is_identity_for_any_tile((width, height, bands, samples) in identity_case()) {
            let op = CopyOp::<U8>::default();
            let region = Region::new(0, 0, width, height);
            let input = Tile::<U8>::new(region, bands, &samples);
            let mut output_samples = vec![0u8; samples.len()];
            let mut output = TileMut::<U8>::new(region, bands, &mut output_samples);
            op.start();

            op.process_region(&mut (), &input, &mut output);

            prop_assert_eq!(output_samples, samples);
        }

        #[test]
        fn copy_op_preserves_boundary_values(bands in 1u32..=4, value in prop_oneof![Just(0u8), Just(u8::MAX)]) {
            let op = CopyOp::<U8>::default();
            let region = Region::new(0, 0, 1, 1);
            let input_samples = vec![value; bands as usize];
            let input = Tile::<U8>::new(region, bands, &input_samples);
            let mut output_samples = vec![0u8; bands as usize];
            let mut output = TileMut::<U8>::new(region, bands, &mut output_samples);
            op.start();

            op.process_region(&mut (), &input, &mut output);

            prop_assert_eq!(output_samples, input_samples);
        }
    }

    #[test]
    fn merge_overrides_replaces_and_inherits_fields() {
        let source = ImageMetadata {
            interpretation: Some(Interpretation::Srgb),
            orientation: Some(6),
            icc_profile: Some(vec![1, 2, 3]),
            exif: Some(vec![9, 9]),
            xmp: Some(vec![8, 8]),
            xres: Some(10.0),
            yres: Some(20.0),
            page_height: Some(32),
            n_pages: Some(4),
            ..ImageMetadata::default()
        };
        let overrides = MetadataOverrides {
            interpretation: Some(Interpretation::Lab),
            orientation: Some(1),
            icc_profile: Some(None),
            xres: Some(72.0),
            yres: None,
        };

        let merged = source.merge_overrides(&overrides);

        assert_eq!(merged.interpretation, Some(Interpretation::Lab));
        assert_eq!(merged.orientation, Some(1));
        assert_eq!(merged.icc_profile, None);
        assert_eq!(merged.exif, source.exif);
        assert_eq!(merged.xres, Some(72.0));
        assert_eq!(merged.yres, Some(20.0));
    }

    #[test]
    fn apply_metadata_uses_sparse_overrides() {
        let op = CopyOp::<U8>::new(MetadataOverrides {
            interpretation: Some(Interpretation::Grey16),
            orientation: None,
            icc_profile: Some(Some(vec![4, 5, 6])),
            xres: None,
            yres: Some(300.0),
        });
        let source = ImageMetadata {
            interpretation: Some(Interpretation::Srgb),
            orientation: Some(8),
            icc_profile: None,
            exif: Some(vec![7, 8]),
            xmp: Some(vec![6, 5, 4]),
            xres: Some(10.0),
            yres: Some(20.0),
            page_height: Some(16),
            n_pages: Some(2),
            ..ImageMetadata::default()
        };

        let merged = op.apply_metadata(&source);

        assert_eq!(merged.interpretation, Some(Interpretation::Grey16));
        assert_eq!(merged.orientation, Some(8));
        assert_eq!(merged.icc_profile, Some(vec![4, 5, 6]));
        assert_eq!(merged.exif, source.exif);
        assert_eq!(merged.xres, Some(10.0));
        assert_eq!(merged.yres, Some(300.0));
    }
}

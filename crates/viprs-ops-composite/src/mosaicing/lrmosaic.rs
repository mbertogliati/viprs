use std::marker::PhantomData;

use crate::mosaicing::{LrMerge, TiePointMatch, TiePointOffset, auto_mosaic::AutoMosaicSearch};

use viprs_core::{
    error::ViprsError, format::BandFormat, image::Tile, shared_ops::sample_conv::ToF64,
};

/// Libvips-style left-right mosaic with automatic tie-point refinement.
///
/// # Examples
/// ```ignore
/// use viprs_ops_composite::mosaicing::lrmosaic::LrMosaicOp;
///
/// let op = LrMosaicOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct LrMosaicOp<F: BandFormat> {
    search: AutoMosaicSearch<F>,
    blend_width: u32,
    bands: u32,
    _format: PhantomData<F>,
}

impl<F: BandFormat> LrMosaicOp<F> {
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    /// Creates a new `LrMosaicOp`.
    pub const fn new(
        ref_width: u32,
        ref_height: u32,
        sec_width: u32,
        sec_height: u32,
        xref: i32,
        yref: i32,
        xsec: i32,
        ysec: i32,
        search_radius: u32,
        blend_width: u32,
        bands: u32,
    ) -> Self {
        Self {
            search: AutoMosaicSearch::new(
                ref_width,
                ref_height,
                sec_width,
                sec_height,
                xref,
                yref,
                xsec,
                ysec,
                search_radius,
            ),
            blend_width,
            bands,
            _format: PhantomData,
        }
    }

    /// Returns or performs detect offset.
    pub fn detect_offset(
        &self,
        reference: &Tile<F>,
        secondary: &Tile<F>,
    ) -> Result<TiePointMatch, ViprsError>
    where
        F::Sample: ToF64,
    {
        self.search.detect_offset(reference, secondary)
    }

    #[must_use]
    /// Returns or performs build merge.
    pub const fn build_merge(&self, offset: TiePointOffset) -> LrMerge<F> {
        LrMerge::new(
            self.search.reference_region().width,
            self.search.reference_region().height,
            self.search.secondary_region().width,
            self.search.secondary_region().height,
            offset.dx,
            offset.dy,
            self.blend_width,
            self.bands,
        )
    }

    /// Returns or performs detect and build merge.
    pub fn detect_and_build_merge(
        &self,
        reference: &Tile<F>,
        secondary: &Tile<F>,
    ) -> Result<(TiePointMatch, LrMerge<F>), ViprsError>
    where
        F::Sample: ToF64,
    {
        let tie_point = self.detect_offset(reference, secondary)?;
        Ok((tie_point, self.build_merge(tie_point.offset)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use viprs_core::{format::U8, image::Region, op::DynOperation};

    fn patterned(width: usize, height: usize, seed: u8) -> Vec<u8> {
        let mut pixels = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let value = (x * x * 17 + y * 29 + x * y * 13 + usize::from(seed)) % 251;
                pixels.push((value + 1) as u8);
            }
        }
        pixels
    }

    fn crop(
        source: &[u8],
        source_width: usize,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> Vec<u8> {
        let mut out = Vec::with_capacity(width * height);
        for row in 0..height {
            let start = (y + row) * source_width + x;
            out.extend_from_slice(&source[start..start + width]);
        }
        out
    }

    fn run_merge(op: &LrMerge<U8>, reference: &[u8], secondary: &[u8]) -> Vec<u8> {
        let output_region = Region::new(0, 0, op.output_width(), op.output_height());
        let input_regions = [
            op.required_input_region_slot(&output_region, 0),
            op.required_input_region_slot(&output_region, 1),
        ];
        let mut output = vec![0u8; output_region.pixel_count()];
        let mut state = op.dyn_start();
        op.dyn_process_region_multi(
            state.as_mut(),
            &[reference, secondary],
            &mut output,
            &input_regions,
            output_region,
        );
        output
    }

    fn compose_expected(
        width: usize,
        height: usize,
        reference: &[u8],
        reference_width: usize,
        reference_height: usize,
        secondary: &[u8],
        secondary_width: usize,
        secondary_height: usize,
        secondary_left: usize,
        secondary_top: usize,
    ) -> Vec<u8> {
        let mut out = vec![0u8; width * height];
        for y in 0..reference_height {
            let src = y * reference_width;
            let dst = y * width;
            out[dst..dst + reference_width].copy_from_slice(&reference[src..src + reference_width]);
        }
        for y in 0..secondary_height {
            let src = y * secondary_width;
            let dst = (secondary_top + y) * width + secondary_left;
            out[dst..dst + secondary_width].copy_from_slice(&secondary[src..src + secondary_width]);
        }
        out
    }

    fn columns(image: &[u8], image_width: usize, x: usize, width: usize, height: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(width * height);
        for y in 0..height {
            let row_start = y * image_width + x;
            out.extend_from_slice(&image[row_start..row_start + width]);
        }
        out
    }

    #[test]
    fn identical_images_find_zero_offset() {
        let pixels = patterned(7, 6, 11);
        let region = Region::new(0, 0, 7, 6);
        let reference = Tile::<U8>::new(region, 1, &pixels);
        let secondary = Tile::<U8>::new(region, 1, &pixels);
        let op = LrMosaicOp::<U8>::new(7, 6, 7, 6, 0, 0, 0, 0, 2, 3, 1);

        let found = op.detect_offset(&reference, &secondary).unwrap();

        assert_eq!(found.offset, TiePointOffset { dx: 0, dy: 0 });
    }

    #[test]
    fn shifted_pair_finds_expected_offset_exactly() {
        let base = patterned(14, 8, 23);
        let reference_pixels = crop(&base, 14, 0, 0, 9, 7);
        let secondary_pixels = crop(&base, 14, 5, 1, 9, 7);
        let reference = Tile::<U8>::new(Region::new(0, 0, 9, 7), 1, &reference_pixels);
        let secondary = Tile::<U8>::new(Region::new(0, 0, 9, 7), 1, &secondary_pixels);
        let op = LrMosaicOp::<U8>::new(9, 7, 9, 7, 4, 0, 0, 0, 2, 4, 1);
        let found = op.detect_offset(&reference, &secondary).unwrap();

        assert_eq!(found.offset, TiePointOffset { dx: -5, dy: -1 });
    }

    #[test]
    fn exact_overlap_delegates_to_lrmerge() {
        let base = patterned(14, 8, 23);
        let reference_pixels = crop(&base, 14, 0, 0, 9, 7);
        let secondary_pixels = crop(&base, 14, 5, 1, 9, 7);
        let reference = Tile::<U8>::new(Region::new(0, 0, 9, 7), 1, &reference_pixels);
        let secondary = Tile::<U8>::new(Region::new(0, 0, 9, 7), 1, &secondary_pixels);
        let op = LrMosaicOp::<U8>::new(9, 7, 9, 7, 5, 1, 0, 0, 2, 4, 1);

        let (found, merge) = op.detect_and_build_merge(&reference, &secondary).unwrap();

        assert_eq!(found.offset, TiePointOffset { dx: -5, dy: -1 });

        let stitched = run_merge(&merge, &reference_pixels, &secondary_pixels);
        let expected = compose_expected(
            14,
            8,
            &reference_pixels,
            9,
            7,
            &secondary_pixels,
            9,
            7,
            5,
            1,
        );
        assert_eq!(stitched, expected);
    }

    #[test]
    fn approximate_tie_point_keeps_lr_seam_aligned() {
        let base = patterned(14, 8, 23);
        let reference_pixels = crop(&base, 14, 0, 0, 9, 7);
        let secondary_pixels = crop(&base, 14, 5, 1, 9, 7);
        let reference = Tile::<U8>::new(Region::new(0, 0, 9, 7), 1, &reference_pixels);
        let secondary = Tile::<U8>::new(Region::new(0, 0, 9, 7), 1, &secondary_pixels);
        let op = LrMosaicOp::<U8>::new(9, 7, 9, 7, 4, 0, 0, 0, 2, 4, 1);

        let (found, merge) = op.detect_and_build_merge(&reference, &secondary).unwrap();
        let stitched = run_merge(&merge, &reference_pixels, &secondary_pixels);
        let expected = compose_expected(
            14,
            8,
            &reference_pixels,
            9,
            7,
            &secondary_pixels,
            9,
            7,
            5,
            1,
        );

        assert_eq!(found.offset, TiePointOffset { dx: -5, dy: -1 });
        assert_eq!(
            columns(&stitched, 14, 4, 4, 8),
            columns(&expected, 14, 4, 4, 8)
        );
    }

    #[test]
    fn single_pixel_tiles_find_zero_offset() {
        let pixels = [173u8];
        let region = Region::new(0, 0, 1, 1);
        let reference = Tile::<U8>::new(region, 1, &pixels);
        let secondary = Tile::<U8>::new(region, 1, &pixels);
        let op = LrMosaicOp::<U8>::new(1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1);

        let found = op.detect_offset(&reference, &secondary).unwrap();

        assert_eq!(found.offset, TiePointOffset { dx: 0, dy: 0 });
        assert!(found.score.is_finite());
    }
}

use std::marker::PhantomData;

use crate::mosaicing::{TbMerge, TiePointMatch, TiePointOffset, auto_mosaic::AutoMosaicSearch};

use viprs_core::{
    error::ViprsError, format::BandFormat, image::Tile, shared_ops::sample_conv::ToF64,
};

/// Libvips-style top-bottom mosaic with automatic tie-point refinement.
///
/// # Examples
/// ```ignore
/// use viprs_ops_composite::mosaicing::tbmosaic::TbMosaicOp;
///
/// let op = TbMosaicOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct TbMosaicOp<F: BandFormat> {
    search: AutoMosaicSearch<F>,
    blend_width: u32,
    bands: u32,
    _format: PhantomData<F>,
}

impl<F: BandFormat> TbMosaicOp<F> {
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    /// Creates a new `TbMosaicOp`.
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
    pub const fn build_merge(&self, offset: TiePointOffset) -> TbMerge<F> {
        TbMerge::new(
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
    ) -> Result<(TiePointMatch, TbMerge<F>), ViprsError>
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

    fn run_merge(op: &TbMerge<U8>, reference: &[u8], secondary: &[u8]) -> Vec<u8> {
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

    fn rows(image: &[u8], image_width: usize, y: usize, height: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(image_width * height);
        for row in y..y + height {
            let start = row * image_width;
            out.extend_from_slice(&image[start..start + image_width]);
        }
        out
    }

    #[test]
    fn identical_images_find_zero_offset() {
        let pixels = patterned(6, 7, 29);
        let region = Region::new(0, 0, 6, 7);
        let reference = Tile::<U8>::new(region, 1, &pixels);
        let secondary = Tile::<U8>::new(region, 1, &pixels);
        let op = TbMosaicOp::<U8>::new(6, 7, 6, 7, 0, 0, 0, 0, 2, 3, 1);

        let found = op.detect_offset(&reference, &secondary).unwrap();

        assert_eq!(found.offset, TiePointOffset { dx: 0, dy: 0 });
    }

    #[test]
    fn shifted_pair_finds_expected_offset_exactly() {
        let base = patterned(8, 14, 31);
        let reference_pixels = crop(&base, 8, 0, 0, 7, 9);
        let secondary_pixels = crop(&base, 8, 1, 5, 7, 9);
        let reference = Tile::<U8>::new(Region::new(0, 0, 7, 9), 1, &reference_pixels);
        let secondary = Tile::<U8>::new(Region::new(0, 0, 7, 9), 1, &secondary_pixels);
        let op = TbMosaicOp::<U8>::new(7, 9, 7, 9, 0, 4, 0, 0, 2, 4, 1);
        let found = op.detect_offset(&reference, &secondary).unwrap();

        assert_eq!(found.offset, TiePointOffset { dx: -1, dy: -5 });
    }

    #[test]
    fn exact_overlap_delegates_to_tbmerge() {
        let base = patterned(8, 14, 31);
        let reference_pixels = crop(&base, 8, 0, 0, 7, 9);
        let secondary_pixels = crop(&base, 8, 1, 5, 7, 9);
        let reference = Tile::<U8>::new(Region::new(0, 0, 7, 9), 1, &reference_pixels);
        let secondary = Tile::<U8>::new(Region::new(0, 0, 7, 9), 1, &secondary_pixels);
        let op = TbMosaicOp::<U8>::new(7, 9, 7, 9, 1, 5, 0, 0, 2, 4, 1);

        let (found, merge) = op.detect_and_build_merge(&reference, &secondary).unwrap();

        assert_eq!(found.offset, TiePointOffset { dx: -1, dy: -5 });

        let stitched = run_merge(&merge, &reference_pixels, &secondary_pixels);
        let expected = compose_expected(
            8,
            14,
            &reference_pixels,
            7,
            9,
            &secondary_pixels,
            7,
            9,
            1,
            5,
        );
        assert_eq!(stitched, expected);
    }

    #[test]
    fn approximate_tie_point_keeps_tb_seam_aligned() {
        let base = patterned(8, 14, 31);
        let reference_pixels = crop(&base, 8, 0, 0, 7, 9);
        let secondary_pixels = crop(&base, 8, 1, 5, 7, 9);
        let reference = Tile::<U8>::new(Region::new(0, 0, 7, 9), 1, &reference_pixels);
        let secondary = Tile::<U8>::new(Region::new(0, 0, 7, 9), 1, &secondary_pixels);
        let op = TbMosaicOp::<U8>::new(7, 9, 7, 9, 0, 4, 0, 0, 2, 4, 1);

        let (found, merge) = op.detect_and_build_merge(&reference, &secondary).unwrap();
        let stitched = run_merge(&merge, &reference_pixels, &secondary_pixels);
        let expected = compose_expected(
            8,
            14,
            &reference_pixels,
            7,
            9,
            &secondary_pixels,
            7,
            9,
            1,
            5,
        );

        assert_eq!(found.offset, TiePointOffset { dx: -1, dy: -5 });
        assert_eq!(rows(&stitched, 8, 4, 4), rows(&expected, 8, 4, 4));
    }

    #[test]
    fn single_pixel_tiles_find_zero_offset() {
        let pixels = [211u8];
        let region = Region::new(0, 0, 1, 1);
        let reference = Tile::<U8>::new(region, 1, &pixels);
        let secondary = Tile::<U8>::new(region, 1, &pixels);
        let op = TbMosaicOp::<U8>::new(1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1);

        let found = op.detect_offset(&reference, &secondary).unwrap();

        assert_eq!(found.offset, TiePointOffset { dx: 0, dy: 0 });
        assert!(found.score.is_finite());
    }
}

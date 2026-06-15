use std::marker::PhantomData;

use crate::domain::{
    error::{MosaicingError, ViprsError},
    format::BandFormat,
    image::{Region, Tile},
    ops::{
        mosaicing::{TiePointMatch, TiePointSearchOp},
        resample::sample_conv::ToF64,
    },
};

pub(super) struct AutoMosaicSearch<F: BandFormat> {
    ref_width: u32,
    ref_height: u32,
    sec_width: u32,
    sec_height: u32,
    xref: i32,
    yref: i32,
    xsec: i32,
    ysec: i32,
    search_radius: u32,
    _format: PhantomData<F>,
}

impl<F: BandFormat> AutoMosaicSearch<F> {
    #[allow(clippy::too_many_arguments)]
    pub(super) const fn new(
        ref_width: u32,
        ref_height: u32,
        sec_width: u32,
        sec_height: u32,
        xref: i32,
        yref: i32,
        xsec: i32,
        ysec: i32,
        search_radius: u32,
    ) -> Self {
        Self {
            ref_width,
            ref_height,
            sec_width,
            sec_height,
            xref,
            yref,
            xsec,
            ysec,
            search_radius,
            _format: PhantomData,
        }
    }

    pub(super) fn detect_offset(
        &self,
        reference: &Tile<F>,
        secondary: &Tile<F>,
    ) -> Result<TiePointMatch, ViprsError>
    where
        F::Sample: ToF64,
    {
        let reference_world =
            Tile::<F>::new(self.reference_region(), reference.bands, reference.data);
        let secondary_world =
            Tile::<F>::new(self.secondary_region(), secondary.bands, secondary.data);
        let overlap = intersect_regions(reference_world.region, secondary_world.region).ok_or(
            MosaicingError::OverlapTooSmall {
                width: 0,
                height: 0,
                minimum_width: 1,
                minimum_height: 1,
            },
        )?;
        let minimum_overlap_pixels = overlap.pixel_count().min(16);
        let match_result = TiePointSearchOp::new(self.search_radius)
            .with_minimum_overlap(minimum_overlap_pixels)
            .search(&reference_world, &secondary_world, overlap)?;
        let actual_left = secondary_world.region.x + match_result.offset.dx;
        let actual_top = secondary_world.region.y + match_result.offset.dy;

        Ok(TiePointMatch {
            offset: crate::domain::ops::mosaicing::TiePointOffset {
                dx: -actual_left,
                dy: -actual_top,
            },
            score: match_result.score,
        })
    }

    pub(super) const fn reference_region(&self) -> Region {
        Region::new(0, 0, self.ref_width, self.ref_height)
    }

    pub(super) const fn secondary_region(&self) -> Region {
        Region::new(
            self.xref - self.xsec,
            self.yref - self.ysec,
            self.sec_width,
            self.sec_height,
        )
    }
}

fn intersect_regions(lhs: Region, rhs: Region) -> Option<Region> {
    let x0 = i64::from(lhs.x).max(i64::from(rhs.x));
    let y0 = i64::from(lhs.y).max(i64::from(rhs.y));
    let x1 = (i64::from(lhs.x) + i64::from(lhs.width)).min(i64::from(rhs.x) + i64::from(rhs.width));
    let y1 =
        (i64::from(lhs.y) + i64::from(lhs.height)).min(i64::from(rhs.y) + i64::from(rhs.height));
    if x1 <= x0 || y1 <= y0 {
        None
    } else {
        Some(Region::new(
            i32::try_from(x0).ok()?,
            i32::try_from(y0).ok()?,
            u32::try_from(x1 - x0).ok()?,
            u32::try_from(y1 - y0).ok()?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::format::U8;

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

    #[test]
    fn detect_offset_recovers_exact_shift_from_approximate_tie_points() {
        let base = patterned(14, 8, 23);
        let reference_pixels = crop(&base, 14, 0, 0, 9, 7);
        let secondary_pixels = crop(&base, 14, 5, 1, 9, 7);
        let reference = Tile::<U8>::new(Region::new(0, 0, 9, 7), 1, &reference_pixels);
        let secondary = Tile::<U8>::new(Region::new(0, 0, 9, 7), 1, &secondary_pixels);
        let search = AutoMosaicSearch::<U8>::new(9, 7, 9, 7, 4, 0, 0, 0, 2);

        let found = search.detect_offset(&reference, &secondary).unwrap();

        assert_eq!(
            found.offset,
            crate::domain::ops::mosaicing::TiePointOffset { dx: -5, dy: -1 }
        );
        assert!(found.score > 0.99);
    }

    #[test]
    fn detect_offset_rejects_non_overlapping_images() {
        let pixels = patterned(4, 4, 19);
        let region = Region::new(0, 0, 4, 4);
        let reference = Tile::<U8>::new(region, 1, &pixels);
        let secondary = Tile::<U8>::new(region, 1, &pixels);
        let search = AutoMosaicSearch::<U8>::new(4, 4, 4, 4, 8, 0, 0, 0, 2);

        let err = search.detect_offset(&reference, &secondary).unwrap_err();

        match err {
            ViprsError::Mosaicing(MosaicingError::OverlapTooSmall {
                width,
                height,
                minimum_width,
                minimum_height,
            }) => {
                assert_eq!(width, 0);
                assert_eq!(height, 0);
                assert_eq!(minimum_width, 1);
                assert_eq!(minimum_height, 1);
            }
            other => panic!("expected OverlapTooSmall, got {other:?}"),
        }
    }

    #[test]
    fn intersect_regions_handles_large_positive_origins_without_panicking() {
        let lhs = Region::new(i32::MAX - 2, i32::MAX - 2, 8, 8);
        let rhs = Region::new(i32::MAX - 1, i32::MAX - 1, 1, 1);

        let overlap = intersect_regions(lhs, rhs);

        assert_eq!(overlap, Some(Region::new(i32::MAX - 1, i32::MAX - 1, 1, 1)));
    }
}

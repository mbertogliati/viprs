use std::sync::Arc;

use crate::{
    domain::op::{NodeSpec, Op, PixelLocalOp},
    domain::{
        format::{BandFormat, PairMinMaxSample},
        image::{DemandHint, Region, Tile, TileMut},
    },
};

/// Element-wise minimum with a synchronized right-hand-side image.
///
/// The right-hand side is stored once as a full-image buffer. It may be either
/// single-band or match the input band count, matching libvips broadcast rules.
pub struct MinPair<F: BandFormat>
where
    F::Sample: PairMinMaxSample,
{
    rhs: Arc<[F::Sample]>,
    image_width: usize,
    rhs_bands: usize,
}

impl<F: BandFormat> MinPair<F>
where
    F::Sample: PairMinMaxSample,
{
    #[must_use]
    /// Creates a new `MinPair`.
    pub fn new<R>(rhs: R, image_width: u32, bands: u32) -> Self
    where
        R: Into<Arc<[F::Sample]>>,
    {
        Self {
            rhs: rhs.into(),
            image_width: image_width as usize,
            rhs_bands: bands as usize,
        }
    }
}

impl<F> Op for MinPair<F>
where
    F: BandFormat,
    F::Sample: PairMinMaxSample,
{
    type Input = F;
    type Output = F;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec::identity(tile_w, tile_h)
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, (): &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        debug_assert_eq!(input.data.len(), output.data.len());
        debug_assert_eq!(input.bands, output.bands);
        debug_assert!(input.region.x >= 0 && input.region.y >= 0);

        let output_bands = input.bands as usize;
        let rhs_bands = self.rhs_bands;
        let tile_width = input.region.width as usize;
        let tile_height = input.region.height as usize;
        let tile_row_samples = tile_width * output_bands;
        let rhs_row_samples = tile_width * rhs_bands;
        let rhs_row_stride = self.image_width * rhs_bands;
        let start_x = input.region.x as usize * rhs_bands;
        let start_y = input.region.y as usize;
        let required_rhs_len = start_y
            .checked_add(tile_height)
            .and_then(|rows| rows.checked_mul(rhs_row_stride))
            .unwrap_or(usize::MAX);

        debug_assert!(rhs_bands == 1 || rhs_bands == output_bands);
        debug_assert!(start_x <= rhs_row_stride);
        debug_assert!(rhs_row_samples <= rhs_row_stride.saturating_sub(start_x));
        debug_assert!(self.rhs.len() >= required_rhs_len);

        for row in 0..tile_height {
            let tile_offset = row * tile_row_samples;
            let rhs_offset = (start_y + row) * rhs_row_stride + start_x;
            let src_row = &input.data[tile_offset..tile_offset + tile_row_samples];
            let dst_row = &mut output.data[tile_offset..tile_offset + tile_row_samples];
            let rhs_row = &self.rhs[rhs_offset..rhs_offset + rhs_row_samples];

            if rhs_bands == output_bands {
                for ((sample, rhs), dst) in
                    src_row.iter().zip(rhs_row.iter()).zip(dst_row.iter_mut())
                {
                    *dst = sample.s_minpair(*rhs);
                }
            } else {
                for (pixel, rhs) in rhs_row.iter().copied().enumerate().take(tile_width) {
                    let src_base = pixel * output_bands;
                    for band in 0..output_bands {
                        let idx = src_base + band;
                        dst_row[idx] = src_row[idx].s_minpair(rhs);
                    }
                }
            }
        }
    }
}

impl<F> PixelLocalOp for MinPair<F>
where
    F: BandFormat,
    F::Sample: PairMinMaxSample,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        format::{F32, U8},
        image::Region,
    };
    use proptest::prelude::*;

    fn run_u8(op: &MinPair<U8>, input_data: &[u8], width: u32, height: u32, bands: u32) -> Vec<u8> {
        let region = Region::new(0, 0, width, height);
        let input = Tile::<U8>::new(region, bands, input_data);
        let mut output_data = vec![0u8; input_data.len()];
        let mut output = TileMut::<U8>::new(region, bands, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    fn run_f32(
        op: &MinPair<F32>,
        input_data: &[f32],
        width: u32,
        height: u32,
        bands: u32,
    ) -> Vec<f32> {
        let region = Region::new(0, 0, width, height);
        let input = Tile::<F32>::new(region, bands, input_data);
        let mut output_data = vec![0.0f32; input_data.len()];
        let mut output = TileMut::<F32>::new(region, bands, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    proptest! {
        #[test]
        fn minpair_identity_when_rhs_matches_input(pixels in proptest::collection::vec(any::<u8>(), 1..=128)) {
            let len = pixels.len();
            let op = MinPair::<U8>::new(pixels.clone(), len as u32, 1);
            let output = run_u8(&op, &pixels, len as u32, 1, 1);
            prop_assert_eq!(output, pixels);
        }

        #[test]
        fn minpair_matches_reference(
            (left, right) in (1usize..=64).prop_flat_map(|len| {
                (
                    proptest::collection::vec(-1000.0f32..=1000.0f32, len),
                    proptest::collection::vec(-1000.0f32..=1000.0f32, len),
                )
            })
        ) {
            let len = left.len();
            let op = MinPair::<F32>::new(right.clone(), len as u32, 1);
            let output = run_f32(&op, &left, len as u32, 1, 1);
            for ((actual, lhs), rhs) in output.iter().zip(left.iter()).zip(right.iter()) {
                prop_assert_eq!(*actual, lhs.min(*rhs));
            }
        }
    }

    #[test]
    fn minpair_boundary_prefers_smaller_extremes() {
        let op = MinPair::<U8>::new(vec![0, 255, 0, 255], 4, 1);
        let output = run_u8(&op, &[255, 0, 1, 254], 4, 1, 1);
        assert_eq!(output, vec![0, 0, 0, 254]);
    }

    #[test]
    fn minpair_single_band_rhs_broadcasts_across_bands() {
        let op = MinPair::<U8>::new(vec![5, 10], 2, 1);
        let output = run_u8(&op, &[1, 9, 20, 3], 2, 1, 2);
        assert_eq!(output, vec![1, 5, 10, 3]);
    }
}

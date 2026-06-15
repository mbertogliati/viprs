#![allow(clippy::needless_range_loop)]
// REASON: explicit index loops keep band-wise arithmetic aligned with the packed pixel layout.

use std::sync::Arc;

use crate::{
    domain::op::{NodeSpec, Op, PixelLocalOp},
    domain::{
        format::{BandFormat, DivSample},
        image::{DemandHint, Region, Tile, TileMut},
    },
};

/// Element-wise division of every pixel sample by a per-pixel divisor buffer.
///
/// For real formats, divide-by-zero yields zero, matching libvips. The rhs image
/// layout may be single-band and is then expanded across a multiband input tile.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::arithmetic::divide::Divide;
///
/// let op = Divide { /* operation parameters */ };
/// // Run `op` through a compiled image pipeline.
/// ```
#[allow(dead_code)]
pub struct Divide<F: BandFormat>
where
    F::Sample: DivSample,
{
    /// Pre-allocated full-image divisor values in row-major order.
    rhs: Arc<[F::Sample]>,
    image_width: usize,
    rhs_bands: usize,
}

#[allow(dead_code)]
impl<F: BandFormat> Divide<F>
where
    F::Sample: DivSample,
{
    #[must_use]
    /// Creates a new `Divide`.
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

impl<F: BandFormat> Op for Divide<F>
where
    F::Sample: DivSample,
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
        debug_assert_eq!(
            input.data.len(),
            output.data.len(),
            "Divide: input and output must have same length"
        );
        debug_assert_eq!(
            input.bands, output.bands,
            "Divide: input and output bands must match"
        );
        debug_assert!(
            input.region.x >= 0 && input.region.y >= 0,
            "Divide: rhs indexing requires in-bounds image coordinates"
        );

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
        debug_assert!(
            start_x <= rhs_row_stride,
            "Divide: tile x offset must fit within the rhs row stride"
        );
        debug_assert!(
            rhs_row_samples <= rhs_row_stride.saturating_sub(start_x),
            "Divide: tile row must fit within the rhs row stride"
        );
        debug_assert!(
            self.rhs.len() >= required_rhs_len,
            "Divide: rhs buffer must cover the requested tile rows"
        );
        debug_assert!(
            rhs_bands == 1 || rhs_bands == output_bands,
            "Divide: rhs must be single-band or match input/output band count"
        );

        for row in 0..tile_height {
            let tile_offset = row * tile_row_samples;
            let rhs_offset = (start_y + row) * rhs_row_stride + start_x;
            let src_row = &input.data[tile_offset..tile_offset + tile_row_samples];
            let dst_row = &mut output.data[tile_offset..tile_offset + tile_row_samples];
            let rhs_row = &self.rhs[rhs_offset..rhs_offset + rhs_row_samples];

            if rhs_bands == output_bands {
                for ((sample, divisor), dst) in
                    src_row.iter().zip(rhs_row.iter()).zip(dst_row.iter_mut())
                {
                    *dst = sample.s_div(*divisor);
                }
            } else {
                for pixel in 0..tile_width {
                    let divisor = rhs_row[pixel];
                    let src_base = pixel * output_bands;
                    for band in 0..output_bands {
                        let idx = src_base + band;
                        dst_row[idx] = src_row[idx].s_div(divisor);
                    }
                }
            }
        }
    }
}

/// `Divide` is pixel-local: output(x,y) depends only on input(x,y).
/// No neighbourhood access; identity tile geometry. See `PixelLocalOp` for invariants.
impl<F: BandFormat> PixelLocalOp for Divide<F> where F::Sample: DivSample {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        format::{F32, I32},
        image::Region,
    };
    use proptest::prelude::*;

    fn make_region(w: u32, h: u32) -> Region {
        Region::new(0, 0, w, h)
    }

    #[test]
    fn divide_f32_known_values() {
        let rhs = vec![2.0f32, 4.0, 1.0];
        let op = Divide::<F32>::new(rhs, 3, 1);
        let r = make_region(3, 1);
        let input_data = vec![4.0f32, 8.0, 3.0];
        let mut output_data = vec![0.0f32; 3];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert!((output_data[0] - 2.0).abs() < f32::EPSILON);
        assert!((output_data[1] - 2.0).abs() < f32::EPSILON);
        assert!((output_data[2] - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn divide_f32_by_one_is_identity() {
        let len = 4usize;
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let rhs = vec![1.0f32; len];
        let op = Divide::<F32>::new(rhs, len as u32, 1);
        let r = make_region(len as u32, 1);
        let mut output_data = vec![0.0f32; len];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        for (a, b) in input_data.iter().zip(output_data.iter()) {
            assert!((a - b).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn divide_i32_by_zero_returns_zero() {
        let op = Divide::<I32>::new(vec![0, 2, -1], 3, 1);
        let r = make_region(3, 1);
        let input_data = vec![7, 8, i32::MIN];
        let mut output_data = vec![0; 3];
        let input = Tile::<I32>::new(r, 1, &input_data);
        let mut output = TileMut::<I32>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, vec![0, 4, i32::MAX]);
    }

    #[test]
    fn divide_metadata_matches_identity_geometry() {
        let op = Divide::<F32>::new(vec![1.0, 1.0, 1.0], 3, 1);
        let region = make_region(3, 1);
        assert_eq!(op.demand_hint(), crate::domain::image::DemandHint::Any);
        assert_eq!(op.required_input_region(&region), region);
        assert_eq!(
            op.node_spec(3, 1),
            crate::domain::op::NodeSpec::identity(3, 1)
        );
    }

    #[test]
    fn divide_reads_rhs_from_non_zero_tile_offset() {
        let rhs = vec![1.0f32, 2.0, 4.0, 8.0, 16.0, 32.0];
        let op = Divide::<F32>::new(rhs, 3, 1);
        let region = Region::new(1, 0, 2, 1);
        let input_data = vec![32.0f32, 64.0];
        let mut output_data = vec![0.0f32; 2];
        let input = Tile::<F32>::new(region, 1, &input_data);
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
        let mut state = ();

        op.process_region(&mut state, &input, &mut output);

        assert_eq!(output_data, vec![16.0, 16.0]);
    }

    #[test]
    fn divide_single_band_rhs_expands_across_multiband_input() {
        let rhs = vec![2.0f32, 4.0];
        let op = Divide::<F32>::new(rhs, 2, 1);
        let region = Region::new(0, 0, 2, 1);
        let input_data = vec![8.0f32, 12.0, 16.0, 20.0, 24.0, 28.0];
        let mut output_data = vec![0.0f32; 6];
        let input = Tile::<F32>::new(region, 3, &input_data);
        let mut output = TileMut::<F32>::new(region, 3, &mut output_data);
        let mut state = ();

        op.process_region(&mut state, &input, &mut output);

        assert_eq!(output_data, vec![4.0, 6.0, 8.0, 5.0, 6.0, 7.0]);
    }

    proptest! {
        #[test]
        fn divide_f32_by_nonzero_consistent(
            pixels in proptest::collection::vec(1.0f32..=1000.0f32, 1..=32),
            divisors in proptest::collection::vec(1.0f32..=100.0f32, 1..=32),
        ) {
            let len = pixels.len().min(divisors.len());
            let pixels = &pixels[..len];
            let divisors = divisors[..len].to_vec();
            let op = Divide::<F32>::new(divisors.clone(), len as u32, 1);
            let r = Region::new(0, 0, len as u32, 1);
            let mut result = vec![0.0f32; len];
            {
                let input = Tile::<F32>::new(r, 1, pixels);
                let mut output = TileMut::<F32>::new(r, 1, &mut result);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
            }
            for ((p, d), r) in pixels.iter().zip(divisors.iter()).zip(result.iter()) {
                let expected = p / d;
                prop_assert!((r - expected).abs() < 1e-4, "{} / {} = {} expected {}", p, d, r, expected);
            }
        }
    }
}

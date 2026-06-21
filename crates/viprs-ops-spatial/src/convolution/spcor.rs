#![allow(dead_code)]
// REASON: cached statistics are reserved for upcoming normalized correlation refinements.
#![allow(clippy::needless_range_loop)]
// REASON: indexed loops keep the correlation window aligned with the packed tile layout.

use bytemuck::Pod;

use viprs_core::{
    error::ViprsError,
    format::{BandFormat, F32},
    image::{DemandHint, Image, Region, Tile, TileMut},
    op::{NodeSpec, Op},
};

use super::common::ToF64;

/// Normalized spatial correlation (`spcor`) against a fixed reference image.
///
/// # Examples
/// ```ignore
/// use viprs_ops_spatial::convolution::spcor::SpcorOp;
///
/// let op = SpcorOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct SpcorOp<F: BandFormat> {
    reference: Image<F>,
    centered_reference: Box<[f64]>,
    means: Box<[f64]>,
    norms: Box<[f64]>,
    ref_width: usize,
    ref_height: usize,
    ref_bands: usize,
    radius_x: u32,
    radius_y: u32,
}

impl<F: BandFormat> SpcorOp<F>
where
    F::Sample: ToF64,
{
    /// Creates a new `SpcorOp`.
    pub fn new(reference: Image<F>) -> Result<Self, ViprsError> {
        let ref_width = reference.width() as usize;
        let ref_height = reference.height() as usize;
        let ref_bands = reference.bands() as usize;

        if ref_width == 0 || ref_height == 0 || ref_bands == 0 {
            return Err(ViprsError::Codec(
                "SpcorOp: reference dimensions must be non-zero".to_owned(),
            ));
        }

        let area = (ref_width * ref_height) as f64;
        let mut means = vec![0.0f64; ref_bands];
        for band in 0..ref_bands {
            let mut sum = 0.0f64;
            for y in 0..ref_height {
                for x in 0..ref_width {
                    let idx = ((y * ref_width + x) * ref_bands) + band;
                    sum += reference.pixels()[idx].to_f64();
                }
            }
            means[band] = sum / area;
        }

        let mut centered_reference = vec![0.0f64; reference.pixels().len()];
        let mut norms = vec![0.0f64; ref_bands];
        for band in 0..ref_bands {
            let mut sum_sq = 0.0f64;
            for y in 0..ref_height {
                for x in 0..ref_width {
                    let idx = ((y * ref_width + x) * ref_bands) + band;
                    let centered = reference.pixels()[idx].to_f64() - means[band];
                    centered_reference[idx] = centered;
                    sum_sq = centered.mul_add(centered, sum_sq);
                }
            }
            norms[band] = sum_sq.sqrt();
        }

        Ok(Self {
            reference,
            centered_reference: centered_reference.into_boxed_slice(),
            means: means.into_boxed_slice(),
            norms: norms.into_boxed_slice(),
            ref_width,
            ref_height,
            ref_bands,
            radius_x: (ref_width / 2) as u32,
            radius_y: (ref_height / 2) as u32,
        })
    }

    /// Creates this value from buffer.
    pub fn from_buffer(
        reference: Vec<F::Sample>,
        ref_width: u32,
        ref_height: u32,
        ref_bands: u32,
    ) -> Result<Self, ViprsError> {
        Self::new(Image::from_buffer(
            ref_width, ref_height, ref_bands, reference,
        )?)
    }

    #[must_use]
    /// Returns or performs reference.
    pub const fn reference(&self) -> &Image<F> {
        &self.reference
    }
}

#[inline]
const fn bands_are_compatible(input_bands: usize, ref_bands: usize, output_bands: usize) -> bool {
    output_bands > 0
        && (input_bands == 1 || input_bands == output_bands)
        && (ref_bands == 1 || ref_bands == output_bands)
}

impl<F> Op for SpcorOp<F>
where
    F: BandFormat,
    F::Sample: ToF64 + Pod,
{
    type Input = F;
    type Output = F32;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::SmallTile
    }

    fn required_input_region(&self, output: &Region) -> Region {
        Region::new(
            output.x - self.radius_x as i32,
            output.y - self.radius_y as i32,
            output.width + 2 * self.radius_x,
            output.height + 2 * self.radius_y,
        )
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec {
            input_tile_w: tile_w + 2 * self.radius_x,
            input_tile_h: tile_h + 2 * self.radius_y,
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F32>) {
        let input_bands = input.bands as usize;
        let output_bands = output.bands as usize;
        if !bands_are_compatible(input_bands, self.ref_bands, output_bands) {
            debug_assert!(
                false,
                "SpcorOp band counts require each input to be one-band or match output bands"
            );
            return;
        }

        let out_w = output.region.width as usize;
        let out_h = output.region.height as usize;
        let in_w = input.region.width as usize;
        let area = (self.ref_width * self.ref_height) as f64;

        for oy in 0..out_h {
            for ox in 0..out_w {
                for band in 0..output_bands {
                    let input_band = if input_bands == 1 { 0 } else { band };
                    let ref_band = if self.ref_bands == 1 { 0 } else { band };
                    let mut sum = 0.0f64;

                    for ky in 0..self.ref_height {
                        for kx in 0..self.ref_width {
                            let idx = ((oy + ky) * in_w + ox + kx) * input_bands + input_band;
                            sum += input.data[idx].to_f64();
                        }
                    }

                    let mean = sum / area;
                    let mut variance = 0.0f64;
                    let mut cross = 0.0f64;
                    for ky in 0..self.ref_height {
                        for kx in 0..self.ref_width {
                            let input_idx = ((oy + ky) * in_w + ox + kx) * input_bands + input_band;
                            let ref_idx = ((ky * self.ref_width + kx) * self.ref_bands) + ref_band;
                            let centered = input.data[input_idx].to_f64() - mean;
                            variance = centered.mul_add(centered, variance);
                            cross = self.centered_reference[ref_idx].mul_add(centered, cross);
                        }
                    }

                    let denom = self.norms[ref_band] * variance.sqrt();
                    let out_idx = (oy * out_w + ox) * output_bands + band;
                    output.data[out_idx] = if denom == 0.0 {
                        0.0
                    } else {
                        (cross / denom).clamp(-1.0, 1.0) as f32
                    };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::F32,
        image::{Region, Tile, TileMut},
    };

    fn run_single(reference: &[f32], width: u32, height: u32, input: &[f32]) -> f32 {
        let op = SpcorOp::<F32>::from_buffer(reference.to_vec(), width, height, 1).unwrap();
        let input_tile = Tile::<F32>::new(Region::new(0, 0, width, height), 1, input);
        let mut output_data = vec![0.0f32; 1];
        let mut output = TileMut::<F32>::new(Region::new(0, 0, 1, 1), 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input_tile, &mut output);
        output_data[0]
    }

    #[test]
    fn correlate_image_with_itself_yields_one() {
        let reference = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert!((run_single(&reference, 3, 3, &reference) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn correlate_image_with_negated_self_yields_minus_one() {
        let reference = vec![1.0f32, 2.0, 3.0, 4.0];
        let negated = reference.iter().map(|value| -*value).collect::<Vec<_>>();
        assert!((run_single(&reference, 2, 2, &negated) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn constant_reference_reports_zero() {
        let reference = vec![4.0f32; 9];
        assert_eq!(run_single(&reference, 3, 3, &reference), 0.0);
    }

    #[test]
    fn metadata_expands_by_reference_radius() {
        let reference = Image::<F32>::from_buffer(5, 3, 1, vec![0.0; 15]).unwrap();
        let op = SpcorOp::<F32>::new(reference).unwrap();
        let output = Region::new(7, 9, 11, 13);

        assert_eq!(op.demand_hint(), DemandHint::SmallTile);
        assert_eq!(op.required_input_region(&output), Region::new(5, 8, 15, 15));
        assert_eq!(
            op.node_spec(11, 13),
            NodeSpec {
                input_tile_w: 15,
                input_tile_h: 15,
                output_tile_w: 11,
                output_tile_h: 13,
                coordinate_driven_source: None,
            }
        );
    }

    #[test]
    fn single_band_reference_is_applied_to_each_input_band() {
        let op = SpcorOp::<F32>::from_buffer(vec![1.0, 2.0, 3.0, 4.0], 2, 2, 1).unwrap();
        let input_data = vec![
            1.0f32, 10.0, 2.0, 20.0, //
            3.0, 30.0, 4.0, 40.0,
        ];
        let input = Tile::<F32>::new(Region::new(0, 0, 2, 2), 2, &input_data);
        let mut output_data = vec![0.0f32; 2];
        let mut output = TileMut::<F32>::new(Region::new(0, 0, 1, 1), 2, &mut output_data);
        let mut state = ();

        op.process_region(&mut state, &input, &mut output);

        assert!((output_data[0] - 1.0).abs() < 1e-6);
        assert!((output_data[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(
        expected = "SpcorOp band counts require each input to be one-band or match output bands"
    )]
    fn band_mismatch_panics_in_debug_builds() {
        let op = SpcorOp::<F32>::from_buffer(vec![1.0, 2.0, 3.0, 4.0], 2, 2, 1).unwrap();
        let input_data = vec![1.0f32; 12];
        let input = Tile::<F32>::new(Region::new(0, 0, 2, 2), 3, &input_data);
        let mut output_data = vec![9.0f32; 2];
        let mut output = TileMut::<F32>::new(Region::new(0, 0, 1, 1), 2, &mut output_data);
        let mut state = ();

        op.process_region(&mut state, &input, &mut output);
    }

    proptest! {
        #[test]
        fn random_patch_correlates_with_itself(
            values in prop::collection::vec(-10.0f32..10.0f32, 9),
        ) {
            let mean = values.iter().copied().sum::<f32>() / values.len() as f32;
            let variance = values
                .iter()
                .map(|value| {
                    let centered = *value - mean;
                    centered * centered
                })
                .sum::<f32>();
            prop_assume!(variance > 1e-4);

            let score = run_single(&values, 3, 3, &values);
            prop_assert!((score - 1.0).abs() < 1e-4, "got {score}");
        }

        #[test]
        fn single_pixel_reference_is_uncorrelated(value in -1_000.0f32..1_000.0f32) {
            let score = run_single(&[value], 1, 1, &[value]);
            prop_assert_eq!(score, 0.0);
        }
    }
}

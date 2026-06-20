use std::marker::PhantomData;

use bytemuck::Pod;

use viprs_core::{
    error::ViprsError,
    format::{BandFormat, F32},
    image::{DemandHint, Region, Tile, TileMut},
    op::{NodeSpec, Op},
};

use super::common::ToF64;

/// Constant value for prewitt compass mask.
pub const PREWITT_COMPASS_MASK: [f32; 9] = [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0];

/// Rotating compass filter: convolve with a mask in successive 45° rotations and keep the max response.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::convolution::compass::CompassOp;
///
/// let op = CompassOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct CompassOp<F: BandFormat> {
    /// Stores the `mask` value for this item.
    pub mask: &'static [f32],
    /// Stores the `times` value for this item.
    pub times: u32,
    side: usize,
    radius: u32,
    rotated_masks: Box<[Box<[f64]>]>,
    _phantom: PhantomData<F>,
}

impl<F: BandFormat> CompassOp<F> {
    /// Creates a new `CompassOp`.
    pub fn new(mask: &'static [f32], times: u32) -> Result<Self, ViprsError> {
        let side = (mask.len() as f64).sqrt() as usize;
        if side * side != mask.len() || side == 0 || side.is_multiple_of(2) {
            return Err(ViprsError::Codec(
                "CompassOp: mask must be a non-empty odd square kernel".to_owned(),
            ));
        }
        let times = times.max(1);
        let rotated_masks = (0..times)
            .map(|step| rotate_kernel(mask, side, step).into_boxed_slice())
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Ok(Self {
            mask,
            times,
            side,
            radius: (side / 2) as u32,
            rotated_masks,
            _phantom: PhantomData,
        })
    }
}

#[inline]
fn rotate_kernel(mask: &'static [f32], side: usize, step: u32) -> Vec<f64> {
    let angle = f64::from(step) * std::f64::consts::FRAC_PI_4;
    let cos = angle.cos();
    let sin = angle.sin();
    let center = (side / 2) as f64;
    let mut rotated = vec![0.0f64; side * side];

    for y in 0..side {
        for x in 0..side {
            let dx = x as f64 - center;
            let dy = y as f64 - center;
            let rx = (dx * cos - dy * sin).round() + center;
            let ry = (dx * sin + dy * cos).round() + center;
            let dst_x = rx.clamp(0.0, (side - 1) as f64) as usize;
            let dst_y = ry.clamp(0.0, (side - 1) as f64) as usize;
            rotated[dst_y * side + dst_x] += f64::from(mask[y * side + x]);
        }
    }

    rotated
}

impl<F> Op for CompassOp<F>
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
            output.x - self.radius as i32,
            output.y - self.radius as i32,
            output.width + 2 * self.radius,
            output.height + 2 * self.radius,
        )
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec {
            input_tile_w: tile_w + 2 * self.radius,
            input_tile_h: tile_h + 2 * self.radius,
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F32>) {
        let out_w = output.region.width as usize;
        let out_h = output.region.height as usize;
        let in_w = input.region.width as usize;
        let bands = input.bands as usize;

        for oy in 0..out_h {
            for ox in 0..out_w {
                for band in 0..bands {
                    let mut best = 0.0f64;
                    for kernel in &self.rotated_masks {
                        let mut response = 0.0f64;
                        for ky in 0..self.side {
                            for kx in 0..self.side {
                                let idx = ((oy + ky) * in_w + ox + kx) * bands + band;
                                response = input.data[idx]
                                    .to_f64()
                                    .mul_add(kernel[ky * self.side + kx], response);
                            }
                        }
                        best = best.max(response.abs());
                    }
                    let out_idx = (oy * out_w + ox) * bands + band;
                    output.data[out_idx] = best as f32;
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

    #[test]
    fn flat_image_has_zero_gradients() {
        let op = CompassOp::<F32>::new(&PREWITT_COMPASS_MASK, 8).unwrap();
        let in_region = Region::new(0, 0, 5, 5);
        let out_region = Region::new(0, 0, 3, 3);
        let input_data = vec![4.0f32; 25];
        let input = Tile::<F32>::new(in_region, 1, &input_data);
        let mut output_data = vec![1.0f32; 9];
        let mut output = TileMut::<F32>::new(out_region, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert!(output_data.iter().all(|value| value.abs() < 1e-6));
    }

    #[test]
    fn rotate_kernel_step_zero_keeps_original_layout() {
        let rotated = rotate_kernel(&PREWITT_COMPASS_MASK, 3, 0);
        let expected: Vec<f64> = PREWITT_COMPASS_MASK
            .iter()
            .map(|value| f64::from(*value))
            .collect();
        assert_eq!(rotated, expected);
    }

    #[test]
    fn constructor_rejects_invalid_masks_and_clamps_zero_rotations() {
        assert!(CompassOp::<F32>::new(&[], 8).is_err());
        assert!(CompassOp::<F32>::new(&[1.0, 2.0], 8).is_err());
        assert!(CompassOp::<F32>::new(&[1.0, 2.0, 3.0, 4.0], 8).is_err());

        let op = CompassOp::<F32>::new(&PREWITT_COMPASS_MASK, 0).unwrap();
        assert_eq!(op.times, 1);
        assert_eq!(op.rotated_masks.len(), 1);
    }

    #[test]
    fn metadata_expands_regions_by_kernel_radius() {
        let op = CompassOp::<F32>::new(&PREWITT_COMPASS_MASK, 8).unwrap();
        let output = Region::new(2, 3, 4, 5);

        assert_eq!(op.demand_hint(), DemandHint::SmallTile);
        assert_eq!(op.required_input_region(&output), Region::new(1, 2, 6, 7));
        assert_eq!(
            op.node_spec(4, 5),
            NodeSpec {
                input_tile_w: 6,
                input_tile_h: 7,
                output_tile_w: 4,
                output_tile_h: 5,
                coordinate_driven_source: None,
            }
        );
    }

    #[test]
    fn process_region_handles_multiple_bands() {
        let op = CompassOp::<F32>::new(&PREWITT_COMPASS_MASK, 4).unwrap();
        let in_region = Region::new(0, 0, 3, 3);
        let out_region = Region::new(0, 0, 1, 1);
        let input_data = vec![
            0.0f32, 0.0, 1.0, 2.0, 2.0, 4.0, //
            0.0, 0.0, 1.0, 2.0, 2.0, 4.0, //
            0.0, 0.0, 1.0, 2.0, 2.0, 4.0,
        ];
        let input = Tile::<F32>::new(in_region, 2, &input_data);
        let mut output_data = vec![0.0f32; 2];
        let mut output = TileMut::<F32>::new(out_region, 2, &mut output_data);
        let mut state = ();

        op.process_region(&mut state, &input, &mut output);

        assert!(output_data[0] > 0.0);
        assert!(output_data[1] > output_data[0]);
    }

    proptest! {
        #[test]
        fn uniform_input_stays_zero(
            width in 1usize..5,
            height in 1usize..5,
            value in any::<f32>(),
        ) {
            let op = CompassOp::<F32>::new(&PREWITT_COMPASS_MASK, 8).unwrap();
            let in_region = Region::new(0, 0, (width + 2) as u32, (height + 2) as u32);
            let out_region = Region::new(0, 0, width as u32, height as u32);
            let input_data = vec![value; (width + 2) * (height + 2)];
            let input = Tile::<F32>::new(in_region, 1, &input_data);
            let mut output_data = vec![1.0f32; width * height];
            let mut output = TileMut::<F32>::new(out_region, 1, &mut output_data);
            let mut state = ();
            op.process_region(&mut state, &input, &mut output);
            prop_assert!(output_data.iter().all(|sample| sample.abs() < 1e-5));
        }
    }
}

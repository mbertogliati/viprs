use std::marker::PhantomData;

use bytemuck::Pod;

use viprs_core::{
    format::{BandFormat, BandFormatId, F32},
    image::{DemandHint, Region, Tile, TileMut},
    op::{NodeSpec, Op},
};

use super::common::{ToF64, convolve_mask3_at};

const SOBEL_X: [[f64; 3]; 3] = [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]];
const SOBEL_Y: [[f64; 3]; 3] = [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]];

/// Sobel gradient magnitude.
pub struct Sobel<F: BandFormat> {
    _format: PhantomData<F>,
}

impl<F: BandFormat> Sobel<F> {
    #[must_use]
    /// Creates a new `Sobel`.
    pub const fn new() -> Self {
        Self {
            _format: PhantomData,
        }
    }
}

impl<F: BandFormat> Default for Sobel<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> Op for Sobel<F>
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
            output.x - 1,
            output.y - 1,
            output.width + 2,
            output.height + 2,
        )
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec {
            input_tile_w: tile_w + 2,
            input_tile_h: tile_h + 2,
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    fn start(&self) -> Self::State {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F32>) {
        let out_w = output.region.width as usize;
        let out_h = output.region.height as usize;
        let in_w = input.region.width as usize;
        let bands = input.bands as usize;

        for oy in 0..out_h {
            for ox in 0..out_w {
                let x = ox + 1;
                let y = oy + 1;
                for band in 0..bands {
                    let gx = convolve_mask3_at(input, in_w, bands, x, y, band, &SOBEL_X);
                    let gy = convolve_mask3_at(input, in_w, bands, x, y, band, &SOBEL_Y);
                    let magnitude = if F::ID == BandFormatId::U8 {
                        (gx.abs() + gy.abs()).min(255.0)
                    } else {
                        gx.hypot(gy)
                    } as f32;
                    let out_idx = (oy * out_w + ox) * bands + band;
                    output.data[out_idx] = magnitude;
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
        format::{F32, U8},
        image::{Region, Tile, TileMut},
    };

    proptest! {
        #[test]
        fn zero_input_stays_zero(width in 1usize..5, height in 1usize..5) {
            let op = Sobel::<F32>::new();
            let in_region = Region::new(0, 0, (width + 2) as u32, (height + 2) as u32);
            let out_region = Region::new(0, 0, width as u32, height as u32);
            let input_data = vec![0.0f32; (width + 2) * (height + 2)];
            let input = Tile::<F32>::new(in_region, 1, &input_data);
            let mut output_data = vec![1.0f32; width * height];
            let mut output = TileMut::<F32>::new(out_region, 1, &mut output_data);
            let mut state = ();

            op.process_region(&mut state, &input, &mut output);

            prop_assert!(output_data.iter().all(|value| value.abs() < 1e-6));
        }
    }

    #[test]
    fn constant_field_has_no_edges() {
        let op = Sobel::<F32>::new();
        let in_region = Region::new(0, 0, 5, 5);
        let out_region = Region::new(0, 0, 3, 3);
        let input_data = vec![7.0f32; 25];
        let input = Tile::<F32>::new(in_region, 1, &input_data);
        let mut output_data = vec![0.0f32; 9];
        let mut output = TileMut::<F32>::new(out_region, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert!(output_data.iter().all(|value| value.abs() < 1e-6));
    }

    #[test]
    fn sobel_metadata_expands_input_by_one_pixel() {
        let op = Sobel::<F32>::default();
        let out_region = Region::new(5, 7, 4, 3);
        assert_eq!(op.demand_hint(), viprs_core::image::DemandHint::SmallTile);
        assert_eq!(
            op.required_input_region(&out_region),
            Region::new(4, 6, 6, 5)
        );
        let spec = op.node_spec(4, 3);
        assert_eq!(spec.input_tile_w, 6);
        assert_eq!(spec.input_tile_h, 5);
        assert_eq!(spec.output_tile_w, 4);
        assert_eq!(spec.output_tile_h, 3);
    }

    #[test]
    fn u8_input_uses_libvips_l1_magnitude() {
        let op = Sobel::<U8>::new();
        let in_region = Region::new(0, 0, 3, 3);
        let out_region = Region::new(0, 0, 1, 1);
        let input_data = vec![0u8, 0, 0, 0, 255, 255, 0, 255, 255];
        let input = Tile::<U8>::new(in_region, 1, &input_data);
        let mut output_data = vec![0.0f32; 1];
        let mut output = TileMut::<F32>::new(out_region, 1, &mut output_data);

        op.process_region(&mut (), &input, &mut output);

        let gx = convolve_mask3_at(&input, 3, 1, 1, 1, 0, &SOBEL_X);
        let gy = convolve_mask3_at(&input, 3, 1, 1, 1, 0, &SOBEL_Y);
        let expected = (gx.abs() + gy.abs()).min(255.0) as f32;
        assert_eq!(output_data[0], expected);
    }
}

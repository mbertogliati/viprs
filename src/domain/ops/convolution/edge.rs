use std::marker::PhantomData;

use bytemuck::Pod;

use crate::{
    domain::op::{NodeSpec, Op},
    domain::{
        format::{BandFormat, U8},
        image::{DemandHint, Region, Tile, TileMut},
    },
};

use super::common::{ToF64, convolve_mask3_at};

/// Constant value for sobel edge mask.
pub const SOBEL_EDGE_MASK: [[f64; 3]; 3] = [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]];
/// Constant value for scharr edge mask.
pub const SCHARR_EDGE_MASK: [[f64; 3]; 3] =
    [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]];
/// Constant value for prewitt edge mask.
pub const PREWITT_EDGE_MASK: [[f64; 3]; 3] = [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]];

/// Shared libvips `edge` primitive for 3×3 gradient masks.
///
/// This follows the accurate float path from libvips `edge.c`: convolve with
/// the supplied mask and the mask rotated by 90 degrees, then write the gradient
/// magnitude, then cast to U8 like libvips `edge`.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::convolution::edge::EdgeOp;
///
/// let op = EdgeOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct EdgeOp<F: BandFormat> {
    mask: [[f64; 3]; 3],
    rotated: [[f64; 3]; 3],
    _format: PhantomData<F>,
}

impl<F: BandFormat> EdgeOp<F> {
    #[must_use]
    /// Creates a new `EdgeOp`.
    pub const fn new(mask: [[f64; 3]; 3]) -> Self {
        Self {
            rotated: rotate90(mask),
            mask,
            _format: PhantomData,
        }
    }

    #[must_use]
    /// Returns or performs sobel.
    pub const fn sobel() -> Self {
        Self::new(SOBEL_EDGE_MASK)
    }

    #[must_use]
    /// Returns or performs scharr.
    pub const fn scharr() -> Self {
        Self::new(SCHARR_EDGE_MASK)
    }

    #[must_use]
    /// Returns or performs prewitt.
    pub const fn prewitt() -> Self {
        Self::new(PREWITT_EDGE_MASK)
    }
}

impl<F: BandFormat> Default for EdgeOp<F> {
    fn default() -> Self {
        Self::sobel()
    }
}

const fn rotate90(mask: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [mask[2][0], mask[1][0], mask[0][0]],
        [mask[2][1], mask[1][1], mask[0][1]],
        [mask[2][2], mask[1][2], mask[0][2]],
    ]
}

impl<F> Op for EdgeOp<F>
where
    F: BandFormat,
    F::Sample: ToF64 + Pod,
{
    type Input = F;
    type Output = U8;
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
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<U8>) {
        let out_w = output.region.width as usize;
        let out_h = output.region.height as usize;
        let in_w = input.region.width as usize;
        let bands = input.bands as usize;

        for oy in 0..out_h {
            for ox in 0..out_w {
                let x = ox + 1;
                let y = oy + 1;
                for band in 0..bands {
                    let gx = convolve_mask3_at(input, in_w, bands, x, y, band, &self.mask);
                    let gy = convolve_mask3_at(input, in_w, bands, x, y, band, &self.rotated);
                    let out_idx = (oy * out_w + ox) * bands + band;
                    output.data[out_idx] =
                        gx.hypot(gy).clamp(0.0, f64::from(u8::MAX)).round() as u8;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        format::{F32, U16},
        image::{Region, Tile, TileMut},
    };
    use proptest::prelude::*;

    fn run_edge<F>(
        op: &EdgeOp<F>,
        input_data: &[F::Sample],
        input_region: Region,
        output_region: Region,
        bands: u32,
    ) -> Vec<u8>
    where
        F: BandFormat,
        F::Sample: ToF64 + Pod,
    {
        let mut output = vec![0u8; output_region.pixel_count() * bands as usize];
        let input = Tile::<F>::new(input_region, bands, input_data);
        let mut output_tile = TileMut::<U8>::new(output_region, bands, &mut output);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output_tile);
        output
    }

    #[test]
    fn metadata_expands_input_by_one_pixel() {
        let op = EdgeOp::<F32>::sobel();
        fn output_is_u8<O: Op<Input = F32, Output = U8>>(_: &O) {}
        output_is_u8(&op);
        let output = Region::new(5, 7, 4, 3);
        assert_eq!(op.demand_hint(), DemandHint::SmallTile);
        assert_eq!(op.required_input_region(&output), Region::new(4, 6, 6, 5));
        let spec = op.node_spec(4, 3);
        assert_eq!(spec.input_tile_w, 6);
        assert_eq!(spec.input_tile_h, 5);
        assert_eq!(spec.output_tile_w, 4);
        assert_eq!(spec.output_tile_h, 3);
    }

    #[test]
    fn zero_mask_is_noop_boundary_case() {
        let op = EdgeOp::<U16>::new([[0.0; 3]; 3]);
        let input = vec![u16::MAX; 25 * 2];
        let output = run_edge(
            &op,
            &input,
            Region::new(0, 0, 5, 5),
            Region::new(0, 0, 3, 3),
            2,
        );
        assert!(output.iter().all(|sample| *sample == 0));
    }

    proptest! {
        #[test]
        fn constant_field_has_no_edges(
            width in 1usize..5,
            height in 1usize..5,
            value in -100.0f32..100.0,
        ) {
            let op = EdgeOp::<F32>::prewitt();
            let input_region = Region::new(0, 0, (width + 2) as u32, (height + 2) as u32);
            let output_region = Region::new(0, 0, width as u32, height as u32);
            let input = vec![value; (width + 2) * (height + 2)];
            let output = run_edge(&op, &input, input_region, output_region, 1);

            prop_assert!(output.iter().all(|sample| *sample == 0));
        }
    }
}

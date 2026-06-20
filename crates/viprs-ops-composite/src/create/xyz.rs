use std::marker::PhantomData;

use viprs_core::{
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
    shared_ops::sample_conv::FromF64,
};

/// Generate an `(x, y, 0)` coordinate image.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::create::xyz::XyzOp;
///
/// let op = XyzOp { /* operation parameters */ };
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Clone, Copy, Debug)]
pub struct XyzOp<F: BandFormat> {
    width: u32,
    height: u32,
    _format: PhantomData<F>,
}

impl<F: BandFormat> XyzOp<F> {
    #[must_use]
    /// Creates a new `XyzOp`.
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            _format: PhantomData,
        }
    }
}

impl<F> Op for XyzOp<F>
where
    F: BandFormat,
    F::Sample: FromF64,
{
    type Input = F;
    type Output = F;
    type State = ();

    const OUTPUT_BANDS: Option<usize> = Some(3);

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), _input: &Tile<F>, output: &mut TileMut<F>) {
        debug_assert_eq!(output.bands, 3, "XyzOp output must be three-band");
        debug_assert!(output.region.x >= 0 && output.region.y >= 0);
        debug_assert!(output.region.x as u32 + output.region.width <= self.width);
        debug_assert!(output.region.y as u32 + output.region.height <= self.height);

        let region_width = output.region.width as usize;
        for row in 0..output.region.height as usize {
            let y = output.region.y as u32 + row as u32;
            for col in 0..region_width {
                let x = output.region.x as u32 + col as u32;
                let pixel_base = (row * region_width + col) * 3;
                output.data[pixel_base] = F::Sample::from_f64(f64::from(x));
                output.data[pixel_base + 1] = F::Sample::from_f64(f64::from(y));
                output.data[pixel_base + 2] = F::Sample::from_f64(0.0);
            }
        }
    }
}

impl<F> PixelLocalOp for XyzOp<F>
where
    F: BandFormat,
    F::Sample: FromF64,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::U16,
        image::{Region, Tile, TileMut},
    };

    fn run_op(width: u32, height: u32) -> Vec<u16> {
        let op = XyzOp::<U16>::new(width, height);
        let region = Region::new(0, 0, width, height);
        let len = width as usize * height as usize * 3;
        let input_data = vec![0u16; len];
        let mut output_data = vec![0u16; len];
        let input = Tile::<U16>::new(region, 3, &input_data);
        let mut output = TileMut::<U16>::new(region, 3, &mut output_data);
        op.process_region(&mut (), &input, &mut output);
        output_data
    }

    #[test]
    fn xyz_op_encodes_coordinates() {
        let samples = run_op(3, 2);
        assert_eq!(
            samples,
            vec![0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 1, 0, 1, 1, 0, 2, 1, 0]
        );
    }

    #[test]
    fn xyz_op_single_pixel_is_origin() {
        assert_eq!(run_op(1, 1), vec![0, 0, 0]);
    }

    #[test]
    fn partial_region_uses_absolute_coordinates() {
        let op = XyzOp::<U16>::new(5, 4);
        let input_region = Region::new(0, 0, 5, 4);
        let output_region = Region::new(2, 1, 2, 2);
        let input_data = vec![0u16; input_region.pixel_count() * 3];
        let mut output_data = vec![0u16; output_region.pixel_count() * 3];
        let input = Tile::<U16>::new(input_region, 3, &input_data);
        let mut output = TileMut::<U16>::new(output_region, 3, &mut output_data);

        op.process_region(&mut (), &input, &mut output);

        assert_eq!(output_data, vec![2, 1, 0, 3, 1, 0, 2, 2, 0, 3, 2, 0]);
    }

    #[test]
    fn xyz_op_reports_identity_geometry_contract() {
        let op = XyzOp::<U16>::new(5, 4);
        let region = Region::new(1, 2, 3, 1);

        assert_eq!(op.demand_hint(), DemandHint::Any);
        assert_eq!(op.required_input_region(&region), region);
        op.start();
    }

    proptest! {
        #[test]
        fn prop_xyz_matches_pixel_coordinates(width in 1u32..=12, height in 1u32..=12) {
            let samples = run_op(width, height);
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let pixel_base = (y * width as usize + x) * 3;
                    prop_assert_eq!(samples[pixel_base], x as u16);
                    prop_assert_eq!(samples[pixel_base + 1], y as u16);
                    prop_assert_eq!(samples[pixel_base + 2], 0u16);
                }
            }
        }
    }
}

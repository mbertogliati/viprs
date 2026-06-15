use std::marker::PhantomData;

use crate::domain::{
    error::ViprsError,
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
    ops::resample::sample_conv::FromF64,
};

/// Generate a Mandelbrot fractal image over the standard libvips viewport.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::create::mandelbrot::MandelbrotOp;
///
/// let op = MandelbrotOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug)]
pub struct MandelbrotOp<F: BandFormat> {
    width: u32,
    height: u32,
    max_iter: u32,
    _format: PhantomData<F>,
}

impl<F: BandFormat> Copy for MandelbrotOp<F> {}

impl<F: BandFormat> Clone for MandelbrotOp<F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: BandFormat> MandelbrotOp<F> {
    /// Creates a new `MandelbrotOp`.
    pub fn new(width: u32, height: u32, max_iter: u32) -> Result<Self, ViprsError> {
        if width == 0 || height == 0 {
            return Err(ViprsError::Scheduler(format!(
                "MandelbrotOp width and height must be > 0, got {width}x{height}"
            )));
        }
        if max_iter == 0 {
            return Err(ViprsError::Scheduler(
                "MandelbrotOp max_iter must be > 0".to_owned(),
            ));
        }

        Ok(Self {
            width,
            height,
            max_iter,
            _format: PhantomData,
        })
    }

    #[must_use]
    /// Returns or performs width.
    pub const fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    /// Returns or performs height.
    pub const fn height(&self) -> u32 {
        self.height
    }
}

impl<F> Op for MandelbrotOp<F>
where
    F: BandFormat,
    F::Sample: FromF64,
{
    type Input = F;
    type Output = F;
    type State = ();

    const OUTPUT_BANDS: Option<usize> = Some(1);

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), _input: &Tile<F>, output: &mut TileMut<F>) {
        debug_assert_eq!(output.bands, 1, "MandelbrotOp output must be single-band");
        debug_assert!(output.region.x >= 0 && output.region.y >= 0);
        debug_assert!(output.region.x as u32 + output.region.width <= self.width);
        debug_assert!(output.region.y as u32 + output.region.height <= self.height);

        let region_width = output.region.width as usize;
        let width = f64::from(self.width);
        let height = f64::from(self.height);

        for row in 0..output.region.height as usize {
            let py = output.region.y as u32 + row as u32;
            let cy = (f64::from(py) / height).mul_add(2.0, -1.0);

            for col in 0..region_width {
                let px = output.region.x as u32 + col as u32;
                let cx = (f64::from(px) / width).mul_add(3.5, -2.5);

                let mut zx = 0.0f64;
                let mut zy = 0.0f64;
                let mut iter = 0u32;

                while iter < self.max_iter && zy.mul_add(zy, zx * zx) <= 4.0 {
                    let next_zx = zy.mul_add(-zy, zx * zx) + cx;
                    zy = (2.0 * zx).mul_add(zy, cy);
                    zx = next_zx;
                    iter += 1;
                }

                let value = (f64::from(iter) / f64::from(self.max_iter)) * 255.0;
                output.data[row * region_width + col] = F::Sample::from_f64(value);
            }
        }
    }
}

impl<F> PixelLocalOp for MandelbrotOp<F>
where
    F: BandFormat,
    F::Sample: FromF64,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        format::{F32, U8},
        image::{Region, Tile, TileMut},
    };
    use proptest::prelude::*;

    fn render(width: u32, height: u32, max_iter: u32) -> Vec<u8> {
        let op = MandelbrotOp::<U8>::new(width, height, max_iter).unwrap();
        let region = Region::new(0, 0, width, height);
        let input_data = vec![0u8; region.pixel_count()];
        let mut output_data = vec![0u8; region.pixel_count()];
        let input = Tile::<U8>::new(region, 1, &input_data);
        let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);
        output_data
    }

    fn render_f32(width: u32, height: u32, max_iter: u32) -> Vec<f32> {
        let op = MandelbrotOp::<F32>::new(width, height, max_iter).unwrap();
        let region = Region::new(0, 0, width, height);
        let input_data = vec![0.0f32; region.pixel_count()];
        let mut output_data = vec![0.0f32; region.pixel_count()];
        let input = Tile::<F32>::new(region, 1, &input_data);
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);
        output_data
    }

    #[test]
    fn constructor_rejects_zero_dimensions_and_iterations() {
        assert!(MandelbrotOp::<U8>::new(0, 8, 32).is_err());
        assert!(MandelbrotOp::<U8>::new(8, 0, 32).is_err());
        assert!(MandelbrotOp::<U8>::new(8, 8, 0).is_err());
    }

    #[test]
    fn interior_point_survives_longer_than_corner() {
        let samples = render(100, 100, 64);
        let corner = samples[0];
        let interior = samples[50 * 100 + 71];

        assert!(interior >= corner);
        assert_eq!(interior, 255);
    }

    #[test]
    fn accessors_report_requested_geometry() {
        let op = MandelbrotOp::<U8>::new(7, 9, 32).unwrap();
        assert_eq!(op.width(), 7);
        assert_eq!(op.height(), 9);
        assert_eq!(op.demand_hint(), DemandHint::Any);
    }

    #[test]
    fn partial_region_matches_full_render_slice() {
        let op = MandelbrotOp::<F32>::new(8, 6, 32).unwrap();
        let full = render_f32(8, 6, 32);
        let input_region = Region::new(0, 0, op.width(), op.height());
        let output_region = Region::new(2, 1, 3, 2);
        let input_data = vec![0.0f32; input_region.pixel_count()];
        let mut output_data = vec![0.0f32; output_region.pixel_count()];
        let input = Tile::<F32>::new(input_region, 1, &input_data);
        let mut output = TileMut::<F32>::new(output_region, 1, &mut output_data);

        op.process_region(&mut (), &input, &mut output);

        assert_eq!(
            output_data,
            vec![
                full[8 + 2],
                full[8 + 3],
                full[8 + 4],
                full[16 + 2],
                full[16 + 3],
                full[16 + 4],
            ]
        );
    }

    proptest! {
        #[test]
        fn prop_output_has_expected_dimensions_and_range(
            width in 1u32..=32,
            height in 1u32..=32,
            max_iter in 1u32..=256,
        ) {
            let samples = render_f32(width, height, max_iter);
            prop_assert_eq!(samples.len(), width as usize * height as usize);
            prop_assert!(samples.iter().all(|sample| (0.0..=255.0).contains(sample)));
        }
    }
}

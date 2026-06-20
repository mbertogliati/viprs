use std::marker::PhantomData;

use viprs_core::{
    error::ViprsError,
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
    shared_ops::sample_conv::FromF64,
};

/// Generate the classic eye spatial-frequency test chart.
///
/// # Examples
/// ```ignore
/// use viprs_ops_composite::create::eye::EyeOp;
///
/// let op = EyeOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug)]
pub struct EyeOp<F: BandFormat> {
    width: u32,
    height: u32,
    factor: f64,
    uchar: bool,
    _format: PhantomData<F>,
}

impl<F: BandFormat> Copy for EyeOp<F> {}

impl<F: BandFormat> Clone for EyeOp<F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: BandFormat> EyeOp<F> {
    /// Creates a new `EyeOp`.
    pub fn new(width: u32, height: u32, factor: f64) -> Result<Self, ViprsError> {
        if width == 0 || height == 0 {
            return Err(ViprsError::Scheduler(format!(
                "EyeOp width and height must be > 0, got {width}x{height}"
            )));
        }
        if !factor.is_finite() || !(0.0..=1.0).contains(&factor) {
            return Err(ViprsError::Scheduler(format!(
                "EyeOp factor must be finite and in [0, 1], got {factor}"
            )));
        }

        Ok(Self {
            width,
            height,
            factor,
            uchar: false,
            _format: PhantomData,
        })
    }

    #[must_use]
    /// Returns this value configured with uchar.
    pub const fn with_uchar(mut self, uchar: bool) -> Self {
        self.uchar = uchar;
        self
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

impl<F> Op for EyeOp<F>
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
        debug_assert_eq!(output.bands, 1, "EyeOp output must be single-band");
        debug_assert!(output.region.x >= 0 && output.region.y >= 0);
        debug_assert!(output.region.x as u32 + output.region.width <= self.width);
        debug_assert!(output.region.y as u32 + output.region.height <= self.height);

        let max_x = self.width.saturating_sub(1).max(1);
        let max_y = self.height.saturating_sub(1).max(1);
        let c = self.factor * std::f64::consts::PI / (2.0 * f64::from(max_x));
        let h = f64::from(max_y * max_y);
        let region_width = output.region.width as usize;

        for row in 0..output.region.height as usize {
            let y = f64::from(output.region.y as u32 + row as u32);
            for col in 0..region_width {
                let x = f64::from(output.region.x as u32 + col as u32);
                let value = (y * y * (c * x * x).cos()) / h;
                let sample = if self.uchar {
                    ((value.clamp(-1.0, 1.0) + 1.0) * 0.5) * 255.0
                } else {
                    value
                };
                output.data[row * region_width + col] = F::Sample::from_f64(sample);
            }
        }
    }
}

impl<F> PixelLocalOp for EyeOp<F>
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
        format::F32,
        image::{Region, Tile, TileMut},
    };

    fn render(op: EyeOp<F32>) -> Vec<f32> {
        let region = Region::new(0, 0, op.width(), op.height());
        let input_data = vec![0.0f32; region.pixel_count()];
        let mut output_data = vec![0.0f32; region.pixel_count()];
        let input = Tile::<F32>::new(region, 1, &input_data);
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);
        output_data
    }

    #[test]
    fn factor_zero_is_a_vertical_ramp() {
        let samples = render(EyeOp::<F32>::new(4, 5, 0.0).unwrap());

        assert!((samples[0] - 0.0).abs() < 1e-6);
        assert!((samples[4 * 4] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn constructor_rejects_invalid_parameters() {
        assert!(EyeOp::<F32>::new(0, 4, 0.5).is_err());
        assert!(EyeOp::<F32>::new(4, 4, 1.5).is_err());
    }

    #[test]
    fn partial_region_uses_absolute_coordinates() {
        let op = EyeOp::<F32>::new(6, 4, 0.5).unwrap();
        let full = render(op);
        let region = Region::new(2, 1, 3, 2);
        let input_region = Region::new(0, 0, op.width(), op.height());
        let input_data = vec![0.0f32; input_region.pixel_count()];
        let mut output_data = vec![0.0f32; region.pixel_count()];
        let input = Tile::<F32>::new(input_region, 1, &input_data);
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);

        op.process_region(&mut (), &input, &mut output);

        let full_width = op.width() as usize;
        assert_eq!(
            output_data,
            vec![
                full[full_width + 2],
                full[full_width + 3],
                full[full_width + 4],
                full[full_width * 2 + 2],
                full[full_width * 2 + 3],
                full[full_width * 2 + 4],
            ]
        );
    }

    #[test]
    fn with_uchar_preserves_dimensions() {
        let op = EyeOp::<F32>::new(4, 3, 0.25).unwrap().with_uchar(true);

        assert_eq!(op.width(), 4);
        assert_eq!(op.height(), 3);
        assert!(
            render(op)
                .iter()
                .all(|sample| (0.0..=255.0).contains(sample))
        );
    }

    proptest! {
        #[test]
        fn prop_output_has_expected_dimensions_and_range(
            width in 1u32..=32,
            height in 1u32..=32,
            factor in 0.0f64..=1.0,
            uchar in any::<bool>(),
        ) {
            let op = EyeOp::<F32>::new(width, height, factor).unwrap().with_uchar(uchar);
            let samples = render(op);

            prop_assert_eq!(samples.len(), width as usize * height as usize);
            for sample in samples {
                if uchar {
                    prop_assert!((0.0..=255.0).contains(&sample));
                } else {
                    prop_assert!((-1.0 - 1e-6..=1.0 + 1e-6).contains(&sample));
                }
            }
        }
    }
}

use std::marker::PhantomData;

use viprs_core::{
    error::ViprsError,
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
    shared_ops::sample_conv::FromF64,
};

const MASK_SANITY: u32 = 5_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Enumerates the available gaussmat precision values.
pub enum GaussmatPrecision {
    /// Uses the `Integer` variant of `GaussmatPrecision`.
    Integer,
    /// Uses the `Float` variant of `GaussmatPrecision`.
    Float,
    /// Uses the `Approximate` variant of `GaussmatPrecision`.
    Approximate,
}

/// Generate a Gaussian kernel image matching libvips `gaussmat` sizing rules.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::create::gaussmat::GaussmatOp;
///
/// let op = GaussmatOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug)]
pub struct GaussmatOp<F: BandFormat> {
    sigma: f64,
    separable: bool,
    precision: GaussmatPrecision,
    radius: u32,
    _format: PhantomData<F>,
}

impl<F: BandFormat> Copy for GaussmatOp<F> {}

impl<F: BandFormat> Clone for GaussmatOp<F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: BandFormat> GaussmatOp<F> {
    /// Creates a new `GaussmatOp`.
    pub fn new(sigma: f64, min_ampl: f64) -> Result<Self, ViprsError> {
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(ViprsError::Scheduler(format!(
                "GaussmatOp sigma must be finite and > 0, got {sigma}"
            )));
        }
        if !min_ampl.is_finite() || min_ampl <= 0.0 {
            return Err(ViprsError::Scheduler(format!(
                "GaussmatOp min_ampl must be finite and > 0, got {min_ampl}"
            )));
        }

        let sig2 = 2.0 * sigma * sigma;
        let max_x = (8.0 * sigma).clamp(0.0, f64::from(MASK_SANITY)) as u32;
        let mut first_below = max_x;
        for x in 0..max_x {
            let value = (-(f64::from(x * x)) / sig2).exp();
            if value < min_ampl {
                first_below = x;
                break;
            }
        }

        Ok(Self {
            sigma,
            separable: false,
            precision: GaussmatPrecision::Integer,
            radius: first_below.saturating_sub(1),
            _format: PhantomData,
        })
    }

    #[must_use]
    /// Returns this value configured with separable.
    pub const fn with_separable(mut self, separable: bool) -> Self {
        self.separable = separable;
        self
    }

    #[must_use]
    /// Returns this value configured with precision.
    pub const fn with_precision(mut self, precision: GaussmatPrecision) -> Self {
        self.precision = precision;
        self
    }

    #[must_use]
    /// Returns or performs width.
    pub const fn width(&self) -> u32 {
        2 * self.radius + 1
    }

    #[must_use]
    /// Returns or performs height.
    pub const fn height(&self) -> u32 {
        if self.separable { 1 } else { self.width() }
    }

    #[inline(always)]
    fn kernel_value(&self, x: u32, y: u32) -> f64 {
        let xo = i64::from(x) - i64::from(self.width() / 2);
        let yo = i64::from(y) - i64::from(self.height() / 2);
        let distance = (xo * xo + yo * yo) as f64;
        let sig2 = 2.0 * self.sigma * self.sigma;
        let mut value = (-distance / sig2).exp();
        if !matches!(self.precision, GaussmatPrecision::Float) {
            value = (20.0 * value).round();
        }
        value
    }
}

impl<F> Op for GaussmatOp<F>
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
        debug_assert_eq!(output.bands, 1, "GaussmatOp output must be single-band");
        debug_assert!(output.region.x >= 0 && output.region.y >= 0);
        debug_assert!(output.region.x as u32 + output.region.width <= self.width());
        debug_assert!(output.region.y as u32 + output.region.height <= self.height());

        let region_width = output.region.width as usize;
        for row in 0..output.region.height as usize {
            let y = output.region.y as u32 + row as u32;
            for col in 0..region_width {
                let x = output.region.x as u32 + col as u32;
                output.data[row * region_width + col] =
                    F::Sample::from_f64(self.kernel_value(x, y));
            }
        }
    }
}

impl<F> PixelLocalOp for GaussmatOp<F>
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

    fn run_op(op: GaussmatOp<F32>) -> Vec<f32> {
        let region = Region::new(0, 0, op.width(), op.height());
        let input_data = vec![0.0f32; region.pixel_count()];
        let mut output_data = vec![0.0f32; region.pixel_count()];
        let input = Tile::<F32>::new(region, 1, &input_data);
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);
        output_data
    }

    #[test]
    fn dimensions_are_odd() {
        let op = GaussmatOp::<F32>::new(1.0, 0.1).unwrap();
        assert_eq!(op.width() % 2, 1);
        assert_eq!(op.height(), op.width());
    }

    #[test]
    fn separable_mode_emits_one_row() {
        let op = GaussmatOp::<F32>::new(1.0, 0.1)
            .unwrap()
            .with_separable(true);
        assert_eq!(op.height(), 1);
    }

    #[test]
    fn float_precision_has_unit_peak_at_centre() {
        let op = GaussmatOp::<F32>::new(1.0, 0.1)
            .unwrap()
            .with_precision(GaussmatPrecision::Float);
        let samples = run_op(op);
        let centre = samples[samples.len() / 2];
        let max = samples.iter().copied().fold(f32::MIN, f32::max);
        assert!((centre - 1.0).abs() < 1e-6);
        assert!((centre - max).abs() < 1e-6);
    }

    #[test]
    fn integer_precision_rounds_to_integer_like_values() {
        let op = GaussmatOp::<F32>::new(1.5, 0.1).unwrap();
        for value in run_op(op) {
            assert!((value - value.round()).abs() < 1e-6);
        }
    }

    #[test]
    fn approximate_precision_uses_integer_rounding_branch() {
        let op = GaussmatOp::<F32>::new(1.5, 0.1)
            .unwrap()
            .with_precision(GaussmatPrecision::Approximate);
        for value in run_op(op) {
            assert!((value - value.round()).abs() < 1e-6);
        }
    }

    #[test]
    fn constructor_rejects_non_positive_inputs() {
        assert!(GaussmatOp::<F32>::new(0.0, 0.1).is_err());
        assert!(GaussmatOp::<F32>::new(1.0, 0.0).is_err());
    }

    #[test]
    fn partial_region_matches_full_kernel_slice() {
        let op = GaussmatOp::<F32>::new(1.0, 0.1)
            .unwrap()
            .with_precision(GaussmatPrecision::Float);
        let full = run_op(op);
        let region = Region::new(1, 1, 2, 2);
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
                full[full_width + 1],
                full[full_width + 2],
                full[full_width * 2 + 1],
                full[full_width * 2 + 2]
            ]
        );
    }

    proptest! {
        #[test]
        fn prop_float_kernels_are_symmetric(
            sigma in 0.2f64..=3.0,
            min_ampl in 0.01f64..=0.5,
        ) {
            let op = GaussmatOp::<F32>::new(sigma, min_ampl)
                .unwrap()
                .with_precision(GaussmatPrecision::Float);
            let width = op.width() as usize;
            let height = op.height() as usize;
            let samples = run_op(op);

            for y in 0..height {
                for x in 0..width {
                    let idx = y * width + x;
                    let mirror_x = y * width + (width - 1 - x);
                    let mirror_y = (height - 1 - y) * width + x;
                    prop_assert!((samples[idx] - samples[mirror_x]).abs() < 1e-5);
                    prop_assert!((samples[idx] - samples[mirror_y]).abs() < 1e-5);
                }
            }
        }
    }
}

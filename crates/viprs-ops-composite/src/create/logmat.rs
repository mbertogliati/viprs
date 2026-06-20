use std::{fmt, marker::PhantomData};

use viprs_core::{
    error::ViprsError,
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
    shared_ops::sample_conv::FromF64,
};

const LOGMAT_SANITY: u32 = 5_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Enumerates the available logmat precision values.
pub enum LogmatPrecision {
    /// Uses the `Integer` variant of `LogmatPrecision`.
    Integer,
    /// Uses the `Float` variant of `LogmatPrecision`.
    Float,
}

/// Generate a Laplacian-of-Gaussian matrix using libvips sizing rules.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::create::logmat::LogmatOp;
///
/// let op = LogmatOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct LogmatOp<F: BandFormat> {
    sigma: f64,
    min_ampl: f64,
    separable: bool,
    precision: LogmatPrecision,
    radius: u32,
    _format: PhantomData<F>,
}

impl<F: BandFormat> fmt::Debug for LogmatOp<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LogmatOp")
            .field("sigma", &self.sigma)
            .field("min_ampl", &self.min_ampl)
            .field("separable", &self.separable)
            .field("precision", &self.precision)
            .field("radius", &self.radius)
            .finish()
    }
}

impl<F: BandFormat> Copy for LogmatOp<F> {}

impl<F: BandFormat> Clone for LogmatOp<F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: BandFormat> LogmatOp<F> {
    /// Creates a new `LogmatOp`.
    pub fn new(sigma: f64, min_ampl: f64) -> Result<Self, ViprsError> {
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(ViprsError::Scheduler(format!(
                "LogmatOp sigma must be finite and > 0, got {sigma}"
            )));
        }
        if !min_ampl.is_finite() || min_ampl <= 0.0 {
            return Err(ViprsError::Scheduler(format!(
                "LogmatOp min_ampl must be finite and > 0, got {min_ampl}"
            )));
        }

        let sig2 = sigma * sigma;
        let mut last = 0.0;
        let mut radius = LOGMAT_SANITY;
        for x in 0..LOGMAT_SANITY {
            let distance = f64::from(x * x);
            let value = logmat_value(distance, sig2);
            if value - last >= 0.0 && value.abs() < min_ampl {
                radius = x;
                break;
            }
            last = value;
        }

        if radius == LOGMAT_SANITY {
            return Err(ViprsError::Scheduler(format!(
                "LogmatOp mask too large for sigma={sigma}, min_ampl={min_ampl}"
            )));
        }

        Ok(Self {
            sigma,
            min_ampl,
            separable: false,
            precision: LogmatPrecision::Integer,
            radius,
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
    pub const fn with_precision(mut self, precision: LogmatPrecision) -> Self {
        self.precision = precision;
        self
    }

    #[must_use]
    /// Returns or performs width.
    pub const fn width(&self) -> u32 {
        self.radius * 2 + 1
    }

    #[must_use]
    /// Returns or performs height.
    pub const fn height(&self) -> u32 {
        if self.separable { 1 } else { self.width() }
    }

    #[must_use]
    /// Returns or performs sigma.
    pub const fn sigma(&self) -> f64 {
        self.sigma
    }

    #[must_use]
    /// Returns or performs min ampl.
    pub const fn min_ampl(&self) -> f64 {
        self.min_ampl
    }

    #[inline(always)]
    fn kernel_value(&self, x: u32, y: u32) -> f64 {
        let xo = i64::from(x) - i64::from(self.width() / 2);
        let yo = i64::from(y) - i64::from(self.height() / 2);
        let distance = (xo * xo + yo * yo) as f64;
        let mut value = logmat_value(distance, self.sigma * self.sigma);
        if matches!(self.precision, LogmatPrecision::Integer) {
            value = (20.0 * value).round();
        }
        value
    }
}

#[inline(always)]
fn logmat_value(distance: f64, sig2: f64) -> f64 {
    0.5 * (2.0 - distance / sig2) * (-distance / (2.0 * sig2)).exp()
}

impl<F> Op for LogmatOp<F>
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
        debug_assert_eq!(output.bands, 1, "LogmatOp output must be single-band");
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

impl<F> PixelLocalOp for LogmatOp<F>
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

    fn render(op: LogmatOp<F32>) -> Vec<f32> {
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
        let op = LogmatOp::<F32>::new(1.0, 0.1).unwrap();
        assert_eq!(op.width() % 2, 1);
        assert_eq!(op.height(), op.width());
    }

    #[test]
    fn separable_mode_emits_one_row() {
        let op = LogmatOp::<F32>::new(1.0, 0.1).unwrap().with_separable(true);
        assert_eq!(op.height(), 1);
    }

    #[test]
    fn accessors_return_constructor_values() {
        let op = LogmatOp::<F32>::new(1.25, 0.05)
            .unwrap()
            .with_precision(LogmatPrecision::Float);

        assert!((op.sigma() - 1.25).abs() < f64::EPSILON);
        assert!((op.min_ampl() - 0.05).abs() < f64::EPSILON);
        assert_eq!(op.width(), op.height());
    }

    #[test]
    fn float_precision_is_positive_at_centre() {
        let op = LogmatOp::<F32>::new(1.0, 0.1)
            .unwrap()
            .with_precision(LogmatPrecision::Float);
        let samples = render(op);
        let centre = samples[samples.len() / 2];
        assert!(centre > 0.0);
    }

    #[test]
    fn integer_precision_rounds_to_integer_like_values() {
        let op = LogmatOp::<F32>::new(1.5, 0.1).unwrap();
        for value in render(op) {
            assert!((value - value.round()).abs() < 1e-6);
        }
    }

    #[test]
    fn op_contract_helpers_return_expected_values() {
        let op = LogmatOp::<F32>::new(1.0, 0.1).unwrap();
        let requested = Region::new(1, 2, 3, 4);

        assert_eq!(op.demand_hint(), DemandHint::Any);
        assert_eq!(op.required_input_region(&requested), requested);
        op.start();
    }

    #[test]
    fn rejects_invalid_constructor_arguments() {
        assert!(LogmatOp::<F32>::new(0.0, 0.1).is_err());
        assert!(LogmatOp::<F32>::new(f64::NAN, 0.1).is_err());
        assert!(LogmatOp::<F32>::new(1.0, 0.0).is_err());
        assert!(LogmatOp::<F32>::new(1.0, f64::INFINITY).is_err());
    }

    #[test]
    fn rejects_masks_that_exceed_sanity_limit() {
        let result = LogmatOp::<F32>::new(10_000.0, 1e-12);
        assert!(result.is_err());
    }

    proptest! {
        #[test]
        fn prop_float_kernel_is_symmetric(
            sigma in 0.2f64..=3.0,
            min_ampl in 0.01f64..=0.5,
        ) {
            let op = LogmatOp::<F32>::new(sigma, min_ampl)
                .unwrap()
                .with_precision(LogmatPrecision::Float);
            let width = op.width() as usize;
            let height = op.height() as usize;
            let samples = render(op);

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

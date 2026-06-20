#![allow(clippy::struct_excessive_bools)]
// REASON: the boolean feature flags directly mirror libvips frequency-mask switches.

use std::marker::PhantomData;

use viprs_core::{
    error::ViprsError,
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
    shared_ops::sample_conv::FromF64,
};

#[derive(Clone, Copy, Debug, PartialEq)]
/// Enumerates the available frequency mask variants.
pub enum FrequencyMaskKind {
    /// Selects the `Ideal` profile for `FrequencyMaskKind`.
    Ideal {
        /// Cutoff frequency that defines the mask response.
        frequency_cutoff: f64,
    },
    /// Selects the `IdealBand` profile for `FrequencyMaskKind`.
    IdealBand {
        /// Horizontal cutoff frequency that defines the mask response.
        frequency_cutoff_x: f64,
        /// Vertical cutoff frequency that defines the mask response.
        frequency_cutoff_y: f64,
        /// Radius parameter associated with this mask profile.
        radius: f64,
    },
    /// Selects the `IdealRing` profile for `FrequencyMaskKind`.
    IdealRing {
        /// Cutoff frequency that defines the mask response.
        frequency_cutoff: f64,
        /// Ring width associated with this mask profile.
        ringwidth: f64,
    },
    /// Selects the `Gaussian` profile for `FrequencyMaskKind`.
    Gaussian {
        /// Cutoff frequency that defines the mask response.
        frequency_cutoff: f64,
        /// Amplitude threshold associated with this mask profile.
        amplitude_cutoff: f64,
    },
    /// Selects the `GaussianBand` profile for `FrequencyMaskKind`.
    GaussianBand {
        /// Horizontal cutoff frequency that defines the mask response.
        frequency_cutoff_x: f64,
        /// Vertical cutoff frequency that defines the mask response.
        frequency_cutoff_y: f64,
        /// Radius parameter associated with this mask profile.
        radius: f64,
        /// Amplitude threshold associated with this mask profile.
        amplitude_cutoff: f64,
    },
    /// Selects the `GaussianRing` profile for `FrequencyMaskKind`.
    GaussianRing {
        /// Cutoff frequency that defines the mask response.
        frequency_cutoff: f64,
        /// Amplitude threshold associated with this mask profile.
        amplitude_cutoff: f64,
        /// Ring width associated with this mask profile.
        ringwidth: f64,
    },
    /// Selects the `Butterworth` profile for `FrequencyMaskKind`.
    Butterworth {
        /// Butterworth order associated with this mask profile.
        order: f64,
        /// Cutoff frequency that defines the mask response.
        frequency_cutoff: f64,
        /// Amplitude threshold associated with this mask profile.
        amplitude_cutoff: f64,
    },
    /// Selects the `ButterworthBand` profile for `FrequencyMaskKind`.
    ButterworthBand {
        /// Butterworth order associated with this mask profile.
        order: f64,
        /// Horizontal cutoff frequency that defines the mask response.
        frequency_cutoff_x: f64,
        /// Vertical cutoff frequency that defines the mask response.
        frequency_cutoff_y: f64,
        /// Radius parameter associated with this mask profile.
        radius: f64,
        /// Amplitude threshold associated with this mask profile.
        amplitude_cutoff: f64,
    },
    /// Selects the `ButterworthRing` profile for `FrequencyMaskKind`.
    ButterworthRing {
        /// Butterworth order associated with this mask profile.
        order: f64,
        /// Cutoff frequency that defines the mask response.
        frequency_cutoff: f64,
        /// Amplitude threshold associated with this mask profile.
        amplitude_cutoff: f64,
        /// Ring width associated with this mask profile.
        ringwidth: f64,
    },
    /// Selects the `Fractal` profile for `FrequencyMaskKind`.
    Fractal {
        /// Fractal dimension associated with this mask profile.
        fractal_dimension: f64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Configures frequency mask.
pub struct FrequencyMaskOptions {
    /// Uses optical coordinates instead of Cartesian frequency coordinates.
    pub optical: bool,
    /// Rejects matching frequencies instead of passing them through.
    pub reject: bool,
    /// Suppresses the DC component at the spectrum centre.
    pub nodc: bool,
    /// Emits the mask as unsigned 8-bit output when enabled.
    pub uchar: bool,
}

impl FrequencyMaskOptions {
    #[must_use]
    /// Creates a new `FrequencyMaskOptions`.
    pub const fn new() -> Self {
        Self {
            optical: false,
            reject: false,
            nodc: false,
            uchar: false,
        }
    }
}

impl Default for FrequencyMaskOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// Creates pixels with the `frequency mask` generator operation. Use it when a pipeline needs a
/// synthetic image source instead of reading existing pixels.
///
/// # Examples
/// ```ignore
/// use viprs_ops_composite::create::frequency_mask::FrequencyMaskOp;
///
/// let op = FrequencyMaskOp { /* operation parameters */ };
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug)]
pub struct FrequencyMaskOp<F: BandFormat> {
    width: u32,
    height: u32,
    kind: FrequencyMaskKind,
    options: FrequencyMaskOptions,
    _format: PhantomData<F>,
}

impl<F: BandFormat> Copy for FrequencyMaskOp<F> {}

impl<F: BandFormat> Clone for FrequencyMaskOp<F> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Type alias for mask ideal op.
pub type MaskIdealOp<F> = FrequencyMaskOp<F>;
/// Type alias for mask ideal band op.
pub type MaskIdealBandOp<F> = FrequencyMaskOp<F>;
/// Type alias for mask ideal ring op.
pub type MaskIdealRingOp<F> = FrequencyMaskOp<F>;
/// Type alias for mask gaussian op.
pub type MaskGaussianOp<F> = FrequencyMaskOp<F>;
/// Type alias for mask gaussian band op.
pub type MaskGaussianBandOp<F> = FrequencyMaskOp<F>;
/// Type alias for mask gaussian ring op.
pub type MaskGaussianRingOp<F> = FrequencyMaskOp<F>;
/// Type alias for mask butterworth op.
pub type MaskButterworthOp<F> = FrequencyMaskOp<F>;
/// Type alias for mask butterworth band op.
pub type MaskButterworthBandOp<F> = FrequencyMaskOp<F>;
/// Type alias for mask butterworth ring op.
pub type MaskButterworthRingOp<F> = FrequencyMaskOp<F>;
/// Type alias for mask fractal op.
pub type MaskFractalOp<F> = FrequencyMaskOp<F>;

impl<F: BandFormat> FrequencyMaskOp<F> {
    /// Creates a mask operation using the `ideal` profile.
    pub fn mask_ideal(width: u32, height: u32, frequency_cutoff: f64) -> Result<Self, ViprsError> {
        Self::new(width, height, FrequencyMaskKind::Ideal { frequency_cutoff })
    }

    /// Creates a mask operation using the `ideal_band` profile.
    pub fn mask_ideal_band(
        width: u32,
        height: u32,
        frequency_cutoff_x: f64,
        frequency_cutoff_y: f64,
        radius: f64,
    ) -> Result<Self, ViprsError> {
        Self::new(
            width,
            height,
            FrequencyMaskKind::IdealBand {
                frequency_cutoff_x,
                frequency_cutoff_y,
                radius,
            },
        )
    }

    /// Creates a mask operation using the `ideal_ring` profile.
    pub fn mask_ideal_ring(
        width: u32,
        height: u32,
        frequency_cutoff: f64,
        ringwidth: f64,
    ) -> Result<Self, ViprsError> {
        Self::new(
            width,
            height,
            FrequencyMaskKind::IdealRing {
                frequency_cutoff,
                ringwidth,
            },
        )
    }

    /// Creates a mask operation using the `gaussian` profile.
    pub fn mask_gaussian(
        width: u32,
        height: u32,
        frequency_cutoff: f64,
        amplitude_cutoff: f64,
    ) -> Result<Self, ViprsError> {
        Self::new(
            width,
            height,
            FrequencyMaskKind::Gaussian {
                frequency_cutoff,
                amplitude_cutoff,
            },
        )
    }

    /// Creates a mask operation using the `gaussian_band` profile.
    pub fn mask_gaussian_band(
        width: u32,
        height: u32,
        frequency_cutoff_x: f64,
        frequency_cutoff_y: f64,
        radius: f64,
        amplitude_cutoff: f64,
    ) -> Result<Self, ViprsError> {
        Self::new(
            width,
            height,
            FrequencyMaskKind::GaussianBand {
                frequency_cutoff_x,
                frequency_cutoff_y,
                radius,
                amplitude_cutoff,
            },
        )
    }

    /// Creates a mask operation using the `gaussian_ring` profile.
    pub fn mask_gaussian_ring(
        width: u32,
        height: u32,
        frequency_cutoff: f64,
        amplitude_cutoff: f64,
        ringwidth: f64,
    ) -> Result<Self, ViprsError> {
        Self::new(
            width,
            height,
            FrequencyMaskKind::GaussianRing {
                frequency_cutoff,
                amplitude_cutoff,
                ringwidth,
            },
        )
    }

    /// Creates a mask operation using the `butterworth` profile.
    pub fn mask_butterworth(
        width: u32,
        height: u32,
        order: f64,
        frequency_cutoff: f64,
        amplitude_cutoff: f64,
    ) -> Result<Self, ViprsError> {
        Self::new(
            width,
            height,
            FrequencyMaskKind::Butterworth {
                order,
                frequency_cutoff,
                amplitude_cutoff,
            },
        )
    }

    /// Creates a mask operation using the `butterworth_band` profile.
    pub fn mask_butterworth_band(
        width: u32,
        height: u32,
        order: f64,
        frequency_cutoff_x: f64,
        frequency_cutoff_y: f64,
        radius: f64,
        amplitude_cutoff: f64,
    ) -> Result<Self, ViprsError> {
        Self::new(
            width,
            height,
            FrequencyMaskKind::ButterworthBand {
                order,
                frequency_cutoff_x,
                frequency_cutoff_y,
                radius,
                amplitude_cutoff,
            },
        )
    }

    /// Creates a mask operation using the `butterworth_ring` profile.
    pub fn mask_butterworth_ring(
        width: u32,
        height: u32,
        order: f64,
        frequency_cutoff: f64,
        amplitude_cutoff: f64,
        ringwidth: f64,
    ) -> Result<Self, ViprsError> {
        Self::new(
            width,
            height,
            FrequencyMaskKind::ButterworthRing {
                order,
                frequency_cutoff,
                amplitude_cutoff,
                ringwidth,
            },
        )
    }

    /// Creates a mask operation using the `fractal` profile.
    pub fn mask_fractal(
        width: u32,
        height: u32,
        fractal_dimension: f64,
    ) -> Result<Self, ViprsError> {
        Self::new(
            width,
            height,
            FrequencyMaskKind::Fractal { fractal_dimension },
        )
    }

    fn new(width: u32, height: u32, kind: FrequencyMaskKind) -> Result<Self, ViprsError> {
        if width == 0 || height == 0 {
            return Err(ViprsError::Scheduler(format!(
                "FrequencyMaskOp width and height must be > 0, got {width}x{height}"
            )));
        }
        validate_kind(kind)?;

        Ok(Self {
            width,
            height,
            kind,
            options: FrequencyMaskOptions::new(),
            _format: PhantomData,
        })
    }

    #[must_use]
    /// Returns this value configured with options.
    pub const fn with_options(mut self, options: FrequencyMaskOptions) -> Self {
        self.options = options;
        self
    }

    #[must_use]
    /// Returns this value configured with optical.
    pub const fn with_optical(mut self, optical: bool) -> Self {
        self.options.optical = optical;
        self
    }

    #[must_use]
    /// Returns this value configured with reject.
    pub const fn with_reject(mut self, reject: bool) -> Self {
        self.options.reject = reject;
        self
    }

    #[must_use]
    /// Returns this value configured with nodc.
    pub const fn with_nodc(mut self, nodc: bool) -> Self {
        self.options.nodc = nodc;
        self
    }

    #[must_use]
    /// Returns this value configured with uchar.
    pub const fn with_uchar(mut self, uchar: bool) -> Self {
        self.options.uchar = uchar;
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

    #[must_use]
    /// Returns or performs kind.
    pub const fn kind(&self) -> FrequencyMaskKind {
        self.kind
    }

    #[inline(always)]
    fn normalized_coordinate(&self, x: u32, y: u32) -> (f64, f64) {
        let half_width = (self.width / 2).max(1);
        let half_height = (self.height / 2).max(1);
        let mut shifted_x = x;
        let mut shifted_y = y;

        if !self.options.optical {
            shifted_x = (shifted_x + half_width) % self.width;
            shifted_y = (shifted_y + half_height) % self.height;
        }

        let dx = i64::from(shifted_x) - i64::from(half_width);
        let dy = i64::from(shifted_y) - i64::from(half_height);

        (
            dx as f64 / f64::from(half_width),
            dy as f64 / f64::from(half_height),
        )
    }

    #[inline(always)]
    fn mask_value(&self, x: u32, y: u32) -> f64 {
        let (dx, dy) = self.normalized_coordinate(x, y);
        let mut value = if !self.options.nodc && dx == 0.0 && dy == 0.0 {
            1.0
        } else {
            let response = kind_response(self.kind, dx, dy);
            if self.options.reject {
                1.0 - response
            } else {
                response
            }
        };

        if self.options.uchar {
            value *= 255.0;
        }

        value
    }
}

fn validate_kind(kind: FrequencyMaskKind) -> Result<(), ViprsError> {
    match kind {
        FrequencyMaskKind::Ideal { frequency_cutoff } => {
            validate_finite_non_negative("frequency_cutoff", frequency_cutoff)
        }
        FrequencyMaskKind::Fractal { fractal_dimension } => {
            if fractal_dimension.is_finite() && (2.0..=3.0).contains(&fractal_dimension) {
                Ok(())
            } else {
                Err(ViprsError::Scheduler(format!(
                    "FrequencyMaskOp fractal_dimension must be finite and in [2, 3], got {fractal_dimension}"
                )))
            }
        }
        FrequencyMaskKind::IdealBand {
            frequency_cutoff_x,
            frequency_cutoff_y,
            radius,
        } => {
            validate_finite_non_negative("frequency_cutoff_x", frequency_cutoff_x)?;
            validate_finite_non_negative("frequency_cutoff_y", frequency_cutoff_y)?;
            validate_positive("radius", radius)
        }
        FrequencyMaskKind::IdealRing {
            frequency_cutoff,
            ringwidth,
        } => {
            validate_finite_non_negative("frequency_cutoff", frequency_cutoff)?;
            validate_positive("ringwidth", ringwidth)
        }
        FrequencyMaskKind::Gaussian {
            frequency_cutoff,
            amplitude_cutoff,
        } => {
            validate_positive("frequency_cutoff", frequency_cutoff)?;
            validate_amplitude(amplitude_cutoff)
        }
        FrequencyMaskKind::GaussianBand {
            frequency_cutoff_x,
            frequency_cutoff_y,
            radius,
            amplitude_cutoff,
        } => {
            validate_finite_non_negative("frequency_cutoff_x", frequency_cutoff_x)?;
            validate_finite_non_negative("frequency_cutoff_y", frequency_cutoff_y)?;
            validate_positive("radius", radius)?;
            validate_amplitude(amplitude_cutoff)
        }
        FrequencyMaskKind::GaussianRing {
            frequency_cutoff,
            amplitude_cutoff,
            ringwidth,
        } => {
            validate_finite_non_negative("frequency_cutoff", frequency_cutoff)?;
            validate_amplitude(amplitude_cutoff)?;
            validate_positive("ringwidth", ringwidth)
        }
        FrequencyMaskKind::Butterworth {
            order,
            frequency_cutoff,
            amplitude_cutoff,
        } => {
            validate_positive("order", order)?;
            validate_positive("frequency_cutoff", frequency_cutoff)?;
            validate_amplitude(amplitude_cutoff)
        }
        FrequencyMaskKind::ButterworthBand {
            order,
            frequency_cutoff_x,
            frequency_cutoff_y,
            radius,
            amplitude_cutoff,
        } => {
            validate_positive("order", order)?;
            validate_finite_non_negative("frequency_cutoff_x", frequency_cutoff_x)?;
            validate_finite_non_negative("frequency_cutoff_y", frequency_cutoff_y)?;
            validate_positive("radius", radius)?;
            validate_amplitude(amplitude_cutoff)
        }
        FrequencyMaskKind::ButterworthRing {
            order,
            frequency_cutoff,
            amplitude_cutoff,
            ringwidth,
        } => {
            validate_positive("order", order)?;
            validate_finite_non_negative("frequency_cutoff", frequency_cutoff)?;
            validate_amplitude(amplitude_cutoff)?;
            validate_positive("ringwidth", ringwidth)
        }
    }
}

fn validate_finite_non_negative(name: &str, value: f64) -> Result<(), ViprsError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(ViprsError::Scheduler(format!(
            "FrequencyMaskOp {name} must be finite and >= 0, got {value}"
        )))
    }
}

fn validate_positive(name: &str, value: f64) -> Result<(), ViprsError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(ViprsError::Scheduler(format!(
            "FrequencyMaskOp {name} must be finite and > 0, got {value}"
        )))
    }
}

fn validate_amplitude(value: f64) -> Result<(), ViprsError> {
    if value.is_finite() && value > 0.0 && value <= 1.0 {
        Ok(())
    } else {
        Err(ViprsError::Scheduler(format!(
            "FrequencyMaskOp amplitude_cutoff must be finite and in (0, 1], got {value}"
        )))
    }
}

#[inline(always)]
fn kind_response(kind: FrequencyMaskKind, dx: f64, dy: f64) -> f64 {
    match kind {
        FrequencyMaskKind::Ideal { frequency_cutoff } => {
            let dist2 = dy.mul_add(dy, dx * dx);
            let fc2 = frequency_cutoff * frequency_cutoff;
            if dist2 <= fc2 { 0.0 } else { 1.0 }
        }
        FrequencyMaskKind::IdealBand {
            frequency_cutoff_x,
            frequency_cutoff_y,
            radius,
        } => {
            let r2 = radius * radius;
            let d1 = (dx - frequency_cutoff_x)
                .mul_add(dx - frequency_cutoff_x, (dy - frequency_cutoff_y).powi(2));
            let d2 = (dx + frequency_cutoff_x)
                .mul_add(dx + frequency_cutoff_x, (dy + frequency_cutoff_y).powi(2));
            if d1 < r2 || d2 < r2 { 1.0 } else { 0.0 }
        }
        FrequencyMaskKind::IdealRing {
            frequency_cutoff,
            ringwidth,
        } => {
            let df = ringwidth / 2.0;
            let dist2 = dy.mul_add(dy, dx * dx);
            let lower = (frequency_cutoff - df) * (frequency_cutoff - df);
            let upper = (frequency_cutoff + df) * (frequency_cutoff + df);
            if dist2 > lower && dist2 < upper {
                1.0
            } else {
                0.0
            }
        }
        FrequencyMaskKind::Gaussian {
            frequency_cutoff,
            amplitude_cutoff,
        } => {
            let dist2 = dy.mul_add(dy, dx * dx) / (frequency_cutoff * frequency_cutoff);
            1.0 - (amplitude_cutoff.ln() * dist2).exp()
        }
        FrequencyMaskKind::GaussianBand {
            frequency_cutoff_x,
            frequency_cutoff_y,
            radius,
            amplitude_cutoff,
        } => {
            let r2 = radius * radius;
            let cnst = amplitude_cutoff.ln();
            let d1 = (dx - frequency_cutoff_x)
                .mul_add(dx - frequency_cutoff_x, (dy - frequency_cutoff_y).powi(2));
            let d2 = (dx + frequency_cutoff_x)
                .mul_add(dx + frequency_cutoff_x, (dy + frequency_cutoff_y).powi(2));
            let cnsta = 1.0
                / (1.0
                    + (cnst
                        * 4.0
                        * frequency_cutoff_x
                            .mul_add(frequency_cutoff_x, frequency_cutoff_y.powi(2))
                        / r2)
                        .exp());

            cnsta * ((cnst * d1 / r2).exp() + (cnst * d2 / r2).exp())
        }
        FrequencyMaskKind::GaussianRing {
            frequency_cutoff,
            amplitude_cutoff,
            ringwidth,
        } => {
            let df = ringwidth / 2.0;
            let dist = dx.hypot(dy);
            (amplitude_cutoff.ln() * (dist - frequency_cutoff).powi(2) / (df * df)).exp()
        }
        FrequencyMaskKind::Butterworth {
            order,
            frequency_cutoff,
            amplitude_cutoff,
        } => {
            let d = dy.mul_add(dy, dx * dx);
            if d == 0.0 {
                0.0
            } else {
                let cnst = (1.0 / amplitude_cutoff) - 1.0;
                1.0 / (1.0 + cnst * ((frequency_cutoff * frequency_cutoff) / d).powf(order))
            }
        }
        FrequencyMaskKind::ButterworthBand {
            order,
            frequency_cutoff_x,
            frequency_cutoff_y,
            radius,
            amplitude_cutoff,
        } => {
            let r2 = radius * radius;
            let cnst = (1.0 / amplitude_cutoff) - 1.0;
            let cnsta = 1.0
                / (1.0
                    + 1.0
                        / (1.0
                            + cnst
                                * (4.0
                                    * frequency_cutoff_x
                                        .mul_add(frequency_cutoff_x, frequency_cutoff_y.powi(2))
                                    / r2)
                                    .powf(order)));
            let d1 = (dx - frequency_cutoff_x)
                .mul_add(dx - frequency_cutoff_x, (dy - frequency_cutoff_y).powi(2));
            let d2 = (dx + frequency_cutoff_x)
                .mul_add(dx + frequency_cutoff_x, (dy + frequency_cutoff_y).powi(2));

            cnsta
                * (1.0 / (1.0 + cnst * (d1 / r2).powf(order))
                    + 1.0 / (1.0 + cnst * (d2 / r2).powf(order)))
        }
        FrequencyMaskKind::ButterworthRing {
            order,
            frequency_cutoff,
            amplitude_cutoff,
            ringwidth,
        } => {
            let df = ringwidth / 2.0;
            let cnst = (1.0 / amplitude_cutoff) - 1.0;
            let dist = dx.hypot(dy);
            1.0 / (1.0 + cnst * ((dist - frequency_cutoff).powi(2) / (df * df)).powf(order))
        }
        FrequencyMaskKind::Fractal { fractal_dimension } => {
            let fd = (fractal_dimension - 4.0) / 2.0;
            dy.mul_add(dy, dx * dx).powf(fd)
        }
    }
}

impl<F> Op for FrequencyMaskOp<F>
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
        debug_assert_eq!(
            output.bands, 1,
            "FrequencyMaskOp output must be single-band"
        );
        debug_assert!(output.region.x >= 0 && output.region.y >= 0);
        debug_assert!(output.region.x as u32 + output.region.width <= self.width);
        debug_assert!(output.region.y as u32 + output.region.height <= self.height);

        let region_width = output.region.width as usize;
        for row in 0..output.region.height as usize {
            let y = output.region.y as u32 + row as u32;
            for col in 0..region_width {
                let x = output.region.x as u32 + col as u32;
                output.data[row * region_width + col] = F::Sample::from_f64(self.mask_value(x, y));
            }
        }
    }
}

impl<F> PixelLocalOp for FrequencyMaskOp<F>
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
        format::{F32, U8},
        image::{Region, Tile, TileMut},
    };

    fn render_f32(op: &FrequencyMaskOp<F32>) -> Vec<f32> {
        let region = Region::new(0, 0, op.width(), op.height());
        let input_data = vec![0.0f32; region.pixel_count()];
        let mut output_data = vec![0.0f32; region.pixel_count()];
        let input = Tile::<F32>::new(region, 1, &input_data);
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);
        output_data
    }

    fn render_u8(op: &FrequencyMaskOp<U8>) -> Vec<u8> {
        let region = Region::new(0, 0, op.width(), op.height());
        let input_data = vec![0u8; region.pixel_count()];
        let mut output_data = vec![0u8; region.pixel_count()];
        let input = Tile::<U8>::new(region, 1, &input_data);
        let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);
        output_data
    }

    #[test]
    fn ideal_default_places_dc_at_origin_and_forces_one() {
        let op = FrequencyMaskOp::<F32>::mask_ideal(4, 4, 0.5).unwrap();
        let samples = render_f32(&op);

        assert_eq!(samples[0], 1.0);
        assert_eq!(samples[1], 0.0);
        assert_eq!(samples[2], 1.0);
    }

    #[test]
    fn nodc_uses_family_response_at_dc() {
        let op = FrequencyMaskOp::<F32>::mask_ideal(4, 4, 0.5)
            .unwrap()
            .with_nodc(true);
        assert_eq!(render_f32(&op)[0], 0.0);
    }

    #[test]
    fn reject_inverts_non_dc_response() {
        let low = FrequencyMaskOp::<F32>::mask_ideal(4, 4, 0.5).unwrap();
        let reject = low.with_reject(true);
        let low_samples = render_f32(&low);
        let reject_samples = render_f32(&reject);

        assert_eq!(low_samples[2], 1.0);
        assert_eq!(reject_samples[2], 0.0);
        assert_eq!(reject_samples[0], 1.0);
    }

    #[test]
    fn optical_places_dc_at_centre() {
        let op = FrequencyMaskOp::<F32>::mask_ideal(5, 5, 0.4)
            .unwrap()
            .with_optical(true);
        let samples = render_f32(&op);

        assert_eq!(samples[2 * 5 + 2], 1.0);
    }

    #[test]
    fn uchar_scales_to_byte_range() {
        let op = FrequencyMaskOp::<U8>::mask_ideal(3, 3, 0.1)
            .unwrap()
            .with_uchar(true);
        let samples = render_u8(&op);

        assert_eq!(samples[0], 255);
    }

    #[test]
    fn all_frequency_mask_variants_are_finite_away_from_fractal_dc() {
        let ops = [
            FrequencyMaskOp::<F32>::mask_ideal_band(8, 8, 0.5, 0.25, 0.2).unwrap(),
            FrequencyMaskOp::<F32>::mask_ideal_ring(8, 8, 0.5, 0.2).unwrap(),
            FrequencyMaskOp::<F32>::mask_gaussian(8, 8, 0.5, 0.5).unwrap(),
            FrequencyMaskOp::<F32>::mask_gaussian_band(8, 8, 0.5, 0.25, 0.2, 0.5).unwrap(),
            FrequencyMaskOp::<F32>::mask_gaussian_ring(8, 8, 0.5, 0.5, 0.2).unwrap(),
            FrequencyMaskOp::<F32>::mask_butterworth(8, 8, 2.0, 0.5, 0.5).unwrap(),
            FrequencyMaskOp::<F32>::mask_butterworth_band(8, 8, 2.0, 0.5, 0.25, 0.2, 0.5).unwrap(),
            FrequencyMaskOp::<F32>::mask_butterworth_ring(8, 8, 2.0, 0.5, 0.5, 0.2).unwrap(),
            FrequencyMaskOp::<F32>::mask_fractal(8, 8, 2.5).unwrap(),
        ];

        for op in ops {
            assert!(render_f32(&op).iter().all(|sample| sample.is_finite()));
        }
    }

    #[test]
    fn rejects_invalid_parameters() {
        assert!(FrequencyMaskOp::<F32>::mask_ideal(0, 1, 0.5).is_err());
        assert!(FrequencyMaskOp::<F32>::mask_gaussian(1, 1, 0.0, 0.5).is_err());
        assert!(FrequencyMaskOp::<F32>::mask_gaussian(1, 1, 0.5, 0.0).is_err());
        assert!(FrequencyMaskOp::<F32>::mask_butterworth(1, 1, 0.0, 0.5, 0.5).is_err());
        assert!(FrequencyMaskOp::<F32>::mask_fractal(1, 1, f64::NAN).is_err());
        assert!(FrequencyMaskOp::<F32>::mask_fractal(1, 1, 1.9).is_err());
        assert!(FrequencyMaskOp::<F32>::mask_fractal(1, 1, 3.1).is_err());
    }

    proptest! {
        #[test]
        fn prop_dc_is_one_for_default_masks(
            width in 1u32..64,
            height in 1u32..64,
            cutoff in 0.01f64..1.0,
        ) {
            let op = FrequencyMaskOp::<F32>::mask_ideal(width, height, cutoff).unwrap();
            let samples = render_f32(&op);

            prop_assert!((samples[0] - 1.0).abs() < 1e-6);
        }

        #[test]
        fn prop_uchar_outputs_stay_in_byte_range(
            width in 1u32..32,
            height in 1u32..32,
            cutoff in 0.01f64..1.0,
            amplitude in 0.01f64..=1.0,
        ) {
            let op = FrequencyMaskOp::<U8>::mask_gaussian(width, height, cutoff, amplitude)
                .unwrap()
                .with_uchar(true);
            let samples = render_u8(&op);

            prop_assert_eq!(samples.len(), width as usize * height as usize);
        }
    }
}

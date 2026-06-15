//! Format-erased gamma correction operation.

use crate::domain::concretize::{Concretize, WideAccum, Width};
use crate::domain::format::{BandFormat, PointSample};

/// Gamma correction: `x^exponent` (normalized to \[0, 1\] range for integers).
///
/// For integer formats: normalizes to \[0, 1\], applies gamma, scales back.
/// For float formats: applies directly (assumes \[0, 1\] range).
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::point::gamma::Gamma;
///
/// let op = Gamma::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Gamma {
    /// Stores the `exponent` value for this item.
    pub exponent: f64,
}

impl Gamma {
    #[must_use]
    /// Creates a new `Gamma`.
    pub const fn new(exponent: f64) -> Self {
        Self { exponent }
    }
}

impl Concretize for Gamma {
    #[inline(always)]
    fn apply_sample<F: BandFormat>(&self, x: F::Sample) -> F::Sample
    where
        F::Sample: PointSample,
    {
        x.pt_pow(self.exponent)
    }

    #[inline(always)]
    fn apply_wide<W: WideAccum>(&self, x: W) -> W {
        let v = x.to_f32();
        W::from_f32(if v == 0.0 {
            0.0
        } else {
            v.powf(self.exponent as f32)
        })
    }

    #[inline(always)]
    fn min_width(&self) -> Width {
        Width::F32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::concretize::{Concretize, Width, apply_chain_to_slice};
    use crate::domain::format::F32;

    #[test]
    fn gamma_identity() {
        let mut pixels: Vec<f32> = vec![0.0, 0.5, 1.0];
        apply_chain_to_slice::<F32, _>(&Gamma::new(1.0), &mut pixels);
        assert!((pixels[0] - 0.0).abs() < 1e-6);
        assert!((pixels[1] - 0.5).abs() < 1e-6);
        assert!((pixels[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn gamma_squared() {
        let mut pixels: Vec<f32> = vec![0.5];
        apply_chain_to_slice::<F32, _>(&Gamma::new(2.0), &mut pixels);
        assert!((pixels[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn gamma_sqrt() {
        let mut pixels: Vec<f32> = vec![0.25];
        apply_chain_to_slice::<F32, _>(&Gamma::new(0.5), &mut pixels);
        assert!((pixels[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn gamma_wide_zero_and_width() {
        let op = Gamma::new(2.2);
        assert_eq!(<Gamma as Concretize>::apply_wide::<f32>(&op, 0.0), 0.0);
        assert_eq!(<Gamma as Concretize>::min_width(&op), Width::F32);
    }

    #[test]
    fn gamma_wide_non_zero() {
        let op = Gamma::new(2.0);
        assert!((<Gamma as Concretize>::apply_wide::<f32>(&op, 0.5) - 0.25).abs() < 1e-6);
    }
}

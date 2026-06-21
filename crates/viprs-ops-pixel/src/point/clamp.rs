//! Format-erased clamp operation.

use viprs_core::concretize::{Concretize, WideAccum, Width};
use viprs_core::format::{BandFormat, PointSample};

/// Clamp each sample to `[min, max]`.
///
/// Bounds are expressed as f64 for format-erased usage. At monomorphization
/// time, the comparison is done in f64 and the result converted back.
///
/// # Examples
/// ```ignore
/// use viprs_ops_pixel::point::clamp::Clamp;
///
/// let op = Clamp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Clamp {
    /// Stores the `min` value for this item.
    pub min: f64,
    /// Stores the `max` value for this item.
    pub max: f64,
}

impl Clamp {
    #[must_use]
    /// Creates a new `Clamp`.
    pub const fn new(min: f64, max: f64) -> Self {
        Self { min, max }
    }
}

impl Concretize for Clamp {
    #[inline(always)]
    fn apply_sample<F: BandFormat>(&self, x: F::Sample) -> F::Sample
    where
        F::Sample: PointSample,
    {
        x.pt_clamp(self.min, self.max)
    }

    #[inline(always)]
    fn apply_wide<W: WideAccum>(&self, x: W) -> W {
        x.max(W::from_f64(self.min)).min(W::from_f64(self.max))
    }

    #[inline(always)]
    fn min_width(&self) -> Width {
        Width::Native
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use viprs_core::concretize::apply_chain_to_slice;
    use viprs_core::format::{F32, I16, U8};

    #[test]
    fn clamp_u8() {
        let mut pixels: Vec<u8> = vec![0, 50, 128, 200, 255];
        apply_chain_to_slice::<U8, _>(&Clamp::new(50.0, 200.0), &mut pixels);
        assert_eq!(pixels, vec![50, 50, 128, 200, 200]);
    }

    #[test]
    fn clamp_i16() {
        let mut pixels: Vec<i16> = vec![-1000, -50, 0, 50, 1000];
        apply_chain_to_slice::<I16, _>(&Clamp::new(-100.0, 100.0), &mut pixels);
        assert_eq!(pixels, vec![-100, -50, 0, 50, 100]);
    }

    #[test]
    fn clamp_f32() {
        let mut pixels: Vec<f32> = vec![-0.5, 0.0, 0.5, 1.0, 1.5];
        apply_chain_to_slice::<F32, _>(&Clamp::new(0.0, 1.0), &mut pixels);
        assert_eq!(pixels, vec![0.0, 0.0, 0.5, 1.0, 1.0]);
    }
}

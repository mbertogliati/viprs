//! Format-erased absolute value operation.

use viprs_core::concretize::{Concretize, WideAccum, Width};
use viprs_core::format::{BandFormat, PointSample};

/// Absolute value of each sample.
///
/// - Unsigned integers: identity (always positive)
/// - Signed integers: `saturating_abs` (`i16::MIN` → `i16::MAX`)
/// - Floats: `f32::abs` / `f64::abs`
///
/// # Examples
/// ```ignore
/// use viprs_ops_pixel::point::abs::Abs;
///
/// let op = Abs;
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Abs;

impl Concretize for Abs {
    #[inline(always)]
    fn apply_sample<F: BandFormat>(&self, x: F::Sample) -> F::Sample
    where
        F::Sample: PointSample,
    {
        x.pt_abs()
    }

    #[inline(always)]
    fn apply_wide<W: WideAccum>(&self, x: W) -> W {
        x.abs()
    }

    #[inline(always)]
    fn min_width(&self) -> Width {
        Width::Native
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use viprs_core::concretize::{Concretize, Width, apply_chain_to_slice};
    use viprs_core::format::{F32, I16, I32, U8};

    #[test]
    fn abs_u8_identity() {
        let mut pixels: Vec<u8> = vec![0, 128, 255];
        apply_chain_to_slice::<U8, _>(&Abs, &mut pixels);
        assert_eq!(pixels, vec![0, 128, 255]);
    }

    #[test]
    fn abs_i16() {
        let mut pixels: Vec<i16> = vec![-100, 0, 100, i16::MIN];
        apply_chain_to_slice::<I16, _>(&Abs, &mut pixels);
        assert_eq!(pixels, vec![100, 0, 100, i16::MAX]); // MIN saturates
    }

    #[test]
    fn abs_i32() {
        let mut pixels: Vec<i32> = vec![-50, 0, 50];
        apply_chain_to_slice::<I32, _>(&Abs, &mut pixels);
        assert_eq!(pixels, vec![50, 0, 50]);
    }

    #[test]
    fn abs_f32() {
        let mut pixels: Vec<f32> = vec![-1.0, 0.0, 0.5, -0.5];
        apply_chain_to_slice::<F32, _>(&Abs, &mut pixels);
        assert_eq!(pixels, vec![1.0, 0.0, 0.5, 0.5]);
    }

    #[test]
    fn abs_wide_and_min_width() {
        assert_eq!(
            <Abs as Concretize>::apply_wide::<i16>(&Abs, i16::MIN),
            i16::MAX
        );
        assert_eq!(<Abs as Concretize>::apply_wide::<f32>(&Abs, -3.5), 3.5);
        assert_eq!(<Abs as Concretize>::min_width(&Abs), Width::Native);
    }
}

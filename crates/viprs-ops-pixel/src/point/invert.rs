//! Format-erased invert operation.

use viprs_core::concretize::{Concretize, WideAccum, Width};
use viprs_core::format::{BandFormat, PointSample};

/// Invert all samples: `max - x` for unsigned, `-x` for signed, `1.0 - x` for float.
///
/// This is the format-erased equivalent of `domain::ops::arithmetic::Invert<F>`.
/// Unlike the typed version, this struct carries no generic parameter — the format
/// is resolved at monomorphization time when the pipeline flushes.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::point::invert::Invert;
///
/// let op = Invert;
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Invert;

impl Concretize for Invert {
    #[inline(always)]
    fn apply_sample<F: BandFormat>(&self, x: F::Sample) -> F::Sample
    where
        F::Sample: PointSample,
    {
        x.pt_invert()
    }

    #[inline(always)]
    fn apply_wide<W: WideAccum>(&self, x: W) -> W {
        W::FORMAT_MAX_U8.sub(x)
    }

    #[inline(always)]
    fn min_width(&self) -> Width {
        Width::Native
    }

    #[inline(always)]
    fn try_apply_bulk_u8(&self, src: &[u8], dst: &mut [u8]) -> bool {
        use crate::arithmetic::invert::Invertible;
        u8::invert_bulk(src, dst);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use viprs_core::concretize::apply_chain_to_slice;
    use viprs_core::format::{F32, F64, I16, I32, U8, U16, U32};

    #[test]
    fn invert_u8() {
        let mut pixels: Vec<u8> = vec![0, 128, 255];
        apply_chain_to_slice::<U8, _>(&Invert, &mut pixels);
        assert_eq!(pixels, vec![255, 127, 0]);
    }

    #[test]
    fn invert_u16() {
        let mut pixels: Vec<u16> = vec![0, 32768, 65535];
        apply_chain_to_slice::<U16, _>(&Invert, &mut pixels);
        assert_eq!(pixels, vec![65535, 32767, 0]);
    }

    #[test]
    fn invert_i16() {
        let mut pixels: Vec<i16> = vec![-100, 0, 100];
        apply_chain_to_slice::<I16, _>(&Invert, &mut pixels);
        assert_eq!(pixels, vec![100, 0, -100]);
    }

    #[test]
    fn invert_u32() {
        let mut pixels: Vec<u32> = vec![0, 1000, u32::MAX];
        apply_chain_to_slice::<U32, _>(&Invert, &mut pixels);
        assert_eq!(pixels, vec![u32::MAX, u32::MAX - 1000, 0]);
    }

    #[test]
    fn invert_i32() {
        let mut pixels: Vec<i32> = vec![-50, 0, 50];
        apply_chain_to_slice::<I32, _>(&Invert, &mut pixels);
        assert_eq!(pixels, vec![50, 0, -50]);
    }

    #[test]
    fn invert_f32() {
        let mut pixels: Vec<f32> = vec![0.0, 0.5, 1.0];
        apply_chain_to_slice::<F32, _>(&Invert, &mut pixels);
        assert_eq!(pixels, vec![1.0, 0.5, 0.0]);
    }

    #[test]
    fn invert_f64() {
        let mut pixels: Vec<f64> = vec![0.0, 0.25, 1.0];
        apply_chain_to_slice::<F64, _>(&Invert, &mut pixels);
        assert_eq!(pixels, vec![1.0, 0.75, 0.0]);
    }
}

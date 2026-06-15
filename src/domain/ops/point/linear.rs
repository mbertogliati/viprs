//! Format-erased linear operation.

use crate::domain::coeff::OptimizedCoeff;
use crate::domain::concretize::{Concretize, WideAccum, Width};
use crate::domain::format::{BandFormat, PointSample};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    vaddq_s16, vcombine_u8, vcombine_u16, vcvtq_f32_u32, vcvtq_u32_f32, vdupq_n_f32, vdupq_n_s16,
    vget_high_u8, vget_high_u16, vget_low_u8, vget_low_u16, vld1_u8, vld1q_u8, vmaxq_f32,
    vmaxq_s16, vminq_f32, vminq_s16, vmlaq_n_f32, vmovl_u8, vmovl_u16, vmulq_s16, vqmovn_u16,
    vqmovn_u32, vqmovun_s16, vreinterpretq_s16_u16, vst1_u8, vst1q_u8,
};

/// Linear transform: `x * scale + offset`, clamped to the valid range for the format.
///
/// Stores [`OptimizedCoeff`] values that pre-analyze narrowing possibilities.
/// The specialization logic lives entirely in `PointSample::pt_linear` —
/// this op has zero knowledge of i16 fast-paths or SIMD lane widths.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::point::linear::Linear;
///
/// let op = Linear::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Linear {
    /// Stores the `scale` value for this item.
    pub scale: OptimizedCoeff,
    /// Stores the `offset` value for this item.
    pub offset: OptimizedCoeff,
}

impl Linear {
    #[must_use]
    /// Creates a new `Linear`.
    pub fn new(scale: f64, offset: f64) -> Self {
        Self {
            scale: OptimizedCoeff::new(scale),
            offset: OptimizedCoeff::new(offset),
        }
    }
}

impl Concretize for Linear {
    #[inline(always)]
    fn apply_sample<F: BandFormat>(&self, x: F::Sample) -> F::Sample
    where
        F::Sample: PointSample,
    {
        x.pt_linear(self.scale, self.offset)
    }

    #[inline(always)]
    fn apply_wide<W: WideAccum>(&self, x: W) -> W {
        x.mul_add(
            W::from_f64(self.scale.as_f64()),
            W::from_f64(self.offset.as_f64()),
        )
    }

    #[inline(always)]
    fn min_width(&self) -> Width {
        if self.scale.is_integer() && self.offset.is_integer() {
            Width::I16
        } else {
            Width::F32
        }
    }

    #[inline(always)]
    fn try_apply_bulk_u8(&self, src: &[u8], dst: &mut [u8]) -> bool {
        linear_bulk_u8(src, dst, self.scale, self.offset);
        true
    }
}

#[inline]
fn linear_bulk_u8(src: &[u8], dst: &mut [u8], scale: OptimizedCoeff, offset: OptimizedCoeff) {
    if let Some((scale_i16, offset_i16)) = OptimizedCoeff::i16_mul_add_unsigned(scale, offset, 255)
    {
        linear_bulk_u8_i16(src, dst, scale_i16, offset_i16);
    } else {
        linear_bulk_u8_f32(src, dst, scale.as_f32(), offset.as_f32());
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
// SAFETY: caller must execute this only on aarch64 NEON and pass byte slices; the loop guards
// ensure every vector load/store remains in bounds and the i16 coefficient pre-check guarantees
// `sample * scale + offset` stays within i16 before the final saturating narrow to u8.
unsafe fn linear_bulk_u8_i16_neon(src: &[u8], dst: &mut [u8], scale: i16, offset: i16) {
    let len = src.len().min(dst.len());
    let mut index = 0usize;
    let scale_vec = vdupq_n_s16(scale);
    let offset_vec = vdupq_n_s16(offset);
    let zero = vdupq_n_s16(0);
    let max = vdupq_n_s16(255);

    while index + 16 <= len {
        // SAFETY: `index + 16 <= len` guarantees 16 readable source bytes and 16 writable
        // destination bytes. AArch64 NEON permits unaligned loads/stores for u8 lanes.
        unsafe {
            let bytes = vld1q_u8(src.as_ptr().add(index));
            let lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(bytes)));
            let hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(bytes)));
            let lo = vmaxq_s16(
                vminq_s16(vaddq_s16(vmulq_s16(lo, scale_vec), offset_vec), max),
                zero,
            );
            let hi = vmaxq_s16(
                vminq_s16(vaddq_s16(vmulq_s16(hi, scale_vec), offset_vec), max),
                zero,
            );
            let packed = vcombine_u8(vqmovun_s16(lo), vqmovun_s16(hi));
            vst1q_u8(dst.as_mut_ptr().add(index), packed);
        }
        index += 16;
    }

    while index + 8 <= len {
        // SAFETY: `index + 8 <= len` guarantees 8 readable source bytes and 8 writable
        // destination bytes for the narrow-lane fallback.
        unsafe {
            let bytes = vld1_u8(src.as_ptr().add(index));
            let widened = vreinterpretq_s16_u16(vmovl_u8(bytes));
            let values = vmaxq_s16(
                vminq_s16(vaddq_s16(vmulq_s16(widened, scale_vec), offset_vec), max),
                zero,
            );
            vst1_u8(dst.as_mut_ptr().add(index), vqmovun_s16(values));
        }
        index += 8;
    }

    for i in index..len {
        let value = (i16::from(src[i]) * scale + offset).clamp(0, 255);
        dst[i] = value as u8;
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn linear_bulk_u8_i16(src: &[u8], dst: &mut [u8], scale: i16, offset: i16) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        let value = (i16::from(src[i]) * scale + offset).clamp(0, 255);
        dst[i] = value as u8;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn linear_bulk_u8_i16(src: &[u8], dst: &mut [u8], scale: i16, offset: i16) {
    // SAFETY: AArch64 guarantees NEON support, and the helper bounds-checks each vector chunk.
    unsafe { linear_bulk_u8_i16_neon(src, dst, scale, offset) }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
// SAFETY: caller must execute this only on aarch64 NEON and pass byte slices; the loop guards
// ensure every 8-byte load and store stays in bounds, and values are clamped to [0, 255]
// before truncating conversion back to u8.
unsafe fn linear_bulk_u8_f32_neon(src: &[u8], dst: &mut [u8], scale: f32, offset: f32) {
    let len = src.len().min(dst.len());
    let mut index = 0usize;
    let zero = vdupq_n_f32(0.0);
    let max = vdupq_n_f32(255.0);
    let offset_vec = vdupq_n_f32(offset);

    while index + 8 <= len {
        // SAFETY: `index + 8 <= len` guarantees 8 readable bytes and 8 writable bytes.
        unsafe {
            let bytes = vld1_u8(src.as_ptr().add(index));
            let widened = vmovl_u8(bytes);
            let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(widened)));
            let hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(widened)));
            let lo = vcvtq_u32_f32(vminq_f32(
                vmaxq_f32(vmlaq_n_f32(offset_vec, lo, scale), zero),
                max,
            ));
            let hi = vcvtq_u32_f32(vminq_f32(
                vmaxq_f32(vmlaq_n_f32(offset_vec, hi, scale), zero),
                max,
            ));
            let packed = vcombine_u16(vqmovn_u32(lo), vqmovn_u32(hi));
            vst1_u8(dst.as_mut_ptr().add(index), vqmovn_u16(packed));
        }
        index += 8;
    }

    for i in index..len {
        dst[i] = (f32::from(src[i]).mul_add(scale, offset)).clamp(0.0, 255.0) as u8;
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn linear_bulk_u8_f32(src: &[u8], dst: &mut [u8], scale: f32, offset: f32) {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = (f32::from(src[i]).mul_add(scale, offset)).clamp(0.0, 255.0) as u8;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn linear_bulk_u8_f32(src: &[u8], dst: &mut [u8], scale: f32, offset: f32) {
    // SAFETY: AArch64 guarantees NEON support, and the helper bounds-checks each vector chunk.
    unsafe { linear_bulk_u8_f32_neon(src, dst, scale, offset) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::concretize::apply_chain_to_slice;
    use crate::domain::format::{F32, F64, I16, U8};

    #[test]
    fn linear_u8_scale_2() {
        let mut pixels: Vec<u8> = vec![10, 50, 128];
        apply_chain_to_slice::<U8, _>(&Linear::new(2.0, 0.0), &mut pixels);
        assert_eq!(pixels, vec![20, 100, 255]); // 256 clamps to 255
    }

    #[test]
    fn linear_u8_with_offset() {
        let mut pixels: Vec<u8> = vec![100, 200];
        apply_chain_to_slice::<U8, _>(&Linear::new(1.0, 50.0), &mut pixels);
        assert_eq!(pixels, vec![150, 250]);
    }

    #[test]
    fn linear_u8_clamps_negative() {
        let mut pixels: Vec<u8> = vec![10];
        apply_chain_to_slice::<U8, _>(&Linear::new(1.0, -20.0), &mut pixels);
        assert_eq!(pixels, vec![0]); // -10 clamps to 0
    }

    #[test]
    fn linear_i16() {
        let mut pixels: Vec<i16> = vec![100, -100];
        apply_chain_to_slice::<I16, _>(&Linear::new(2.0, 10.0), &mut pixels);
        assert_eq!(pixels, vec![210, -190]);
    }

    #[test]
    fn linear_f32_no_clamp() {
        let mut pixels: Vec<f32> = vec![0.5, 1.0];
        apply_chain_to_slice::<F32, _>(&Linear::new(2.0, -0.5), &mut pixels);
        assert!((pixels[0] - 0.5).abs() < 1e-6);
        assert!((pixels[1] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn linear_f64() {
        let mut pixels: Vec<f64> = vec![0.25];
        apply_chain_to_slice::<F64, _>(&Linear::new(4.0, 0.0), &mut pixels);
        assert!((pixels[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn linear_bulk_u8_matches_sample_path_for_fractional_coefficients() {
        let op = Linear::new(1.5, 10.0);
        let input: Vec<u8> = (0..37).map(|value| ((value * 7) % 256) as u8).collect();
        let expected: Vec<u8> = input
            .iter()
            .copied()
            .map(|sample| sample.pt_linear(op.scale, op.offset))
            .collect();
        let mut actual = vec![0u8; input.len()];

        assert!(op.try_apply_bulk_u8(&input, &mut actual));
        assert_eq!(actual, expected);
    }

    #[test]
    fn linear_bulk_u8_matches_sample_path_for_integer_coefficients() {
        let op = Linear::new(2.0, 5.0);
        let input: Vec<u8> = (0..53).map(|value| ((value * 11) % 256) as u8).collect();
        let expected: Vec<u8> = input
            .iter()
            .copied()
            .map(|sample| sample.pt_linear(op.scale, op.offset))
            .collect();
        let mut actual = vec![0u8; input.len()];

        assert!(op.try_apply_bulk_u8(&input, &mut actual));
        assert_eq!(actual, expected);
    }
}

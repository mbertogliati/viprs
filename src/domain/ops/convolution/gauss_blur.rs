//! Separable Gaussian blur — O(2*(2r+1)) per pixel instead of O((2r+1)²).
//!
//! A Gaussian kernel is separable: G(x,y) = G(x) * G(y). Applying two 1-D
//! passes (horizontal then vertical) gives identical output to a single 2-D
//! convolution with the full Gaussian kernel, at a fraction of the cost.
//!
//! For sigma=3.0 (radius=9, kernel size=19):
//! - `Conv2d`: 19*19 = 361 multiplications per pixel per band.
//! - `GaussBlurH` + `GaussBlurV:` 2*19 = 38 multiplications per pixel per band.
//!
//! # Usage
//!
//! ```rust,ignore
//! let blur = GaussBlur::new(3.0);
//! let pipeline = PipelineBuilder::from_source(source)
//!     .then_op(blur.h)
//!     .then_op(blur.v)
//!     .build()?;
//! ```
//!
//! `F32` inputs keep the current floating-point pipeline. `U8` inputs switch to
//! a libvips-parity fixed-point path that clips after each separable pass.

#![allow(clippy::needless_range_loop)]
// REASON: indexed loops keep the separable-kernel implementation aligned with packed band layout.

use std::marker::PhantomData;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    int32x4_t, uint8x8_t, uint8x8x3_t, uint8x8x4_t, vaddq_u32, vcombine_u16, vcombine_u32,
    vdup_n_s16, vdup_n_u32, vdupq_n_s32, vdupq_n_u32, vget_high_s16, vget_high_u32, vget_low_s16,
    vget_low_u32, vld1_u8, vld3_u8, vld4_u8, vmlal_s16, vmovl_u8, vmull_u32, vqmovn_u16,
    vqmovn_u32, vreinterpretq_s16_u16, vreinterpretq_u32_s32, vshrn_n_u64, vst1_u8, vst1q_s32,
    vst3_u8, vst4_u8,
};

use crate::{
    domain::op::{NodeSpec, Op},
    domain::{
        format::{BandFormat, F32, F64, I16, I32, U8, U16, U32},
        image::{DemandHint, Region, Tile, TileMut},
    },
};

const GAUSSBLUR_COPY_SIGMA_THRESHOLD: f64 = 0.2;
const GAUSSBLUR_MIN_AMPL: f64 = 0.2;
const SHARPEN_MIN_AMPL: f64 = 0.1;
const GAUSSBLUR_INTEGER_SCALE: f64 = 20.0;
const SIGMA15_FAST_COEFFS: [i16; 5] = [8, 16, 20, 16, 8];
const SIGMA15_FAST_SCALE: i64 = 68;
const SIGMA15_FAST_ROUNDING: i64 = 34;
#[cfg(target_arch = "aarch64")]
const SIGMA15_FAST_RECIPROCAL: u32 = 15_421;
#[cfg(target_arch = "aarch64")]
const SIGMA15_FAST_SHIFT: i32 = 20;

#[derive(Clone)]
struct IntegerKernel1d {
    coeffs: Box<[i16]>,
    scale: i32,
    rounding: i64,
}

impl IntegerKernel1d {
    fn radius(&self) -> usize {
        (self.coeffs.len() - 1) / 2
    }
}

fn integer_kernel_with_precision(sigma: f32, min_ampl: f64) -> IntegerKernel1d {
    let sigma = f64::from(sigma);
    if sigma < GAUSSBLUR_COPY_SIGMA_THRESHOLD {
        return IntegerKernel1d {
            coeffs: vec![1].into_boxed_slice(),
            scale: 1,
            rounding: 0,
        };
    }

    let sig2 = 2.0 * sigma * sigma;
    let max_x = (8.0 * sigma).ceil() as usize;
    let mut x = 0usize;
    while x < max_x {
        let value = (-((x * x) as f64) / sig2).exp();
        if value < min_ampl {
            break;
        }
        x += 1;
    }

    let radius = x.saturating_sub(1);
    let size = 2 * radius + 1;
    let mut coeffs = Vec::with_capacity(size);
    let mut scale = 0i32;
    for i in 0..size {
        let offset = i as i32 - radius as i32;
        let value = (-f64::from(offset * offset) / sig2).exp();
        let tap = (GAUSSBLUR_INTEGER_SCALE * value).round_ties_even() as i32;
        coeffs.push(tap as i16);
        scale += tap;
    }

    if scale == 0 {
        return IntegerKernel1d {
            coeffs: vec![1].into_boxed_slice(),
            scale: 1,
            rounding: 0,
        };
    }

    IntegerKernel1d {
        coeffs: coeffs.into_boxed_slice(),
        scale,
        rounding: i64::from(scale / 2),
    }
}

fn normalise_integer_kernel(kernel: &IntegerKernel1d) -> Vec<f64> {
    let scale = f64::from(kernel.scale);
    kernel
        .coeffs
        .iter()
        .map(|&coeff| f64::from(coeff) / scale)
        .collect()
}

#[inline]
fn is_sigma15_fast_kernel(coeffs: &[i16], scale: i64, rounding: i64) -> bool {
    coeffs == SIGMA15_FAST_COEFFS
        && scale == SIGMA15_FAST_SCALE
        && rounding == SIGMA15_FAST_ROUNDING
}

/// Compute a normalised 1-D Gaussian kernel with libvips `gaussblur` sizing.
///
/// `vips_gaussblur()` treats `sigma < 0.2` as a passthrough and otherwise
/// builds an integer `gaussmat` with `min_ampl = 0.2`, then normalises by the
/// matrix scale. Matching those taps keeps non-uniform inputs and border pixels
/// aligned with libvips.
///
/// Returned as `Vec<f64>` because the kernel is computed at construction time
/// and never reallocated on the pixel path — it is stored inside the struct.
fn gaussian_kernel_1d_with_precision(
    sigma: f32,
    min_ampl: f64,
    integer_precision: bool,
) -> Vec<f64> {
    if integer_precision {
        return normalise_integer_kernel(&integer_kernel_with_precision(sigma, min_ampl));
    }

    let sigma = f64::from(sigma);
    if sigma < GAUSSBLUR_COPY_SIGMA_THRESHOLD {
        return vec![1.0];
    }

    let sig2 = 2.0 * sigma * sigma;
    let max_x = (8.0 * sigma).ceil() as usize;
    let mut x = 0usize;
    while x < max_x {
        let value = (-((x * x) as f64) / sig2).exp();
        if value < min_ampl {
            break;
        }
        x += 1;
    }

    let radius = x.saturating_sub(1);
    let size = 2 * radius + 1;
    let mut kernel = Vec::with_capacity(size);
    let mut sum = 0.0f64;
    for i in 0..size {
        let offset = i as i32 - radius as i32;
        let value = (-f64::from(offset * offset) / sig2).exp();
        kernel.push(value);
        sum += value;
    }

    if sum == 0.0 {
        return vec![1.0];
    }

    for coeff in &mut kernel {
        *coeff /= sum;
    }
    kernel
}

#[must_use]
/// Returns or performs gaussian kernel 1d.
pub fn gaussian_kernel_1d(sigma: f32) -> Vec<f64> {
    gaussian_kernel_1d_with_precision(sigma, GAUSSBLUR_MIN_AMPL, true)
}

pub(crate) fn gaussian_kernel_1d_float(sigma: f32) -> Vec<f64> {
    gaussian_kernel_1d_with_precision(sigma, GAUSSBLUR_MIN_AMPL, false)
}

pub(crate) fn sharpen_kernel_1d(sigma: f32) -> Vec<f64> {
    gaussian_kernel_1d_with_precision(sigma, SHARPEN_MIN_AMPL, true)
}

mod gauss_output_private {
    pub trait Sealed {}
}

/// Selects the intermediate/output format used by separable Gaussian blur.
///
/// `U8` keeps libvips' fixed-point path and therefore stays `U8` across both
/// passes. All other formats use the floating-point path and therefore produce
/// `F32`.
pub trait GaussOutput: gauss_output_private::Sealed + BandFormat {
    /// Associated type for format.
    type Format: BandFormat;
}

/// Type alias for the library-selected separable Gaussian intermediate format.
pub type GaussOutputFormat<F> = <F as GaussOutput>::Format;

trait GaussProcessFormat: GaussOutput + Sized {
    fn process_horizontal(
        float_kernel: &[f64],
        integer_kernel: &IntegerKernel1d,
        input: &Tile<Self>,
        output: &mut TileMut<GaussOutputFormat<Self>>,
    );

    fn process_vertical(
        float_kernel: &[f64],
        integer_kernel: &IntegerKernel1d,
        input: &Tile<Self>,
        output: &mut TileMut<GaussOutputFormat<Self>>,
    );
}

/// Trait for converting `F::Sample` to `f32` for Gaussian accumulation.
pub trait ToF32: Copy {
    /// Converts this value to f32.
    fn to_f32(self) -> f32;
}

impl ToF32 for u8 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        f32::from(self)
    }
}
impl ToF32 for u16 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        f32::from(self)
    }
}
impl ToF32 for i16 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        f32::from(self)
    }
}
impl ToF32 for u32 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
}
impl ToF32 for i32 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
}
impl ToF32 for f32 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
}
impl ToF32 for f64 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
}

impl gauss_output_private::Sealed for U8 {}
impl GaussOutput for U8 {
    type Format = Self;
}

impl GaussProcessFormat for U8 {
    #[allow(unreachable_code)]
    #[inline]
    fn process_horizontal(
        _float_kernel: &[f64],
        integer_kernel: &IntegerKernel1d,
        input: &Tile<Self>,
        output: &mut TileMut<GaussOutputFormat<Self>>,
    ) {
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: aarch64 guarantees NEON and the helper only reads from the halo-extended tile.
            unsafe {
                return process_horizontal_u8_neon(integer_kernel, input, output);
            }
        }
        process_horizontal_u8_scalar(integer_kernel, input, output);
    }

    #[allow(unreachable_code)]
    #[inline]
    fn process_vertical(
        _float_kernel: &[f64],
        integer_kernel: &IntegerKernel1d,
        input: &Tile<Self>,
        output: &mut TileMut<GaussOutputFormat<Self>>,
    ) {
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: aarch64 guarantees NEON and the helper only reads from the halo-extended tile.
            unsafe {
                return process_vertical_u8_neon(integer_kernel, input, output);
            }
        }
        process_vertical_u8_scalar(integer_kernel, input, output);
    }
}

macro_rules! impl_gauss_output_float_format {
    ($($format:ty),+ $(,)?) => {
        $(
            impl gauss_output_private::Sealed for $format {}

            impl GaussOutput for $format {
                type Format = F32;
            }

            impl GaussProcessFormat for $format {
                #[inline]
                fn process_horizontal(
                    float_kernel: &[f64],
                    _integer_kernel: &IntegerKernel1d,
                    input: &Tile<Self>,
                    output: &mut TileMut<GaussOutputFormat<Self>>,
                ) {
                    process_horizontal_float(
                        float_kernel,
                        input.data,
                        input.region.width as usize,
                        input.bands as usize,
                        output.data,
                        output.region.width as usize,
                        output.region.height as usize,
                    );
                }

                #[inline]
                fn process_vertical(
                    float_kernel: &[f64],
                    _integer_kernel: &IntegerKernel1d,
                    input: &Tile<Self>,
                    output: &mut TileMut<GaussOutputFormat<Self>>,
                ) {
                    process_vertical_float(
                        float_kernel,
                        input.data,
                        input.region.width as usize,
                        input.bands as usize,
                        output.data,
                        output.region.width as usize,
                        output.region.height as usize,
                    );
                }
            }
        )+
    };
}

impl_gauss_output_float_format!(U16, I16, U32, I32, F32, F64);

#[inline(always)]
fn clip_u8_fixed(sum: i64, scale: i64, rounding: i64) -> u8 {
    ((sum + rounding) / scale).clamp(i64::from(u8::MIN), i64::from(u8::MAX)) as u8
}

#[inline(always)]
#[allow(clippy::suboptimal_flops)]
fn process_horizontal_float<T: ToF32>(
    kernel: &[f64],
    input: &[T],
    in_w: usize,
    bands: usize,
    output: &mut [f32],
    out_w: usize,
    out_h: usize,
) {
    for y in 0..out_h {
        for ox in 0..out_w {
            for band in 0..bands {
                let mut acc = 0.0f64;
                for (tap, &weight) in kernel.iter().enumerate() {
                    let idx = (y * in_w + ox + tap) * bands + band;
                    // REASON: the scalar accumulation order must stay bit-for-bit aligned with the
                    // existing libvips-parity tests for border handling.
                    acc += f64::from(input[idx].to_f32()) * weight;
                }
                let out_idx = (y * out_w + ox) * bands + band;
                output[out_idx] = acc as f32;
            }
        }
    }
}

#[inline(always)]
#[allow(clippy::suboptimal_flops)]
fn process_vertical_float<T: ToF32>(
    kernel: &[f64],
    input: &[T],
    in_w: usize,
    bands: usize,
    output: &mut [f32],
    out_w: usize,
    out_h: usize,
) {
    for oy in 0..out_h {
        for x in 0..out_w {
            for band in 0..bands {
                let mut acc = 0.0f64;
                for (tap, &weight) in kernel.iter().enumerate() {
                    let idx = ((oy + tap) * in_w + x) * bands + band;
                    // REASON: the scalar accumulation order must stay bit-for-bit aligned with the
                    // existing libvips-parity tests for border handling.
                    acc += f64::from(input[idx].to_f32()) * weight;
                }
                let out_idx = (oy * out_w + x) * bands + band;
                output[out_idx] = acc as f32;
            }
        }
    }
}

#[inline(always)]
fn process_horizontal_u8_scalar(
    kernel: &IntegerKernel1d,
    input: &Tile<U8>,
    output: &mut TileMut<U8>,
) {
    let out_w = output.region.width as usize;
    let out_h = output.region.height as usize;
    let in_w = input.region.width as usize;
    let bands = input.bands as usize;
    let scale = i64::from(kernel.scale);

    for y in 0..out_h {
        for ox in 0..out_w {
            for band in 0..bands {
                let mut sum = 0i64;
                for (tap, &weight) in kernel.coeffs.iter().enumerate() {
                    let idx = (y * in_w + ox + tap) * bands + band;
                    sum += i64::from(weight) * i64::from(input.data[idx]);
                }
                let out_idx = (y * out_w + ox) * bands + band;
                output.data[out_idx] = clip_u8_fixed(sum, scale, kernel.rounding);
            }
        }
    }
}

#[inline(always)]
fn process_vertical_u8_scalar(
    kernel: &IntegerKernel1d,
    input: &Tile<U8>,
    output: &mut TileMut<U8>,
) {
    let out_w = output.region.width as usize;
    let out_h = output.region.height as usize;
    let in_w = input.region.width as usize;
    let bands = input.bands as usize;
    let scale = i64::from(kernel.scale);

    for oy in 0..out_h {
        for x in 0..out_w {
            for band in 0..bands {
                let mut sum = 0i64;
                for (tap, &weight) in kernel.coeffs.iter().enumerate() {
                    let idx = ((oy + tap) * in_w + x) * bands + band;
                    sum += i64::from(weight) * i64::from(input.data[idx]);
                }
                let out_idx = (oy * out_w + x) * bands + band;
                output.data[out_idx] = clip_u8_fixed(sum, scale, kernel.rounding);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
// SAFETY: caller must execute this only on NEON-capable aarch64 and pass an `output` slice with at least 8 writable bytes for the packed result lanes.
unsafe fn store_eight_u8(
    acc_lo: int32x4_t,
    acc_hi: int32x4_t,
    output: &mut [u8],
    scale: i64,
    rounding: i64,
) {
    let mut lo = [0i32; 4];
    let mut hi = [0i32; 4];
    // SAFETY: `lo` and `hi` are stack arrays with space for exactly four i32 lanes each.
    unsafe {
        vst1q_s32(lo.as_mut_ptr(), acc_lo);
        vst1q_s32(hi.as_mut_ptr(), acc_hi);
    }
    for lane in 0..4 {
        output[lane] = clip_u8_fixed(i64::from(lo[lane]), scale, rounding);
        output[4 + lane] = clip_u8_fixed(i64::from(hi[lane]), scale, rounding);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn divide_round_pack_sigma15_u8x8(acc_lo: int32x4_t, acc_hi: int32x4_t) -> uint8x8_t {
    // SAFETY: constructing constant vectors does not access memory.
    let rounding = unsafe { vdupq_n_u32(SIGMA15_FAST_ROUNDING as u32) };
    // SAFETY: constructing constant vectors does not access memory.
    let reciprocal = unsafe { vdup_n_u32(SIGMA15_FAST_RECIPROCAL) };

    // SAFETY: all operations are lane-wise on non-negative accumulators whose quotient fits in u8.
    unsafe {
        let numerators_lo = vaddq_u32(vreinterpretq_u32_s32(acc_lo), rounding);
        let numerators_hi = vaddq_u32(vreinterpretq_u32_s32(acc_hi), rounding);

        let quot_lo = vcombine_u32(
            vshrn_n_u64::<SIGMA15_FAST_SHIFT>(vmull_u32(vget_low_u32(numerators_lo), reciprocal)),
            vshrn_n_u64::<SIGMA15_FAST_SHIFT>(vmull_u32(vget_high_u32(numerators_lo), reciprocal)),
        );
        let quot_hi = vcombine_u32(
            vshrn_n_u64::<SIGMA15_FAST_SHIFT>(vmull_u32(vget_low_u32(numerators_hi), reciprocal)),
            vshrn_n_u64::<SIGMA15_FAST_SHIFT>(vmull_u32(vget_high_u32(numerators_hi), reciprocal)),
        );

        vqmovn_u16(vcombine_u16(vqmovn_u32(quot_lo), vqmovn_u32(quot_hi)))
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
// SAFETY: caller must execute this only on NEON-capable aarch64 and provide a halo-extended input row plus an output row of `out_w` bytes so each 8-sample window is valid.
unsafe fn gauss_blur_h_u8_neon_row_1(
    coeffs: &[i16],
    scale: i64,
    rounding: i64,
    input_row: &[u8],
    output_row: &mut [u8],
    out_w: usize,
) {
    let mut x = 0usize;
    let sigma15_fast = is_sigma15_fast_kernel(coeffs, scale, rounding);
    while x + 8 <= out_w {
        // SAFETY: each iteration reads exactly eight contiguous samples from `input_row[x + tap..]`, and the halo guarantees every tap window is in-bounds.
        let (acc_lo, acc_hi) = unsafe {
            let mut acc_lo = vdupq_n_s32(0);
            let mut acc_hi = vdupq_n_s32(0);
            for (tap, &coeff) in coeffs.iter().enumerate() {
                let samples = vld1_u8(input_row.as_ptr().add(x + tap));
                let samples = vreinterpretq_s16_u16(vmovl_u8(samples));
                let weight = vdup_n_s16(coeff);
                acc_lo = vmlal_s16(acc_lo, vget_low_s16(samples), weight);
                acc_hi = vmlal_s16(acc_hi, vget_high_s16(samples), weight);
            }
            (acc_lo, acc_hi)
        };
        if sigma15_fast {
            // SAFETY: `output_row[x..x + 8]` has space for eight output samples.
            unsafe {
                vst1_u8(
                    output_row.as_mut_ptr().add(x),
                    divide_round_pack_sigma15_u8x8(acc_lo, acc_hi),
                );
            }
        } else {
            // SAFETY: `output_row[x..x + 8]` has space for eight output samples.
            unsafe {
                store_eight_u8(acc_lo, acc_hi, &mut output_row[x..x + 8], scale, rounding);
            }
        }
        x += 8;
    }

    for ox in x..out_w {
        let mut sum = 0i64;
        for (tap, &weight) in coeffs.iter().enumerate() {
            sum += i64::from(weight) * i64::from(input_row[ox + tap]);
        }
        output_row[ox] = clip_u8_fixed(sum, scale, rounding);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
// SAFETY: caller must execute this only on NEON-capable aarch64 and provide halo-extended RGB rows so each 8-pixel interleaved load and 24-byte output write stays in bounds.
unsafe fn gauss_blur_h_u8_neon_row_3(
    coeffs: &[i16],
    scale: i64,
    rounding: i64,
    input_row: &[u8],
    output_row: &mut [u8],
    out_w: usize,
) {
    let mut x = 0usize;
    let sigma15_fast = is_sigma15_fast_kernel(coeffs, scale, rounding);
    while x + 8 <= out_w {
        let mut acc_lo;
        let mut acc_hi;
        // SAFETY: each `vld3_u8` reads eight contiguous RGB pixels from the halo-extended row.
        unsafe {
            acc_lo = [vdupq_n_s32(0); 3];
            acc_hi = [vdupq_n_s32(0); 3];
            for (tap, &coeff) in coeffs.iter().enumerate() {
                let sample = vld3_u8(input_row.as_ptr().add((x + tap) * 3));
                let channels = [sample.0, sample.1, sample.2];
                let weight = vdup_n_s16(coeff);
                for channel in 0..3 {
                    let widened = vreinterpretq_s16_u16(vmovl_u8(channels[channel]));
                    acc_lo[channel] = vmlal_s16(acc_lo[channel], vget_low_s16(widened), weight);
                    acc_hi[channel] = vmlal_s16(acc_hi[channel], vget_high_s16(widened), weight);
                }
            }
        }
        if sigma15_fast {
            let packed = uint8x8x3_t(
                divide_round_pack_sigma15_u8x8(acc_lo[0], acc_hi[0]),
                divide_round_pack_sigma15_u8x8(acc_lo[1], acc_hi[1]),
                divide_round_pack_sigma15_u8x8(acc_lo[2], acc_hi[2]),
            );
            // SAFETY: `output_row[x * 3..]` holds eight RGB pixels.
            unsafe {
                vst3_u8(output_row.as_mut_ptr().add(x * 3), packed);
            }
        } else {
            let mut tmp = [0u8; 24];
            for channel in 0..3 {
                let mut lo = [0i32; 4];
                let mut hi = [0i32; 4];
                // SAFETY: the stack arrays hold exactly one 128-bit vector each.
                unsafe {
                    vst1q_s32(lo.as_mut_ptr(), acc_lo[channel]);
                    vst1q_s32(hi.as_mut_ptr(), acc_hi[channel]);
                }
                for lane in 0..4 {
                    tmp[lane * 3 + channel] = clip_u8_fixed(i64::from(lo[lane]), scale, rounding);
                    tmp[(lane + 4) * 3 + channel] =
                        clip_u8_fixed(i64::from(hi[lane]), scale, rounding);
                }
            }
            output_row[x * 3..(x + 8) * 3].copy_from_slice(&tmp);
        }
        x += 8;
    }

    for ox in x..out_w {
        let out_pixel = &mut output_row[ox * 3..(ox + 1) * 3];
        for channel in 0..3 {
            let mut sum = 0i64;
            for (tap, &weight) in coeffs.iter().enumerate() {
                sum += i64::from(weight) * i64::from(input_row[(ox + tap) * 3 + channel]);
            }
            out_pixel[channel] = clip_u8_fixed(sum, scale, rounding);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
// SAFETY: caller must execute this only on NEON-capable aarch64 and provide halo-extended RGBA rows so each 8-pixel interleaved load and 32-byte output write stays in bounds.
unsafe fn gauss_blur_h_u8_neon_row_4(
    coeffs: &[i16],
    scale: i64,
    rounding: i64,
    input_row: &[u8],
    output_row: &mut [u8],
    out_w: usize,
) {
    let mut x = 0usize;
    let sigma15_fast = is_sigma15_fast_kernel(coeffs, scale, rounding);
    while x + 8 <= out_w {
        let mut acc_lo;
        let mut acc_hi;
        // SAFETY: each `vld4_u8` reads eight contiguous RGBA pixels from the halo-extended row.
        unsafe {
            acc_lo = [vdupq_n_s32(0); 4];
            acc_hi = [vdupq_n_s32(0); 4];
            for (tap, &coeff) in coeffs.iter().enumerate() {
                let sample = vld4_u8(input_row.as_ptr().add((x + tap) * 4));
                let channels = [sample.0, sample.1, sample.2, sample.3];
                let weight = vdup_n_s16(coeff);
                for channel in 0..4 {
                    let widened = vreinterpretq_s16_u16(vmovl_u8(channels[channel]));
                    acc_lo[channel] = vmlal_s16(acc_lo[channel], vget_low_s16(widened), weight);
                    acc_hi[channel] = vmlal_s16(acc_hi[channel], vget_high_s16(widened), weight);
                }
            }
        }
        if sigma15_fast {
            let packed = uint8x8x4_t(
                divide_round_pack_sigma15_u8x8(acc_lo[0], acc_hi[0]),
                divide_round_pack_sigma15_u8x8(acc_lo[1], acc_hi[1]),
                divide_round_pack_sigma15_u8x8(acc_lo[2], acc_hi[2]),
                divide_round_pack_sigma15_u8x8(acc_lo[3], acc_hi[3]),
            );
            // SAFETY: `output_row[x * 4..]` holds eight RGBA pixels.
            unsafe {
                vst4_u8(output_row.as_mut_ptr().add(x * 4), packed);
            }
        } else {
            let mut tmp = [0u8; 32];
            for channel in 0..4 {
                let mut lo = [0i32; 4];
                let mut hi = [0i32; 4];
                // SAFETY: the stack arrays hold exactly one 128-bit vector each.
                unsafe {
                    vst1q_s32(lo.as_mut_ptr(), acc_lo[channel]);
                    vst1q_s32(hi.as_mut_ptr(), acc_hi[channel]);
                }
                for lane in 0..4 {
                    tmp[lane * 4 + channel] = clip_u8_fixed(i64::from(lo[lane]), scale, rounding);
                    tmp[(lane + 4) * 4 + channel] =
                        clip_u8_fixed(i64::from(hi[lane]), scale, rounding);
                }
            }
            output_row[x * 4..(x + 8) * 4].copy_from_slice(&tmp);
        }
        x += 8;
    }

    for ox in x..out_w {
        let out_pixel = &mut output_row[ox * 4..(ox + 1) * 4];
        for channel in 0..4 {
            let mut sum = 0i64;
            for (tap, &weight) in coeffs.iter().enumerate() {
                sum += i64::from(weight) * i64::from(input_row[(ox + tap) * 4 + channel]);
            }
            out_pixel[channel] = clip_u8_fixed(sum, scale, rounding);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
// SAFETY: caller must execute this only on NEON-capable aarch64 and ensure the tile contains every tapped source row for `y`, with `output_row` sized for the current output width.
unsafe fn gauss_blur_v_u8_neon_row_1(
    coeffs: &[i16],
    scale: i64,
    rounding: i64,
    input: &Tile<U8>,
    output_row: &mut [u8],
    y: usize,
) {
    let out_w = output_row.len();
    let in_w = input.region.width as usize;
    let mut x = 0usize;
    let sigma15_fast = is_sigma15_fast_kernel(coeffs, scale, rounding);
    while x + 8 <= out_w {
        // SAFETY: for each tap we read eight contiguous pixels from row `y + tap` at column `x`.
        let (acc_lo, acc_hi) = unsafe {
            let mut acc_lo = vdupq_n_s32(0);
            let mut acc_hi = vdupq_n_s32(0);
            for (tap, &coeff) in coeffs.iter().enumerate() {
                let row_start = (y + tap) * in_w;
                let samples = vld1_u8(input.data.as_ptr().add(row_start + x));
                let samples = vreinterpretq_s16_u16(vmovl_u8(samples));
                let weight = vdup_n_s16(coeff);
                acc_lo = vmlal_s16(acc_lo, vget_low_s16(samples), weight);
                acc_hi = vmlal_s16(acc_hi, vget_high_s16(samples), weight);
            }
            (acc_lo, acc_hi)
        };
        if sigma15_fast {
            // SAFETY: `output_row[x..x + 8]` has space for eight output samples.
            unsafe {
                vst1_u8(
                    output_row.as_mut_ptr().add(x),
                    divide_round_pack_sigma15_u8x8(acc_lo, acc_hi),
                );
            }
        } else {
            // SAFETY: `output_row[x..x + 8]` has space for eight output samples.
            unsafe {
                store_eight_u8(acc_lo, acc_hi, &mut output_row[x..x + 8], scale, rounding);
            }
        }
        x += 8;
    }

    for column in x..out_w {
        let mut sum = 0i64;
        for (tap, &weight) in coeffs.iter().enumerate() {
            let idx = (y + tap) * in_w + column;
            sum += i64::from(weight) * i64::from(input.data[idx]);
        }
        output_row[column] = clip_u8_fixed(sum, scale, rounding);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
// SAFETY: caller must execute this only on NEON-capable aarch64 and ensure the tile contains every tapped RGB source row for `y`, with `output_row` sized for one RGB row.
unsafe fn gauss_blur_v_u8_neon_row_3(
    coeffs: &[i16],
    scale: i64,
    rounding: i64,
    input: &Tile<U8>,
    output_row: &mut [u8],
    y: usize,
) {
    let out_w = output_row.len() / 3;
    let in_w = input.region.width as usize;
    let bands = 3usize;
    let mut x = 0usize;
    let sigma15_fast = is_sigma15_fast_kernel(coeffs, scale, rounding);
    while x + 8 <= out_w {
        let mut acc_lo;
        let mut acc_hi;
        // SAFETY: each `vld3_u8` reads eight contiguous RGB pixels from row `y + tap`.
        unsafe {
            acc_lo = [vdupq_n_s32(0); 3];
            acc_hi = [vdupq_n_s32(0); 3];
            for (tap, &coeff) in coeffs.iter().enumerate() {
                let row_offset = ((y + tap) * in_w + x) * bands;
                let sample = vld3_u8(input.data.as_ptr().add(row_offset));
                let channels = [sample.0, sample.1, sample.2];
                let weight = vdup_n_s16(coeff);
                for channel in 0..3 {
                    let widened = vreinterpretq_s16_u16(vmovl_u8(channels[channel]));
                    acc_lo[channel] = vmlal_s16(acc_lo[channel], vget_low_s16(widened), weight);
                    acc_hi[channel] = vmlal_s16(acc_hi[channel], vget_high_s16(widened), weight);
                }
            }
        }

        if sigma15_fast {
            let packed = uint8x8x3_t(
                divide_round_pack_sigma15_u8x8(acc_lo[0], acc_hi[0]),
                divide_round_pack_sigma15_u8x8(acc_lo[1], acc_hi[1]),
                divide_round_pack_sigma15_u8x8(acc_lo[2], acc_hi[2]),
            );
            // SAFETY: `output_row[x * 3..]` holds eight RGB pixels.
            unsafe {
                vst3_u8(output_row.as_mut_ptr().add(x * 3), packed);
            }
        } else {
            let mut tmp = [0u8; 24];
            for channel in 0..3 {
                let mut lo = [0i32; 4];
                let mut hi = [0i32; 4];
                // SAFETY: the stack arrays hold exactly one 128-bit vector each.
                unsafe {
                    vst1q_s32(lo.as_mut_ptr(), acc_lo[channel]);
                    vst1q_s32(hi.as_mut_ptr(), acc_hi[channel]);
                }
                for lane in 0..4 {
                    tmp[lane * 3 + channel] = clip_u8_fixed(i64::from(lo[lane]), scale, rounding);
                    tmp[(lane + 4) * 3 + channel] =
                        clip_u8_fixed(i64::from(hi[lane]), scale, rounding);
                }
            }
            output_row[x * 3..(x + 8) * 3].copy_from_slice(&tmp);
        }
        x += 8;
    }

    for column in x..out_w {
        let out_pixel = &mut output_row[column * 3..(column + 1) * 3];
        for channel in 0..3 {
            let mut sum = 0i64;
            for (tap, &weight) in coeffs.iter().enumerate() {
                let idx = ((y + tap) * in_w + column) * bands + channel;
                sum += i64::from(weight) * i64::from(input.data[idx]);
            }
            out_pixel[channel] = clip_u8_fixed(sum, scale, rounding);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
// SAFETY: caller must execute this only on NEON-capable aarch64 and ensure the tile contains every tapped RGBA source row for `y`, with `output_row` sized for one RGBA row.
unsafe fn gauss_blur_v_u8_neon_row_4(
    coeffs: &[i16],
    scale: i64,
    rounding: i64,
    input: &Tile<U8>,
    output_row: &mut [u8],
    y: usize,
) {
    let out_w = output_row.len() / 4;
    let in_w = input.region.width as usize;
    let bands = 4usize;
    let mut x = 0usize;
    let sigma15_fast = is_sigma15_fast_kernel(coeffs, scale, rounding);
    while x + 8 <= out_w {
        let mut acc_lo;
        let mut acc_hi;
        // SAFETY: each `vld4_u8` reads eight contiguous RGBA pixels from row `y + tap`.
        unsafe {
            acc_lo = [vdupq_n_s32(0); 4];
            acc_hi = [vdupq_n_s32(0); 4];
            for (tap, &coeff) in coeffs.iter().enumerate() {
                let row_offset = ((y + tap) * in_w + x) * bands;
                let sample = vld4_u8(input.data.as_ptr().add(row_offset));
                let channels = [sample.0, sample.1, sample.2, sample.3];
                let weight = vdup_n_s16(coeff);
                for channel in 0..4 {
                    let widened = vreinterpretq_s16_u16(vmovl_u8(channels[channel]));
                    acc_lo[channel] = vmlal_s16(acc_lo[channel], vget_low_s16(widened), weight);
                    acc_hi[channel] = vmlal_s16(acc_hi[channel], vget_high_s16(widened), weight);
                }
            }
        }

        if sigma15_fast {
            let packed = uint8x8x4_t(
                divide_round_pack_sigma15_u8x8(acc_lo[0], acc_hi[0]),
                divide_round_pack_sigma15_u8x8(acc_lo[1], acc_hi[1]),
                divide_round_pack_sigma15_u8x8(acc_lo[2], acc_hi[2]),
                divide_round_pack_sigma15_u8x8(acc_lo[3], acc_hi[3]),
            );
            // SAFETY: `output_row[x * 4..]` holds eight RGBA pixels.
            unsafe {
                vst4_u8(output_row.as_mut_ptr().add(x * 4), packed);
            }
        } else {
            let mut tmp = [0u8; 32];
            for channel in 0..4 {
                let mut lo = [0i32; 4];
                let mut hi = [0i32; 4];
                // SAFETY: the stack arrays hold exactly one 128-bit vector each.
                unsafe {
                    vst1q_s32(lo.as_mut_ptr(), acc_lo[channel]);
                    vst1q_s32(hi.as_mut_ptr(), acc_hi[channel]);
                }
                for lane in 0..4 {
                    tmp[lane * 4 + channel] = clip_u8_fixed(i64::from(lo[lane]), scale, rounding);
                    tmp[(lane + 4) * 4 + channel] =
                        clip_u8_fixed(i64::from(hi[lane]), scale, rounding);
                }
            }
            output_row[x * 4..(x + 8) * 4].copy_from_slice(&tmp);
        }
        x += 8;
    }

    for column in x..out_w {
        let out_pixel = &mut output_row[column * 4..(column + 1) * 4];
        for channel in 0..4 {
            let mut sum = 0i64;
            for (tap, &weight) in coeffs.iter().enumerate() {
                let idx = ((y + tap) * in_w + column) * bands + channel;
                sum += i64::from(weight) * i64::from(input.data[idx]);
            }
            out_pixel[channel] = clip_u8_fixed(sum, scale, rounding);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
// SAFETY: caller must execute this only on NEON-capable aarch64 and provide tiles whose row slices already include the convolution halo required by the selected kernel.
unsafe fn process_horizontal_u8_neon(
    kernel: &IntegerKernel1d,
    input: &Tile<U8>,
    output: &mut TileMut<U8>,
) {
    let out_w = output.region.width as usize;
    let out_h = output.region.height as usize;
    let in_w = input.region.width as usize;
    let bands = input.bands as usize;
    if !matches!(bands, 1 | 3 | 4) {
        process_horizontal_u8_scalar(kernel, input, output);
        return;
    }
    let in_stride = in_w * bands;
    let out_stride = out_w * bands;
    let scale = i64::from(kernel.scale);

    for y in 0..out_h {
        let input_row = &input.data[y * in_stride..(y + 1) * in_stride];
        let output_row = &mut output.data[y * out_stride..(y + 1) * out_stride];
        match bands {
            1 => {
                // SAFETY: the helper only reads within the halo-extended row and writes the matching output row.
                unsafe {
                    gauss_blur_h_u8_neon_row_1(
                        &kernel.coeffs,
                        scale,
                        kernel.rounding,
                        input_row,
                        output_row,
                        out_w,
                    );
                }
            }
            3 => {
                // SAFETY: the helper only reads and writes complete RGB pixel batches within the current row.
                unsafe {
                    gauss_blur_h_u8_neon_row_3(
                        &kernel.coeffs,
                        scale,
                        kernel.rounding,
                        input_row,
                        output_row,
                        out_w,
                    );
                }
            }
            4 => {
                // SAFETY: the helper only reads and writes complete RGBA pixel batches within the current row.
                unsafe {
                    gauss_blur_h_u8_neon_row_4(
                        &kernel.coeffs,
                        scale,
                        kernel.rounding,
                        input_row,
                        output_row,
                        out_w,
                    );
                }
            }
            _ => {
                debug_assert!(
                    false,
                    "GaussBlurH NEON path only specializes 1-, 3-, and 4-band tiles"
                );
                return;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
// SAFETY: caller must execute this only on NEON-capable aarch64 and provide tiles containing every tapped source row for each output row.
unsafe fn process_vertical_u8_neon(
    kernel: &IntegerKernel1d,
    input: &Tile<U8>,
    output: &mut TileMut<U8>,
) {
    let out_w = output.region.width as usize;
    let out_h = output.region.height as usize;
    let bands = input.bands as usize;
    if !matches!(bands, 1 | 3 | 4) {
        process_vertical_u8_scalar(kernel, input, output);
        return;
    }
    let out_stride = out_w * bands;
    let scale = i64::from(kernel.scale);

    for y in 0..out_h {
        let output_row = &mut output.data[y * out_stride..(y + 1) * out_stride];
        match bands {
            1 => {
                // SAFETY: the helper reads valid rows `y..y + kernel.len()` and writes one output row.
                unsafe {
                    gauss_blur_v_u8_neon_row_1(
                        &kernel.coeffs,
                        scale,
                        kernel.rounding,
                        input,
                        output_row,
                        y,
                    );
                }
            }
            3 => {
                // SAFETY: the helper reads valid RGB rows `y..y + kernel.len()` and writes one output row.
                unsafe {
                    gauss_blur_v_u8_neon_row_3(
                        &kernel.coeffs,
                        scale,
                        kernel.rounding,
                        input,
                        output_row,
                        y,
                    );
                }
            }
            4 => {
                // SAFETY: the helper reads valid RGBA rows `y..y + kernel.len()` and writes one output row.
                unsafe {
                    gauss_blur_v_u8_neon_row_4(
                        &kernel.coeffs,
                        scale,
                        kernel.rounding,
                        input,
                        output_row,
                        y,
                    );
                }
            }
            _ => {
                debug_assert!(
                    false,
                    "GaussBlurV NEON path only specializes 1-, 3-, and 4-band tiles"
                );
                return;
            }
        }
    }
}

// ── GaussBlurH ───────────────────────────────────────────────────────────────

/// Horizontal pass of a separable Gaussian blur.
///
/// The intermediate format is selected by [`GaussOutput`]: `U8` stays `U8`,
/// while every other input format writes `F32`.
pub struct GaussBlurH<F: BandFormat> {
    float_kernel: Box<[f64]>,
    integer_kernel: IntegerKernel1d,
    radius: usize,
    _fmt: PhantomData<F>,
}

impl<F: BandFormat> GaussBlurH<F> {
    /// Construct a horizontal Gaussian blur for the given `sigma`.
    ///
    /// `sigma` controls the blur width. `sigma < 0.2` is a passthrough, matching
    /// libvips `gaussblur`.
    #[must_use]
    pub fn new(sigma: f32) -> Self {
        let integer_kernel = integer_kernel_with_precision(sigma, GAUSSBLUR_MIN_AMPL);
        let radius = integer_kernel.radius();
        let float_kernel = gaussian_kernel_1d(sigma).into_boxed_slice();
        Self {
            float_kernel,
            integer_kernel,
            radius,
            _fmt: PhantomData,
        }
    }

    /// The kernel radius (number of extra input pixels needed on each side).
    #[must_use]
    pub const fn radius(&self) -> usize {
        self.radius
    }
}

// ── GaussBlurV ───────────────────────────────────────────────────────────────

/// Vertical pass of a separable Gaussian blur.
///
/// The output format is selected by [`GaussOutput`]: `U8` stays `U8`, while
/// every other input format writes `F32`.
pub struct GaussBlurV<F: BandFormat> {
    float_kernel: Box<[f64]>,
    integer_kernel: IntegerKernel1d,
    radius: usize,
    _fmt: PhantomData<F>,
}

impl<F: BandFormat> GaussBlurV<F> {
    /// Construct a vertical Gaussian blur for the given `sigma`.
    #[must_use]
    pub fn new(sigma: f32) -> Self {
        let integer_kernel = integer_kernel_with_precision(sigma, GAUSSBLUR_MIN_AMPL);
        let radius = integer_kernel.radius();
        let float_kernel = gaussian_kernel_1d(sigma).into_boxed_slice();
        Self {
            float_kernel,
            integer_kernel,
            radius,
            _fmt: PhantomData,
        }
    }

    /// The kernel radius (number of extra input pixels needed on each side).
    #[must_use]
    pub const fn radius(&self) -> usize {
        self.radius
    }
}

impl<F> Op for GaussBlurH<F>
where
    F: GaussProcessFormat,
{
    type Input = F;
    type Output = GaussOutputFormat<F>;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        Region::new(
            output.x - self.radius as i32,
            output.y,
            output.width + 2 * self.radius as u32,
            output.height,
        )
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec {
            input_tile_w: tile_w + 2 * self.radius as u32,
            input_tile_h: tile_h,
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    fn start(&self) -> Self::State {}

    #[inline]
    fn process_region(
        &self,
        _state: &mut (),
        input: &Tile<F>,
        output: &mut TileMut<GaussOutputFormat<F>>,
    ) {
        F::process_horizontal(&self.float_kernel, &self.integer_kernel, input, output);
    }
}

impl<F> Op for GaussBlurV<F>
where
    F: GaussProcessFormat,
{
    type Input = F;
    type Output = GaussOutputFormat<F>;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        Region::new(
            output.x,
            output.y - self.radius as i32,
            output.width,
            output.height + 2 * self.radius as u32,
        )
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec {
            input_tile_w: tile_w,
            input_tile_h: tile_h + 2 * self.radius as u32,
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    fn start(&self) -> Self::State {}

    #[inline]
    fn process_region(
        &self,
        _state: &mut (),
        input: &Tile<F>,
        output: &mut TileMut<GaussOutputFormat<F>>,
    ) {
        F::process_vertical(&self.float_kernel, &self.integer_kernel, input, output);
    }
}

// ── GaussBlur facade ─────────────────────────────────────────────────────────

/// Convenience wrapper that holds both passes of a separable Gaussian blur for
/// the floating-point pipeline.
pub struct GaussBlur {
    /// Stores the `sigma` value for this item.
    pub sigma: f32,
    /// Stores the `h` value for this item.
    pub h: GaussBlurH<F32>,
    /// Stores the `v` value for this item.
    pub v: GaussBlurV<F32>,
}

impl GaussBlur {
    /// Create both passes for the given `sigma`.
    #[must_use]
    pub fn new(sigma: f32) -> Self {
        Self {
            sigma,
            h: GaussBlurH::new(sigma),
            v: GaussBlurV::new(sigma),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        format::{F32, U8, U16},
        image::{DemandHint, Region, Tile, TileMut},
        op::Op,
    };
    use proptest::prelude::*;

    // ── Helper: run GaussBlurH<F32> on a given input/output tile pair ──────

    fn run_gauss_blur_h(
        input_data: &[f32],
        in_region: Region,
        out_region: Region,
        sigma: f32,
        bands: u32,
    ) -> Vec<f32> {
        let op = GaussBlurH::<F32>::new(sigma);
        let out_len = (out_region.width * out_region.height) as usize * bands as usize;
        let mut output_data = vec![0.0f32; out_len];
        let input = Tile::<F32>::new(in_region, bands, input_data);
        let mut output = TileMut::<F32>::new(out_region, bands, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    fn run_gauss_blur_v(
        input_data: &[f32],
        in_region: Region,
        out_region: Region,
        sigma: f32,
        bands: u32,
    ) -> Vec<f32> {
        let op = GaussBlurV::<F32>::new(sigma);
        let out_len = (out_region.width * out_region.height) as usize * bands as usize;
        let mut output_data = vec![0.0f32; out_len];
        let input = Tile::<F32>::new(in_region, bands, input_data);
        let mut output = TileMut::<F32>::new(out_region, bands, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    fn run_gauss_blur_h_u8(
        input_data: &[u8],
        in_region: Region,
        out_region: Region,
        sigma: f32,
        bands: u32,
    ) -> Vec<u8> {
        let op = GaussBlurH::<U8>::new(sigma);
        let out_len = (out_region.width * out_region.height) as usize * bands as usize;
        let mut output_data = vec![0u8; out_len];
        let input = Tile::<U8>::new(in_region, bands, input_data);
        let mut output = TileMut::<U8>::new(out_region, bands, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    fn run_gauss_blur_v_u8(
        input_data: &[u8],
        in_region: Region,
        out_region: Region,
        sigma: f32,
        bands: u32,
    ) -> Vec<u8> {
        let op = GaussBlurV::<U8>::new(sigma);
        let out_len = (out_region.width * out_region.height) as usize * bands as usize;
        let mut output_data = vec![0u8; out_len];
        let input = Tile::<U8>::new(in_region, bands, input_data);
        let mut output = TileMut::<U8>::new(out_region, bands, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    fn edge_extended_rows(input: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
        let extended_width = width + 2 * radius;
        let mut extended = Vec::with_capacity(extended_width * height);
        for y in 0..height {
            for x in 0..extended_width {
                let src_x = (x as i32 - radius as i32).clamp(0, width as i32 - 1) as usize;
                extended.push(input[y * width + src_x]);
            }
        }
        extended
    }

    fn edge_extended_columns(
        input: &[f32],
        width: usize,
        height: usize,
        radius: usize,
    ) -> Vec<f32> {
        let extended_height = height + 2 * radius;
        let mut extended = Vec::with_capacity(width * extended_height);
        for y in 0..extended_height {
            let src_y = (y as i32 - radius as i32).clamp(0, height as i32 - 1) as usize;
            for x in 0..width {
                extended.push(input[src_y * width + x]);
            }
        }
        extended
    }

    fn naive_separable_blur(input: &[f32], width: usize, height: usize, sigma: f32) -> Vec<f32> {
        let kernel = gaussian_kernel_1d(sigma);
        let radius = kernel.len() / 2;

        let mut horizontal = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let mut acc = 0.0f64;
                for (k, weight) in kernel.iter().enumerate() {
                    let src_x =
                        (x as i32 + k as i32 - radius as i32).clamp(0, width as i32 - 1) as usize;
                    acc += f64::from(input[y * width + src_x]) * weight;
                }
                horizontal[y * width + x] = acc as f32;
            }
        }

        let mut output = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let mut acc = 0.0f64;
                for (k, weight) in kernel.iter().enumerate() {
                    let src_y =
                        (y as i32 + k as i32 - radius as i32).clamp(0, height as i32 - 1) as usize;
                    acc += f64::from(horizontal[src_y * width + x]) * weight;
                }
                output[y * width + x] = acc as f32;
            }
        }

        output
    }

    fn naive_integer_separable_blur_u8(
        input: &[u8],
        width: usize,
        height: usize,
        bands: usize,
        sigma: f32,
    ) -> Vec<u8> {
        let kernel = integer_kernel_with_precision(sigma, GAUSSBLUR_MIN_AMPL);
        let scale = i64::from(kernel.scale);
        let radius = kernel.radius();

        let mut horizontal = vec![0u8; width * height * bands];
        for y in 0..height {
            for x in 0..width {
                for band in 0..bands {
                    let mut sum = 0i64;
                    for (tap, &weight) in kernel.coeffs.iter().enumerate() {
                        let src_x = (x as i32 + tap as i32 - radius as i32)
                            .clamp(0, width as i32 - 1)
                            as usize;
                        let idx = (y * width + src_x) * bands + band;
                        sum += i64::from(weight) * i64::from(input[idx]);
                    }
                    horizontal[(y * width + x) * bands + band] =
                        clip_u8_fixed(sum, scale, kernel.rounding);
                }
            }
        }

        let mut output = vec![0u8; width * height * bands];
        for y in 0..height {
            for x in 0..width {
                for band in 0..bands {
                    let mut sum = 0i64;
                    for (tap, &weight) in kernel.coeffs.iter().enumerate() {
                        let src_y = (y as i32 + tap as i32 - radius as i32)
                            .clamp(0, height as i32 - 1)
                            as usize;
                        let idx = (src_y * width + x) * bands + band;
                        sum += i64::from(weight) * i64::from(horizontal[idx]);
                    }
                    output[(y * width + x) * bands + band] =
                        clip_u8_fixed(sum, scale, kernel.rounding);
                }
            }
        }

        output
    }

    fn edge_extend_u8_rows(
        input: &[u8],
        width: usize,
        height: usize,
        bands: usize,
        radius: usize,
    ) -> Vec<u8> {
        let mut extended = Vec::with_capacity((width + 2 * radius) * height * bands);
        for y in 0..height {
            for x in 0..(width + 2 * radius) {
                let src_x = (x as i32 - radius as i32).clamp(0, width as i32 - 1) as usize;
                let src = (y * width + src_x) * bands;
                extended.extend_from_slice(&input[src..src + bands]);
            }
        }
        extended
    }

    fn edge_extend_u8_columns(
        input: &[u8],
        width: usize,
        height: usize,
        bands: usize,
        radius: usize,
    ) -> Vec<u8> {
        let mut extended = Vec::with_capacity(width * (height + 2 * radius) * bands);
        for y in 0..(height + 2 * radius) {
            let src_y = (y as i32 - radius as i32).clamp(0, height as i32 - 1) as usize;
            for x in 0..width {
                let src = (src_y * width + x) * bands;
                extended.extend_from_slice(&input[src..src + bands]);
            }
        }
        extended
    }

    // ── 1. Kernel sum = 1.0 ───────────────────────────────────────────────

    #[test]
    fn kernel_sum_is_one() {
        for &sigma in &[0.5f32, 1.0, 2.0, 3.0, 5.0, 10.0] {
            let k = gaussian_kernel_1d(sigma);
            let sum: f64 = k.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-9,
                "sigma={sigma}: kernel sum={sum}, expected ≈ 1.0"
            );
        }
    }

    #[test]
    fn sigma15_fast_kernel_matches_integer_path() {
        let kernel = integer_kernel_with_precision(1.5, GAUSSBLUR_MIN_AMPL);
        assert!(is_sigma15_fast_kernel(
            &kernel.coeffs,
            i64::from(kernel.scale),
            kernel.rounding
        ));
        assert_eq!(kernel.coeffs.as_ref(), SIGMA15_FAST_COEFFS);
    }

    // ── 2. Uniform input: blur(C) = C ─────────────────────────────────────

    #[test]
    fn uniform_image_unchanged_by_blur_h() {
        let sigma = 3.0f32;
        let op_h = GaussBlurH::<F32>::new(sigma);
        let radius = op_h.radius() as u32;

        let out_w = 8u32;
        let out_h = 4u32;
        let in_w = out_w + 2 * radius;
        let val = 42.0f32;

        let out_region = Region::new(0, 0, out_w, out_h);
        let in_region = Region::new(-(radius as i32), 0, in_w, out_h);
        let input_data = vec![val; (in_w * out_h) as usize];

        let result = run_gauss_blur_h(&input_data, in_region, out_region, sigma, 1);

        for &v in &result {
            assert!(
                (v - val).abs() < 1e-4,
                "horizontal pass on uniform input: expected {val}, got {v}"
            );
        }
    }

    #[test]
    fn uniform_image_unchanged_by_blur_v() {
        let sigma = 3.0f32;
        let op_v = GaussBlurV::<F32>::new(sigma);
        let radius = op_v.radius() as u32;

        let out_w = 4u32;
        let out_h = 8u32;
        let in_h = out_h + 2 * radius;
        let val = 77.0f32;

        let out_region = Region::new(0, 0, out_w, out_h);
        let in_region = Region::new(0, -(radius as i32), out_w, in_h);
        let input_data = vec![val; (out_w * in_h) as usize];

        let result = run_gauss_blur_v(&input_data, in_region, out_region, sigma, 1);

        for &v in &result {
            assert!(
                (v - val).abs() < 1e-4,
                "vertical pass on uniform input: expected {val}, got {v}"
            );
        }
    }

    // ── 3. Identity sigma≈0 (sigma=0.5): proptest output ≈ input ─────────

    proptest! {
        /// For sigma=0.5, the Gaussian kernel is very narrow (radius=2, but the
        /// center weight dominates). A uniform row must survive the horizontal pass
        /// nearly unchanged.
        #[test]
        fn small_sigma_uniform_row_preserved_h(
            val in 0.0f32..=255.0f32,
            out_w in 1u32..=16,
        ) {
            let sigma = 0.5f32;
            let op = GaussBlurH::<F32>::new(sigma);
            let radius = op.radius() as u32;
            let in_w = out_w + 2 * radius;

            let out_region = Region::new(0, 0, out_w, 1);
            let in_region = Region::new(-(radius as i32), 0, in_w, 1);
            let input_data = vec![val; in_w as usize];
            let mut output_data = vec![0.0f32; out_w as usize];

            let input = Tile::<F32>::new(in_region, 1, &input_data);
            let mut output = TileMut::<F32>::new(out_region, 1, &mut output_data);
            let mut state = ();
            op.process_region(&mut state, &input, &mut output);

            for &got in &output_data {
                prop_assert!(
                    (got - val).abs() < 1e-3,
                    "small sigma H: expected {val}, got {got}"
                );
            }
        }

        /// Same test for the vertical pass.
        #[test]
        fn small_sigma_uniform_column_preserved_v(
            val in 0.0f32..=255.0f32,
            out_h in 1u32..=16,
        ) {
            let sigma = 0.5f32;
            let op = GaussBlurV::<F32>::new(sigma);
            let radius = op.radius() as u32;
            let in_h = out_h + 2 * radius;

            let out_region = Region::new(0, 0, 1, out_h);
            let in_region = Region::new(0, -(radius as i32), 1, in_h);
            let input_data = vec![val; in_h as usize];
            let mut output_data = vec![0.0f32; out_h as usize];

            let input = Tile::<F32>::new(in_region, 1, &input_data);
            let mut output = TileMut::<F32>::new(out_region, 1, &mut output_data);
            let mut state = ();
            op.process_region(&mut state, &input, &mut output);

            for &got in &output_data {
                prop_assert!(
                    (got - val).abs() < 1e-3,
                    "small sigma V: expected {val}, got {got}"
                );
            }
        }
    }

    // ── 4. Separability test ──────────────────────────────────────────────
    //
    // GaussBlurH then GaussBlurV must produce the same result as Conv2d with
    // the outer product Gaussian 2D kernel on a 5×5 image (tolerance 1e-4).

    /// Build the 2D Gaussian kernel as the outer product of two 1D kernels.
    fn gaussian_kernel_2d(sigma: f32) -> Vec<Vec<f64>> {
        let k1d = gaussian_kernel_1d(sigma);
        let n = k1d.len();
        (0..n)
            .map(|r| (0..n).map(|c| k1d[r] * k1d[c]).collect())
            .collect()
    }

    #[test]
    fn separable_equals_conv2d_on_5x5() {
        use crate::domain::ops::convolution::conv2d::Conv2d;

        let sigma = 1.5f32;
        let op_h = GaussBlurH::<F32>::new(sigma);
        let op_v = GaussBlurV::<F32>::new(sigma);
        let radius_h = op_h.radius() as u32;
        let radius_v = op_v.radius() as u32;

        // 5×5 output image
        let out_w = 5u32;
        let out_h = 5u32;

        // Build a simple gradient input image (values 0..25).
        let img_data: Vec<f32> = (0..25).map(|i| i as f32).collect();

        // ── Two-pass separable path ──────────────────────────────────────

        // Horizontal pass: input is wider by `radius_h` on each side.
        let h_in_w = out_w + 2 * radius_h;
        let h_in_h = out_h; // height unchanged

        // For a standalone tile test we need to embed the 5×5 image inside a
        // wider buffer. Each row of the input is:
        // [left_halo | image_row | right_halo]
        // We use edge-replication: the halo pixels repeat the border value.
        let h_in_data: Vec<f32> = {
            let mut d = Vec::with_capacity((h_in_w * h_in_h) as usize);
            for row in 0..out_h as usize {
                for hx in 0..h_in_w as usize {
                    let img_x = (hx as i32 - radius_h as i32).clamp(0, out_w as i32 - 1) as usize;
                    d.push(img_data[row * out_w as usize + img_x]);
                }
            }
            d
        };

        let h_in_region = Region::new(-(radius_h as i32), 0, h_in_w, h_in_h);
        let h_out_region = Region::new(0, 0, out_w, out_h);

        let intermediate = run_gauss_blur_h(&h_in_data, h_in_region, h_out_region, sigma, 1);

        // Vertical pass: input is taller by `radius_v` on each side.
        let v_in_w = out_w;
        let v_in_h = out_h + 2 * radius_v;

        let v_in_data: Vec<f32> = {
            let mut d = Vec::with_capacity((v_in_w * v_in_h) as usize);
            for vy in 0..v_in_h as usize {
                let img_y = (vy as i32 - radius_v as i32).clamp(0, out_h as i32 - 1) as usize;
                for x in 0..v_in_w as usize {
                    d.push(intermediate[img_y * out_w as usize + x]);
                }
            }
            d
        };

        let v_in_region = Region::new(0, -(radius_v as i32), v_in_w, v_in_h);
        let v_out_region = Region::new(0, 0, out_w, out_h);

        let separable_result = run_gauss_blur_v(&v_in_data, v_in_region, v_out_region, sigma, 1);

        // ── Conv2d 2D path ────────────────────────────────────────────────

        let kernel_2d = gaussian_kernel_2d(sigma);
        let conv_radius = (kernel_2d.len() / 2) as u32;

        let conv_in_w = out_w + 2 * conv_radius;
        let conv_in_h = out_h + 2 * conv_radius;

        // Build the 2D halo-extended input with edge replication.
        let conv_in_data: Vec<f32> = {
            let mut d = Vec::with_capacity((conv_in_w * conv_in_h) as usize);
            for cy in 0..conv_in_h as usize {
                let img_y = (cy as i32 - conv_radius as i32).clamp(0, out_h as i32 - 1) as usize;
                for cx in 0..conv_in_w as usize {
                    let img_x =
                        (cx as i32 - conv_radius as i32).clamp(0, out_w as i32 - 1) as usize;
                    d.push(img_data[img_y * out_w as usize + img_x]);
                }
            }
            d
        };

        let conv_in_region = Region::new(
            -(conv_radius as i32),
            -(conv_radius as i32),
            conv_in_w,
            conv_in_h,
        );
        let conv_out_region = Region::new(0, 0, out_w, out_h);

        let conv_op = Conv2d::<F32>::new(kernel_2d).unwrap();
        let mut conv_out_data = vec![0.0f32; (out_w * out_h) as usize];
        let conv_in_tile = Tile::<F32>::new(conv_in_region, 1, &conv_in_data);
        let mut conv_out_tile = TileMut::<F32>::new(conv_out_region, 1, &mut conv_out_data);
        let mut state = ();
        conv_op.process_region(&mut state, &conv_in_tile, &mut conv_out_tile);

        // ── Compare ───────────────────────────────────────────────────────

        for (i, (&sep, &conv)) in separable_result
            .iter()
            .zip(conv_out_data.iter())
            .enumerate()
        {
            assert!(
                (sep - conv).abs() < 1e-4,
                "separability test failed at pixel {i}: separable={sep}, conv2d={conv}"
            );
        }
    }

    // ── 5. required_input_region expands correctly ─────────────────────────

    #[test]
    fn required_input_region_h_expands_horizontally() {
        let sigma = 3.0f32;
        let op = GaussBlurH::<F32>::new(sigma);
        let radius = op.radius() as u32;
        let output = Region::new(0, 0, 10, 8);
        let input = op.required_input_region(&output);
        assert_eq!(input.x, -(radius as i32));
        assert_eq!(input.y, 0);
        assert_eq!(input.width, 10 + 2 * radius);
        assert_eq!(input.height, 8);
    }

    #[test]
    fn required_input_region_v_expands_vertically() {
        let sigma = 3.0f32;
        let op = GaussBlurV::<F32>::new(sigma);
        let radius = op.radius() as u32;
        let output = Region::new(0, 0, 10, 8);
        let input = op.required_input_region(&output);
        assert_eq!(input.x, 0);
        assert_eq!(input.y, -(radius as i32));
        assert_eq!(input.width, 10);
        assert_eq!(input.height, 8 + 2 * radius);
    }

    // ── 6. node_spec reports expanded tile ────────────────────────────────

    #[test]
    fn node_spec_h_expands_width() {
        let sigma = 3.0f32;
        let op = GaussBlurH::<F32>::new(sigma);
        let radius = op.radius() as u32;
        let spec = op.node_spec(64, 64);
        assert_eq!(spec.input_tile_w, 64 + 2 * radius);
        assert_eq!(spec.input_tile_h, 64);
        assert_eq!(spec.output_tile_w, 64);
        assert_eq!(spec.output_tile_h, 64);
    }

    #[test]
    fn node_spec_v_expands_height() {
        let sigma = 3.0f32;
        let op = GaussBlurV::<F32>::new(sigma);
        let radius = op.radius() as u32;
        let spec = op.node_spec(64, 64);
        assert_eq!(spec.input_tile_w, 64);
        assert_eq!(spec.input_tile_h, 64 + 2 * radius);
        assert_eq!(spec.output_tile_w, 64);
        assert_eq!(spec.output_tile_h, 64);
    }

    // ── 7. U8 input: GaussBlurH accepts non-float formats ─────────────────

    #[test]
    fn gauss_blur_h_accepts_u8_uniform_input() {
        let sigma = 2.0f32;
        let op = GaussBlurH::<U8>::new(sigma);
        let radius = op.radius() as u32;
        let val = 128u8;
        let out_w = 4u32;
        let out_h = 1u32;
        let in_w = out_w + 2 * radius;

        let out_region = Region::new(0, 0, out_w, out_h);
        let in_region = Region::new(-(radius as i32), 0, in_w, out_h);
        let input_data = vec![val; in_w as usize];
        let mut output_data = vec![0u8; out_w as usize];

        let input = Tile::<U8>::new(in_region, 1, &input_data);
        let mut output = TileMut::<U8>::new(out_region, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);

        for &v in &output_data {
            assert_eq!(v, val, "U8 uniform horizontal: expected {val}, got {v}");
        }
    }

    #[test]
    fn sample_conversions_cover_supported_input_types() {
        assert_eq!(0u8.to_f32(), 0.0);
        assert_eq!(7u16.to_f32(), 7.0);
        assert_eq!((-3i16).to_f32(), -3.0);
        assert_eq!(11u32.to_f32(), 11.0);
        assert_eq!((-9i32).to_f32(), -9.0);
        assert_eq!(1.25f32.to_f32(), 1.25);
        assert_eq!(2.5f64.to_f32(), 2.5);
    }

    #[test]
    fn sigma_below_threshold_uses_identity_kernel() {
        let integer = integer_kernel_with_precision(0.0, GAUSSBLUR_MIN_AMPL);
        assert_eq!(&*integer.coeffs, &[1]);
        assert_eq!(integer.scale, 1);
        assert_eq!(gaussian_kernel_1d(0.0), vec![1.0]);
        assert_eq!(gaussian_kernel_1d_float(0.0), vec![1.0]);
    }

    #[test]
    fn u8_integer_path_matches_scalar_reference() {
        let width = 7usize;
        let height = 5usize;
        let sigma = 1.5f32;
        let input: Vec<u8> = (0..width * height)
            .map(|idx| ((idx * 29 + 11) % 256) as u8)
            .collect();
        let radius = integer_kernel_with_precision(sigma, GAUSSBLUR_MIN_AMPL).radius();

        let h_input = {
            let mut extended = Vec::with_capacity((width + 2 * radius) * height);
            for y in 0..height {
                for x in 0..(width + 2 * radius) {
                    let src_x = (x as i32 - radius as i32).clamp(0, width as i32 - 1) as usize;
                    extended.push(input[y * width + src_x]);
                }
            }
            extended
        };
        let intermediate = run_gauss_blur_h_u8(
            &h_input,
            Region::new(
                -(radius as i32),
                0,
                (width + 2 * radius) as u32,
                height as u32,
            ),
            Region::new(0, 0, width as u32, height as u32),
            sigma,
            1,
        );

        let v_input = {
            let mut extended = Vec::with_capacity(width * (height + 2 * radius));
            for y in 0..(height + 2 * radius) {
                let src_y = (y as i32 - radius as i32).clamp(0, height as i32 - 1) as usize;
                for x in 0..width {
                    extended.push(intermediate[src_y * width + x]);
                }
            }
            extended
        };
        let actual = run_gauss_blur_v_u8(
            &v_input,
            Region::new(
                0,
                -(radius as i32),
                width as u32,
                (height + 2 * radius) as u32,
            ),
            Region::new(0, 0, width as u32, height as u32),
            sigma,
            1,
        );

        assert_eq!(
            actual,
            naive_integer_separable_blur_u8(&input, width, height, 1, sigma)
        );
    }

    #[test]
    fn u8_integer_path_preserves_uniform_rgb_and_rgba() {
        for bands in [3u32, 4u32] {
            let sigma = 1.5f32;
            let op_h = GaussBlurH::<U8>::new(sigma);
            let op_v = GaussBlurV::<U8>::new(sigma);
            let radius = op_h.radius() as u32;
            let width = 8u32;
            let height = 4u32;
            let value = if bands == 3 {
                vec![9u8, 127u8, 240u8]
            } else {
                vec![9u8, 127u8, 240u8, 33u8]
            };

            let mut h_input = Vec::with_capacity(
                (width + 2 * radius) as usize * height as usize * bands as usize,
            );
            for _ in 0..(width + 2 * radius) * height {
                h_input.extend_from_slice(&value);
            }
            let intermediate = run_gauss_blur_h_u8(
                &h_input,
                Region::new(-(radius as i32), 0, width + 2 * radius, height),
                Region::new(0, 0, width, height),
                sigma,
                bands,
            );

            let mut v_input = Vec::with_capacity(
                width as usize * (height + 2 * radius) as usize * bands as usize,
            );
            for _ in 0..width * (height + 2 * radius) {
                v_input.extend_from_slice(&value);
            }
            let actual = run_gauss_blur_v_u8(
                &v_input,
                Region::new(0, -(radius as i32), width, height + 2 * radius),
                Region::new(0, 0, width, height),
                sigma,
                bands,
            );

            let expected = value
                .iter()
                .copied()
                .cycle()
                .take((width * height * bands) as usize)
                .collect::<Vec<_>>();
            assert_eq!(intermediate, expected);
            assert_eq!(actual, expected);
            assert_eq!(op_h.radius(), op_v.radius());
        }
    }

    #[test]
    fn u8_integer_path_matches_reference_for_vector_loop_and_tail() {
        let sigma = 1.5f32;
        let width = 9usize;
        let height = 4usize;
        let radius = integer_kernel_with_precision(sigma, GAUSSBLUR_MIN_AMPL).radius();

        for bands in [1usize, 3, 4] {
            let input = (0..width * height * bands)
                .map(|idx| ((idx * 17 + 23) % 256) as u8)
                .collect::<Vec<_>>();
            let h_input = edge_extend_u8_rows(&input, width, height, bands, radius);
            let intermediate = run_gauss_blur_h_u8(
                &h_input,
                Region::new(
                    -(radius as i32),
                    0,
                    (width + 2 * radius) as u32,
                    height as u32,
                ),
                Region::new(0, 0, width as u32, height as u32),
                sigma,
                bands as u32,
            );
            let v_input = edge_extend_u8_columns(&intermediate, width, height, bands, radius);
            let actual = run_gauss_blur_v_u8(
                &v_input,
                Region::new(
                    0,
                    -(radius as i32),
                    width as u32,
                    (height + 2 * radius) as u32,
                ),
                Region::new(0, 0, width as u32, height as u32),
                sigma,
                bands as u32,
            );
            assert_eq!(
                actual,
                naive_integer_separable_blur_u8(&input, width, height, bands, sigma)
            );
        }
    }

    #[test]
    fn u8_metadata_and_scalar_fallback_cover_non_simd_band_counts() {
        let sigma = 1.2f32;
        let h = GaussBlurH::<U8>::new(sigma);
        let v = GaussBlurV::<U8>::new(sigma);
        let radius = h.radius();
        let output = Region::new(0, 0, 5, 3);
        assert_eq!(h.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(v.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(
            h.required_input_region(&output),
            Region::new(
                -(radius as i32),
                0,
                output.width + 2 * radius as u32,
                output.height
            )
        );
        assert_eq!(
            v.required_input_region(&output),
            Region::new(
                0,
                -(radius as i32),
                output.width,
                output.height + 2 * radius as u32
            )
        );
        let _ = h.start();
        let _ = v.start();

        let width = 5usize;
        let height = 3usize;
        let bands = 2usize;
        let input = (0..width * height * bands)
            .map(|idx| ((idx * 31 + 7) % 256) as u8)
            .collect::<Vec<_>>();
        let h_input = edge_extend_u8_rows(&input, width, height, bands, radius);
        let intermediate = run_gauss_blur_h_u8(
            &h_input,
            Region::new(
                -(radius as i32),
                0,
                (width + 2 * radius) as u32,
                height as u32,
            ),
            Region::new(0, 0, width as u32, height as u32),
            sigma,
            bands as u32,
        );
        let v_input = edge_extend_u8_columns(&intermediate, width, height, bands, radius);
        let actual = run_gauss_blur_v_u8(
            &v_input,
            Region::new(
                0,
                -(radius as i32),
                width as u32,
                (height + 2 * radius) as u32,
            ),
            Region::new(0, 0, width as u32, height as u32),
            sigma,
            bands as u32,
        );
        assert_eq!(
            actual,
            naive_integer_separable_blur_u8(&input, width, height, bands, sigma)
        );
    }

    #[test]
    fn trait_entrypoints_cover_non_u8_formats() {
        let sigma = 1.0f32;
        let op_h = GaussBlurH::<U16>::new(sigma);
        let op_v = GaussBlurV::<GaussOutputFormat<U16>>::new(sigma);
        let radius = op_h.radius();

        let input = vec![5u16, 7, 11, 13];
        let h_input = edge_extend_u8_rows(
            &input.iter().map(|&v| v as u8).collect::<Vec<_>>(),
            2,
            2,
            1,
            radius,
        )
        .into_iter()
        .map(u16::from)
        .collect::<Vec<_>>();
        let mut h_output = vec![0.0f32; 4];
        let mut h_state = op_h.start();
        op_h.process_region(
            &mut h_state,
            &Tile::<U16>::new(
                Region::new(-(radius as i32), 0, (2 + 2 * radius) as u32, 2),
                1,
                &h_input,
            ),
            &mut TileMut::<F32>::new(Region::new(0, 0, 2, 2), 1, &mut h_output),
        );

        let v_input = edge_extended_columns(&h_output, 2, 2, radius);
        let mut v_output = vec![0.0f32; 4];
        let mut v_state = op_v.start();
        op_v.process_region(
            &mut v_state,
            &Tile::<F32>::new(
                Region::new(0, -(radius as i32), 2, (2 + 2 * radius) as u32),
                1,
                &v_input,
            ),
            &mut TileMut::<F32>::new(Region::new(0, 0, 2, 2), 1, &mut v_output),
        );

        assert!(v_output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn gauss_output_supports_generic_callers() {
        fn accepts_generic_chain<F>()
        where
            F: GaussOutput,
            GaussBlurH<F>: Op<Input = F, Output = GaussOutputFormat<F>>,
            GaussBlurV<GaussOutputFormat<F>>:
                Op<Input = GaussOutputFormat<F>, Output = GaussOutputFormat<F>>,
        {
        }

        fn accepts_u8_output<O: Op<Input = U8, Output = U8>>(_: &O) {}
        fn accepts_u16_output<O: Op<Input = U16, Output = F32>>(_: &O) {}

        accepts_generic_chain::<U8>();
        accepts_generic_chain::<U16>();
        accepts_generic_chain::<F32>();
        accepts_u8_output(&GaussBlurH::<U8>::new(1.0));
        accepts_u8_output(&GaussBlurV::<U8>::new(1.0));
        accepts_u16_output(&GaussBlurH::<U16>::new(1.0));
        accepts_u16_output(&GaussBlurV::<U16>::new(1.0));
    }

    // ── 8. GaussBlur facade creates matching H and V radii ────────────────

    #[test]
    fn gauss_blur_facade_h_v_have_same_radius() {
        let blur = GaussBlur::new(2.5);
        assert_eq!(blur.h.radius(), blur.v.radius());
        assert_eq!(blur.sigma, 2.5);
    }

    #[test]
    fn float_precision_kernel_avoids_integer_quantisation_for_canny() {
        let integer = gaussian_kernel_1d(1.4);
        let float = gaussian_kernel_1d_float(1.4);

        assert_eq!(integer.len(), float.len());
        assert!(
            integer
                .iter()
                .zip(float.iter())
                .any(|(lhs, rhs)| (lhs - rhs).abs() > 1e-6)
        );
    }

    #[test]
    fn sharpen_kernel_keeps_wider_support_than_gaussblur_at_same_sigma() {
        let sharpen = sharpen_kernel_1d(0.5);
        let blur = gaussian_kernel_1d(0.5);

        assert!(sharpen.len() >= blur.len());
        assert!((sharpen.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    proptest! {
        #[test]
        fn sigma_zero_is_passthrough_h(
            width in 1usize..=6,
            height in 1usize..=6,
            pixels in proptest::collection::vec(0u8..=255, 1..=36),
        ) {
            prop_assume!(pixels.len() >= width * height);
            let pixels = &pixels[..width * height];
            let region = Region::new(0, 0, width as u32, height as u32);
            let input_data: Vec<f32> = pixels.iter().map(|&value| f32::from(value)).collect();
            let result = run_gauss_blur_h(&input_data, region, region, 0.0, 1);
            prop_assert_eq!(result, input_data);
        }

        #[test]
        fn sigma_zero_is_passthrough_v(
            width in 1usize..=6,
            height in 1usize..=6,
            pixels in proptest::collection::vec(0u8..=255, 1..=36),
        ) {
            prop_assume!(pixels.len() >= width * height);
            let pixels = &pixels[..width * height];
            let region = Region::new(0, 0, width as u32, height as u32);
            let input_data: Vec<f32> = pixels.iter().map(|&value| f32::from(value)).collect();
            let result = run_gauss_blur_v(&input_data, region, region, 0.0, 1);
            prop_assert_eq!(result, input_data);
        }

        #[test]
        fn sigma_zero_is_passthrough_u8(
            width in 1usize..=6,
            height in 1usize..=6,
            pixels in proptest::collection::vec(0u8..=255, 1..=36),
        ) {
            prop_assume!(pixels.len() >= width * height);
            let pixels = pixels[..width * height].to_vec();
            let region = Region::new(0, 0, width as u32, height as u32);
            let h = run_gauss_blur_h_u8(&pixels, region, region, 0.0, 1);
            let v = run_gauss_blur_v_u8(&pixels, region, region, 0.0, 1);
            prop_assert_eq!(h, pixels.clone());
            prop_assert_eq!(v, pixels);
        }

        #[test]
        fn non_square_border_pixels_match_edge_extension(
            pixels in proptest::collection::vec(0.0f32..=255.0f32, 15),
        ) {
            let width = 3usize;
            let height = 5usize;
            let sigma = 1.0f32;
            let radius = gaussian_kernel_1d(sigma).len() / 2;

            let h_input = edge_extended_rows(&pixels, width, height, radius);
            let intermediate = run_gauss_blur_h(
                &h_input,
                Region::new(-(radius as i32), 0, (width + 2 * radius) as u32, height as u32),
                Region::new(0, 0, width as u32, height as u32),
                sigma,
                1,
            );

            let v_input = edge_extended_columns(&intermediate, width, height, radius);
            let actual = run_gauss_blur_v(
                &v_input,
                Region::new(0, -(radius as i32), width as u32, (height + 2 * radius) as u32),
                Region::new(0, 0, width as u32, height as u32),
                sigma,
                1,
            );
            let expected = naive_separable_blur(&pixels, width, height, sigma);

            prop_assert!((actual[0] - expected[0]).abs() <= f32::EPSILON);
            prop_assert!((actual[width - 1] - expected[width - 1]).abs() <= f32::EPSILON);
            prop_assert!(
                (actual[(height - 1) * width] - expected[(height - 1) * width]).abs()
                    <= f32::EPSILON
            );
            prop_assert!(
                (actual[height * width - 1] - expected[height * width - 1]).abs()
                    <= f32::EPSILON
            );
        }
    }
}

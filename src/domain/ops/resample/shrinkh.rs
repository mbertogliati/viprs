//! Horizontal integer shrink by a box filter.
//!
//! `ShrinkH` averages `factor` consecutive pixels per output pixel, producing a
//! downscaled image with `output_width = floor(input_width / factor)` by default.
//! Set `ceil=true` to round the output width up (libvips parity).
//! It is the fast path for large integer downscales: O(n/factor) work compared
//! to O(n * `kernel_support`) for `ReduceH`. Use `ShrinkH` followed by `ReduceH`
//! to handle the fractional remainder.

#![allow(dead_code)]
// REASON: alternate horizontal shrink constructors are retained for upcoming planner integration.
#![allow(clippy::needless_range_loop)]
// REASON: indexed loops keep the shrink kernel aligned with packed tile buffers.

use std::marker::PhantomData;

use crate::domain::{
    error::BuildError,
    format::{BandFormat, BandFormatId},
    image::{DemandHint, Region, Tile, TileMut},
    op::{NodeSpec, Op},
    resample::ResampleOp,
};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    uint8x8_t, uint8x8x3_t, uint8x8x4_t, uint8x16_t, uint8x16x3_t, uint8x16x4_t, uint32x2_t,
    vaddq_u32, vcombine_u8, vcombine_u16, vdup_n_u16, vdupq_n_u32, vget_high_u32, vget_lane_u32,
    vget_low_u16, vget_low_u32, vld1_u8, vld1q_u8, vld3q_u8, vld4q_u8, vmovl_u8, vmovl_u16,
    vmovn_u16, vmovn_u32, vmulq_n_u32, vpadd_u32, vpaddlq_u8, vpaddlq_u16, vreinterpret_u16_u8,
    vreinterpret_u32_u8, vrshrn_n_u16, vshrq_n_u32, vst1_lane_u8, vst1_lane_u16, vst1_lane_u32,
    vst1_u8, vst1q_u8, vst3_u8, vst3q_u8, vst4_u8, vst4q_u8,
};

/// Converts a sample to `f64` for accumulation and back, with clamping.
///
/// This is a private implementation detail of the shrink ops — it does NOT live
/// in `ports/` because it expresses a runtime conversion strategy, not a
/// capability contract. It is crate-visible because the pipeline bridges need
/// it in generic bounds, but it remains confined to the resample implementation.
pub(crate) trait ShrinkSample: Copy + 'static {
    fn to_f64(self) -> f64;
    fn from_f64_clamped(v: f64) -> Self;
}

impl ShrinkSample for u8 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    #[inline(always)]
    fn from_f64_clamped(v: f64) -> Self {
        v.round().clamp(f64::from(Self::MIN), f64::from(Self::MAX)) as Self
    }
}

impl ShrinkSample for u16 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    #[inline(always)]
    fn from_f64_clamped(v: f64) -> Self {
        v.round().clamp(f64::from(Self::MIN), f64::from(Self::MAX)) as Self
    }
}

impl ShrinkSample for i16 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    #[inline(always)]
    fn from_f64_clamped(v: f64) -> Self {
        (v + 0.5)
            .clamp(f64::from(Self::MIN), f64::from(Self::MAX))
            .trunc() as Self
    }
}

impl ShrinkSample for u32 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    #[inline(always)]
    fn from_f64_clamped(v: f64) -> Self {
        v.round().clamp(f64::from(Self::MIN), f64::from(Self::MAX)) as Self
    }
}

impl ShrinkSample for i32 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    #[inline(always)]
    fn from_f64_clamped(v: f64) -> Self {
        (v + 0.5)
            .clamp(f64::from(Self::MIN), f64::from(Self::MAX))
            .trunc() as Self
    }
}

impl ShrinkSample for f32 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    #[inline(always)]
    fn from_f64_clamped(v: f64) -> Self {
        v as Self
    }
}

impl ShrinkSample for f64 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline(always)]
    fn from_f64_clamped(v: f64) -> Self {
        v
    }
}

/// Horizontal integer shrink by `factor`.
///
/// Averages `factor` consecutive pixels per output pixel using f64 accumulation
/// with saturating cast back to `F::Sample`. `factor` must be >= 1. Factor=1
/// is identity.
///
/// This is the fast path for large downscales: O(n/`factor`) work vs
/// O(`n*kernel_support`) for `ReduceH`. Use `ShrinkH` followed by `ReduceH` for
/// the fractional remainder.
///
/// # Construction
///
/// ```rust,ignore
/// let op = ShrinkH::<U8>::new(2)?;
/// ```
pub struct ShrinkH<F: BandFormat> {
    factor: u32,
    ceil: bool,
    source_width: Option<usize>,
    _format: PhantomData<F>,
}

fn validate_shrink_h_factor(factor: u32) -> Result<(), BuildError> {
    if factor == 0 {
        return Err(BuildError::SourceHint {
            context: "shrink_h",
            message: "factor must be >= 1".to_string(),
        });
    }

    Ok(())
}

impl<F: BandFormat + Send + Sync> ShrinkH<F> {
    /// Create a new `ShrinkH` that shrinks the image width by `factor`.
    ///
    /// `factor` must be >= 1. Factor=1 is identity (output == input width).
    pub fn new(factor: u32) -> Result<Self, BuildError> {
        Self::new_with_ceil(factor, false)
    }

    /// Create a new `ShrinkH` that shrinks the image width by `factor`,
    /// optionally rounding the output width up.
    pub fn new_with_ceil(factor: u32, ceil: bool) -> Result<Self, BuildError> {
        validate_shrink_h_factor(factor)?;
        debug_assert!(factor >= 1, "ShrinkH: factor must be >= 1");
        Ok(Self {
            factor,
            ceil,
            source_width: None,
            _format: PhantomData,
        })
    }

    /// Return a copy configured with the given `ceil` output-dimension mode.
    #[must_use]
    pub const fn with_ceil(mut self, ceil: bool) -> Self {
        self.ceil = ceil;
        self
    }

    #[must_use]
    /// Returns this value configured with source width.
    pub const fn with_source_width(mut self, source_width: usize) -> Self {
        self.source_width = Some(source_width);
        self
    }
}

#[inline]
fn saturating_scale_i32(value: i32, factor: u32) -> i32 {
    i64::from(value)
        .saturating_mul(i64::from(factor))
        .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

impl<F> Op for ShrinkH<F>
where
    F: BandFormat,
    F::Sample: ShrinkSample + bytemuck::Pod,
{
    type Input = F;
    type Output = F;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        // Horizontal shrink reads full rows; thin-strip demand is correct.
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        let required_width = (output.width as usize).saturating_mul(self.factor as usize);
        if let Some(source_width) = self
            .source_width
            .filter(|&width| output.x == 0 && required_width <= width)
        {
            return Region::new(0, output.y, source_width as u32, output.height);
        }

        Region::new(
            saturating_scale_i32(output.x, self.factor),
            output.y,
            output.width.saturating_mul(self.factor),
            output.height,
        )
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        let factor = self.factor;
        let requested_input_tile_w = tile_w.saturating_mul(factor);
        NodeSpec {
            input_tile_w: self
                .source_width
                .map_or(requested_input_tile_w, |source_width| {
                    requested_input_tile_w.max(source_width as u32)
                }),
            input_tile_h: tile_h,
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        let input_stride = input.region.width as usize;
        if F::ID == BandFormatId::U8 {
            shrink_h_u8(
                self.factor as usize,
                bytemuck::cast_slice(input.data),
                input.bands as usize,
                input_stride,
                output.region.width as usize,
                output.region.height as usize,
                bytemuck::cast_slice_mut(output.data),
            );
            return;
        }

        let factor = self.factor as usize;
        let bands = input.bands as usize;
        let out_w = output.region.width as usize;
        let out_h = output.region.height as usize;
        let inv_factor = 1.0_f64 / factor as f64;
        let input_row_width = input_stride * bands;
        let output_row_width = out_w * bands;

        assert_eq!(input.data.len(), input_row_width * out_h);
        assert_eq!(output.data.len(), output_row_width * out_h);
        assert!(input_stride >= out_w * factor);

        for (input_row, output_row) in input
            .data
            .chunks_exact(input_row_width)
            .zip(output.data.chunks_exact_mut(output_row_width))
        {
            let mut input_offset = 0usize;
            for output_pixel in output_row.chunks_exact_mut(bands) {
                let input_pixel = &input_row[input_offset..input_offset + factor * bands];
                for (band, out_sample) in output_pixel.iter_mut().enumerate() {
                    let mut sum = 0.0_f64;
                    for sample in input_pixel[band..].iter().step_by(bands) {
                        sum += sample.to_f64();
                    }
                    *out_sample = F::Sample::from_f64_clamped(sum * inv_factor);
                }
                input_offset += factor * bands;
            }
        }
    }
}

impl<F> ResampleOp for ShrinkH<F>
where
    F: BandFormat,
    F::Sample: ShrinkSample + bytemuck::Pod,
{
    fn output_width(&self, input_w: u32) -> u32 {
        if self.ceil {
            input_w.div_ceil(self.factor)
        } else {
            input_w / self.factor
        }
    }

    fn output_height(&self, input_h: u32) -> u32 {
        // Horizontal pass: height is unchanged.
        input_h
    }
}

#[inline]
fn shrink_h_u8(
    factor: usize,
    input: &[u8],
    bands: usize,
    input_stride: usize,
    out_w: usize,
    out_h: usize,
    output: &mut [u8],
) {
    if factor == 1 {
        copy_u8_rows(input, bands, input_stride, out_w, out_h, output);
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if factor == 2 && (bands == 1 || bands == 3 || bands == 4) {
            // SAFETY: the factor-2 NEON helpers operate on complete row chunks and handle any scalar tail separately.
            unsafe {
                shrink_h_u8_factor2_neon(input, bands, input_stride, out_w, out_h, output);
            }
            return;
        }
        if (bands == 3 || bands == 4) && factor > 2 {
            if factor >= 16 {
                // Large-factor path: vld3q_u8/vld4q_u8 processes 16 input pixels per iteration.
                // For factor=19 this means 1 SIMD chunk + 3 scalar tail pixels per output pixel,
                // vs 9 vld1_u8 iterations for the generic path — significantly fewer loads.
                // Stage profiling confirms ~24ms (chunked) vs ~27ms (vld1_u8) for ShrinkH x19.
                // For very large factors (≥39), both approaches tie because the per-output-pixel
                // hsum cost (vpadd_u32 + vget_lane_u32) grows proportionally with chunk count,
                // offsetting the load reduction. Threshold calibrated against factors 10/19/39.
                // SAFETY: shrink_h_u8_neon_chunked proves safety via 16-pixel chunk invariant.
                unsafe {
                    shrink_h_u8_neon_chunked(
                        factor,
                        input,
                        bands,
                        input_stride,
                        out_w,
                        out_h,
                        output,
                    );
                }
            } else {
                // Small-factor path (3 ≤ factor ≤ 15): vld1_u8 with 2-at-a-time unrolling.
                // Inner loop is short enough that LLVM cannot auto-vectorize reliably;
                // explicit vld1_u8 + vaddq_u32 accumulation beats the scalar path here.
                // SAFETY: shrink_h_u8_neon handles the last-pixel OOB guard internally.
                unsafe {
                    shrink_h_u8_neon(factor, input, bands, input_stride, out_w, out_h, output);
                }
            }
            return;
        }
    }

    if factor == 2 {
        shrink_h_u8_factor2_scalar(input, bands, input_stride, out_w, out_h, output);
        return;
    }

    if factor == 5 {
        shrink_h_u8_factor5_scalar(input, bands, input_stride, out_w, out_h, output);
        return;
    }

    match bands {
        1 => shrink_h_u8_sequential_1(factor, input, input_stride, out_w, out_h, output),
        3 => shrink_h_u8_sequential_3(factor, input, input_stride, out_w, out_h, output),
        4 => shrink_h_u8_sequential_4(factor, input, input_stride, out_w, out_h, output),
        _ => shrink_h_u8_scalar(factor, input, bands, input_stride, out_w, out_h, output),
    }
}

#[inline]
fn shrink_h_u8_factor2_scalar(
    input: &[u8],
    bands: usize,
    input_stride: usize,
    out_w: usize,
    out_h: usize,
    output: &mut [u8],
) {
    let input_row_width = input_stride * bands;
    let output_row_width = out_w * bands;

    assert_eq!(input.len(), input_row_width * out_h);
    assert_eq!(output.len(), output_row_width * out_h);
    assert!(input_stride >= out_w * 2);

    for (input_row, output_row) in input
        .chunks_exact(input_row_width)
        .zip(output.chunks_exact_mut(output_row_width))
    {
        for (input_pixel, output_pixel) in input_row[..out_w * bands * 2]
            .chunks_exact(bands * 2)
            .zip(output_row.chunks_exact_mut(bands))
        {
            let (left, right) = input_pixel.split_at(bands);
            for ((left_sample, right_sample), out_sample) in
                left.iter().zip(right.iter()).zip(output_pixel.iter_mut())
            {
                let sum = u16::from(*left_sample) + u16::from(*right_sample);
                *out_sample = ((sum + 1) >> 1) as u8;
            }
        }
    }
}

#[inline]
fn shrink_h_u8_scalar(
    factor: usize,
    input: &[u8],
    bands: usize,
    input_stride: usize,
    out_w: usize,
    out_h: usize,
    output: &mut [u8],
) {
    let amend = factor / 2;
    let multiplier = (1_u64 << 32) / ((1_u64 << 8) * factor as u64);
    let pixel_span = factor * bands;
    let input_row_width = input_stride * bands;
    let output_row_width = out_w * bands;

    assert_eq!(input.len(), input_row_width * out_h);
    assert_eq!(output.len(), output_row_width * out_h);
    assert!(input_stride >= out_w * factor);

    for (input_row, output_row) in input
        .chunks_exact(input_row_width)
        .zip(output.chunks_exact_mut(output_row_width))
    {
        let mut input_offset = 0usize;
        for output_pixel in output_row.chunks_exact_mut(bands) {
            let input_pixel = &input_row[input_offset..input_offset + pixel_span];
            for (band, out_sample) in output_pixel.iter_mut().enumerate() {
                let mut sum = amend;
                for sample in input_pixel[band..].iter().step_by(bands) {
                    sum += usize::from(*sample);
                }
                *out_sample = (((sum as u64) * multiplier) >> 24) as u8;
            }
            input_offset += pixel_span;
        }
    }
}

#[inline]
fn shrink_h_u8_factor5_scalar(
    input: &[u8],
    bands: usize,
    input_stride: usize,
    out_w: usize,
    out_h: usize,
    output: &mut [u8],
) {
    const FACTOR: usize = 5;
    const AMEND: u16 = 2;

    let input_row_width = input_stride * bands;
    assert_eq!(input.len(), input_row_width * out_h);
    assert!(input_stride >= out_w * FACTOR);

    for y in 0..out_h {
        let input_row = &input[y * input_row_width..(y + 1) * input_row_width];
        let output_row = &mut output[y * out_w * bands..(y + 1) * out_w * bands];

        match bands {
            1 => {
                let mut src = input_row.as_ptr();
                let mut dst = output_row.as_mut_ptr();
                let pairs = out_w / 2;
                for _ in 0..pairs {
                    // SAFETY: `input_row` contains exactly `out_w * 5` bytes and this loop advances
                    // by 10 input bytes and 2 output bytes per iteration while staying within bounds.
                    unsafe {
                        let sum0 = u16::from(*src)
                            + u16::from(*src.add(1))
                            + u16::from(*src.add(2))
                            + u16::from(*src.add(3))
                            + u16::from(*src.add(4))
                            + AMEND;
                        let sum1 = u16::from(*src.add(5))
                            + u16::from(*src.add(6))
                            + u16::from(*src.add(7))
                            + u16::from(*src.add(8))
                            + u16::from(*src.add(9))
                            + AMEND;
                        *dst = (sum0 / FACTOR as u16) as u8;
                        *dst.add(1) = (sum1 / FACTOR as u16) as u8;
                        src = src.add(FACTOR * 2);
                        dst = dst.add(2);
                    }
                }

                if out_w & 1 != 0 {
                    // SAFETY: one output remains, so reading the final 5 input bytes and writing
                    // one output byte stays within the row slices.
                    unsafe {
                        let sum = u16::from(*src)
                            + u16::from(*src.add(1))
                            + u16::from(*src.add(2))
                            + u16::from(*src.add(3))
                            + u16::from(*src.add(4))
                            + AMEND;
                        *dst = (sum / FACTOR as u16) as u8;
                    }
                }
            }
            3 => {
                let mut src = input_row.as_ptr();
                let mut dst = output_row.as_mut_ptr();
                let pairs = out_w / 2;
                for _ in 0..pairs {
                    // SAFETY: `input_row` contains exactly `out_w * 15` bytes and this loop advances
                    // by 30 input bytes and 6 output bytes per iteration while staying within bounds.
                    unsafe {
                        let red0 = u16::from(*src)
                            + u16::from(*src.add(3))
                            + u16::from(*src.add(6))
                            + u16::from(*src.add(9))
                            + u16::from(*src.add(12))
                            + AMEND;
                        let green0 = u16::from(*src.add(1))
                            + u16::from(*src.add(4))
                            + u16::from(*src.add(7))
                            + u16::from(*src.add(10))
                            + u16::from(*src.add(13))
                            + AMEND;
                        let blue0 = u16::from(*src.add(2))
                            + u16::from(*src.add(5))
                            + u16::from(*src.add(8))
                            + u16::from(*src.add(11))
                            + u16::from(*src.add(14))
                            + AMEND;
                        let red1 = u16::from(*src.add(15))
                            + u16::from(*src.add(18))
                            + u16::from(*src.add(21))
                            + u16::from(*src.add(24))
                            + u16::from(*src.add(27))
                            + AMEND;
                        let green1 = u16::from(*src.add(16))
                            + u16::from(*src.add(19))
                            + u16::from(*src.add(22))
                            + u16::from(*src.add(25))
                            + u16::from(*src.add(28))
                            + AMEND;
                        let blue1 = u16::from(*src.add(17))
                            + u16::from(*src.add(20))
                            + u16::from(*src.add(23))
                            + u16::from(*src.add(26))
                            + u16::from(*src.add(29))
                            + AMEND;

                        *dst = (red0 / FACTOR as u16) as u8;
                        *dst.add(1) = (green0 / FACTOR as u16) as u8;
                        *dst.add(2) = (blue0 / FACTOR as u16) as u8;
                        *dst.add(3) = (red1 / FACTOR as u16) as u8;
                        *dst.add(4) = (green1 / FACTOR as u16) as u8;
                        *dst.add(5) = (blue1 / FACTOR as u16) as u8;

                        src = src.add(FACTOR * 6);
                        dst = dst.add(6);
                    }
                }

                if out_w & 1 != 0 {
                    // SAFETY: one RGB output remains, so reading the final 15 input bytes and writing
                    // the final 3 output bytes stays within the row slices.
                    unsafe {
                        let red = u16::from(*src)
                            + u16::from(*src.add(3))
                            + u16::from(*src.add(6))
                            + u16::from(*src.add(9))
                            + u16::from(*src.add(12))
                            + AMEND;
                        let green = u16::from(*src.add(1))
                            + u16::from(*src.add(4))
                            + u16::from(*src.add(7))
                            + u16::from(*src.add(10))
                            + u16::from(*src.add(13))
                            + AMEND;
                        let blue = u16::from(*src.add(2))
                            + u16::from(*src.add(5))
                            + u16::from(*src.add(8))
                            + u16::from(*src.add(11))
                            + u16::from(*src.add(14))
                            + AMEND;

                        *dst = (red / FACTOR as u16) as u8;
                        *dst.add(1) = (green / FACTOR as u16) as u8;
                        *dst.add(2) = (blue / FACTOR as u16) as u8;
                    }
                }
            }
            4 => {
                let mut src = input_row.as_ptr();
                let mut dst = output_row.as_mut_ptr();
                let pairs = out_w / 2;
                for _ in 0..pairs {
                    // SAFETY: `input_row` contains exactly `out_w * 20` bytes and this loop advances
                    // by 40 input bytes and 8 output bytes per iteration while staying within bounds.
                    unsafe {
                        let c0_0 = u16::from(*src)
                            + u16::from(*src.add(4))
                            + u16::from(*src.add(8))
                            + u16::from(*src.add(12))
                            + u16::from(*src.add(16))
                            + AMEND;
                        let c1_0 = u16::from(*src.add(1))
                            + u16::from(*src.add(5))
                            + u16::from(*src.add(9))
                            + u16::from(*src.add(13))
                            + u16::from(*src.add(17))
                            + AMEND;
                        let c2_0 = u16::from(*src.add(2))
                            + u16::from(*src.add(6))
                            + u16::from(*src.add(10))
                            + u16::from(*src.add(14))
                            + u16::from(*src.add(18))
                            + AMEND;
                        let c3_0 = u16::from(*src.add(3))
                            + u16::from(*src.add(7))
                            + u16::from(*src.add(11))
                            + u16::from(*src.add(15))
                            + u16::from(*src.add(19))
                            + AMEND;
                        let c0_1 = u16::from(*src.add(20))
                            + u16::from(*src.add(24))
                            + u16::from(*src.add(28))
                            + u16::from(*src.add(32))
                            + u16::from(*src.add(36))
                            + AMEND;
                        let c1_1 = u16::from(*src.add(21))
                            + u16::from(*src.add(25))
                            + u16::from(*src.add(29))
                            + u16::from(*src.add(33))
                            + u16::from(*src.add(37))
                            + AMEND;
                        let c2_1 = u16::from(*src.add(22))
                            + u16::from(*src.add(26))
                            + u16::from(*src.add(30))
                            + u16::from(*src.add(34))
                            + u16::from(*src.add(38))
                            + AMEND;
                        let c3_1 = u16::from(*src.add(23))
                            + u16::from(*src.add(27))
                            + u16::from(*src.add(31))
                            + u16::from(*src.add(35))
                            + u16::from(*src.add(39))
                            + AMEND;

                        *dst = (c0_0 / FACTOR as u16) as u8;
                        *dst.add(1) = (c1_0 / FACTOR as u16) as u8;
                        *dst.add(2) = (c2_0 / FACTOR as u16) as u8;
                        *dst.add(3) = (c3_0 / FACTOR as u16) as u8;
                        *dst.add(4) = (c0_1 / FACTOR as u16) as u8;
                        *dst.add(5) = (c1_1 / FACTOR as u16) as u8;
                        *dst.add(6) = (c2_1 / FACTOR as u16) as u8;
                        *dst.add(7) = (c3_1 / FACTOR as u16) as u8;

                        src = src.add(FACTOR * 8);
                        dst = dst.add(8);
                    }
                }

                if out_w & 1 != 0 {
                    // SAFETY: one RGBA output remains, so reading the final 20 input bytes and writing
                    // the final 4 output bytes stays within the row slices.
                    unsafe {
                        let c0 = u16::from(*src)
                            + u16::from(*src.add(4))
                            + u16::from(*src.add(8))
                            + u16::from(*src.add(12))
                            + u16::from(*src.add(16))
                            + AMEND;
                        let c1 = u16::from(*src.add(1))
                            + u16::from(*src.add(5))
                            + u16::from(*src.add(9))
                            + u16::from(*src.add(13))
                            + u16::from(*src.add(17))
                            + AMEND;
                        let c2 = u16::from(*src.add(2))
                            + u16::from(*src.add(6))
                            + u16::from(*src.add(10))
                            + u16::from(*src.add(14))
                            + u16::from(*src.add(18))
                            + AMEND;
                        let c3 = u16::from(*src.add(3))
                            + u16::from(*src.add(7))
                            + u16::from(*src.add(11))
                            + u16::from(*src.add(15))
                            + u16::from(*src.add(19))
                            + AMEND;

                        *dst = (c0 / FACTOR as u16) as u8;
                        *dst.add(1) = (c1 / FACTOR as u16) as u8;
                        *dst.add(2) = (c2 / FACTOR as u16) as u8;
                        *dst.add(3) = (c3 / FACTOR as u16) as u8;
                    }
                }
            }
            _ => shrink_h_u8_scalar(FACTOR, input_row, bands, input_stride, out_w, 1, output_row),
        }
    }
}

#[inline]
fn copy_u8_rows(
    input: &[u8],
    bands: usize,
    input_stride: usize,
    out_w: usize,
    out_h: usize,
    output: &mut [u8],
) {
    let input_row_width = input_stride * bands;
    let output_row_width = out_w * bands;

    assert_eq!(input.len(), input_row_width * out_h);
    assert_eq!(output.len(), output_row_width * out_h);
    assert!(input_stride >= out_w);

    for (input_row, output_row) in input
        .chunks_exact(input_row_width)
        .zip(output.chunks_exact_mut(output_row_width))
    {
        output_row.copy_from_slice(&input_row[..output_row_width]);
    }
}

#[inline]
fn shrink_h_u8_sequential_1(
    factor: usize,
    input: &[u8],
    input_stride: usize,
    out_w: usize,
    out_h: usize,
    output: &mut [u8],
) {
    let amend = factor / 2;
    let multiplier = (1_u64 << 32) / ((1_u64 << 8) * factor as u64);
    let input_row_width = input_stride;

    assert_eq!(input.len(), input_row_width * out_h);
    assert_eq!(output.len(), out_w * out_h);
    assert!(input_stride >= out_w * factor);

    for y in 0..out_h {
        let input_row = &input[y * input_row_width..(y + 1) * input_row_width];
        let output_row = &mut output[y * out_w..(y + 1) * out_w];
        let mut src = 0usize;

        for out_sample in output_row.iter_mut() {
            let end = src + factor;
            let mut sum = amend;
            while src < end {
                sum += usize::from(input_row[src]);
                src += 1;
            }
            *out_sample = (((sum as u64) * multiplier) >> 24) as u8;
        }
    }
}

#[inline]
fn shrink_h_u8_sequential_3(
    factor: usize,
    input: &[u8],
    input_stride: usize,
    out_w: usize,
    out_h: usize,
    output: &mut [u8],
) {
    let amend = factor / 2;
    let multiplier = (1_u64 << 32) / ((1_u64 << 8) * factor as u64);
    let input_row_width = input_stride * 3;
    let output_row_width = out_w * 3;

    assert_eq!(input.len(), input_row_width * out_h);
    assert_eq!(output.len(), output_row_width * out_h);
    assert!(input_stride >= out_w * factor);

    for y in 0..out_h {
        let input_row = &input[y * input_row_width..(y + 1) * input_row_width];
        let output_row = &mut output[y * output_row_width..(y + 1) * output_row_width];
        let mut src = 0usize;

        for output_pixel in output_row.chunks_exact_mut(3) {
            let end = src + factor * 3;
            let mut red = amend;
            let mut green = amend;
            let mut blue = amend;
            while src < end {
                red += usize::from(input_row[src]);
                green += usize::from(input_row[src + 1]);
                blue += usize::from(input_row[src + 2]);
                src += 3;
            }
            output_pixel[0] = (((red as u64) * multiplier) >> 24) as u8;
            output_pixel[1] = (((green as u64) * multiplier) >> 24) as u8;
            output_pixel[2] = (((blue as u64) * multiplier) >> 24) as u8;
        }
    }
}

#[inline]
fn shrink_h_u8_sequential_4(
    factor: usize,
    input: &[u8],
    input_stride: usize,
    out_w: usize,
    out_h: usize,
    output: &mut [u8],
) {
    let amend = factor / 2;
    let multiplier = (1_u64 << 32) / ((1_u64 << 8) * factor as u64);
    let input_row_width = input_stride * 4;
    let output_row_width = out_w * 4;

    assert_eq!(input.len(), input_row_width * out_h);
    assert_eq!(output.len(), output_row_width * out_h);
    assert!(input_stride >= out_w * factor);

    for y in 0..out_h {
        let input_row = &input[y * input_row_width..(y + 1) * input_row_width];
        let output_row = &mut output[y * output_row_width..(y + 1) * output_row_width];
        let mut src = 0usize;

        for output_pixel in output_row.chunks_exact_mut(4) {
            let end = src + factor * 4;
            let mut c0 = amend;
            let mut c1 = amend;
            let mut c2 = amend;
            let mut c3 = amend;
            while src < end {
                c0 += usize::from(input_row[src]);
                c1 += usize::from(input_row[src + 1]);
                c2 += usize::from(input_row[src + 2]);
                c3 += usize::from(input_row[src + 3]);
                src += 4;
            }
            output_pixel[0] = (((c0 as u64) * multiplier) >> 24) as u8;
            output_pixel[1] = (((c1 as u64) * multiplier) >> 24) as u8;
            output_pixel[2] = (((c2 as u64) * multiplier) >> 24) as u8;
            output_pixel[3] = (((c3 as u64) * multiplier) >> 24) as u8;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
// SAFETY: caller must ensure `input` has `out_h * input_stride * bands` bytes,
// `output` has `out_h * out_w * bands` bytes, and `bands` is 3 or 4.
// The function handles the case where the last output pixel's vld1_u8 read would
// exceed the input row by falling back to byte-by-byte scalar for that pixel.
unsafe fn shrink_h_u8_neon(
    factor: usize,
    input: &[u8],
    bands: usize,
    input_stride: usize,
    out_w: usize,
    out_h: usize,
    output: &mut [u8],
) {
    // Mirrors libvips shrinkh_hwy.cpp:
    //   sum0 = amend + sum_of(PromoteTo(du32, LoadU(du8x32, p))[0..bands])
    //   output = (sum0 * multiplier) >> 24
    // where multiplier = 2^32 / (256 * factor).
    //
    // `vld1_u8(p)` loads 8 bytes → uint8x8_t.
    // `vmovl_u8` → uint16x8_t; `vget_low_u16` → uint16x4_t (lanes 0-3);
    // `vmovl_u16` → uint32x4_t with lanes [B0, B1, B2, B3] (= [R, G, B, junk/A]).
    // For bands=3 lane-3 is junk; only lanes 0-2 are stored.
    let amend = (factor / 2) as u32;
    let multiplier = ((1u64 << 32) / ((1u64 << 8) * factor as u64)) as u32;
    let input_row_bytes = input_stride * bands;
    let output_row_bytes = out_w * bands;

    // How many pixels can safely use NEON (vld1_u8 reads 8 bytes, so the last read
    // at offset (x*factor + factor-1)*bands must satisfy offset+7 < input_row_bytes).
    // For pixel x: last_read_start = (x*factor + factor-1)*bands.
    // Safe condition: (x*factor + factor-1)*bands + 7 < input_row_bytes.
    // Equivalently: x < (input_row_bytes - (factor-1)*bands - 7) / (factor*bands).
    // Compute conservatively: all pixels where the last read fits. At worst, only the
    // final output pixel falls back to scalar (the previous pixel always has ≥factor*bands
    // bytes of slack from the existence of a next pixel, and factor*bands ≥ 9 when
    // factor≥3 and bands≥3).
    let neon_count = if out_w == 0 {
        0
    } else {
        let last_read_start = (out_w.wrapping_sub(1) * factor + factor - 1) * bands;
        if last_read_start + 7 < input_row_bytes {
            out_w // All pixels safe for NEON.
        } else {
            out_w - 1 // Last pixel needs scalar fallback.
        }
    };

    for y in 0..out_h {
        // SAFETY: y < out_h; `input` contains `out_h * input_row_bytes` bytes by caller contract,
        // so advancing to the start of row `y` stays in-bounds and preserves pointer alignment.
        let row_in = unsafe { input.as_ptr().add(y * input_row_bytes) };
        // SAFETY: y < out_h; `output` contains `out_h * output_row_bytes` writable bytes, so this
        // points to the start of row `y` within the same allocation with the original lifetime.
        let row_out = unsafe { output.as_mut_ptr().add(y * output_row_bytes) };

        // NEON path for the first `neon_count` output pixels.
        for x in 0..neon_count {
            // SAFETY: neon_count guarantees vld1_u8 stays within the input row.
            unsafe {
                let mut p = row_in.add(x * factor * bands);
                let mut sum = vdupq_n_u32(amend);

                let mut xx = 0usize;
                while xx + 2 <= factor {
                    let pix0 = vmovl_u16(vget_low_u16(vmovl_u8(vld1_u8(p))));
                    p = p.add(bands);
                    let pix1 = vmovl_u16(vget_low_u16(vmovl_u8(vld1_u8(p))));
                    p = p.add(bands);
                    sum = vaddq_u32(sum, vaddq_u32(pix0, pix1));
                    xx += 2;
                }
                if xx < factor {
                    let pix0 = vmovl_u16(vget_low_u16(vmovl_u8(vld1_u8(p))));
                    sum = vaddq_u32(sum, pix0);
                }

                sum = vmulq_n_u32(sum, multiplier);
                sum = vshrq_n_u32(sum, 24);

                // Narrow uint32x4_t → uint8x8_t without NEON→integer register crossings:
                //   vmovn_u32: [R,G,B,junk] u32 → [R,G,B,junk] u16 (keeps low 16 bits)
                //   vcombine_u16 + vmovn_u16: → [R,G,B,junk, 0,0,0,0] u8
                //   vst1_lane_u16 / vst1_lane_u8: store bytes from NEON register directly.
                let narrow16 = vmovn_u32(sum);
                let narrow8 = vmovn_u16(vcombine_u16(narrow16, vdup_n_u16(0)));
                let q = row_out.add(x * bands);
                if bands == 4 {
                    // SAFETY: unaligned 4-byte store; vst1_lane_u32 handles non-aligned writes.
                    vst1_lane_u32::<0>(q.cast(), vreinterpret_u32_u8(narrow8));
                } else {
                    // Store 2 bytes (R, G) then 1 byte (B) — no integer pipeline crossing.
                    vst1_lane_u16::<0>(q.cast(), vreinterpret_u16_u8(narrow8));
                    vst1_lane_u8::<2>(q.add(2), narrow8);
                }
            }
        }

        // Scalar fallback for the last pixel when vld1_u8 would overread (at most 1 pixel/row).
        for x in neon_count..out_w {
            // SAFETY: x < out_w; strides validated by caller contract.
            unsafe {
                let p = row_in.add(x * factor * bands);
                let q = row_out.add(x * bands);
                let mut sums = [amend; 4];
                for step in 0..factor {
                    for b in 0..bands {
                        sums[b] += u32::from(*p.add(step * bands + b));
                    }
                }
                let finalize = |s: u32| ((u64::from(s) * u64::from(multiplier)) >> 24) as u8;
                *q = finalize(sums[0]);
                *q.add(1) = finalize(sums[1]);
                *q.add(2) = finalize(sums[2]);
                if bands == 4 {
                    *q.add(3) = finalize(sums[3]);
                }
            }
        }
    }
}

/// NEON shrink for bands=3/4, factor ≥ 16.
///
/// Uses `vld3q_u8`/`vld4q_u8` to de-interleave and load 16 input pixels at a time, then
/// `vpaddlq_u8` → `vpaddlq_u16` to efficiently reduce them into a running `uint32x4_t`
/// accumulator. For `factor=19` this means 1 SIMD chunk + 3 scalar tail pixels per output
/// pixel — significantly fewer loads than the 9 × `vld1_u8` iterations of the generic path.
/// Stage profiling: ~24ms (chunked) vs ~27ms (`vld1_u8`) for `ShrinkH` x19.
///
/// Safety proof for `vld3q_u8:` when `remaining >= 16`, the loop reads exactly 48 bytes from `p`.
/// At entry to each iteration `p = base + (factor − remaining) × bands`, so the last byte is at
/// `p + 47 = base + (factor − remaining) × bands + 47`.  The pixel's territory ends at
/// `base + factor × bands − 1`.  The inequality `p + 47 ≤ territory_end` simplifies to
/// `48 ≤ remaining × bands`, which holds whenever `remaining ≥ 16` and `bands ≥ 3`. ∎
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn shrink_h_u8_neon_chunked(
    factor: usize,
    input: &[u8],
    bands: usize,
    input_stride: usize,
    out_w: usize,
    out_h: usize,
    output: &mut [u8],
) {
    let amend = (factor / 2) as u32;
    let multiplier = ((1u64 << 32) / ((1u64 << 8) * factor as u64)) as u32;
    let input_row_bytes = input_stride * bands;
    let output_row_bytes = out_w * bands;

    for y in 0..out_h {
        // SAFETY: `y < out_h`, so both row offsets stay within the input/output slices guaranteed
        // by the caller contract for this kernel.
        let (row_in, row_out) = unsafe {
            (
                input.as_ptr().add(y * input_row_bytes),
                output.as_mut_ptr().add(y * output_row_bytes),
            )
        };

        for x in 0..out_w {
            // SAFETY: `x < out_w`, so the start of this reduction window stays inside the current
            // input row described by `row_in`.
            let base = unsafe { row_in.add(x * factor * bands) };
            let mut p = base;
            let mut remaining = factor;

            let mut acc_r = vdupq_n_u32(0u32);
            let mut acc_g = vdupq_n_u32(0u32);
            let mut acc_b = vdupq_n_u32(0u32);
            let mut acc_a = vdupq_n_u32(0u32);

            if bands == 3 {
                // SAFETY: loop invariant remaining ≥ 16 guarantees p+47 ≤ territory_end (see above).
                while remaining >= 16 {
                    let rgb = unsafe { vld3q_u8(p) };
                    acc_r = vaddq_u32(acc_r, vpaddlq_u16(vpaddlq_u8(rgb.0)));
                    acc_g = vaddq_u32(acc_g, vpaddlq_u16(vpaddlq_u8(rgb.1)));
                    acc_b = vaddq_u32(acc_b, vpaddlq_u16(vpaddlq_u8(rgb.2)));
                    // SAFETY: consuming exactly the 48 bytes just loaded keeps `p` within the
                    // same validated reduction window.
                    p = unsafe { p.add(48) }; // 16 pixels × 3 bytes
                    remaining -= 16;
                }
            } else {
                // bands == 4 (RGBA)
                // SAFETY: loop invariant remaining ≥ 16 guarantees p+63 ≤ territory_end
                // because 64 = 16×4 ≤ remaining×bands when remaining ≥ 16 and bands = 4.
                while remaining >= 16 {
                    let rgba = unsafe { vld4q_u8(p) };
                    acc_r = vaddq_u32(acc_r, vpaddlq_u16(vpaddlq_u8(rgba.0)));
                    acc_g = vaddq_u32(acc_g, vpaddlq_u16(vpaddlq_u8(rgba.1)));
                    acc_b = vaddq_u32(acc_b, vpaddlq_u16(vpaddlq_u8(rgba.2)));
                    acc_a = vaddq_u32(acc_a, vpaddlq_u16(vpaddlq_u8(rgba.3)));
                    // SAFETY: consuming exactly the 64 bytes just loaded keeps `p` within the
                    // same validated reduction window.
                    p = unsafe { p.add(64) }; // 16 pixels × 4 bytes
                    remaining -= 16;
                }
            }

            // Horizontal sum of uint32x4_t → u32 (once per channel, once per output pixel).
            macro_rules! hsum {
                ($v:expr) => {{
                    let lo: uint32x2_t = vget_low_u32($v);
                    let hi: uint32x2_t = vget_high_u32($v);
                    let s1: uint32x2_t = vpadd_u32(lo, hi);
                    let s2: uint32x2_t = vpadd_u32(s1, s1);
                    vget_lane_u32::<0>(s2)
                }};
            }

            let mut sum_r = hsum!(acc_r) + amend;
            let mut sum_g = hsum!(acc_g) + amend;
            let mut sum_b = hsum!(acc_b) + amend;
            let mut sum_a = if bands == 4 { hsum!(acc_a) + amend } else { 0 };

            // Scalar tail for remaining < 16 pixels not covered by the chunk loop.
            // SAFETY: `p` starts within this window and advances by exactly `bands` bytes per
            // iteration, so each channel load remains within the remaining source territory.
            unsafe {
                for _ in 0..remaining {
                    sum_r += u32::from(*p);
                    sum_g += u32::from(*p.add(1));
                    sum_b += u32::from(*p.add(2));
                    if bands == 4 {
                        sum_a += u32::from(*p.add(3));
                    }
                    p = p.add(bands);
                }
            }

            let finalize = |s: u32| ((u64::from(s) * u64::from(multiplier)) >> 24) as u8;
            // SAFETY: `x < out_w`, so the output pixel at `q` and its channel offsets fit within
            // the writable output row for this call.
            unsafe {
                let q = row_out.add(x * bands);
                *q = finalize(sum_r);
                *q.add(1) = finalize(sum_g);
                *q.add(2) = finalize(sum_b);
                if bands == 4 {
                    *q.add(3) = finalize(sum_a);
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn shrink_h_u8_factor2_neon(
    input: &[u8],
    bands: usize,
    input_stride: usize,
    out_w: usize,
    out_h: usize,
    output: &mut [u8],
) {
    let input_row_width = input_stride * bands;

    for y in 0..out_h {
        let input_row = &input[y * input_row_width..(y + 1) * input_row_width];
        let output_row = &mut output[y * out_w * bands..(y + 1) * out_w * bands];

        match bands {
            1 => {
                let simd16_outputs = out_w / 16;
                for chunk in 0..simd16_outputs {
                    let src = input_row.as_ptr().wrapping_add(chunk * 32);
                    let dst = output_row.as_mut_ptr().wrapping_add(chunk * 16);
                    // SAFETY: `src` points to 32 contiguous input bytes for 16 factor-2 outputs.
                    let pixels_lo = unsafe { vld1q_u8(src) };
                    // SAFETY: second 16-byte load stays within the same 32-byte chunk.
                    let pixels_hi = unsafe { vld1q_u8(src.add(16)) };
                    // SAFETY: `shrink_h_u8_factor2_neon` requires NEON support, so combining the
                    // two 8-lane averages into one 16-lane vector is valid here.
                    let avg = unsafe {
                        vcombine_u8(
                            average_adjacent_pairs_u8x16(pixels_lo),
                            average_adjacent_pairs_u8x16(pixels_hi),
                        )
                    };
                    // SAFETY: `dst` points to 16 writable output bytes for this chunk.
                    unsafe { vst1q_u8(dst, avg) };
                }

                let processed_outputs = simd16_outputs * 16;
                let remaining_outputs = out_w - processed_outputs;
                let simd8_outputs = remaining_outputs / 8;
                for chunk in 0..simd8_outputs {
                    let src = input_row
                        .as_ptr()
                        .wrapping_add(processed_outputs * 2 + chunk * 16);
                    let dst = output_row
                        .as_mut_ptr()
                        .wrapping_add(processed_outputs + chunk * 8);
                    // SAFETY: `src` points to 16 contiguous input bytes for 8 factor-2 outputs.
                    let pixels = unsafe { vld1q_u8(src) };
                    let avg = average_adjacent_pairs_u8x16(pixels);
                    // SAFETY: `dst` points to 8 writable output bytes for this chunk.
                    unsafe { vst1_u8(dst, avg) };
                }

                let dst_start = processed_outputs + simd8_outputs * 8;
                if dst_start < out_w {
                    let src_start = dst_start * 2;
                    shrink_h_u8_factor2_scalar(
                        &input_row[src_start..],
                        1,
                        input_row[src_start..].len(),
                        out_w - dst_start,
                        1,
                        &mut output_row[dst_start..],
                    );
                }
            }
            3 => {
                let simd16_outputs = out_w / 16;
                for chunk in 0..simd16_outputs {
                    let src = input_row.as_ptr().wrapping_add(chunk * 96);
                    let dst = output_row.as_mut_ptr().wrapping_add(chunk * 48);
                    // SAFETY: `src` points to 32 interleaved RGB pixels (96 bytes).
                    let pixels_lo: uint8x16x3_t = unsafe { vld3q_u8(src) };
                    // SAFETY: second 48-byte load stays within the same 96-byte chunk.
                    let pixels_hi: uint8x16x3_t = unsafe { vld3q_u8(src.add(48)) };
                    // SAFETY: `shrink_h_u8_factor2_neon` requires NEON support, so combining each
                    // pair of 8-lane channel averages into 16-lane vectors is valid here.
                    let avg: uint8x16x3_t = unsafe {
                        uint8x16x3_t(
                            vcombine_u8(
                                average_adjacent_pairs_u8x16(pixels_lo.0),
                                average_adjacent_pairs_u8x16(pixels_hi.0),
                            ),
                            vcombine_u8(
                                average_adjacent_pairs_u8x16(pixels_lo.1),
                                average_adjacent_pairs_u8x16(pixels_hi.1),
                            ),
                            vcombine_u8(
                                average_adjacent_pairs_u8x16(pixels_lo.2),
                                average_adjacent_pairs_u8x16(pixels_hi.2),
                            ),
                        )
                    };
                    // SAFETY: `dst` points to space for 16 interleaved RGB output pixels.
                    unsafe { vst3q_u8(dst, avg) };
                }

                let processed_outputs = simd16_outputs * 16;
                let remaining_outputs = out_w - processed_outputs;
                let simd8_outputs = remaining_outputs / 8;
                for chunk in 0..simd8_outputs {
                    let src = input_row
                        .as_ptr()
                        .wrapping_add(processed_outputs * 6 + chunk * 48);
                    let dst = output_row
                        .as_mut_ptr()
                        .wrapping_add(processed_outputs * 3 + chunk * 24);
                    // SAFETY: `src` points to 16 interleaved RGB pixels (48 bytes).
                    let pixels: uint8x16x3_t = unsafe { vld3q_u8(src) };
                    let avg: uint8x8x3_t = uint8x8x3_t(
                        average_adjacent_pairs_u8x16(pixels.0),
                        average_adjacent_pairs_u8x16(pixels.1),
                        average_adjacent_pairs_u8x16(pixels.2),
                    );
                    // SAFETY: `dst` points to space for 8 interleaved RGB output pixels.
                    unsafe { vst3_u8(dst, avg) };
                }

                let dst_start = processed_outputs + simd8_outputs * 8;
                if dst_start < out_w {
                    let src_start = dst_start * 6;
                    shrink_h_u8_factor2_scalar(
                        &input_row[src_start..],
                        3,
                        input_row[src_start..].len() / 3,
                        out_w - dst_start,
                        1,
                        &mut output_row[dst_start * 3..],
                    );
                }
            }
            4 => {
                let simd16_outputs = out_w / 16;
                for chunk in 0..simd16_outputs {
                    let src = input_row.as_ptr().wrapping_add(chunk * 128);
                    let dst = output_row.as_mut_ptr().wrapping_add(chunk * 64);
                    // SAFETY: `src` points to 32 interleaved RGBA pixels (128 bytes).
                    let pixels_lo: uint8x16x4_t = unsafe { vld4q_u8(src) };
                    // SAFETY: second 64-byte load stays within the same 128-byte chunk.
                    let pixels_hi: uint8x16x4_t = unsafe { vld4q_u8(src.add(64)) };
                    // SAFETY: `shrink_h_u8_factor2_neon` requires NEON support, so combining each
                    // pair of 8-lane channel averages into 16-lane vectors is valid here.
                    let avg: uint8x16x4_t = unsafe {
                        uint8x16x4_t(
                            vcombine_u8(
                                average_adjacent_pairs_u8x16(pixels_lo.0),
                                average_adjacent_pairs_u8x16(pixels_hi.0),
                            ),
                            vcombine_u8(
                                average_adjacent_pairs_u8x16(pixels_lo.1),
                                average_adjacent_pairs_u8x16(pixels_hi.1),
                            ),
                            vcombine_u8(
                                average_adjacent_pairs_u8x16(pixels_lo.2),
                                average_adjacent_pairs_u8x16(pixels_hi.2),
                            ),
                            vcombine_u8(
                                average_adjacent_pairs_u8x16(pixels_lo.3),
                                average_adjacent_pairs_u8x16(pixels_hi.3),
                            ),
                        )
                    };
                    // SAFETY: `dst` points to space for 16 interleaved RGBA output pixels.
                    unsafe { vst4q_u8(dst, avg) };
                }

                let processed_outputs = simd16_outputs * 16;
                let remaining_outputs = out_w - processed_outputs;
                let simd8_outputs = remaining_outputs / 8;
                for chunk in 0..simd8_outputs {
                    let src = input_row
                        .as_ptr()
                        .wrapping_add(processed_outputs * 8 + chunk * 64);
                    let dst = output_row
                        .as_mut_ptr()
                        .wrapping_add(processed_outputs * 4 + chunk * 32);
                    // SAFETY: `src` points to 16 interleaved RGBA pixels (64 bytes).
                    let pixels: uint8x16x4_t = unsafe { vld4q_u8(src) };
                    let avg: uint8x8x4_t = uint8x8x4_t(
                        average_adjacent_pairs_u8x16(pixels.0),
                        average_adjacent_pairs_u8x16(pixels.1),
                        average_adjacent_pairs_u8x16(pixels.2),
                        average_adjacent_pairs_u8x16(pixels.3),
                    );
                    // SAFETY: `dst` points to space for 8 interleaved RGBA output pixels.
                    unsafe { vst4_u8(dst, avg) };
                }

                let dst_start = processed_outputs + simd8_outputs * 8;
                if dst_start < out_w {
                    let src_start = dst_start * 8;
                    shrink_h_u8_factor2_scalar(
                        &input_row[src_start..],
                        4,
                        input_row[src_start..].len() / 4,
                        out_w - dst_start,
                        1,
                        &mut output_row[dst_start * 4..],
                    );
                }
            }
            _ => {
                debug_assert!(
                    false,
                    "non-specialized band counts must use the scalar fallback"
                );
                return;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn average_adjacent_pairs_u8x16(pixels: uint8x16_t) -> uint8x8_t {
    // SAFETY: this helper is only called from the aarch64 NEON fast path.
    unsafe { vrshrn_n_u16::<1>(vpaddlq_u8(pixels)) }
}

pub(crate) struct ShrinkHBridge<F: BandFormat>
where
    F::Sample: bytemuck::Pod + ShrinkSample,
{
    inner: crate::domain::op::OperationBridge<ShrinkH<F>>,
}

impl<F: BandFormat> ShrinkHBridge<F>
where
    F::Sample: bytemuck::Pod + ShrinkSample,
{
    pub fn new(factor: u32, bands: u32) -> Result<Self, BuildError> {
        Self::new_with_ceil(factor, false, bands)
    }

    pub fn new_with_ceil(factor: u32, ceil: bool, bands: u32) -> Result<Self, BuildError> {
        Self::new_with_ceil_and_source_width(factor, ceil, bands, None)
    }

    pub fn new_with_ceil_and_source_width(
        factor: u32,
        ceil: bool,
        bands: u32,
        source_width: Option<u32>,
    ) -> Result<Self, BuildError> {
        let op = ShrinkH::new_with_ceil(factor, ceil)?;
        let op = if let Some(width) = source_width {
            op.with_source_width(width as usize)
        } else {
            op
        };

        Ok(Self {
            inner: crate::domain::op::OperationBridge::new(op, bands),
        })
    }
}

impl<F: BandFormat> crate::domain::op::DynOperation for ShrinkHBridge<F>
where
    F::Sample: bytemuck::Pod + ShrinkSample + Send,
{
    fn input_format(&self) -> crate::domain::format::BandFormatId {
        self.inner.input_format()
    }

    fn output_format(&self) -> crate::domain::format::BandFormatId {
        self.inner.output_format()
    }

    fn bands(&self) -> u32 {
        self.inner.bands()
    }

    fn demand_hint(&self) -> DemandHint {
        self.inner.demand_hint()
    }

    fn required_input_region(&self, output: &Region) -> Region {
        self.inner.required_input_region(output)
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        self.inner.node_spec(tile_w, tile_h)
    }

    fn output_width(&self, input_w: u32) -> u32 {
        self.inner.op.output_width(input_w)
    }

    fn output_height(&self, input_h: u32) -> u32 {
        input_h
    }

    fn dyn_start(&self) -> Box<dyn std::any::Any + Send> {
        self.inner.dyn_start()
    }

    fn dyn_start_with_tile(&self, tile_w: u32, tile_h: u32) -> Box<dyn std::any::Any + Send> {
        self.inner.dyn_start_with_tile(tile_w, tile_h)
    }

    fn dyn_process_region(
        &self,
        state: &mut dyn std::any::Any,
        input: &[u8],
        output: &mut [u8],
        input_region: Region,
        output_region: Region,
    ) {
        self.inner
            .dyn_process_region(state, input, output, input_region, output_region);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::sources::memory::MemorySource;
    use crate::{
        domain::{
            error::BuildError,
            format::{BandFormatId, I16, U8},
            image::{DemandHint, Region, Tile, TileMut},
            op::DynOperation,
        },
        ports::source::ImageSource,
    };
    use proptest::prelude::*;

    fn run_shrinkh<F>(
        factor: u32,
        ceil: bool,
        input_data: &[F::Sample],
        in_w: u32,
        in_h: u32,
        bands: u32,
    ) -> Vec<F::Sample>
    where
        F: BandFormat,
        F::Sample: ShrinkSample + bytemuck::Pod,
    {
        let op = ShrinkH::<F>::new_with_ceil(factor, ceil).unwrap();
        let out_w = op.output_width(in_w);
        let in_region = Region::new(0, 0, in_w, in_h);
        let out_region = Region::new(0, 0, out_w, in_h);
        let input = Tile::<F>::new(in_region, bands, input_data);
        let mut out_data =
            vec![F::Sample::from_f64_clamped(0.0); out_w as usize * in_h as usize * bands as usize];
        let mut output = TileMut::<F>::new(out_region, bands, &mut out_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        out_data
    }

    #[test]
    fn shrinkh_factor2_averages_pairs() {
        // Input: 1 row, 4 pixels, 1 band: [10, 20, 30, 40]
        // Expected output: [15, 35]
        let input = vec![10u8, 20u8, 30u8, 40u8];
        let output = run_shrinkh::<U8>(2, false, &input, 4, 1, 1);
        assert_eq!(output, vec![15u8, 35u8]);
    }

    #[test]
    fn shrinkh_factor2_averages_rgb_pairs() {
        let input = vec![
            10u8, 20u8, 30u8, 20u8, 40u8, 60u8, 30u8, 60u8, 90u8, 40u8, 80u8, 120u8,
        ];
        let output = run_shrinkh::<U8>(2, false, &input, 4, 1, 3);
        assert_eq!(output, vec![15u8, 30u8, 45u8, 35u8, 70u8, 105u8]);
    }

    #[test]
    fn shrinkh_factor5_averages_rgb_groups() {
        let input = vec![
            10u8, 20u8, 30u8, 20u8, 30u8, 40u8, 30u8, 40u8, 50u8, 40u8, 50u8, 60u8, 50u8, 60u8,
            70u8, 100u8, 110u8, 120u8, 110u8, 120u8, 130u8, 120u8, 130u8, 140u8, 130u8, 140u8,
            150u8, 140u8, 150u8, 160u8,
        ];
        let output = run_shrinkh::<U8>(5, false, &input, 10, 1, 3);
        assert_eq!(output, vec![30u8, 40u8, 50u8, 120u8, 130u8, 140u8]);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn shrinkh_generic_neon_matches_scalar_for_thumbnail_factor() {
        let factor = 19usize;
        let out_w = 7usize;
        let out_h = 3usize;

        for bands in [1usize, 3, 4] {
            let len = out_w * out_h * factor * bands;
            let input: Vec<u8> = (0..len).map(|idx| ((idx * 37 + 11) % 251) as u8).collect();
            let mut expected = vec![0u8; out_w * out_h * bands];
            let mut actual = vec![0u8; out_w * out_h * bands];

            let input_stride = out_w * factor;
            shrink_h_u8_scalar(
                factor,
                &input,
                bands,
                input_stride,
                out_w,
                out_h,
                &mut expected,
            );
            shrink_h_u8(
                factor,
                &input,
                bands,
                input_stride,
                out_w,
                out_h,
                &mut actual,
            );

            assert_eq!(actual, expected, "bands={bands}");
        }
    }

    /// Verify that shrink_h_u8_neon produces the same result as scalar for
    /// thumbnail-realistic factors including exact-divisor cases (no stride slack).
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn shrinkh_neon_generic_matches_scalar_for_all_thumbnail_factors() {
        // (factor, out_w, source_width):
        // - Slack cases: source_width mod factor != 0 → all pixels use NEON.
        // - Exact-divisor: source_width mod factor == 0 → last pixel uses scalar fallback.
        let cases: &[(usize, usize, usize)] = &[
            (10, 819, 8192), // 8k→800: slack=2 pixels
            (19, 431, 8192), // 8k→400: slack=3 pixels
            (39, 210, 8192), // 8k→200: slack=2 pixels
            (4, 512, 2048),  // 2k→400: exact divisor, last pixel → scalar
            (8, 256, 2048),  // exact divisor
            (32, 256, 8192), // exact divisor (8192/32=256)
            (3, 100, 300),   // exact divisor, small image
        ];
        for &(factor, out_w, source_width) in cases {
            for bands in [3usize, 4] {
                let out_h = 2usize;
                let row_bytes = source_width * bands;
                let len = out_h * row_bytes;
                let input: Vec<u8> = (0..len).map(|i| ((i * 41 + 7) % 251) as u8).collect();
                let mut expected = vec![0u8; out_w * out_h * bands];
                let mut actual = vec![0u8; out_w * out_h * bands];

                shrink_h_u8_scalar(
                    factor,
                    &input,
                    bands,
                    source_width,
                    out_w,
                    out_h,
                    &mut expected,
                );
                shrink_h_u8(
                    factor,
                    &input,
                    bands,
                    source_width,
                    out_w,
                    out_h,
                    &mut actual,
                );

                assert_eq!(
                    actual, expected,
                    "factor={factor} out_w={out_w} source_w={source_width} bands={bands}"
                );
            }
        }
    }

    #[test]
    fn shrinkh_factor1_is_identity() {
        let input = vec![7u8, 42u8, 100u8, 200u8];
        let output = run_shrinkh::<U8>(1, false, &input, 4, 1, 1);
        assert_eq!(output, input);
    }

    #[test]
    fn shrinkh_output_width() {
        let op = ShrinkH::<U8>::new(2).unwrap();
        assert_eq!(op.output_width(100), 50);
        assert_eq!(op.output_width(5), 2); // integer division
    }

    #[test]
    fn shrinkh_output_width_with_ceil_matches_libvips() {
        let floor = ShrinkH::<U8>::new_with_ceil(3, false).unwrap();
        let ceil = ShrinkH::<U8>::new_with_ceil(3, true).unwrap();
        assert_eq!(floor.output_width(10), 3);
        assert_eq!(ceil.output_width(10), 4);
    }

    #[test]
    fn shrinkh_output_height_unchanged() {
        let op = ShrinkH::<U8>::new(4).unwrap();
        assert_eq!(op.output_height(200), 200);
    }

    #[test]
    fn shrinkh_required_input_region() {
        let op = ShrinkH::<U8>::new(2).unwrap();
        let out_region = Region::new(0, 0, 10, 1);
        let in_region = op.required_input_region(&out_region);
        assert_eq!(in_region.x, 0);
        assert_eq!(in_region.width, 20);
        assert_eq!(in_region.height, 1);
    }

    #[test]
    fn shrinkh_required_input_region_uses_full_row_borrow_when_source_width_is_wider() {
        let op = ShrinkH::<U8>::new(19).unwrap().with_source_width(8192);
        let out_region = Region::new(0, 32, 431, 16);
        assert_eq!(
            op.required_input_region(&out_region),
            Region::new(0, 32, 8192, 16)
        );
    }

    #[test]
    fn shrinkh_required_input_region_falls_back_when_source_width_cannot_cover_request() {
        let op = ShrinkH::<U8>::new_with_ceil(3, true)
            .unwrap()
            .with_source_width(10);
        let out_region = Region::new(0, 0, 4, 1);
        assert_eq!(
            op.required_input_region(&out_region),
            Region::new(0, 0, 12, 1)
        );
    }

    #[test]
    fn shrinkh_required_input_region_saturates_huge_factor() {
        let op = ShrinkH::<U8>::new(u32::MAX).unwrap();
        assert_eq!(
            op.required_input_region(&Region::new(1, 0, 2, 1)),
            Region::new(i32::MAX, 0, u32::MAX, 1)
        );
    }

    #[test]
    fn shrinkh_node_spec() {
        let op = ShrinkH::<U8>::new(3).unwrap();
        let spec = op.node_spec(9, 4);
        assert_eq!(spec.input_tile_w, 27);
        assert_eq!(spec.output_tile_w, 9);
        assert_eq!(spec.input_tile_h, 4);
        assert_eq!(spec.output_tile_h, 4);
    }

    #[test]
    fn shrinkh_uniform_image_shrink2_shrink2_equals_shrink4() {
        // For a uniform image, shrink(2) then shrink(2) == shrink(4) because
        // all pixels are equal and averaging equal values is idempotent.
        let uniform = vec![128u8; 16]; // 16 pixels, 1 row, 1 band

        // shrink by 4 directly
        let direct = run_shrinkh::<U8>(4, false, &uniform, 16, 1, 1);

        // shrink by 2 twice
        let step1 = run_shrinkh::<U8>(2, false, &uniform, 16, 1, 1);
        let step2 = run_shrinkh::<U8>(2, false, &step1, 8, 1, 1);

        assert_eq!(
            direct, step2,
            "shrink(4) must equal shrink(2) applied twice for a uniform image"
        );
    }

    #[test]
    fn shrink_sample_conversions_cover_all_formats() {
        assert_eq!(u8::to_f64(3), 3.0);
        assert_eq!(u8::from_f64_clamped(260.0), 255);
        assert_eq!(u16::to_f64(7), 7.0);
        assert_eq!(u16::from_f64_clamped(70000.0), u16::MAX);
        assert_eq!(i16::to_f64(-7), -7.0);
        assert_eq!(i16::from_f64_clamped(-40000.0), i16::MIN);
        assert_eq!(i16::from_f64_clamped(-1.5), -1);
        assert_eq!(u32::to_f64(9), 9.0);
        assert_eq!(u32::from_f64_clamped(f64::from(u32::MAX) + 1.0), u32::MAX);
        assert_eq!(i32::to_f64(-9), -9.0);
        assert_eq!(i32::from_f64_clamped(f64::from(i32::MIN) - 1.0), i32::MIN);
        assert_eq!(i32::from_f64_clamped(-1.5), -1);
        assert_eq!(f32::to_f64(1.25), 1.25);
        assert_eq!(f32::from_f64_clamped(2.5), 2.5);
        assert_eq!(f64::to_f64(3.5), 3.5);
        assert_eq!(f64::from_f64_clamped(4.5), 4.5);
    }

    #[test]
    fn shrinkh_signed_integer_rounding_matches_libvips_bias() {
        let input = vec![-2i16, -1i16];
        let output = run_shrinkh::<I16>(2, false, &input, 2, 1, 1);
        assert_eq!(output, vec![-1]);
    }

    #[test]
    fn shrinkh_bridge_exposes_dyn_operation_contract() {
        let bridge = ShrinkHBridge::<U8>::new(2, 1).unwrap();
        assert_eq!(bridge.input_format(), BandFormatId::U8);
        assert_eq!(bridge.output_format(), BandFormatId::U8);
        assert_eq!(bridge.bands(), 1);
        assert_eq!(bridge.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(bridge.output_width(4), 2);
        assert_eq!(bridge.output_height(1), 1);
        assert_eq!(
            bridge.node_spec(2, 1),
            ShrinkH::<U8>::new(2).unwrap().node_spec(2, 1)
        );

        let source = MemorySource::<U8>::new(4, 1, 1, vec![0, 10, 20, 30]).unwrap();
        let out_region = Region::new(0, 0, 2, 1);
        let input_region = bridge.required_input_region(&out_region);
        let mut input_bytes = vec![0u8; input_region.pixel_count()];
        source.read_region(input_region, &mut input_bytes).unwrap();
        let mut output_bytes = vec![0u8; out_region.pixel_count()];
        let mut state = bridge.dyn_start();
        bridge.dyn_process_region(
            &mut *state,
            &input_bytes,
            &mut output_bytes,
            input_region,
            out_region,
        );
        assert_eq!(output_bytes, vec![5, 25]);
    }

    #[test]
    fn shrinkh_bridge_with_ceil_exposes_dimension_round_up() {
        let bridge = ShrinkHBridge::<U8>::new_with_ceil(3, true, 1).unwrap();
        assert_eq!(bridge.output_width(10), 4);
    }

    #[test]
    fn shrinkh_new_rejects_zero_factor() {
        let result = ShrinkH::<U8>::new(0);
        assert!(matches!(
            result,
            Err(BuildError::SourceHint {
                context: "shrink_h",
                message,
            }) if message == "factor must be >= 1"
        ));
    }

    #[test]
    fn shrinkh_bridge_new_rejects_zero_factor() {
        let result = ShrinkHBridge::<U8>::new(0, 1);
        assert!(matches!(
            result,
            Err(BuildError::SourceHint {
                context: "shrink_h",
                message,
            }) if message == "factor must be >= 1"
        ));
    }

    #[test]
    fn factor5_scalar_handles_odd_single_band_tail() {
        let input = vec![
            10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
        ];
        let mut output = vec![0u8; 3];
        shrink_h_u8_factor5_scalar(&input, 1, 15, 3, 1, &mut output);
        assert_eq!(output, vec![30, 80, 130]);
    }

    #[test]
    fn factor5_scalar_handles_rgb_and_rgba_tails() {
        let rgb_input = vec![
            10u8, 20, 30, 20, 30, 40, 30, 40, 50, 40, 50, 60, 50, 60, 70, 100, 110, 120, 110, 120,
            130, 120, 130, 140, 130, 140, 150, 140, 150, 160, 200, 210, 220, 210, 220, 230, 220,
            230, 240, 230, 240, 250, 240, 250, 255,
        ];
        let mut rgb_output = vec![0u8; 9];
        shrink_h_u8_factor5_scalar(&rgb_input, 3, 15, 3, 1, &mut rgb_output);
        assert_eq!(rgb_output, vec![30, 40, 50, 120, 130, 140, 220, 230, 239]);

        let rgba_input = vec![
            1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        ];
        let mut rgba_output = vec![0u8; 8];
        shrink_h_u8_factor5_scalar(&rgba_input, 4, 10, 2, 1, &mut rgba_output);
        assert_eq!(rgba_output, vec![9, 10, 11, 12, 29, 30, 31, 32]);
    }

    #[test]
    fn factor5_scalar_handles_odd_rgba_tail() {
        let input = vec![
            1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
            46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
        ];
        let mut output = vec![0u8; 12];
        shrink_h_u8_factor5_scalar(&input, 4, 15, 3, 1, &mut output);
        assert_eq!(output, vec![9, 10, 11, 12, 29, 30, 31, 32, 49, 50, 51, 52]);
    }

    #[test]
    fn shrinkh_dispatches_factor5_single_band_to_specialized_scalar_path() {
        let input = vec![
            10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
        ];
        let mut actual = vec![0u8; 3];
        let mut expected = vec![0u8; 3];

        shrink_h_u8(5, &input, 1, 15, 3, 1, &mut actual);
        shrink_h_u8_scalar(5, &input, 1, 15, 3, 1, &mut expected);

        assert_eq!(actual, expected);
    }

    #[test]
    fn sequential_rgb_helpers_match_scalar_reference() {
        let factor = 19usize;
        let out_w = 3usize;
        let out_h = 2usize;

        let rgb_input: Vec<u8> = (0..out_w * out_h * factor * 3)
            .map(|idx| ((idx * 17 + 5) % 251) as u8)
            .collect();
        let mut rgb_expected = vec![0u8; out_w * out_h * 3];
        let mut rgb_actual = vec![0u8; out_w * out_h * 3];
        shrink_h_u8_scalar(
            factor,
            &rgb_input,
            3,
            out_w * factor,
            out_w,
            out_h,
            &mut rgb_expected,
        );
        shrink_h_u8_sequential_3(
            factor,
            &rgb_input,
            out_w * factor,
            out_w,
            out_h,
            &mut rgb_actual,
        );
        assert_eq!(rgb_actual, rgb_expected);

        let rgba_input: Vec<u8> = (0..out_w * out_h * factor * 4)
            .map(|idx| ((idx * 29 + 3) % 251) as u8)
            .collect();
        let mut rgba_expected = vec![0u8; out_w * out_h * 4];
        let mut rgba_actual = vec![0u8; out_w * out_h * 4];
        shrink_h_u8_scalar(
            factor,
            &rgba_input,
            4,
            out_w * factor,
            out_w,
            out_h,
            &mut rgba_expected,
        );
        shrink_h_u8_sequential_4(
            factor,
            &rgba_input,
            out_w * factor,
            out_w,
            out_h,
            &mut rgba_actual,
        );
        assert_eq!(rgba_actual, rgba_expected);
    }

    #[test]
    fn generic_dispatch_covers_non_specialized_band_counts() {
        let input = vec![10u8, 100, 20, 110, 30, 120, 40, 130, 50, 140, 60, 150];
        let mut output = vec![0u8; 4];
        shrink_h_u8(3, &input, 2, 6, 2, 1, &mut output);
        assert_eq!(output, vec![20, 110, 50, 140]);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn shrinkh_neon_generic_handles_zero_width_without_touching_buffers() {
        let input = Vec::<u8>::new();
        let mut output = Vec::<u8>::new();

        // SAFETY: zero-width rows produce zero NEON/scalar iterations, so no pointer dereference occurs.
        unsafe {
            shrink_h_u8_neon(19, &input, 3, 0, 0, 1, &mut output);
        }

        assert!(output.is_empty());
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    #[should_panic(expected = "non-specialized band counts must use the scalar fallback")]
    fn shrinkh_factor2_neon_rejects_non_specialized_band_counts() {
        let input = vec![10u8, 20, 30, 40];
        let mut output = vec![0u8; 2];

        // SAFETY: the slices have the exact sizes requested by the helper; this test asserts
        // that its explicit non-specialized-band precondition remains enforced.
        unsafe {
            shrink_h_u8_factor2_neon(&input, 2, 2, 1, 1, &mut output);
        }
    }

    #[test]
    fn shrinkh_process_region_honors_borrowed_full_row_stride() {
        let op = ShrinkH::<U8>::new(2).unwrap().with_source_width(6);
        let input_region = Region::new(0, 0, 6, 1);
        let output_region = Region::new(0, 0, 2, 1);
        let input = Tile::<U8>::new(input_region, 1, &[10, 20, 30, 40, 250, 251]);
        let mut output_data = [0u8; 2];
        let mut output = TileMut::<U8>::new(output_region, 1, &mut output_data);

        op.process_region(&mut (), &input, &mut output);

        assert_eq!(output_data, [15, 35]);
    }

    #[test]
    fn with_ceil_builder_and_required_input_region_cover_non_zero_output_origin() {
        let op = ShrinkH::<U8>::new(2).unwrap().with_ceil(true);
        assert_eq!(op.output_width(5), 3);
        assert_eq!(
            op.required_input_region(&Region::new(3, 4, 2, 1)),
            Region::new(6, 4, 4, 1)
        );
    }

    proptest! {
        #[test]
        fn shrinkh_factor1_is_identity_prop(
            (width, height, bands, pixels) in (1u32..=32, 1u32..=16, 1u32..=4).prop_flat_map(|(width, height, bands)| {
                (
                    Just(width),
                    Just(height),
                    Just(bands),
                    prop::collection::vec(any::<u8>(), (width * height * bands) as usize),
                )
            }),
        ) {
            let output = run_shrinkh::<U8>(1, false, &pixels, width, height, bands);
            prop_assert_eq!(output, pixels);
        }

        #[test]
        fn shrinkh_uniform_factor2_preserves_value(
            value in any::<u8>(),
            width in 1u32..=32,
            height in 1u32..=16,
            bands in 1u32..=4,
        ) {
            let input = vec![value; (width * 2 * height * bands) as usize];
            let output = run_shrinkh::<U8>(2, false, &input, width * 2, height, bands);
            prop_assert!(output.iter().all(|sample| *sample == value));
        }

        #[test]
        fn shrinkh_ceil_output_width_at_least_floor(
            width in 1u32..=1024,
            factor in 1u32..=32,
        ) {
            let floor = ShrinkH::<U8>::new_with_ceil(factor, false)
                .unwrap()
                .output_width(width);
            let ceil = ShrinkH::<U8>::new_with_ceil(factor, true)
                .unwrap()
                .output_width(width);
            prop_assert!(ceil >= floor);
        }
    }
}

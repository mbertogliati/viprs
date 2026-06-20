#![allow(private_bounds)]
// REASON: recombination sample conversion stays crate-private while the public op remains typed.

use std::marker::PhantomData;

use crate::{
    error::{BuildError, ViprsError},
    format::{BandFormat, BandFormatId, F32, F64},
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    float32x4_t, uint8x8_t, uint8x8x3_t, vaddq_f32, vcombine_u16, vcvtq_f32_u32, vcvtq_u32_f32,
    vdupq_n_f32, vget_high_u16, vget_low_u16, vld3_u8, vmaxq_f32, vminq_f32, vmlaq_n_f32, vmovl_u8,
    vmovl_u16, vmulq_n_f32, vqmovn_u16, vqmovn_u32, vst3_u8,
};

/// Row-major `rows × cols` recombination matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    values: Vec<f64>,
}

impl Matrix {
    #[must_use]
    /// Creates a new `Matrix`.
    ///
    /// # Panics
    ///
    /// Panics if `values.len()` does not equal `rows * cols`.
    pub fn new(rows: usize, cols: usize, values: Vec<f64>) -> Self {
        assert_eq!(
            values.len(),
            rows * cols,
            "Matrix: values.len() must equal rows * cols"
        );
        Self { rows, cols, values }
    }

    #[must_use]
    /// Returns or performs identity.
    pub fn identity(size: usize) -> Self {
        let mut values = vec![0.0f64; size * size];
        for i in 0..size {
            values[i * size + i] = 1.0;
        }
        Self::new(size, size, values)
    }

    #[must_use]
    /// Returns or performs zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::new(rows, cols, vec![0.0f64; rows * cols])
    }

    #[must_use]
    /// Returns or performs rows.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    #[must_use]
    /// Returns or performs cols.
    pub const fn cols(&self) -> usize {
        self.cols
    }

    #[must_use]
    /// Returns or performs values.
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

trait RecombSample: Copy {
    fn from_f64(value: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl RecombSample for u8 {
    #[inline(always)]
    fn from_f64(value: f64) -> Self {
        let clamped = value.clamp(f64::from(Self::MIN), f64::from(Self::MAX));
        ((clamped + 0.5).floor() as u16).min(u16::from(Self::MAX)) as Self
    }

    #[inline(always)]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl RecombSample for f32 {
    #[inline(always)]
    fn from_f64(value: f64) -> Self {
        value as Self
    }

    #[inline(always)]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl RecombSample for f64 {
    #[inline(always)]
    fn from_f64(value: f64) -> Self {
        value
    }

    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }
}

/// Pixel-local band recombination.
///
/// Each input pixel is treated as an `N`-element vector and multiplied by an
/// `M × N` matrix, producing an `M`-band output pixel.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::arithmetic::recomb::RecombOp;
///
/// let op = RecombOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct RecombOp<F: BandFormat>
where
    F::Sample: RecombSample,
{
    matrix: Matrix,
    _format: PhantomData<F>,
}

impl<F: BandFormat> RecombOp<F>
where
    F::Sample: RecombSample,
{
    const fn validate_matrix_build(
        &self,
        input_bands: u32,
        output_bands: u32,
    ) -> Result<(), BuildError> {
        let rows = self.matrix.rows();
        let cols = self.matrix.cols();

        if rows == 0 {
            return Err(BuildError::InvalidRecombMatrix {
                rows,
                cols,
                input_bands,
                reason: "matrix must have at least 1 row",
            });
        }
        if cols != input_bands as usize {
            return Err(BuildError::InvalidRecombMatrix {
                rows,
                cols,
                input_bands,
                reason: "matrix columns must equal input bands",
            });
        }
        if rows != output_bands as usize {
            return Err(BuildError::InvalidRecombMatrix {
                rows,
                cols,
                input_bands,
                reason: "matrix rows must equal output bands",
            });
        }

        Ok(())
    }

    const fn validate_matrix_runtime(
        &self,
        input_bands: u32,
        output_bands: u32,
    ) -> Result<(), ViprsError> {
        let rows = self.matrix.rows();
        let cols = self.matrix.cols();

        if rows == 0 {
            return Err(ViprsError::InvalidRecombMatrix {
                rows,
                cols,
                input_bands: Some(input_bands),
                reason: "matrix must have at least 1 row",
            });
        }
        if cols != input_bands as usize {
            return Err(ViprsError::InvalidRecombMatrix {
                rows,
                cols,
                input_bands: Some(input_bands),
                reason: "matrix columns must equal input bands",
            });
        }
        if rows != output_bands as usize {
            return Err(ViprsError::InvalidRecombMatrix {
                rows,
                cols,
                input_bands: Some(input_bands),
                reason: "matrix rows must equal output bands",
            });
        }

        Ok(())
    }

    #[must_use]
    /// Creates a new `RecombOp`.
    pub const fn new(matrix: Matrix) -> Self {
        Self {
            matrix,
            _format: PhantomData,
        }
    }

    #[must_use]
    /// Returns or performs identity.
    pub fn identity(size: usize) -> Self {
        Self::new(Matrix::identity(size))
    }

    #[must_use]
    /// Returns or performs matrix.
    pub const fn matrix(&self) -> &Matrix {
        &self.matrix
    }
}

impl<F: BandFormat> Op for RecombOp<F>
where
    F::Sample: RecombSample,
{
    type Input = F;
    type Output = F;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    fn validate_build_contract(
        &self,
        input_bands: u32,
        output_bands: u32,
    ) -> Result<(), BuildError> {
        self.validate_matrix_build(input_bands, output_bands)
    }

    fn validate_region_contract(
        &self,
        input_region: Region,
        input_bands: u32,
        output_region: Region,
        output_bands: u32,
    ) -> Result<(), ViprsError> {
        let _ = (input_region, output_region);
        self.validate_matrix_runtime(input_bands, output_bands)
    }

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        debug_assert_eq!(input.bands as usize, self.matrix.cols());
        debug_assert_eq!(output.bands as usize, self.matrix.rows());

        if F::ID == BandFormatId::U8
            && input.bands == 3
            && output.bands == 3
            && self.matrix.rows() == 3
            && self.matrix.cols() == 3
        {
            let mut matrix = [0.0f32; 9];
            for (dst, src) in matrix.iter_mut().zip(self.matrix.values()) {
                *dst = *src as f32;
            }

            recomb_u8_3x3(
                bytemuck::cast_slice(input.data),
                bytemuck::cast_slice_mut(output.data),
                matrix,
            );
            return;
        }

        for (in_pixel, out_pixel) in input
            .data
            .chunks_exact(self.matrix.cols())
            .zip(output.data.chunks_exact_mut(self.matrix.rows()))
        {
            for (row_index, out_band) in out_pixel.iter_mut().enumerate() {
                let row_start = row_index * self.matrix.cols();
                let row = &self.matrix.values()[row_start..row_start + self.matrix.cols()];
                let mut acc = 0.0f64;
                for (sample, coeff) in in_pixel.iter().zip(row.iter()) {
                    acc += sample.to_f64() * coeff;
                }
                *out_band = F::Sample::from_f64(acc);
            }
        }
    }
}

impl<F: BandFormat> PixelLocalOp for RecombOp<F> where F::Sample: RecombSample {}

/// Type alias for recomb.
pub type Recomb = RecombOp<F32>;
/// Type alias for recomb64.
pub type Recomb64 = RecombOp<F64>;

#[inline]
fn recomb_u8_3x3(input: &[u8], output: &mut [u8], matrix: [f32; 9]) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: aarch64 guarantees NEON support; the helper handles only complete
        // 8-pixel chunks and defers any remainder to the scalar fallback.
        unsafe {
            recomb_u8_3x3_neon(input, output, matrix);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    recomb_u8_3x3_scalar(input, output, matrix);
}

#[inline]
fn recomb_u8_3x3_scalar(input: &[u8], output: &mut [u8], matrix: [f32; 9]) {
    for (in_pixel, out_pixel) in input.chunks_exact(3).zip(output.chunks_exact_mut(3)) {
        let red = f32::from(in_pixel[0]);
        let green = f32::from(in_pixel[1]);
        let blue = f32::from(in_pixel[2]);

        out_pixel[0] =
            clamp_f32_to_u8(blue.mul_add(matrix[2], green.mul_add(matrix[1], red * matrix[0])));
        out_pixel[1] =
            clamp_f32_to_u8(blue.mul_add(matrix[5], green.mul_add(matrix[4], red * matrix[3])));
        out_pixel[2] =
            clamp_f32_to_u8(blue.mul_add(matrix[8], green.mul_add(matrix[7], red * matrix[6])));
    }
}

#[inline(always)]
fn clamp_f32_to_u8(value: f32) -> u8 {
    let clamped = value.clamp(0.0, f32::from(u8::MAX));
    ((clamped + 0.5).floor() as u16).min(u16::from(u8::MAX)) as u8
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
// SAFETY: caller must execute this only on NEON-capable aarch64 and provide `input`/`output` slices sized for interleaved RGB pixels so each 24-byte chunk load/store stays in bounds.
unsafe fn recomb_u8_3x3_neon(input: &[u8], output: &mut [u8], matrix: [f32; 9]) {
    let simd_pixels = input.len() / 24;

    for chunk in 0..simd_pixels {
        let src = input.as_ptr().wrapping_add(chunk * 24);
        let dst = output.as_mut_ptr().wrapping_add(chunk * 24);

        // SAFETY: `src` points to 8 interleaved RGB pixels (24 bytes) for this chunk.
        let pixels = unsafe { vld3_u8(src) };

        let (red_low, red_high) = u8x8_to_f32x4_pair(pixels.0);
        let (green_low, green_high) = u8x8_to_f32x4_pair(pixels.1);
        let (blue_low, blue_high) = u8x8_to_f32x4_pair(pixels.2);

        let out = uint8x8x3_t(
            recomb_lane_pair_to_u8x8(
                red_low,
                red_high,
                green_low,
                green_high,
                blue_low,
                blue_high,
                [matrix[0], matrix[1], matrix[2]],
            ),
            recomb_lane_pair_to_u8x8(
                red_low,
                red_high,
                green_low,
                green_high,
                blue_low,
                blue_high,
                [matrix[3], matrix[4], matrix[5]],
            ),
            recomb_lane_pair_to_u8x8(
                red_low,
                red_high,
                green_low,
                green_high,
                blue_low,
                blue_high,
                [matrix[6], matrix[7], matrix[8]],
            ),
        );

        // SAFETY: `dst` points to 24 writable bytes for 8 interleaved RGB outputs.
        unsafe { vst3_u8(dst, out) };
    }

    let tail_pixels = input.len() / 3 - simd_pixels * 8;
    if tail_pixels > 0 {
        let src_start = simd_pixels * 24;
        let dst_start = simd_pixels * 24;
        recomb_u8_3x3_scalar(&input[src_start..], &mut output[dst_start..], matrix);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn u8x8_to_f32x4_pair(values: uint8x8_t) -> (float32x4_t, float32x4_t) {
    // SAFETY: this helper is only called from the aarch64 NEON fast path.
    unsafe {
        let widened = vmovl_u8(values);
        let low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(widened)));
        let high = vcvtq_f32_u32(vmovl_u16(vget_high_u16(widened)));
        (low, high)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn recomb_lane_pair_to_u8x8(
    red_low: float32x4_t,
    red_high: float32x4_t,
    green_low: float32x4_t,
    green_high: float32x4_t,
    blue_low: float32x4_t,
    blue_high: float32x4_t,
    coeffs: [f32; 3],
) -> uint8x8_t {
    let low = recomb_row_f32x4(red_low, green_low, blue_low, coeffs);
    let high = recomb_row_f32x4(red_high, green_high, blue_high, coeffs);
    pack_f32x4_pair_to_u8x8(low, high)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn recomb_row_f32x4(
    red: float32x4_t,
    green: float32x4_t,
    blue: float32x4_t,
    coeffs: [f32; 3],
) -> float32x4_t {
    // SAFETY: this helper is only called from the aarch64 NEON fast path.
    unsafe {
        let acc = vmlaq_n_f32(vmulq_n_f32(red, coeffs[0]), green, coeffs[1]);
        vmlaq_n_f32(acc, blue, coeffs[2])
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn pack_f32x4_pair_to_u8x8(low: float32x4_t, high: float32x4_t) -> uint8x8_t {
    // SAFETY: this helper is only called from the aarch64 NEON fast path.
    unsafe {
        let zero = vdupq_n_f32(0.0);
        let max = vdupq_n_f32(f32::from(u8::MAX));
        let round = vdupq_n_f32(0.5);

        let low = vcvtq_u32_f32(vaddq_f32(vminq_f32(vmaxq_f32(low, zero), max), round));
        let high = vcvtq_u32_f32(vaddq_f32(vminq_f32(vmaxq_f32(high, zero), max), round));

        let packed = vcombine_u16(vqmovn_u32(low), vqmovn_u32(high));
        vqmovn_u16(packed)
    }
}

// Integration tests for Recomb live in the root viprs crate.

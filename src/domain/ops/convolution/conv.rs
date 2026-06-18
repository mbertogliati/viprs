#![allow(clippy::unused_self)]
// REASON: compatibility helpers stay as methods alongside the rest of the convolution API.
#![allow(unused_imports)]
// REASON: BandFormatId is used only in aarch64-gated NEON dispatch blocks.

use std::marker::PhantomData;

use bytemuck::Pod;

use crate::{
    domain::op::{NodeSpec, Op},
    domain::{
        error::ViprsError,
        format::{BandFormat, BandFormatId, F32, F64, I16, I32, U8, U16, U32},
        image::{DemandHint, Region, Tile, TileMut},
    },
};

use super::common::{ConvolutionMask2d, FromF64, ToF64, validate_kernel_2d};
use super::gauss_blur::ToF32;

/// Precision mode for compatibility constructors on [`ConvOp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvPrecision {
    /// Uses the `Float` variant of `ConvPrecision`.
    Float,
    /// Uses the `Integer` variant of `ConvPrecision`.
    Integer,
    /// Uses the `Approximate` variant of `ConvPrecision`.
    Approximate,
}

/// libvips float precision marker.
pub struct FloatPrecision;
/// libvips integer precision marker: rounded mask, same output format as input.
pub struct IntegerPrecision;
/// libvips approximate precision marker: rounded mask, same output format as input.
pub struct ApproximatePrecision;

/// libvips-style `conv` facade.
///
/// `Op::Output` is an associated type, so libvips output-format parity is exposed
/// with type-level precision markers. `ConvOp<F>` keeps the existing float facade:
/// integer inputs output F32, F64 input outputs F64. `ConvOp<F, IntegerPrecision>`
/// and `ConvOp<F, ApproximatePrecision>` keep the input format.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::convolution::conv::ConvOp;
///
/// let op = ConvOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct ConvOp<F: BandFormat, P = FloatPrecision> {
    kernel: Box<[f64]>,
    kernel_f32: Box<[f32]>,
    kernel_w: u32,
    kernel_h: u32,
    radius_x: u32,
    radius_y: u32,
    offset: f64,
    offset_f32: f32,
    kernel3x3_f32: Option<Kernel3x3F32>,
    _format: PhantomData<F>,
    _precision: PhantomData<P>,
}

#[derive(Clone, Copy, Debug, Default)]
struct KernelTap3x3F32 {
    x: usize,
    y: usize,
    weight: f32,
}

#[derive(Clone, Debug)]
struct Kernel3x3F32 {
    taps: [KernelTap3x3F32; 9],
    nnz: usize,
    offset: f32,
    kind: Kernel3x3Kind,
}

#[derive(Clone, Debug)]
enum Kernel3x3Kind {
    Cross {
        north: f32,
        west: f32,
        center: f32,
        east: f32,
        south: f32,
    },
    SobelX,
    Full([f32; 9]),
    Sparse,
}

#[derive(Debug, Default)]
/// Represents a conv state.
pub struct ConvState {
    layout_3x3: Kernel3x3State,
}

#[derive(Debug, Default)]
struct Kernel3x3State {
    last_in_w: usize,
    last_bands: usize,
    valid: bool,
    sample_offsets: [usize; 9],
}

impl<F> ConvOp<F, FloatPrecision>
where
    F: BandFormat,
    F::Sample: ToF64 + Pod,
{
    /// Creates a new `ConvOp`.
    pub fn new(kernel: Vec<Vec<f64>>) -> Result<Self, ViprsError> {
        Self::with_mask(ConvolutionMask2d::from_coefficients(kernel)?)
    }

    /// Returns this value configured with mask.
    pub fn with_mask(mask: ConvolutionMask2d) -> Result<Self, ViprsError> {
        Self::from_mask(mask)
    }

    /// Returns this value configured with precision.
    pub fn with_precision(
        kernel: Vec<Vec<f64>>,
        precision: ConvPrecision,
    ) -> Result<Self, ViprsError> {
        Self::with_mask_precision(ConvolutionMask2d::from_coefficients(kernel)?, precision)
    }

    /// Returns this value configured with mask precision.
    pub fn with_mask_precision(
        mask: ConvolutionMask2d,
        precision: ConvPrecision,
    ) -> Result<Self, ViprsError> {
        let mask = match precision {
            ConvPrecision::Float => mask,
            ConvPrecision::Integer | ConvPrecision::Approximate => round_mask(mask)?,
        };
        Self::from_mask(mask)
    }

    /// Returns or performs approximate.
    pub fn approximate(kernel: Vec<Vec<f64>>) -> Result<Self, ViprsError> {
        Self::with_precision(kernel, ConvPrecision::Approximate)
    }
}

impl<F> ConvOp<F, IntegerPrecision>
where
    F: BandFormat,
    F::Sample: ToF64 + FromF64 + Pod,
{
    /// Creates a new `ConvOp`.
    pub fn new(kernel: Vec<Vec<f64>>) -> Result<Self, ViprsError> {
        Self::with_mask(ConvolutionMask2d::from_coefficients(kernel)?)
    }

    /// Returns this value configured with mask.
    pub fn with_mask(mask: ConvolutionMask2d) -> Result<Self, ViprsError> {
        Self::from_mask(round_mask(mask)?)
    }
}

impl<F> ConvOp<F, ApproximatePrecision>
where
    F: BandFormat,
    F::Sample: ToF64 + FromF64 + Pod,
{
    /// Creates a new `ConvOp`.
    pub fn new(kernel: Vec<Vec<f64>>) -> Result<Self, ViprsError> {
        Self::with_mask(ConvolutionMask2d::from_coefficients(kernel)?)
    }

    /// Returns this value configured with mask.
    pub fn with_mask(mask: ConvolutionMask2d) -> Result<Self, ViprsError> {
        Self::from_mask(round_mask(mask)?)
    }
}

impl<F, P> ConvOp<F, P>
where
    F: BandFormat,
{
    fn from_mask(mask: ConvolutionMask2d) -> Result<Self, ViprsError> {
        let (kernel_w, kernel_h) = validate_kernel_2d("ConvOp", mask.coefficients())?;
        let scale = mask.scale();
        let offset = mask.offset();
        let kernel = mask
            .into_coefficients()
            .into_iter()
            .flatten()
            .map(|coeff| coeff / scale)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let kernel_f32 = kernel
            .iter()
            .map(|&coeff| coeff as f32)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let kernel_w = kernel_w as u32;
        let kernel_h = kernel_h as u32;

        Ok(Self {
            kernel3x3_f32: build_kernel3x3_f32(&kernel, kernel_w, kernel_h, offset),
            kernel,
            kernel_f32,
            kernel_w,
            kernel_h,
            radius_x: kernel_w / 2,
            radius_y: kernel_h / 2,
            offset,
            offset_f32: offset as f32,
            _format: PhantomData,
            _precision: PhantomData,
        })
    }

    const fn required_region(&self, output: &Region) -> Region {
        Region::new(
            output.x - self.radius_x as i32,
            output.y - self.radius_y as i32,
            output.width + 2 * self.radius_x,
            output.height + 2 * self.radius_y,
        )
    }

    const fn spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec {
            input_tile_w: tile_w + 2 * self.radius_x,
            input_tile_h: tile_h + 2 * self.radius_y,
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    #[inline]
    fn process_as<O>(&self, input: &Tile<F>, output: &mut TileMut<O>)
    where
        O: BandFormat,
        F::Sample: ToF64 + Pod,
        O::Sample: FromF64 + Pod,
    {
        let out_w = output.region.width as usize;
        let out_h = output.region.height as usize;
        let in_w = input.region.width as usize;
        let bands = input.bands as usize;
        let kw = self.kernel_w as usize;
        let kh = self.kernel_h as usize;
        let row_stride = in_w * bands;
        let out_row_stride = out_w * bands;

        assert_eq!(input.data.len(), input.region.pixel_count() * bands);
        assert_eq!(output.data.len(), output.region.pixel_count() * bands);

        for (oy, output_row) in output
            .data
            .chunks_exact_mut(out_row_stride)
            .enumerate()
            .take(out_h)
        {
            let input_rows = &input.data[oy * row_stride..];
            for (ox, output_pixel) in output_row.chunks_exact_mut(bands).enumerate() {
                let column_offset = ox * bands;
                for (band, out_sample) in output_pixel.iter_mut().enumerate() {
                    let mut acc = self.offset;
                    for (kernel_row, source_row) in self
                        .kernel
                        .chunks_exact(kw)
                        .zip(input_rows.chunks_exact(row_stride).take(kh))
                    {
                        for (sample, weight) in source_row[column_offset + band..]
                            .iter()
                            .step_by(bands)
                            .take(kw)
                            .zip(kernel_row.iter())
                        {
                            acc = sample.to_f64().mul_add(*weight, acc);
                        }
                    }
                    *out_sample = O::Sample::from_f64(acc);
                }
            }
        }
    }

    #[inline]
    fn process_as_f32(&self, state: &mut ConvState, input: &Tile<F>, output: &mut TileMut<F32>)
    where
        F::Sample: ToF32 + Pod,
    {
        if let Some(kernel3x3) = &self.kernel3x3_f32 {
            self.process_as_f32_3x3(&mut state.layout_3x3, kernel3x3, input, output);
            return;
        }

        let out_w = output.region.width as usize;
        let out_h = output.region.height as usize;
        let in_w = input.region.width as usize;
        let bands = input.bands as usize;
        let kw = self.kernel_w as usize;
        let kh = self.kernel_h as usize;
        let row_stride = in_w * bands;
        let out_row_stride = out_w * bands;

        assert_eq!(input.data.len(), input.region.pixel_count() * bands);
        assert_eq!(output.data.len(), output.region.pixel_count() * bands);

        for (oy, output_row) in output
            .data
            .chunks_exact_mut(out_row_stride)
            .enumerate()
            .take(out_h)
        {
            let input_rows = &input.data[oy * row_stride..];
            for (ox, output_pixel) in output_row.chunks_exact_mut(bands).enumerate() {
                let column_offset = ox * bands;
                for (band, out_sample) in output_pixel.iter_mut().enumerate() {
                    let mut acc = self.offset_f32;
                    for (kernel_row, source_row) in self
                        .kernel_f32
                        .chunks_exact(kw)
                        .zip(input_rows.chunks_exact(row_stride).take(kh))
                    {
                        for (sample, weight) in source_row[column_offset + band..]
                            .iter()
                            .step_by(bands)
                            .take(kw)
                            .zip(kernel_row.iter())
                        {
                            acc = sample.to_f32().mul_add(*weight, acc);
                        }
                    }
                    *out_sample = acc;
                }
            }
        }
    }

    #[inline]
    fn process_as_f32_3x3(
        &self,
        state: &mut Kernel3x3State,
        kernel: &Kernel3x3F32,
        input: &Tile<F>,
        output: &mut TileMut<F32>,
    ) where
        F::Sample: ToF32 + Pod,
    {
        if process_as_f32_3x3_specialized(kernel, input, output) {
            return;
        }

        let out_w = output.region.width as usize;
        let out_h = output.region.height as usize;
        let in_w = input.region.width as usize;
        let bands = input.bands as usize;
        let row_stride = in_w * bands;
        let offsets = state.offsets_for(kernel, in_w, bands);
        let out_row_stride = out_w * bands;

        assert_eq!(input.data.len(), input.region.pixel_count() * bands);
        assert_eq!(output.data.len(), output.region.pixel_count() * bands);

        for (oy, output_row) in output
            .data
            .chunks_exact_mut(out_row_stride)
            .enumerate()
            .take(out_h)
        {
            let input_rows = &input.data[oy * row_stride..];
            for (ox, output_pixel) in output_row.chunks_exact_mut(bands).enumerate() {
                let pixel_base = ox * bands;
                match bands {
                    1 => {
                        let mut acc0 = kernel.offset;
                        for (offset, tap) in offsets.iter().zip(kernel.taps.iter()).take(kernel.nnz)
                        {
                            acc0 = input_rows[pixel_base + *offset]
                                .to_f32()
                                .mul_add(tap.weight, acc0);
                        }
                        output_pixel[0] = acc0;
                    }
                    3 => {
                        let mut acc0 = kernel.offset;
                        let mut acc1 = kernel.offset;
                        let mut acc2 = kernel.offset;
                        for (offset, tap) in offsets.iter().zip(kernel.taps.iter()).take(kernel.nnz)
                        {
                            let samples = &input_rows[pixel_base + *offset..];
                            acc0 = samples[0].to_f32().mul_add(tap.weight, acc0);
                            acc1 = samples[1].to_f32().mul_add(tap.weight, acc1);
                            acc2 = samples[2].to_f32().mul_add(tap.weight, acc2);
                        }
                        output_pixel[0] = acc0;
                        output_pixel[1] = acc1;
                        output_pixel[2] = acc2;
                    }
                    4 => {
                        let mut acc0 = kernel.offset;
                        let mut acc1 = kernel.offset;
                        let mut acc2 = kernel.offset;
                        let mut acc3 = kernel.offset;
                        for (offset, tap) in offsets.iter().zip(kernel.taps.iter()).take(kernel.nnz)
                        {
                            let samples = &input_rows[pixel_base + *offset..];
                            acc0 = samples[0].to_f32().mul_add(tap.weight, acc0);
                            acc1 = samples[1].to_f32().mul_add(tap.weight, acc1);
                            acc2 = samples[2].to_f32().mul_add(tap.weight, acc2);
                            acc3 = samples[3].to_f32().mul_add(tap.weight, acc3);
                        }
                        output_pixel[0] = acc0;
                        output_pixel[1] = acc1;
                        output_pixel[2] = acc2;
                        output_pixel[3] = acc3;
                    }
                    _ => {
                        for (band, out_sample) in output_pixel.iter_mut().enumerate() {
                            let mut acc = kernel.offset;
                            for (offset, tap) in
                                offsets.iter().zip(kernel.taps.iter()).take(kernel.nnz)
                            {
                                acc = input_rows[pixel_base + *offset + band]
                                    .to_f32()
                                    .mul_add(tap.weight, acc);
                            }
                            *out_sample = acc;
                        }
                    }
                }
            }
        }
    }
}

#[inline]
fn process_as_f32_3x3_specialized<F>(
    kernel: &Kernel3x3F32,
    input: &Tile<F>,
    output: &mut TileMut<F32>,
) -> bool
where
    F: BandFormat,
    F::Sample: ToF32 + Pod,
{
    match &kernel.kind {
        Kernel3x3Kind::Cross {
            north,
            west,
            center,
            east,
            south,
        } => {
            process_cross_3x3(
                input,
                output,
                kernel.offset,
                *north,
                *west,
                *center,
                *east,
                *south,
            );
            true
        }
        Kernel3x3Kind::SobelX => {
            process_sobel_x_3x3(input, output, kernel.offset);
            true
        }
        Kernel3x3Kind::Full(coefficients) => {
            process_full_3x3(input, output, kernel.offset, coefficients);
            true
        }
        Kernel3x3Kind::Sparse => false,
    }
}

#[inline]
fn process_cross_3x3<F>(
    input: &Tile<F>,
    output: &mut TileMut<F32>,
    offset: f32,
    north_weight: f32,
    west_weight: f32,
    center_weight: f32,
    east_weight: f32,
    south_weight: f32,
) where
    F: BandFormat,
    F::Sample: ToF32 + Pod,
{
    let out_w = output.region.width as usize;
    let out_h = output.region.height as usize;
    let in_w = input.region.width as usize;
    let bands = input.bands as usize;
    let row_stride = in_w * bands;
    let out_row_stride = out_w * bands;

    assert_eq!(input.data.len(), input.region.pixel_count() * bands);
    assert_eq!(output.data.len(), output.region.pixel_count() * bands);

    #[cfg(target_arch = "aarch64")]
    {
        use crate::domain::simd::SimdLevel;
        if F::ID == BandFormatId::U8
            && SimdLevel::detect().has_neon()
            && let (Some(north), Some(west), Some(center), Some(east), Some(south)) = (
                f32_to_i16_exact(north_weight),
                f32_to_i16_exact(west_weight),
                f32_to_i16_exact(center_weight),
                f32_to_i16_exact(east_weight),
                f32_to_i16_exact(south_weight),
            )
        {
            let input_u8: &[u8] = bytemuck::cast_slice(input.data);
            let coeffs = [north, west, center, east, south];
            match bands {
                1 => {
                    // SAFETY: runtime NEON detection passed, `input_u8` matches `input.data`, and the helper stays within halo/output bounds.
                    unsafe {
                        for oy in 0..out_h {
                            let row0 = &input_u8[oy * row_stride..(oy + 1) * row_stride];
                            let row1 = &input_u8[(oy + 1) * row_stride..(oy + 2) * row_stride];
                            let row2 = &input_u8[(oy + 2) * row_stride..(oy + 3) * row_stride];
                            let out_row =
                                &mut output.data[oy * out_row_stride..(oy + 1) * out_row_stride];
                            cross_u8_neon_row_1(row0, row1, row2, out_row, out_w, offset, coeffs);
                        }
                    }
                    return;
                }
                3 => {
                    // SAFETY: runtime NEON detection passed, `input_u8` matches `input.data`, and the helper stays within halo/output bounds.
                    unsafe {
                        for oy in 0..out_h {
                            let row0 = &input_u8[oy * row_stride..(oy + 1) * row_stride];
                            let row1 = &input_u8[(oy + 1) * row_stride..(oy + 2) * row_stride];
                            let row2 = &input_u8[(oy + 2) * row_stride..(oy + 3) * row_stride];
                            let out_row =
                                &mut output.data[oy * out_row_stride..(oy + 1) * out_row_stride];
                            cross_u8_neon_row_3(row0, row1, row2, out_row, out_w, offset, coeffs);
                        }
                    }
                    return;
                }
                _ => {}
            }
        }
    }

    for (((row0, row1), row2), out_row) in input
        .data
        .chunks_exact(row_stride)
        .zip(input.data[row_stride..].chunks_exact(row_stride))
        .zip(input.data[2 * row_stride..].chunks_exact(row_stride))
        .zip(output.data.chunks_exact_mut(out_row_stride))
        .take(out_h)
    {
        for (((((north_pixel, west_pixel), center_pixel), east_pixel), south_pixel), out_pixel) in
            row0[bands..]
                .chunks_exact(bands)
                .zip(row1.chunks_exact(bands))
                .zip(row1[bands..].chunks_exact(bands))
                .zip(row1[2 * bands..].chunks_exact(bands))
                .zip(row2[bands..].chunks_exact(bands))
                .zip(out_row.chunks_exact_mut(bands))
                .take(out_w)
        {
            for (((((north, west), center), east), south), out_sample) in north_pixel
                .iter()
                .zip(west_pixel.iter())
                .zip(center_pixel.iter())
                .zip(east_pixel.iter())
                .zip(south_pixel.iter())
                .zip(out_pixel.iter_mut())
            {
                *out_sample = south.to_f32().mul_add(
                    south_weight,
                    east.to_f32().mul_add(
                        east_weight,
                        center.to_f32().mul_add(
                            center_weight,
                            west.to_f32()
                                .mul_add(west_weight, north.to_f32().mul_add(north_weight, offset)),
                        ),
                    ),
                );
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn f32_to_i16_exact(value: f32) -> Option<i16> {
    if value.fract() != 0.0 {
        return None;
    }
    let rounded = value as i32;
    i16::try_from(rounded).ok()
}

#[inline]
fn process_sobel_x_3x3<F>(input: &Tile<F>, output: &mut TileMut<F32>, offset: f32)
where
    F: BandFormat,
    F::Sample: ToF32 + Pod,
{
    let out_w = output.region.width as usize;
    let out_h = output.region.height as usize;
    let in_w = input.region.width as usize;
    let bands = input.bands as usize;
    let row_stride = in_w * bands;
    let out_row_stride = out_w * bands;

    assert_eq!(input.data.len(), input.region.pixel_count() * bands);
    assert_eq!(output.data.len(), output.region.pixel_count() * bands);

    #[cfg(target_arch = "aarch64")]
    {
        use crate::domain::simd::SimdLevel;
        if F::ID == BandFormatId::U8 && SimdLevel::detect().has_neon() {
            let input_u8: &[u8] = bytemuck::cast_slice(input.data);
            match bands {
                1 => {
                    // SAFETY: runtime NEON detection passed, `input_u8` matches `input.data`, and the helper stays within halo/output bounds.
                    unsafe {
                        for oy in 0..out_h {
                            let row0 = &input_u8[oy * row_stride..(oy + 1) * row_stride];
                            let row1 = &input_u8[(oy + 1) * row_stride..(oy + 2) * row_stride];
                            let row2 = &input_u8[(oy + 2) * row_stride..(oy + 3) * row_stride];
                            let out_row =
                                &mut output.data[oy * out_row_stride..(oy + 1) * out_row_stride];
                            sobel_x_u8_neon_row_1(row0, row1, row2, out_row, out_w, offset);
                        }
                    }
                    return;
                }
                3 => {
                    // SAFETY: runtime NEON detection passed, `input_u8` matches `input.data`, and the helper stays within halo/output bounds.
                    unsafe {
                        for oy in 0..out_h {
                            let row0 = &input_u8[oy * row_stride..(oy + 1) * row_stride];
                            let row1 = &input_u8[(oy + 1) * row_stride..(oy + 2) * row_stride];
                            let row2 = &input_u8[(oy + 2) * row_stride..(oy + 3) * row_stride];
                            let out_row =
                                &mut output.data[oy * out_row_stride..(oy + 1) * out_row_stride];
                            sobel_x_u8_neon_row_3(row0, row1, row2, out_row, out_w, offset);
                        }
                    }
                    return;
                }
                _ => {}
            }
        }
    }

    for (((row0, row1), row2), out_row) in input
        .data
        .chunks_exact(row_stride)
        .zip(input.data[row_stride..].chunks_exact(row_stride))
        .zip(input.data[2 * row_stride..].chunks_exact(row_stride))
        .zip(output.data.chunks_exact_mut(out_row_stride))
        .take(out_h)
    {
        for (
            (((((top_left, top_right), mid_left), mid_right), bottom_left), bottom_right),
            out_pixel,
        ) in row0
            .chunks_exact(bands)
            .zip(row0[2 * bands..].chunks_exact(bands))
            .zip(row1.chunks_exact(bands))
            .zip(row1[2 * bands..].chunks_exact(bands))
            .zip(row2.chunks_exact(bands))
            .zip(row2[2 * bands..].chunks_exact(bands))
            .zip(out_row.chunks_exact_mut(bands))
            .take(out_w)
        {
            for ((((((top_l, top_r), mid_l), mid_r), bottom_l), bottom_r), out_sample) in top_left
                .iter()
                .zip(top_right.iter())
                .zip(mid_left.iter())
                .zip(mid_right.iter())
                .zip(bottom_left.iter())
                .zip(bottom_right.iter())
                .zip(out_pixel.iter_mut())
            {
                let top = top_r.to_f32() - top_l.to_f32();
                let middle = mid_r.to_f32() - mid_l.to_f32();
                let bottom = bottom_r.to_f32() - bottom_l.to_f32();
                *out_sample = middle.mul_add(2.0, offset + top + bottom);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn widen_diff_u8x8(
    left: std::arch::aarch64::uint8x8_t,
    right: std::arch::aarch64::uint8x8_t,
) -> std::arch::aarch64::int16x8_t {
    use std::arch::aarch64::{vmovl_u8, vreinterpretq_s16_u16, vsubq_s16};

    // SAFETY: widening and subtracting lane-wise values touches only register data.
    unsafe {
        let left = vreinterpretq_s16_u16(vmovl_u8(left));
        let right = vreinterpretq_s16_u16(vmovl_u8(right));
        vsubq_s16(right, left)
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn cross_u8_neon_row_1(
    row0: &[u8],
    row1: &[u8],
    row2: &[u8],
    output_row: &mut [f32],
    out_w: usize,
    offset: f32,
    coeffs: [i16; 5],
) {
    use std::arch::aarch64::{
        vdup_n_s16, vdupq_n_s32, vget_high_s16, vget_low_s16, vld1_u8, vmlal_s16, vmovl_u8,
        vreinterpretq_s16_u16, vst1q_s32,
    };

    let [north, west, center, east, south] = coeffs;
    let mut x = 0usize;
    while x + 8 <= out_w {
        // SAFETY: `x + 8 <= out_w` keeps every 8-lane load within the halo-extended rows, and the NEON arithmetic only touches those registers.
        let (acc_lo, acc_hi) = unsafe {
            let north = vdup_n_s16(north);
            let west = vdup_n_s16(west);
            let center = vdup_n_s16(center);
            let east = vdup_n_s16(east);
            let south = vdup_n_s16(south);
            let north_vec = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(row0.as_ptr().add(x + 1))));
            let west_vec = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(row1.as_ptr().add(x))));
            let center_vec = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(row1.as_ptr().add(x + 1))));
            let east_vec = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(row1.as_ptr().add(x + 2))));
            let south_vec = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(row2.as_ptr().add(x + 1))));

            let mut acc_lo = vdupq_n_s32(0);
            let mut acc_hi = vdupq_n_s32(0);
            acc_lo = vmlal_s16(acc_lo, vget_low_s16(north_vec), north);
            acc_hi = vmlal_s16(acc_hi, vget_high_s16(north_vec), north);
            acc_lo = vmlal_s16(acc_lo, vget_low_s16(west_vec), west);
            acc_hi = vmlal_s16(acc_hi, vget_high_s16(west_vec), west);
            acc_lo = vmlal_s16(acc_lo, vget_low_s16(center_vec), center);
            acc_hi = vmlal_s16(acc_hi, vget_high_s16(center_vec), center);
            acc_lo = vmlal_s16(acc_lo, vget_low_s16(east_vec), east);
            acc_hi = vmlal_s16(acc_hi, vget_high_s16(east_vec), east);
            acc_lo = vmlal_s16(acc_lo, vget_low_s16(south_vec), south);
            acc_hi = vmlal_s16(acc_hi, vget_high_s16(south_vec), south);
            (acc_lo, acc_hi)
        };

        let mut lo = [0i32; 4];
        let mut hi = [0i32; 4];
        // SAFETY: the arrays above are exactly four lanes wide and fully valid for stores.
        unsafe {
            vst1q_s32(lo.as_mut_ptr(), acc_lo);
            vst1q_s32(hi.as_mut_ptr(), acc_hi);
        }
        for lane in 0..4 {
            output_row[x + lane] = lo[lane] as f32 + offset;
            output_row[x + lane + 4] = hi[lane] as f32 + offset;
        }
        x += 8;
    }

    for ox in x..out_w {
        output_row[ox] = f32::from(coeffs[4]).mul_add(
            f32::from(row2[ox + 1]),
            f32::from(coeffs[3]).mul_add(
                f32::from(row1[ox + 2]),
                f32::from(coeffs[2]).mul_add(
                    f32::from(row1[ox + 1]),
                    f32::from(coeffs[1]).mul_add(
                        f32::from(row1[ox]),
                        f32::from(coeffs[0]).mul_add(f32::from(row0[ox + 1]), offset),
                    ),
                ),
            ),
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn cross_u8_neon_row_3(
    row0: &[u8],
    row1: &[u8],
    row2: &[u8],
    output_row: &mut [f32],
    out_w: usize,
    offset: f32,
    coeffs: [i16; 5],
) {
    use std::arch::aarch64::{
        vdup_n_s16, vdupq_n_s32, vget_high_s16, vget_low_s16, vld3_u8, vmlal_s16, vmovl_u8,
        vreinterpretq_s16_u16, vst1q_s32,
    };

    let [north, west, center, east, south] = coeffs;
    let mut x = 0usize;
    while x + 8 <= out_w {
        // SAFETY: `x + 8 <= out_w` keeps each `vld3_u8` inside the halo-extended rows, and the subsequent NEON ops stay in registers.
        let (acc_lo, acc_hi) = unsafe {
            let north = vdup_n_s16(north);
            let west = vdup_n_s16(west);
            let center = vdup_n_s16(center);
            let east = vdup_n_s16(east);
            let south = vdup_n_s16(south);
            let north_px = vld3_u8(row0.as_ptr().add((x + 1) * 3));
            let west_px = vld3_u8(row1.as_ptr().add(x * 3));
            let center_px = vld3_u8(row1.as_ptr().add((x + 1) * 3));
            let east_px = vld3_u8(row1.as_ptr().add((x + 2) * 3));
            let south_px = vld3_u8(row2.as_ptr().add((x + 1) * 3));
            let north_px = [north_px.0, north_px.1, north_px.2];
            let west_px = [west_px.0, west_px.1, west_px.2];
            let center_px = [center_px.0, center_px.1, center_px.2];
            let east_px = [east_px.0, east_px.1, east_px.2];
            let south_px = [south_px.0, south_px.1, south_px.2];

            let mut acc_lo = [vdupq_n_s32(0); 3];
            let mut acc_hi = [vdupq_n_s32(0); 3];
            for channel in 0..3 {
                let north_vec = vreinterpretq_s16_u16(vmovl_u8(north_px[channel]));
                let west_vec = vreinterpretq_s16_u16(vmovl_u8(west_px[channel]));
                let center_vec = vreinterpretq_s16_u16(vmovl_u8(center_px[channel]));
                let east_vec = vreinterpretq_s16_u16(vmovl_u8(east_px[channel]));
                let south_vec = vreinterpretq_s16_u16(vmovl_u8(south_px[channel]));
                acc_lo[channel] = vmlal_s16(acc_lo[channel], vget_low_s16(north_vec), north);
                acc_hi[channel] = vmlal_s16(acc_hi[channel], vget_high_s16(north_vec), north);
                acc_lo[channel] = vmlal_s16(acc_lo[channel], vget_low_s16(west_vec), west);
                acc_hi[channel] = vmlal_s16(acc_hi[channel], vget_high_s16(west_vec), west);
                acc_lo[channel] = vmlal_s16(acc_lo[channel], vget_low_s16(center_vec), center);
                acc_hi[channel] = vmlal_s16(acc_hi[channel], vget_high_s16(center_vec), center);
                acc_lo[channel] = vmlal_s16(acc_lo[channel], vget_low_s16(east_vec), east);
                acc_hi[channel] = vmlal_s16(acc_hi[channel], vget_high_s16(east_vec), east);
                acc_lo[channel] = vmlal_s16(acc_lo[channel], vget_low_s16(south_vec), south);
                acc_hi[channel] = vmlal_s16(acc_hi[channel], vget_high_s16(south_vec), south);
            }
            (acc_lo, acc_hi)
        };

        for channel in 0..3 {
            let mut lo = [0i32; 4];
            let mut hi = [0i32; 4];
            // SAFETY: the arrays above are exactly four lanes wide and fully valid for stores.
            unsafe {
                vst1q_s32(lo.as_mut_ptr(), acc_lo[channel]);
                vst1q_s32(hi.as_mut_ptr(), acc_hi[channel]);
            }
            for lane in 0..4 {
                output_row[(x + lane) * 3 + channel] = lo[lane] as f32 + offset;
                output_row[(x + lane + 4) * 3 + channel] = hi[lane] as f32 + offset;
            }
        }
        x += 8;
    }

    for ox in x..out_w {
        let out_pixel = &mut output_row[ox * 3..(ox + 1) * 3];
        for channel in 0..3 {
            out_pixel[channel] = f32::from(coeffs[4]).mul_add(
                f32::from(row2[(ox + 1) * 3 + channel]),
                f32::from(coeffs[3]).mul_add(
                    f32::from(row1[(ox + 2) * 3 + channel]),
                    f32::from(coeffs[2]).mul_add(
                        f32::from(row1[(ox + 1) * 3 + channel]),
                        f32::from(coeffs[1]).mul_add(
                            f32::from(row1[ox * 3 + channel]),
                            f32::from(coeffs[0])
                                .mul_add(f32::from(row0[(ox + 1) * 3 + channel]), offset),
                        ),
                    ),
                ),
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sobel_x_u8_neon_row_1(
    row0: &[u8],
    row1: &[u8],
    row2: &[u8],
    output_row: &mut [f32],
    out_w: usize,
    offset: f32,
) {
    use std::arch::aarch64::{
        vdup_n_s16, vdupq_n_s32, vget_high_s16, vget_low_s16, vld1_u8, vmlal_s16, vst1q_s32,
    };

    let mut x = 0usize;
    while x + 8 <= out_w {
        // SAFETY: `x + 8 <= out_w` keeps the Sobel taps inside the halo-extended rows, and the helper only performs register arithmetic.
        let (acc_lo, acc_hi) = unsafe {
            let one = vdup_n_s16(1);
            let two = vdup_n_s16(2);
            let top_diff = widen_diff_u8x8(
                vld1_u8(row0.as_ptr().add(x)),
                vld1_u8(row0.as_ptr().add(x + 2)),
            );
            let mid_diff = widen_diff_u8x8(
                vld1_u8(row1.as_ptr().add(x)),
                vld1_u8(row1.as_ptr().add(x + 2)),
            );
            let bottom_diff = widen_diff_u8x8(
                vld1_u8(row2.as_ptr().add(x)),
                vld1_u8(row2.as_ptr().add(x + 2)),
            );

            let mut acc_lo = vdupq_n_s32(0);
            let mut acc_hi = vdupq_n_s32(0);
            acc_lo = vmlal_s16(acc_lo, vget_low_s16(top_diff), one);
            acc_hi = vmlal_s16(acc_hi, vget_high_s16(top_diff), one);
            acc_lo = vmlal_s16(acc_lo, vget_low_s16(mid_diff), two);
            acc_hi = vmlal_s16(acc_hi, vget_high_s16(mid_diff), two);
            acc_lo = vmlal_s16(acc_lo, vget_low_s16(bottom_diff), one);
            acc_hi = vmlal_s16(acc_hi, vget_high_s16(bottom_diff), one);
            (acc_lo, acc_hi)
        };

        let mut lo = [0i32; 4];
        let mut hi = [0i32; 4];
        // SAFETY: the arrays above are exactly four lanes wide and fully valid for stores.
        unsafe {
            vst1q_s32(lo.as_mut_ptr(), acc_lo);
            vst1q_s32(hi.as_mut_ptr(), acc_hi);
        }
        for lane in 0..4 {
            output_row[x + lane] = lo[lane] as f32 + offset;
            output_row[x + lane + 4] = hi[lane] as f32 + offset;
        }
        x += 8;
    }

    for ox in x..out_w {
        output_row[ox] = 2.0f32.mul_add(
            f32::from(i16::from(row1[ox + 2]) - i16::from(row1[ox])),
            offset + f32::from(i16::from(row0[ox + 2]) - i16::from(row0[ox])),
        ) + f32::from(i16::from(row2[ox + 2]) - i16::from(row2[ox]));
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sobel_x_u8_neon_row_3(
    row0: &[u8],
    row1: &[u8],
    row2: &[u8],
    output_row: &mut [f32],
    out_w: usize,
    offset: f32,
) {
    use std::arch::aarch64::{
        vdup_n_s16, vdupq_n_s32, vget_high_s16, vget_low_s16, vld3_u8, vmlal_s16, vst1q_s32,
    };

    let mut x = 0usize;
    while x + 8 <= out_w {
        // SAFETY: `x + 8 <= out_w` keeps each `vld3_u8` Sobel tap pair inside the halo-extended interleaved rows, and the rest stays in registers.
        let (acc_lo, acc_hi) = unsafe {
            let one = vdup_n_s16(1);
            let two = vdup_n_s16(2);
            let top_left = vld3_u8(row0.as_ptr().add(x * 3));
            let top_right = vld3_u8(row0.as_ptr().add((x + 2) * 3));
            let mid_left = vld3_u8(row1.as_ptr().add(x * 3));
            let mid_right = vld3_u8(row1.as_ptr().add((x + 2) * 3));
            let bottom_left = vld3_u8(row2.as_ptr().add(x * 3));
            let bottom_right = vld3_u8(row2.as_ptr().add((x + 2) * 3));

            let left = [top_left.0, top_left.1, top_left.2];
            let right = [top_right.0, top_right.1, top_right.2];
            let mid_l = [mid_left.0, mid_left.1, mid_left.2];
            let mid_r = [mid_right.0, mid_right.1, mid_right.2];
            let bottom_l = [bottom_left.0, bottom_left.1, bottom_left.2];
            let bottom_r = [bottom_right.0, bottom_right.1, bottom_right.2];

            let mut acc_lo = [vdupq_n_s32(0); 3];
            let mut acc_hi = [vdupq_n_s32(0); 3];
            for channel in 0..3 {
                let top_diff = widen_diff_u8x8(left[channel], right[channel]);
                let mid_diff = widen_diff_u8x8(mid_l[channel], mid_r[channel]);
                let bottom_diff = widen_diff_u8x8(bottom_l[channel], bottom_r[channel]);
                acc_lo[channel] = vmlal_s16(acc_lo[channel], vget_low_s16(top_diff), one);
                acc_hi[channel] = vmlal_s16(acc_hi[channel], vget_high_s16(top_diff), one);
                acc_lo[channel] = vmlal_s16(acc_lo[channel], vget_low_s16(mid_diff), two);
                acc_hi[channel] = vmlal_s16(acc_hi[channel], vget_high_s16(mid_diff), two);
                acc_lo[channel] = vmlal_s16(acc_lo[channel], vget_low_s16(bottom_diff), one);
                acc_hi[channel] = vmlal_s16(acc_hi[channel], vget_high_s16(bottom_diff), one);
            }
            (acc_lo, acc_hi)
        };

        for channel in 0..3 {
            let mut lo = [0i32; 4];
            let mut hi = [0i32; 4];
            // SAFETY: the arrays above are exactly four lanes wide and fully valid for stores.
            unsafe {
                vst1q_s32(lo.as_mut_ptr(), acc_lo[channel]);
                vst1q_s32(hi.as_mut_ptr(), acc_hi[channel]);
            }
            for lane in 0..4 {
                output_row[(x + lane) * 3 + channel] = lo[lane] as f32 + offset;
                output_row[(x + lane + 4) * 3 + channel] = hi[lane] as f32 + offset;
            }
        }
        x += 8;
    }

    for ox in x..out_w {
        let out_pixel = &mut output_row[ox * 3..(ox + 1) * 3];
        for channel in 0..3 {
            out_pixel[channel] = 2.0f32.mul_add(
                f32::from(
                    i16::from(row1[(ox + 2) * 3 + channel]) - i16::from(row1[ox * 3 + channel]),
                ),
                offset
                    + f32::from(
                        i16::from(row0[(ox + 2) * 3 + channel]) - i16::from(row0[ox * 3 + channel]),
                    ),
            ) + f32::from(
                i16::from(row2[(ox + 2) * 3 + channel]) - i16::from(row2[ox * 3 + channel]),
            );
        }
    }
}

#[inline]
fn process_full_3x3<F>(
    input: &Tile<F>,
    output: &mut TileMut<F32>,
    offset: f32,
    coefficients: &[f32; 9],
) where
    F: BandFormat,
    F::Sample: ToF32 + Pod,
{
    let out_w = output.region.width as usize;
    let out_h = output.region.height as usize;
    let in_w = input.region.width as usize;
    let bands = input.bands as usize;
    let row_stride = in_w * bands;
    let out_row_stride = out_w * bands;
    let [k00, k01, k02, k10, k11, k12, k20, k21, k22] = *coefficients;

    assert_eq!(input.data.len(), input.region.pixel_count() * bands);
    assert_eq!(output.data.len(), output.region.pixel_count() * bands);

    for (((row0, row1), row2), out_row) in input
        .data
        .chunks_exact(row_stride)
        .zip(input.data[row_stride..].chunks_exact(row_stride))
        .zip(input.data[2 * row_stride..].chunks_exact(row_stride))
        .zip(output.data.chunks_exact_mut(out_row_stride))
        .take(out_h)
    {
        for (((((((((p00, p01), p02), p10), p11), p12), p20), p21), p22), out_pixel) in row0
            .chunks_exact(bands)
            .zip(row0[bands..].chunks_exact(bands))
            .zip(row0[2 * bands..].chunks_exact(bands))
            .zip(row1.chunks_exact(bands))
            .zip(row1[bands..].chunks_exact(bands))
            .zip(row1[2 * bands..].chunks_exact(bands))
            .zip(row2.chunks_exact(bands))
            .zip(row2[bands..].chunks_exact(bands))
            .zip(row2[2 * bands..].chunks_exact(bands))
            .zip(out_row.chunks_exact_mut(bands))
            .take(out_w)
        {
            for (((((((((s00, s01), s02), s10), s11), s12), s20), s21), s22), out_sample) in p00
                .iter()
                .zip(p01.iter())
                .zip(p02.iter())
                .zip(p10.iter())
                .zip(p11.iter())
                .zip(p12.iter())
                .zip(p20.iter())
                .zip(p21.iter())
                .zip(p22.iter())
                .zip(out_pixel.iter_mut())
            {
                *out_sample = s22.to_f32().mul_add(
                    k22,
                    s21.to_f32().mul_add(
                        k21,
                        s20.to_f32().mul_add(
                            k20,
                            s12.to_f32().mul_add(
                                k12,
                                s11.to_f32().mul_add(
                                    k11,
                                    s10.to_f32().mul_add(
                                        k10,
                                        s02.to_f32().mul_add(
                                            k02,
                                            s01.to_f32()
                                                .mul_add(k01, s00.to_f32().mul_add(k00, offset)),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                );
            }
        }
    }
}

impl Kernel3x3State {
    #[inline]
    fn offsets_for<'a>(
        &'a mut self,
        kernel: &Kernel3x3F32,
        in_w: usize,
        bands: usize,
    ) -> &'a [usize] {
        if !self.valid || self.last_in_w != in_w || self.last_bands != bands {
            for (slot, tap) in self
                .sample_offsets
                .iter_mut()
                .zip(kernel.taps.iter())
                .take(kernel.nnz)
            {
                *slot = (tap.y * in_w + tap.x) * bands;
            }
            self.last_in_w = in_w;
            self.last_bands = bands;
            self.valid = true;
        }

        &self.sample_offsets[..kernel.nnz]
    }
}

fn build_kernel3x3_f32(
    kernel: &[f64],
    kernel_w: u32,
    kernel_h: u32,
    offset: f64,
) -> Option<Kernel3x3F32> {
    if kernel_w != 3 || kernel_h != 3 {
        return None;
    }

    let mut taps = [KernelTap3x3F32::default(); 9];
    let mut nnz = 0usize;
    let mut full = [0.0f32; 9];
    for y in 0..3usize {
        for x in 0..3usize {
            let weight = kernel[y * 3 + x] as f32;
            full[y * 3 + x] = weight;
            if weight != 0.0 {
                taps[nnz] = KernelTap3x3F32 { x, y, weight };
                nnz += 1;
            }
        }
    }

    if nnz == 0 {
        taps[0] = KernelTap3x3F32 {
            x: 0,
            y: 0,
            weight: 0.0,
        };
        nnz = 1;
    }

    Some(Kernel3x3F32 {
        taps,
        nnz,
        offset: offset as f32,
        kind: classify_kernel3x3_f32(full),
    })
}

fn classify_kernel3x3_f32(kernel: [f32; 9]) -> Kernel3x3Kind {
    if kernel[0] == 0.0 && kernel[2] == 0.0 && kernel[6] == 0.0 && kernel[8] == 0.0 {
        return Kernel3x3Kind::Cross {
            north: kernel[1],
            west: kernel[3],
            center: kernel[4],
            east: kernel[5],
            south: kernel[7],
        };
    }

    if kernel == [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0] {
        return Kernel3x3Kind::SobelX;
    }

    if kernel.iter().all(|weight| *weight != 0.0) {
        return Kernel3x3Kind::Full(kernel);
    }

    Kernel3x3Kind::Sparse
}

fn round_mask(mask: ConvolutionMask2d) -> Result<ConvolutionMask2d, ViprsError> {
    let scale = mask.scale();
    let offset = mask.offset();
    let coefficients = mask
        .into_coefficients()
        .into_iter()
        .map(|row| row.into_iter().map(f64::round_ties_even).collect())
        .collect();
    ConvolutionMask2d::new(coefficients, scale, offset)
}

macro_rules! impl_conv_float_f32_output {
    ($($format:ty),+ $(,)?) => {
        $(
            impl Op for ConvOp<$format, FloatPrecision> {
                type Input = $format;
                type Output = F32;
                type State = ConvState;

                fn demand_hint(&self) -> DemandHint {
                    DemandHint::SmallTile
                }

                fn required_input_region(&self, output: &Region) -> Region {
                    self.required_region(output)
                }

                fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
                    self.spec(tile_w, tile_h)
                }

                fn start(&self) -> Self::State {
                    ConvState::default()
                }

                #[inline]
                fn process_region(
                    &self,
                    _state: &mut Self::State,
                    input: &Tile<Self::Input>,
                    output: &mut TileMut<F32>,
                ) {
                    self.process_as_f32(_state, input, output);
                }
            }
        )+
    };
}

macro_rules! impl_conv_same_output {
    ($precision:ty; $($format:ty),+ $(,)?) => {
        $(
            impl Op for ConvOp<$format, $precision> {
                type Input = $format;
                type Output = $format;
                type State = ();

                fn demand_hint(&self) -> DemandHint {
                    DemandHint::SmallTile
                }

                fn required_input_region(&self, output: &Region) -> Region {
                    self.required_region(output)
                }

                fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
                    self.spec(tile_w, tile_h)
                }

                fn start(&self) -> Self::State {}

                #[inline]
                fn process_region(
                    &self,
                    _state: &mut Self::State,
                    input: &Tile<Self::Input>,
                    output: &mut TileMut<Self::Output>,
                ) {
                    self.process_as(input, output);
                }
            }
        )+
    };
}

impl_conv_float_f32_output!(U8, U16, I16, U32, I32, F32);

impl Op for ConvOp<F64, FloatPrecision> {
    type Input = F64;
    type Output = F64;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::SmallTile
    }

    fn required_input_region(&self, output: &Region) -> Region {
        self.required_region(output)
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        self.spec(tile_w, tile_h)
    }

    fn start(&self) -> Self::State {}

    #[inline]
    fn process_region(
        &self,
        _state: &mut Self::State,
        input: &Tile<F64>,
        output: &mut TileMut<F64>,
    ) {
        self.process_as(input, output);
    }
}

impl_conv_same_output!(IntegerPrecision; U8, U16, I16, U32, I32, F32, F64);
impl_conv_same_output!(ApproximatePrecision; U8, U16, I16, U32, I32, F32, F64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        format::{F32, F64, U8, U16},
        image::{Region, Tile, TileMut},
    };
    use proptest::prelude::*;

    fn run_conv_f32(input_data: &[f32], region: Region, kernel: Vec<Vec<f64>>) -> Vec<f32> {
        let op = ConvOp::<F32>::new(kernel).unwrap();
        let mut output = vec![0.0f32; region.pixel_count()];
        let input = Tile::<F32>::new(region, 1, input_data);
        let mut output_tile = TileMut::<F32>::new(region, 1, &mut output);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output_tile);
        output
    }

    fn run_conv_mask(input_data: &[f32], region: Region, mask: ConvolutionMask2d) -> Vec<f32> {
        run_conv_mask_regions(input_data, region, region, mask)
    }

    fn run_conv_mask_regions(
        input_data: &[f32],
        input_region: Region,
        output_region: Region,
        mask: ConvolutionMask2d,
    ) -> Vec<f32> {
        let op = ConvOp::<F32>::with_mask(mask).unwrap();
        let mut output = vec![0.0f32; output_region.pixel_count()];
        let input = Tile::<F32>::new(input_region, 1, input_data);
        let mut output_tile = TileMut::<F32>::new(output_region, 1, &mut output);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output_tile);
        output
    }

    fn run_conv_u8_mask_regions(
        input_data: &[u8],
        input_region: Region,
        output_region: Region,
        bands: u32,
        mask: ConvolutionMask2d,
    ) -> Vec<f32> {
        let op = ConvOp::<U8>::with_mask(mask).unwrap();
        let mut output = vec![0.0f32; output_region.pixel_count() * bands as usize];
        let input = Tile::<U8>::new(input_region, bands, input_data);
        let mut output_tile = TileMut::<F32>::new(output_region, bands, &mut output);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output_tile);
        output
    }

    fn run_conv_f32_mask_regions_bands(
        input_data: &[f32],
        input_region: Region,
        output_region: Region,
        bands: u32,
        mask: ConvolutionMask2d,
    ) -> Vec<f32> {
        let op = ConvOp::<F32>::with_mask(mask).unwrap();
        let mut output = vec![0.0f32; output_region.pixel_count() * bands as usize];
        let input = Tile::<F32>::new(input_region, bands, input_data);
        let mut output_tile = TileMut::<F32>::new(output_region, bands, &mut output);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output_tile);
        output
    }

    fn reference_conv_from_u8(
        input_data: &[u8],
        input_region: Region,
        output_region: Region,
        bands: u32,
        mask: &ConvolutionMask2d,
    ) -> Vec<f32> {
        let in_w = input_region.width as usize;
        let bands = bands as usize;
        let coeffs = mask.coefficients();
        let kh = coeffs.len();
        let kw = coeffs[0].len();
        let scale = mask.scale() as f32;
        let offset = mask.offset() as f32;
        let mut output = vec![0.0f32; output_region.pixel_count() * bands];

        for oy in 0..output_region.height as usize {
            for ox in 0..output_region.width as usize {
                for band in 0..bands {
                    let mut acc = offset;
                    for (ky, row) in coeffs.iter().enumerate().take(kh) {
                        for (kx, &weight) in row.iter().enumerate().take(kw) {
                            let idx = ((oy + ky) * in_w + ox + kx) * bands + band;
                            acc += f32::from(input_data[idx]) * (weight as f32 / scale);
                        }
                    }
                    output[(oy * output_region.width as usize + ox) * bands + band] = acc;
                }
            }
        }

        output
    }

    fn reference_conv_from_f32(
        input_data: &[f32],
        input_region: Region,
        output_region: Region,
        bands: u32,
        mask: &ConvolutionMask2d,
    ) -> Vec<f32> {
        let in_w = input_region.width as usize;
        let bands = bands as usize;
        let coeffs = mask.coefficients();
        let kh = coeffs.len();
        let kw = coeffs[0].len();
        let scale = mask.scale() as f32;
        let offset = mask.offset() as f32;
        let mut output = vec![0.0f32; output_region.pixel_count() * bands];

        for oy in 0..output_region.height as usize {
            for ox in 0..output_region.width as usize {
                for band in 0..bands {
                    let mut acc = offset;
                    for (ky, row) in coeffs.iter().enumerate().take(kh) {
                        for (kx, &weight) in row.iter().enumerate().take(kw) {
                            let idx = ((oy + ky) * in_w + ox + kx) * bands + band;
                            acc += input_data[idx] * (weight as f32 / scale);
                        }
                    }
                    output[(oy * output_region.width as usize + ox) * bands + band] = acc;
                }
            }
        }

        output
    }

    proptest! {
        #[test]
        fn identity_kernel_preserves_samples(samples in prop::collection::vec(-100.0f32..100.0, 1..64)) {
            let region = Region::new(0, 0, samples.len() as u32, 1);
            let output = run_conv_f32(&samples, region, vec![vec![1.0]]);

            for (actual, expected) in output.iter().zip(samples.iter()) {
                prop_assert!((actual - expected).abs() < 1e-6);
            }
        }

        #[test]
        fn zero_kernel_outputs_zero(width in 1usize..6, height in 1usize..6, value in -50.0f32..50.0) {
            let region = Region::new(0, 0, width as u32, height as u32);
            let input = vec![value; width * height];
            let output = run_conv_f32(&input, region, vec![vec![0.0]]);

            prop_assert!(output.iter().all(|sample| sample.abs() < 1e-6));
        }
    }

    #[test]
    fn approximate_precision_uses_same_region_metadata_as_conva() {
        let kernel = vec![
            vec![1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
            vec![1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
            vec![1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
        ];
        let op = ConvOp::<F32>::approximate(kernel).unwrap();
        let output = Region::new(4, 5, 7, 8);
        assert_eq!(op.required_input_region(&output), Region::new(3, 4, 9, 10));
        let spec = op.node_spec(7, 8);
        assert_eq!(spec.input_tile_w, 9);
        assert_eq!(spec.input_tile_h, 10);
        assert_eq!(spec.output_tile_w, 7);
        assert_eq!(spec.output_tile_h, 8);
    }

    #[test]
    fn mask_scale_and_offset_follow_libvips_formula() {
        let region = Region::new(0, 0, 3, 1);
        let input = vec![10.0, 20.0, 30.0];
        let mask = ConvolutionMask2d::new(vec![vec![2.0]], 4.0, 5.0).unwrap();

        assert_eq!(run_conv_mask(&input, region, mask), vec![10.0, 15.0, 20.0]);
    }

    #[test]
    fn sparse_3x3_mask_uses_libvips_scale_offset_formula() {
        let input_region = Region::new(-1, -1, 3, 3);
        let output_region = Region::new(0, 0, 1, 1);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mask = ConvolutionMask2d::new(
            vec![
                vec![0.0, -1.0, 0.0],
                vec![-2.0, 8.0, -2.0],
                vec![0.0, -1.0, 0.0],
            ],
            2.0,
            3.0,
        )
        .unwrap();

        assert_eq!(
            run_conv_mask_regions(&input, input_region, output_region, mask),
            vec![8.0]
        );
    }

    #[test]
    fn float_precision_promotes_integer_input_to_f32() {
        fn output_is_f32<O: Op<Input = U8, Output = F32>>(_: &O) {}
        let op = ConvOp::<U8>::new(vec![vec![1.0]]).unwrap();
        output_is_f32(&op);
    }

    #[test]
    fn float_precision_preserves_f64_output() {
        let op = ConvOp::<F64>::new(vec![vec![1.0]]).unwrap();
        let region = Region::new(0, 0, 2, 1);
        let input_data = vec![1.25f64, 2.5];
        let mut output = vec![0.0f64; 2];
        let input = Tile::<F64>::new(region, 1, &input_data);
        let mut output_tile = TileMut::<F64>::new(region, 1, &mut output);
        let mut state = op.start();

        op.process_region(&mut state, &input, &mut output_tile);

        assert_eq!(output, input_data);
    }

    #[test]
    fn integer_precision_keeps_input_format_and_rounds_mask() {
        fn output_is_u16<O: Op<Input = U16, Output = U16>>(_: &O) {}
        let op = ConvOp::<U16, IntegerPrecision>::new(vec![vec![1.4]]).unwrap();
        output_is_u16(&op);
        let region = Region::new(0, 0, 1, 1);
        let input_data = vec![7u16];
        let mut output = vec![0u16; 1];
        let input = Tile::<U16>::new(region, 1, &input_data);
        let mut output_tile = TileMut::<U16>::new(region, 1, &mut output);
        let mut state = op.start();

        op.process_region(&mut state, &input, &mut output_tile);

        assert_eq!(output, vec![7]);
    }

    #[test]
    fn approximate_precision_keeps_input_format() {
        fn output_is_u8<O: Op<Input = U8, Output = U8>>(_: &O) {}
        let op = ConvOp::<U8, ApproximatePrecision>::new(vec![vec![1.0]]).unwrap();
        output_is_u8(&op);
    }

    #[test]
    fn kernel3x3_builder_classifies_variants_and_rounds_masks() {
        let cross =
            build_kernel3x3_f32(&[0.0, 1.0, 0.0, 2.0, 3.0, 4.0, 0.0, 5.0, 0.0], 3, 3, 7.0).unwrap();
        assert_eq!(cross.nnz, 5);
        assert_eq!(cross.offset, 7.0);
        match cross.kind {
            Kernel3x3Kind::Cross {
                north,
                west,
                center,
                east,
                south,
            } => assert_eq!(
                (north, west, center, east, south),
                (1.0, 2.0, 3.0, 4.0, 5.0)
            ),
            _ => panic!("expected cross kernel"),
        }

        assert!(matches!(
            build_kernel3x3_f32(&[-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0], 3, 3, 0.0)
                .unwrap()
                .kind,
            Kernel3x3Kind::SobelX
        ));
        assert!(matches!(
            build_kernel3x3_f32(&[1.0; 9], 3, 3, 0.0).unwrap().kind,
            Kernel3x3Kind::Full(_)
        ));
        assert!(matches!(
            build_kernel3x3_f32(&[1.0, 0.0, 2.0, 0.0, 3.0, 4.0, 5.0, 0.0, 6.0], 3, 3, 0.0)
                .unwrap()
                .kind,
            Kernel3x3Kind::Sparse
        ));

        let zero = build_kernel3x3_f32(&[0.0; 9], 3, 3, 0.0).unwrap();
        assert_eq!(zero.nnz, 1);
        assert!(build_kernel3x3_f32(&[1.0], 1, 1, 0.0).is_none());

        let rounded =
            round_mask(ConvolutionMask2d::new(vec![vec![1.5, -2.5, 0.5]], 3.0, 2.0).unwrap())
                .unwrap();
        assert_eq!(rounded.coefficients(), &[vec![2.0, -2.0, 0.0]]);
        assert_eq!(rounded.scale(), 3.0);
        assert_eq!(rounded.offset(), 2.0);
        #[cfg(target_arch = "aarch64")]
        {
            assert_eq!(f32_to_i16_exact(12.0), Some(12));
            assert_eq!(f32_to_i16_exact(12.25), None);
        }
    }

    #[test]
    fn u8_cross_kernel_matches_reference_for_large_rows() {
        let mask = ConvolutionMask2d::new(
            vec![
                vec![0.0, 1.0, 0.0],
                vec![2.0, 3.0, 4.0],
                vec![0.0, 5.0, 0.0],
            ],
            2.0,
            1.5,
        )
        .unwrap();
        let input_region = Region::new(0, 0, 22, 4);
        let output_region = Region::new(0, 0, 20, 2);
        let input_data = (0..(input_region.pixel_count() as usize))
            .map(|index| ((index * 17 + 9) % 251) as u8)
            .collect::<Vec<_>>();

        let actual =
            run_conv_u8_mask_regions(&input_data, input_region, output_region, 1, mask.clone());
        let expected = reference_conv_from_u8(&input_data, input_region, output_region, 1, &mask);

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-4);
        }
    }

    #[test]
    fn u8_sobel_x_kernel_matches_reference_for_rgb() {
        let mask = ConvolutionMask2d::from_coefficients(vec![
            vec![-1.0, 0.0, 1.0],
            vec![-2.0, 0.0, 2.0],
            vec![-1.0, 0.0, 1.0],
        ])
        .unwrap();
        let input_region = Region::new(0, 0, 22, 4);
        let output_region = Region::new(0, 0, 20, 2);
        let input_data = (0..(input_region.pixel_count() as usize * 3))
            .map(|index| ((index * 7 + 13) % 255) as u8)
            .collect::<Vec<_>>();

        let actual =
            run_conv_u8_mask_regions(&input_data, input_region, output_region, 3, mask.clone());
        let expected = reference_conv_from_u8(&input_data, input_region, output_region, 3, &mask);

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-3);
        }
    }

    #[test]
    fn u8_full_kernel_matches_reference_for_rgb() {
        let mask = ConvolutionMask2d::from_coefficients(vec![
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
        ])
        .unwrap();
        let input_region = Region::new(0, 0, 22, 4);
        let output_region = Region::new(0, 0, 20, 2);
        let input_data = (0..(input_region.pixel_count() as usize * 3))
            .map(|index| (index % 7) as u8)
            .collect::<Vec<_>>();

        let actual =
            run_conv_u8_mask_regions(&input_data, input_region, output_region, 3, mask.clone());
        let expected = reference_conv_from_u8(&input_data, input_region, output_region, 3, &mask);

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-4);
        }
    }

    #[test]
    fn sparse_kernel_matches_reference_for_band_counts_two_and_four() {
        let mask = ConvolutionMask2d::new(
            vec![
                vec![1.0, 0.0, 2.0],
                vec![0.0, 3.0, 4.0],
                vec![5.0, 0.0, 6.0],
            ],
            2.0,
            0.5,
        )
        .unwrap();
        let input_region = Region::new(0, 0, 6, 4);
        let output_region = Region::new(0, 0, 4, 2);

        for &bands in &[2u32, 4u32] {
            let input_data = (0..(input_region.pixel_count() as usize * bands as usize))
                .map(|index| (index as f32 * 0.5) - 3.0)
                .collect::<Vec<_>>();
            let actual = run_conv_f32_mask_regions_bands(
                &input_data,
                input_region,
                output_region,
                bands,
                mask.clone(),
            );
            let expected =
                reference_conv_from_f32(&input_data, input_region, output_region, bands, &mask);

            for (actual, expected) in actual.iter().zip(expected.iter()) {
                assert!((actual - expected).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn non_three_by_three_float_path_matches_reference_for_rgb() {
        let mask = ConvolutionMask2d::new(
            vec![
                vec![0.0, 1.0, 0.0, 1.0, 0.0],
                vec![1.0, 2.0, 3.0, 2.0, 1.0],
                vec![0.0, 3.0, 4.0, 3.0, 0.0],
                vec![1.0, 2.0, 3.0, 2.0, 1.0],
                vec![0.0, 1.0, 0.0, 1.0, 0.0],
            ],
            4.0,
            -1.0,
        )
        .unwrap();
        let input_region = Region::new(0, 0, 8, 6);
        let output_region = Region::new(0, 0, 4, 2);
        let input_data = (0..(input_region.pixel_count() as usize * 3))
            .map(|index| index as f32 / 10.0)
            .collect::<Vec<_>>();

        let actual = run_conv_f32_mask_regions_bands(
            &input_data,
            input_region,
            output_region,
            3,
            mask.clone(),
        );
        let expected = reference_conv_from_f32(&input_data, input_region, output_region, 3, &mask);

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn integer_precision_non_three_by_three_identity_returns_center_pixels() {
        let mask = vec![
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let op = ConvOp::<U16, IntegerPrecision>::new(mask).unwrap();
        let input_region = Region::new(0, 0, 8, 6);
        let output_region = Region::new(0, 0, 4, 2);
        let bands = 2u32;
        let input_data = (0..(input_region.pixel_count() as usize * bands as usize))
            .map(|index| (index as u16).wrapping_mul(3))
            .collect::<Vec<_>>();
        let mut output = vec![0u16; output_region.pixel_count() * bands as usize];
        let input = Tile::<U16>::new(input_region, bands, &input_data);
        let mut output_tile = TileMut::<U16>::new(output_region, bands, &mut output);
        let mut state = op.start();

        op.process_region(&mut state, &input, &mut output_tile);

        let mut expected = Vec::with_capacity(output.len());
        for oy in 0..output_region.height as usize {
            for ox in 0..output_region.width as usize {
                let src_base = ((oy + 2) * input_region.width as usize + (ox + 2)) * bands as usize;
                expected.extend_from_slice(&input_data[src_base..src_base + bands as usize]);
            }
        }
        assert_eq!(output, expected);
    }

    #[test]
    fn f32_specialized_cross_and_full_paths_match_reference() {
        let input_region = Region::new(0, 0, 7, 5);
        let output_region = Region::new(0, 0, 5, 3);
        let input_data = (0..(input_region.pixel_count() as usize * 3))
            .map(|index| (index % 9) as f32 - 4.0)
            .collect::<Vec<_>>();

        for mask in [
            ConvolutionMask2d::from_coefficients(vec![
                vec![0.0, 1.0, 0.0],
                vec![2.0, 3.0, 4.0],
                vec![0.0, 5.0, 0.0],
            ])
            .unwrap(),
            ConvolutionMask2d::from_coefficients(vec![
                vec![1.0, 1.0, 1.0],
                vec![1.0, 1.0, 1.0],
                vec![1.0, 1.0, 1.0],
            ])
            .unwrap(),
        ] {
            let actual = run_conv_f32_mask_regions_bands(
                &input_data,
                input_region,
                output_region,
                3,
                mask.clone(),
            );
            let expected =
                reference_conv_from_f32(&input_data, input_region, output_region, 3, &mask);
            for (actual, expected) in actual.iter().zip(expected.iter()) {
                assert!((actual - expected).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn five_band_cross_and_sobel_paths_use_generic_specialized_loops() {
        let input_region = Region::new(0, 0, 7, 5);
        let output_region = Region::new(0, 0, 5, 3);
        let bands = 5u32;
        let input_data = (0..(input_region.pixel_count() as usize * bands as usize))
            .map(|index| ((index * 11 + 3) % 251) as u8)
            .collect::<Vec<_>>();

        for mask in [
            ConvolutionMask2d::from_coefficients(vec![
                vec![0.0, 1.0, 0.0],
                vec![2.0, 3.0, 4.0],
                vec![0.0, 5.0, 0.0],
            ])
            .unwrap(),
            ConvolutionMask2d::from_coefficients(vec![
                vec![-1.0, 0.0, 1.0],
                vec![-2.0, 0.0, 2.0],
                vec![-1.0, 0.0, 1.0],
            ])
            .unwrap(),
        ] {
            let actual = run_conv_u8_mask_regions(
                &input_data,
                input_region,
                output_region,
                bands,
                mask.clone(),
            );
            let expected =
                reference_conv_from_u8(&input_data, input_region, output_region, bands, &mask);
            for (actual, expected) in actual.iter().zip(expected.iter()) {
                assert!((actual - expected).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn conv_state_reuses_cached_sparse_offsets_across_calls() {
        let mask = ConvolutionMask2d::new(
            vec![
                vec![1.0, 0.0, 2.0],
                vec![0.0, 3.0, 0.0],
                vec![4.0, 0.0, 5.0],
            ],
            1.0,
            0.0,
        )
        .unwrap();
        let op = ConvOp::<F32>::with_mask(mask).unwrap();
        let input_region = Region::new(0, 0, 7, 5);
        let output_region = Region::new(0, 0, 5, 3);
        let input_data = (0..(input_region.pixel_count() as usize * 5))
            .map(|index| index as f32 - 10.0)
            .collect::<Vec<_>>();
        let input = Tile::<F32>::new(input_region, 5, &input_data);
        let mut output_a = vec![0.0f32; output_region.pixel_count() * 5];
        let mut output_b = vec![0.0f32; output_region.pixel_count() * 5];
        let mut tile_a = TileMut::<F32>::new(output_region, 5, &mut output_a);
        let mut tile_b = TileMut::<F32>::new(output_region, 5, &mut output_b);
        let mut state = op.start();

        op.process_region(&mut state, &input, &mut tile_a);
        assert!(state.layout_3x3.valid);
        let cached_offsets = state.layout_3x3.sample_offsets;
        op.process_region(&mut state, &input, &mut tile_b);

        assert_eq!(output_a, output_b);
        assert_eq!(state.layout_3x3.sample_offsets, cached_offsets);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_row_helpers_match_scalar_reference() {
        let coeffs = [1i16, 2, 3, 4, 5];
        let row0 = (0..33).map(|index| (index % 11) as u8).collect::<Vec<_>>();
        let row1 = (0..33)
            .map(|index| ((index * 3) % 17) as u8)
            .collect::<Vec<_>>();
        let row2 = (0..33)
            .map(|index| ((index * 5) % 19) as u8)
            .collect::<Vec<_>>();

        let mut cross1 = vec![0.0f32; 9];
        let mut cross3 = vec![0.0f32; 27];
        let mut sobel1 = vec![0.0f32; 9];
        let mut sobel3 = vec![0.0f32; 27];

        // SAFETY: the test runs only on aarch64, rows include the required halo bytes, and the output buffers are sized for the requested widths.
        unsafe {
            cross_u8_neon_row_1(&row0, &row1, &row2, &mut cross1, 9, 0.5, coeffs);
            sobel_x_u8_neon_row_1(&row0, &row1, &row2, &mut sobel1, 9, -1.0);
            cross_u8_neon_row_3(&row0, &row1, &row2, &mut cross3, 9, 0.5, coeffs);
            sobel_x_u8_neon_row_3(&row0, &row1, &row2, &mut sobel3, 9, -1.0);
        }

        for ox in 0..9usize {
            let expected_cross = 0.5
                + f32::from(coeffs[0]) * f32::from(row0[ox + 1])
                + f32::from(coeffs[1]) * f32::from(row1[ox])
                + f32::from(coeffs[2]) * f32::from(row1[ox + 1])
                + f32::from(coeffs[3]) * f32::from(row1[ox + 2])
                + f32::from(coeffs[4]) * f32::from(row2[ox + 1]);
            let expected_sobel = -1.0
                + f32::from(row0[ox + 2] as i16 - row0[ox] as i16)
                + 2.0 * f32::from(row1[ox + 2] as i16 - row1[ox] as i16)
                + f32::from(row2[ox + 2] as i16 - row2[ox] as i16);
            assert!((cross1[ox] - expected_cross).abs() < 1e-4);
            assert!((sobel1[ox] - expected_sobel).abs() < 1e-4);
            for channel in 0..3usize {
                let expected_cross_rgb = 0.5
                    + f32::from(coeffs[0]) * f32::from(row0[(ox + 1) * 3 + channel])
                    + f32::from(coeffs[1]) * f32::from(row1[ox * 3 + channel])
                    + f32::from(coeffs[2]) * f32::from(row1[(ox + 1) * 3 + channel])
                    + f32::from(coeffs[3]) * f32::from(row1[(ox + 2) * 3 + channel])
                    + f32::from(coeffs[4]) * f32::from(row2[(ox + 1) * 3 + channel]);
                let expected_sobel_rgb =
                    -1.0 + f32::from(
                        row0[(ox + 2) * 3 + channel] as i16 - row0[ox * 3 + channel] as i16,
                    ) + 2.0
                        * f32::from(
                            row1[(ox + 2) * 3 + channel] as i16 - row1[ox * 3 + channel] as i16,
                        )
                        + f32::from(
                            row2[(ox + 2) * 3 + channel] as i16 - row2[ox * 3 + channel] as i16,
                        );
                assert!((cross3[ox * 3 + channel] - expected_cross_rgb).abs() < 1e-4);
                assert!((sobel3[ox * 3 + channel] - expected_sobel_rgb).abs() < 1e-4);
            }
        }
    }
}

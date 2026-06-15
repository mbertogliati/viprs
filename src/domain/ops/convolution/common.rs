use bytemuck::Pod;

use crate::domain::{
    error::{ConvolutionError, ViprsError},
    format::BandFormat,
    image::Tile,
};

pub use crate::domain::ops::resample::sample_conv::{FromF64, ToF64};

pub fn validate_kernel_1d(name: &'static str, kernel: &[f64]) -> Result<usize, ViprsError> {
    if kernel.is_empty() {
        return Err(ConvolutionError::EmptyKernel { op: name }.into());
    }
    if kernel.len().is_multiple_of(2) {
        return Err(ConvolutionError::EvenKernelLength {
            op: name,
            len: kernel.len(),
        }
        .into());
    }
    for (x, value) in kernel.iter().copied().enumerate() {
        validate_finite_coefficient(name, x, 0, value)?;
    }

    Ok(kernel.len() / 2)
}

/// Convolution matrix with libvips mask metadata.
///
/// libvips stores `scale` and `offset` on the matrix image; each output sample is:
/// `sum(pixel * coeff) / scale + offset`.
pub struct ConvolutionMask2d {
    coefficients: Vec<Vec<f64>>,
    scale: f64,
    offset: f64,
}

impl ConvolutionMask2d {
    /// Creates a new `ConvolutionMask2d`.
    pub fn new(coefficients: Vec<Vec<f64>>, scale: f64, offset: f64) -> Result<Self, ViprsError> {
        validate_scale_offset("ConvolutionMask2d", scale, offset)?;
        validate_kernel_2d("ConvolutionMask2d", &coefficients)?;
        Ok(Self {
            coefficients,
            scale,
            offset,
        })
    }

    /// Creates this value from coefficients.
    pub fn from_coefficients(coefficients: Vec<Vec<f64>>) -> Result<Self, ViprsError> {
        Self::new(coefficients, 1.0, 0.0)
    }

    #[must_use]
    /// Returns or performs coefficients.
    pub fn coefficients(&self) -> &[Vec<f64>] {
        &self.coefficients
    }

    #[must_use]
    /// Returns or performs scale.
    pub const fn scale(&self) -> f64 {
        self.scale
    }

    #[must_use]
    /// Returns or performs offset.
    pub const fn offset(&self) -> f64 {
        self.offset
    }

    #[must_use]
    /// Returns or performs into coefficients.
    pub fn into_coefficients(self) -> Vec<Vec<f64>> {
        self.coefficients
    }
}

impl Clone for ConvolutionMask2d {
    fn clone(&self) -> Self {
        Self {
            coefficients: self.coefficients.clone(),
            scale: self.scale,
            offset: self.offset,
        }
    }
}

/// Separable convolution vector with libvips mask metadata.
pub struct ConvolutionMask1d {
    coefficients: Vec<f64>,
    scale: f64,
    offset: f64,
}

impl ConvolutionMask1d {
    /// Creates a new `ConvolutionMask1d`.
    pub fn new(coefficients: Vec<f64>, scale: f64, offset: f64) -> Result<Self, ViprsError> {
        validate_scale_offset("ConvolutionMask1d", scale, offset)?;
        validate_kernel_1d("ConvolutionMask1d", &coefficients)?;
        Ok(Self {
            coefficients,
            scale,
            offset,
        })
    }

    /// Creates this value from coefficients.
    pub fn from_coefficients(coefficients: Vec<f64>) -> Result<Self, ViprsError> {
        Self::new(coefficients, 1.0, 0.0)
    }

    #[must_use]
    /// Returns or performs coefficients.
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    #[must_use]
    /// Returns or performs scale.
    pub const fn scale(&self) -> f64 {
        self.scale
    }

    #[must_use]
    /// Returns or performs offset.
    pub const fn offset(&self) -> f64 {
        self.offset
    }

    #[must_use]
    /// Returns or performs into coefficients.
    pub fn into_coefficients(self) -> Vec<f64> {
        self.coefficients
    }
}

impl Clone for ConvolutionMask1d {
    fn clone(&self) -> Self {
        Self {
            coefficients: self.coefficients.clone(),
            scale: self.scale,
            offset: self.offset,
        }
    }
}

#[inline(always)]
pub fn apply_scale_offset(acc: f64, scale: f64, offset: f64) -> f64 {
    acc / scale + offset
}

pub fn validate_kernel_2d(
    op: &'static str,
    kernel: &[Vec<f64>],
) -> Result<(usize, usize), ViprsError> {
    if kernel.is_empty() {
        return Err(ConvolutionError::EmptyKernel { op }.into());
    }

    let height = kernel.len();
    let width = kernel[0].len();
    if width == 0 {
        return Err(ConvolutionError::EmptyKernelRow { op }.into());
    }
    if height.is_multiple_of(2) || width.is_multiple_of(2) {
        return Err(ConvolutionError::EvenKernelDimensions { op, width, height }.into());
    }

    for (y, row) in kernel.iter().enumerate() {
        if row.len() != width {
            return Err(ConvolutionError::RaggedKernel {
                op,
                row: y,
                width: row.len(),
                expected: width,
            }
            .into());
        }
        for (x, value) in row.iter().copied().enumerate() {
            validate_finite_coefficient(op, x, y, value)?;
        }
    }

    Ok((width, height))
}

fn validate_scale_offset(op: &'static str, scale: f64, offset: f64) -> Result<(), ViprsError> {
    if !scale.is_finite() || scale == 0.0 {
        return Err(ConvolutionError::InvalidScale { op, scale }.into());
    }
    if !offset.is_finite() {
        return Err(ConvolutionError::InvalidOffset { op, offset }.into());
    }
    Ok(())
}

fn validate_finite_coefficient(
    op: &'static str,
    x: usize,
    y: usize,
    value: f64,
) -> Result<(), ViprsError> {
    if !value.is_finite() {
        return Err(ConvolutionError::NonFiniteCoefficient { op, x, y, value }.into());
    }
    Ok(())
}

#[inline(always)]
pub fn convolve_separable_at<F>(
    input: &Tile<F>,
    in_w: usize,
    bands: usize,
    x: usize,
    y: usize,
    band: usize,
    kernel: &[f64],
) -> f64
where
    F: BandFormat,
    F::Sample: ToF64 + Pod,
{
    let radius = kernel.len() / 2;
    let start_x = x - radius;
    let start_y = y - radius;
    let mut acc = 0.0f64;

    for (ky, &wy) in kernel.iter().enumerate() {
        let iy = start_y + ky;
        for (kx, &wx) in kernel.iter().enumerate() {
            let ix = start_x + kx;
            let idx = (iy * in_w + ix) * bands + band;
            acc = (input.data[idx].to_f64() * wx).mul_add(wy, acc);
        }
    }

    acc
}

#[inline(always)]
pub fn convolve_mask3_at<F>(
    input: &Tile<F>,
    in_w: usize,
    bands: usize,
    x: usize,
    y: usize,
    band: usize,
    mask: &[[f64; 3]; 3],
) -> f64
where
    F: BandFormat,
    F::Sample: ToF64 + Pod,
{
    let start_x = x - 1;
    let start_y = y - 1;
    let mut acc = 0.0f64;

    for (ky, row) in mask.iter().enumerate() {
        let iy = start_y + ky;
        for (kx, weight) in row.iter().enumerate() {
            let ix = start_x + kx;
            let idx = (iy * in_w + ix) * bands + band;
            acc = input.data[idx].to_f64().mul_add(*weight, acc);
        }
    }

    acc
}

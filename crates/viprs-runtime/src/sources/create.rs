//! Create image source adapter.
//!
//! This module exposes concrete source implementations or helpers that feed
//! pixels into compiled pipelines.

#![allow(private_bounds)]
// REASON: source sample traits are internal implementation details behind public source constructors.

use std::marker::PhantomData;

use bytemuck::Pod;

use crate::{
    domain::{
        error::ViprsError,
        format::BandFormat,
        image::{DemandHint, Region},
    },
    ports::source::{ImageSource, RandomAccessSource},
};

trait WhiteSample: Copy + Pod + 'static {
    fn white_value() -> Self;
}

impl WhiteSample for u8 {
    #[inline(always)]
    fn white_value() -> Self {
        Self::MAX
    }
}

impl WhiteSample for u16 {
    #[inline(always)]
    fn white_value() -> Self {
        Self::MAX
    }
}

impl WhiteSample for i16 {
    #[inline(always)]
    fn white_value() -> Self {
        Self::MAX
    }
}

impl WhiteSample for u32 {
    #[inline(always)]
    fn white_value() -> Self {
        Self::MAX
    }
}

impl WhiteSample for i32 {
    #[inline(always)]
    fn white_value() -> Self {
        Self::MAX
    }
}

impl WhiteSample for f32 {
    #[inline(always)]
    fn white_value() -> Self {
        1.0
    }
}

impl WhiteSample for f64 {
    #[inline(always)]
    fn white_value() -> Self {
        1.0
    }
}

trait FloatSourceSample: Copy + Pod + 'static {
    fn from_f64(v: f64) -> Self;
}

impl FloatSourceSample for f32 {
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v as Self
    }
}

impl FloatSourceSample for f64 {
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v
    }
}

trait IdentitySample: Copy + Pod + 'static {
    fn from_index(index: u32) -> Self;
}

impl IdentitySample for u8 {
    #[inline(always)]
    fn from_index(index: u32) -> Self {
        index.min(u32::from(Self::MAX)) as Self
    }
}

impl IdentitySample for u16 {
    #[inline(always)]
    fn from_index(index: u32) -> Self {
        index.min(u32::from(Self::MAX)) as Self
    }
}

impl IdentitySample for i16 {
    #[inline(always)]
    fn from_index(index: u32) -> Self {
        index.min(Self::MAX as u32) as Self
    }
}

impl IdentitySample for u32 {
    #[inline(always)]
    fn from_index(index: u32) -> Self {
        index
    }
}

impl IdentitySample for i32 {
    #[inline(always)]
    fn from_index(index: u32) -> Self {
        index.min(Self::MAX as u32) as Self
    }
}

#[inline(always)]
fn write_sample<S: Pod>(output: &mut [u8], sample_index: usize, value: S) {
    let sample_size = std::mem::size_of::<S>();
    let byte_start = sample_index * sample_size;
    output[byte_start..byte_start + sample_size].copy_from_slice(bytemuck::bytes_of(&value));
}

#[inline(always)]
fn clamped_x(x: i32, width: u32) -> u32 {
    if width == 0 {
        0
    } else {
        x.clamp(0, width as i32 - 1) as u32
    }
}

#[inline(always)]
fn clamped_y(y: i32, height: u32) -> u32 {
    if height == 0 {
        0
    } else {
        y.clamp(0, height as i32 - 1) as u32
    }
}

/// Synthetic source that fills every sample with zero.
pub struct BlackSource<F: BandFormat> {
    width: u32,
    height: u32,
    bands: u32,
    _format: PhantomData<F>,
}

impl<F: BandFormat> BlackSource<F> {
    /// `new` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::sources::create::new;
    /// ```
    #[must_use]
    pub const fn new(width: u32, height: u32, bands: u32) -> Self {
        Self {
            width,
            height,
            bands,
            _format: PhantomData,
        }
    }
}

impl<F: BandFormat> ImageSource for BlackSource<F> {
    type Format = F;

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn bands(&self) -> u32 {
        self.bands
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    #[inline]
    fn read_region(&self, _region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        output.fill(0);
        Ok(())
    }
}

impl<F: BandFormat> RandomAccessSource for BlackSource<F> {}

/// Synthetic source that fills every sample with the maximum representable white value.
pub struct WhiteSource<F: BandFormat>
where
    F::Sample: WhiteSample,
{
    width: u32,
    height: u32,
    bands: u32,
    pixel_bytes: Box<[u8]>,
    _format: PhantomData<F>,
}

#[allow(private_bounds)] // WhiteSample is module-local; it only constrains supported sample types.
impl<F: BandFormat> WhiteSource<F>
where
    F::Sample: WhiteSample,
{
    /// `new` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::sources::create::new;
    /// ```
    #[must_use]
    pub fn new(width: u32, height: u32, bands: u32) -> Self {
        let pixel = vec![F::Sample::white_value(); bands as usize];
        let pixel_bytes = bytemuck::cast_slice(&pixel).to_vec().into_boxed_slice();

        Self {
            width,
            height,
            bands,
            pixel_bytes,
            _format: PhantomData,
        }
    }
}

impl<F: BandFormat> ImageSource for WhiteSource<F>
where
    F::Sample: WhiteSample,
{
    type Format = F;

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn bands(&self) -> u32 {
        self.bands
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    #[inline]
    fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        let pixel_len = self.pixel_bytes.len();
        let pixel_count = region.pixel_count();

        for pixel_index in 0..pixel_count {
            let byte_start = pixel_index * pixel_len;
            output[byte_start..byte_start + pixel_len].copy_from_slice(&self.pixel_bytes);
        }

        Ok(())
    }
}

impl<F: BandFormat> RandomAccessSource for WhiteSource<F> where F::Sample: WhiteSample {}

/// Synthetic source that creates a left-to-right linear ramp in `[0, 1]`.
pub struct GreySource<F: BandFormat>
where
    F::Sample: FloatSourceSample,
{
    width: u32,
    height: u32,
    _format: PhantomData<F>,
}

#[allow(private_bounds)] // FloatSourceSample stays private to keep conversion details out of the public API.
impl<F: BandFormat> GreySource<F>
where
    F::Sample: FloatSourceSample,
{
    /// `new` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::sources::create::new;
    /// ```
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            _format: PhantomData,
        }
    }
}

impl<F: BandFormat> ImageSource for GreySource<F>
where
    F::Sample: FloatSourceSample,
{
    type Format = F;

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn bands(&self) -> u32 {
        1
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    #[inline]
    fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        let denom = self.width.saturating_sub(1);

        for row in 0..region.height as usize {
            for col in 0..region.width as usize {
                let x = clamped_x(region.x + col as i32, self.width);
                let value = if denom == 0 {
                    0.0
                } else {
                    f64::from(x) / f64::from(denom)
                };
                write_sample(
                    output,
                    row * region.width as usize + col,
                    F::Sample::from_f64(value),
                );
            }
        }

        Ok(())
    }
}

impl<F: BandFormat> RandomAccessSource for GreySource<F> where F::Sample: FloatSourceSample {}

/// Synthetic source that creates a one-row identity LUT.
pub struct IdentitySource<F: BandFormat>
where
    F::Sample: IdentitySample,
{
    size: u32,
    bands: u32,
    _format: PhantomData<F>,
}

#[allow(private_bounds)] // IdentitySample is private because only the builtin numeric sample types are supported.
impl<F: BandFormat> IdentitySource<F>
where
    F::Sample: IdentitySample,
{
    /// `new` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::sources::create::new;
    /// ```
    #[must_use]
    pub const fn new(size: u32, bands: u32) -> Self {
        Self {
            size,
            bands,
            _format: PhantomData,
        }
    }
}

impl<F: BandFormat> ImageSource for IdentitySource<F>
where
    F::Sample: IdentitySample,
{
    type Format = F;

    fn width(&self) -> u32 {
        self.size
    }

    fn height(&self) -> u32 {
        1
    }

    fn bands(&self) -> u32 {
        self.bands
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    #[inline]
    fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        let region_width = region.width as usize;
        let bands = self.bands as usize;

        for row in 0..region.height as usize {
            for col in 0..region_width {
                let x = clamped_x(region.x + col as i32, self.size);
                let value = F::Sample::from_index(x);
                let pixel_base = (row * region_width + col) * bands;

                for band in 0..bands {
                    write_sample(output, pixel_base + band, value);
                }
            }
        }

        Ok(())
    }
}

impl<F: BandFormat> RandomAccessSource for IdentitySource<F> where F::Sample: IdentitySample {}

/// Synthetic source that creates deterministic gaussian noise via the libvips 12-sample approximation.
pub struct GaussNoiseSource<F: BandFormat>
where
    F::Sample: FloatSourceSample,
{
    width: u32,
    height: u32,
    mean: f64,
    sigma: f64,
    seed: u32,
    _format: PhantomData<F>,
}

#[allow(private_bounds)] // FloatSourceSample is a local conversion helper.
impl<F: BandFormat> GaussNoiseSource<F>
where
    F::Sample: FloatSourceSample,
{
    /// `new` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::sources::create::new;
    /// ```
    #[must_use]
    pub fn new(width: u32, height: u32, mean: f64, sigma: f64) -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |duration| duration.subsec_nanos());

        Self::with_seed(width, height, mean, sigma, seed)
    }

    /// `with_seed` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::sources::create::with_seed;
    /// ```
    #[must_use]
    pub const fn with_seed(width: u32, height: u32, mean: f64, sigma: f64, seed: u32) -> Self {
        Self {
            width,
            height,
            mean,
            sigma,
            seed,
            _format: PhantomData,
        }
    }
}

#[inline(always)]
fn vips_random_add(mut hash: u32, value: i32) -> u32 {
    for shift in [0, 8, 16, 24] {
        hash = (hash ^ ((value >> shift) as u32 & 0xff)).wrapping_mul(16_777_619);
    }
    hash
}

#[inline(always)]
fn vips_random(seed: u32) -> u32 {
    vips_random_add(2_166_136_261, seed as i32)
}

impl<F: BandFormat> ImageSource for GaussNoiseSource<F>
where
    F::Sample: FloatSourceSample,
{
    type Format = F;

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn bands(&self) -> u32 {
        1
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    #[inline]
    fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        let region_width = region.width as usize;

        for row in 0..region.height as usize {
            for col in 0..region_width {
                let x = clamped_x(region.x + col as i32, self.width);
                let y = clamped_y(region.y + row as i32, self.height);
                let mut state = self.seed;
                state = vips_random_add(state, x as i32);
                state = vips_random_add(state, y as i32);
                let mut sum = 0.0;

                for _ in 0..12 {
                    let sample = vips_random(state);
                    state = sample;
                    sum += f64::from(sample) / f64::from(u32::MAX);
                }

                let value = (sum - 6.0).mul_add(self.sigma, self.mean);
                write_sample(output, row * region_width + col, F::Sample::from_f64(value));
            }
        }

        Ok(())
    }
}

impl<F: BandFormat> RandomAccessSource for GaussNoiseSource<F> where F::Sample: FloatSourceSample {}

/// Synthetic source that creates a cosine zone plate.
pub struct ZoneSource<F: BandFormat>
where
    F::Sample: FloatSourceSample,
{
    width: u32,
    height: u32,
    _format: PhantomData<F>,
}

#[allow(private_bounds)] // FloatSourceSample is private module plumbing.
impl<F: BandFormat> ZoneSource<F>
where
    F::Sample: FloatSourceSample,
{
    /// `new` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::sources::create::new;
    /// ```
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            _format: PhantomData,
        }
    }
}

impl<F: BandFormat> ImageSource for ZoneSource<F>
where
    F::Sample: FloatSourceSample,
{
    type Format = F;

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn bands(&self) -> u32 {
        1
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    #[inline]
    fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        let hwidth = f64::from(self.width / 2);
        let hheight = f64::from(self.height / 2);
        let c = std::f64::consts::PI / f64::from(self.width.max(1));
        let region_width = region.width as usize;

        for row in 0..region.height as usize {
            for col in 0..region_width {
                let x = f64::from(clamped_x(region.x + col as i32, self.width));
                let y = f64::from(clamped_y(region.y + row as i32, self.height));
                let dx = x - hwidth;
                let dy = y - hheight;
                let value = (c * (dx * dx + dy * dy)).cos();
                write_sample(output, row * region_width + col, F::Sample::from_f64(value));
            }
        }

        Ok(())
    }
}

impl<F: BandFormat> RandomAccessSource for ZoneSource<F> where F::Sample: FloatSourceSample {}

/// Synthetic source that mirrors libvips' eye response pattern.
pub struct EyeSource<F: BandFormat>
where
    F::Sample: FloatSourceSample,
{
    width: u32,
    height: u32,
    factor: f64,
    _format: PhantomData<F>,
}

#[allow(private_bounds)] // FloatSourceSample is an internal conversion helper.
impl<F: BandFormat> EyeSource<F>
where
    F::Sample: FloatSourceSample,
{
    /// `new` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::sources::create::new;
    /// ```
    pub fn new(width: u32, height: u32, factor: f64) -> Result<Self, ViprsError> {
        if !(0.0..=1.0).contains(&factor) {
            return Err(ViprsError::Codec(format!(
                "EyeSource: factor must be in [0, 1], got {factor}"
            )));
        }

        Ok(Self {
            width,
            height,
            factor,
            _format: PhantomData,
        })
    }
}

impl<F: BandFormat> ImageSource for EyeSource<F>
where
    F::Sample: FloatSourceSample,
{
    type Format = F;

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn bands(&self) -> u32 {
        1
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    #[inline]
    fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        let max_x = self.width.saturating_sub(1).max(1);
        let max_y = self.height.saturating_sub(1).max(1);
        let c = self.factor * std::f64::consts::PI / (2.0 * f64::from(max_x));
        let h = f64::from(max_y * max_y);
        let region_width = region.width as usize;

        for row in 0..region.height as usize {
            for col in 0..region_width {
                let x = f64::from(clamped_x(region.x + col as i32, self.width));
                let y = f64::from(clamped_y(region.y + row as i32, self.height));
                let value = (y * y * (c * x * x).cos()) / h;
                write_sample(output, row * region_width + col, F::Sample::from_f64(value));
            }
        }

        Ok(())
    }
}

impl<F: BandFormat> RandomAccessSource for EyeSource<F> where F::Sample: FloatSourceSample {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::format::{F32, U8, U16};

    #[test]
    fn black_source_fills_region_with_zeros() {
        let source = BlackSource::<U8>::new(4, 4, 2);
        let mut output = vec![0xffu8; 4 * 4 * 2];
        ImageSource::read_region(&source, Region::new(0, 0, 4, 4), &mut output).unwrap();
        assert!(output.iter().all(|&sample| sample == 0));
    }

    #[test]
    fn white_source_fills_region_with_max_samples() {
        let source = WhiteSource::<U16>::new(2, 2, 1);
        let mut output = vec![0u8; 2 * 2 * std::mem::size_of::<u16>()];
        ImageSource::read_region(&source, Region::new(0, 0, 2, 2), &mut output).unwrap();
        let samples: &[u16] = bytemuck::cast_slice(&output);
        assert!(samples.iter().all(|&sample| sample == u16::MAX));
    }

    #[test]
    fn grey_source_spans_zero_to_one() {
        let source = GreySource::<F32>::new(4, 1);
        let mut output = vec![0u8; 4 * std::mem::size_of::<f32>()];
        ImageSource::read_region(&source, Region::new(0, 0, 4, 1), &mut output).unwrap();
        let samples: &[f32] = bytemuck::cast_slice(&output);

        assert!((samples[0] - 0.0).abs() < 1e-6);
        assert!((samples[1] - (1.0 / 3.0)).abs() < 1e-6);
        assert!((samples[2] - (2.0 / 3.0)).abs() < 1e-6);
        assert!((samples[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn identity_source_emits_pixel_indices_for_each_band() {
        let source = IdentitySource::<U16>::new(4, 2);
        let mut output = vec![0u8; 4 * 2 * std::mem::size_of::<u16>()];
        ImageSource::read_region(&source, Region::new(0, 0, 4, 1), &mut output).unwrap();
        let samples: &[u16] = bytemuck::cast_slice(&output);
        assert_eq!(samples, &[0, 0, 1, 1, 2, 2, 3, 3]);
    }

    #[test]
    fn gauss_noise_has_non_zero_variance() {
        let source = GaussNoiseSource::<F32>::with_seed(64, 64, 0.0, 1.0, 1234);
        let mut output = vec![0u8; 64 * 64 * std::mem::size_of::<f32>()];
        ImageSource::read_region(&source, Region::new(0, 0, 64, 64), &mut output).unwrap();
        let samples: &[f32] = bytemuck::cast_slice(&output);
        let mean = samples.iter().copied().sum::<f32>() / samples.len() as f32;
        let variance = samples
            .iter()
            .map(|sample| {
                let delta = *sample - mean;
                delta * delta
            })
            .sum::<f32>()
            / samples.len() as f32;

        assert!(
            variance > 0.01,
            "expected visible gaussian variance, got {variance}"
        );
    }

    #[test]
    fn zone_source_has_peak_at_centre() {
        let source = ZoneSource::<F32>::new(5, 5);
        let mut output = vec![0u8; std::mem::size_of::<f32>()];
        ImageSource::read_region(&source, Region::new(2, 2, 1, 1), &mut output).unwrap();
        let sample = bytemuck::cast_slice::<u8, f32>(&output)[0];
        assert!((sample - 1.0).abs() < 1e-6);
    }

    #[test]
    fn eye_source_factor_zero_is_vertical_ramp() {
        let source = EyeSource::<F32>::new(4, 5, 0.0).unwrap();
        let mut output = vec![0u8; 5 * std::mem::size_of::<f32>()];
        ImageSource::read_region(&source, Region::new(0, 0, 1, 5), &mut output).unwrap();
        let samples: &[f32] = bytemuck::cast_slice(&output);

        assert!((samples[0] - 0.0).abs() < 1e-6);
        assert!((samples[4] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn eye_source_rejects_out_of_range_factor() {
        assert!(matches!(
            EyeSource::<F32>::new(4, 4, 1.5),
            Err(ViprsError::Codec(_))
        ));
    }
}

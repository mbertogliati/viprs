#![allow(clippy::missing_fields_in_debug)]
// REASON: the custom debug view intentionally omits large generated tables to keep logs readable.

use std::{fmt, marker::PhantomData};

use crate::domain::{
    error::ViprsError,
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
    ops::resample::sample_conv::FromF64,
};

const FRACTSURF_SEED: u64 = 0xd1b5_4a32_917c_8ef1;
const OCTAVE_WEIGHT_OFFSET: f32 = 3.25;

/// Generate a rectangular fractal surface with dyadic Gaussian octaves.
///
/// libvips builds `FractSurf` with `gaussnoise + mask_fractal + freqmult`, which
/// shapes white noise by a radial `f^(D-4)` coefficient mask in Fourier space
/// (`fractsurf.c` + `mask_fractal.c`). The default `viprs` build keeps `fft`
/// optional, so it precomputes a deterministic spatial-domain equivalent:
/// bilinearly filtered Gaussian octaves whose per-octave RMS is weighted to
/// match the same radial power-law after octave band integration. The `0.25`
/// correction in `OCTAVE_WEIGHT_OFFSET` compensates for the tent filter
/// introduced by bilinear lattice reconstruction; the spectral-slope test
/// verifies the measured Hurst exponent stays aligned with the libvips mask.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::create::fractsurf::FractSurfOp;
///
/// let op = FractSurfOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct FractSurfOp<F: BandFormat> {
    width: u32,
    height: u32,
    fractal_dimension: f64,
    surface: Box<[f32]>,
    _format: PhantomData<F>,
}

impl<F: BandFormat> fmt::Debug for FractSurfOp<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FractSurfOp")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("fractal_dimension", &self.fractal_dimension)
            .finish()
    }
}

impl<F: BandFormat> FractSurfOp<F> {
    /// Creates a new `FractSurfOp`.
    pub fn new(width: u32, height: u32, fractal_dimension: f64) -> Result<Self, ViprsError> {
        if width == 0 || height == 0 {
            return Err(ViprsError::Scheduler(format!(
                "FractSurfOp width and height must be > 0, got {width}x{height}"
            )));
        }
        if !fractal_dimension.is_finite() || !(2.0..=3.0).contains(&fractal_dimension) {
            return Err(ViprsError::Scheduler(format!(
                "FractSurfOp fractal_dimension must be finite and in [2, 3], got {fractal_dimension}"
            )));
        }

        let pixel_count = checked_surface_pixel_count(width, height)?;

        Ok(Self {
            width,
            height,
            fractal_dimension,
            surface: generate_surface(width, height, pixel_count, fractal_dimension)?,
            _format: PhantomData,
        })
    }

    #[must_use]
    /// Returns or performs width.
    pub const fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    /// Returns or performs height.
    pub const fn height(&self) -> u32 {
        self.height
    }
}

const fn fractsurf_image_too_large(
    width: u32,
    height: u32,
    bytes: u128,
    details: &'static str,
) -> ViprsError {
    ViprsError::ImageTooLarge {
        width,
        height,
        bands: 1,
        bytes,
        limit_bytes: usize::MAX as u128,
        details,
    }
}

fn checked_surface_pixel_count(width: u32, height: u32) -> Result<usize, ViprsError> {
    let bytes_per_sample = std::mem::size_of::<f32>() as u128;
    let total_bytes = u128::from(width) * u128::from(height) * bytes_per_sample;
    let pixel_count = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| {
            fractsurf_image_too_large(
                width,
                height,
                total_bytes,
                "FractSurf surface pixel count exceeds addressable memory",
            )
        })?;

    if total_bytes > usize::MAX as u128 {
        return Err(fractsurf_image_too_large(
            width,
            height,
            total_bytes,
            "FractSurf surface buffer exceeds addressable memory",
        ));
    }

    Ok(pixel_count)
}

#[inline(always)]
const fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[inline(always)]
fn next_unit_f64(state: &mut u64) -> f64 {
    let bits = splitmix64(*state) >> 11;
    *state = (*state).wrapping_add(0x94d0_49bb_1331_11eb);
    ((bits as f64) + 1.0) / (((1_u64 << 53) as f64) + 1.0)
}

#[inline(always)]
fn gaussian_sample(seed: u64, x: usize, y: usize) -> f32 {
    let mut state = seed ^ ((x as u64) << 32) ^ y as u64;
    let u1 = next_unit_f64(&mut state);
    let u2 = next_unit_f64(&mut state);
    let magnitude = (-2.0 * u1.ln()).sqrt();
    let z0 = magnitude * (std::f64::consts::TAU * u2).cos();
    z0 as f32
}

#[inline(always)]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    (b - a).mul_add(t, a)
}

fn build_gaussian_lattice(width: u32, height: u32, octave: u64) -> Result<Box<[f32]>, ViprsError> {
    let pixel_count = checked_surface_pixel_count(width, height)?;
    let width = width as usize;
    let height = height as usize;
    let mut lattice = vec![0.0f32; pixel_count];
    let octave_seed = FRACTSURF_SEED ^ octave.wrapping_mul(0x517c_c1b7_2722_0a95);

    for y in 0..height {
        for x in 0..width {
            lattice[y * width + x] = gaussian_sample(octave_seed, x, y);
        }
    }

    Ok(lattice.into_boxed_slice())
}

fn accumulate_octave(
    surface: &mut [f32],
    width: u32,
    height: u32,
    lattice_width: u32,
    lattice_height: u32,
    amplitude: f32,
    octave: u64,
) -> Result<(), ViprsError> {
    let lattice = build_gaussian_lattice(lattice_width, lattice_height, octave)?;
    let width = width as usize;
    let height = height as usize;
    let lattice_width = lattice_width as usize;
    let lattice_height = lattice_height as usize;
    let x_scale = lattice_width as f32 / width.max(1) as f32;
    let y_scale = lattice_height as f32 / height.max(1) as f32;

    for y in 0..height {
        let sample_y = y as f32 * y_scale;
        let y0 = sample_y.floor() as usize % lattice_height;
        let y1 = (y0 + 1) % lattice_height;
        let ty = sample_y - y0 as f32;

        for x in 0..width {
            let sample_x = x as f32 * x_scale;
            let x0 = sample_x.floor() as usize % lattice_width;
            let x1 = (x0 + 1) % lattice_width;
            let tx = sample_x - x0 as f32;

            let v00 = lattice[y0 * lattice_width + x0];
            let v10 = lattice[y0 * lattice_width + x1];
            let v01 = lattice[y1 * lattice_width + x0];
            let v11 = lattice[y1 * lattice_width + x1];
            let top = lerp(v00, v10, tx);
            let bottom = lerp(v01, v11, tx);

            surface[y * width + x] += lerp(top, bottom, ty) * amplitude;
        }
    }

    Ok(())
}

fn normalize_surface(surface: &mut [f32]) {
    let mut min_value = f32::INFINITY;
    let mut max_value = f32::NEG_INFINITY;
    for &value in surface.iter() {
        min_value = min_value.min(value);
        max_value = max_value.max(value);
    }

    let range = max_value - min_value;
    if range <= f32::EPSILON {
        surface.fill(0.5);
        return;
    }

    for value in surface.iter_mut() {
        *value = (*value - min_value) / range;
    }
}

fn generate_surface(
    width: u32,
    height: u32,
    pixel_count: usize,
    fractal_dimension: f64,
) -> Result<Box<[f32]>, ViprsError> {
    let mut surface = vec![0.0f32; pixel_count];
    let max_dimension = width.max(height).max(1);
    let octave_weight_exponent = fractal_dimension as f32 - OCTAVE_WEIGHT_OFFSET;

    let mut frequency = 1u32;
    let mut octave = 0u64;
    loop {
        let lattice_width = frequency.min(width).max(1);
        let lattice_height = frequency.min(height).max(1);
        let amplitude = (frequency as f32).powf(octave_weight_exponent);
        accumulate_octave(
            &mut surface,
            width,
            height,
            lattice_width,
            lattice_height,
            amplitude,
            octave,
        )?;

        if frequency >= max_dimension {
            break;
        }
        frequency = frequency.saturating_mul(2);
        octave = octave.wrapping_add(1);
    }

    normalize_surface(&mut surface);
    Ok(surface.into_boxed_slice())
}

impl<F> Op for FractSurfOp<F>
where
    F: BandFormat,
    F::Sample: FromF64,
{
    type Input = F;
    type Output = F;
    type State = ();

    const OUTPUT_BANDS: Option<usize> = Some(1);

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), _input: &Tile<F>, output: &mut TileMut<F>) {
        debug_assert_eq!(output.bands, 1, "FractSurfOp output must be single-band");
        debug_assert!(output.region.x >= 0 && output.region.y >= 0);
        debug_assert!(output.region.x as u32 + output.region.width <= self.width);
        debug_assert!(output.region.y as u32 + output.region.height <= self.height);

        let region_width = output.region.width as usize;
        let image_width = self.width as usize;

        for row in 0..output.region.height as usize {
            let src_row = (output.region.y as usize + row) * image_width + output.region.x as usize;
            let dst_row = row * region_width;

            for col in 0..region_width {
                output.data[dst_row + col] =
                    F::Sample::from_f64(f64::from(self.surface[src_row + col]));
            }
        }
    }
}

impl<F> PixelLocalOp for FractSurfOp<F>
where
    F: BandFormat,
    F::Sample: FromF64,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use crate::domain::{
        format::F32,
        image::{Region, Tile, TileMut},
    };
    use proptest::prelude::*;

    const GOLDEN_FRACTSURF_SIZE: u32 = 48;
    const GOLDEN_FRACTSURF_DIMENSION: f64 = 2.5;
    const HURST_TOLERANCE: f64 = 0.1;

    fn render(op: &FractSurfOp<F32>) -> Vec<f32> {
        let region = Region::new(0, 0, op.width(), op.height());
        let input_data = vec![0.0f32; region.pixel_count()];
        let mut output_data = vec![0.0f32; region.pixel_count()];
        let input = Tile::<F32>::new(region, 1, &input_data);
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);
        output_data
    }

    fn dft_rows(signal: &[f64], width: usize, height: usize) -> Vec<(f64, f64)> {
        let mut rows = vec![(0.0, 0.0); width * height];

        for y in 0..height {
            let row_offset = y * width;
            for kx in 0..width {
                let theta = -std::f64::consts::TAU * kx as f64 / width as f64;
                let step_re = theta.cos();
                let step_im = theta.sin();
                let mut phase_re = 1.0;
                let mut phase_im = 0.0;
                let mut sum_re = 0.0;
                let mut sum_im = 0.0;

                for x in 0..width {
                    let sample = signal[row_offset + x];
                    sum_re += sample * phase_re;
                    sum_im += sample * phase_im;

                    let next_re = phase_re * step_re - phase_im * step_im;
                    let next_im = phase_re * step_im + phase_im * step_re;
                    phase_re = next_re;
                    phase_im = next_im;
                }

                rows[row_offset + kx] = (sum_re, sum_im);
            }
        }

        rows
    }

    fn estimate_hurst_exponent(surface: &[f32], width: usize, height: usize) -> f64 {
        let mean =
            surface.iter().map(|&value| f64::from(value)).sum::<f64>() / surface.len() as f64;
        let centered = surface
            .iter()
            .map(|&value| f64::from(value) - mean)
            .collect::<Vec<_>>();
        let rows = dft_rows(&centered, width, height);
        let mut radial_power = BTreeMap::<u32, (f64, usize)>::new();

        for ky in 0..height {
            let fy = if ky <= height / 2 {
                ky as i32
            } else {
                ky as i32 - height as i32
            };

            for kx in 0..width {
                let fx = if kx <= width / 2 {
                    kx as i32
                } else {
                    kx as i32 - width as i32
                };
                let radius_sq = (fx * fx + fy * fy) as u32;
                if radius_sq == 0 {
                    continue;
                }

                let radius = f64::from(radius_sq).sqrt();
                if !(1.5..=(width.min(height) as f64 / 3.0)).contains(&radius) {
                    continue;
                }

                let theta = -std::f64::consts::TAU * ky as f64 / height as f64;
                let step_re = theta.cos();
                let step_im = theta.sin();
                let mut phase_re = 1.0;
                let mut phase_im = 0.0;
                let mut sum_re = 0.0;
                let mut sum_im = 0.0;

                for y in 0..height {
                    let (row_re, row_im) = rows[y * width + kx];
                    sum_re += row_re * phase_re - row_im * phase_im;
                    sum_im += row_re * phase_im + row_im * phase_re;

                    let next_re = phase_re * step_re - phase_im * step_im;
                    let next_im = phase_re * step_im + phase_im * step_re;
                    phase_re = next_re;
                    phase_im = next_im;
                }

                let power = sum_re.mul_add(sum_re, sum_im * sum_im);
                let entry = radial_power.entry(radius_sq).or_insert((0.0, 0));
                entry.0 += power;
                entry.1 += 1;
            }
        }

        let mut points = Vec::with_capacity(radial_power.len());
        for (radius_sq, (power_sum, count)) in radial_power {
            if count == 0 || power_sum <= 0.0 {
                continue;
            }
            let radius = f64::from(radius_sq).sqrt();
            let mean_power = power_sum / count as f64;
            points.push((radius.ln(), mean_power.ln()));
        }

        let count = points.len() as f64;
        let sum_x = points.iter().map(|(x, _)| *x).sum::<f64>();
        let sum_y = points.iter().map(|(_, y)| *y).sum::<f64>();
        let sum_xx = points.iter().map(|(x, _)| x * x).sum::<f64>();
        let sum_xy = points.iter().map(|(x, y)| x * y).sum::<f64>();
        let slope = (count * sum_xy - sum_x * sum_y) / (count * sum_xx - sum_x * sum_x);

        -(slope + 2.0) / 2.0
    }

    #[test]
    fn constructor_rejects_invalid_dimensions_and_dimension() {
        assert!(FractSurfOp::<F32>::new(0, 8, 2.5).is_err());
        assert!(FractSurfOp::<F32>::new(8, 0, 2.5).is_err());
        assert!(FractSurfOp::<F32>::new(8, 8, 1.5).is_err());
    }

    #[test]
    fn constructor_rejects_oversized_dimensions() {
        assert!(matches!(
            FractSurfOp::<F32>::new(u32::MAX, u32::MAX, 2.5),
            Err(ViprsError::ImageTooLarge { .. })
        ));
    }

    #[test]
    fn constructor_rejects_non_finite_fractal_dimensions() {
        assert!(FractSurfOp::<F32>::new(8, 8, f64::NAN).is_err());
        assert!(FractSurfOp::<F32>::new(8, 8, f64::INFINITY).is_err());
    }

    #[test]
    fn constructor_accepts_boundary_fractal_dimensions() {
        let low = FractSurfOp::<F32>::new(7, 11, 2.0).unwrap();
        let high = FractSurfOp::<F32>::new(7, 11, 3.0).unwrap();

        assert_eq!(render(&low).len(), 7 * 11);
        assert_eq!(render(&high).len(), 7 * 11);
    }

    #[test]
    fn checked_surface_pixel_count_accepts_non_trivial_dimensions() {
        assert_eq!(checked_surface_pixel_count(13, 29).unwrap(), 13 * 29);
    }

    #[test]
    fn checked_surface_pixel_count_reports_buffer_overflow_details() {
        assert!(matches!(
            checked_surface_pixel_count(u32::MAX, u32::MAX),
            Err(ViprsError::ImageTooLarge {
                width,
                height,
                bands: 1,
                details: "FractSurf surface buffer exceeds addressable memory",
                ..
            }) if width == u32::MAX && height == u32::MAX
        ));
    }

    #[test]
    fn build_gaussian_lattice_covers_valid_and_error_paths() {
        assert_eq!(build_gaussian_lattice(3, 5, 7).unwrap().len(), 15);
        assert!(matches!(
            build_gaussian_lattice(u32::MAX, u32::MAX, 0),
            Err(ViprsError::ImageTooLarge { .. })
        ));
    }

    #[test]
    fn debug_output_reports_shape_and_dimension() {
        let op = FractSurfOp::<F32>::new(4, 6, 2.75).unwrap();
        let debug = format!("{op:?}");

        assert!(debug.contains("FractSurfOp"));
        assert!(debug.contains("width: 4"));
        assert!(debug.contains("height: 6"));
        assert!(debug.contains("fractal_dimension: 2.75"));
    }

    #[test]
    fn generation_is_deterministic_for_rectangular_images() {
        let first = FractSurfOp::<F32>::new(13, 29, 2.5).unwrap();
        let second = FractSurfOp::<F32>::new(13, 29, 2.5).unwrap();

        assert_eq!(render(&first), render(&second));
    }

    #[test]
    fn rectangular_constructor_preserves_requested_dimensions() {
        let op = FractSurfOp::<F32>::new(21, 9, 2.3).unwrap();

        assert_eq!(op.width(), 21);
        assert_eq!(op.height(), 9);
        assert_eq!(render(&op).len(), 21 * 9);
    }

    #[test]
    fn single_pixel_surface_normalizes_to_mid_gray() {
        let surface = generate_surface(1, 1, 1, 2.5).unwrap();

        assert_eq!(&*surface, &[0.5]);
    }

    #[test]
    fn normalize_surface_maps_flat_input_to_mid_gray() {
        let mut surface = [3.5f32; 4];

        normalize_surface(&mut surface);

        assert_eq!(surface, [0.5; 4]);
    }

    #[test]
    fn op_trait_methods_keep_passthrough_contract() {
        let op = FractSurfOp::<F32>::new(5, 7, 2.5).unwrap();
        let region = Region::new(1, 2, 3, 4);

        assert_eq!(<FractSurfOp<F32> as Op>::OUTPUT_BANDS, Some(1));
        assert!(matches!(op.demand_hint(), DemandHint::Any));
        assert_eq!(op.required_input_region(&region), region);
        op.start();
    }

    #[test]
    fn golden_power_spectrum_matches_expected_hurst_exponent() {
        let op = FractSurfOp::<F32>::new(
            GOLDEN_FRACTSURF_SIZE,
            GOLDEN_FRACTSURF_SIZE,
            GOLDEN_FRACTSURF_DIMENSION,
        )
        .unwrap();
        let estimated_hurst = estimate_hurst_exponent(
            &render(&op),
            GOLDEN_FRACTSURF_SIZE as usize,
            GOLDEN_FRACTSURF_SIZE as usize,
        );
        let expected_hurst = 3.0 - GOLDEN_FRACTSURF_DIMENSION;

        assert!(
            (estimated_hurst - expected_hurst).abs() <= HURST_TOLERANCE,
            "expected Hurst exponent {expected_hurst:.3}, got {estimated_hurst:.3}",
        );
    }

    #[test]
    fn hurst_estimator_skips_zero_power_bins() {
        let estimated_hurst = estimate_hurst_exponent(&vec![0.0; 16], 4, 4);

        assert!(estimated_hurst.is_nan());
    }

    proptest! {
        #[test]
        fn prop_output_has_expected_dimensions_and_range(
            width in 1u32..=32,
            height in 1u32..=32,
            fractal_dimension in 2.0f64..=3.0,
        ) {
            let op = FractSurfOp::<F32>::new(width, height, fractal_dimension).unwrap();
            let samples = render(&op);

            prop_assert_eq!(samples.len(), width as usize * height as usize);
            prop_assert!(samples.iter().all(|sample| sample.is_finite()));
            prop_assert!(samples.iter().all(|sample| (0.0..=1.0).contains(sample)));
        }
    }
}

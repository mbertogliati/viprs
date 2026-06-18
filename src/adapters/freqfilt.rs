//! Frequency-domain helper functions backed by the FFT adapter.
//!
//! These utilities expose the libvips-style `fwfft` and `invfft` entrypoints as
//! ergonomic functions over `Image` values rather than pipeline nodes.

use crate::domain::{
    error::{FreqfiltError, ViprsError},
    format::{BandFormat, F64},
    image::Image,
    ops::resample::sample_conv::ToF64,
};
use rustfft::{FftPlanner, num_complex::Complex};

/// Frequency-domain images are stored as `Image<F64>` with bands `[real, imag]`.
/// The zero-frequency bin is centered with an `fftshift`-style layout so
/// frequency masks can be built around `(width / 2, height / 2)`.
///
/// This constant documents the storage layout used by [`fwfft`] and [`invfft`]
/// so callers can validate intermediate buffers or construct compatible test
/// fixtures.
///
/// # Examples
///
/// ```rust
/// use viprs::adapters::freqfilt::FFT_COMPLEX_BANDS;
///
/// assert_eq!(FFT_COMPLEX_BANDS, 2);
/// ```
pub const FFT_COMPLEX_BANDS: u32 = 2;

/// Transform a single-band real image to Fourier space.
///
/// The output layout matches libvips' normalization contract: each complex sample
/// is divided by `width * height`, and the zero-frequency bin is shifted to the
/// image centre, so `invfft(fwfft(image))` round-trips to the original real image
/// within floating-point error.
///
/// # Examples
///
/// ```rust
/// use viprs::{
///     adapters::freqfilt::{fwfft, FFT_COMPLEX_BANDS},
///     domain::{format::F32, image::Image},
/// };
///
/// let image = Image::<F32>::from_buffer(1, 1, 1, vec![1.0])?;
/// let spectrum = fwfft(&image)?;
/// assert_eq!(spectrum.bands(), FFT_COMPLEX_BANDS);
/// # Ok::<(), viprs::domain::error::ViprsError>(())
/// ```
pub fn fwfft<F>(input: &Image<F>) -> Result<Image<F64>, ViprsError>
where
    F: BandFormat,
    F::Sample: ToF64,
{
    if input.bands() != 1 {
        return Err(FreqfiltError::FwfftBands {
            bands: input.bands(),
        }
        .into());
    }

    let (width, height, pixel_count) = checked_dimensions(input.width(), input.height())?;
    if pixel_count == 0 {
        return Image::<F64>::from_buffer(
            input.width(),
            input.height(),
            FFT_COMPLEX_BANDS,
            Vec::new(),
        );
    }

    let mut spectrum: Vec<Complex<f64>> = input
        .pixels()
        .iter()
        .map(|sample| Complex::new(sample.to_f64(), 0.0))
        .collect();

    fft_2d_in_place(&mut spectrum, width, height, FftDirection::Forward);
    fftshift_2d(&mut spectrum, width, height);

    let scale = pixel_count as f64;
    let mut data = Vec::with_capacity(spectrum.len() * FFT_COMPLEX_BANDS as usize);
    for value in spectrum {
        data.push(value.re / scale);
        data.push(value.im / scale);
    }

    Image::<F64>::from_buffer(input.width(), input.height(), FFT_COMPLEX_BANDS, data)
}

/// Transform a complex frequency-domain image back to a real image.
///
/// `input` must be an `Image<F64>` with bands `[real, imag]`, typically produced by
/// [`fwfft`]. The inverse follows libvips' `real=true` path and returns only the
/// real component of the spatial-domain image.
///
/// # Examples
///
/// ```rust
/// use viprs::{
///     adapters::freqfilt::{fwfft, invfft},
///     domain::{format::F32, image::Image},
/// };
///
/// let image = Image::<F32>::from_buffer(1, 1, 1, vec![1.0])?;
/// let spatial = invfft(&fwfft(&image)?)?;
/// assert_eq!(spatial.bands(), 1);
/// # Ok::<(), viprs::domain::error::ViprsError>(())
/// ```
pub fn invfft(input: &Image<F64>) -> Result<Image<F64>, ViprsError> {
    if input.bands() != FFT_COMPLEX_BANDS {
        return Err(FreqfiltError::InvfftBands {
            bands: input.bands(),
        }
        .into());
    }

    let (width, height, pixel_count) = checked_dimensions(input.width(), input.height())?;
    if pixel_count == 0 {
        return Image::<F64>::from_buffer(input.width(), input.height(), 1, Vec::new());
    }

    let mut spatial: Vec<Complex<f64>> = input
        .pixels()
        .chunks_exact(FFT_COMPLEX_BANDS as usize)
        .map(|chunk| Complex::new(chunk[0], chunk[1]))
        .collect();

    ifftshift_2d(&mut spatial, width, height);
    fft_2d_in_place(&mut spatial, width, height, FftDirection::Inverse);

    let data = spatial.into_iter().map(|value| value.re).collect();
    Image::<F64>::from_buffer(input.width(), input.height(), 1, data)
}

#[derive(Copy, Clone)]
enum FftDirection {
    Forward,
    Inverse,
}

fn checked_dimensions(width: u32, height: u32) -> Result<(usize, usize, usize), ViprsError> {
    let width_usize =
        usize::try_from(width).map_err(|_| FreqfiltError::DimensionsOverflow { width, height })?;
    let height_usize =
        usize::try_from(height).map_err(|_| FreqfiltError::DimensionsOverflow { width, height })?;
    let pixel_count = width_usize
        .checked_mul(height_usize)
        .ok_or(FreqfiltError::DimensionsOverflow { width, height })?;
    Ok((width_usize, height_usize, pixel_count))
}

fn fft_2d_in_place(
    buffer: &mut [Complex<f64>],
    width: usize,
    height: usize,
    direction: FftDirection,
) {
    let mut planner = FftPlanner::<f64>::new();
    let row_fft = match direction {
        FftDirection::Forward => planner.plan_fft_forward(width),
        FftDirection::Inverse => planner.plan_fft_inverse(width),
    };
    for row in buffer.chunks_exact_mut(width) {
        row_fft.process(row);
    }

    let col_fft = match direction {
        FftDirection::Forward => planner.plan_fft_forward(height),
        FftDirection::Inverse => planner.plan_fft_inverse(height),
    };
    let mut column = vec![Complex::default(); height];

    for x in 0..width {
        for y in 0..height {
            column[y] = buffer[y * width + x];
        }
        col_fft.process(&mut column);
        for y in 0..height {
            buffer[y * width + x] = column[y];
        }
    }
}

fn fftshift_2d(buffer: &mut [Complex<f64>], width: usize, height: usize) {
    shift_2d(buffer, width, height, width / 2, height / 2);
}

fn ifftshift_2d(buffer: &mut [Complex<f64>], width: usize, height: usize) {
    shift_2d(buffer, width, height, width.div_ceil(2), height.div_ceil(2));
}

fn shift_2d(
    buffer: &mut [Complex<f64>],
    width: usize,
    height: usize,
    x_shift: usize,
    y_shift: usize,
) {
    let mut shifted = vec![Complex::default(); buffer.len()];
    for y in 0..height {
        for x in 0..width {
            let src_x = (x + x_shift) % width;
            let src_y = (y + y_shift) % height;
            shifted[y * width + x] = buffer[src_y * width + src_x];
        }
    }
    buffer.copy_from_slice(&shifted);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::format::{F32, U8};
    use proptest::prelude::*;

    prop_compose! {
        fn mono_images()
            (width in 1_u32..=4, height in 1_u32..=4)
            (
                width in Just(width),
                height in Just(height),
                pixels in prop::collection::vec(-1000.0_f32..1000.0_f32, (width * height) as usize),
            ) -> (u32, u32, Vec<f32>) {
                (width, height, pixels)
            }
    }

    proptest! {
        #[test]
        fn fwfft_then_invfft_round_trip_is_identity((width, height, pixels) in mono_images()) {
            let image = Image::<F32>::from_buffer(width, height, 1, pixels.clone()).unwrap();

            let spectrum = fwfft(&image).unwrap();
            let reconstructed = invfft(&spectrum).unwrap();

            prop_assert_eq!(spectrum.bands(), FFT_COMPLEX_BANDS);
            prop_assert_eq!(reconstructed.bands(), 1);
            for (expected, actual) in pixels.iter().zip(reconstructed.pixels().iter()) {
                prop_assert!((f64::from(*expected) - *actual).abs() <= 1e-6);
            }
        }
    }

    #[test]
    fn single_pixel_round_trip_preserves_value() {
        let image = Image::<U8>::from_buffer(1, 1, 1, vec![7]).unwrap();

        let spectrum = fwfft(&image).unwrap();
        let reconstructed = invfft(&spectrum).unwrap();

        assert_eq!(spectrum.pixels(), &[7.0, 0.0]);
        assert_eq!(reconstructed.pixels(), &[7.0]);
    }

    #[test]
    fn constant_signal_places_dc_at_image_center() {
        let image = Image::<F32>::from_buffer(4, 4, 1, vec![2.0; 16]).unwrap();

        let spectrum = fwfft(&image).unwrap();

        let dc_index = ((image.height() as usize / 2) * image.width() as usize
            + image.width() as usize / 2)
            * FFT_COMPLEX_BANDS as usize;
        for (index, pair) in spectrum
            .pixels()
            .chunks_exact(FFT_COMPLEX_BANDS as usize)
            .enumerate()
        {
            if index == dc_index / FFT_COMPLEX_BANDS as usize {
                assert!((pair[0] - 2.0).abs() <= 1e-6);
                assert!(pair[1].abs() <= 1e-6);
            } else {
                assert!(pair[0].abs() <= 1e-6);
                assert!(pair[1].abs() <= 1e-6);
            }
        }
    }

    #[test]
    fn empty_image_returns_empty_buffers() {
        let image = Image::<F32>::from_buffer(0, 0, 1, Vec::new()).unwrap();

        let spectrum = fwfft(&image).unwrap();
        let reconstructed = invfft(&spectrum).unwrap();

        assert!(spectrum.pixels().is_empty());
        assert!(reconstructed.pixels().is_empty());
    }

    #[test]
    fn fwfft_rejects_multiband_input() {
        let image = Image::<F32>::from_buffer(1, 1, 2, vec![1.0, 0.0]).unwrap();

        let err = fwfft(&image).unwrap_err();

        assert!(matches!(
            err,
            ViprsError::Freqfilt(FreqfiltError::FwfftBands { bands: 2 })
        ));
    }

    #[test]
    fn invfft_rejects_non_complex_layout() {
        let image = Image::<F64>::from_buffer(1, 1, 1, vec![1.0]).unwrap();

        let err = invfft(&image).unwrap_err();

        assert!(matches!(
            err,
            ViprsError::Freqfilt(FreqfiltError::InvfftBands { bands: 1 })
        ));
    }
}

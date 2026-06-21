//! Shared Gaussian-kernel construction used by convolution and conversion ops.

const GAUSSBLUR_COPY_SIGMA_THRESHOLD: f64 = 0.2;
const GAUSSBLUR_MIN_AMPL: f64 = 0.2;
const GAUSSBLUR_INTEGER_SCALE: f64 = 20.0;

#[derive(Clone)]
struct IntegerKernel1d {
    coeffs: Box<[i16]>,
    scale: i32,
}

fn integer_kernel_with_precision(sigma: f32, min_ampl: f64) -> IntegerKernel1d {
    let sigma = f64::from(sigma);
    if sigma < GAUSSBLUR_COPY_SIGMA_THRESHOLD {
        return IntegerKernel1d {
            coeffs: vec![1].into_boxed_slice(),
            scale: 1,
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
        };
    }

    IntegerKernel1d {
        coeffs: coeffs.into_boxed_slice(),
        scale,
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
/// Returns a normalised libvips-style 1-D Gaussian kernel.
pub fn gaussian_kernel_1d(sigma: f32) -> Vec<f64> {
    gaussian_kernel_1d_with_precision(sigma, GAUSSBLUR_MIN_AMPL, true)
}

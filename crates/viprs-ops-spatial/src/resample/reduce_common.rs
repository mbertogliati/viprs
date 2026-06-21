use viprs_core::{
    error::BuildError, image::Region, kernel::InterpolationKernel, resample::ReduceConfig,
};

pub const REDUCE_PHASES: usize = 64;
pub const REDUCE_FIXED_SHIFT: i32 = 12;
pub const REDUCE_FIXED_SCALE: i64 = 1_i64 << REDUCE_FIXED_SHIFT;
// libvips MAX_POINT (presample.h:70): reject kernels wider than this to prevent
// runaway memory and match reference behaviour exactly.
const MAX_REDUCE_TAPS: usize = 2_000;
const MAX_REDUCE_TAP_RADIUS: usize = (MAX_REDUCE_TAPS - 1) / 2;

#[inline]
fn saturating_output_end(origin: i32, len: u32) -> i32 {
    i64::from(origin)
        .saturating_add(i64::from(len.saturating_sub(1)))
        .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

#[inline]
fn clamp_i64_to_i32(value: i64) -> i32 {
    value.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

#[inline]
pub fn validate_reduce_factors(h_factor: f64, v_factor: f64) -> Result<(), BuildError> {
    if !h_factor.is_finite() || !v_factor.is_finite() {
        return Err(invalid_reduce_parameters(
            h_factor,
            v_factor,
            "factors must be finite",
        ));
    }

    if h_factor < 1.0 || v_factor < 1.0 {
        return Err(invalid_reduce_parameters(
            h_factor,
            v_factor,
            "factors must be >= 1.0",
        ));
    }

    Ok(())
}

#[inline]
const fn invalid_reduce_parameters(
    h_factor: f64,
    v_factor: f64,
    reason: &'static str,
) -> BuildError {
    BuildError::InvalidReduceParameters {
        h_factor,
        v_factor,
        reason,
    }
}

#[inline]
pub fn validate_reduce_tap_limits(
    h_factor: f64,
    v_factor: f64,
    kernel: InterpolationKernel,
) -> Result<(), BuildError> {
    validate_reduce_tap_limit(h_factor, v_factor, h_factor, kernel)?;
    validate_reduce_tap_limit(h_factor, v_factor, v_factor, kernel)
}

#[inline]
fn validate_reduce_tap_limit(
    h_factor: f64,
    v_factor: f64,
    factor: f64,
    kernel: InterpolationKernel,
) -> Result<(), BuildError> {
    if tap_count(kernel, factor).is_none() {
        return Err(invalid_reduce_parameters(
            h_factor,
            v_factor,
            "factor produces too many filter taps (limit: 2000, matching libvips MAX_POINT)",
        ));
    }

    Ok(())
}

pub struct ReduceKernel {
    config: ReduceConfig,
    coeffs_f64: Box<[f64]>,
    coeffs_i16: Box<[i16]>,
    offset: f64,
}

#[inline]
pub fn validate_reduce_kernel(
    op: &'static str,
    kernel: InterpolationKernel,
) -> Result<(), BuildError> {
    if kernel == InterpolationKernel::Lbb {
        return Err(BuildError::InvalidKernel {
            op,
            kernel,
            reason: "LBB is a nonlinear 2-D affine interpolator and is not valid for separable reduce kernels; use affine/resize instead",
        });
    }

    if kernel == InterpolationKernel::Vsqbs {
        return Err(BuildError::InvalidKernel {
            op,
            kernel,
            reason: "VSQBS is non-separable and is not valid for separable reduce kernels; use affine/resize instead",
        });
    }

    if kernel == InterpolationKernel::Nohalo {
        return Err(BuildError::InvalidKernel {
            op,
            kernel,
            reason: "Nohalo is a nonlinear 2-D interpolator in libvips and has no separable reduce kernel; use resize()/affine() mapping or one of: Nearest, Bilinear, Bicubic/CatmullRom, Lanczos2, Lanczos3",
        });
    }

    Ok(())
}

impl ReduceKernel {
    #[inline]
    fn plan_for_source_position(&self, source_position: f64) -> (i64, usize) {
        let floor = source_position.floor();
        let phase = (source_position - floor)
            .mul_add(REDUCE_PHASES as f64, 0.0)
            .round()
            .clamp(0.0, REDUCE_PHASES as f64) as usize;
        (floor as i64 - self.config.pad_before, phase)
    }

    pub(crate) fn new(factor: f64, kernel: InterpolationKernel) -> Result<Self, BuildError> {
        validate_reduce_factors(factor, factor)?;
        let taps = tap_count(kernel, factor).ok_or_else(|| {
            invalid_reduce_parameters(factor, factor, "factor produces too many filter taps")
        })?;
        let phase_count = REDUCE_PHASES + 1;
        let pad_before = ((taps as f64) / 2.0).ceil() as i64 - 1;
        let mut coeffs_f64 = Vec::with_capacity(phase_count * taps);
        let mut coeffs_i16 = Vec::with_capacity(phase_count * taps);

        for phase in 0..phase_count {
            let coeffs =
                build_coefficients(kernel, taps, factor, phase as f64 / REDUCE_PHASES as f64);
            coeffs_i16.extend(quantize_coefficients(&coeffs).iter().copied());
            coeffs_f64.extend(coeffs);
        }

        Ok(Self {
            config: ReduceConfig {
                factor,
                taps: taps as u32,
                pad_before,
            },
            coeffs_f64: coeffs_f64.into_boxed_slice(),
            coeffs_i16: coeffs_i16.into_boxed_slice(),
            offset: 0.0,
        })
    }

    #[inline]
    pub(crate) const fn config(&self) -> ReduceConfig {
        self.config
    }

    pub(crate) fn bind_input_len(&mut self, input_len: u32) {
        self.offset = if input_len == 0 || (self.config.factor - 1.0).abs() < f64::EPSILON {
            0.0
        } else {
            let output_len = self.config.output_width(input_len);
            let extra_pixels =
                f64::from(output_len).mul_add(self.config.factor, -f64::from(input_len));
            f64::midpoint(1.0, extra_pixels) - 1.0
        };
    }

    #[inline]
    pub(crate) fn source_position(&self, index: f64) -> f64 {
        (index + 0.5).mul_add(self.config.factor, -0.5) - self.offset
    }

    #[inline]
    pub(crate) fn taps_for_f64(&self, source_position: f64) -> (i64, &[f64]) {
        let (start, phase) = self.plan_for_source_position(source_position);
        (start, self.coeffs_f64_for_phase(phase))
    }

    #[inline]
    pub(crate) fn taps_for_i16(&self, source_position: f64) -> (i64, &[i16]) {
        let (start, phase) = self.plan_for_source_position(source_position);
        (start, self.coeffs_i16_for_phase(phase))
    }

    #[inline]
    pub(crate) fn plan_i16(&self, source_position: f64) -> (i64, usize) {
        self.plan_for_source_position(source_position)
    }

    #[inline]
    pub(crate) fn coeffs_i16_for_phase(&self, phase: usize) -> &[i16] {
        let taps = self.config.taps as usize;
        let start = phase * taps;
        &self.coeffs_i16[start..start + taps]
    }

    #[inline]
    fn coeffs_f64_for_phase(&self, phase: usize) -> &[f64] {
        let taps = self.config.taps as usize;
        let start = phase * taps;
        &self.coeffs_f64[start..start + taps]
    }

    #[inline]
    pub(crate) fn required_input_region_h(&self, output: &Region) -> Region {
        if output.width == 0 {
            return Region::new(output.x, output.y, 0, output.height);
        }

        let first_src = self.source_position(f64::from(output.x)).floor() as i64;
        let last_x = saturating_output_end(output.x, output.width);
        let last_src = self.source_position(f64::from(last_x)).floor() as i64;
        let start = first_src - self.config.pad_before;
        let end = last_src - self.config.pad_before + i64::from(self.config.taps) - 1;
        let clamped_start = clamp_i64_to_i32(start);
        let clamped_end = clamp_i64_to_i32(end);
        let width = i64::from(clamped_end)
            .saturating_sub(i64::from(clamped_start))
            .saturating_add(1) as u32;
        Region::new(clamped_start, output.y, width, output.height)
    }

    #[inline]
    pub(crate) fn required_input_region_v(&self, output: &Region) -> Region {
        if output.height == 0 {
            return Region::new(output.x, output.y, output.width, 0);
        }

        let first_src = self.source_position(f64::from(output.y)).floor() as i64;
        let last_y = saturating_output_end(output.y, output.height);
        let last_src = self.source_position(f64::from(last_y)).floor() as i64;
        let start = first_src - self.config.pad_before;
        let end = last_src - self.config.pad_before + i64::from(self.config.taps) - 1;
        let clamped_start = clamp_i64_to_i32(start);
        let clamped_end = clamp_i64_to_i32(end);
        let height = i64::from(clamped_end)
            .saturating_sub(i64::from(clamped_start))
            .saturating_add(1) as u32;
        Region::new(output.x, clamped_start, output.width, height)
    }
}

#[inline]
fn tap_count(kernel: InterpolationKernel, factor: f64) -> Option<usize> {
    if (factor - 1.0).abs() < f64::EPSILON {
        return Some(1);
    }

    if matches!(kernel, InterpolationKernel::Nearest) {
        return Some(1);
    }

    let tap_radius = (kernel.support() * factor).round();
    if !tap_radius.is_finite() || tap_radius < 0.0 || tap_radius > MAX_REDUCE_TAP_RADIUS as f64 {
        return None;
    }

    let tap_radius = tap_radius as usize;
    tap_radius
        .checked_mul(2)
        .and_then(|taps| taps.checked_add(1))
        .filter(|&taps| taps <= MAX_REDUCE_TAPS)
}

fn build_coefficients(
    kernel: InterpolationKernel,
    taps: usize,
    factor: f64,
    phase: f64,
) -> Vec<f64> {
    if matches!(kernel, InterpolationKernel::Nearest) {
        return vec![1.0];
    }

    let half = phase + taps as f64 / 2.0 - 1.0;
    let scale = 1.0 / factor;
    let mut coeffs = Vec::with_capacity(taps);
    let mut sum = 0.0;

    for tap in 0..taps {
        let distance = ((tap as f64) - half) * scale;
        let weight = kernel.weight(distance.abs());
        coeffs.push(weight);
        sum += weight;
    }

    if sum > 0.0 {
        for coeff in &mut coeffs {
            *coeff /= sum;
        }
    }

    coeffs
}

fn quantize_coefficients(coeffs: &[f64]) -> Box<[i16]> {
    coeffs
        .iter()
        .map(|coeff| (*coeff * REDUCE_FIXED_SCALE as f64) as i16)
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

#[inline]
pub fn clamp_axis(abs: i64, tile_origin: i32, tile_len: usize) -> usize {
    let rel = abs - i64::from(tile_origin);
    rel.clamp(0, tile_len.saturating_sub(1) as i64) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sum_phase(coeffs: &[f64]) -> f64 {
        coeffs.iter().sum::<f64>()
    }

    #[test]
    fn linear_coefficients_normalize_for_every_phase() {
        let kernel = ReduceKernel::new(2.0, InterpolationKernel::Bilinear).unwrap();
        for phase in 0..=REDUCE_PHASES {
            let coeffs = kernel.coeffs_f64_for_phase(phase);
            assert!((sum_phase(coeffs) - 1.0).abs() < 1e-12, "phase {phase}");
        }
    }

    #[test]
    fn lanczos3_coefficients_normalize_for_every_phase() {
        let kernel = ReduceKernel::new(1.5, InterpolationKernel::Lanczos3).unwrap();
        for phase in 0..=REDUCE_PHASES {
            let coeffs = kernel.coeffs_f64_for_phase(phase);
            assert!((sum_phase(coeffs) - 1.0).abs() < 1e-12, "phase {phase}");
        }
    }

    #[test]
    fn fixed_point_coefficients_track_libvips_truncation_budget() {
        for kernel in [
            InterpolationKernel::Nearest,
            InterpolationKernel::Bilinear,
            InterpolationKernel::Bicubic,
            InterpolationKernel::CatmullRom,
            InterpolationKernel::Lanczos2,
            InterpolationKernel::Lanczos3,
        ] {
            for factor in [1.0, 1.5, 2.0, 2.5, 4.0] {
                let reduce_kernel = ReduceKernel::new(factor, kernel).unwrap();
                for phase in 0..=REDUCE_PHASES {
                    let coeffs = reduce_kernel.coeffs_i16_for_phase(phase);
                    let sum = coeffs.iter().map(|coeff| i64::from(*coeff)).sum::<i64>();
                    assert!(
                        (sum - REDUCE_FIXED_SCALE).abs() <= i64::from(reduce_kernel.config.taps),
                        "kernel={kernel:?} factor={factor} phase={phase} sum={sum}"
                    );
                }
            }
        }
    }

    #[test]
    fn bound_offset_matches_libvips_centering_for_factor2() {
        let mut kernel = ReduceKernel::new(2.0, InterpolationKernel::Bilinear).unwrap();
        kernel.bind_input_len(8);
        assert!((kernel.offset + 0.5).abs() < 1e-12);

        let source_x = kernel.source_position(0.0);
        assert!((source_x - 1.0).abs() < 1e-12);
        let (start_x, _) = kernel.taps_for_i16(source_x);
        assert_eq!(start_x, -1);
    }

    #[test]
    fn required_input_region_keeps_negative_halo_for_edge_copy() {
        let mut kernel = ReduceKernel::new(2.0, InterpolationKernel::Bilinear).unwrap();
        kernel.bind_input_len(4);
        let region = kernel.required_input_region_h(&Region::new(0, 0, 2, 1));
        assert_eq!(region, Region::new(-1, 0, 7, 1));
    }

    #[test]
    fn required_input_region_h_clamps_max_output_coordinates() {
        let kernel = ReduceKernel::new(1.5, InterpolationKernel::Lanczos3).unwrap();

        assert_eq!(
            kernel.required_input_region_h(&Region::new(i32::MAX, 0, 2, 1)),
            Region::new(i32::MAX, 0, 1, 1)
        );
    }

    #[test]
    fn required_input_region_v_clamps_max_output_coordinates() {
        let kernel = ReduceKernel::new(1.5, InterpolationKernel::Lanczos3).unwrap();

        assert_eq!(
            kernel.required_input_region_v(&Region::new(0, i32::MAX, 1, 2)),
            Region::new(0, i32::MAX, 1, 1)
        );
    }

    // Lanczos3 support = 3.0; factor needed so taps > MAX_REDUCE_TAPS (2000).
    // taps = 2 * ceil(3 * factor) + 1 > 2000 → factor > (2000-1)/2/3 ≈ 333.17
    #[test]
    fn validate_tap_limits_rejects_factor_exceeding_max_point() {
        use viprs_core::kernel::InterpolationKernel;
        let result = validate_reduce_tap_limits(400.0, 400.0, InterpolationKernel::Lanczos3);
        assert!(
            result.is_err(),
            "factor 400 should exceed libvips MAX_POINT tap limit"
        );
    }

    // Modest factor well within limit must succeed.
    #[test]
    fn validate_tap_limits_accepts_factor_below_max_point() {
        use viprs_core::kernel::InterpolationKernel;
        let result = validate_reduce_tap_limits(4.0, 4.0, InterpolationKernel::Lanczos3);
        assert!(result.is_ok(), "factor 4 should be within tap limit");
    }
}

use std::marker::PhantomData;

use viprs_core::{
    error::BuildError,
    format::NumericBand,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

/// Linear transform: `output = input * scale + offset`, applied per-sample.
///
/// The same scale and offset are applied to every band of every pixel.
///
/// # Examples
/// ```ignore
/// use viprs_ops_pixel::arithmetic::linear::Linear;
///
/// let op = Linear::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct Linear<F: NumericBand> {
    scale: f64,
    offset: f64,
    _format: PhantomData<F>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LinearI16Coefficients {
    scale: i16,
    offset: i16,
}

impl LinearI16Coefficients {
    #[inline]
    fn for_u8(scale: f64, offset: f64) -> Option<Self> {
        if !scale.is_finite() || !offset.is_finite() {
            return None;
        }

        let scale_i32 = scale as i32;
        let offset_i32 = offset as i32;
        if f64::from(scale_i32) != scale || f64::from(offset_i32) != offset {
            return None;
        }

        let lo = offset_i32;
        let hi = (i32::from(u8::MAX) * scale_i32) + offset_i32;
        let min_value = lo.min(hi);
        let max_value = lo.max(hi);
        if min_value < i32::from(i16::MIN) || max_value > i32::from(i16::MAX) {
            return None;
        }

        Some(Self {
            scale: scale_i32 as i16,
            offset: offset_i32 as i16,
        })
    }
}

impl<F: NumericBand> Linear<F> {
    /// Creates a new `Linear`.
    pub fn new<Scale, Offset>(scale: Scale, offset: Offset) -> Result<Self, BuildError>
    where
        Scale: Into<f64>,
        Offset: Into<f64>,
    {
        let scale = scale.into();
        let offset = offset.into();
        validate_linear_parameters(scale, offset)?;

        Ok(Self {
            scale,
            offset,
            _format: PhantomData,
        })
    }
}

#[inline]
const fn validate_linear_parameters(scale: f64, offset: f64) -> Result<(), BuildError> {
    if scale.is_nan() || scale.is_infinite() || offset.is_nan() || offset.is_infinite() {
        return Err(BuildError::InvalidLinearParameters { scale, offset });
    }

    Ok(())
}

trait LinearSample: Copy + 'static {
    fn linear(self, scale: f64, offset: f64) -> Self;

    #[inline]
    fn linear_bulk(input: &[Self], output: &mut [Self], scale: f64, offset: f64) {
        for (s, d) in input.iter().zip(output.iter_mut()) {
            *d = s.linear(scale, offset);
        }
    }
}

macro_rules! impl_linear_sample_int {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl LinearSample for $ty {
                #[inline(always)]
                fn linear(self, scale: f64, offset: f64) -> Self {
                    let value = f64::from(self).mul_add(scale, offset);
                    value.min(<$ty>::MAX as f64).max(<$ty>::MIN as f64) as $ty
                }
            }
        )+
    };
}

impl_linear_sample_int!(u8, u16, i16, u32, i32);

impl LinearSample for f32 {
    #[inline(always)]
    fn linear(self, scale: f64, offset: f64) -> Self {
        self.mul_add(scale as Self, offset as Self)
    }

    #[inline]
    fn linear_bulk(input: &[Self], output: &mut [Self], scale: f64, offset: f64) {
        linear_bulk_f32(input, output, scale as Self, offset as Self);
    }
}

impl LinearSample for f64 {
    #[inline(always)]
    fn linear(self, scale: f64, offset: f64) -> Self {
        self.mul_add(scale, offset)
    }
}

impl<F> Op for Linear<F>
where
    F: NumericBand,
    F::Sample: LinearSample,
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

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        F::Sample::linear_bulk(input.data, output.data, self.scale, self.offset);
    }
}

/// `Linear` is pixel-local: output(x,y) depends only on input(x,y).
/// No neighbourhood access; identity tile geometry. See `PixelLocalOp` for invariants.
impl<F> PixelLocalOp for Linear<F>
where
    F: NumericBand,
    F::Sample: LinearSample,
{
}

/// Specialized linear kernel for `Linear<U8>` with integer coefficients.
///
/// Uses i16 arithmetic that LLVM can auto-vectorize into NEON/AVX2 smull+sqxtun.
/// This avoids the f64 round-trip of the generic `LinearSample::linear` path.
pub struct LinearKernelU8 {
    scale: i16,
    offset: i16,
}

impl LinearKernelU8 {
    /// Construct from a `Linear<U8>` if its coefficients fit in i16.
    /// Returns `None` if the coefficients require f64 precision.
    #[must_use]
    pub fn from_linear(linear: &Linear<viprs_core::format::U8>) -> Option<Self> {
        LinearI16Coefficients::for_u8(linear.scale, linear.offset).map(|c| Self {
            scale: c.scale,
            offset: c.offset,
        })
    }

    /// Construct directly from i16 coefficients.
    #[must_use]
    pub const fn new(scale: i16, offset: i16) -> Self {
        Self { scale, offset }
    }
}

impl Op for LinearKernelU8 {
    type Input = viprs_core::format::U8;
    type Output = viprs_core::format::U8;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(
        &self,
        _state: &mut (),
        input: &Tile<Self::Input>,
        output: &mut TileMut<Self::Output>,
    ) {
        let scale = self.scale;
        let offset = self.offset;
        for (s, d) in input.data.iter().zip(output.data.iter_mut()) {
            let v = i16::from(*s) * scale + offset;
            *d = v.clamp(0, 255) as u8;
        }
    }
}

impl PixelLocalOp for LinearKernelU8 {}

#[inline]
fn linear_bulk_f32_scalar(input: &[f32], output: &mut [f32], scale: f32, offset: f32) {
    for (s, d) in input.iter().zip(output.iter_mut()) {
        *d = s.mul_add(scale, offset);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn linear_bulk_f32(input: &[f32], output: &mut [f32], scale: f32, offset: f32) {
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: runtime dispatch guarantees the required AVX2+FMA features and the helper
        // only touches the common in-bounds prefix of `input` and `output`.
        unsafe {
            linear_bulk_f32_avx2(input, output, scale, offset);
        }
    } else {
        linear_bulk_f32_scalar(input, output, scale, offset);
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline]
fn linear_bulk_f32(input: &[f32], output: &mut [f32], scale: f32, offset: f32) {
    linear_bulk_f32_scalar(input, output, scale, offset);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
// SAFETY: caller must dispatch only when AVX2 and FMA are available and pass slices whose
// common prefix is valid for unaligned 8-lane loads/stores plus the scalar tail.
#[allow(clippy::cast_ptr_alignment)]
unsafe fn linear_bulk_f32_avx2(input: &[f32], output: &mut [f32], scale: f32, offset: f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps};

    let len = input.len().min(output.len());
    let chunks = len / 16;
    let remainder_start = chunks * 16;
    let scale_v = _mm256_set1_ps(scale);
    let offset_v = _mm256_set1_ps(offset);

    for chunk in 0..chunks {
        let base = chunk * 16;
        // SAFETY: `base + 16 <= len`, both pointers are valid for two unaligned 8-lane
        // float loads/stores, and AVX2+FMA availability is guaranteed by the caller.
        unsafe {
            let v0 = _mm256_loadu_ps(input.as_ptr().add(base));
            let v1 = _mm256_loadu_ps(input.as_ptr().add(base + 8));
            _mm256_storeu_ps(
                output.as_mut_ptr().add(base),
                _mm256_fmadd_ps(v0, scale_v, offset_v),
            );
            _mm256_storeu_ps(
                output.as_mut_ptr().add(base + 8),
                _mm256_fmadd_ps(v1, scale_v, offset_v),
            );
        }
    }

    linear_bulk_f32_scalar(
        &input[remainder_start..len],
        &mut output[remainder_start..len],
        scale,
        offset,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::{F32, U8, U16},
        image::{Region, Tile, TileMut},
    };

    // Allocation tests require the root crate test_support (global allocator).
    // Run via: cargo test -p viprs --lib

    fn make_region(w: u32, h: u32) -> Region {
        Region::new(0, 0, w, h)
    }

    #[test]
    fn linear_identity_scale_zero_offset() {
        let op = Linear::<F32>::new(1.0, 0.0).unwrap();
        let r = make_region(2, 2);
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut output_data = vec![0.0f32; 4];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, input_data);
    }

    proptest! {
        #[test]
        fn linear_identity_prop(
            pixels in proptest::collection::vec(0.0f32..=1.0f32, 1..=64)
        ) {
            let len = pixels.len();
            let op = Linear::<F32>::new(1.0, 0.0).unwrap();
            let r = Region::new(0, 0, len as u32, 1);
            let mut output_data = vec![0.0f32; len];
            let input = Tile::<F32>::new(r, 1, &pixels);
            let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
            let mut state = ();
            op.process_region(&mut state, &input, &mut output);
            prop_assert_eq!(output_data, pixels);
        }
    }

    /// Ported from libvips `test_arithmetic.py` via the `linear` operation.
    ///
    /// libvips uses `vips_linear` for `im * [1,2,3] + [2,3,4]` in test setup.
    /// This test verifies the per-channel linear transform: `output = input * scale + offset`.
    ///
    /// Reference: input=3.0, scale=2.0, offset=4.0 → output=10.0
    #[test]
    fn linear_scale_and_offset_known_values() {
        let op = Linear::<F32>::new(2.0, 4.0).unwrap();
        let r = make_region(3, 1);
        let input_data = vec![0.0f32, 1.5, 3.0];
        let mut output_data = vec![0.0f32; 3];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);

        assert!(
            (output_data[0] - 4.0).abs() < f32::EPSILON,
            "0*2+4={}",
            output_data[0]
        );
        assert!(
            (output_data[1] - 7.0).abs() < f32::EPSILON,
            "1.5*2+4={}",
            output_data[1]
        );
        assert!(
            (output_data[2] - 10.0).abs() < f32::EPSILON,
            "3*2+4={}",
            output_data[2]
        );
    }

    /// Ported from libvips `test_arithmetic.py`.
    ///
    /// libvips test: `(image + 100).deviate() ≈ 0` when the image is constant.
    /// A constant image has zero variance regardless of scale or offset.
    /// Here we verify: `linear(constant_tile`, scale=5, offset=7) → constant output.
    #[test]
    fn linear_constant_input_stays_constant() {
        let op = Linear::<F32>::new(5.0, 7.0).unwrap();
        let r = make_region(4, 1);
        let input_data = vec![2.0f32; 4];
        let mut output_data = vec![0.0f32; 4];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);

        // 2.0 * 5.0 + 7.0 = 17.0 for every pixel
        assert!(
            output_data.iter().all(|&v| (v - 17.0).abs() < f32::EPSILON),
            "constant input must produce constant output: {:?}",
            &output_data
        );
    }

    fn run_linear_u8(scale: f64, offset: f64, input_data: &[u8]) -> Vec<u8> {
        run_linear_u8_with_bands(scale, offset, 1, input_data)
    }

    fn run_linear_u8_with_bands(scale: f64, offset: f64, bands: u32, input_data: &[u8]) -> Vec<u8> {
        debug_assert_eq!(input_data.len() % bands as usize, 0);
        let width = (input_data.len() / bands as usize) as u32;
        let region = Region::new(0, 0, width, 1);
        let op = Linear::<U8>::new(scale, offset).unwrap();
        let mut output_data = vec![0u8; input_data.len()];
        let input = Tile::<U8>::new(region, bands, input_data);
        let mut output = TileMut::<U8>::new(region, bands, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    fn run_linear_u16(scale: f64, offset: f64, input_data: &[u16]) -> Vec<u16> {
        let region = Region::new(0, 0, input_data.len() as u32, 1);
        let op = Linear::<U16>::new(scale, offset).unwrap();
        let mut output_data = vec![0u16; input_data.len()];
        let input = Tile::<U16>::new(region, 1, input_data);
        let mut output = TileMut::<U16>::new(region, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn run_linear_f32(scale: f32, offset: f32, input_data: &[f32]) -> Vec<f32> {
        let region = Region::new(0, 0, input_data.len() as u32, 1);
        let op = Linear::<F32>::new(scale, offset).unwrap();
        let mut output_data = vec![0.0f32; input_data.len()];
        let input = Tile::<F32>::new(region, 1, input_data);
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    #[test]
    fn linear_new_rejects_non_finite_parameters() {
        assert!(matches!(
            Linear::<F32>::new(f64::NAN, 0.0),
            Err(BuildError::InvalidLinearParameters { scale, offset })
                if scale.is_nan() && offset == 0.0
        ));
        assert!(matches!(
            Linear::<F32>::new(f64::INFINITY, 0.0),
            Err(BuildError::InvalidLinearParameters { scale, offset })
                if scale.is_infinite() && offset == 0.0
        ));
        assert!(matches!(
            Linear::<F32>::new(1.0, f64::NAN),
            Err(BuildError::InvalidLinearParameters { scale, offset })
                if scale == 1.0 && offset.is_nan()
        ));
        assert!(matches!(
            Linear::<F32>::new(1.0, f64::INFINITY),
            Err(BuildError::InvalidLinearParameters { scale, offset })
                if scale == 1.0 && offset.is_infinite()
        ));
    }

    #[test]
    fn linear_new_accepts_zero_scale() {
        assert!(Linear::<F32>::new(0.0, 0.0).is_ok());
    }

    #[test]
    fn linear_valid_params_succeeds() {
        assert!(Linear::<F32>::new(1.5, -2.0).is_ok());
    }

    fn linear_reference_u8(sample: u8, scale: f64, offset: f64) -> u8 {
        let value = f64::from(sample).mul_add(scale, offset);
        value.min(f64::from(u8::MAX)).max(f64::from(u8::MIN)) as u8
    }

    fn linear_reference_u16(sample: u16, scale: f64, offset: f64) -> u16 {
        let value = f64::from(sample).mul_add(scale, offset);
        value.min(f64::from(u16::MAX)).max(f64::from(u16::MIN)) as u16
    }

    #[test]
    fn linear_u8_clips_and_truncates_like_libvips() {
        let output = run_linear_u8(1.5, 10.9, &[0, 10, 250, 255]);
        assert_eq!(output, vec![10, 25, 255, 255]);
    }

    #[test]
    fn linear_u16_clips_and_truncates_like_libvips() {
        let output = run_linear_u16(1.5, -3.25, &[0, 1, 40_000, u16::MAX]);
        assert_eq!(output, vec![0, 0, 59_996, u16::MAX]);
    }

    #[test]
    fn linear_u8_integer_coefficients_match_reference() {
        let input = [0, 1, 8, 16, 64, 120, 125, 255];
        let output = run_linear_u8(2.0, 5.0, &input);
        let expected: Vec<u8> = input
            .iter()
            .copied()
            .map(|sample| linear_reference_u8(sample, 2.0, 5.0))
            .collect();
        assert_eq!(output, expected);
    }

    #[test]
    fn linear_u8_negative_integer_scale_matches_reference() {
        let input = [0, 10, 64, 120, 200, 255];
        let output = run_linear_u8(-2.0, 255.0, &input);
        let expected: Vec<u8> = input
            .iter()
            .copied()
            .map(|sample| linear_reference_u8(sample, -2.0, 255.0))
            .collect();
        assert_eq!(output, expected);
    }

    #[test]
    fn linear_u8_f32_exact_coefficients_match_reference_for_1_3_4_bands() {
        let input = [
            0, 12, 64, 255, 7, 31, 120, 240, 9, 18, 33, 66, 99, 123, 200, 250, 3, 15, 27, 39, 51,
            63, 75, 87,
        ];

        for bands in [1_u32, 3, 4] {
            let usable_len = input.len() - (input.len() % bands as usize);
            let pixels = &input[..usable_len];
            let output = run_linear_u8_with_bands(1.25, 10.0, bands, pixels);
            let expected: Vec<u8> = pixels
                .iter()
                .copied()
                .map(|sample| linear_reference_u8(sample, 1.25, 10.0))
                .collect();
            assert_eq!(output, expected, "bands={bands}");
        }
    }

    #[test]
    fn linear_metadata_and_large_exact_u8_rows_cover_runtime_dispatch() {
        let op = Linear::<U8>::new(1.25, 10.0).unwrap();
        let region = Region::new(4, -2, 17, 1);
        assert_eq!(op.demand_hint(), viprs_core::image::DemandHint::ThinStrip);
        assert_eq!(op.required_input_region(&region), region);
        assert_eq!(
            op.node_spec(17, 1),
            viprs_core::op::NodeSpec::identity(17, 1)
        );
        op.start();

        let input = (0..17u8).collect::<Vec<_>>();
        let output = run_linear_u8(1.25, 10.0, &input);
        let expected: Vec<u8> = input
            .iter()
            .copied()
            .map(|sample| linear_reference_u8(sample, 1.25, 10.0))
            .collect();
        assert_eq!(output, expected);
    }

    #[test]
    fn coefficient_helpers_and_scalar_paths_cover_integer_and_float_formats() {
        assert_eq!(
            LinearI16Coefficients::for_u8(2.0, 5.0),
            Some(LinearI16Coefficients {
                scale: 2,
                offset: 5
            })
        );
        assert_eq!(LinearI16Coefficients::for_u8(1.25, 10.0), None);
        assert_eq!(LinearI16Coefficients::for_u8(f64::INFINITY, 1.0), None);

        assert_eq!(<i32 as LinearSample>::linear(7, 2.0, -3.0), 11);
        assert!((<f64 as LinearSample>::linear(0.5, 2.0, -0.25) - 0.75).abs() < f64::EPSILON);

        let coefficients = LinearI16Coefficients::for_u8(2.0, 5.0).unwrap();
        let input: &[u8] = &[0, 10, 200];
        let output: Vec<u8> = input
            .iter()
            .map(|&s| {
                let v = (i16::from(s) * coefficients.scale) + coefficients.offset;
                v.clamp(0, i16::from(u8::MAX)) as u8
            })
            .collect();
        assert_eq!(output, vec![5, 25, 255]);
    }

    proptest! {
        #[test]
        fn linear_u8_matches_clipped_float_reference(
            pixels in proptest::collection::vec(any::<u8>(), 1..=64),
            scale in -8.0f64..8.0,
            offset in -1024.0f64..1024.0,
        ) {
            let output = run_linear_u8(scale, offset, &pixels);
            let expected: Vec<u8> = pixels
                .iter()
                .copied()
                .map(|sample| linear_reference_u8(sample, scale, offset))
                .collect();
            prop_assert_eq!(output, expected);
        }

        #[test]
        fn linear_u16_matches_clipped_float_reference(
            pixels in proptest::collection::vec(any::<u16>(), 1..=64),
            scale in -8.0f64..8.0,
            offset in -131_072.0_f64..131_072.0,
        ) {
            let output = run_linear_u16(scale, offset, &pixels);
            let expected: Vec<u16> = pixels
                .iter()
                .copied()
                .map(|sample| linear_reference_u16(sample, scale, offset))
                .collect();
            prop_assert_eq!(output, expected);
        }

        #[test]
        fn linear_u8_f32_exact_coefficients_match_reference_across_band_counts(
            bands in prop_oneof![Just(1_u32), Just(3_u32), Just(4_u32)],
            pixels in proptest::collection::vec(any::<u8>(), 4..=96),
        ) {
            let usable_len = pixels.len() - (pixels.len() % bands as usize);
            prop_assume!(usable_len > 0);
            let pixels = &pixels[..usable_len];
            let output = run_linear_u8_with_bands(1.25, 10.0, bands, pixels);
            let expected: Vec<u8> = pixels
                .iter()
                .copied()
                .map(|sample| linear_reference_u8(sample, 1.25, 10.0))
                .collect();
            prop_assert_eq!(output, expected);
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    proptest! {
        #[test]
        fn linear_f32_avx2_matches_scalar_reference(
            pixels in proptest::collection::vec(-1024.0f32..1024.0f32, 17..=257),
            scale in -8.0f32..8.0f32,
            offset in -2048.0f32..2048.0f32,
        ) {
            if !(std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma"))
            {
                return;
            }

            let output = run_linear_f32(scale, offset, &pixels);
            let expected: Vec<f32> = pixels
                .iter()
                .map(|&sample| sample.mul_add(scale, offset))
                .collect();
            prop_assert_eq!(output, expected);
        }
    }
}

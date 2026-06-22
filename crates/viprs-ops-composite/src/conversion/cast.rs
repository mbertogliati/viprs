use std::marker::PhantomData;

use bytemuck::{Pod, try_cast_slice, try_cast_slice_mut};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    vaddq_f32, vcombine_u16, vcvtq_f32_u32, vcvtq_u32_f32, vdup_n_u16, vdupq_n_f32, vget_high_u16,
    vget_lane_u32, vget_low_u16, vld1_u8, vld1q_f32, vmaxq_f32, vminq_f32, vmovl_u8, vmovl_u16,
    vmulq_n_f32, vqmovn_u16, vqmovn_u32, vreinterpret_u32_u8, vst1q_f32,
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128i, _mm_loadl_epi64, _mm_packus_epi16, _mm_packus_epi32, _mm_setzero_si128,
    _mm_storel_epi64, _mm256_add_ps, _mm256_castsi256_si128, _mm256_cvtepi32_ps,
    _mm256_cvtepu8_epi32, _mm256_cvttps_epi32, _mm256_extracti128_si256, _mm256_loadu_ps,
    _mm256_max_ps, _mm256_min_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps,
};

use viprs_core::{
    format::{BandFormat, BandFormatId},
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
    simd::SimdLevel,
};

const U8_TO_F32_SCALE: f32 = 1.0 / 255.0;
const F32_TO_U8_SCALE: f32 = 255.0;
const F32_TO_U8_ROUND_BIAS: f32 = 0.5;

pub use viprs_core::shared_ops::cast_sample::CastSample;

/// Converts image samples from one band format to another.
///
/// `Cast` uses `Op` with `Input = From` and `Output = To`, so the compiler
/// verifies format compatibility when it is chained with other operations. This
/// lets `Cast` participate in the normal operation bridge instead of needing a
/// special `DynOperation` implementation.
///
/// # Examples
/// ```ignore
/// let op = Cast::<U8, F32>::new(3);
/// // Apply `op` to convert a three-band image from U8 to F32 samples.
/// ```
pub struct Cast<From: BandFormat, To: BandFormat> {
    bands: u32,
    _from: PhantomData<From>,
    _to: PhantomData<To>,
}

impl<From: BandFormat, To: BandFormat> Cast<From, To>
where
    From::Sample: CastSample<To::Sample> + Pod,
    To::Sample: Pod,
{
    #[must_use]
    /// Creates a new `Cast`.
    pub const fn new(bands: u32) -> Self {
        Self {
            bands,
            _from: PhantomData,
            _to: PhantomData,
        }
    }

    /// Returns the number of bands this Cast was constructed for.
    ///
    /// Used by `Compilable` to propagate band count into `OperationBridge`.
    #[must_use]
    pub const fn bands(&self) -> u32 {
        self.bands
    }

    #[inline]
    fn process_scalar(input: &[From::Sample], output: &mut [To::Sample]) {
        for (s, d) in input.iter().zip(output.iter_mut()) {
            *d = s.cast_to();
        }
    }

    #[inline]
    fn process_samples(input: &[From::Sample], output: &mut [To::Sample]) {
        if Self::try_process_avx2(input, output) || Self::try_process_neon(input, output) {
            return;
        }

        Self::process_scalar(input, output);
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn try_process_avx2(input: &[From::Sample], output: &mut [To::Sample]) -> bool {
        if !SimdLevel::detect().has_avx2() {
            return false;
        }

        match (From::ID, To::ID) {
            (BandFormatId::U8, BandFormatId::F32) => {
                let Ok(src) = try_cast_slice::<From::Sample, u8>(input) else {
                    return false;
                };
                let Ok(dst) = try_cast_slice_mut::<To::Sample, f32>(output) else {
                    return false;
                };
                // SAFETY: `has_avx2()` confirmed AVX2 support before dispatch, and the
                // bytemuck casts above guarantee the slices have the expected element types.
                unsafe { cast_u8_to_f32_avx2(src, dst) };
                true
            }
            (BandFormatId::F32, BandFormatId::U8) => {
                let Ok(src) = try_cast_slice::<From::Sample, f32>(input) else {
                    return false;
                };
                let Ok(dst) = try_cast_slice_mut::<To::Sample, u8>(output) else {
                    return false;
                };
                // SAFETY: `has_avx2()` confirmed AVX2 support before dispatch, and the
                // bytemuck casts above guarantee the slices have the expected element types.
                unsafe { cast_f32_to_u8_avx2(src, dst) };
                true
            }
            _ => false,
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    #[inline]
    const fn try_process_avx2(_input: &[From::Sample], _output: &mut [To::Sample]) -> bool {
        false
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    fn try_process_neon(input: &[From::Sample], output: &mut [To::Sample]) -> bool {
        if !SimdLevel::detect().has_neon() {
            return false;
        }

        match (From::ID, To::ID) {
            (BandFormatId::U8, BandFormatId::F32) => {
                let Ok(src) = try_cast_slice::<From::Sample, u8>(input) else {
                    return false;
                };
                let Ok(dst) = try_cast_slice_mut::<To::Sample, f32>(output) else {
                    return false;
                };
                // SAFETY: `has_neon()` confirmed NEON support before dispatch, and the
                // bytemuck casts above guarantee the slices have the expected element types.
                unsafe { cast_u8_to_f32_neon(src, dst) };
                true
            }
            (BandFormatId::F32, BandFormatId::U8) => {
                let Ok(src) = try_cast_slice::<From::Sample, f32>(input) else {
                    return false;
                };
                let Ok(dst) = try_cast_slice_mut::<To::Sample, u8>(output) else {
                    return false;
                };
                // SAFETY: `has_neon()` confirmed NEON support before dispatch, and the
                // bytemuck casts above guarantee the slices have the expected element types.
                unsafe { cast_f32_to_u8_neon(src, dst) };
                true
            }
            _ => false,
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    #[inline]
    #[allow(clippy::missing_const_for_fn)]
    // REASON: Uses x86 SIMD intrinsics not available in const context.
    fn try_process_neon(_input: &[From::Sample], _output: &mut [To::Sample]) -> bool {
        false
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
// SAFETY: caller must execute this only on NEON-capable aarch64 and pass equal-length slices so every 8-byte load and two 4-lane stores stay in bounds.
unsafe fn cast_u8_to_f32_neon(input: &[u8], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());

    let mut index = 0usize;
    while index + 8 <= input.len() {
        // SAFETY: `index + 8 <= input.len()` guarantees eight readable bytes, and the
        // destination has eight writable `f32` lanes at the same logical positions.
        // AArch64 NEON permits unaligned loads/stores for these element sizes.
        unsafe {
            let bytes = vld1_u8(input.as_ptr().add(index));
            let widened_u16 = vmovl_u8(bytes);
            let lo_u32 = vmovl_u16(vget_low_u16(widened_u16));
            let hi_u32 = vmovl_u16(vget_high_u16(widened_u16));
            let lo_f32 = vmulq_n_f32(vcvtq_f32_u32(lo_u32), U8_TO_F32_SCALE);
            let hi_f32 = vmulq_n_f32(vcvtq_f32_u32(hi_u32), U8_TO_F32_SCALE);
            vst1q_f32(output.as_mut_ptr().add(index), lo_f32);
            vst1q_f32(output.as_mut_ptr().add(index + 4), hi_f32);
        }
        index += 8;
    }

    for (src, dst) in input[index..].iter().zip(output[index..].iter_mut()) {
        *dst = src.cast_to();
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
// SAFETY: caller must execute this only on NEON-capable aarch64 and pass equal-length slices so every 4-lane load and 4-byte packed store stays in bounds.
unsafe fn cast_f32_to_u8_neon(input: &[f32], output: &mut [u8]) {
    debug_assert_eq!(input.len(), output.len());

    let mut index = 0usize;
    while index + 4 <= input.len() {
        // SAFETY: `index + 4 <= input.len()` guarantees four readable `f32` values and
        // four writable output bytes. The narrowing sequence saturates before extracting
        // the low 32-bit lane, and `copy_nonoverlapping` writes those four packed bytes
        // to the byte-addressed destination without requiring `u32` alignment.
        unsafe {
            let pixels = vld1q_f32(input.as_ptr().add(index));
            let clamped = vmaxq_f32(vminq_f32(pixels, vdupq_n_f32(1.0)), vdupq_n_f32(0.0));
            let scaled = vmulq_n_f32(clamped, F32_TO_U8_SCALE);
            let rounded = vaddq_f32(scaled, vdupq_n_f32(F32_TO_U8_ROUND_BIAS));
            let as_u32 = vcvtq_u32_f32(rounded);
            let narrowed_u16 = vqmovn_u32(as_u32);
            let narrowed_u8 = vqmovn_u16(vcombine_u16(narrowed_u16, vdup_n_u16(0)));
            let packed = vreinterpret_u32_u8(narrowed_u8);
            let packed_bytes = vget_lane_u32::<0>(packed).to_ne_bytes();
            std::ptr::copy_nonoverlapping(packed_bytes.as_ptr(), output.as_mut_ptr().add(index), 4);
        }
        index += 4;
    }

    for (src, dst) in input[index..].iter().zip(output[index..].iter_mut()) {
        *dst = src.cast_to();
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
// SAFETY: caller must dispatch only when AVX2 is available and pass equal-length slices so every 8-byte load and 8-lane store stays in bounds.
// REASON: SIMD intrinsics (_mm_loadl_epi64 / _mm_storel_epi64) handle alignment internally;
// the pointer cast is intentional and safe within unsafe SIMD blocks.
#[allow(clippy::cast_ptr_alignment)]
unsafe fn cast_u8_to_f32_avx2(input: &[u8], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());

    let mut index = 0usize;
    let scale = _mm256_set1_ps(U8_TO_F32_SCALE);

    while index + 8 <= input.len() {
        // SAFETY: `index + 8 <= input.len()` guarantees eight readable input bytes and
        // eight writable `f32` outputs. The AVX2 load/store intrinsics used here accept
        // unaligned pointers.
        unsafe {
            let bytes = _mm_loadl_epi64(input.as_ptr().add(index).cast::<__m128i>());
            let widened = _mm256_cvtepu8_epi32(bytes);
            let floats = _mm256_cvtepi32_ps(widened);
            let scaled = _mm256_mul_ps(floats, scale);
            _mm256_storeu_ps(output.as_mut_ptr().add(index), scaled);
        }
        index += 8;
    }

    for (src, dst) in input[index..].iter().zip(output[index..].iter_mut()) {
        *dst = src.cast_to();
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
// SAFETY: caller must dispatch only when AVX2 is available and pass equal-length slices so every 8-lane load and 8-byte packed store stays in bounds.
// REASON: SIMD intrinsics (_mm_loadl_epi64 / _mm_storel_epi64) handle alignment internally;
// the pointer cast is intentional and safe within unsafe SIMD blocks.
#[allow(clippy::cast_ptr_alignment)]
unsafe fn cast_f32_to_u8_avx2(input: &[f32], output: &mut [u8]) {
    debug_assert_eq!(input.len(), output.len());

    let mut index = 0usize;
    let zero = _mm256_set1_ps(0.0);
    let one = _mm256_set1_ps(1.0);
    let scale = _mm256_set1_ps(F32_TO_U8_SCALE);
    let round_bias = _mm256_set1_ps(F32_TO_U8_ROUND_BIAS);

    while index + 8 <= input.len() {
        // SAFETY: `index + 8 <= input.len()` guarantees eight readable `f32` values and
        // eight writable output bytes. The AVX2/SSE packing sequence saturates each lane
        // before the final 64-bit store, which writes exactly the eight bytes for this chunk.
        unsafe {
            let pixels = _mm256_loadu_ps(input.as_ptr().add(index));
            let clamped = _mm256_max_ps(zero, _mm256_min_ps(one, pixels));
            let rounded = _mm256_add_ps(_mm256_mul_ps(clamped, scale), round_bias);
            let as_i32 = _mm256_cvttps_epi32(rounded);
            let lo = _mm256_castsi256_si128(as_i32);
            let hi = _mm256_extracti128_si256::<1>(as_i32);
            let packed_u16 = _mm_packus_epi32(lo, hi);
            let packed_u8 = _mm_packus_epi16(packed_u16, _mm_setzero_si128());
            _mm_storel_epi64(output.as_mut_ptr().add(index).cast::<__m128i>(), packed_u8);
        }
        index += 8;
    }

    for (src, dst) in input[index..].iter().zip(output[index..].iter_mut()) {
        *dst = src.cast_to();
    }
}

impl<From, To> Op for Cast<From, To>
where
    From: BandFormat,
    To: BandFormat,
    From::Sample: CastSample<To::Sample> + Pod,
    To::Sample: Pod,
{
    type Input = From;
    type Output = To;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<From>, output: &mut TileMut<To>) {
        Self::process_samples(input.data, output.data);
    }
}

/// `Cast` is pixel-local: output(x,y) depends only on input(x,y).
/// No neighbourhood access; identity tile geometry. See `PixelLocalOp` for invariants.
impl<From, To> PixelLocalOp for Cast<From, To>
where
    From: BandFormat,
    To: BandFormat,
    From::Sample: CastSample<To::Sample> + Pod,
    To::Sample: Pod,
{
}
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::op::{DynOperation, OperationBridge};
    use viprs_core::{
        format::{F32, U8, U16},
        image::Region,
    };

    fn cast_via_bridge<From: BandFormat, To: BandFormat>(
        cast: Cast<From, To>,
        input: &[u8],
        output: &mut [u8],
        bands: u32,
    ) where
        From::Sample: CastSample<To::Sample> + Pod,
        To::Sample: Pod,
    {
        // Region width = total samples / bands (number of pixels per row).
        let total_samples = input.len() / std::mem::size_of::<From::Sample>();
        let pixel_count = total_samples / bands as usize;
        let region = Region::new(0, 0, pixel_count as u32, 1);
        let bridge = OperationBridge::new(cast, bands);
        let mut state = bridge.dyn_start();
        // Cast is pixel-local: input_region == output_region.
        bridge.dyn_process_region(state.as_mut(), input, output, region, region);
    }

    fn scalar_u8_to_f32(input: &[u8]) -> Vec<f32> {
        input.iter().map(|sample| sample.cast_to()).collect()
    }

    fn scalar_f32_to_u8(input: &[f32]) -> Vec<u8> {
        input.iter().map(|sample| sample.cast_to()).collect()
    }

    #[test]
    fn cast_u8_to_f32_scales_correctly() {
        let cast = Cast::<U8, F32>::new(1);
        let input = [0u8, 128, 255];
        let mut output = vec![0u8; 3 * std::mem::size_of::<f32>()];
        cast_via_bridge(cast, bytemuck::cast_slice(&input), &mut output, 1);
        let result: &[f32] = bytemuck::cast_slice(&output);
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[1] - 128.0 / 255.0).abs() < 1e-4);
        assert!((result[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cast_f32_to_u8_clamps_and_scales() {
        let cast = Cast::<F32, U8>::new(1);
        let input = [0.0f32, 0.5, 1.0];
        let mut output = vec![0u8; 3];
        cast_via_bridge(cast, bytemuck::cast_slice(&input), &mut output, 1);
        assert_eq!(output[0], 0);
        // 0.5 * 255 = 127.5, rounded = 128
        assert_eq!(output[1], 128);
        assert_eq!(output[2], 255);
    }

    #[test]
    fn cast_preserves_pixel_count() {
        let cast = Cast::<U8, F32>::new(3);
        // 4 pixels × 3 bands × 1 byte = 12 input bytes → 12 × 4 = 48 output bytes
        let input = vec![0u8; 12];
        let mut output = vec![0u8; 48];
        cast_via_bridge(cast, &input, &mut output, 3);
        let result: &[f32] = bytemuck::cast_slice(&output);
        assert_eq!(result.len(), 12);
    }

    #[test]
    fn cast_op_directly_processes_tile() {
        // Verify that Op::process_region works directly, without going through the bridge.
        let cast = Cast::<U8, F32>::new(1);
        let region = Region::new(0, 0, 3, 1);
        let input_data = [0u8, 128, 255];
        let mut output_data = [0.0f32; 3];
        let input = Tile::<U8>::new(region, 1, &input_data);
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
        cast.start();
        cast.process_region(&mut (), &input, &mut output);
        assert!((output_data[0] - 0.0).abs() < 1e-6);
        assert!((output_data[1] - 128.0 / 255.0).abs() < 1e-4);
        assert!((output_data[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cast_sample_dispatch_covers_all_conversion_impls() {
        assert_eq!(<u8 as CastSample<u8>>::cast_to(7), 7);
        assert_eq!(<u8 as CastSample<u16>>::cast_to(1), 257);
        assert!((<u16 as CastSample<f32>>::cast_to(u16::MAX) - 1.0).abs() < 1e-6);
        assert!((<f32 as CastSample<u8>>::cast_to(0.5) as i32 - 128).abs() <= 0);
        assert_eq!(<f32 as CastSample<f32>>::cast_to(0.25), 0.25);
        assert_eq!(<f32 as CastSample<f64>>::cast_to(0.5), 0.5f64);
        assert_eq!(<f64 as CastSample<f32>>::cast_to(0.75), 0.75f32);
    }

    #[test]
    fn cast_metadata_matches_identity_geometry() {
        let cast = Cast::<U8, F32>::new(3);
        let region = Region::new(0, 0, 4, 2);
        assert_eq!(cast.bands(), 3);
        assert_eq!(cast.demand_hint(), viprs_core::image::DemandHint::ThinStrip);
        assert_eq!(cast.required_input_region(&region), region);
    }

    /// Ported from libvips `test_conversion.py::test_cast`.
    ///
    /// libvips test: "casting negative pixels to an unsigned format should clip to zero".
    /// In viprs, the f32 → u8 cast already clamps to [0,1] before scaling:
    /// negative f32 values clip to 0u8.
    #[test]
    fn cast_negative_f32_clips_to_zero_when_cast_to_u8() {
        let cast = Cast::<F32, U8>::new(1);
        let region = Region::new(0, 0, 3, 1);
        let input_data = [-1.0f32, -0.5, -0.001];
        let mut output_data = [255u8; 3];
        let input = Tile::<F32>::new(region, 1, &input_data);
        let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
        cast.start();
        cast.process_region(&mut (), &input, &mut output);
        assert_eq!(output_data[0], 0, "cast(-1.0f32 → u8) must clip to 0");
        assert_eq!(output_data[1], 0, "cast(-0.5f32 → u8) must clip to 0");
        assert_eq!(output_data[2], 0, "cast(-0.001f32 → u8) must clip to 0");
    }

    /// Ported from libvips `test_conversion.py::test_cast`.
    ///
    /// libvips test: "casting very positive pixels to a signed format should clip to max".
    /// f32 values > 1.0 clip to 255u8, and 1.0 maps to 255u8.
    #[test]
    fn cast_out_of_range_f32_clips_to_u8_max() {
        let cast = Cast::<F32, U8>::new(1);
        let region = Region::new(0, 0, 3, 1);
        let input_data = [1.0f32, 1.5, 100.0];
        let mut output_data = [0u8; 3];
        let input = Tile::<F32>::new(region, 1, &input_data);
        let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
        cast.start();
        cast.process_region(&mut (), &input, &mut output);
        assert_eq!(output_data[0], 255, "cast(1.0f32 → u8) must be 255");
        assert_eq!(output_data[1], 255, "cast(1.5f32 → u8) must clip to 255");
        assert_eq!(output_data[2], 255, "cast(100.0f32 → u8) must clip to 255");
    }

    #[test]
    fn cast_u8_to_f32_simd_matches_scalar() {
        let cast = Cast::<U8, F32>::new(1);
        let region = Region::new(0, 0, 17, 1);
        let input_data = [
            0u8, 1, 2, 3, 15, 31, 63, 64, 65, 127, 128, 129, 191, 223, 254, 255, 42,
        ];
        let mut output_data = [0.0f32; 17];
        let input = Tile::<U8>::new(region, 1, &input_data);
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
        cast.start();
        cast.process_region(&mut (), &input, &mut output);

        let expected = scalar_u8_to_f32(&input_data);
        assert_eq!(output_data.len(), expected.len());
        for (actual, expected) in output_data.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn cast_f32_to_u8_saturates_out_of_range() {
        let cast = Cast::<F32, U8>::new(1);
        let region = Region::new(0, 0, 9, 1);
        let input_data = [-5.0f32, -0.25, 0.0, 0.25, 0.5, 1.0, 1.001, 5.0, 100.0];
        let mut output_data = [0u8; 9];
        let input = Tile::<F32>::new(region, 1, &input_data);
        let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
        cast.start();
        cast.process_region(&mut (), &input, &mut output);

        assert_eq!(output_data, [0, 0, 0, 64, 128, 255, 255, 255, 255]);
    }

    #[test]
    fn cast_f32_to_u8_simd_matches_scalar() {
        let cast = Cast::<F32, U8>::new(1);
        let region = Region::new(0, 0, 13, 1);
        let input_data = [
            -1.0f32, 0.0, 0.01, 0.1, 0.2, 0.33, 0.5, 0.66, 0.75, 0.9, 0.99, 1.0, 1.25,
        ];
        let mut output_data = [0u8; 13];
        let input = Tile::<F32>::new(region, 1, &input_data);
        let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
        cast.start();
        cast.process_region(&mut (), &input, &mut output);

        assert_eq!(output_data, scalar_f32_to_u8(&input_data).as_slice());
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn cast_f32_to_u8_neon_supports_unaligned_output() {
        if !SimdLevel::detect().has_neon() {
            return;
        }

        let input = [0.0f32, 0.25, 0.5, 1.0];
        let expected = scalar_f32_to_u8(&input);
        let mut storage = [0xAAu8; 8];
        let offset = (0..=4)
            .find(|candidate| storage.as_ptr().wrapping_add(*candidate).align_offset(4) != 0)
            .unwrap_or(1);
        let output = &mut storage[offset..offset + input.len()];

        // SAFETY: the test guards NEON availability before calling the target-feature
        // function, and `output` has exactly one byte per `input` lane.
        unsafe { cast_f32_to_u8_neon(&input, output) };

        assert_ne!(output.as_ptr().align_offset(4), 0);
        assert_eq!(output, expected.as_slice());
    }

    /// Ported from libvips `test_conversion.py::test_cast`.
    ///
    /// libvips test: u8 0 → u16 0, u8 255 → u16 65535.
    /// In libvips cast, u8 is scaled to u16 by multiplying by 257 (= 65535/255).
    #[test]
    fn cast_u8_to_u16_scales_by_257() {
        let cast = Cast::<U8, U16>::new(1);
        let region = Region::new(0, 0, 3, 1);
        let input_data = [0u8, 1, 255];
        let mut output_data = [0u16; 3];
        let input = Tile::<U8>::new(region, 1, &input_data);
        let mut output = TileMut::<U16>::new(region, 1, &mut output_data);
        cast.start();
        cast.process_region(&mut (), &input, &mut output);
        assert_eq!(output_data[0], 0, "u8(0) → u16 must be 0");
        assert_eq!(output_data[1], 257, "u8(1) → u16 must be 257");
        assert_eq!(output_data[2], 65535, "u8(255) → u16 must be 65535");
    }

    proptest! {
        #[test]
        fn cast_u8_to_u8_identity(samples in prop::collection::vec(any::<u8>(), 0..128)) {
            let cast = Cast::<U8, U8>::new(1);
            let region = Region::new(0, 0, samples.len() as u32, 1);
            let mut output_data = vec![0u8; samples.len()];
            let input = Tile::<U8>::new(region, 1, &samples);
            let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
            cast.start();
            cast.process_region(&mut (), &input, &mut output);
            prop_assert_eq!(output_data, samples);
        }
    }
}

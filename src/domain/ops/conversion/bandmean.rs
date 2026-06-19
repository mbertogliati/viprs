use std::marker::PhantomData;

use bytemuck::{try_cast_slice, try_cast_slice_mut};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    uint8x16_t, uint8x16x3_t, uint8x16x4_t, uint16x8_t, vaddl_u8, vaddq_u16, vaddw_u8, vcombine_u8,
    vcombine_u16, vdupq_n_u16, vget_high_u8, vget_high_u16, vget_low_u8, vget_low_u16, vld3q_u8,
    vld4q_u8, vmull_n_u16, vqmovn_u16, vshrn_n_u32, vshrq_n_u16, vshrq_n_u32, vst1q_u8,
};

use crate::{
    domain::op::{Op, OperationBridge, PixelLocalOp},
    domain::{
        format::{BandFormat, BandFormatId},
        image::{DemandHint, Region, Tile, TileMut},
    },
};

/// Average all bands of each pixel into a single-band image.
pub struct BandMean<F: BandFormat> {
    input_bands: usize,
    _f: PhantomData<F>,
}

impl<F: BandFormat> BandMean<F> {
    /// Construct a `BandMean` that reduces `input_bands` to one averaged band.
    #[must_use]
    pub fn new(input_bands: usize) -> Self {
        debug_assert!(input_bands > 0, "BandMean: input_bands must be at least 1");
        Self {
            input_bands,
            _f: PhantomData,
        }
    }
}

impl<F> BandMean<F>
where
    F: BandFormat,
    F::Sample: BandMeanSample,
{
    /// Build an `OperationBridge` configured with the correct input/output band counts.
    ///
    /// Input is `self.input_bands` (N-band), output is always 1-band.
    #[must_use]
    pub fn into_bridge(self) -> OperationBridge<Self> {
        let input_bands = self.input_bands as u32;
        OperationBridge::with_dynamic_bands_pixel_local(self, input_bands, 1)
    }

    #[inline]
    fn process_scalar(&self, input: &[F::Sample], output: &mut [F::Sample]) {
        for (src, dst) in input.chunks_exact(self.input_bands).zip(output.iter_mut()) {
            let mut sum = <F::Sample as BandMeanSample>::zero();
            for &sample in src {
                <F::Sample as BandMeanSample>::accumulate(&mut sum, sample);
            }
            *dst = <F::Sample as BandMeanSample>::finish(sum, self.input_bands);
        }
    }

    #[inline]
    fn process_u8(&self, input: &[F::Sample], output: &mut [F::Sample]) -> bool {
        let Ok(src) = try_cast_slice::<F::Sample, u8>(input) else {
            return false;
        };
        let Ok(dst) = try_cast_slice_mut::<F::Sample, u8>(output) else {
            return false;
        };

        process_u8_bandmean(src, dst, self.input_bands);
        true
    }

    #[inline]
    fn process_samples(&self, input: &[F::Sample], output: &mut [F::Sample]) {
        if self.input_bands == 1 {
            output.copy_from_slice(input);
            return;
        }

        if F::ID == BandFormatId::U8 && self.process_u8(input, output) {
            return;
        }

        self.process_scalar(input, output);
    }
}

impl<F> Op for BandMean<F>
where
    F: BandFormat,
    F::Sample: BandMeanSample,
{
    type Input = F;
    type Output = F;
    type State = ();

    const OUTPUT_BANDS: Option<usize> = Some(1);

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        debug_assert_eq!(
            input.bands as usize, self.input_bands,
            "BandMean input tile band count must match constructor input_bands"
        );
        debug_assert_eq!(
            output.bands, 1,
            "BandMean output tile must have exactly 1 band"
        );

        self.process_samples(input.data, output.data);
    }
}

impl<F> PixelLocalOp for BandMean<F>
where
    F: BandFormat,
    F::Sample: BandMeanSample,
{
}

/// Defines the contract for band mean sample.
pub trait BandMeanSample: Copy + 'static {
    /// Associated type for accumulator.
    type Accumulator: Copy;

    /// Returns or performs zero.
    fn zero() -> Self::Accumulator;
    /// Returns or performs accumulate.
    fn accumulate(acc: &mut Self::Accumulator, sample: Self);
    /// Returns or performs finish.
    fn finish(acc: Self::Accumulator, bands: usize) -> Self;
}

impl BandMeanSample for u8 {
    type Accumulator = u128;

    #[inline(always)]
    fn zero() -> Self::Accumulator {
        0
    }

    #[inline(always)]
    fn accumulate(acc: &mut Self::Accumulator, sample: Self) {
        *acc += u128::from(sample);
    }

    #[inline(always)]
    fn finish(acc: Self::Accumulator, bands: usize) -> Self {
        ((acc + (bands / 2) as u128) / bands as u128) as Self
    }
}

impl BandMeanSample for u16 {
    type Accumulator = u128;

    #[inline(always)]
    fn zero() -> Self::Accumulator {
        0
    }

    #[inline(always)]
    fn accumulate(acc: &mut Self::Accumulator, sample: Self) {
        *acc += u128::from(sample);
    }

    #[inline(always)]
    fn finish(acc: Self::Accumulator, bands: usize) -> Self {
        ((acc + (bands / 2) as u128) / bands as u128) as Self
    }
}

impl BandMeanSample for u32 {
    type Accumulator = u128;

    #[inline(always)]
    fn zero() -> Self::Accumulator {
        0
    }

    #[inline(always)]
    fn accumulate(acc: &mut Self::Accumulator, sample: Self) {
        *acc += u128::from(sample);
    }

    #[inline(always)]
    fn finish(acc: Self::Accumulator, bands: usize) -> Self {
        ((acc + (bands / 2) as u128) / bands as u128) as Self
    }
}

impl BandMeanSample for i16 {
    type Accumulator = i128;

    #[inline(always)]
    fn zero() -> Self::Accumulator {
        0
    }

    #[inline(always)]
    fn accumulate(acc: &mut Self::Accumulator, sample: Self) {
        *acc += i128::from(sample);
    }

    #[inline(always)]
    fn finish(acc: Self::Accumulator, bands: usize) -> Self {
        if acc > 0 {
            ((acc + (bands / 2) as i128) / bands as i128) as Self
        } else {
            ((acc - (bands / 2) as i128) / bands as i128) as Self
        }
    }
}

impl BandMeanSample for i32 {
    type Accumulator = i128;

    #[inline(always)]
    fn zero() -> Self::Accumulator {
        0
    }

    #[inline(always)]
    fn accumulate(acc: &mut Self::Accumulator, sample: Self) {
        *acc += i128::from(sample);
    }

    #[inline(always)]
    fn finish(acc: Self::Accumulator, bands: usize) -> Self {
        if acc > 0 {
            ((acc + (bands / 2) as i128) / bands as i128) as Self
        } else {
            ((acc - (bands / 2) as i128) / bands as i128) as Self
        }
    }
}

impl BandMeanSample for f32 {
    type Accumulator = Self;

    #[inline(always)]
    fn zero() -> Self::Accumulator {
        0.0
    }

    #[inline(always)]
    fn accumulate(acc: &mut Self::Accumulator, sample: Self) {
        *acc += sample;
    }

    #[inline(always)]
    fn finish(acc: Self::Accumulator, bands: usize) -> Self {
        acc / bands as Self
    }
}

impl BandMeanSample for f64 {
    type Accumulator = Self;

    #[inline(always)]
    fn zero() -> Self::Accumulator {
        0.0
    }

    #[inline(always)]
    fn accumulate(acc: &mut Self::Accumulator, sample: Self) {
        *acc += sample;
    }

    #[inline(always)]
    fn finish(acc: Self::Accumulator, bands: usize) -> Self {
        acc / bands as Self
    }
}

#[inline]
fn process_u8_bandmean(input: &[u8], output: &mut [u8], bands: usize) {
    debug_assert_eq!(input.len(), output.len() * bands);

    #[cfg(target_arch = "aarch64")]
    if matches!(bands, 3 | 4) {
        // SAFETY: this branch only exists on aarch64 where NEON is mandatory, and the helper
        // validates chunk boundaries before every load/store while preserving slice lengths.
        unsafe { bandmean_u8_neon(input, output, bands) };
        return;
    }

    match bands {
        1 => output.copy_from_slice(input),
        3 => bandmean_u8_rgb_scalar(input, output),
        4 => bandmean_u8_rgba_scalar(input, output),
        _ => bandmean_u8_scalar(input, output, bands),
    }
}

#[inline]
fn bandmean_u8_scalar(input: &[u8], output: &mut [u8], bands: usize) {
    for (src, dst) in input.chunks_exact(bands).zip(output.iter_mut()) {
        let mut sum = 0u32;
        for &sample in src {
            sum += u32::from(sample);
        }
        *dst = ((sum + (bands / 2) as u32) / bands as u32) as u8;
    }
}

#[inline]
fn bandmean_u8_rgb_scalar(input: &[u8], output: &mut [u8]) {
    for (src, dst) in input.chunks_exact(3).zip(output.iter_mut()) {
        let sum = u16::from(src[0]) + u16::from(src[1]) + u16::from(src[2]);
        *dst = ((sum + 1) / 3) as u8;
    }
}

#[inline]
fn bandmean_u8_rgba_scalar(input: &[u8], output: &mut [u8]) {
    for (src, dst) in input.chunks_exact(4).zip(output.iter_mut()) {
        let sum = u16::from(src[0]) + u16::from(src[1]) + u16::from(src[2]) + u16::from(src[3]);
        *dst = ((sum + 2) >> 2) as u8;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
// SAFETY: caller must execute this only on NEON-capable aarch64 and provide matching `input`/`output` lengths for the selected band count so each helper sees whole pixels.
unsafe fn bandmean_u8_neon(input: &[u8], output: &mut [u8], bands: usize) {
    match bands {
        3 => {
            // SAFETY: the caller guarantees NEON availability and matching RGB slice lengths.
            unsafe { bandmean_u8_rgb_neon(input, output) }
        }
        4 => {
            // SAFETY: the caller guarantees NEON availability and matching RGBA slice lengths.
            unsafe { bandmean_u8_rgba_neon(input, output) }
        }
        _ => bandmean_u8_scalar(input, output, bands),
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
// SAFETY: caller must execute this only on NEON-capable aarch64 and provide RGB-packed input with `input.len() == output.len() * 3` so each 16-pixel load and store stays in bounds.
unsafe fn bandmean_u8_rgb_neon(input: &[u8], output: &mut [u8]) {
    let simd_pixels = output.len() / 16 * 16;
    let reciprocal = 0xAAABu16;

    for chunk in 0..(simd_pixels / 16) {
        let input_offset = chunk * 48;
        let output_offset = chunk * 16;

        // SAFETY: `input_offset + 48 <= input.len()` for every SIMD chunk and `vld3q_u8`
        // accepts unaligned pointers to 16 interleaved RGB pixels.
        let rgb: uint8x16x3_t = unsafe { vld3q_u8(input.as_ptr().add(input_offset)) };
        // SAFETY: this helper shares the same NEON precondition and only operates on local vectors.
        let averages = unsafe { divide_by_3_rounded_u8x16(sum_rgb_u8x16(rgb), reciprocal) };

        // SAFETY: `output_offset + 16 <= output.len()` for every SIMD chunk and NEON stores
        // permit unaligned destinations.
        unsafe { vst1q_u8(output.as_mut_ptr().add(output_offset), averages) };
    }

    let input_tail = simd_pixels * 3;
    bandmean_u8_rgb_scalar(&input[input_tail..], &mut output[simd_pixels..]);
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
// SAFETY: caller must execute this only on NEON-capable aarch64 and provide RGBA-packed input with `input.len() == output.len() * 4` so each 16-pixel load and store stays in bounds.
unsafe fn bandmean_u8_rgba_neon(input: &[u8], output: &mut [u8]) {
    let simd_pixels = output.len() / 16 * 16;
    let round_bias = vdupq_n_u16(2);

    for chunk in 0..(simd_pixels / 16) {
        let input_offset = chunk * 64;
        let output_offset = chunk * 16;

        // SAFETY: `input_offset + 64 <= input.len()` for every SIMD chunk and `vld4q_u8`
        // accepts unaligned pointers to 16 interleaved RGBA pixels.
        let rgba: uint8x16x4_t = unsafe { vld4q_u8(input.as_ptr().add(input_offset)) };
        // SAFETY: this helper shares the same NEON precondition and only operates on local vectors.
        let sums = unsafe { sum_rgba_u8x16(rgba) };
        let lo = vshrq_n_u16::<2>(vaddq_u16(sums.0, round_bias));
        let hi = vshrq_n_u16::<2>(vaddq_u16(sums.1, round_bias));
        let averages = vcombine_u8(vqmovn_u16(lo), vqmovn_u16(hi));

        // SAFETY: `output_offset + 16 <= output.len()` for every SIMD chunk and NEON stores
        // permit unaligned destinations.
        unsafe { vst1q_u8(output.as_mut_ptr().add(output_offset), averages) };
    }

    let input_tail = simd_pixels * 4;
    bandmean_u8_rgba_scalar(&input[input_tail..], &mut output[simd_pixels..]);
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
// SAFETY: caller must execute this only on NEON-capable aarch64; the helper only transforms register values and does not touch memory.
unsafe fn sum_rgb_u8x16(rgb: uint8x16x3_t) -> (uint16x8_t, uint16x8_t) {
    let lo = vaddw_u8(
        vaddl_u8(vget_low_u8(rgb.0), vget_low_u8(rgb.1)),
        vget_low_u8(rgb.2),
    );
    let hi = vaddw_u8(
        vaddl_u8(vget_high_u8(rgb.0), vget_high_u8(rgb.1)),
        vget_high_u8(rgb.2),
    );
    (lo, hi)
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
// SAFETY: caller must execute this only on NEON-capable aarch64; the helper only transforms register values and does not touch memory.
unsafe fn sum_rgba_u8x16(rgba: uint8x16x4_t) -> (uint16x8_t, uint16x8_t) {
    let lo = vaddw_u8(
        vaddw_u8(
            vaddl_u8(vget_low_u8(rgba.0), vget_low_u8(rgba.1)),
            vget_low_u8(rgba.2),
        ),
        vget_low_u8(rgba.3),
    );
    let hi = vaddw_u8(
        vaddw_u8(
            vaddl_u8(vget_high_u8(rgba.0), vget_high_u8(rgba.1)),
            vget_high_u8(rgba.2),
        ),
        vget_high_u8(rgba.3),
    );
    (lo, hi)
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
// SAFETY: caller must execute this only on NEON-capable aarch64; the helper only performs lane-wise arithmetic on the provided vectors.
unsafe fn divide_by_3_rounded_u8x16(sums: (uint16x8_t, uint16x8_t), reciprocal: u16) -> uint8x16_t {
    let round_bias = vdupq_n_u16(1);
    // SAFETY: the caller guarantees NEON availability for the full duration of this helper.
    let lo = unsafe { divide_by_3_rounded_u16x8(vaddq_u16(sums.0, round_bias), reciprocal) };
    // SAFETY: the caller guarantees NEON availability for the full duration of this helper.
    let hi = unsafe { divide_by_3_rounded_u16x8(vaddq_u16(sums.1, round_bias), reciprocal) };
    vcombine_u8(vqmovn_u16(lo), vqmovn_u16(hi))
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
// SAFETY: caller must execute this only on NEON-capable aarch64; the helper only performs lane-wise arithmetic on the provided vectors.
unsafe fn divide_by_3_rounded_u16x8(values: uint16x8_t, reciprocal: u16) -> uint16x8_t {
    vcombine_u16(
        vshrn_n_u32::<16>(vshrq_n_u32::<1>(vmull_n_u16(
            vget_low_u16(values),
            reciprocal,
        ))),
        vshrn_n_u32::<16>(vshrq_n_u32::<1>(vmull_n_u16(
            vget_high_u16(values),
            reciprocal,
        ))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        format::{F64, I16, I32, U8, U16, U32},
        image::{Region, Tile, TileMut},
        op::DynOperation,
    };
    use proptest::prelude::*;

    fn make_region(w: u32, h: u32) -> Region {
        Region::new(0, 0, w, h)
    }

    fn run_bandmean_u8(
        input_bands: usize,
        input_data: &[u8],
        output_data: &mut [u8],
        pixels: usize,
    ) {
        let region = make_region(pixels as u32, 1);
        let op = BandMean::<U8>::new(input_bands);
        let input = Tile::<U8>::new(region, input_bands as u32, input_data);
        let mut output = TileMut::<U8>::new(region, 1, output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
    }

    fn run_bandmean_i16(
        input_bands: usize,
        input_data: &[i16],
        output_data: &mut [i16],
        pixels: usize,
    ) {
        let region = make_region(pixels as u32, 1);
        let op = BandMean::<I16>::new(input_bands);
        let input = Tile::<I16>::new(region, input_bands as u32, input_data);
        let mut output = TileMut::<I16>::new(region, 1, output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
    }

    fn expected_u8_bandmean(input: &[u8], bands: usize) -> Vec<u8> {
        input
            .chunks_exact(bands)
            .map(|samples| {
                let sum: u32 = samples.iter().map(|&sample| u32::from(sample)).sum();
                ((sum + (bands / 2) as u32) / bands as u32) as u8
            })
            .collect()
    }

    #[test]
    fn into_bridge_reports_one_band() {
        let bridge = BandMean::<U8>::new(4).into_bridge();
        assert_eq!(bridge.bands(), 1);
    }

    #[test]
    fn single_band_is_identity() {
        let input = [7u8, 13, 42, 255];
        let mut output = [0u8; 4];
        run_bandmean_u8(1, &input, &mut output, 4);
        assert_eq!(output, input);
    }

    #[test]
    fn signed_integer_mean_rounds_away_from_zero() {
        let input = [1i16, 2, -1, -2];
        let mut output = [0i16; 2];
        run_bandmean_i16(2, &input, &mut output, 2);
        assert_eq!(output, [2, -2]);
    }

    #[test]
    fn rgb_u8_matches_reference_across_simd_tail() {
        let pixels = 17usize;
        let input: Vec<u8> = (0..pixels * 3).map(|value| (value % 251) as u8).collect();
        let mut output = vec![0u8; pixels];
        run_bandmean_u8(3, &input, &mut output, pixels);
        assert_eq!(output, expected_u8_bandmean(&input, 3));
    }

    #[test]
    fn rgba_u8_matches_reference_across_simd_tail() {
        let pixels = 19usize;
        let input: Vec<u8> = (0..pixels * 4)
            .map(|value| ((value * 17 + 29) % 256) as u8)
            .collect();
        let mut output = vec![0u8; pixels];
        run_bandmean_u8(4, &input, &mut output, pixels);
        assert_eq!(output, expected_u8_bandmean(&input, 4));
    }

    #[test]
    fn bandmean_sample_implementations_cover_all_formats() {
        let mut acc_u16 = <u16 as BandMeanSample>::zero();
        <u16 as BandMeanSample>::accumulate(&mut acc_u16, 1000);
        <u16 as BandMeanSample>::accumulate(&mut acc_u16, 1001);
        assert_eq!(<u16 as BandMeanSample>::finish(acc_u16, 2), 1001);

        let mut acc_u32 = <u32 as BandMeanSample>::zero();
        <u32 as BandMeanSample>::accumulate(&mut acc_u32, 8);
        <u32 as BandMeanSample>::accumulate(&mut acc_u32, 9);
        assert_eq!(<u32 as BandMeanSample>::finish(acc_u32, 2), 9);

        let mut acc_i32 = <i32 as BandMeanSample>::zero();
        <i32 as BandMeanSample>::accumulate(&mut acc_i32, -3);
        <i32 as BandMeanSample>::accumulate(&mut acc_i32, -4);
        assert_eq!(<i32 as BandMeanSample>::finish(acc_i32, 2), -4);

        let mut acc_f32 = <f32 as BandMeanSample>::zero();
        <f32 as BandMeanSample>::accumulate(&mut acc_f32, 1.5);
        <f32 as BandMeanSample>::accumulate(&mut acc_f32, 2.5);
        assert!((<f32 as BandMeanSample>::finish(acc_f32, 2) - 2.0).abs() < f32::EPSILON);

        let mut acc_f64 = <f64 as BandMeanSample>::zero();
        <f64 as BandMeanSample>::accumulate(&mut acc_f64, 1.5);
        <f64 as BandMeanSample>::accumulate(&mut acc_f64, 3.5);
        assert!((<f64 as BandMeanSample>::finish(acc_f64, 2) - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn bandmean_metadata_matches_identity_geometry() {
        let op = BandMean::<U8>::new(3);
        let region = make_region(4, 2);
        assert_eq!(
            op.demand_hint(),
            crate::domain::image::DemandHint::ThinStrip
        );
        assert_eq!(op.required_input_region(&region), region);
    }

    proptest! {
        #[test]
        fn identical_bands_reduce_to_same_values(
            pixels in proptest::collection::vec(0u8..=255u8, 1..=64),
            bands in 1usize..=8,
        ) {
            let mut input = Vec::with_capacity(pixels.len() * bands);
            for &pixel in &pixels {
                for _ in 0..bands {
                    input.push(pixel);
                }
            }

            let mut output = vec![0u8; pixels.len()];
            run_bandmean_u8(bands, &input, &mut output, pixels.len());
            prop_assert_eq!(output, pixels);
        }
    }
}

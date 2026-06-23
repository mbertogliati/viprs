#![allow(missing_docs)]
// REASON: these bridge helpers are public only for cross-crate workspace wiring, not end-user API.

//! Flatten an image with alpha channel onto a solid background colour.
//!
//! Alpha-composites the source image over `background`, producing an output
//! image without an alpha channel. Useful as the final step before encoding
//! to a format that does not support transparency (JPEG, etc.).
//!
//! # Band-count change
//!
//! `Flatten` reduces the band count by 1 only for images with an alpha band
//! (`2` or `4` bands in the current pipeline conventions). Images without alpha
//! pass through unchanged, matching libvips `vips_flatten()`.
//! `OperationBridge` uses a single `bands` field for both input and output, so
//! callers must construct the bridge manually with the correct output band count
//! and pass `input_bands` to `Flatten::new`.

use std::marker::PhantomData;

use viprs_core::{
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    op::{DynOperation, Op},
};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    float32x4_t, uint8x8_t, uint8x16_t, uint8x16x3_t, uint8x16x4_t, uint16x8_t, uint32x4_t,
    vcombine_u8, vcombine_u16, vcvtq_f32_u32, vcvtq_u32_f32, vdupq_n_f32, vdupq_n_u8, vget_high_u8,
    vget_high_u16, vget_low_u8, vget_low_u16, vld4q_u8, vmlal_n_u16, vmovl_u8, vmull_u16,
    vmulq_f32, vqmovn_u16, vqmovn_u32, vst3q_u8, vsubq_u8,
};

/// Alpha-composite an RGBA image over a solid background, producing an RGB output.
///
/// For each pixel:
/// ```text
/// alpha_f = pixel[last_band] / max_value      // normalised to [0, 1]
/// out[b]  = pixel[b] * alpha_f + background[b] * (1 - alpha_f)
/// ```
///
/// - **U8**: `max_value` = 255; output uses libvips-style integer division.
/// - **F32**: `max_value` = 1.0; alpha already in [0, 1].
///
/// `background` must have exactly `input_bands - 1` samples (one per colour band,
/// no alpha). If `background.len() != input_bands - 1` the struct will assert in
/// debug mode; in release mode the extra/missing bands are ignored/zeroed.
pub struct Flatten<F: BandFormat> {
    /// Number of bands in the **input** image (including alpha).
    input_bands: u32,
    /// Background colour — one sample per colour band (no alpha sample).
    background: Vec<F::Sample>,
    _fmt: PhantomData<F>,
}

#[must_use]
pub const fn flatten_has_alpha(input_bands: u32) -> bool {
    matches!(input_bands, 2 | 4)
}

#[must_use]
pub const fn flatten_output_bands(input_bands: u32) -> u32 {
    if flatten_has_alpha(input_bands) {
        input_bands - 1
    } else {
        input_bands
    }
}

impl Flatten<viprs_core::format::U8> {
    /// Create a new `Flatten<U8>`.
    ///
    /// - `input_bands`: number of bands in the source image.
    /// - `background`: colour of the canvas behind the source image
    ///   (`input_bands - 1` samples when alpha is present, otherwise one sample
    ///   per input band for the no-op passthrough case).
    #[must_use]
    pub fn new(input_bands: u32, background: Vec<u8>) -> Self {
        debug_assert_eq!(
            background.len(),
            flatten_output_bands(input_bands) as usize,
            "background must match the flatten output band count"
        );
        Self {
            input_bands,
            background,
            _fmt: PhantomData,
        }
    }
}

impl Flatten<viprs_core::format::F32> {
    /// Create a new `Flatten<F32>`.
    ///
    /// - `input_bands`: number of bands in the source image.
    /// - `background`: colour of the canvas behind the source image
    ///   (`input_bands - 1` samples when alpha is present, otherwise one sample
    ///   per input band for the no-op passthrough case, in [0.0, 1.0]).
    #[must_use]
    pub fn new(input_bands: u32, background: Vec<f32>) -> Self {
        debug_assert_eq!(
            background.len(),
            flatten_output_bands(input_bands) as usize,
            "background must match the flatten output band count"
        );
        Self {
            input_bands,
            background,
            _fmt: PhantomData,
        }
    }
}

impl Op for Flatten<viprs_core::format::U8> {
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

    /// Process a tile of pixels, flattening alpha when present and otherwise
    /// acting as a passthrough.
    ///
    /// # Band-count mismatch
    ///
    /// `input.bands == self.input_bands` and `output.bands == flatten_output_bands(self.input_bands)`.
    #[inline]
    fn process_region(
        &self,
        _state: &mut (),
        input: &Tile<viprs_core::format::U8>,
        output: &mut TileMut<viprs_core::format::U8>,
    ) {
        let in_bands = self.input_bands as usize;
        if !flatten_has_alpha(self.input_bands) {
            output.data.copy_from_slice(input.data);
            return;
        }

        if in_bands == 4
            && output.bands == 3
            && let [bg_r, bg_g, bg_b, ..] = self.background.as_slice()
        {
            flatten_rgba_u8(input.data, [*bg_r, *bg_g, *bg_b], output.data);
            return;
        }

        flatten_u8_scalar(
            input.data,
            in_bands,
            self.background.as_slice(),
            output.data,
        );
    }
}

#[inline]
fn flatten_u8_scalar(input: &[u8], input_bands: usize, background: &[u8], output: &mut [u8]) {
    let output_bands = input_bands - 1;
    let alpha_idx = input_bands - 1;

    for (in_pixel, out_pixel) in input
        .chunks_exact(input_bands)
        .zip(output.chunks_exact_mut(output_bands))
    {
        let alpha = u32::from(in_pixel[alpha_idx]);
        let inverse_alpha = u32::from(u8::MAX) - alpha;

        for (band, out_sample) in out_pixel.iter_mut().enumerate() {
            let src = u32::from(in_pixel[band]);
            let bg = u32::from(background.get(band).copied().unwrap_or_default());
            let blended = (src * alpha) + (bg * inverse_alpha);
            *out_sample = (blended / u32::from(u8::MAX)) as u8;
        }
    }
}

#[inline]
fn flatten_rgba_u8_scalar(input: &[u8], background: [u8; 3], output: &mut [u8]) {
    for (in_pixel, out_pixel) in input.chunks_exact(4).zip(output.chunks_exact_mut(3)) {
        let alpha = u32::from(in_pixel[3]);
        let inverse_alpha = u32::from(u8::MAX) - alpha;

        out_pixel[0] = (((u32::from(in_pixel[0]) * alpha)
            + (u32::from(background[0]) * inverse_alpha))
            / u32::from(u8::MAX)) as u8;
        out_pixel[1] = (((u32::from(in_pixel[1]) * alpha)
            + (u32::from(background[1]) * inverse_alpha))
            / u32::from(u8::MAX)) as u8;
        out_pixel[2] = (((u32::from(in_pixel[2]) * alpha)
            + (u32::from(background[2]) * inverse_alpha))
            / u32::from(u8::MAX)) as u8;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn flatten_rgba_u8(input: &[u8], background: [u8; 3], output: &mut [u8]) {
    // SAFETY: aarch64 guarantees NEON support, the helper processes full 16-pixel chunks,
    // and any remainder is delegated to the scalar fallback.
    unsafe {
        flatten_rgba_u8_neon(input, background, output);
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn flatten_rgba_u8(input: &[u8], background: [u8; 3], output: &mut [u8]) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime dispatch guarantees AVX2 support and the helper only touches
        // full 8-pixel chunks plus the scalar tail.
        unsafe {
            flatten_rgba_u8_avx2(input, background, output);
        }
        return;
    }

    flatten_rgba_u8_scalar(input, background, output);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
// SAFETY: caller must execute this only on NEON-capable aarch64 and provide RGBA-packed input plus RGB output with matching pixel counts so each 16-pixel interleaved load/store stays in bounds.
unsafe fn flatten_rgba_u8_neon(input: &[u8], background: [u8; 3], output: &mut [u8]) {
    let pixel_count = input.len() / 4;
    let simd_pixels = pixel_count / 16 * 16;
    let inv255 = vdupq_n_f32(1.0 / 255.0);

    for chunk in 0..(simd_pixels / 16) {
        let input_offset = chunk * 64;
        let output_offset = chunk * 48;

        // SAFETY: `input_offset + 64 <= input.len()`, the pointer is valid for 16 interleaved
        // RGBA pixels, and `vld4q_u8` accepts unaligned pointers.
        let rgba: uint8x16x4_t = unsafe { vld4q_u8(input.as_ptr().add(input_offset)) };
        let rgb = uint8x16x3_t(
            // SAFETY: `flatten_rgba_channel_u8` shares this function's NEON precondition.
            unsafe { flatten_rgba_channel_u8(rgba.0, rgba.3, background[0], inv255) },
            // SAFETY: `flatten_rgba_channel_u8` shares this function's NEON precondition.
            unsafe { flatten_rgba_channel_u8(rgba.1, rgba.3, background[1], inv255) },
            // SAFETY: `flatten_rgba_channel_u8` shares this function's NEON precondition.
            unsafe { flatten_rgba_channel_u8(rgba.2, rgba.3, background[2], inv255) },
        );

        // SAFETY: `output_offset + 48 <= output.len()`, the pointer is valid for 16 interleaved
        // RGB pixels, and `vst3q_u8` accepts unaligned pointers.
        unsafe { vst3q_u8(output.as_mut_ptr().add(output_offset), rgb) };
    }

    let input_tail = simd_pixels * 4;
    let output_tail = simd_pixels * 3;
    flatten_rgba_u8_scalar(&input[input_tail..], background, &mut output[output_tail..]);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: caller must execute this only when AVX2 is available and provide RGBA-packed input
// plus RGB output with matching pixel counts so each 8-pixel chunk and scalar tail stay in bounds.
// REASON: SIMD intrinsics operate on unaligned memory via explicit load/store intrinsics; the
// pointer casts are intentional and remain within the chunk-local stack arrays and output slices.
#[allow(clippy::cast_ptr_alignment)]
unsafe fn flatten_rgba_u8_avx2(input: &[u8], background: [u8; 3], output: &mut [u8]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        __m128i, __m256i, _mm_loadl_epi64, _mm256_add_epi32, _mm256_cvtepu8_epi32,
        _mm256_mullo_epi32, _mm256_set1_epi32, _mm256_srli_epi32, _mm256_storeu_si256,
        _mm256_sub_epi32,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        __m128i, __m256i, _mm_loadl_epi64, _mm256_add_epi32, _mm256_cvtepu8_epi32,
        _mm256_mullo_epi32, _mm256_set1_epi32, _mm256_srli_epi32, _mm256_storeu_si256,
        _mm256_sub_epi32,
    };

    let pixel_count = input.len() / 4;
    let simd_pixels = (pixel_count / 8) * 8;
    let max255 = _mm256_set1_epi32(i32::from(u8::MAX));
    let one = _mm256_set1_epi32(1);
    let bg_r = _mm256_set1_epi32(i32::from(background[0]));
    let bg_g = _mm256_set1_epi32(i32::from(background[1]));
    let bg_b = _mm256_set1_epi32(i32::from(background[2]));

    for chunk in 0..(simd_pixels / 8) {
        let input_offset = chunk * 32;
        let output_offset = chunk * 24;
        let mut r_bytes = [0u8; 8];
        let mut g_bytes = [0u8; 8];
        let mut b_bytes = [0u8; 8];
        let mut a_bytes = [0u8; 8];

        for lane in 0..8 {
            let base = input_offset + (lane * 4);
            r_bytes[lane] = input[base];
            g_bytes[lane] = input[base + 1];
            b_bytes[lane] = input[base + 2];
            a_bytes[lane] = input[base + 3];
        }

        // SAFETY: each local byte array contains exactly 8 lanes, `_mm_loadl_epi64` reads those
        // 8 bytes, and the AVX2 widening/multiply/add/store operations stay within stack buffers.
        unsafe {
            let r = _mm256_cvtepu8_epi32(_mm_loadl_epi64(r_bytes.as_ptr().cast::<__m128i>()));
            let g = _mm256_cvtepu8_epi32(_mm_loadl_epi64(g_bytes.as_ptr().cast::<__m128i>()));
            let b = _mm256_cvtepu8_epi32(_mm_loadl_epi64(b_bytes.as_ptr().cast::<__m128i>()));
            let alpha = _mm256_cvtepu8_epi32(_mm_loadl_epi64(a_bytes.as_ptr().cast::<__m128i>()));
            let inverse_alpha = _mm256_sub_epi32(max255, alpha);

            let r_blended = _mm256_add_epi32(
                _mm256_mullo_epi32(r, alpha),
                _mm256_mullo_epi32(bg_r, inverse_alpha),
            );
            let g_blended = _mm256_add_epi32(
                _mm256_mullo_epi32(g, alpha),
                _mm256_mullo_epi32(bg_g, inverse_alpha),
            );
            let b_blended = _mm256_add_epi32(
                _mm256_mullo_epi32(b, alpha),
                _mm256_mullo_epi32(bg_b, inverse_alpha),
            );

            let r_div = _mm256_srli_epi32(
                _mm256_add_epi32(
                    _mm256_add_epi32(r_blended, _mm256_srli_epi32(r_blended, 8)),
                    one,
                ),
                8,
            );
            let g_div = _mm256_srli_epi32(
                _mm256_add_epi32(
                    _mm256_add_epi32(g_blended, _mm256_srli_epi32(g_blended, 8)),
                    one,
                ),
                8,
            );
            let b_div = _mm256_srli_epi32(
                _mm256_add_epi32(
                    _mm256_add_epi32(b_blended, _mm256_srli_epi32(b_blended, 8)),
                    one,
                ),
                8,
            );

            let mut r_out = [0u32; 8];
            let mut g_out = [0u32; 8];
            let mut b_out = [0u32; 8];
            _mm256_storeu_si256(r_out.as_mut_ptr().cast::<__m256i>(), r_div);
            _mm256_storeu_si256(g_out.as_mut_ptr().cast::<__m256i>(), g_div);
            _mm256_storeu_si256(b_out.as_mut_ptr().cast::<__m256i>(), b_div);

            for lane in 0..8 {
                let out_base = output_offset + (lane * 3);
                output[out_base] = r_out[lane] as u8;
                output[out_base + 1] = g_out[lane] as u8;
                output[out_base + 2] = b_out[lane] as u8;
            }
        }
    }

    let input_tail = simd_pixels * 4;
    let output_tail = simd_pixels * 3;
    flatten_rgba_u8_scalar(&input[input_tail..], background, &mut output[output_tail..]);
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
// SAFETY: caller must execute this only on NEON-capable aarch64; this helper only combines live NEON registers for one 16-lane channel.
unsafe fn flatten_rgba_channel_u8(
    src: uint8x16_t,
    alpha: uint8x16_t,
    background: u8,
    inv255: float32x4_t,
) -> uint8x16_t {
    // SAFETY: the caller guarantees NEON availability for the full duration of this helper.
    unsafe {
        let inverse_alpha = vsubq_u8(vdupq_n_u8(u8::MAX), alpha);

        let low = flatten_rgba_channel_half_u8(
            vmovl_u8(vget_low_u8(src)),
            vmovl_u8(vget_low_u8(alpha)),
            vmovl_u8(vget_low_u8(inverse_alpha)),
            background,
            inv255,
        );
        let high = flatten_rgba_channel_half_u8(
            vmovl_u8(vget_high_u8(src)),
            vmovl_u8(vget_high_u8(alpha)),
            vmovl_u8(vget_high_u8(inverse_alpha)),
            background,
            inv255,
        );

        vcombine_u8(low, high)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
// SAFETY: caller must execute this only on NEON-capable aarch64; this helper only combines live NEON registers for one 8-lane half-channel.
unsafe fn flatten_rgba_channel_half_u8(
    src: uint16x8_t,
    alpha: uint16x8_t,
    inverse_alpha: uint16x8_t,
    background: u8,
    inv255: float32x4_t,
) -> uint8x8_t {
    // SAFETY: the caller guarantees NEON availability for the full duration of this helper.
    unsafe {
        let background = u16::from(background);
        let low = flatten_rgba_divide_u32(
            vmlal_n_u16(
                vmull_u16(vget_low_u16(src), vget_low_u16(alpha)),
                vget_low_u16(inverse_alpha),
                background,
            ),
            inv255,
        );
        let high = flatten_rgba_divide_u32(
            vmlal_n_u16(
                vmull_u16(vget_high_u16(src), vget_high_u16(alpha)),
                vget_high_u16(inverse_alpha),
                background,
            ),
            inv255,
        );

        vqmovn_u16(vcombine_u16(vqmovn_u32(low), vqmovn_u32(high)))
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
// SAFETY: caller must execute this only on NEON-capable aarch64; this helper only performs lane-wise arithmetic on the provided vectors.
unsafe fn flatten_rgba_divide_u32(values: uint32x4_t, inv255: float32x4_t) -> uint32x4_t {
    // SAFETY: the caller guarantees NEON availability for the full duration of this helper.
    vcvtq_u32_f32(vmulq_f32(vcvtq_f32_u32(values), inv255))
}

impl Op for Flatten<viprs_core::format::F32> {
    type Input = viprs_core::format::F32;
    type Output = viprs_core::format::F32;
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
        input: &Tile<viprs_core::format::F32>,
        output: &mut TileMut<viprs_core::format::F32>,
    ) {
        let in_bands = self.input_bands as usize;
        if !flatten_has_alpha(self.input_bands) {
            output.data.copy_from_slice(input.data);
            return;
        }
        let out_bands = in_bands - 1;
        let alpha_idx = in_bands - 1;
        let pixel_count = input.region.pixel_count();

        for p in 0..pixel_count {
            let in_base = p * in_bands;
            let out_base = p * out_bands;
            let alpha_f = input.data[in_base + alpha_idx];
            let one_minus_alpha = 1.0 - alpha_f;
            for b in 0..out_bands {
                let src = input.data[in_base + b];
                let bg = self.background[b];
                output.data[out_base + b] = bg.mul_add(one_minus_alpha, src * alpha_f);
            }
        }
    }
}

pub struct FlattenBridge<F: BandFormat> {
    op: Flatten<F>,
    input_bands: u32,
    output_bands: u32,
}

impl FlattenBridge<viprs_core::format::U8> {
    #[must_use]
    pub fn new(input_bands: u32, background: Vec<u8>) -> Self {
        Self {
            op: Flatten::<viprs_core::format::U8>::new(input_bands, background),
            input_bands,
            output_bands: flatten_output_bands(input_bands),
        }
    }
}

impl FlattenBridge<viprs_core::format::F32> {
    #[must_use]
    pub fn new(input_bands: u32, background: Vec<f32>) -> Self {
        Self {
            op: Flatten::<viprs_core::format::F32>::new(input_bands, background),
            input_bands,
            output_bands: flatten_output_bands(input_bands),
        }
    }
}

impl DynOperation for FlattenBridge<viprs_core::format::U8> {
    fn input_format(&self) -> viprs_core::format::BandFormatId {
        viprs_core::format::BandFormatId::U8
    }

    fn output_format(&self) -> viprs_core::format::BandFormatId {
        viprs_core::format::BandFormatId::U8
    }

    fn bands(&self) -> u32 {
        self.output_bands
    }

    fn demand_hint(&self) -> DemandHint {
        self.op.demand_hint()
    }

    fn required_input_region(&self, output: &Region) -> Region {
        self.op.required_input_region(output)
    }

    fn dyn_start(&self) -> Box<dyn std::any::Any + Send> {
        Box::new(())
    }

    fn dyn_process_region(
        &self,
        state: &mut dyn std::any::Any,
        input: &[u8],
        output: &mut [u8],
        input_region: Region,
        output_region: Region,
    ) {
        let Some(state) = state.downcast_mut::<()>() else {
            debug_assert!(false, "flatten bridge state type mismatch");
            return;
        };
        let Ok(input_samples) = bytemuck::try_cast_slice::<u8, u8>(input) else {
            debug_assert!(false, "flatten bridge input cast mismatch");
            return;
        };
        let Ok(output_samples) = bytemuck::try_cast_slice_mut::<u8, u8>(output) else {
            debug_assert!(false, "flatten bridge output cast mismatch");
            return;
        };
        let input_tile =
            Tile::<viprs_core::format::U8>::new(input_region, self.input_bands, input_samples);
        let mut output_tile = TileMut::<viprs_core::format::U8>::new(
            output_region,
            self.output_bands,
            output_samples,
        );
        self.op.process_region(state, &input_tile, &mut output_tile);
    }
}

impl DynOperation for FlattenBridge<viprs_core::format::F32> {
    fn input_format(&self) -> viprs_core::format::BandFormatId {
        viprs_core::format::BandFormatId::F32
    }

    fn output_format(&self) -> viprs_core::format::BandFormatId {
        viprs_core::format::BandFormatId::F32
    }

    fn bands(&self) -> u32 {
        self.output_bands
    }

    fn demand_hint(&self) -> DemandHint {
        self.op.demand_hint()
    }

    fn required_input_region(&self, output: &Region) -> Region {
        self.op.required_input_region(output)
    }

    fn dyn_start(&self) -> Box<dyn std::any::Any + Send> {
        Box::new(())
    }

    fn dyn_process_region(
        &self,
        state: &mut dyn std::any::Any,
        input: &[u8],
        output: &mut [u8],
        input_region: Region,
        output_region: Region,
    ) {
        let Some(state) = state.downcast_mut::<()>() else {
            debug_assert!(false, "flatten bridge state type mismatch");
            return;
        };
        let Ok(input_samples) = bytemuck::try_cast_slice::<u8, f32>(input) else {
            debug_assert!(false, "flatten bridge input cast mismatch");
            return;
        };
        let Ok(output_samples) = bytemuck::try_cast_slice_mut::<u8, f32>(output) else {
            debug_assert!(false, "flatten bridge output cast mismatch");
            return;
        };
        let input_tile =
            Tile::<viprs_core::format::F32>::new(input_region, self.input_bands, input_samples);
        let mut output_tile = TileMut::<viprs_core::format::F32>::new(
            output_region,
            self.output_bands,
            output_samples,
        );
        self.op.process_region(state, &input_tile, &mut output_tile);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::{BandFormatId, F32, U8},
        image::Region,
        op::DynOperation,
    };

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Run `Flatten<U8>::process_region`.
    ///
    /// `input_data` has `input_bands` samples per pixel.
    /// Returned buffer has `input_bands - 1` samples per pixel.
    fn run_flatten_u8(input_data: &[u8], input_bands: u32, background: Vec<u8>) -> Vec<u8> {
        let pixel_count = input_data.len() / input_bands as usize;
        let out_bands = flatten_output_bands(input_bands);
        let width = pixel_count as u32;
        let in_region = Region::new(0, 0, width, 1);
        let out_region = Region::new(0, 0, width, 1);
        let op = Flatten::<U8>::new(input_bands, background);
        let mut out = vec![0u8; pixel_count * out_bands as usize];
        let input = Tile::<U8>::new(in_region, input_bands, input_data);
        let mut output = TileMut::<U8>::new(out_region, out_bands, &mut out);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        out
    }

    fn run_flatten_f32(input_data: &[f32], input_bands: u32, background: Vec<f32>) -> Vec<f32> {
        let pixel_count = input_data.len() / input_bands as usize;
        let out_bands = flatten_output_bands(input_bands);
        let width = pixel_count as u32;
        let in_region = Region::new(0, 0, width, 1);
        let out_region = Region::new(0, 0, width, 1);
        let op = Flatten::<F32>::new(input_bands, background);
        let mut out = vec![0.0f32; pixel_count * out_bands as usize];
        let input = Tile::<F32>::new(in_region, input_bands, input_data);
        let mut output = TileMut::<F32>::new(out_region, out_bands, &mut out);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        out
    }

    fn run_flatten_bridge_u8(
        input_data: &[u8],
        input_bands: u32,
        background: Vec<u8>,
    ) -> (FlattenBridge<U8>, Vec<u8>) {
        let pixel_count = input_data.len() / input_bands as usize;
        let out_bands = flatten_output_bands(input_bands);
        let region = Region::new(0, 0, pixel_count as u32, 1);
        let bridge = FlattenBridge::<U8>::new(input_bands, background);
        let mut state = bridge.dyn_start();
        let mut output = vec![0u8; pixel_count * out_bands as usize];
        bridge.dyn_process_region(
            state.as_mut(),
            bytemuck::cast_slice(input_data),
            bytemuck::cast_slice_mut(output.as_mut_slice()),
            region,
            region,
        );
        (bridge, output)
    }

    fn run_flatten_bridge_f32(
        input_data: &[f32],
        input_bands: u32,
        background: Vec<f32>,
    ) -> (FlattenBridge<F32>, Vec<f32>) {
        let pixel_count = input_data.len() / input_bands as usize;
        let out_bands = flatten_output_bands(input_bands);
        let region = Region::new(0, 0, pixel_count as u32, 1);
        let bridge = FlattenBridge::<F32>::new(input_bands, background);
        let mut state = bridge.dyn_start();
        let mut output = vec![0.0f32; pixel_count * out_bands as usize];
        bridge.dyn_process_region(
            state.as_mut(),
            bytemuck::cast_slice(input_data),
            bytemuck::cast_slice_mut(output.as_mut_slice()),
            region,
            region,
        );
        (bridge, output)
    }

    // ── U8: fully transparent → background ───────────────────────────────────

    #[test]
    fn u8_alpha_zero_yields_background() {
        // alpha=0 → output == background
        let input = vec![200u8, 100u8, 50u8, 0u8]; // R=200, G=100, B=50, A=0
        let bg = vec![10u8, 20u8, 30u8];
        let result = run_flatten_u8(&input, 4, bg.clone());
        assert_eq!(result, bg);
    }

    #[test]
    fn u8_alpha_max_yields_source() {
        // alpha=255 → output == source RGB
        let input = vec![200u8, 100u8, 50u8, 255u8];
        let bg = vec![10u8, 20u8, 30u8];
        let result = run_flatten_u8(&input, 4, bg);
        assert_eq!(result[0], 200);
        assert_eq!(result[1], 100);
        assert_eq!(result[2], 50);
    }

    #[test]
    fn u8_half_alpha_blends_correctly() {
        // alpha=128 ≈ 0.502
        // out = 200 * 0.502 + 0 * 0.498 ≈ 100.4 → round → 100
        // 2-band image (grey + alpha): [grey=200, alpha=128]
        let input = vec![200u8, 128u8]; // grey=200, A=128
        let bg = vec![0u8];
        let result = run_flatten_u8(&input, 2, bg);
        // 200 * (128/255) = 200 * 0.5020 ≈ 100.4 → 100
        assert_eq!(result[0], 100);
    }

    #[test]
    fn u8_uses_libvips_integer_division_semantics() {
        let input = vec![0u8, 0u8, 0u8, 1u8];
        let bg = vec![1u8, 1u8, 1u8];
        let result = run_flatten_u8(&input, 4, bg);

        assert_eq!(result, vec![0u8, 0u8, 0u8]);
    }

    #[test]
    fn flatten_rgb_is_noop() {
        let input = vec![10u8, 20u8, 30u8, 40u8, 50u8, 60u8];
        let result = run_flatten_u8(&input, 3, vec![1u8, 2u8, 3u8]);

        assert_eq!(result, input);
    }

    // ── F32: fully transparent → background ──────────────────────────────────

    #[test]
    fn f32_alpha_zero_yields_background() {
        let input = vec![0.8f32, 0.5f32, 0.3f32, 0.0f32];
        let bg = vec![0.1f32, 0.2f32, 0.3f32];
        let result = run_flatten_f32(&input, 4, bg.clone());
        assert!((result[0] - 0.1).abs() < f32::EPSILON);
        assert!((result[1] - 0.2).abs() < f32::EPSILON);
        assert!((result[2] - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn f32_alpha_one_yields_source() {
        let input = vec![0.8f32, 0.5f32, 0.3f32, 1.0f32];
        let bg = vec![0.1f32, 0.2f32, 0.3f32];
        let result = run_flatten_f32(&input, 4, bg);
        assert!((result[0] - 0.8).abs() < f32::EPSILON);
        assert!((result[1] - 0.5).abs() < f32::EPSILON);
        assert!((result[2] - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn f32_half_alpha_blends_correctly() {
        // src=1.0, bg=0.0, alpha=0.5 → out = 0.5
        // 2-band image (grey + alpha): [grey=1.0, alpha=0.5]
        let input = vec![1.0f32, 0.5f32]; // grey=1.0, alpha=0.5
        let bg = vec![0.0f32];
        let result = run_flatten_f32(&input, 2, bg);
        assert!((result[0] - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn flatten_bridge_u8_rgba_to_rgb_honours_transparency_extremes() {
        let input = [10u8, 20, 30, 0, 40, 50, 60, 255];
        let background = vec![1u8, 2, 3];
        let (bridge, result) = run_flatten_bridge_u8(&input, 4, background);

        assert_eq!(bridge.input_format(), BandFormatId::U8);
        assert_eq!(bridge.output_format(), BandFormatId::U8);
        assert_eq!(bridge.bands(), 3);
        assert_eq!(
            bridge.demand_hint(),
            viprs_core::image::DemandHint::ThinStrip
        );
        assert_eq!(
            bridge.required_input_region(&Region::new(0, 0, 2, 1)),
            Region::new(0, 0, 2, 1)
        );
        assert_eq!(result, vec![1u8, 2, 3, 40, 50, 60]);
    }

    #[test]
    fn flatten_bridge_f32_rgba_to_rgb_honours_transparency_extremes() {
        let input = [0.2f32, 0.4, 0.6, 0.0, 0.1, 0.3, 0.5, 1.0];
        let background = vec![0.7f32, 0.8, 0.9];
        let (bridge, result) = run_flatten_bridge_f32(&input, 4, background);

        assert_eq!(bridge.input_format(), BandFormatId::F32);
        assert_eq!(bridge.output_format(), BandFormatId::F32);
        assert_eq!(bridge.bands(), 3);
        assert!((result[0] - 0.7).abs() < f32::EPSILON);
        assert!((result[1] - 0.8).abs() < f32::EPSILON);
        assert!((result[2] - 0.9).abs() < f32::EPSILON);
        assert!((result[3] - 0.1).abs() < f32::EPSILON);
        assert!((result[4] - 0.3).abs() < f32::EPSILON);
        assert!((result[5] - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn flatten_bridge_u8_rgb_is_noop() {
        let input = [10u8, 20, 30, 40, 50, 60];
        let (bridge, result) = run_flatten_bridge_u8(&input, 3, vec![1u8, 2, 3]);

        assert_eq!(bridge.bands(), 3);
        assert_eq!(result, input);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_rgba_path_matches_scalar_reference() {
        let mut input = Vec::with_capacity(19 * 4);
        for idx in 0..19u8 {
            input.extend_from_slice(&[
                idx.wrapping_mul(17),
                idx.wrapping_mul(31),
                idx.wrapping_mul(47),
                idx.wrapping_mul(13),
            ]);
        }

        let background = [11u8, 97u8, 203u8];
        let mut scalar = vec![0u8; 19 * 3];
        let mut neon = vec![0u8; 19 * 3];

        flatten_rgba_u8_scalar(&input, background, &mut scalar);
        // SAFETY: this test only compiles on aarch64, where NEON is guaranteed.
        unsafe {
            flatten_rgba_u8_neon(&input, background, &mut neon);
        }

        assert_eq!(neon, scalar);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    proptest! {
        #[test]
        fn avx2_rgba_path_matches_scalar_reference(
            pixels in proptest::collection::vec(any::<u8>(), 32..=512),
            bg_r in any::<u8>(),
            bg_g in any::<u8>(),
            bg_b in any::<u8>(),
        ) {
            if !std::arch::is_x86_feature_detected!("avx2") {
                return Ok(());
            }

            let usable_len = pixels.len() - (pixels.len() % 4);
            prop_assume!(usable_len > 0);
            let input = &pixels[..usable_len];
            let background = [bg_r, bg_g, bg_b];

            let mut scalar = vec![0u8; (usable_len / 4) * 3];
            flatten_rgba_u8_scalar(input, background, &mut scalar);

            let avx2 = run_flatten_u8(input, 4, background.to_vec());
            prop_assert_eq!(avx2, scalar);
        }
    }

    // ── Proptest ──────────────────────────────────────────────────────────────

    proptest! {
        /// alpha=0 always yields the background colour exactly.
        #[test]
        fn f32_zero_alpha_always_background(
            r in 0.0f32..=1.0f32,
            g in 0.0f32..=1.0f32,
            b in 0.0f32..=1.0f32,
            bg_r in 0.0f32..=1.0f32,
            bg_g in 0.0f32..=1.0f32,
            bg_b in 0.0f32..=1.0f32,
        ) {
            let input = vec![r, g, b, 0.0f32];
            let bg = vec![bg_r, bg_g, bg_b];
            let result = run_flatten_f32(&input, 4, bg.clone());
            prop_assert!((result[0] - bg_r).abs() < f32::EPSILON);
            prop_assert!((result[1] - bg_g).abs() < f32::EPSILON);
            prop_assert!((result[2] - bg_b).abs() < f32::EPSILON);
        }

        /// alpha=1 always yields the source colour exactly.
        #[test]
        fn f32_full_alpha_always_source(
            r in 0.0f32..=1.0f32,
            g in 0.0f32..=1.0f32,
            b in 0.0f32..=1.0f32,
            bg_r in 0.0f32..=1.0f32,
            bg_g in 0.0f32..=1.0f32,
            bg_b in 0.0f32..=1.0f32,
        ) {
            let input = vec![r, g, b, 1.0f32];
            let bg = vec![bg_r, bg_g, bg_b];
            let result = run_flatten_f32(&input, 4, bg);
            prop_assert!((result[0] - r).abs() < f32::EPSILON);
            prop_assert!((result[1] - g).abs() < f32::EPSILON);
            prop_assert!((result[2] - b).abs() < f32::EPSILON);
        }

        /// U8: alpha=0 always yields the background colour exactly.
        #[test]
        fn u8_zero_alpha_always_background(
            r in 0u8..=255u8,
            g in 0u8..=255u8,
            b in 0u8..=255u8,
            bg_r in 0u8..=255u8,
            bg_g in 0u8..=255u8,
            bg_b in 0u8..=255u8,
        ) {
            let input = vec![r, g, b, 0u8];
            let bg = vec![bg_r, bg_g, bg_b];
            let result = run_flatten_u8(&input, 4, bg.clone());
            prop_assert_eq!(result[0], bg_r);
            prop_assert_eq!(result[1], bg_g);
            prop_assert_eq!(result[2], bg_b);
        }

        /// U8: alpha=255 always yields the source colour exactly.
        #[test]
        fn u8_full_alpha_always_source(
            r in 0u8..=255u8,
            g in 0u8..=255u8,
            b in 0u8..=255u8,
            bg_r in 0u8..=255u8,
            bg_g in 0u8..=255u8,
            bg_b in 0u8..=255u8,
        ) {
            let input = vec![r, g, b, 255u8];
            let bg = vec![bg_r, bg_g, bg_b];
            let result = run_flatten_u8(&input, 4, bg);
            prop_assert_eq!(result[0], r);
            prop_assert_eq!(result[1], g);
            prop_assert_eq!(result[2], b);
        }
    }
}

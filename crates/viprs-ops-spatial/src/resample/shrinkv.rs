#![allow(missing_docs)]
// REASON: these bridge helpers are public only for cross-crate workspace wiring, not end-user API.

//! Vertical integer shrink by a box filter.
//!
//! `ShrinkV` averages `factor` consecutive rows per output row, producing a
//! downscaled image with `output_height = floor(input_height / factor)` by default.
//! Set `ceil=true` to round the output height up (libvips parity).
//! It is the fast path for large integer downscales along the vertical axis.
//! Use `ShrinkV` followed by `ReduceV` to handle the fractional remainder.

#![allow(dead_code)]
// REASON: alternate vertical shrink constructors are retained for upcoming planner integration.
#![allow(clippy::used_underscore_binding)]
// REASON: underscore-prefixed parameters document intentionally ignored planner-only inputs.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    vaddw_u16, vcombine_u8, vcombine_u16, vdupq_n_s32, vget_high_u8, vget_high_u16, vget_low_u8,
    vget_low_u16, vld1q_u8, vld1q_u32, vmovl_u8, vqdmulhq_s32, vqmovn_u16, vqmovun_s32,
    vreinterpretq_s32_u32, vst1q_u8, vst1q_u32,
};
use std::marker::PhantomData;

use viprs_core::{
    error::BuildError,
    format::{BandFormat, BandFormatId},
    image::{DemandHint, Region, Tile, TileMut},
    op::{NodeSpec, Op},
    resample::ResampleOp,
};

use super::shrinkh::ShrinkSample;

/// Vertical integer shrink by `factor`.
///
/// Averages `factor` consecutive rows per output row using f64 accumulation
/// with saturating cast back to `F::Sample`. `factor` must be >= 1. Factor=1
/// is identity.
///
/// This is the fast path for large vertical downscales: O(n/factor) work vs
/// O(`n*kernel_support`) for `ReduceV`. Use `ShrinkV` followed by `ReduceV` for
/// the fractional remainder.
///
/// # Construction
///
/// ```rust,ignore
/// let op = ShrinkV::<U8>::new(2)?;
/// ```
pub struct ShrinkV<F: BandFormat> {
    factor: u32,
    ceil: bool,
    _format: PhantomData<F>,
}

type ShrinkVU8Scratch = Vec<u32>;

fn validate_shrink_v_factor(factor: u32) -> Result<(), BuildError> {
    if factor == 0 {
        return Err(BuildError::SourceHint {
            context: "shrink_v",
            message: "factor must be >= 1".to_string(),
        });
    }

    Ok(())
}

impl<F: BandFormat + Send + Sync> ShrinkV<F> {
    /// Create a new `ShrinkV` that shrinks the image height by `factor`.
    ///
    /// `factor` must be >= 1. Factor=1 is identity (output == input height).
    pub fn new(factor: u32) -> Result<Self, BuildError> {
        Self::new_with_ceil(factor, false)
    }

    /// Create a new `ShrinkV` that shrinks the image height by `factor`,
    /// optionally rounding the output height up.
    pub fn new_with_ceil(factor: u32, ceil: bool) -> Result<Self, BuildError> {
        validate_shrink_v_factor(factor)?;
        debug_assert!(factor >= 1, "ShrinkV: factor must be >= 1");
        Ok(Self {
            factor,
            ceil,
            _format: PhantomData,
        })
    }

    /// Return a copy configured with the given `ceil` output-dimension mode.
    #[must_use]
    pub const fn with_ceil(mut self, ceil: bool) -> Self {
        self.ceil = ceil;
        self
    }
}

#[inline]
fn saturating_scale_i32(value: i32, factor: u32) -> i32 {
    i64::from(value)
        .saturating_mul(i64::from(factor))
        .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

impl<F> Op for ShrinkV<F>
where
    F: BandFormat,
    F::Sample: ShrinkSample + bytemuck::Pod,
{
    type Input = F;
    type Output = F;
    type State = ShrinkVU8Scratch;

    fn demand_hint(&self) -> DemandHint {
        // Vertical shrink reads `factor` consecutive rows per output row. ThinStrip keeps
        // per-thread buffer sizes proportional to factor×tile_h rows, which is far more
        // memory-efficient than FatStrip for large shrink factors (e.g. factor=19 with
        // FatStrip tile_h=256 → 4864 rows per tile × source_width × threads ≈ 120 MB).
        // ThinStrip with tile_h=16 → 304 rows ≈ 7 MB, and allows finer-grained parallelism.
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        Region::new(
            output.x,
            saturating_scale_i32(output.y, self.factor),
            output.width,
            output.height.saturating_mul(self.factor),
        )
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec {
            input_tile_w: tile_w,
            input_tile_h: tile_h.saturating_mul(self.factor),
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    fn start(&self) -> Self::State {
        Vec::new()
    }

    fn start_with_tile_and_bands(&self, tile_w: u32, _tile_h: u32, bands: u32) -> Self::State {
        if self.factor <= 2 {
            Vec::new()
        } else {
            vec![0; tile_w as usize * bands as usize]
        }
    }

    #[inline]
    fn process_region(&self, state: &mut Self::State, input: &Tile<F>, output: &mut TileMut<F>) {
        if F::ID == BandFormatId::U8 {
            shrink_v_u8(
                self.factor as usize,
                bytemuck::cast_slice(input.data),
                input.bands as usize,
                output.region.width as usize,
                output.region.height as usize,
                state,
                bytemuck::cast_slice_mut(output.data),
            );
            return;
        }

        let factor = self.factor as usize;
        let bands = input.bands as usize;
        let out_w = output.region.width as usize;
        let out_h = output.region.height as usize;
        let inv_factor = 1.0_f64 / factor as f64;

        for y_out in 0..out_h {
            for x in 0..out_w {
                for b in 0..bands {
                    let mut sum = 0.0_f64;
                    for k in 0..factor {
                        let y_in = y_out * factor + k;
                        let idx = (y_in * out_w + x) * bands + b;
                        sum += input.data[idx].to_f64();
                    }
                    let out_idx = (y_out * out_w + x) * bands + b;
                    output.data[out_idx] = F::Sample::from_f64_clamped(sum * inv_factor);
                }
            }
        }
    }
}

#[inline]
fn shrink_v_u8(
    factor: usize,
    input: &[u8],
    bands: usize,
    out_w: usize,
    out_h: usize,
    scratch: &mut Vec<u32>,
    output: &mut [u8],
) {
    if factor == 1 {
        output.copy_from_slice(input);
        return;
    }

    if factor == 2 {
        shrink_v_u8_factor2_scalar(input, bands, out_w, out_h, output);
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: only runs on aarch64 with NEON always available.
        // Operates on contiguous row slices with bounds checked by row_len.
        unsafe {
            shrink_v_u8_generic_neon(factor, input, bands, out_w, out_h, scratch, output);
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: the runtime AVX2 check guarantees the required ISA support,
            // and the helper only touches rows and scratch slices derived from
            // the validated output geometry.
            unsafe {
                shrink_v_u8_generic_avx2(factor, input, bands, out_w, out_h, scratch, output);
            }
        } else {
            shrink_v_u8_generic_scalar(factor, input, bands, out_w, out_h, scratch, output);
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
    shrink_v_u8_generic_scalar(factor, input, bands, out_w, out_h, scratch, output);
}

#[inline]
fn shrink_v_u8_factor2_scalar(
    input: &[u8],
    bands: usize,
    out_w: usize,
    out_h: usize,
    output: &mut [u8],
) {
    let row_len = out_w * bands;

    for y_out in 0..out_h {
        let src_idx = y_out * row_len * 2;
        let src_row0 = &input[src_idx..src_idx + row_len];
        let src_row1 = &input[src_idx + row_len..src_idx + row_len * 2];
        let output_row = &mut output[y_out * row_len..(y_out + 1) * row_len];

        match bands {
            3 => {
                let mut idx = 0usize;
                while idx < row_len {
                    let red = u16::from(src_row0[idx]) + u16::from(src_row1[idx]);
                    let green = u16::from(src_row0[idx + 1]) + u16::from(src_row1[idx + 1]);
                    let blue = u16::from(src_row0[idx + 2]) + u16::from(src_row1[idx + 2]);
                    output_row[idx] = ((red + 1) >> 1) as u8;
                    output_row[idx + 1] = ((green + 1) >> 1) as u8;
                    output_row[idx + 2] = ((blue + 1) >> 1) as u8;
                    idx += 3;
                }
            }
            4 => {
                let mut idx = 0usize;
                while idx < row_len {
                    let c0 = u16::from(src_row0[idx]) + u16::from(src_row1[idx]);
                    let c1 = u16::from(src_row0[idx + 1]) + u16::from(src_row1[idx + 1]);
                    let c2 = u16::from(src_row0[idx + 2]) + u16::from(src_row1[idx + 2]);
                    let c3 = u16::from(src_row0[idx + 3]) + u16::from(src_row1[idx + 3]);
                    output_row[idx] = ((c0 + 1) >> 1) as u8;
                    output_row[idx + 1] = ((c1 + 1) >> 1) as u8;
                    output_row[idx + 2] = ((c2 + 1) >> 1) as u8;
                    output_row[idx + 3] = ((c3 + 1) >> 1) as u8;
                    idx += 4;
                }
            }
            _ => {
                for ((dst, sample0), sample1) in output_row
                    .iter_mut()
                    .zip(src_row0.iter().copied())
                    .zip(src_row1.iter().copied())
                {
                    let sum = u16::from(sample0) + u16::from(sample1);
                    *dst = ((sum + 1) >> 1) as u8;
                }
            }
        }
    }
}

#[allow(dead_code)]
fn shrink_v_u8_generic_scalar(
    factor: usize,
    input: &[u8],
    bands: usize,
    out_w: usize,
    _out_h: usize,
    scratch: &mut Vec<u32>,
    output: &mut [u8],
) {
    let row_len = out_w * bands;
    if scratch.len() < row_len {
        scratch.resize(row_len, 0);
    }
    let amend = factor as u32 / 2;
    let multiplier = shrink_u8_multiplier(factor);

    for y_out in 0..output.len() / row_len {
        let sums = &mut scratch[..row_len];
        sums.fill(amend);

        for k in 0..factor {
            let src_idx = (y_out * factor + k) * row_len;
            let src_row = &input[src_idx..src_idx + row_len];
            for (sum, sample) in sums.iter_mut().zip(src_row.iter().copied()) {
                *sum += u32::from(sample);
            }
        }

        let output_row = &mut output[y_out * row_len..(y_out + 1) * row_len];
        for (dst, sum) in output_row.iter_mut().zip(sums.iter().copied()) {
            *dst = shrink_u8_fixed_point_average(sum, multiplier);
        }
    }
}

#[inline(always)]
const fn shrink_u8_multiplier(factor: usize) -> u32 {
    ((1_u64 << 32) / ((1_u64 << 8) * factor as u64)) as u32
}

#[inline(always)]
const fn shrink_u8_fixed_point_average(sum: u32, multiplier: u32) -> u8 {
    (((sum as u64) * multiplier as u64) >> 24) as u8
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
// SAFETY: caller must dispatch only when AVX2 is available; `input`, `scratch`,
// and `output` must cover `out_h` rows of `out_w * bands` samples so every
// vector load/store stays within the provided slices.
// REASON: AVX2 load/store intrinsics accept unaligned pointers; the casts are
// intentional to operate on contiguous pixel and accumulator buffers.
#[allow(clippy::cast_ptr_alignment)]
unsafe fn shrink_v_u8_generic_avx2(
    factor: usize,
    input: &[u8],
    bands: usize,
    out_w: usize,
    out_h: usize,
    scratch: &mut Vec<u32>,
    output: &mut [u8],
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        __m128i, __m256i, _mm_loadl_epi64, _mm256_add_epi32, _mm256_cvtepu8_epi32,
        _mm256_loadu_si256, _mm256_mul_epu32, _mm256_set1_epi32, _mm256_srli_epi64,
        _mm256_storeu_si256,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        __m128i, __m256i, _mm_loadl_epi64, _mm256_add_epi32, _mm256_cvtepu8_epi32,
        _mm256_loadu_si256, _mm256_mul_epu32, _mm256_set1_epi32, _mm256_srli_epi64,
        _mm256_storeu_si256,
    };

    let row_len = out_w * bands;
    let amend = factor as u32 / 2;
    let multiplier = shrink_u8_multiplier(factor);
    let chunks = row_len / 32;

    if scratch.len() < row_len {
        scratch.resize(row_len, 0);
    }

    if (factor as u64 * u64::from(u8::MAX)) + u64::from(amend) > i32::MAX as u64 {
        shrink_v_u8_generic_scalar(factor, input, bands, out_w, out_h, scratch, output);
        return;
    }

    // AVX2 is enabled for this function, so splatting the scalar multiplier is valid.
    let multiplier_vec = _mm256_set1_epi32(multiplier as i32);

    for y_out in 0..out_h {
        scratch[..row_len].fill(amend);

        for k in 0..factor {
            let src_idx = (y_out * factor + k) * row_len;
            let src_row = &input[src_idx..src_idx + row_len];

            for chunk in 0..chunks {
                let offset = chunk * 32;

                // SAFETY: each 8-byte input load and 8-lane scratch load/store stays
                // within the current row chunk because `offset + 32 <= row_len`.
                unsafe {
                    let src_ptr = src_row.as_ptr().add(offset);
                    let scratch_ptr = scratch.as_mut_ptr().add(offset);

                    let pixels0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(src_ptr.cast::<__m128i>()));
                    let acc0 = _mm256_loadu_si256(scratch_ptr.cast::<__m256i>());
                    _mm256_storeu_si256(
                        scratch_ptr.cast::<__m256i>(),
                        _mm256_add_epi32(acc0, pixels0),
                    );

                    let pixels1 =
                        _mm256_cvtepu8_epi32(_mm_loadl_epi64(src_ptr.add(8).cast::<__m128i>()));
                    let acc1 = _mm256_loadu_si256(scratch_ptr.add(8).cast::<__m256i>());
                    _mm256_storeu_si256(
                        scratch_ptr.add(8).cast::<__m256i>(),
                        _mm256_add_epi32(acc1, pixels1),
                    );

                    let pixels2 =
                        _mm256_cvtepu8_epi32(_mm_loadl_epi64(src_ptr.add(16).cast::<__m128i>()));
                    let acc2 = _mm256_loadu_si256(scratch_ptr.add(16).cast::<__m256i>());
                    _mm256_storeu_si256(
                        scratch_ptr.add(16).cast::<__m256i>(),
                        _mm256_add_epi32(acc2, pixels2),
                    );

                    let pixels3 =
                        _mm256_cvtepu8_epi32(_mm_loadl_epi64(src_ptr.add(24).cast::<__m128i>()));
                    let acc3 = _mm256_loadu_si256(scratch_ptr.add(24).cast::<__m256i>());
                    _mm256_storeu_si256(
                        scratch_ptr.add(24).cast::<__m256i>(),
                        _mm256_add_epi32(acc3, pixels3),
                    );
                }
            }

            for i in chunks * 32..row_len {
                scratch[i] += u32::from(src_row[i]);
            }
        }

        let output_row = &mut output[y_out * row_len..(y_out + 1) * row_len];
        for chunk in 0..chunks {
            let offset = chunk * 32;

            // SAFETY: each scratch vector covers 8 contiguous accumulators within the
            // current chunk and the temporary arrays are sized for the AVX2 stores.
            unsafe {
                let scratch_ptr = scratch.as_ptr().add(offset);

                let sums0 = _mm256_loadu_si256(scratch_ptr.cast::<__m256i>());
                let even0 = _mm256_srli_epi64(_mm256_mul_epu32(sums0, multiplier_vec), 24);
                let odd0 = _mm256_srli_epi64(
                    _mm256_mul_epu32(_mm256_srli_epi64(sums0, 32), multiplier_vec),
                    24,
                );
                let mut even0_lanes = [0_u64; 4];
                let mut odd0_lanes = [0_u64; 4];
                _mm256_storeu_si256(even0_lanes.as_mut_ptr().cast::<__m256i>(), even0);
                _mm256_storeu_si256(odd0_lanes.as_mut_ptr().cast::<__m256i>(), odd0);
                output_row[offset] = even0_lanes[0] as u8;
                output_row[offset + 1] = odd0_lanes[0] as u8;
                output_row[offset + 2] = even0_lanes[1] as u8;
                output_row[offset + 3] = odd0_lanes[1] as u8;
                output_row[offset + 4] = even0_lanes[2] as u8;
                output_row[offset + 5] = odd0_lanes[2] as u8;
                output_row[offset + 6] = even0_lanes[3] as u8;
                output_row[offset + 7] = odd0_lanes[3] as u8;

                let sums1 = _mm256_loadu_si256(scratch_ptr.add(8).cast::<__m256i>());
                let even1 = _mm256_srli_epi64(_mm256_mul_epu32(sums1, multiplier_vec), 24);
                let odd1 = _mm256_srli_epi64(
                    _mm256_mul_epu32(_mm256_srli_epi64(sums1, 32), multiplier_vec),
                    24,
                );
                let mut even1_lanes = [0_u64; 4];
                let mut odd1_lanes = [0_u64; 4];
                _mm256_storeu_si256(even1_lanes.as_mut_ptr().cast::<__m256i>(), even1);
                _mm256_storeu_si256(odd1_lanes.as_mut_ptr().cast::<__m256i>(), odd1);
                output_row[offset + 8] = even1_lanes[0] as u8;
                output_row[offset + 9] = odd1_lanes[0] as u8;
                output_row[offset + 10] = even1_lanes[1] as u8;
                output_row[offset + 11] = odd1_lanes[1] as u8;
                output_row[offset + 12] = even1_lanes[2] as u8;
                output_row[offset + 13] = odd1_lanes[2] as u8;
                output_row[offset + 14] = even1_lanes[3] as u8;
                output_row[offset + 15] = odd1_lanes[3] as u8;

                let sums2 = _mm256_loadu_si256(scratch_ptr.add(16).cast::<__m256i>());
                let even2 = _mm256_srli_epi64(_mm256_mul_epu32(sums2, multiplier_vec), 24);
                let odd2 = _mm256_srli_epi64(
                    _mm256_mul_epu32(_mm256_srli_epi64(sums2, 32), multiplier_vec),
                    24,
                );
                let mut even2_lanes = [0_u64; 4];
                let mut odd2_lanes = [0_u64; 4];
                _mm256_storeu_si256(even2_lanes.as_mut_ptr().cast::<__m256i>(), even2);
                _mm256_storeu_si256(odd2_lanes.as_mut_ptr().cast::<__m256i>(), odd2);
                output_row[offset + 16] = even2_lanes[0] as u8;
                output_row[offset + 17] = odd2_lanes[0] as u8;
                output_row[offset + 18] = even2_lanes[1] as u8;
                output_row[offset + 19] = odd2_lanes[1] as u8;
                output_row[offset + 20] = even2_lanes[2] as u8;
                output_row[offset + 21] = odd2_lanes[2] as u8;
                output_row[offset + 22] = even2_lanes[3] as u8;
                output_row[offset + 23] = odd2_lanes[3] as u8;

                let sums3 = _mm256_loadu_si256(scratch_ptr.add(24).cast::<__m256i>());
                let even3 = _mm256_srli_epi64(_mm256_mul_epu32(sums3, multiplier_vec), 24);
                let odd3 = _mm256_srli_epi64(
                    _mm256_mul_epu32(_mm256_srli_epi64(sums3, 32), multiplier_vec),
                    24,
                );
                let mut even3_lanes = [0_u64; 4];
                let mut odd3_lanes = [0_u64; 4];
                _mm256_storeu_si256(even3_lanes.as_mut_ptr().cast::<__m256i>(), even3);
                _mm256_storeu_si256(odd3_lanes.as_mut_ptr().cast::<__m256i>(), odd3);
                output_row[offset + 24] = even3_lanes[0] as u8;
                output_row[offset + 25] = odd3_lanes[0] as u8;
                output_row[offset + 26] = even3_lanes[1] as u8;
                output_row[offset + 27] = odd3_lanes[1] as u8;
                output_row[offset + 28] = even3_lanes[2] as u8;
                output_row[offset + 29] = odd3_lanes[2] as u8;
                output_row[offset + 30] = even3_lanes[3] as u8;
                output_row[offset + 31] = odd3_lanes[3] as u8;
            }
        }

        for (dst, sum) in output_row[chunks * 32..]
            .iter_mut()
            .zip(scratch[chunks * 32..row_len].iter().copied())
        {
            *dst = shrink_u8_fixed_point_average(sum, multiplier);
        }
    }
}

/// NEON-accelerated vertical shrink for any factor > 2.
///
/// Accumulates `factor` input rows into u32 accumulators using widening adds
/// (u8→u16→u32), then converts sums to u8 with the same fixed-point reciprocal
/// multiply used by libvips and the horizontal shrink path.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn shrink_v_u8_generic_neon(
    factor: usize,
    input: &[u8],
    _bands: usize,
    out_w: usize,
    out_h: usize,
    scratch: &mut Vec<u32>,
    output: &mut [u8],
) {
    let row_len = out_w * _bands;
    let amend = factor as u32 / 2;
    let multiplier = shrink_u8_multiplier(factor);
    let chunks = row_len / 16;

    if scratch.len() < row_len {
        scratch.resize(row_len, 0);
    }

    if (factor as u64 * u64::from(u8::MAX)) + u64::from(amend) > i32::MAX as u64 {
        shrink_v_u8_generic_scalar(factor, input, _bands, out_w, _bands, scratch, output);
        return;
    }

    // SAFETY: aarch64 guarantees NEON support for this kernel.
    let reciprocal = unsafe { vdupq_n_s32((multiplier << 7) as i32) };

    for y_out in 0..out_h {
        // Initialize accumulators with rounding bias
        scratch[..row_len].fill(amend);

        // Accumulate factor rows — process each row sequentially (cache-friendly)
        for k in 0..factor {
            let src_idx = (y_out * factor + k) * row_len;
            let src_row = &input[src_idx..src_idx + row_len];

            // NEON accumulation: 16 bytes → widen to u32 and add
            for chunk in 0..chunks {
                let offset = chunk * 16;
                // SAFETY: `offset..offset+16` is within both `src_row` and `scratch`.
                unsafe {
                    let pixels = vld1q_u8(src_row.as_ptr().add(offset));

                    // Load current accumulators (4 × u32x4)
                    let scratch_ptr = scratch.as_mut_ptr().add(offset);
                    let mut acc0 = vld1q_u32(scratch_ptr);
                    let mut acc1 = vld1q_u32(scratch_ptr.add(4));
                    let mut acc2 = vld1q_u32(scratch_ptr.add(8));
                    let mut acc3 = vld1q_u32(scratch_ptr.add(12));

                    // Widen u8→u16
                    let lo16 = vmovl_u8(vget_low_u8(pixels));
                    let hi16 = vmovl_u8(vget_high_u8(pixels));

                    // Widen u16→u32 and accumulate
                    acc0 = vaddw_u16(acc0, vget_low_u16(lo16));
                    acc1 = vaddw_u16(acc1, vget_high_u16(lo16));
                    acc2 = vaddw_u16(acc2, vget_low_u16(hi16));
                    acc3 = vaddw_u16(acc3, vget_high_u16(hi16));

                    // Store back
                    vst1q_u32(scratch_ptr, acc0);
                    vst1q_u32(scratch_ptr.add(4), acc1);
                    vst1q_u32(scratch_ptr.add(8), acc2);
                    vst1q_u32(scratch_ptr.add(12), acc3);
                }
            }

            // Scalar tail for accumulation
            let tail_start = chunks * 16;
            for i in tail_start..row_len {
                scratch[i] += u32::from(src_row[i]);
            }
        }

        let output_row = &mut output[y_out * row_len..(y_out + 1) * row_len];
        for chunk in 0..chunks {
            let offset = chunk * 16;
            // SAFETY: `offset..offset+16` is within both `scratch` and `output_row`.
            unsafe {
                let scratch_ptr = scratch.as_ptr().add(offset);

                let acc0 = vreinterpretq_s32_u32(vld1q_u32(scratch_ptr));
                let acc1 = vreinterpretq_s32_u32(vld1q_u32(scratch_ptr.add(4)));
                let acc2 = vreinterpretq_s32_u32(vld1q_u32(scratch_ptr.add(8)));
                let acc3 = vreinterpretq_s32_u32(vld1q_u32(scratch_ptr.add(12)));

                let quot0 = vqdmulhq_s32(acc0, reciprocal);
                let quot1 = vqdmulhq_s32(acc1, reciprocal);
                let quot2 = vqdmulhq_s32(acc2, reciprocal);
                let quot3 = vqdmulhq_s32(acc3, reciprocal);

                let packed_lo = vcombine_u16(vqmovun_s32(quot0), vqmovun_s32(quot1));
                let packed_hi = vcombine_u16(vqmovun_s32(quot2), vqmovun_s32(quot3));
                let pixels = vcombine_u8(vqmovn_u16(packed_lo), vqmovn_u16(packed_hi));

                vst1q_u8(output_row.as_mut_ptr().add(offset), pixels);
            }
        }

        for (dst, sum) in output_row[chunks * 16..]
            .iter_mut()
            .zip(scratch[chunks * 16..row_len].iter().copied())
        {
            *dst = shrink_u8_fixed_point_average(sum, multiplier);
        }
    }
}

impl<F> ResampleOp for ShrinkV<F>
where
    F: BandFormat,
    F::Sample: ShrinkSample + bytemuck::Pod,
{
    fn output_width(&self, input_w: u32) -> u32 {
        // Vertical pass: width is unchanged.
        input_w
    }

    fn output_height(&self, input_h: u32) -> u32 {
        if self.ceil {
            input_h.div_ceil(self.factor)
        } else {
            input_h / self.factor
        }
    }
}

pub struct ShrinkVBridge<F: BandFormat>
where
    F::Sample: bytemuck::Pod + ShrinkSample,
{
    inner: viprs_core::op::OperationBridge<ShrinkV<F>>,
}

impl<F: BandFormat> ShrinkVBridge<F>
where
    F::Sample: bytemuck::Pod + ShrinkSample,
{
    pub fn new(factor: u32, bands: u32) -> Result<Self, BuildError> {
        Self::new_with_ceil(factor, false, bands)
    }

    pub fn new_with_ceil(factor: u32, ceil: bool, bands: u32) -> Result<Self, BuildError> {
        validate_shrink_v_factor(factor)?;
        if bands == 0 {
            return Err(BuildError::SourceHint {
                context: "shrink_v",
                message: "band count must be >= 1".to_string(),
            });
        }

        Ok(Self {
            inner: viprs_core::op::OperationBridge::new(
                ShrinkV::new_with_ceil(factor, ceil)?,
                bands,
            ),
        })
    }
}

impl<F: BandFormat> viprs_core::op::DynOperation for ShrinkVBridge<F>
where
    F::Sample: bytemuck::Pod + ShrinkSample + Send,
{
    fn input_format(&self) -> viprs_core::format::BandFormatId {
        self.inner.input_format()
    }

    fn output_format(&self) -> viprs_core::format::BandFormatId {
        self.inner.output_format()
    }

    fn bands(&self) -> u32 {
        self.inner.bands()
    }

    fn demand_hint(&self) -> DemandHint {
        self.inner.demand_hint()
    }

    fn required_input_region(&self, output: &Region) -> Region {
        self.inner.required_input_region(output)
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        self.inner.node_spec(tile_w, tile_h)
    }

    fn output_width(&self, input_w: u32) -> u32 {
        input_w
    }

    fn output_height(&self, input_h: u32) -> u32 {
        self.inner.op.output_height(input_h)
    }

    fn dyn_start(&self) -> Box<dyn std::any::Any + Send> {
        self.inner.dyn_start()
    }

    fn dyn_start_with_tile(&self, tile_w: u32, tile_h: u32) -> Box<dyn std::any::Any + Send> {
        self.inner.dyn_start_with_tile(tile_w, tile_h)
    }

    fn dyn_process_region(
        &self,
        state: &mut dyn std::any::Any,
        input: &[u8],
        output: &mut [u8],
        input_region: Region,
        output_region: Region,
    ) {
        self.inner
            .dyn_process_region(state, input, output, input_region, output_region);
    }
}

#[cfg(all(test, feature = "_integration"))]
mod tests {
    use super::*;
    use crate::resample::shrinkh::ShrinkSample;
    use proptest::prelude::*;
    use viprs_core::{
        format::{BandFormatId, I16, U8},
        image::{DemandHint, Region, Tile, TileMut},
        op::DynOperation,
    };
    use viprs_ports::source::ImageSource;
    use viprs_runtime::sources::memory::MemorySource;

    fn run_shrinkv<F>(
        factor: u32,
        ceil: bool,
        input_data: &[F::Sample],
        in_w: u32,
        in_h: u32,
        bands: u32,
    ) -> Vec<F::Sample>
    where
        F: BandFormat,
        F::Sample: ShrinkSample + bytemuck::Pod,
    {
        let op = ShrinkV::<F>::new_with_ceil(factor, ceil).unwrap();
        let out_h = op.output_height(in_h);
        let in_region = Region::new(0, 0, in_w, in_h);
        let out_region = Region::new(0, 0, in_w, out_h);
        let input = Tile::<F>::new(in_region, bands, input_data);
        let mut out_data =
            vec![F::Sample::from_f64_clamped(0.0); in_w as usize * out_h as usize * bands as usize];
        let mut output = TileMut::<F>::new(out_region, bands, &mut out_data);
        let mut state = op.start_with_tile_and_bands(out_region.width, out_region.height, bands);
        op.process_region(&mut state, &input, &mut output);
        out_data
    }

    fn shrinkv_u8_reference(
        input: &[u8],
        width: u32,
        height: u32,
        bands: u32,
        factor: u32,
    ) -> Vec<u8> {
        let row_len = width as usize * bands as usize;
        let out_h = height / factor;
        let multiplier = shrink_u8_multiplier(factor as usize);
        let amend = factor / 2;
        let mut output = vec![0u8; row_len * out_h as usize];

        for y_out in 0..out_h as usize {
            let src_base = y_out * factor as usize * row_len;
            let dst_base = y_out * row_len;
            for x in 0..row_len {
                let mut sum = amend;
                for k in 0..factor as usize {
                    sum += u32::from(input[src_base + k * row_len + x]);
                }
                output[dst_base + x] = shrink_u8_fixed_point_average(sum, multiplier);
            }
        }

        output
    }

    #[test]
    fn shrinkv_factor2_averages_row_pairs() {
        // Input: 2 rows, 2 pixels, 1 band
        // row0: [100, 100], row1: [200, 200]
        // Expected output: 1 row: [150, 150]
        let input = vec![100u8, 100u8, 200u8, 200u8];
        let output = run_shrinkv::<U8>(2, false, &input, 2, 2, 1);
        assert_eq!(output, vec![150u8, 150u8]);
    }

    #[test]
    fn shrinkv_factor1_is_identity() {
        let input = vec![10u8, 20u8, 30u8, 40u8];
        let output = run_shrinkv::<U8>(1, false, &input, 4, 1, 1);
        assert_eq!(output, input);
    }

    #[test]
    fn shrinkv_new_rejects_zero_factor_with_typed_error() {
        assert!(matches!(
            ShrinkV::<U8>::new(0),
            Err(BuildError::SourceHint {
                context: "shrink_v",
                ..
            })
        ));
    }

    #[test]
    fn shrinkv_output_height() {
        let op = ShrinkV::<U8>::new(2).unwrap();
        assert_eq!(op.output_height(100), 50);
        assert_eq!(op.output_height(5), 2); // integer division
    }

    #[test]
    fn shrinkv_output_height_with_ceil_matches_libvips() {
        let floor = ShrinkV::<U8>::new_with_ceil(3, false).unwrap();
        let ceil = ShrinkV::<U8>::new_with_ceil(3, true).unwrap();
        assert_eq!(floor.output_height(10), 3);
        assert_eq!(ceil.output_height(10), 4);
    }

    #[test]
    fn shrinkv_output_width_unchanged() {
        let op = ShrinkV::<U8>::new(4).unwrap();
        assert_eq!(op.output_width(200), 200);
    }

    #[test]
    fn shrinkv_required_input_region() {
        let op = ShrinkV::<U8>::new(2).unwrap();
        let out_region = Region::new(0, 0, 1, 10);
        let in_region = op.required_input_region(&out_region);
        assert_eq!(in_region.y, 0);
        assert_eq!(in_region.height, 20);
        assert_eq!(in_region.width, 1);
    }

    #[test]
    fn shrinkv_required_input_region_saturates_huge_factor() {
        let op = ShrinkV::<U8>::new(u32::MAX).unwrap();
        assert_eq!(
            op.required_input_region(&Region::new(0, 1, 1, 2)),
            Region::new(0, i32::MAX, 1, u32::MAX)
        );
    }

    #[test]
    fn shrinkv_node_spec() {
        let op = ShrinkV::<U8>::new(3).unwrap();
        let spec = op.node_spec(4, 9);
        assert_eq!(spec.input_tile_h, 27);
        assert_eq!(spec.output_tile_h, 9);
        assert_eq!(spec.input_tile_w, 4);
        assert_eq!(spec.output_tile_w, 4);
    }

    #[test]
    fn shrinkv_uniform_image_shrink2_shrink2_equals_shrink4() {
        // For a uniform image, shrink(2) then shrink(2) == shrink(4) because
        // all pixels are equal and averaging equal values is idempotent.
        let uniform = vec![64u8; 16]; // 1 column, 16 rows, 1 band

        // shrink by 4 directly
        let direct = run_shrinkv::<U8>(4, false, &uniform, 1, 16, 1);

        // shrink by 2 twice
        let step1 = run_shrinkv::<U8>(2, false, &uniform, 1, 16, 1);
        let step2 = run_shrinkv::<U8>(2, false, &step1, 1, 8, 1);

        assert_eq!(
            direct, step2,
            "shrink(4) must equal shrink(2) applied twice for a uniform image"
        );
    }

    #[test]
    fn shrinkv_bridge_exposes_dyn_operation_contract() {
        let bridge = ShrinkVBridge::<U8>::new(2, 1).unwrap();
        assert_eq!(bridge.input_format(), BandFormatId::U8);
        assert_eq!(bridge.output_format(), BandFormatId::U8);
        assert_eq!(bridge.bands(), 1);
        assert_eq!(bridge.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(bridge.output_width(1), 1);
        assert_eq!(bridge.output_height(4), 2);
        assert_eq!(
            bridge.node_spec(1, 2),
            ShrinkV::<U8>::new(2).unwrap().node_spec(1, 2)
        );

        let source = MemorySource::<U8>::new(1, 4, 1, vec![0, 10, 20, 30]).unwrap();
        let out_region = Region::new(0, 0, 1, 2);
        let input_region = bridge.required_input_region(&out_region);
        let mut input_bytes = vec![0u8; input_region.pixel_count()];
        source.read_region(input_region, &mut input_bytes).unwrap();
        let mut output_bytes = vec![0u8; out_region.pixel_count()];
        let mut state = bridge.dyn_start();
        bridge.dyn_process_region(
            &mut *state,
            &input_bytes,
            &mut output_bytes,
            input_region,
            out_region,
        );
        assert_eq!(output_bytes, vec![5, 25]);
    }

    #[test]
    fn shrinkv_bridge_with_ceil_exposes_dimension_round_up() {
        let bridge = ShrinkVBridge::<U8>::new_with_ceil(3, true, 1).unwrap();
        assert_eq!(bridge.output_height(10), 4);
    }

    #[test]
    fn shrinkv_bridge_rejects_zero_band_source_with_typed_error() {
        assert!(matches!(
            ShrinkVBridge::<U8>::new(2, 0),
            Err(BuildError::SourceHint {
                context: "shrink_v",
                ..
            })
        ));
    }

    #[test]
    fn shrinkv_signed_integer_rounding_matches_libvips_bias() {
        let input = vec![-2i16, -1i16];
        let output = run_shrinkv::<I16>(2, false, &input, 1, 2, 1);
        assert_eq!(output, vec![-1]);
    }

    #[test]
    fn shrinkv_factor19_matches_libvips_fixed_point_average() {
        let width = 6u32;
        let bands = 3u32;
        let factor = 19u32;
        let height = factor * 2;
        let input: Vec<u8> = (0..(width * height * bands))
            .map(|i| ((i * 37 + 11) & 0xff) as u8)
            .collect();

        let output = run_shrinkv::<U8>(factor, false, &input, width, height, bands);
        let expected = shrinkv_u8_reference(&input, width, height, bands, factor);

        assert_eq!(output, expected);
    }

    proptest! {
        #[test]
        fn shrinkv_factor1_is_identity_prop(
            (width, height, bands, pixels) in (1u32..=16, 1u32..=32, 1u32..=4).prop_flat_map(|(width, height, bands)| {
                (
                    Just(width),
                    Just(height),
                    Just(bands),
                    prop::collection::vec(any::<u8>(), (width * height * bands) as usize),
                )
            }),
        ) {
            let output = run_shrinkv::<U8>(1, false, &pixels, width, height, bands);
            prop_assert_eq!(output, pixels);
        }

        #[test]
        fn shrinkv_uniform_factor2_preserves_value(
            value in any::<u8>(),
            width in 1u32..=16,
            height in 1u32..=32,
            bands in 1u32..=4,
        ) {
            let input = vec![value; (width * height * 2 * bands) as usize];
            let output = run_shrinkv::<U8>(2, false, &input, width, height * 2, bands);
            prop_assert!(output.iter().all(|sample| *sample == value));
        }

        #[test]
        fn shrinkv_ceil_output_height_at_least_floor(
            height in 1u32..=1024,
            factor in 1u32..=32,
        ) {
            let floor = ShrinkV::<U8>::new_with_ceil(factor, false).unwrap().output_height(height);
            let ceil = ShrinkV::<U8>::new_with_ceil(factor, true).unwrap().output_height(height);
            prop_assert!(ceil >= floor);
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[test]
        fn shrinkv_avx2_matches_scalar_prop(
            (width, out_h, bands, factor, pixels) in (1u32..=16, 1u32..=8, 1u32..=4, 3u32..=19).prop_flat_map(
                |(width, out_h, bands, factor)| {
                    let height = out_h * factor;
                    (
                        Just(width),
                        Just(out_h),
                        Just(bands),
                        Just(factor),
                        prop::collection::vec(any::<u8>(), (width * height * bands) as usize),
                    )
                }
            ),
        ) {
            if !std::arch::is_x86_feature_detected!("avx2") {
                return Ok(());
            }

            let factor_usize = factor as usize;
            let out_h_usize = out_h as usize;
            let bands_usize = bands as usize;
            let width_usize = width as usize;
            let mut expected = vec![0u8; width_usize * out_h_usize * bands_usize];
            let mut actual = vec![0u8; width_usize * out_h_usize * bands_usize];
            let mut scalar_scratch = Vec::new();
            let mut avx2_scratch = Vec::new();

            shrink_v_u8_generic_scalar(
                factor_usize,
                &pixels,
                bands_usize,
                width_usize,
                out_h_usize,
                &mut scalar_scratch,
                &mut expected,
            );

            // SAFETY: the test gates execution on runtime AVX2 support and both
            // buffers are sized from the exact same geometry passed to the helper.
            unsafe {
                shrink_v_u8_generic_avx2(
                    factor_usize,
                    &pixels,
                    bands_usize,
                    width_usize,
                    out_h_usize,
                    &mut avx2_scratch,
                    &mut actual,
                );
            }

            prop_assert_eq!(actual, expected);
        }
    }
}

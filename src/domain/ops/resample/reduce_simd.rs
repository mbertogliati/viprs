#![allow(dead_code)]
// REASON: several specialized kernels remain available for architecture- and feature-specific dispatch.
#![allow(clippy::needless_range_loop, clippy::items_after_statements)]
// REASON: indexed loops and local helpers keep the SIMD gather paths close to the hardware layout.

use crate::domain::image::Region;

use super::{
    reduce_common::{ReduceKernel, clamp_axis},
    sample_conv::ReduceSample,
};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    int32x4_t, int64x2_t, vaddq_f64, vaddq_s32, vaddq_s64, vaddvq_f64, vaddvq_s32, vaddvq_s64,
    vcombine_s16, vcombine_s32, vcvt_f64_f32, vdup_n_s16, vdupq_n_f64, vdupq_n_s16, vdupq_n_s32,
    vdupq_n_s64, vget_high_s16, vget_high_s32, vget_high_u8, vget_high_u32, vget_low_s16,
    vget_low_s32, vget_low_u8, vget_low_u32, vld1_f32, vld1_s16, vld1_u8, vld1_u16, vld1q_f64,
    vld1q_s16, vld1q_u8, vld2_f32, vld2_u8, vld2_u16, vld3_f32, vld3_u8, vld3_u16, vld4_f32,
    vld4_u8, vld4_u16, vmlal_s16, vmlal_s32, vmovl_s16, vmovl_u8, vmovl_u16, vmulq_f64, vqmovn_s32,
    vqmovn_s64, vqmovun_s16, vqmovun_s32, vreinterpret_s32_u32, vreinterpretq_s16_u16, vshrq_n_s32,
    vshrq_n_s64, vst1_u8, vst1_u16, vst1q_f64,
};
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use std::arch::x86_64::*;

#[inline]
pub fn reduce_h_scalar<T: ReduceSample>(
    filter: &ReduceKernel,
    input_region: &Region,
    input: &[T],
    bands: u32,
    output_region: &Region,
    output: &mut [T],
) {
    let out_w = output_region.width as usize;
    let out_h = output_region.height as usize;
    let in_w = input_region.width as usize;
    let bands = bands as usize;

    for y in 0..out_h {
        for x_out in 0..out_w {
            let source_x = filter.source_position(f64::from(output_region.x) + x_out as f64);
            if T::USE_FIXED_POINT {
                let (start_x, weights) = filter.taps_for_i16(source_x);
                for band in 0..bands {
                    let mut acc = 0_i64;
                    for (tap, weight) in weights.iter().copied().enumerate() {
                        let tile_x = clamp_axis(start_x + tap as i64, input_region.x, in_w);
                        let idx = (y * in_w + tile_x) * bands + band;
                        acc += i64::from(weight) * input[idx].to_i64();
                    }

                    let out_idx = (y * out_w + x_out) * bands + band;
                    output[out_idx] = T::from_fixed_i64(acc);
                }
            } else {
                let (start_x, weights) = filter.taps_for_f64(source_x);
                for band in 0..bands {
                    let mut acc = 0.0;
                    for (tap, weight) in weights.iter().copied().enumerate() {
                        let tile_x = clamp_axis(start_x + tap as i64, input_region.x, in_w);
                        let idx = (y * in_w + tile_x) * bands + band;
                        acc = input[idx].to_f64().mul_add(weight, acc);
                    }

                    let out_idx = (y * out_w + x_out) * bands + band;
                    output[out_idx] = T::from_f64(acc);
                }
            }
        }
    }
}

#[inline]
pub fn reduce_v_scalar<T: ReduceSample>(
    filter: &ReduceKernel,
    input_region: &Region,
    input: &[T],
    bands: u32,
    output_region: &Region,
    output: &mut [T],
) {
    let out_w = output_region.width as usize;
    let out_h = output_region.height as usize;
    let in_w = input_region.width as usize;
    let in_h = input_region.height as usize;
    let bands = bands as usize;

    for y_out in 0..out_h {
        let source_y = filter.source_position(f64::from(output_region.y) + y_out as f64);
        if T::USE_FIXED_POINT {
            let (start_y, weights) = filter.taps_for_i16(source_y);
            for x in 0..out_w {
                for band in 0..bands {
                    let mut acc = 0_i64;
                    for (tap, weight) in weights.iter().copied().enumerate() {
                        let tile_y = clamp_axis(start_y + tap as i64, input_region.y, in_h);
                        let idx = (tile_y * in_w + x) * bands + band;
                        acc += i64::from(weight) * input[idx].to_i64();
                    }

                    let out_idx = (y_out * out_w + x) * bands + band;
                    output[out_idx] = T::from_fixed_i64(acc);
                }
            }
        } else {
            let (start_y, weights) = filter.taps_for_f64(source_y);
            for x in 0..out_w {
                for band in 0..bands {
                    let mut acc = 0.0;
                    for (tap, weight) in weights.iter().copied().enumerate() {
                        let tile_y = clamp_axis(start_y + tap as i64, input_region.y, in_h);
                        let idx = (tile_y * in_w + x) * bands + band;
                        acc = input[idx].to_f64().mul_add(weight, acc);
                    }

                    let out_idx = (y_out * out_w + x) * bands + band;
                    output[out_idx] = T::from_f64(acc);
                }
            }
        }
    }
}

#[allow(unreachable_code)]
#[inline]
pub fn reduce_h_u8(
    filter: &ReduceKernel,
    input_region: &Region,
    input: &[u8],
    bands: u32,
    output_region: &Region,
    output: &mut [u8],
) {
    let out_w = output_region.width as usize;
    let mut starts = vec![0_i64; out_w];
    let mut phases = vec![0_u8; out_w];
    for x_out in 0..out_w {
        let source_x = filter.source_position(f64::from(output_region.x) + x_out as f64);
        let (start_x, phase) = filter.plan_i16(source_x);
        starts[x_out] = start_x;
        phases[x_out] = phase as u8;
    }
    reduce_h_u8_planned(
        filter,
        input_region,
        input,
        bands,
        output_region,
        output,
        &starts,
        &phases,
    );
}

#[allow(unreachable_code)]
#[inline]
pub fn reduce_h_u8_planned(
    filter: &ReduceKernel,
    input_region: &Region,
    input: &[u8],
    bands: u32,
    output_region: &Region,
    output: &mut [u8],
    starts: &[i64],
    phases: &[u8],
) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: aarch64 guarantees NEON and the helper only reads within the provided slices.
        unsafe {
            return reduce_h_u8_neon(
                filter,
                input_region,
                input,
                bands,
                output_region,
                output,
                starts,
                phases,
            );
        };
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: the helper is only compiled when AVX2 is enabled for the current target.
        unsafe {
            return reduce_h_u8_avx2(
                filter,
                input_region,
                input,
                bands,
                output_region,
                output,
                starts,
                phases,
            );
        };
    }
    reduce_h_u8_scalar_planned(
        filter,
        input_region,
        input,
        bands,
        output_region,
        output,
        starts,
        phases,
    );
}

#[allow(unreachable_code)]
#[inline]
pub fn reduce_h_u16(
    filter: &ReduceKernel,
    input_region: &Region,
    input: &[u16],
    bands: u32,
    output_region: &Region,
    output: &mut [u16],
) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: aarch64 guarantees NEON and the helper only reads within the provided slices.
        unsafe {
            return reduce_h_u16_neon(filter, input_region, input, bands, output_region, output);
        };
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: the helper is only compiled when AVX2 is enabled for the current target.
        unsafe {
            return reduce_h_u16_avx2(filter, input_region, input, bands, output_region, output);
        };
    }
    reduce_h_scalar(filter, input_region, input, bands, output_region, output);
}

#[allow(unreachable_code)]
#[inline]
pub fn reduce_h_f32(
    filter: &ReduceKernel,
    input_region: &Region,
    input: &[f32],
    bands: u32,
    output_region: &Region,
    output: &mut [f32],
) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: aarch64 guarantees NEON and the helper only reads within the provided slices.
        unsafe {
            return reduce_h_f32_neon(filter, input_region, input, bands, output_region, output);
        };
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: the helper is only compiled when AVX2 is enabled for the current target.
        unsafe {
            return reduce_h_f32_avx2(filter, input_region, input, bands, output_region, output);
        };
    }
    reduce_h_scalar(filter, input_region, input, bands, output_region, output);
}

#[allow(unreachable_code)]
#[inline]
pub fn reduce_v_u8(
    filter: &ReduceKernel,
    input_region: &Region,
    input: &[u8],
    bands: u32,
    output_region: &Region,
    starts: &[i64],
    phases: &[u8],
    output: &mut [u8],
) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: aarch64 guarantees NEON and the helper only reads within the provided slices.
        unsafe {
            return reduce_v_u8_neon(
                filter,
                input_region,
                input,
                bands,
                output_region,
                starts,
                phases,
                output,
            );
        };
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: the helper is only compiled when AVX2 is enabled for the current target.
        unsafe {
            return reduce_v_u8_avx2(filter, input_region, input, bands, output_region, output);
        };
    }
    reduce_v_u8_scalar_planned(
        filter,
        input_region,
        input,
        bands,
        output_region,
        starts,
        phases,
        output,
    );
}

#[allow(unreachable_code)]
#[inline]
pub fn reduce_v_u16(
    filter: &ReduceKernel,
    input_region: &Region,
    input: &[u16],
    bands: u32,
    output_region: &Region,
    output: &mut [u16],
) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: aarch64 guarantees NEON and the helper only reads within the provided slices.
        unsafe {
            return reduce_v_u16_neon(filter, input_region, input, bands, output_region, output);
        };
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: the helper is only compiled when AVX2 is enabled for the current target.
        unsafe {
            return reduce_v_u16_avx2(filter, input_region, input, bands, output_region, output);
        };
    }
    reduce_v_scalar(filter, input_region, input, bands, output_region, output);
}

#[allow(unreachable_code)]
#[inline]
pub fn reduce_v_f32(
    filter: &ReduceKernel,
    input_region: &Region,
    input: &[f32],
    bands: u32,
    output_region: &Region,
    output: &mut [f32],
) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: aarch64 guarantees NEON and the helper only reads within the provided slices.
        unsafe {
            return reduce_v_f32_neon(filter, input_region, input, bands, output_region, output);
        };
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: the helper is only compiled when AVX2 is enabled for the current target.
        unsafe {
            return reduce_v_f32_avx2(filter, input_region, input, bands, output_region, output);
        };
    }
    reduce_v_scalar(filter, input_region, input, bands, output_region, output);
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
// SAFETY: caller must execute this only on NEON-capable aarch64 and provide row-major `input`/`output` buffers that cover `input_region` and `output_region`, with `starts` and `phases` indexed for every output column.
unsafe fn reduce_h_u8_neon(
    filter: &ReduceKernel,
    input_region: &Region,
    input: &[u8],
    bands: u32,
    output_region: &Region,
    output: &mut [u8],
    starts: &[i64],
    phases: &[u8],
) {
    let out_w = output_region.width as usize;
    let out_h = output_region.height as usize;
    let in_w = input_region.width as usize;
    let bands = bands as usize;
    let in_row_stride = in_w * bands;
    let out_row_stride = out_w * bands;
    let exact2 = (filter.config().factor - 2.0).abs() < f64::EPSILON;
    match bands {
        1 => {
            let input_end = i64::from(input_region.x) + in_w as i64;
            for y in 0..out_h {
                let in_row = &input[y * in_row_stride..(y + 1) * in_row_stride];
                let out_row = &mut output[y * out_row_stride..(y + 1) * out_row_stride];
                let mut x_out = 0usize;
                while x_out < out_w {
                    let start_x = starts[x_out];
                    let weights = filter.coeffs_i16_for_phase(phases[x_out] as usize);
                    if exact2
                        && x_out + 1 < out_w
                        && phases[x_out + 1] == phases[x_out]
                        && starts[x_out + 1] == start_x + 2
                        && start_x >= i64::from(input_region.x)
                        && start_x + (weights.len() as i64) < input_end
                    {
                        let pixel_offset = (start_x - i64::from(input_region.x)) as usize;
                        let (left, right) = out_row[x_out..x_out + 2].split_at_mut(1);
                        // SAFETY: the exact-2 pair window is fully in-bounds for both output pixels.
                        unsafe {
                            reduce_h_u8_neon_pair_exact2_1(
                                in_row,
                                pixel_offset,
                                weights,
                                &mut left[0],
                                &mut right[0],
                            );
                        };
                        x_out += 2;
                        continue;
                    }

                    out_row[x_out] = if start_x >= i64::from(input_region.x)
                        && start_x + weights.len() as i64 <= input_end
                    {
                        let pixel_offset = (start_x - i64::from(input_region.x)) as usize;
                        // SAFETY: the full tap window is in-bounds for this row.
                        unsafe { reduce_h_u8_neon_pixel_interior(in_row, pixel_offset, weights) }
                    } else {
                        // SAFETY: the helper only gathers from `in_row` with clamped indices.
                        unsafe { reduce_h_u8_neon_pixel(in_row, input_region.x, start_x, weights) }
                    };
                    x_out += 1;
                }
            }
        }
        3 => {
            let input_end = i64::from(input_region.x) + in_w as i64;
            for y in 0..out_h {
                let in_row = &input[y * in_row_stride..(y + 1) * in_row_stride];
                let out_row = &mut output[y * out_row_stride..(y + 1) * out_row_stride];
                let mut x_out = 0usize;
                while x_out < out_w {
                    let start_x = starts[x_out];
                    let weights = filter.coeffs_i16_for_phase(phases[x_out] as usize);
                    if exact2
                        && x_out + 1 < out_w
                        && phases[x_out + 1] == phases[x_out]
                        && starts[x_out + 1] == start_x + 2
                        && start_x >= i64::from(input_region.x)
                        && start_x + (weights.len() as i64) < input_end
                    {
                        let pixel_offset = (start_x - i64::from(input_region.x)) as usize * 3;
                        let out_idx = x_out * 3;
                        let (head, tail) = out_row[out_idx..out_idx + 6].split_at_mut(3);
                        // SAFETY: the exact-2 pair window is fully in-bounds for both RGB output pixels.
                        unsafe {
                            reduce_h_u8_neon_pair_exact2_3(
                                in_row,
                                pixel_offset,
                                weights,
                                head,
                                tail,
                            );
                        };
                        x_out += 2;
                        continue;
                    }

                    let rgb = if start_x >= i64::from(input_region.x)
                        && start_x + weights.len() as i64 <= input_end
                    {
                        let pixel_offset = (start_x - i64::from(input_region.x)) as usize * 3;
                        // SAFETY: the full RGB tap window is in-bounds for this row.
                        unsafe {
                            reduce_h_u8_neon_pixel_rgb_interior(in_row, pixel_offset, weights)
                        }
                    } else {
                        // SAFETY: the helper only reads the current row through clamped pixel indices.
                        unsafe {
                            reduce_h_u8_neon_pixel_rgb(
                                in_row,
                                input_region.x,
                                in_w,
                                start_x,
                                weights,
                            )
                        }
                    };
                    let out_idx = x_out * 3;
                    out_row[out_idx] = rgb[0];
                    out_row[out_idx + 1] = rgb[1];
                    out_row[out_idx + 2] = rgb[2];
                    x_out += 1;
                }
            }
        }
        4 => {
            let input_end = i64::from(input_region.x) + in_w as i64;
            for y in 0..out_h {
                let in_row = &input[y * in_row_stride..(y + 1) * in_row_stride];
                let out_row = &mut output[y * out_row_stride..(y + 1) * out_row_stride];
                let mut x_out = 0usize;
                while x_out < out_w {
                    let start_x = starts[x_out];
                    let weights = filter.coeffs_i16_for_phase(phases[x_out] as usize);
                    if exact2
                        && x_out + 1 < out_w
                        && phases[x_out + 1] == phases[x_out]
                        && starts[x_out + 1] == start_x + 2
                        && start_x >= i64::from(input_region.x)
                        && start_x + (weights.len() as i64) < input_end
                    {
                        let pixel_offset = (start_x - i64::from(input_region.x)) as usize * 4;
                        let out_idx = x_out * 4;
                        let (head, tail) = out_row[out_idx..out_idx + 8].split_at_mut(4);
                        // SAFETY: the exact-2 pair window is fully in-bounds for both RGBA output pixels.
                        unsafe {
                            reduce_h_u8_neon_pair_exact2_4(
                                in_row,
                                pixel_offset,
                                weights,
                                head,
                                tail,
                            );
                        };
                        x_out += 2;
                        continue;
                    }

                    let rgba = if start_x >= i64::from(input_region.x)
                        && start_x + weights.len() as i64 <= input_end
                    {
                        let pixel_offset = (start_x - i64::from(input_region.x)) as usize * 4;
                        // SAFETY: the full RGBA tap window is in-bounds for this row.
                        unsafe {
                            reduce_h_u8_neon_pixel_rgba_interior(in_row, pixel_offset, weights)
                        }
                    } else {
                        // SAFETY: the helper only reads the current row through clamped pixel indices.
                        unsafe {
                            reduce_h_u8_neon_pixel_rgba(
                                in_row,
                                input_region.x,
                                in_w,
                                start_x,
                                weights,
                            )
                        }
                    };
                    let out_idx = x_out * 4;
                    out_row[out_idx] = rgba[0];
                    out_row[out_idx + 1] = rgba[1];
                    out_row[out_idx + 2] = rgba[2];
                    out_row[out_idx + 3] = rgba[3];
                    x_out += 1;
                }
            }
        }
        _ => {
            for y in 0..out_h {
                let in_row = &input[y * in_row_stride..(y + 1) * in_row_stride];
                let out_row = &mut output[y * out_row_stride..(y + 1) * out_row_stride];
                for x_out in 0..out_w {
                    let start_x = starts[x_out];
                    let weights = filter.coeffs_i16_for_phase(phases[x_out] as usize);
                    let out_pixel = &mut out_row[x_out * bands..(x_out + 1) * bands];
                    for (band, out_sample) in out_pixel.iter_mut().enumerate() {
                        // SAFETY: the helper only gathers from `in_row` with clamped pixel indices.
                        *out_sample = unsafe {
                            reduce_h_u8_neon_pixel_nb(
                                in_row,
                                input_region.x,
                                in_w,
                                bands,
                                band,
                                start_x,
                                weights,
                            )
                        };
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn reduce_h_u8_neon_pair_exact2_1(
    row: &[u8],
    pixel_offset: usize,
    weights: &[i16],
    out_0: &mut u8,
    out_1: &mut u8,
) {
    let pair_taps = weights.len() + 2;
    let mut acc_0 = 0_i64;
    let mut acc_1 = 0_i64;
    let mut src = 0usize;
    let mut weight_buf_0 = [0_i16; 8];
    let mut weight_buf_1 = [0_i16; 8];
    // SAFETY: the zero vectors are valid NEON values with no memory access.
    let mut acc0_lo = unsafe { vdupq_n_s32(0) };
    // SAFETY: the zero vectors are valid NEON values with no memory access.
    let mut acc0_hi = unsafe { vdupq_n_s32(0) };
    // SAFETY: the zero vectors are valid NEON values with no memory access.
    let mut acc1_lo = unsafe { vdupq_n_s32(0) };
    // SAFETY: the zero vectors are valid NEON values with no memory access.
    let mut acc1_hi = unsafe { vdupq_n_s32(0) };

    while src + 8 <= pair_taps {
        for lane in 0..8 {
            let idx = src + lane;
            weight_buf_0[lane] = weights.get(idx).copied().unwrap_or(0);
            weight_buf_1[lane] = if idx >= 2 {
                weights.get(idx - 2).copied().unwrap_or(0)
            } else {
                0
            };
        }

        // SAFETY: the caller guarantees the exact-2 pair window is in-bounds for all lanes.
        unsafe {
            let sample_vec = vld1_u8(row.as_ptr().add(pixel_offset + src));
            let sample_vec = vreinterpretq_s16_u16(vmovl_u8(sample_vec));
            let weight_vec_0 = vld1q_s16(weight_buf_0.as_ptr());
            let weight_vec_1 = vld1q_s16(weight_buf_1.as_ptr());
            acc0_lo = vmlal_s16(
                acc0_lo,
                vget_low_s16(sample_vec),
                vget_low_s16(weight_vec_0),
            );
            acc0_hi = vmlal_s16(
                acc0_hi,
                vget_high_s16(sample_vec),
                vget_high_s16(weight_vec_0),
            );
            acc1_lo = vmlal_s16(
                acc1_lo,
                vget_low_s16(sample_vec),
                vget_low_s16(weight_vec_1),
            );
            acc1_hi = vmlal_s16(
                acc1_hi,
                vget_high_s16(sample_vec),
                vget_high_s16(weight_vec_1),
            );
        }
        src += 8;
    }

    // SAFETY: horizontal adds only read the accumulator lanes.
    unsafe {
        acc_0 += i64::from(vaddvq_s32(acc0_lo));
        acc_0 += i64::from(vaddvq_s32(acc0_hi));
        acc_1 += i64::from(vaddvq_s32(acc1_lo));
        acc_1 += i64::from(vaddvq_s32(acc1_hi));
    }

    while src < pair_taps {
        let sample = i64::from(row[pixel_offset + src]);
        if src < weights.len() {
            acc_0 += i64::from(weights[src]) * sample;
        }
        if src >= 2 {
            acc_1 += i64::from(weights[src - 2]) * sample;
        }
        src += 1;
    }
    *out_0 = <u8 as ReduceSample>::from_fixed_i64(acc_0);
    *out_1 = <u8 as ReduceSample>::from_fixed_i64(acc_1);
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn reduce_h_u8_neon_pair_exact2_3(
    row: &[u8],
    pixel_offset: usize,
    weights: &[i16],
    out_0: &mut [u8],
    out_1: &mut [u8],
) {
    let pair_taps = weights.len() + 2;
    let mut acc_0 = [0_i64; 3];
    let mut acc_1 = [0_i64; 3];
    let mut src = 0usize;
    let mut weight_buf_0 = [0_i16; 8];
    let mut weight_buf_1 = [0_i16; 8];
    // SAFETY: the zero vectors are valid NEON values with no memory access.
    let mut acc0_lo = [unsafe { vdupq_n_s32(0) }; 3];
    // SAFETY: the zero vectors are valid NEON values with no memory access.
    let mut acc0_hi = [unsafe { vdupq_n_s32(0) }; 3];
    // SAFETY: the zero vectors are valid NEON values with no memory access.
    let mut acc1_lo = [unsafe { vdupq_n_s32(0) }; 3];
    // SAFETY: the zero vectors are valid NEON values with no memory access.
    let mut acc1_hi = [unsafe { vdupq_n_s32(0) }; 3];

    while src + 8 <= pair_taps {
        for lane in 0..8 {
            let idx = src + lane;
            weight_buf_0[lane] = weights.get(idx).copied().unwrap_or(0);
            weight_buf_1[lane] = if idx >= 2 {
                weights.get(idx - 2).copied().unwrap_or(0)
            } else {
                0
            };
        }

        // SAFETY: the caller guarantees the exact-2 pair RGB window is in-bounds for all lanes.
        unsafe {
            let sample_vecs = vld3_u8(row.as_ptr().add(pixel_offset + src * 3));
            let sample_vecs = [sample_vecs.0, sample_vecs.1, sample_vecs.2];
            let weight_vec_0 = vld1q_s16(weight_buf_0.as_ptr());
            let weight_vec_1 = vld1q_s16(weight_buf_1.as_ptr());
            for channel in 0..3 {
                let sample_vec = vreinterpretq_s16_u16(vmovl_u8(sample_vecs[channel]));
                acc0_lo[channel] = vmlal_s16(
                    acc0_lo[channel],
                    vget_low_s16(sample_vec),
                    vget_low_s16(weight_vec_0),
                );
                acc0_hi[channel] = vmlal_s16(
                    acc0_hi[channel],
                    vget_high_s16(sample_vec),
                    vget_high_s16(weight_vec_0),
                );
                acc1_lo[channel] = vmlal_s16(
                    acc1_lo[channel],
                    vget_low_s16(sample_vec),
                    vget_low_s16(weight_vec_1),
                );
                acc1_hi[channel] = vmlal_s16(
                    acc1_hi[channel],
                    vget_high_s16(sample_vec),
                    vget_high_s16(weight_vec_1),
                );
            }
        }
        src += 8;
    }

    // SAFETY: horizontal adds only read the accumulator lanes.
    unsafe {
        for channel in 0..3 {
            acc_0[channel] += i64::from(vaddvq_s32(acc0_lo[channel]));
            acc_0[channel] += i64::from(vaddvq_s32(acc0_hi[channel]));
            acc_1[channel] += i64::from(vaddvq_s32(acc1_lo[channel]));
            acc_1[channel] += i64::from(vaddvq_s32(acc1_hi[channel]));
        }
    }

    while src < pair_taps {
        let base = pixel_offset + src * 3;
        let samples = [
            i64::from(row[base]),
            i64::from(row[base + 1]),
            i64::from(row[base + 2]),
        ];
        if src < weights.len() {
            let weight = i64::from(weights[src]);
            acc_0[0] += weight * samples[0];
            acc_0[1] += weight * samples[1];
            acc_0[2] += weight * samples[2];
        }
        if src >= 2 {
            let weight = i64::from(weights[src - 2]);
            acc_1[0] += weight * samples[0];
            acc_1[1] += weight * samples[1];
            acc_1[2] += weight * samples[2];
        }
        src += 1;
    }
    out_0[0] = <u8 as ReduceSample>::from_fixed_i64(acc_0[0]);
    out_0[1] = <u8 as ReduceSample>::from_fixed_i64(acc_0[1]);
    out_0[2] = <u8 as ReduceSample>::from_fixed_i64(acc_0[2]);
    out_1[0] = <u8 as ReduceSample>::from_fixed_i64(acc_1[0]);
    out_1[1] = <u8 as ReduceSample>::from_fixed_i64(acc_1[1]);
    out_1[2] = <u8 as ReduceSample>::from_fixed_i64(acc_1[2]);
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn reduce_h_u8_neon_pair_exact2_4(
    row: &[u8],
    pixel_offset: usize,
    weights: &[i16],
    out_0: &mut [u8],
    out_1: &mut [u8],
) {
    let pair_taps = weights.len() + 2;
    let mut acc_0 = [0_i64; 4];
    let mut acc_1 = [0_i64; 4];
    let mut src = 0usize;
    let mut weight_buf_0 = [0_i16; 8];
    let mut weight_buf_1 = [0_i16; 8];
    // SAFETY: the zero vectors are valid NEON values with no memory access.
    let mut acc0_lo = [unsafe { vdupq_n_s32(0) }; 4];
    // SAFETY: the zero vectors are valid NEON values with no memory access.
    let mut acc0_hi = [unsafe { vdupq_n_s32(0) }; 4];
    // SAFETY: the zero vectors are valid NEON values with no memory access.
    let mut acc1_lo = [unsafe { vdupq_n_s32(0) }; 4];
    // SAFETY: the zero vectors are valid NEON values with no memory access.
    let mut acc1_hi = [unsafe { vdupq_n_s32(0) }; 4];

    while src + 8 <= pair_taps {
        for lane in 0..8 {
            let idx = src + lane;
            weight_buf_0[lane] = weights.get(idx).copied().unwrap_or(0);
            weight_buf_1[lane] = if idx >= 2 {
                weights.get(idx - 2).copied().unwrap_or(0)
            } else {
                0
            };
        }

        // SAFETY: the caller guarantees the exact-2 pair RGBA window is in-bounds for all lanes.
        unsafe {
            let sample_vecs = vld4_u8(row.as_ptr().add(pixel_offset + src * 4));
            let sample_vecs = [sample_vecs.0, sample_vecs.1, sample_vecs.2, sample_vecs.3];
            let weight_vec_0 = vld1q_s16(weight_buf_0.as_ptr());
            let weight_vec_1 = vld1q_s16(weight_buf_1.as_ptr());
            for channel in 0..4 {
                let sample_vec = vreinterpretq_s16_u16(vmovl_u8(sample_vecs[channel]));
                acc0_lo[channel] = vmlal_s16(
                    acc0_lo[channel],
                    vget_low_s16(sample_vec),
                    vget_low_s16(weight_vec_0),
                );
                acc0_hi[channel] = vmlal_s16(
                    acc0_hi[channel],
                    vget_high_s16(sample_vec),
                    vget_high_s16(weight_vec_0),
                );
                acc1_lo[channel] = vmlal_s16(
                    acc1_lo[channel],
                    vget_low_s16(sample_vec),
                    vget_low_s16(weight_vec_1),
                );
                acc1_hi[channel] = vmlal_s16(
                    acc1_hi[channel],
                    vget_high_s16(sample_vec),
                    vget_high_s16(weight_vec_1),
                );
            }
        }
        src += 8;
    }

    // SAFETY: horizontal adds only read the accumulator lanes.
    unsafe {
        for channel in 0..4 {
            acc_0[channel] += i64::from(vaddvq_s32(acc0_lo[channel]));
            acc_0[channel] += i64::from(vaddvq_s32(acc0_hi[channel]));
            acc_1[channel] += i64::from(vaddvq_s32(acc1_lo[channel]));
            acc_1[channel] += i64::from(vaddvq_s32(acc1_hi[channel]));
        }
    }

    while src < pair_taps {
        let base = pixel_offset + src * 4;
        let samples = [
            i64::from(row[base]),
            i64::from(row[base + 1]),
            i64::from(row[base + 2]),
            i64::from(row[base + 3]),
        ];
        if src < weights.len() {
            let weight = i64::from(weights[src]);
            acc_0[0] += weight * samples[0];
            acc_0[1] += weight * samples[1];
            acc_0[2] += weight * samples[2];
            acc_0[3] += weight * samples[3];
        }
        if src >= 2 {
            let weight = i64::from(weights[src - 2]);
            acc_1[0] += weight * samples[0];
            acc_1[1] += weight * samples[1];
            acc_1[2] += weight * samples[2];
            acc_1[3] += weight * samples[3];
        }
        src += 1;
    }
    out_0[0] = <u8 as ReduceSample>::from_fixed_i64(acc_0[0]);
    out_0[1] = <u8 as ReduceSample>::from_fixed_i64(acc_0[1]);
    out_0[2] = <u8 as ReduceSample>::from_fixed_i64(acc_0[2]);
    out_0[3] = <u8 as ReduceSample>::from_fixed_i64(acc_0[3]);
    out_1[0] = <u8 as ReduceSample>::from_fixed_i64(acc_1[0]);
    out_1[1] = <u8 as ReduceSample>::from_fixed_i64(acc_1[1]);
    out_1[2] = <u8 as ReduceSample>::from_fixed_i64(acc_1[2]);
    out_1[3] = <u8 as ReduceSample>::from_fixed_i64(acc_1[3]);
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
// SAFETY: caller must execute this only on NEON-capable aarch64 and pass a `dst` pointer to at least 8 writable `u8` lanes.
unsafe fn store_rounded_u8x8(dst: *mut u8, acc_lo: int32x4_t, acc_hi: int32x4_t) {
    // SAFETY: constructing constant vectors does not access memory.
    let round = unsafe { vdupq_n_s32(1 << 11) };
    // SAFETY: all operations are lane-wise on live NEON values and `dst` points to 8 writable bytes.
    unsafe {
        let rounded_lo = vshrq_n_s32(vaddq_s32(acc_lo, round), 12);
        let rounded_hi = vshrq_n_s32(vaddq_s32(acc_hi, round), 12);
        let packed_s16 = vcombine_s16(vqmovn_s32(rounded_lo), vqmovn_s32(rounded_hi));
        let packed_u8 = vqmovun_s16(packed_s16);
        vst1_u8(dst, packed_u8);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn store_rounded_u8x16(
    dst: *mut u8,
    acc0_lo: int32x4_t,
    acc0_hi: int32x4_t,
    acc1_lo: int32x4_t,
    acc1_hi: int32x4_t,
) {
    // SAFETY: `dst` points to 16 writable bytes and each half writes a disjoint 8-byte lane group.
    unsafe {
        store_rounded_u8x8(dst, acc0_lo, acc0_hi);
        store_rounded_u8x8(dst.add(8), acc1_lo, acc1_hi);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
// SAFETY: caller must execute this only on NEON-capable aarch64 and pass a `dst` pointer to at least 4 writable `u16` lanes.
unsafe fn store_rounded_u16x4(dst: *mut u16, acc_lo: int64x2_t, acc_hi: int64x2_t) {
    // SAFETY: constructing constant vectors does not access memory.
    let round = unsafe { vdupq_n_s64(1 << 11) };
    // SAFETY: all operations are lane-wise on live NEON values and `dst` points to 4 writable u16s.
    unsafe {
        let rounded_lo = vshrq_n_s64(vaddq_s64(acc_lo, round), 12);
        let rounded_hi = vshrq_n_s64(vaddq_s64(acc_hi, round), 12);
        let packed_s32 = vcombine_s32(vqmovn_s64(rounded_lo), vqmovn_s64(rounded_hi));
        let packed_u16 = vqmovun_s32(packed_s32);
        vst1_u16(dst, packed_u16);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
// SAFETY: caller must execute this only on NEON-capable aarch64; the function either reads an in-bounds 8-tap window or gathers through clamped indices within `row`.
unsafe fn reduce_h_u8_neon_pixel(
    row: &[u8],
    input_origin: i32,
    start_x: i64,
    weights: &[i16],
) -> u8 {
    let mut acc = 0_i64;
    let mut tap = 0usize;
    let input_end = i64::from(input_origin) + row.len() as i64;
    // SAFETY: the zero vector is a valid NEON value with no memory access.
    let mut acc_lo = unsafe { vdupq_n_s32(0) };
    // SAFETY: the zero vector is a valid NEON value with no memory access.
    let mut acc_hi = unsafe { vdupq_n_s32(0) };

    while tap + 8 <= weights.len() {
        let first_x = start_x + tap as i64;
        let last_x = first_x + 7;
        let mut samples = [0_u8; 8];
        let sample_vec = if first_x >= i64::from(input_origin) && last_x < input_end {
            let pixel_offset = (first_x - i64::from(input_origin)) as usize;
            // SAFETY: the 8-tap window is fully in bounds, so this reads 8 contiguous pixels.
            unsafe { vld1_u8(row.as_ptr().add(pixel_offset)) }
        } else {
            for lane in 0..8 {
                let tile_x = clamp_axis(start_x + (tap + lane) as i64, input_origin, row.len());
                samples[lane] = row[tile_x];
            }
            // SAFETY: `samples` is an 8-lane stack buffer.
            unsafe { vld1_u8(samples.as_ptr()) }
        };

        // SAFETY: `sample_vec` and `weights[tap..tap + 8]` have exactly 8 lanes.
        unsafe {
            let sample_vec = vreinterpretq_s16_u16(vmovl_u8(sample_vec));
            let weight_vec = vld1q_s16(weights.as_ptr().add(tap));
            acc_lo = vmlal_s16(acc_lo, vget_low_s16(sample_vec), vget_low_s16(weight_vec));
            acc_hi = vmlal_s16(acc_hi, vget_high_s16(sample_vec), vget_high_s16(weight_vec));
        }
        tap += 8;
    }

    // SAFETY: horizontal adds only read the accumulator lanes.
    unsafe {
        acc += i64::from(vaddvq_s32(acc_lo));
        acc += i64::from(vaddvq_s32(acc_hi));
    }

    while tap < weights.len() {
        let tile_x = clamp_axis(start_x + tap as i64, input_origin, row.len());
        acc += i64::from(weights[tap]) * i64::from(row[tile_x]);
        tap += 1;
    }

    <u8 as ReduceSample>::from_fixed_i64(acc)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
// SAFETY: caller must execute this only on NEON-capable aarch64 and guarantee `pixel_offset + weights.len() <= row.len()` for the single-band tap window.
unsafe fn reduce_h_u8_neon_pixel_interior(row: &[u8], pixel_offset: usize, weights: &[i16]) -> u8 {
    let mut acc = 0_i64;
    let mut tap = 0usize;
    // SAFETY: the zero vector is a valid NEON value with no memory access.
    let mut acc_lo = unsafe { vdupq_n_s32(0) };
    // SAFETY: the zero vector is a valid NEON value with no memory access.
    let mut acc_hi = unsafe { vdupq_n_s32(0) };

    while tap + 8 <= weights.len() {
        // SAFETY: the caller guarantees the full tap window is in-bounds.
        unsafe {
            let sample_vec = vld1_u8(row.as_ptr().add(pixel_offset + tap));
            let sample_vec = vreinterpretq_s16_u16(vmovl_u8(sample_vec));
            let weight_vec = vld1q_s16(weights.as_ptr().add(tap));
            acc_lo = vmlal_s16(acc_lo, vget_low_s16(sample_vec), vget_low_s16(weight_vec));
            acc_hi = vmlal_s16(acc_hi, vget_high_s16(sample_vec), vget_high_s16(weight_vec));
        }
        tap += 8;
    }

    // SAFETY: horizontal adds only read the accumulator lanes.
    unsafe {
        acc += i64::from(vaddvq_s32(acc_lo));
        acc += i64::from(vaddvq_s32(acc_hi));
    }

    while tap < weights.len() {
        acc += i64::from(weights[tap]) * i64::from(row[pixel_offset + tap]);
        tap += 1;
    }

    <u8 as ReduceSample>::from_fixed_i64(acc)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
// SAFETY: caller must execute this only on NEON-capable aarch64; the function either reads an in-bounds 8-pixel RGB window or gathers through clamped pixel indices within `row`.
unsafe fn reduce_h_u8_neon_pixel_rgb(
    row: &[u8],
    input_origin: i32,
    input_width: usize,
    start_x: i64,
    weights: &[i16],
) -> [u8; 3] {
    let mut acc = [0_i64; 3];
    let mut tap = 0usize;
    let input_end = i64::from(input_origin) + input_width as i64;
    // SAFETY: the zero vectors are valid NEON values with no memory access.
    let mut acc_lo = [unsafe { vdupq_n_s32(0) }; 3];
    // SAFETY: the zero vectors are valid NEON values with no memory access.
    let mut acc_hi = [unsafe { vdupq_n_s32(0) }; 3];

    while tap + 8 <= weights.len() {
        let first_x = start_x + tap as i64;
        let last_x = first_x + 7;
        let mut gather = [[0_u8; 8]; 3];
        let sample_vecs = if first_x >= i64::from(input_origin) && last_x < input_end {
            let pixel_offset = (first_x - i64::from(input_origin)) as usize * 3;
            // SAFETY: the range covers 8 in-bounds RGB pixels, so vld3 reads a contiguous chunk.
            unsafe {
                let sample_vecs = vld3_u8(row.as_ptr().add(pixel_offset));
                [sample_vecs.0, sample_vecs.1, sample_vecs.2]
            }
        } else {
            for lane in 0..8 {
                let tile_x = clamp_axis(start_x + (tap + lane) as i64, input_origin, input_width);
                let pixel = tile_x * 3;
                gather[0][lane] = row[pixel];
                gather[1][lane] = row[pixel + 1];
                gather[2][lane] = row[pixel + 2];
            }
            // SAFETY: each gather buffer has exactly 8 contiguous lanes.
            unsafe {
                [
                    vld1_u8(gather[0].as_ptr()),
                    vld1_u8(gather[1].as_ptr()),
                    vld1_u8(gather[2].as_ptr()),
                ]
            }
        };

        // SAFETY: `sample_vecs` and `weights[tap..tap + 8]` each provide exactly 8 lanes.
        unsafe {
            let weight_vec = vld1q_s16(weights.as_ptr().add(tap));
            for channel in 0..3 {
                let sample_vec = vreinterpretq_s16_u16(vmovl_u8(sample_vecs[channel]));
                acc_lo[channel] = vmlal_s16(
                    acc_lo[channel],
                    vget_low_s16(sample_vec),
                    vget_low_s16(weight_vec),
                );
                acc_hi[channel] = vmlal_s16(
                    acc_hi[channel],
                    vget_high_s16(sample_vec),
                    vget_high_s16(weight_vec),
                );
            }
        }
        tap += 8;
    }

    // SAFETY: horizontal adds only read the accumulator lanes.
    unsafe {
        for channel in 0..3 {
            acc[channel] += i64::from(vaddvq_s32(acc_lo[channel]));
            acc[channel] += i64::from(vaddvq_s32(acc_hi[channel]));
        }
    }

    while tap < weights.len() {
        let tile_x = clamp_axis(start_x + tap as i64, input_origin, input_width);
        let pixel = tile_x * 3;
        let weight = i64::from(weights[tap]);
        acc[0] += weight * i64::from(row[pixel]);
        acc[1] += weight * i64::from(row[pixel + 1]);
        acc[2] += weight * i64::from(row[pixel + 2]);
        tap += 1;
    }

    [
        <u8 as ReduceSample>::from_fixed_i64(acc[0]),
        <u8 as ReduceSample>::from_fixed_i64(acc[1]),
        <u8 as ReduceSample>::from_fixed_i64(acc[2]),
    ]
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
// SAFETY: caller must execute this only on NEON-capable aarch64 and guarantee `pixel_offset + weights.len() * 3 <= row.len()` for the RGB tap window.
unsafe fn reduce_h_u8_neon_pixel_rgb_interior(
    row: &[u8],
    pixel_offset: usize,
    weights: &[i16],
) -> [u8; 3] {
    let mut acc = [0_i64; 3];
    let mut tap = 0usize;
    // SAFETY: the zero vectors are valid NEON values with no memory access.
    let mut acc_lo = [unsafe { vdupq_n_s32(0) }; 3];
    // SAFETY: the zero vectors are valid NEON values with no memory access.
    let mut acc_hi = [unsafe { vdupq_n_s32(0) }; 3];

    while tap + 8 <= weights.len() {
        // SAFETY: the caller guarantees the full RGB tap window is in-bounds.
        let sample_vecs = unsafe {
            let sample_vecs = vld3_u8(row.as_ptr().add(pixel_offset + tap * 3));
            [sample_vecs.0, sample_vecs.1, sample_vecs.2]
        };
        // SAFETY: `sample_vecs` and `weights[tap..tap + 8]` each provide exactly 8 lanes.
        unsafe {
            let weight_vec = vld1q_s16(weights.as_ptr().add(tap));
            for channel in 0..3 {
                let sample_vec = vreinterpretq_s16_u16(vmovl_u8(sample_vecs[channel]));
                acc_lo[channel] = vmlal_s16(
                    acc_lo[channel],
                    vget_low_s16(sample_vec),
                    vget_low_s16(weight_vec),
                );
                acc_hi[channel] = vmlal_s16(
                    acc_hi[channel],
                    vget_high_s16(sample_vec),
                    vget_high_s16(weight_vec),
                );
            }
        }
        tap += 8;
    }

    // SAFETY: horizontal adds only read the accumulator lanes.
    unsafe {
        for channel in 0..3 {
            acc[channel] += i64::from(vaddvq_s32(acc_lo[channel]));
            acc[channel] += i64::from(vaddvq_s32(acc_hi[channel]));
        }
    }

    while tap < weights.len() {
        let pixel = pixel_offset + tap * 3;
        let weight = i64::from(weights[tap]);
        acc[0] += weight * i64::from(row[pixel]);
        acc[1] += weight * i64::from(row[pixel + 1]);
        acc[2] += weight * i64::from(row[pixel + 2]);
        tap += 1;
    }

    [
        <u8 as ReduceSample>::from_fixed_i64(acc[0]),
        <u8 as ReduceSample>::from_fixed_i64(acc[1]),
        <u8 as ReduceSample>::from_fixed_i64(acc[2]),
    ]
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
// SAFETY: caller must execute this only on NEON-capable aarch64; the function either reads an in-bounds 8-pixel RGBA window or gathers through clamped pixel indices within `row`.
unsafe fn reduce_h_u8_neon_pixel_rgba(
    row: &[u8],
    input_origin: i32,
    input_width: usize,
    start_x: i64,
    weights: &[i16],
) -> [u8; 4] {
    let mut acc = [0_i64; 4];
    let mut tap = 0usize;
    let input_end = i64::from(input_origin) + input_width as i64;
    // SAFETY: the zero vectors are valid NEON values with no memory access.
    let mut acc_lo = [unsafe { vdupq_n_s32(0) }; 4];
    // SAFETY: the zero vectors are valid NEON values with no memory access.
    let mut acc_hi = [unsafe { vdupq_n_s32(0) }; 4];

    while tap + 8 <= weights.len() {
        let first_x = start_x + tap as i64;
        let last_x = first_x + 7;
        let mut gather = [[0_u8; 8]; 4];
        let sample_vecs = if first_x >= i64::from(input_origin) && last_x < input_end {
            let pixel_offset = (first_x - i64::from(input_origin)) as usize * 4;
            // SAFETY: the range covers 8 in-bounds RGBA pixels, so vld4 reads a contiguous chunk.
            unsafe {
                let sample_vecs = vld4_u8(row.as_ptr().add(pixel_offset));
                [sample_vecs.0, sample_vecs.1, sample_vecs.2, sample_vecs.3]
            }
        } else {
            for lane in 0..8 {
                let tile_x = clamp_axis(start_x + (tap + lane) as i64, input_origin, input_width);
                let pixel = tile_x * 4;
                gather[0][lane] = row[pixel];
                gather[1][lane] = row[pixel + 1];
                gather[2][lane] = row[pixel + 2];
                gather[3][lane] = row[pixel + 3];
            }
            // SAFETY: each gather buffer has exactly 8 contiguous lanes.
            unsafe {
                [
                    vld1_u8(gather[0].as_ptr()),
                    vld1_u8(gather[1].as_ptr()),
                    vld1_u8(gather[2].as_ptr()),
                    vld1_u8(gather[3].as_ptr()),
                ]
            }
        };

        // SAFETY: `sample_vecs` and `weights[tap..tap + 8]` each provide exactly 8 lanes.
        unsafe {
            let weight_vec = vld1q_s16(weights.as_ptr().add(tap));
            for channel in 0..4 {
                let sample_vec = vreinterpretq_s16_u16(vmovl_u8(sample_vecs[channel]));
                acc_lo[channel] = vmlal_s16(
                    acc_lo[channel],
                    vget_low_s16(sample_vec),
                    vget_low_s16(weight_vec),
                );
                acc_hi[channel] = vmlal_s16(
                    acc_hi[channel],
                    vget_high_s16(sample_vec),
                    vget_high_s16(weight_vec),
                );
            }
        }
        tap += 8;
    }

    // SAFETY: horizontal adds only read the accumulator lanes.
    unsafe {
        for channel in 0..4 {
            acc[channel] += i64::from(vaddvq_s32(acc_lo[channel]));
            acc[channel] += i64::from(vaddvq_s32(acc_hi[channel]));
        }
    }

    while tap < weights.len() {
        let tile_x = clamp_axis(start_x + tap as i64, input_origin, input_width);
        let pixel = tile_x * 4;
        let weight = i64::from(weights[tap]);
        acc[0] += weight * i64::from(row[pixel]);
        acc[1] += weight * i64::from(row[pixel + 1]);
        acc[2] += weight * i64::from(row[pixel + 2]);
        acc[3] += weight * i64::from(row[pixel + 3]);
        tap += 1;
    }

    [
        <u8 as ReduceSample>::from_fixed_i64(acc[0]),
        <u8 as ReduceSample>::from_fixed_i64(acc[1]),
        <u8 as ReduceSample>::from_fixed_i64(acc[2]),
        <u8 as ReduceSample>::from_fixed_i64(acc[3]),
    ]
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
// SAFETY: caller must execute this only on NEON-capable aarch64 and guarantee `pixel_offset + weights.len() * 4 <= row.len()` for the RGBA tap window.
unsafe fn reduce_h_u8_neon_pixel_rgba_interior(
    row: &[u8],
    pixel_offset: usize,
    weights: &[i16],
) -> [u8; 4] {
    let mut acc = [0_i64; 4];
    let mut tap = 0usize;
    // SAFETY: the zero vectors are valid NEON values with no memory access.
    let mut acc_lo = [unsafe { vdupq_n_s32(0) }; 4];
    // SAFETY: the zero vectors are valid NEON values with no memory access.
    let mut acc_hi = [unsafe { vdupq_n_s32(0) }; 4];

    while tap + 8 <= weights.len() {
        // SAFETY: the caller guarantees the full RGBA tap window is in-bounds.
        let sample_vecs = unsafe {
            let sample_vecs = vld4_u8(row.as_ptr().add(pixel_offset + tap * 4));
            [sample_vecs.0, sample_vecs.1, sample_vecs.2, sample_vecs.3]
        };
        // SAFETY: `sample_vecs` and `weights[tap..tap + 8]` each provide exactly 8 lanes.
        unsafe {
            let weight_vec = vld1q_s16(weights.as_ptr().add(tap));
            for channel in 0..4 {
                let sample_vec = vreinterpretq_s16_u16(vmovl_u8(sample_vecs[channel]));
                acc_lo[channel] = vmlal_s16(
                    acc_lo[channel],
                    vget_low_s16(sample_vec),
                    vget_low_s16(weight_vec),
                );
                acc_hi[channel] = vmlal_s16(
                    acc_hi[channel],
                    vget_high_s16(sample_vec),
                    vget_high_s16(weight_vec),
                );
            }
        }
        tap += 8;
    }

    // SAFETY: horizontal adds only read the accumulator lanes.
    unsafe {
        for channel in 0..4 {
            acc[channel] += i64::from(vaddvq_s32(acc_lo[channel]));
            acc[channel] += i64::from(vaddvq_s32(acc_hi[channel]));
        }
    }

    while tap < weights.len() {
        let pixel = pixel_offset + tap * 4;
        let weight = i64::from(weights[tap]);
        acc[0] += weight * i64::from(row[pixel]);
        acc[1] += weight * i64::from(row[pixel + 1]);
        acc[2] += weight * i64::from(row[pixel + 2]);
        acc[3] += weight * i64::from(row[pixel + 3]);
        tap += 1;
    }

    [
        <u8 as ReduceSample>::from_fixed_i64(acc[0]),
        <u8 as ReduceSample>::from_fixed_i64(acc[1]),
        <u8 as ReduceSample>::from_fixed_i64(acc[2]),
        <u8 as ReduceSample>::from_fixed_i64(acc[3]),
    ]
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
// SAFETY: caller must execute this only on NEON-capable aarch64; the function either reads an in-bounds interleaved 8-pixel window for the selected band or gathers through clamped indices.
unsafe fn reduce_h_u8_neon_pixel_nb(
    row: &[u8],
    input_origin: i32,
    input_width: usize,
    bands: usize,
    band: usize,
    start_x: i64,
    weights: &[i16],
) -> u8 {
    let mut acc = 0_i64;
    let mut tap = 0usize;
    let input_end = i64::from(input_origin) + input_width as i64;
    // SAFETY: the zero vector is a valid NEON value with no memory access.
    let mut acc_lo = unsafe { vdupq_n_s32(0) };
    // SAFETY: the zero vector is a valid NEON value with no memory access.
    let mut acc_hi = unsafe { vdupq_n_s32(0) };

    while tap + 8 <= weights.len() {
        let first_x = start_x + tap as i64;
        let last_x = first_x + 7;
        let mut gather = [0_u8; 8];
        let sample_vec = if (2..=4).contains(&bands)
            && first_x >= i64::from(input_origin)
            && last_x < input_end
        {
            let pixel_offset = (first_x - i64::from(input_origin)) as usize * bands;
            // SAFETY: the range covers 8 in-bounds interleaved pixels, so vldN reads a contiguous chunk.
            unsafe {
                match bands {
                    2 => {
                        let sample_vecs = vld2_u8(row.as_ptr().add(pixel_offset));
                        if band == 0 {
                            sample_vecs.0
                        } else {
                            sample_vecs.1
                        }
                    }
                    3 => {
                        let sample_vecs = vld3_u8(row.as_ptr().add(pixel_offset));
                        match band {
                            0 => sample_vecs.0,
                            1 => sample_vecs.1,
                            _ => sample_vecs.2,
                        }
                    }
                    4 => {
                        let sample_vecs = vld4_u8(row.as_ptr().add(pixel_offset));
                        match band {
                            0 => sample_vecs.0,
                            1 => sample_vecs.1,
                            2 => sample_vecs.2,
                            _ => sample_vecs.3,
                        }
                    }
                    _ => {
                        debug_assert!(
                            false,
                            "NEON reduce_h_u8 specialization only supports 2-4 interleaved bands"
                        );
                        // SAFETY: the surrounding branch restricts `bands` to `2..=4`.
                        std::hint::unreachable_unchecked()
                    }
                }
            }
        } else {
            for lane in 0..8 {
                let tile_x = clamp_axis(start_x + (tap + lane) as i64, input_origin, input_width);
                gather[lane] = row[tile_x * bands + band];
            }
            // SAFETY: `gather` is an 8-lane stack buffer.
            unsafe { vld1_u8(gather.as_ptr()) }
        };

        // SAFETY: `sample_vec` and `weights[tap..tap + 8]` each provide exactly 8 lanes.
        unsafe {
            let sample_vec = vreinterpretq_s16_u16(vmovl_u8(sample_vec));
            let weight_vec = vld1q_s16(weights.as_ptr().add(tap));
            acc_lo = vmlal_s16(acc_lo, vget_low_s16(sample_vec), vget_low_s16(weight_vec));
            acc_hi = vmlal_s16(acc_hi, vget_high_s16(sample_vec), vget_high_s16(weight_vec));
        }
        tap += 8;
    }

    // SAFETY: horizontal adds only read the accumulator lanes.
    unsafe {
        acc += i64::from(vaddvq_s32(acc_lo));
        acc += i64::from(vaddvq_s32(acc_hi));
    }

    while tap < weights.len() {
        let tile_x = clamp_axis(start_x + tap as i64, input_origin, input_width);
        acc += i64::from(weights[tap]) * i64::from(row[tile_x * bands + band]);
        tap += 1;
    }

    <u8 as ReduceSample>::from_fixed_i64(acc)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
// SAFETY: caller must execute this only on NEON-capable aarch64 and provide row-major `input`/`output` buffers that cover `input_region` and `output_region`.
unsafe fn reduce_h_u16_neon(
    filter: &ReduceKernel,
    input_region: &Region,
    input: &[u16],
    bands: u32,
    output_region: &Region,
    output: &mut [u16],
) {
    let out_w = output_region.width as usize;
    let out_h = output_region.height as usize;
    let in_w = input_region.width as usize;
    let bands = bands as usize;
    let in_row_stride = in_w * bands;
    let out_row_stride = out_w * bands;

    for y in 0..out_h {
        let in_row = &input[y * in_row_stride..(y + 1) * in_row_stride];
        let out_row = &mut output[y * out_row_stride..(y + 1) * out_row_stride];
        for x_out in 0..out_w {
            let source_x = filter.source_position(f64::from(output_region.x) + x_out as f64);
            let (start_x, weights) = filter.taps_for_i16(source_x);
            if bands == 1 {
                out_row[x_out] =
                    // SAFETY: the helper only gathers from `in_row` with clamped indices.
                    unsafe { reduce_h_u16_neon_pixel(in_row, input_region.x, start_x, weights) };
                continue;
            }

            let out_pixel = &mut out_row[x_out * bands..(x_out + 1) * bands];
            for (band, out_sample) in out_pixel.iter_mut().enumerate() {
                // SAFETY: the helper only gathers from `in_row` with clamped pixel indices.
                *out_sample = unsafe {
                    reduce_h_u16_neon_pixel_nb(
                        in_row,
                        input_region.x,
                        in_w,
                        bands,
                        band,
                        start_x,
                        weights,
                    )
                };
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
// SAFETY: caller must execute this only on NEON-capable aarch64; the function either reads an in-bounds 4-tap window or gathers through clamped indices within `row`.
unsafe fn reduce_h_u16_neon_pixel(
    row: &[u16],
    input_origin: i32,
    start_x: i64,
    weights: &[i16],
) -> u16 {
    let mut acc = 0_i64;
    let mut tap = 0usize;
    let input_end = i64::from(input_origin) + row.len() as i64;
    // SAFETY: the zero vector is a valid NEON value with no memory access.
    let mut acc_lo = unsafe { vdupq_n_s64(0) };
    // SAFETY: the zero vector is a valid NEON value with no memory access.
    let mut acc_hi = unsafe { vdupq_n_s64(0) };

    while tap + 4 <= weights.len() {
        let first_x = start_x + tap as i64;
        let last_x = first_x + 3;
        let mut samples = [0_u16; 4];
        let sample_vec = if first_x >= i64::from(input_origin) && last_x < input_end {
            let pixel_offset = (first_x - i64::from(input_origin)) as usize;
            // SAFETY: the 4-tap window is fully in bounds, so this reads 4 contiguous pixels.
            unsafe { vld1_u16(row.as_ptr().add(pixel_offset)) }
        } else {
            for lane in 0..4 {
                let tile_x = clamp_axis(start_x + (tap + lane) as i64, input_origin, row.len());
                samples[lane] = row[tile_x];
            }
            // SAFETY: `samples` is a 4-lane stack buffer.
            unsafe { vld1_u16(samples.as_ptr()) }
        };

        // SAFETY: `sample_vec` and `weights[tap..tap + 4]` each provide exactly 4 lanes.
        unsafe {
            let sample_vec = vmovl_u16(sample_vec);
            let weight_vec = vmovl_s16(vld1_s16(weights.as_ptr().add(tap)));
            acc_lo = vmlal_s32(
                acc_lo,
                vreinterpret_s32_u32(vget_low_u32(sample_vec)),
                vget_low_s32(weight_vec),
            );
            acc_hi = vmlal_s32(
                acc_hi,
                vreinterpret_s32_u32(vget_high_u32(sample_vec)),
                vget_high_s32(weight_vec),
            );
        }
        tap += 4;
    }

    // SAFETY: horizontal adds only read the accumulator lanes.
    unsafe {
        acc += vaddvq_s64(acc_lo);
        acc += vaddvq_s64(acc_hi);
    }

    while tap < weights.len() {
        let tile_x = clamp_axis(start_x + tap as i64, input_origin, row.len());
        acc += i64::from(weights[tap]) * i64::from(row[tile_x]);
        tap += 1;
    }

    <u16 as ReduceSample>::from_fixed_i64(acc)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
// SAFETY: caller must execute this only on NEON-capable aarch64; the function either reads an in-bounds interleaved 4-pixel window for the selected band or gathers through clamped indices.
unsafe fn reduce_h_u16_neon_pixel_nb(
    row: &[u16],
    input_origin: i32,
    input_width: usize,
    bands: usize,
    band: usize,
    start_x: i64,
    weights: &[i16],
) -> u16 {
    let mut acc = 0_i64;
    let mut tap = 0usize;
    let input_end = i64::from(input_origin) + input_width as i64;
    // SAFETY: the zero vector is a valid NEON value with no memory access.
    let mut acc_lo = unsafe { vdupq_n_s64(0) };
    // SAFETY: the zero vector is a valid NEON value with no memory access.
    let mut acc_hi = unsafe { vdupq_n_s64(0) };

    while tap + 4 <= weights.len() {
        let first_x = start_x + tap as i64;
        let last_x = first_x + 3;
        let mut gather = [0_u16; 4];
        let sample_vec = if (2..=4).contains(&bands)
            && first_x >= i64::from(input_origin)
            && last_x < input_end
        {
            let pixel_offset = (first_x - i64::from(input_origin)) as usize * bands;
            // SAFETY: the range covers 4 in-bounds interleaved pixels, so vldN reads a contiguous chunk.
            unsafe {
                match bands {
                    2 => {
                        let sample_vecs = vld2_u16(row.as_ptr().add(pixel_offset));
                        if band == 0 {
                            sample_vecs.0
                        } else {
                            sample_vecs.1
                        }
                    }
                    3 => {
                        let sample_vecs = vld3_u16(row.as_ptr().add(pixel_offset));
                        match band {
                            0 => sample_vecs.0,
                            1 => sample_vecs.1,
                            _ => sample_vecs.2,
                        }
                    }
                    4 => {
                        let sample_vecs = vld4_u16(row.as_ptr().add(pixel_offset));
                        match band {
                            0 => sample_vecs.0,
                            1 => sample_vecs.1,
                            2 => sample_vecs.2,
                            _ => sample_vecs.3,
                        }
                    }
                    _ => {
                        debug_assert!(
                            false,
                            "NEON reduce_h_u16 specialization only supports 2-4 interleaved bands"
                        );
                        // SAFETY: the surrounding branch restricts `bands` to `2..=4`.
                        std::hint::unreachable_unchecked()
                    }
                }
            }
        } else {
            for lane in 0..4 {
                let tile_x = clamp_axis(start_x + (tap + lane) as i64, input_origin, input_width);
                gather[lane] = row[tile_x * bands + band];
            }
            // SAFETY: `gather` is a 4-lane stack buffer.
            unsafe { vld1_u16(gather.as_ptr()) }
        };

        // SAFETY: `sample_vec` and `weights[tap..tap + 4]` each provide exactly 4 lanes.
        unsafe {
            let sample_vec = vmovl_u16(sample_vec);
            let weight_vec = vmovl_s16(vld1_s16(weights.as_ptr().add(tap)));
            acc_lo = vmlal_s32(
                acc_lo,
                vreinterpret_s32_u32(vget_low_u32(sample_vec)),
                vget_low_s32(weight_vec),
            );
            acc_hi = vmlal_s32(
                acc_hi,
                vreinterpret_s32_u32(vget_high_u32(sample_vec)),
                vget_high_s32(weight_vec),
            );
        }
        tap += 4;
    }

    // SAFETY: horizontal adds only read the accumulator lanes.
    unsafe {
        acc += vaddvq_s64(acc_lo);
        acc += vaddvq_s64(acc_hi);
    }

    while tap < weights.len() {
        let tile_x = clamp_axis(start_x + tap as i64, input_origin, input_width);
        acc += i64::from(weights[tap]) * i64::from(row[tile_x * bands + band]);
        tap += 1;
    }

    <u16 as ReduceSample>::from_fixed_i64(acc)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
// SAFETY: caller must execute this only on NEON-capable aarch64 and provide row-major `input`/`output` buffers that cover `input_region` and `output_region`.
unsafe fn reduce_h_f32_neon(
    filter: &ReduceKernel,
    input_region: &Region,
    input: &[f32],
    bands: u32,
    output_region: &Region,
    output: &mut [f32],
) {
    let out_w = output_region.width as usize;
    let out_h = output_region.height as usize;
    let in_w = input_region.width as usize;
    let bands = bands as usize;
    let in_row_stride = in_w * bands;
    let out_row_stride = out_w * bands;

    for y in 0..out_h {
        let in_row = &input[y * in_row_stride..(y + 1) * in_row_stride];
        let out_row = &mut output[y * out_row_stride..(y + 1) * out_row_stride];
        for x_out in 0..out_w {
            let source_x = filter.source_position(f64::from(output_region.x) + x_out as f64);
            let (start_x, weights) = filter.taps_for_f64(source_x);
            if bands == 1 {
                out_row[x_out] =
                    // SAFETY: the helper only gathers from `in_row` with clamped indices.
                    unsafe { reduce_h_f32_neon_pixel(in_row, input_region.x, start_x, weights) };
                continue;
            }

            let out_pixel = &mut out_row[x_out * bands..(x_out + 1) * bands];
            for (band, out_sample) in out_pixel.iter_mut().enumerate() {
                // SAFETY: the helper only gathers from `in_row` with clamped pixel indices.
                *out_sample = unsafe {
                    reduce_h_f32_neon_pixel_nb(
                        in_row,
                        input_region.x,
                        in_w,
                        bands,
                        band,
                        start_x,
                        weights,
                    )
                };
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
// SAFETY: caller must execute this only on NEON-capable aarch64; the helper gathers every sample through clamped indices within `row` before vectorizing the multiply-add.
unsafe fn reduce_h_f32_neon_pixel(
    row: &[f32],
    input_origin: i32,
    start_x: i64,
    weights: &[f64],
) -> f32 {
    let mut acc = 0.0_f64;
    let mut tap = 0usize;
    // SAFETY: the zero vector is a valid NEON value with no memory access.
    let mut acc_vec = unsafe { vdupq_n_f64(0.0) };

    while tap + 2 <= weights.len() {
        let mut samples = [0.0_f32; 2];
        let mut weight_arr = [0.0_f64; 2];
        for lane in 0..2 {
            let tile_x = clamp_axis(start_x + (tap + lane) as i64, input_origin, row.len());
            samples[lane] = row[tile_x];
            weight_arr[lane] = weights[tap + lane];
        }

        // SAFETY: `samples` and `weight_arr` each provide exactly 2 contiguous lanes.
        unsafe {
            let product = vmulq_f64(
                vcvt_f64_f32(vld1_f32(samples.as_ptr())),
                vld1q_f64(weight_arr.as_ptr()),
            );
            acc_vec = vaddq_f64(acc_vec, product);
        }
        tap += 2;
    }

    // SAFETY: horizontal add only reads the accumulator lanes.
    unsafe {
        acc += vaddvq_f64(acc_vec);
    }

    while tap < weights.len() {
        let tile_x = clamp_axis(start_x + tap as i64, input_origin, row.len());
        acc = f64::from(row[tile_x]).mul_add(weights[tap], acc);
        tap += 1;
    }

    acc as f32
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
// SAFETY: caller must execute this only on NEON-capable aarch64; the helper either reads an in-bounds interleaved 2-pixel window for the selected band or gathers through clamped indices.
unsafe fn reduce_h_f32_neon_pixel_nb(
    row: &[f32],
    input_origin: i32,
    input_width: usize,
    bands: usize,
    band: usize,
    start_x: i64,
    weights: &[f64],
) -> f32 {
    let mut acc = 0.0_f64;
    let mut tap = 0usize;
    let input_end = i64::from(input_origin) + input_width as i64;
    // SAFETY: the zero vector is a valid NEON value with no memory access.
    let mut acc_vec = unsafe { vdupq_n_f64(0.0) };

    while tap + 2 <= weights.len() {
        let first_x = start_x + tap as i64;
        let last_x = first_x + 1;
        let mut samples = [0.0_f32; 2];
        let mut weight_arr = [0.0_f64; 2];
        let sample_vec = if (2..=4).contains(&bands)
            && first_x >= i64::from(input_origin)
            && last_x < input_end
        {
            let pixel_offset = (first_x - i64::from(input_origin)) as usize * bands;
            // SAFETY: the range covers 2 in-bounds interleaved pixels, so vldN reads a contiguous chunk.
            unsafe {
                match bands {
                    2 => {
                        let sample_vecs = vld2_f32(row.as_ptr().add(pixel_offset));
                        if band == 0 {
                            sample_vecs.0
                        } else {
                            sample_vecs.1
                        }
                    }
                    3 => {
                        let sample_vecs = vld3_f32(row.as_ptr().add(pixel_offset));
                        match band {
                            0 => sample_vecs.0,
                            1 => sample_vecs.1,
                            _ => sample_vecs.2,
                        }
                    }
                    4 => {
                        let sample_vecs = vld4_f32(row.as_ptr().add(pixel_offset));
                        match band {
                            0 => sample_vecs.0,
                            1 => sample_vecs.1,
                            2 => sample_vecs.2,
                            _ => sample_vecs.3,
                        }
                    }
                    _ => {
                        debug_assert!(
                            false,
                            "NEON reduce_h_f32 specialization only supports 2-4 interleaved bands"
                        );
                        // SAFETY: the surrounding branch restricts `bands` to `2..=4`.
                        std::hint::unreachable_unchecked()
                    }
                }
            }
        } else {
            for lane in 0..2 {
                let tile_x = clamp_axis(start_x + (tap + lane) as i64, input_origin, input_width);
                samples[lane] = row[tile_x * bands + band];
            }
            // SAFETY: `samples` is a 2-lane stack buffer.
            unsafe { vld1_f32(samples.as_ptr()) }
        };
        weight_arr.copy_from_slice(&weights[tap..tap + 2]);

        // SAFETY: `sample_vec` and `weight_arr` each provide exactly 2 contiguous lanes.
        unsafe {
            let product = vmulq_f64(vcvt_f64_f32(sample_vec), vld1q_f64(weight_arr.as_ptr()));
            acc_vec = vaddq_f64(acc_vec, product);
        }
        tap += 2;
    }

    // SAFETY: horizontal add only reads the accumulator lanes.
    unsafe {
        acc += vaddvq_f64(acc_vec);
    }

    while tap < weights.len() {
        let tile_x = clamp_axis(start_x + tap as i64, input_origin, input_width);
        acc = f64::from(row[tile_x * bands + band]).mul_add(weights[tap], acc);
        tap += 1;
    }

    acc as f32
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
// SAFETY: caller must execute this only on NEON-capable aarch64 and provide buffers that cover the full input and output regions so each precomputed row offset stays valid.
unsafe fn reduce_v_u8_neon(
    filter: &ReduceKernel,
    input_region: &Region,
    input: &[u8],
    bands: u32,
    output_region: &Region,
    starts: &[i64],
    phases: &[u8],
    output: &mut [u8],
) {
    let out_w = output_region.width as usize;
    let out_h = output_region.height as usize;
    let in_h = input_region.height as usize;
    let input_origin = i64::from(input_region.y);
    let input_end = input_origin + in_h as i64;
    let row_stride = input_region.width as usize * bands as usize;
    let row_len = out_w * bands as usize;
    let vector_len = row_len / 8 * 8;
    let vector_len_wide = row_len / 16 * 16;
    const STACK_TAP_LIMIT: usize = 64;

    let mut y_out = 0usize;
    while y_out < out_h {
        let start_y = starts[y_out];
        let weights = filter.coeffs_i16_for_phase(phases[y_out] as usize);
        let out_row_start = y_out * row_len;
        let out_row_end = out_row_start + row_len;

        if (filter.config().factor - 2.0).abs() < f64::EPSILON
            && y_out + 1 < out_h
            && phases[y_out + 1] == phases[y_out]
            && starts[y_out + 1] == start_y + 2
            && start_y >= input_origin
            && start_y + weights.len() as i64 + 1 < input_end
        {
            let next_row_start = out_row_end;
            let start_row = (start_y - input_origin) as usize;
            let (head, tail) = output.split_at_mut(next_row_start);
            let out_row = &mut head[out_row_start..out_row_end];
            let next_out_row = &mut tail[..row_len];
            // SAFETY: the exact-2 fast path only runs for interior rows with two full overlapping windows in-bounds.
            unsafe {
                reduce_v_u8_neon_pair_exact2(
                    input,
                    row_stride,
                    row_len,
                    vector_len,
                    vector_len_wide,
                    start_row,
                    weights,
                    out_row,
                    next_out_row,
                );
            }
            y_out += 2;
            continue;
        }

        let out_row = &mut output[out_row_start..out_row_end];
        let mut row_offsets = [0usize; STACK_TAP_LIMIT];
        let precomputed_offsets = if weights.len() <= STACK_TAP_LIMIT {
            for (tap, offset) in row_offsets.iter_mut().take(weights.len()).enumerate() {
                *offset = clamp_axis(start_y + tap as i64, input_region.y, in_h) * row_stride;
            }
            Some(&row_offsets[..weights.len()])
        } else {
            None
        };

        let mut x = 0usize;
        while x + 32 <= vector_len {
            // SAFETY: the zero vectors are valid NEON values with no memory access.
            let mut acc0_lo = unsafe { vdupq_n_s32(0) };
            // SAFETY: the zero vectors are valid NEON values with no memory access.
            let mut acc0_hi = unsafe { vdupq_n_s32(0) };
            // SAFETY: the zero vectors are valid NEON values with no memory access.
            let mut acc1_lo = unsafe { vdupq_n_s32(0) };
            // SAFETY: the zero vectors are valid NEON values with no memory access.
            let mut acc1_hi = unsafe { vdupq_n_s32(0) };
            // SAFETY: the zero vectors are valid NEON values with no memory access.
            let mut acc2_lo = unsafe { vdupq_n_s32(0) };
            // SAFETY: the zero vectors are valid NEON values with no memory access.
            let mut acc2_hi = unsafe { vdupq_n_s32(0) };
            // SAFETY: the zero vectors are valid NEON values with no memory access.
            let mut acc3_lo = unsafe { vdupq_n_s32(0) };
            // SAFETY: the zero vectors are valid NEON values with no memory access.
            let mut acc3_hi = unsafe { vdupq_n_s32(0) };
            for (tap, &weight) in weights.iter().enumerate() {
                let row_offset = precomputed_offsets.map_or_else(
                    || clamp_axis(start_y + tap as i64, input_region.y, in_h) * row_stride,
                    |offsets| offsets[tap],
                ) + x;

                // SAFETY: `row_offset + 32 <= input.len()` for vector chunks and the row slice is contiguous.
                unsafe {
                    let sample_vec0 = vld1q_u8(input.as_ptr().add(row_offset));
                    let sample_vec1 = vld1q_u8(input.as_ptr().add(row_offset + 16));
                    let weight_vec = vdup_n_s16(weight);

                    let sample0_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(sample_vec0)));
                    acc0_lo = vmlal_s16(acc0_lo, vget_low_s16(sample0_lo), weight_vec);
                    acc0_hi = vmlal_s16(acc0_hi, vget_high_s16(sample0_lo), weight_vec);
                    let sample0_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(sample_vec0)));
                    acc1_lo = vmlal_s16(acc1_lo, vget_low_s16(sample0_hi), weight_vec);
                    acc1_hi = vmlal_s16(acc1_hi, vget_high_s16(sample0_hi), weight_vec);

                    let sample1_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(sample_vec1)));
                    acc2_lo = vmlal_s16(acc2_lo, vget_low_s16(sample1_lo), weight_vec);
                    acc2_hi = vmlal_s16(acc2_hi, vget_high_s16(sample1_lo), weight_vec);
                    let sample1_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(sample_vec1)));
                    acc3_lo = vmlal_s16(acc3_lo, vget_low_s16(sample1_hi), weight_vec);
                    acc3_hi = vmlal_s16(acc3_hi, vget_high_s16(sample1_hi), weight_vec);
                }
            }

            // SAFETY: `x + 32 <= vector_len` guarantees 32 writable output lanes.
            unsafe {
                store_rounded_u8x16(
                    out_row.as_mut_ptr().add(x),
                    acc0_lo,
                    acc0_hi,
                    acc1_lo,
                    acc1_hi,
                );
                store_rounded_u8x16(
                    out_row.as_mut_ptr().add(x + 16),
                    acc2_lo,
                    acc2_hi,
                    acc3_lo,
                    acc3_hi,
                );
            }
            x += 32;
        }

        while x + 16 <= vector_len {
            // SAFETY: the zero vectors are valid NEON values with no memory access.
            let mut acc0_lo = unsafe { vdupq_n_s32(0) };
            // SAFETY: the zero vectors are valid NEON values with no memory access.
            let mut acc0_hi = unsafe { vdupq_n_s32(0) };
            // SAFETY: the zero vectors are valid NEON values with no memory access.
            let mut acc1_lo = unsafe { vdupq_n_s32(0) };
            // SAFETY: the zero vectors are valid NEON values with no memory access.
            let mut acc1_hi = unsafe { vdupq_n_s32(0) };
            for (tap, &weight) in weights.iter().enumerate() {
                let row_offset = precomputed_offsets.map_or_else(
                    || clamp_axis(start_y + tap as i64, input_region.y, in_h) * row_stride,
                    |offsets| offsets[tap],
                ) + x;

                // SAFETY: `row_offset + 16 <= input.len()` for vector chunks and the row slice is contiguous.
                unsafe {
                    let sample_vec = vld1q_u8(input.as_ptr().add(row_offset));
                    let weight_vec = vdup_n_s16(weight);

                    let sample_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(sample_vec)));
                    acc0_lo = vmlal_s16(acc0_lo, vget_low_s16(sample_lo), weight_vec);
                    acc0_hi = vmlal_s16(acc0_hi, vget_high_s16(sample_lo), weight_vec);

                    let sample_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(sample_vec)));
                    acc1_lo = vmlal_s16(acc1_lo, vget_low_s16(sample_hi), weight_vec);
                    acc1_hi = vmlal_s16(acc1_hi, vget_high_s16(sample_hi), weight_vec);
                }
            }

            // SAFETY: `x + 16 <= vector_len` guarantees 16 writable output lanes.
            unsafe {
                store_rounded_u8x16(
                    out_row.as_mut_ptr().add(x),
                    acc0_lo,
                    acc0_hi,
                    acc1_lo,
                    acc1_hi,
                );
            };
            x += 16;
        }

        while x < vector_len {
            // SAFETY: the zero vectors are valid NEON values with no memory access.
            let mut acc_lo = unsafe { vdupq_n_s32(0) };
            // SAFETY: the zero vectors are valid NEON values with no memory access.
            let mut acc_hi = unsafe { vdupq_n_s32(0) };
            for (tap, &weight) in weights.iter().enumerate() {
                let row_offset =
                    clamp_axis(start_y + tap as i64, input_region.y, in_h) * row_stride + x;

                // SAFETY: `row_offset + 8 <= input.len()` for vector chunks and the row slice is contiguous.
                unsafe {
                    let sample_vec = vld1_u8(input.as_ptr().add(row_offset));
                    let sample_vec = vreinterpretq_s16_u16(vmovl_u8(sample_vec));
                    let weight_vec = vdupq_n_s16(weight);
                    acc_lo = vmlal_s16(acc_lo, vget_low_s16(sample_vec), vget_low_s16(weight_vec));
                    acc_hi =
                        vmlal_s16(acc_hi, vget_high_s16(sample_vec), vget_high_s16(weight_vec));
                }
            }

            // SAFETY: `x < vector_len` guarantees 8 writable output lanes.
            unsafe { store_rounded_u8x8(out_row.as_mut_ptr().add(x), acc_lo, acc_hi) };
            x += 8;
        }

        while x < row_len {
            let mut acc = 0_i64;
            for (tap, &weight) in weights.iter().enumerate() {
                let row_offset =
                    clamp_axis(start_y + tap as i64, input_region.y, in_h) * row_stride;
                acc += i64::from(weight) * i64::from(input[row_offset + x]);
            }
            out_row[x] = <u8 as ReduceSample>::from_fixed_i64(acc);
            x += 1;
        }
        y_out += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn reduce_v_u8_neon_pair_exact2(
    input: &[u8],
    row_stride: usize,
    row_len: usize,
    vector_len: usize,
    vector_len_wide: usize,
    start_row: usize,
    weights: &[i16],
    out_row_0: &mut [u8],
    out_row_1: &mut [u8],
) {
    let pair_taps = weights.len() + 2;
    let mut x = 0usize;
    while x < vector_len_wide {
        // SAFETY: the zero vectors are valid NEON values with no memory access.
        let mut acc0_0 = unsafe { vdupq_n_s32(0) };
        // SAFETY: the zero vectors are valid NEON values with no memory access.
        let mut acc0_1 = unsafe { vdupq_n_s32(0) };
        // SAFETY: the zero vectors are valid NEON values with no memory access.
        let mut acc0_2 = unsafe { vdupq_n_s32(0) };
        // SAFETY: the zero vectors are valid NEON values with no memory access.
        let mut acc0_3 = unsafe { vdupq_n_s32(0) };
        // SAFETY: the zero vectors are valid NEON values with no memory access.
        let mut acc1_0 = unsafe { vdupq_n_s32(0) };
        // SAFETY: the zero vectors are valid NEON values with no memory access.
        let mut acc1_1 = unsafe { vdupq_n_s32(0) };
        // SAFETY: the zero vectors are valid NEON values with no memory access.
        let mut acc1_2 = unsafe { vdupq_n_s32(0) };
        // SAFETY: the zero vectors are valid NEON values with no memory access.
        let mut acc1_3 = unsafe { vdupq_n_s32(0) };

        for src in 0..pair_taps {
            let row_offset = (start_row + src) * row_stride + x;
            // SAFETY: the exact-2 caller guarantees two full overlapping windows are in-bounds.
            unsafe {
                let row_ptr = input.as_ptr().add(row_offset);
                let sample_lo = vld1_u8(row_ptr);
                let sample_hi = vld1_u8(row_ptr.add(8));
                let sample_lo = vreinterpretq_s16_u16(vmovl_u8(sample_lo));
                let sample_hi = vreinterpretq_s16_u16(vmovl_u8(sample_hi));

                if src < weights.len() {
                    let weight_vec = vdupq_n_s16(weights[src]);
                    acc0_0 = vmlal_s16(acc0_0, vget_low_s16(sample_lo), vget_low_s16(weight_vec));
                    acc0_1 = vmlal_s16(acc0_1, vget_high_s16(sample_lo), vget_high_s16(weight_vec));
                    acc0_2 = vmlal_s16(acc0_2, vget_low_s16(sample_hi), vget_low_s16(weight_vec));
                    acc0_3 = vmlal_s16(acc0_3, vget_high_s16(sample_hi), vget_high_s16(weight_vec));
                }

                if src >= 2 {
                    let weight_vec = vdupq_n_s16(weights[src - 2]);
                    acc1_0 = vmlal_s16(acc1_0, vget_low_s16(sample_lo), vget_low_s16(weight_vec));
                    acc1_1 = vmlal_s16(acc1_1, vget_high_s16(sample_lo), vget_high_s16(weight_vec));
                    acc1_2 = vmlal_s16(acc1_2, vget_low_s16(sample_hi), vget_low_s16(weight_vec));
                    acc1_3 = vmlal_s16(acc1_3, vget_high_s16(sample_hi), vget_high_s16(weight_vec));
                }
            }
        }

        // SAFETY: `x < vector_len_wide` guarantees 16 writable output lanes in both rows.
        unsafe {
            store_rounded_u8x8(out_row_0.as_mut_ptr().add(x), acc0_0, acc0_1);
            store_rounded_u8x8(out_row_0.as_mut_ptr().add(x + 8), acc0_2, acc0_3);
            store_rounded_u8x8(out_row_1.as_mut_ptr().add(x), acc1_0, acc1_1);
            store_rounded_u8x8(out_row_1.as_mut_ptr().add(x + 8), acc1_2, acc1_3);
        };
        x += 16;
    }

    while x < vector_len {
        // SAFETY: the zero vectors are valid NEON values with no memory access.
        let mut acc0_lo = unsafe { vdupq_n_s32(0) };
        // SAFETY: the zero vectors are valid NEON values with no memory access.
        let mut acc0_hi = unsafe { vdupq_n_s32(0) };
        // SAFETY: the zero vectors are valid NEON values with no memory access.
        let mut acc1_lo = unsafe { vdupq_n_s32(0) };
        // SAFETY: the zero vectors are valid NEON values with no memory access.
        let mut acc1_hi = unsafe { vdupq_n_s32(0) };

        for src in 0..pair_taps {
            let row_offset = (start_row + src) * row_stride + x;
            // SAFETY: the exact-2 caller guarantees two full overlapping windows are in-bounds.
            unsafe {
                let sample_vec = vld1_u8(input.as_ptr().add(row_offset));
                let sample_vec = vreinterpretq_s16_u16(vmovl_u8(sample_vec));

                if src < weights.len() {
                    let weight_vec = vdupq_n_s16(weights[src]);
                    acc0_lo =
                        vmlal_s16(acc0_lo, vget_low_s16(sample_vec), vget_low_s16(weight_vec));
                    acc0_hi = vmlal_s16(
                        acc0_hi,
                        vget_high_s16(sample_vec),
                        vget_high_s16(weight_vec),
                    );
                }

                if src >= 2 {
                    let weight_vec = vdupq_n_s16(weights[src - 2]);
                    acc1_lo =
                        vmlal_s16(acc1_lo, vget_low_s16(sample_vec), vget_low_s16(weight_vec));
                    acc1_hi = vmlal_s16(
                        acc1_hi,
                        vget_high_s16(sample_vec),
                        vget_high_s16(weight_vec),
                    );
                }
            }
        }

        // SAFETY: `x < vector_len` guarantees 8 writable output lanes in both rows.
        unsafe {
            store_rounded_u8x8(out_row_0.as_mut_ptr().add(x), acc0_lo, acc0_hi);
            store_rounded_u8x8(out_row_1.as_mut_ptr().add(x), acc1_lo, acc1_hi);
        };
        x += 8;
    }

    while x < row_len {
        let mut acc0 = 0_i64;
        let mut acc1 = 0_i64;
        for src in 0..pair_taps {
            let sample = i64::from(input[(start_row + src) * row_stride + x]);
            if src < weights.len() {
                acc0 += i64::from(weights[src]) * sample;
            }
            if src >= 2 {
                acc1 += i64::from(weights[src - 2]) * sample;
            }
        }
        out_row_0[x] = <u8 as ReduceSample>::from_fixed_i64(acc0);
        out_row_1[x] = <u8 as ReduceSample>::from_fixed_i64(acc1);
        x += 1;
    }
}

#[inline]
fn reduce_v_u8_scalar_planned(
    filter: &ReduceKernel,
    input_region: &Region,
    input: &[u8],
    bands: u32,
    output_region: &Region,
    starts: &[i64],
    phases: &[u8],
    output: &mut [u8],
) {
    let out_w = output_region.width as usize;
    let out_h = output_region.height as usize;
    let in_h = input_region.height as usize;
    let row_stride = input_region.width as usize * bands as usize;
    let row_len = out_w * bands as usize;

    for y_out in 0..out_h {
        let start_y = starts[y_out];
        let weights = filter.coeffs_i16_for_phase(phases[y_out] as usize);
        let out_row = &mut output[y_out * row_len..(y_out + 1) * row_len];
        for x in 0..row_len {
            let mut acc = 0_i64;
            for (tap, &weight) in weights.iter().enumerate() {
                let tile_y = clamp_axis(start_y + tap as i64, input_region.y, in_h);
                acc += i64::from(weight) * i64::from(input[tile_y * row_stride + x]);
            }
            out_row[x] = <u8 as ReduceSample>::from_fixed_i64(acc);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
// SAFETY: caller must execute this only on NEON-capable aarch64 and provide buffers that cover the full input and output regions so each row-offset vector load stays valid.
unsafe fn reduce_v_u16_neon(
    filter: &ReduceKernel,
    input_region: &Region,
    input: &[u16],
    bands: u32,
    output_region: &Region,
    output: &mut [u16],
) {
    let out_w = output_region.width as usize;
    let out_h = output_region.height as usize;
    let in_h = input_region.height as usize;
    let row_stride = input_region.width as usize * bands as usize;
    let row_len = out_w * bands as usize;
    let vector_len = row_len / 4 * 4;

    for y_out in 0..out_h {
        let source_y = filter.source_position(f64::from(output_region.y) + y_out as f64);
        let (start_y, weights) = filter.taps_for_i16(source_y);
        let out_row = &mut output[y_out * row_len..(y_out + 1) * row_len];

        let mut x = 0usize;
        while x < vector_len {
            // SAFETY: the zero vectors are valid NEON values with no memory access.
            let mut acc_lo = unsafe { vdupq_n_s64(0) };
            // SAFETY: the zero vectors are valid NEON values with no memory access.
            let mut acc_hi = unsafe { vdupq_n_s64(0) };
            for (tap, &weight) in weights.iter().enumerate() {
                let tile_y = clamp_axis(start_y + tap as i64, input_region.y, in_h);
                let row_offset = tile_y * row_stride + x;

                // SAFETY: `row_offset + 4 <= input.len()` for vector chunks and the row slice is contiguous.
                unsafe {
                    let sample_vec = vmovl_u16(vld1_u16(input.as_ptr().add(row_offset)));
                    let weight_vec = vmovl_s16(vdup_n_s16(weight));
                    acc_lo = vmlal_s32(
                        acc_lo,
                        vreinterpret_s32_u32(vget_low_u32(sample_vec)),
                        vget_low_s32(weight_vec),
                    );
                    acc_hi = vmlal_s32(
                        acc_hi,
                        vreinterpret_s32_u32(vget_high_u32(sample_vec)),
                        vget_high_s32(weight_vec),
                    );
                }
            }

            // SAFETY: `x < vector_len` guarantees 4 writable output lanes.
            unsafe { store_rounded_u16x4(out_row.as_mut_ptr().add(x), acc_lo, acc_hi) };
            x += 4;
        }

        while x < row_len {
            let mut acc = 0_i64;
            for (tap, &weight) in weights.iter().enumerate() {
                let tile_y = clamp_axis(start_y + tap as i64, input_region.y, in_h);
                acc += i64::from(weight) * i64::from(input[tile_y * row_stride + x]);
            }
            out_row[x] = <u16 as ReduceSample>::from_fixed_i64(acc);
            x += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
// SAFETY: caller must execute this only on NEON-capable aarch64 and provide buffers that cover the full input and output regions so each row-offset vector load stays valid.
unsafe fn reduce_v_f32_neon(
    filter: &ReduceKernel,
    input_region: &Region,
    input: &[f32],
    bands: u32,
    output_region: &Region,
    output: &mut [f32],
) {
    let out_w = output_region.width as usize;
    let out_h = output_region.height as usize;
    let in_h = input_region.height as usize;
    let row_stride = input_region.width as usize * bands as usize;
    let row_len = out_w * bands as usize;
    let vector_len = row_len / 4 * 4;

    for y_out in 0..out_h {
        let source_y = filter.source_position(f64::from(output_region.y) + y_out as f64);
        let (start_y, weights) = filter.taps_for_f64(source_y);
        let out_row = &mut output[y_out * row_len..(y_out + 1) * row_len];

        let mut x = 0usize;
        while x + 2 <= vector_len {
            // SAFETY: the zero vector is a valid NEON value with no memory access.
            let mut acc_vec = unsafe { vdupq_n_f64(0.0) };
            for (tap, &weight) in weights.iter().enumerate() {
                let tile_y = clamp_axis(start_y + tap as i64, input_region.y, in_h);
                let row_offset = tile_y * row_stride + x;

                // SAFETY: `row_offset + 2 <= input.len()` for vector chunks and the row slice is contiguous.
                unsafe {
                    let product = vmulq_f64(
                        vcvt_f64_f32(vld1_f32(input.as_ptr().add(row_offset))),
                        vdupq_n_f64(weight),
                    );
                    acc_vec = vaddq_f64(acc_vec, product);
                }
            }

            let mut lanes = [0.0_f64; 2];
            // SAFETY: `lanes` has exactly enough space for the accumulator result.
            unsafe { vst1q_f64(lanes.as_mut_ptr(), acc_vec) };
            out_row[x] = lanes[0] as f32;
            out_row[x + 1] = lanes[1] as f32;
            x += 2;
        }

        while x < row_len {
            let mut acc = 0.0_f64;
            for (tap, &weight) in weights.iter().enumerate() {
                let tile_y = clamp_axis(start_y + tap as i64, input_region.y, in_h);
                acc = f64::from(input[tile_y * row_stride + x]).mul_add(weight, acc);
            }
            out_row[x] = acc as f32;
            x += 1;
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline(always)]
// SAFETY: caller must dispatch only when AVX2 is available and provide row-major `input`/`output` buffers plus `starts`/`phases` entries for every output column.
unsafe fn reduce_h_u8_avx2(
    filter: &ReduceKernel,
    input_region: &Region,
    input: &[u8],
    bands: u32,
    output_region: &Region,
    output: &mut [u8],
    starts: &[i64],
    phases: &[u8],
) {
    if bands != 1 {
        reduce_h_u8_scalar_planned(
            filter,
            input_region,
            input,
            bands,
            output_region,
            output,
            starts,
            phases,
        );
        return;
    }

    let out_w = output_region.width as usize;
    let out_h = output_region.height as usize;
    let in_w = input_region.width as usize;

    for y in 0..out_h {
        let in_row = &input[y * in_w..(y + 1) * in_w];
        let out_row = &mut output[y * out_w..(y + 1) * out_w];
        for (x_out, out_sample) in out_row.iter_mut().enumerate() {
            let start_x = starts[x_out];
            let weights = filter.coeffs_i16_for_phase(phases[x_out] as usize);
            *out_sample =
                // SAFETY: the helper only gathers from `in_row` with clamped indices.
                unsafe { reduce_h_u8_avx2_pixel(in_row, input_region.x, start_x, weights) };
        }
    }
}

#[inline]
fn reduce_h_u8_scalar_planned(
    filter: &ReduceKernel,
    input_region: &Region,
    input: &[u8],
    bands: u32,
    output_region: &Region,
    output: &mut [u8],
    starts: &[i64],
    phases: &[u8],
) {
    let out_w = output_region.width as usize;
    let out_h = output_region.height as usize;
    let in_w = input_region.width as usize;
    let bands = bands as usize;

    for y in 0..out_h {
        for x_out in 0..out_w {
            let start_x = starts[x_out];
            let weights = filter.coeffs_i16_for_phase(phases[x_out] as usize);
            for band in 0..bands {
                let mut acc = 0_i64;
                for (tap, weight) in weights.iter().copied().enumerate() {
                    let tile_x = clamp_axis(start_x + tap as i64, input_region.x, in_w);
                    let idx = (y * in_w + tile_x) * bands + band;
                    acc += i64::from(weight) * i64::from(input[idx]);
                }

                let out_idx = (y * out_w + x_out) * bands + band;
                output[out_idx] = <u8 as ReduceSample>::from_fixed_i64(acc);
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline(always)]
// SAFETY: caller must dispatch only when AVX2 is available; the helper gathers every sample through clamped indices within `row` before vectorizing the multiply-add.
unsafe fn reduce_h_u8_avx2_pixel(
    row: &[u8],
    input_origin: i32,
    start_x: i64,
    weights: &[i16],
) -> u8 {
    let mut acc = 0_i64;
    let mut tap = 0usize;

    while tap + 8 <= weights.len() {
        let mut samples = [0_u8; 8];
        for lane in 0..8 {
            let tile_x = clamp_axis(start_x + (tap + lane) as i64, input_origin, row.len());
            samples[lane] = row[tile_x];
        }

        // SAFETY: `samples` has 8 bytes and `weights[tap..tap + 8]` has 8 i16 lanes.
        let product = unsafe {
            let sample_vec = _mm256_cvtepu8_epi32(_mm_loadl_epi64(samples.as_ptr().cast()));
            let weight_vec =
                _mm256_cvtepi16_epi32(_mm_loadu_si128(weights.as_ptr().add(tap).cast()));
            _mm256_mullo_epi32(sample_vec, weight_vec)
        };
        let mut lanes = [0_i32; 8];
        // SAFETY: `lanes` has exact storage for the AVX2 result.
        unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), product) };
        acc += lanes.into_iter().map(i64::from).sum::<i64>();
        tap += 8;
    }

    while tap < weights.len() {
        let tile_x = clamp_axis(start_x + tap as i64, input_origin, row.len());
        acc += i64::from(weights[tap]) * i64::from(row[tile_x]);
        tap += 1;
    }

    <u8 as ReduceSample>::from_fixed_i64(acc)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline(always)]
// SAFETY: caller must dispatch only when AVX2 is available and provide row-major `input`/`output` buffers that cover the requested regions.
unsafe fn reduce_h_u16_avx2(
    filter: &ReduceKernel,
    input_region: &Region,
    input: &[u16],
    bands: u32,
    output_region: &Region,
    output: &mut [u16],
) {
    if bands != 1 {
        reduce_h_scalar(filter, input_region, input, bands, output_region, output);
        return;
    }

    let out_w = output_region.width as usize;
    let out_h = output_region.height as usize;
    let in_w = input_region.width as usize;

    for y in 0..out_h {
        let in_row = &input[y * in_w..(y + 1) * in_w];
        let out_row = &mut output[y * out_w..(y + 1) * out_w];
        for (x_out, out_sample) in out_row.iter_mut().enumerate() {
            let source_x = filter.source_position(output_region.x as f64 + x_out as f64);
            let (start_x, weights) = filter.taps_for_i16(source_x);
            *out_sample =
                // SAFETY: the helper only gathers from `in_row` with clamped indices.
                unsafe { reduce_h_u16_avx2_pixel(in_row, input_region.x, start_x, weights) };
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline(always)]
// SAFETY: caller must dispatch only when AVX2 is available; the helper gathers every sample through clamped indices within `row` before vectorizing the multiply-add.
unsafe fn reduce_h_u16_avx2_pixel(
    row: &[u16],
    input_origin: i32,
    start_x: i64,
    weights: &[i16],
) -> u16 {
    let mut acc = 0_i64;
    let mut tap = 0usize;

    while tap + 8 <= weights.len() {
        let mut samples = [0_u16; 8];
        for lane in 0..8 {
            let tile_x = clamp_axis(start_x + (tap + lane) as i64, input_origin, row.len());
            samples[lane] = row[tile_x];
        }

        // SAFETY: `samples` and `weights[tap..tap + 8]` each provide 8 contiguous lanes.
        let product = unsafe {
            let sample_vec = _mm256_cvtepu16_epi32(_mm_loadu_si128(samples.as_ptr().cast()));
            let weight_vec =
                _mm256_cvtepi16_epi32(_mm_loadu_si128(weights.as_ptr().add(tap).cast()));
            _mm256_mullo_epi32(sample_vec, weight_vec)
        };
        let mut lanes = [0_i32; 8];
        // SAFETY: `lanes` has exact storage for the AVX2 result.
        unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), product) };
        acc += lanes.into_iter().map(i64::from).sum::<i64>();
        tap += 8;
    }

    while tap < weights.len() {
        let tile_x = clamp_axis(start_x + tap as i64, input_origin, row.len());
        acc += i64::from(weights[tap]) * i64::from(row[tile_x]);
        tap += 1;
    }

    <u16 as ReduceSample>::from_fixed_i64(acc)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline(always)]
// SAFETY: caller must dispatch only when AVX2 is available and provide row-major `input`/`output` buffers that cover the requested regions.
unsafe fn reduce_h_f32_avx2(
    filter: &ReduceKernel,
    input_region: &Region,
    input: &[f32],
    bands: u32,
    output_region: &Region,
    output: &mut [f32],
) {
    if bands != 1 {
        reduce_h_scalar(filter, input_region, input, bands, output_region, output);
        return;
    }

    let out_w = output_region.width as usize;
    let out_h = output_region.height as usize;
    let in_w = input_region.width as usize;

    for y in 0..out_h {
        let in_row = &input[y * in_w..(y + 1) * in_w];
        let out_row = &mut output[y * out_w..(y + 1) * out_w];
        for (x_out, out_sample) in out_row.iter_mut().enumerate() {
            let source_x = filter.source_position(output_region.x as f64 + x_out as f64);
            let (start_x, weights) = filter.taps_for_f64(source_x);
            *out_sample =
                // SAFETY: the helper only gathers from `in_row` with clamped indices.
                unsafe { reduce_h_f32_avx2_pixel(in_row, input_region.x, start_x, weights) };
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline(always)]
// SAFETY: caller must dispatch only when AVX2 is available; the helper gathers every sample through clamped indices within `row` before vectorizing the multiply-add.
unsafe fn reduce_h_f32_avx2_pixel(
    row: &[f32],
    input_origin: i32,
    start_x: i64,
    weights: &[f64],
) -> f32 {
    let mut acc = 0.0_f64;
    let mut tap = 0usize;

    while tap + 4 <= weights.len() {
        let mut samples = [0.0_f32; 4];
        let mut weight_arr = [0.0_f64; 4];
        for lane in 0..4 {
            let tile_x = clamp_axis(start_x + (tap + lane) as i64, input_origin, row.len());
            samples[lane] = row[tile_x];
            weight_arr[lane] = weights[tap + lane];
        }

        // SAFETY: `samples` and `weight_arr` each provide 4 contiguous lanes.
        let product = unsafe {
            _mm256_mul_pd(
                _mm256_cvtps_pd(_mm_loadu_ps(samples.as_ptr())),
                _mm256_loadu_pd(weight_arr.as_ptr()),
            )
        };
        let mut lanes = [0.0_f64; 4];
        // SAFETY: `lanes` has exact storage for the AVX2 result.
        unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), product) };
        acc += lanes.into_iter().sum::<f64>();
        tap += 4;
    }

    while tap < weights.len() {
        let tile_x = clamp_axis(start_x + tap as i64, input_origin, row.len());
        acc += f64::from(row[tile_x]) * weights[tap];
        tap += 1;
    }

    acc as f32
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline(always)]
// SAFETY: caller must dispatch only when AVX2 is available and provide buffers that cover the full input and output regions so each row-offset vector load stays valid.
unsafe fn reduce_v_u8_avx2(
    filter: &ReduceKernel,
    input_region: &Region,
    input: &[u8],
    bands: u32,
    output_region: &Region,
    output: &mut [u8],
) {
    let out_w = output_region.width as usize;
    let out_h = output_region.height as usize;
    let in_h = input_region.height as usize;
    let row_stride = input_region.width as usize * bands as usize;
    let row_len = out_w * bands as usize;
    let vector_len = row_len / 4 * 4;

    for y_out in 0..out_h {
        let source_y = filter.source_position(output_region.y as f64 + y_out as f64);
        let (start_y, weights) = filter.taps_for_i16(source_y);
        let out_row = &mut output[y_out * row_len..(y_out + 1) * row_len];

        let mut x = 0usize;
        while x < vector_len {
            let mut acc = [0_i64; 8];
            for (tap, &weight) in weights.iter().enumerate() {
                let tile_y = clamp_axis(start_y + tap as i64, input_region.y, in_h);
                let row_offset = tile_y * row_stride + x;

                // SAFETY: `row_offset + 8 <= input.len()` for vector chunks and the row slice is contiguous.
                let product = unsafe {
                    let sample_vec = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                        input.as_ptr().add(row_offset).cast(),
                    ));
                    let weight_vec = _mm256_set1_epi32(i32::from(weight));
                    _mm256_mullo_epi32(sample_vec, weight_vec)
                };
                let mut lanes = [0_i32; 8];
                // SAFETY: `lanes` has exact storage for the AVX2 result.
                unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), product) };
                for lane in 0..8 {
                    acc[lane] += i64::from(lanes[lane]);
                }
            }

            for lane in 0..8 {
                out_row[x + lane] = <u8 as ReduceSample>::from_fixed_i64(acc[lane]);
            }
            x += 8;
        }

        while x < row_len {
            let mut acc = 0_i64;
            for (tap, &weight) in weights.iter().enumerate() {
                let tile_y = clamp_axis(start_y + tap as i64, input_region.y, in_h);
                acc += i64::from(weight) * i64::from(input[tile_y * row_stride + x]);
            }
            out_row[x] = <u8 as ReduceSample>::from_fixed_i64(acc);
            x += 1;
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline(always)]
// SAFETY: caller must dispatch only when AVX2 is available and provide buffers that cover the full input and output regions so each row-offset vector load stays valid.
unsafe fn reduce_v_u16_avx2(
    filter: &ReduceKernel,
    input_region: &Region,
    input: &[u16],
    bands: u32,
    output_region: &Region,
    output: &mut [u16],
) {
    let out_w = output_region.width as usize;
    let out_h = output_region.height as usize;
    let in_h = input_region.height as usize;
    let row_stride = input_region.width as usize * bands as usize;
    let row_len = out_w * bands as usize;
    let vector_len = row_len / 8 * 8;

    for y_out in 0..out_h {
        let source_y = filter.source_position(output_region.y as f64 + y_out as f64);
        let (start_y, weights) = filter.taps_for_i16(source_y);
        let out_row = &mut output[y_out * row_len..(y_out + 1) * row_len];

        let mut x = 0usize;
        while x < vector_len {
            let mut acc = [0_i64; 8];
            for (tap, &weight) in weights.iter().enumerate() {
                let tile_y = clamp_axis(start_y + tap as i64, input_region.y, in_h);
                let row_offset = tile_y * row_stride + x;

                // SAFETY: `row_offset + 8 <= input.len()` for vector chunks and the row slice is contiguous.
                let product = unsafe {
                    let sample_vec = _mm256_cvtepu16_epi32(_mm_loadu_si128(
                        input.as_ptr().add(row_offset).cast(),
                    ));
                    let weight_vec = _mm256_set1_epi32(i32::from(weight));
                    _mm256_mullo_epi32(sample_vec, weight_vec)
                };
                let mut lanes = [0_i32; 8];
                // SAFETY: `lanes` has exact storage for the AVX2 result.
                unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), product) };
                for lane in 0..8 {
                    acc[lane] += i64::from(lanes[lane]);
                }
            }

            for lane in 0..8 {
                out_row[x + lane] = <u16 as ReduceSample>::from_fixed_i64(acc[lane]);
            }
            x += 8;
        }

        while x < row_len {
            let mut acc = 0_i64;
            for (tap, &weight) in weights.iter().enumerate() {
                let tile_y = clamp_axis(start_y + tap as i64, input_region.y, in_h);
                acc += i64::from(weight) * i64::from(input[tile_y * row_stride + x]);
            }
            out_row[x] = <u16 as ReduceSample>::from_fixed_i64(acc);
            x += 1;
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline(always)]
// SAFETY: caller must dispatch only when AVX2 is available and provide buffers that cover the full input and output regions so each row-offset vector load stays valid.
unsafe fn reduce_v_f32_avx2(
    filter: &ReduceKernel,
    input_region: &Region,
    input: &[f32],
    bands: u32,
    output_region: &Region,
    output: &mut [f32],
) {
    let out_w = output_region.width as usize;
    let out_h = output_region.height as usize;
    let in_h = input_region.height as usize;
    let row_stride = input_region.width as usize * bands as usize;
    let row_len = out_w * bands as usize;
    let vector_len = row_len / 8 * 8;

    for y_out in 0..out_h {
        let source_y = filter.source_position(output_region.y as f64 + y_out as f64);
        let (start_y, weights) = filter.taps_for_f64(source_y);
        let out_row = &mut output[y_out * row_len..(y_out + 1) * row_len];

        let mut x = 0usize;
        while x < vector_len {
            let mut acc = [0.0_f64; 4];
            for (tap, &weight) in weights.iter().enumerate() {
                let tile_y = clamp_axis(start_y + tap as i64, input_region.y, in_h);
                let row_offset = tile_y * row_stride + x;

                // SAFETY: `row_offset + 4 <= input.len()` for vector chunks and the row slice is contiguous.
                let product = unsafe {
                    _mm256_mul_pd(
                        _mm256_cvtps_pd(_mm_loadu_ps(input.as_ptr().add(row_offset))),
                        _mm256_set1_pd(weight),
                    )
                };
                let mut lanes = [0.0_f64; 4];
                // SAFETY: `lanes` has exact storage for the AVX2 result.
                unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), product) };
                for lane in 0..4 {
                    acc[lane] += lanes[lane];
                }
            }

            for lane in 0..4 {
                out_row[x + lane] = acc[lane] as f32;
            }
            x += 4;
        }

        while x < row_len {
            let mut acc = 0.0_f64;
            for (tap, &weight) in weights.iter().enumerate() {
                let tile_y = clamp_axis(start_y + tap as i64, input_region.y, in_h);
                acc += f64::from(input[tile_y * row_stride + x]) * weight;
            }
            out_row[x] = acc as f32;
            x += 1;
        }
    }
}

#[cfg(test)]
mod horizontal_multiband_tests {
    use super::*;
    use crate::domain::{image::Region, kernel::InterpolationKernel};
    use proptest::prelude::*;

    fn kernel_strategy() -> impl Strategy<Value = InterpolationKernel> {
        prop_oneof![
            Just(InterpolationKernel::Nearest),
            Just(InterpolationKernel::Bilinear),
            Just(InterpolationKernel::Bicubic),
            Just(InterpolationKernel::CatmullRom),
            Just(InterpolationKernel::Lanczos2),
            Just(InterpolationKernel::Lanczos3),
        ]
    }

    fn factor_strategy() -> impl Strategy<Value = f64> {
        prop_oneof![Just(1.0), Just(1.5), Just(2.0), Just(3.0), Just(4.0)]
    }

    fn reduce_case_u8(
        bands: u32,
    ) -> impl Strategy<Value = (u32, u32, f64, InterpolationKernel, Vec<u8>)> {
        (1u32..=12, 1u32..=4, factor_strategy(), kernel_strategy()).prop_flat_map(
            move |(width, height, factor, kernel)| {
                let len = width as usize * height as usize * bands as usize;
                prop::collection::vec(any::<u8>(), len)
                    .prop_map(move |input| (width, height, factor, kernel, input))
            },
        )
    }

    fn reduce_case_u16(
        bands: u32,
    ) -> impl Strategy<Value = (u32, u32, f64, InterpolationKernel, Vec<u16>)> {
        (1u32..=12, 1u32..=4, factor_strategy(), kernel_strategy()).prop_flat_map(
            move |(width, height, factor, kernel)| {
                let len = width as usize * height as usize * bands as usize;
                prop::collection::vec(any::<u16>(), len)
                    .prop_map(move |input| (width, height, factor, kernel, input))
            },
        )
    }

    fn reduce_case_f32(
        bands: u32,
    ) -> impl Strategy<Value = (u32, u32, f64, InterpolationKernel, Vec<f32>)> {
        (1u32..=12, 1u32..=4, factor_strategy(), kernel_strategy()).prop_flat_map(
            move |(width, height, factor, kernel)| {
                let len = width as usize * height as usize * bands as usize;
                prop::collection::vec(-10_000.0f32..10_000.0f32, len)
                    .prop_map(move |input| (width, height, factor, kernel, input))
            },
        )
    }

    fn run_reduce_h_u8_pair(
        input: &[u8],
        width: u32,
        height: u32,
        bands: u32,
        factor: f64,
        kernel: InterpolationKernel,
    ) -> (Vec<u8>, Vec<u8>) {
        let mut filter = ReduceKernel::new(factor, kernel).unwrap();
        filter.bind_input_len(width);
        let input_region = Region::new(0, 0, width, height);
        let output_region = Region::new(0, 0, filter.config().output_width(width), height);
        let starts: Vec<i64> = (0..output_region.width as usize)
            .map(|x_out| {
                let source_x = filter.source_position(output_region.x as f64 + x_out as f64);
                filter.plan_i16(source_x).0
            })
            .collect();
        let phases: Vec<u8> = (0..output_region.width as usize)
            .map(|x_out| {
                let source_x = filter.source_position(output_region.x as f64 + x_out as f64);
                filter.plan_i16(source_x).1 as u8
            })
            .collect();
        let mut scalar = vec![0u8; output_region.pixel_count() * bands as usize];
        let mut simd = vec![0u8; output_region.pixel_count() * bands as usize];
        reduce_h_scalar(
            &filter,
            &input_region,
            input,
            bands,
            &output_region,
            &mut scalar,
        );
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: this test branch is compiled only on aarch64, so the NEON-targeted
            // function is available, and `scalar`/`simd` were allocated for exactly
            // `output_region.pixel_count() * bands` samples, matching the callee contract.
            unsafe {
                reduce_h_u8_neon(
                    &filter,
                    &input_region,
                    input,
                    bands,
                    &output_region,
                    &mut simd,
                    &starts,
                    &phases,
                );
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        reduce_h_u8(
            &filter,
            &input_region,
            input,
            bands,
            &output_region,
            &mut simd,
        );
        (scalar, simd)
    }

    fn run_reduce_h_u16_pair(
        input: &[u16],
        width: u32,
        height: u32,
        bands: u32,
        factor: f64,
        kernel: InterpolationKernel,
    ) -> (Vec<u16>, Vec<u16>) {
        let mut filter = ReduceKernel::new(factor, kernel).unwrap();
        filter.bind_input_len(width);
        let input_region = Region::new(0, 0, width, height);
        let output_region = Region::new(0, 0, filter.config().output_width(width), height);
        let mut scalar = vec![0u16; output_region.pixel_count() * bands as usize];
        let mut simd = vec![0u16; output_region.pixel_count() * bands as usize];
        reduce_h_scalar(
            &filter,
            &input_region,
            input,
            bands,
            &output_region,
            &mut scalar,
        );
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: this test branch is compiled only on aarch64, so the NEON-targeted
            // function is available, and `scalar`/`simd` were allocated for exactly
            // `output_region.pixel_count() * bands` samples, matching the callee contract.
            unsafe {
                reduce_h_u16_neon(
                    &filter,
                    &input_region,
                    input,
                    bands,
                    &output_region,
                    &mut simd,
                );
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        reduce_h_u16(
            &filter,
            &input_region,
            input,
            bands,
            &output_region,
            &mut simd,
        );
        (scalar, simd)
    }

    fn run_reduce_h_f32_pair(
        input: &[f32],
        width: u32,
        height: u32,
        bands: u32,
        factor: f64,
        kernel: InterpolationKernel,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut filter = ReduceKernel::new(factor, kernel).unwrap();
        filter.bind_input_len(width);
        let input_region = Region::new(0, 0, width, height);
        let output_region = Region::new(0, 0, filter.config().output_width(width), height);
        let mut scalar = vec![0.0f32; output_region.pixel_count() * bands as usize];
        let mut simd = vec![0.0f32; output_region.pixel_count() * bands as usize];
        reduce_h_scalar(
            &filter,
            &input_region,
            input,
            bands,
            &output_region,
            &mut scalar,
        );
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: this test branch is compiled only on aarch64, so the NEON-targeted
            // function is available, and `scalar`/`simd` were allocated for exactly
            // `output_region.pixel_count() * bands` samples, matching the callee contract.
            unsafe {
                reduce_h_f32_neon(
                    &filter,
                    &input_region,
                    input,
                    bands,
                    &output_region,
                    &mut simd,
                );
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        reduce_h_f32(
            &filter,
            &input_region,
            input,
            bands,
            &output_region,
            &mut simd,
        );
        (scalar, simd)
    }

    fn reduce_case_v_u8(
        bands: u32,
    ) -> impl Strategy<Value = (u32, u32, f64, InterpolationKernel, Vec<u8>)> {
        (1u32..=8, 4u32..=12, factor_strategy(), kernel_strategy()).prop_flat_map(
            move |(width, height, factor, kernel)| {
                let len = width as usize * height as usize * bands as usize;
                prop::collection::vec(any::<u8>(), len)
                    .prop_map(move |input| (width, height, factor, kernel, input))
            },
        )
    }

    fn reduce_case_v_u16(
        bands: u32,
    ) -> impl Strategy<Value = (u32, u32, f64, InterpolationKernel, Vec<u16>)> {
        (1u32..=8, 4u32..=12, factor_strategy(), kernel_strategy()).prop_flat_map(
            move |(width, height, factor, kernel)| {
                let len = width as usize * height as usize * bands as usize;
                prop::collection::vec(any::<u16>(), len)
                    .prop_map(move |input| (width, height, factor, kernel, input))
            },
        )
    }

    fn reduce_case_v_f32(
        bands: u32,
    ) -> impl Strategy<Value = (u32, u32, f64, InterpolationKernel, Vec<f32>)> {
        (1u32..=8, 4u32..=12, factor_strategy(), kernel_strategy()).prop_flat_map(
            move |(width, height, factor, kernel)| {
                let len = width as usize * height as usize * bands as usize;
                prop::collection::vec(-10_000.0f32..10_000.0f32, len)
                    .prop_map(move |input| (width, height, factor, kernel, input))
            },
        )
    }

    fn run_reduce_v_u8_pair(
        input: &[u8],
        width: u32,
        height: u32,
        bands: u32,
        factor: f64,
        kernel: InterpolationKernel,
    ) -> (Vec<u8>, Vec<u8>) {
        let mut filter = ReduceKernel::new(factor, kernel).unwrap();
        filter.bind_input_len(height);
        let input_region = Region::new(0, 0, width, height);
        let output_region = Region::new(0, 0, width, filter.config().output_height(height));
        let mut scalar = vec![0u8; output_region.pixel_count() * bands as usize];
        let mut simd = vec![0u8; output_region.pixel_count() * bands as usize];
        let out_h = output_region.height as usize;
        let mut starts = vec![0_i64; out_h];
        let mut phases = vec![0_u8; out_h];
        let step = filter.config().factor;
        let mut source_y = filter.source_position(output_region.y as f64);
        for y_out in 0..out_h {
            let (start_y, phase) = filter.plan_i16(source_y);
            starts[y_out] = start_y;
            phases[y_out] = phase as u8;
            source_y += step;
        }
        reduce_v_scalar(
            &filter,
            &input_region,
            input,
            bands,
            &output_region,
            &mut scalar,
        );
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: this test branch is compiled only on aarch64, so the NEON-targeted
            // function is available, and `starts`, `phases`, and `simd` were derived from
            // the same filter/output geometry that the callee expects.
            unsafe {
                reduce_v_u8_neon(
                    &filter,
                    &input_region,
                    input,
                    bands,
                    &output_region,
                    &starts,
                    &phases,
                    &mut simd,
                );
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        reduce_v_u8(
            &filter,
            &input_region,
            input,
            bands,
            &output_region,
            &starts,
            &phases,
            &mut simd,
        );
        (scalar, simd)
    }

    fn run_reduce_v_u16_pair(
        input: &[u16],
        width: u32,
        height: u32,
        bands: u32,
        factor: f64,
        kernel: InterpolationKernel,
    ) -> (Vec<u16>, Vec<u16>) {
        let mut filter = ReduceKernel::new(factor, kernel).unwrap();
        filter.bind_input_len(height);
        let input_region = Region::new(0, 0, width, height);
        let output_region = Region::new(0, 0, width, filter.config().output_height(height));
        let mut scalar = vec![0u16; output_region.pixel_count() * bands as usize];
        let mut simd = vec![0u16; output_region.pixel_count() * bands as usize];
        reduce_v_scalar(
            &filter,
            &input_region,
            input,
            bands,
            &output_region,
            &mut scalar,
        );
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: this test branch is compiled only on aarch64, so the NEON-targeted
            // function is available, and `scalar`/`simd` were allocated for exactly
            // `output_region.pixel_count() * bands` samples, matching the callee contract.
            unsafe {
                reduce_v_u16_neon(
                    &filter,
                    &input_region,
                    input,
                    bands,
                    &output_region,
                    &mut simd,
                );
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        reduce_v_u16(
            &filter,
            &input_region,
            input,
            bands,
            &output_region,
            &mut simd,
        );
        (scalar, simd)
    }

    fn run_reduce_v_f32_pair(
        input: &[f32],
        width: u32,
        height: u32,
        bands: u32,
        factor: f64,
        kernel: InterpolationKernel,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut filter = ReduceKernel::new(factor, kernel).unwrap();
        filter.bind_input_len(height);
        let input_region = Region::new(0, 0, width, height);
        let output_region = Region::new(0, 0, width, filter.config().output_height(height));
        let mut scalar = vec![0.0f32; output_region.pixel_count() * bands as usize];
        let mut simd = vec![0.0f32; output_region.pixel_count() * bands as usize];
        reduce_v_scalar(
            &filter,
            &input_region,
            input,
            bands,
            &output_region,
            &mut scalar,
        );
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: this test branch is compiled only on aarch64, so the NEON-targeted
            // function is available, and `scalar`/`simd` were allocated for exactly
            // `output_region.pixel_count() * bands` samples, matching the callee contract.
            unsafe {
                reduce_v_f32_neon(
                    &filter,
                    &input_region,
                    input,
                    bands,
                    &output_region,
                    &mut simd,
                );
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        reduce_v_f32(
            &filter,
            &input_region,
            input,
            bands,
            &output_region,
            &mut simd,
        );
        (scalar, simd)
    }

    macro_rules! prop_reduce_h_u8_matches_scalar {
        ($name:ident, $bands:expr) => {
            proptest! {
                #[test]
                fn $name((width, height, factor, kernel, input) in reduce_case_u8($bands)) {
                    let (scalar, simd) =
                        run_reduce_h_u8_pair(&input, width, height, $bands, factor, kernel);
                    prop_assert_eq!(simd, scalar);
                }
            }
        };
    }

    macro_rules! prop_reduce_h_u16_matches_scalar {
        ($name:ident, $bands:expr) => {
            proptest! {
                #[test]
                fn $name((width, height, factor, kernel, input) in reduce_case_u16($bands)) {
                    let (scalar, simd) =
                        run_reduce_h_u16_pair(&input, width, height, $bands, factor, kernel);
                    prop_assert_eq!(simd, scalar);
                }
            }
        };
    }

    macro_rules! prop_reduce_h_f32_matches_scalar {
        ($name:ident, $bands:expr) => {
            proptest! {
                #[test]
                fn $name((width, height, factor, kernel, input) in reduce_case_f32($bands)) {
                    let (scalar, simd) =
                        run_reduce_h_f32_pair(&input, width, height, $bands, factor, kernel);
                    prop_assert_eq!(simd.len(), scalar.len());
                    for (lhs, rhs) in simd.iter().zip(scalar.iter()) {
                        prop_assert!((lhs - rhs).abs() <= 1e-3);
                    }
                }
            }
        };
    }

    macro_rules! prop_reduce_v_u8_matches_scalar {
        ($name:ident, $bands:expr) => {
            proptest! {
                #[test]
                fn $name((width, height, factor, kernel, input) in reduce_case_v_u8($bands)) {
                    let (scalar, simd) =
                        run_reduce_v_u8_pair(&input, width, height, $bands, factor, kernel);
                    prop_assert_eq!(simd, scalar);
                }
            }
        };
    }

    macro_rules! prop_reduce_v_u16_matches_scalar {
        ($name:ident, $bands:expr) => {
            proptest! {
                #[test]
                fn $name((width, height, factor, kernel, input) in reduce_case_v_u16($bands)) {
                    let (scalar, simd) =
                        run_reduce_v_u16_pair(&input, width, height, $bands, factor, kernel);
                    prop_assert_eq!(simd, scalar);
                }
            }
        };
    }

    macro_rules! prop_reduce_v_f32_matches_scalar {
        ($name:ident, $bands:expr) => {
            proptest! {
                #[test]
                fn $name((width, height, factor, kernel, input) in reduce_case_v_f32($bands)) {
                    let (scalar, simd) =
                        run_reduce_v_f32_pair(&input, width, height, $bands, factor, kernel);
                    prop_assert_eq!(simd.len(), scalar.len());
                    for (lhs, rhs) in simd.iter().zip(scalar.iter()) {
                        prop_assert!((lhs - rhs).abs() <= 1e-3);
                    }
                }
            }
        };
    }

    prop_reduce_h_u8_matches_scalar!(reduce_h_u8_matches_scalar_bands_1, 1);
    prop_reduce_h_u8_matches_scalar!(reduce_h_u8_matches_scalar_bands_2, 2);
    prop_reduce_h_u8_matches_scalar!(reduce_h_u8_matches_scalar_bands_3, 3);
    prop_reduce_h_u8_matches_scalar!(reduce_h_u8_matches_scalar_bands_4, 4);

    prop_reduce_h_u16_matches_scalar!(reduce_h_u16_matches_scalar_bands_1, 1);
    prop_reduce_h_u16_matches_scalar!(reduce_h_u16_matches_scalar_bands_2, 2);
    prop_reduce_h_u16_matches_scalar!(reduce_h_u16_matches_scalar_bands_3, 3);
    prop_reduce_h_u16_matches_scalar!(reduce_h_u16_matches_scalar_bands_4, 4);

    prop_reduce_h_f32_matches_scalar!(reduce_h_f32_matches_scalar_bands_1, 1);
    prop_reduce_h_f32_matches_scalar!(reduce_h_f32_matches_scalar_bands_2, 2);
    prop_reduce_h_f32_matches_scalar!(reduce_h_f32_matches_scalar_bands_3, 3);
    prop_reduce_h_f32_matches_scalar!(reduce_h_f32_matches_scalar_bands_4, 4);

    prop_reduce_v_u8_matches_scalar!(reduce_v_u8_matches_scalar_bands_1, 1);
    prop_reduce_v_u8_matches_scalar!(reduce_v_u8_matches_scalar_bands_2, 2);
    prop_reduce_v_u8_matches_scalar!(reduce_v_u8_matches_scalar_bands_3, 3);
    prop_reduce_v_u8_matches_scalar!(reduce_v_u8_matches_scalar_bands_4, 4);

    prop_reduce_v_u16_matches_scalar!(reduce_v_u16_matches_scalar_bands_1, 1);
    prop_reduce_v_u16_matches_scalar!(reduce_v_u16_matches_scalar_bands_2, 2);
    prop_reduce_v_u16_matches_scalar!(reduce_v_u16_matches_scalar_bands_3, 3);
    prop_reduce_v_u16_matches_scalar!(reduce_v_u16_matches_scalar_bands_4, 4);

    prop_reduce_v_f32_matches_scalar!(reduce_v_f32_matches_scalar_bands_1, 1);
    prop_reduce_v_f32_matches_scalar!(reduce_v_f32_matches_scalar_bands_2, 2);
    prop_reduce_v_f32_matches_scalar!(reduce_v_f32_matches_scalar_bands_3, 3);
    prop_reduce_v_f32_matches_scalar!(reduce_v_f32_matches_scalar_bands_4, 4);

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn reduce_h_u8_rgb_dispatch_matches_direct_neon() {
        let width = 8;
        let height = 2;
        let bands = 3;
        let input: Vec<u8> = (0..width * height * bands)
            .map(|value| value as u8)
            .collect();
        let mut filter = ReduceKernel::new(2.0, InterpolationKernel::Bilinear).unwrap();
        filter.bind_input_len(width);
        let input_region = Region::new(0, 0, width, height);
        let output_region = Region::new(0, 0, filter.config().output_width(width), height);
        let mut dispatch = vec![0u8; output_region.pixel_count() * bands as usize];
        let mut direct = vec![0u8; output_region.pixel_count() * bands as usize];
        let starts: Vec<i64> = (0..output_region.width as usize)
            .map(|x_out| {
                let source_x = filter.source_position(output_region.x as f64 + x_out as f64);
                filter.plan_i16(source_x).0
            })
            .collect();
        let phases: Vec<u8> = (0..output_region.width as usize)
            .map(|x_out| {
                let source_x = filter.source_position(output_region.x as f64 + x_out as f64);
                filter.plan_i16(source_x).1 as u8
            })
            .collect();

        reduce_h_u8(
            &filter,
            &input_region,
            &input,
            bands,
            &output_region,
            &mut dispatch,
        );
        // SAFETY: the test runs on aarch64 and uses valid regions/slices.
        unsafe {
            reduce_h_u8_neon(
                &filter,
                &input_region,
                &input,
                bands,
                &output_region,
                &mut direct,
                &starts,
                &phases,
            );
        }

        assert_eq!(dispatch, direct);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{image::Region, kernel::InterpolationKernel};

    fn rgb_input_u8(width: usize) -> Vec<u8> {
        (0..width)
            .flat_map(|x| {
                [
                    ((x * 3) % 251) as u8,
                    ((x * 5 + 1) % 251) as u8,
                    ((x * 7 + 2) % 251) as u8,
                ]
            })
            .collect()
    }

    fn rgb_input_u16(width: usize) -> Vec<u16> {
        (0..width)
            .flat_map(|x| {
                [
                    ((x * 97) % 65_535) as u16,
                    ((x * 193 + 11) % 65_535) as u16,
                    ((x * 389 + 23) % 65_535) as u16,
                ]
            })
            .collect()
    }

    fn rgb_input_f32(width: usize) -> Vec<f32> {
        (0..width)
            .flat_map(|x| {
                [
                    x as f32 / 31.0,
                    (x as f32 + 0.5) / 17.0,
                    (x as f32 + 1.0) / 13.0,
                ]
            })
            .collect()
    }

    fn reduce_fixture() -> (ReduceKernel, Region, Region) {
        let mut filter = ReduceKernel::new(2.0, InterpolationKernel::Lanczos3).unwrap();
        filter.bind_input_len(16);
        (filter, Region::new(0, 0, 16, 1), Region::new(0, 0, 8, 1))
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn reduce_h_u8_neon_matches_scalar_for_rgb_rows() {
        let (filter, input_region, output_region) = reduce_fixture();
        let input = rgb_input_u8(input_region.width as usize);
        let mut scalar = vec![0_u8; output_region.width as usize * 3];
        let mut neon = vec![0_u8; output_region.width as usize * 3];
        let starts: Vec<i64> = (0..output_region.width as usize)
            .map(|x_out| {
                let source_x = filter.source_position(output_region.x as f64 + x_out as f64);
                filter.plan_i16(source_x).0
            })
            .collect();
        let phases: Vec<u8> = (0..output_region.width as usize)
            .map(|x_out| {
                let source_x = filter.source_position(output_region.x as f64 + x_out as f64);
                filter.plan_i16(source_x).1 as u8
            })
            .collect();

        reduce_h_scalar(
            &filter,
            &input_region,
            &input,
            3,
            &output_region,
            &mut scalar,
        );
        // SAFETY: test only runs on aarch64 targets where NEON is guaranteed.
        unsafe {
            reduce_h_u8_neon(
                &filter,
                &input_region,
                &input,
                3,
                &output_region,
                &mut neon,
                &starts,
                &phases,
            )
        };

        assert_eq!(neon, scalar);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn reduce_h_u16_neon_matches_scalar_for_rgb_rows() {
        let (filter, input_region, output_region) = reduce_fixture();
        let input = rgb_input_u16(input_region.width as usize);
        let mut scalar = vec![0_u16; output_region.width as usize * 3];
        let mut neon = vec![0_u16; output_region.width as usize * 3];

        reduce_h_scalar(
            &filter,
            &input_region,
            &input,
            3,
            &output_region,
            &mut scalar,
        );
        // SAFETY: test only runs on aarch64 targets where NEON is guaranteed.
        unsafe { reduce_h_u16_neon(&filter, &input_region, &input, 3, &output_region, &mut neon) };

        assert_eq!(neon, scalar);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn reduce_h_f32_neon_matches_scalar_for_rgb_rows() {
        let (filter, input_region, output_region) = reduce_fixture();
        let input = rgb_input_f32(input_region.width as usize);
        let mut scalar = vec![0.0_f32; output_region.width as usize * 3];
        let mut neon = vec![0.0_f32; output_region.width as usize * 3];

        reduce_h_scalar(
            &filter,
            &input_region,
            &input,
            3,
            &output_region,
            &mut scalar,
        );
        // SAFETY: test only runs on aarch64 targets where NEON is guaranteed.
        unsafe { reduce_h_f32_neon(&filter, &input_region, &input, 3, &output_region, &mut neon) };

        for (actual, expected) in neon.iter().zip(scalar.iter()) {
            assert!((actual - expected).abs() < 1.0e-5);
        }
    }
}

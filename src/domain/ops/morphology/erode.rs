//! Binary morphological erosion — `vips_morph(ERODE)` equivalent.
//!
//! The structuring element (mask) contains values 0, 128, or 255:
//! - 255 → foreground match (object pixel)
//! - 0   → background match (complement of pixel is used)
//! - 128 → don't-care (element excluded from the test)
//!
//! The output pixel is set only if **all** active mask elements match:
//! `result = AND over all active positions of (coeff==0 ? ~pixel : pixel)`.
//! Result is initialized to 255 (AND identity).
//!
//! This matches libvips `vips_erode_gen` exactly (morph.c line 813-815).
//!
//! Edge extension is the caller's responsibility: the source must supply a
//! tile expanded by `(mask_w/2, mask_h/2)` on each side (`VIPS_EXTEND_COPY`).

use crate::domain::{
    format::U8,
    image::{DemandHint, Region, Tile, TileMut},
    op::{NodeSpec, Op},
    simd::SimdLevel,
};

#[derive(Clone, Copy)]
struct MorphTap {
    dx: usize,
    dy: usize,
}

/// Represents an erode state.
pub struct ErodeState {
    fg_offsets: Vec<usize>,
    bg_offsets: Vec<usize>,
    cached_row_stride: usize,
    cached_bands: usize,
}

#[derive(Clone, Copy)]
enum RectKernelSimd {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2,
}

#[inline]
fn detect_rect_kernel_simd() -> RectKernelSimd {
    let simd_level = SimdLevel::detect();

    #[cfg(target_arch = "aarch64")]
    if simd_level.has_neon() {
        return RectKernelSimd::Neon;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if simd_level.has_avx2() {
        return RectKernelSimd::Avx2;
    }

    RectKernelSimd::Scalar
}

/// Binary morphological erosion with a flat structuring element.
///
/// Only operates on `U8` images (binary convention: 0 = background, 255 = object).
pub struct Erode {
    #[allow(dead_code)]
    // REASON: mask dimensions are retained for planned public inspection helpers.
    mask_w: u32,
    #[allow(dead_code)]
    // REASON: mask dimensions are retained for planned public inspection helpers.
    mask_h: u32,
    radius_x: u32,
    radius_y: u32,
    fg_taps: Vec<MorphTap>,
    bg_taps: Vec<MorphTap>,
    full_foreground_rect: Option<usize>,
    rect_kernel_simd: RectKernelSimd,
}

impl Erode {
    /// Construct an `Erode` op from a row-major mask.
    ///
    /// Mask elements must be 0, 128, or 255.
    ///
    /// # Errors
    /// Returns `Err(&'static str)` if the mask is empty, non-rectangular, or
    /// contains values outside `{0, 128, 255}`.
    pub fn new(mask: Vec<Vec<u8>>) -> Result<Self, &'static str> {
        if mask.is_empty() {
            return Err("Erode: mask must not be empty");
        }
        let mask_h = mask.len() as u32;
        let mask_w = mask[0].len() as u32;
        if mask_w == 0 {
            return Err("Erode: mask rows must not be empty");
        }
        for row in &mask {
            if row.len() as u32 != mask_w {
                return Err("Erode: mask must be rectangular");
            }
            for &v in row {
                if v != 0 && v != 128 && v != 255 {
                    return Err("Erode: mask values must be 0, 128, or 255");
                }
            }
        }

        let radius_x = mask_w / 2;
        let radius_y = mask_h / 2;
        let mut fg_taps = Vec::with_capacity((mask_w * mask_h) as usize);
        let mut bg_taps = Vec::with_capacity((mask_w * mask_h) as usize);

        for (dy, row) in mask.into_iter().enumerate() {
            for (dx, coeff) in row.into_iter().enumerate() {
                let tap = MorphTap { dx, dy };
                match coeff {
                    0 => bg_taps.push(tap),
                    255 => fg_taps.push(tap),
                    128 => {}
                    _ => return Err("Erode: mask values must be 0, 128, or 255"),
                }
            }
        }

        let full_foreground_rect = if bg_taps.is_empty()
            && mask_w == mask_h
            && fg_taps.len() == (mask_w * mask_h) as usize
        {
            Some(mask_w as usize)
        } else {
            None
        };

        Ok(Self {
            mask_w,
            mask_h,
            radius_x,
            radius_y,
            fg_taps,
            bg_taps,
            full_foreground_rect,
            rect_kernel_simd: detect_rect_kernel_simd(),
        })
    }

    #[inline]
    fn rebuild_offsets(&self, state: &mut ErodeState, row_stride: usize, bands: usize) {
        for (offset, tap) in state.fg_offsets.iter_mut().zip(self.fg_taps.iter()) {
            *offset = tap.dy * row_stride + tap.dx * bands;
        }
        for (offset, tap) in state.bg_offsets.iter_mut().zip(self.bg_taps.iter()) {
            *offset = tap.dy * row_stride + tap.dx * bands;
        }
        state.cached_row_stride = row_stride;
        state.cached_bands = bands;
    }

    #[inline]
    fn process_full_rect_3(&self, input: &Tile<U8>, output: &mut TileMut<U8>) {
        let out_h = output.region.height as usize;
        let bands = input.bands as usize;
        let row_stride = input.region.width as usize * bands;
        let out_row_stride = output.region.width as usize * bands;

        for oy in 0..out_h {
            let row0 = &input.data[oy * row_stride..];
            let row1 = &input.data[(oy + 1) * row_stride..];
            let row2 = &input.data[(oy + 2) * row_stride..];
            let output_row = &mut output.data[oy * out_row_stride..(oy + 1) * out_row_stride];

            match self.rect_kernel_simd {
                #[cfg(target_arch = "aarch64")]
                RectKernelSimd::Neon => {
                    // SAFETY: `detect_rect_kernel_simd()` only selects NEON on aarch64, and
                    // the row slices always include the halo bytes needed by the 3×3 kernel
                    // regardless of band count because horizontal neighbors remain `bands`
                    // bytes apart in the interleaved rows.
                    unsafe { erode_rect_3_neon(row0, row1, row2, bands, output_row) };
                }
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                RectKernelSimd::Avx2 => {
                    // SAFETY: `detect_rect_kernel_simd()` only selects AVX2 when supported, and
                    // the row slices always include the halo bytes needed by the 3×3 kernel
                    // regardless of band count because horizontal neighbors remain `bands`
                    // bytes apart in the interleaved rows.
                    unsafe { erode_rect_3_avx2(row0, row1, row2, bands, output_row) };
                }
                RectKernelSimd::Scalar => erode_rect_3_scalar(row0, row1, row2, bands, output_row),
            }
        }
    }

    #[inline]
    fn process_full_rect_5(&self, input: &Tile<U8>, output: &mut TileMut<U8>) {
        let out_h = output.region.height as usize;
        let bands = input.bands as usize;
        let row_stride = input.region.width as usize * bands;
        let out_row_stride = output.region.width as usize * bands;

        for oy in 0..out_h {
            let row0 = &input.data[oy * row_stride..];
            let row1 = &input.data[(oy + 1) * row_stride..];
            let row2 = &input.data[(oy + 2) * row_stride..];
            let row3 = &input.data[(oy + 3) * row_stride..];
            let row4 = &input.data[(oy + 4) * row_stride..];
            let output_row = &mut output.data[oy * out_row_stride..(oy + 1) * out_row_stride];

            match self.rect_kernel_simd {
                #[cfg(target_arch = "aarch64")]
                RectKernelSimd::Neon => {
                    // SAFETY: `detect_rect_kernel_simd()` only selects NEON on aarch64, and
                    // the row slices always include the halo bytes needed by the 5×5 kernel
                    // regardless of band count because horizontal neighbors remain `bands`
                    // bytes apart in the interleaved rows.
                    unsafe { erode_rect_5_neon(row0, row1, row2, row3, row4, bands, output_row) };
                }
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                RectKernelSimd::Avx2 => {
                    // SAFETY: `detect_rect_kernel_simd()` only selects AVX2 when supported, and
                    // the row slices always include the halo bytes needed by the 5×5 kernel
                    // regardless of band count because horizontal neighbors remain `bands`
                    // bytes apart in the interleaved rows.
                    unsafe { erode_rect_5_avx2(row0, row1, row2, row3, row4, bands, output_row) };
                }
                RectKernelSimd::Scalar => {
                    erode_rect_5_scalar(row0, row1, row2, row3, row4, bands, output_row);
                }
            }
        }
    }

    /// Build a full N×N rectangular structuring element (all positions 255).
    pub fn rect(n: u32) -> Result<Self, &'static str> {
        if n == 0 {
            return Err("Erode::rect: size must be >= 1");
        }
        let row = vec![255u8; n as usize];
        let mask = vec![row; n as usize];
        Self::new(mask)
    }
}

impl Op for Erode {
    type Input = U8;
    type Output = U8;
    type State = ErodeState;

    fn demand_hint(&self) -> DemandHint {
        DemandHint::SmallTile
    }

    fn required_input_region(&self, output: &Region) -> Region {
        Region::new(
            output.x - self.radius_x as i32,
            output.y - self.radius_y as i32,
            output.width + 2 * self.radius_x,
            output.height + 2 * self.radius_y,
        )
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec {
            input_tile_w: tile_w + 2 * self.radius_x,
            input_tile_h: tile_h + 2 * self.radius_y,
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    fn start(&self) -> Self::State {
        ErodeState {
            fg_offsets: vec![0; self.fg_taps.len()],
            bg_offsets: vec![0; self.bg_taps.len()],
            cached_row_stride: 0,
            cached_bands: 0,
        }
    }

    #[inline]
    fn process_region(&self, state: &mut Self::State, input: &Tile<U8>, output: &mut TileMut<U8>) {
        match self.full_foreground_rect {
            Some(3) => {
                self.process_full_rect_3(input, output);
                return;
            }
            Some(5) => {
                self.process_full_rect_5(input, output);
                return;
            }
            _ => {}
        }

        let out_w = output.region.width as usize;
        let out_h = output.region.height as usize;
        let bands = input.bands as usize;
        let row_stride = input.region.width as usize * bands;
        let out_row_stride = out_w * bands;

        if state.cached_row_stride != row_stride || state.cached_bands != bands {
            self.rebuild_offsets(state, row_stride, bands);
        }

        for oy in 0..out_h {
            let input_row = &input.data[oy * row_stride..];
            let output_row = &mut output.data[oy * out_row_stride..(oy + 1) * out_row_stride];

            for (sample_idx, out_sample) in output_row.iter_mut().enumerate() {
                let mut result = u8::MAX;

                for &offset in &state.fg_offsets {
                    result &= input_row[sample_idx + offset];
                    if result == 0 {
                        break;
                    }
                }

                if result != 0 {
                    for &offset in &state.bg_offsets {
                        result &= !input_row[sample_idx + offset];
                        if result == 0 {
                            break;
                        }
                    }
                }

                *out_sample = result;
            }
        }
    }
}

#[inline]
fn erode_rect_3_scalar(row0: &[u8], row1: &[u8], row2: &[u8], bands: usize, output_row: &mut [u8]) {
    let mut sample_idx = 0usize;
    let len = output_row.len();
    while sample_idx < len {
        output_row[sample_idx] = row0[sample_idx]
            & row0[sample_idx + bands]
            & row0[sample_idx + 2 * bands]
            & row1[sample_idx]
            & row1[sample_idx + bands]
            & row1[sample_idx + 2 * bands]
            & row2[sample_idx]
            & row2[sample_idx + bands]
            & row2[sample_idx + 2 * bands];
        sample_idx += 1;
    }
}

#[inline]
fn erode_rect_5_scalar(
    row0: &[u8],
    row1: &[u8],
    row2: &[u8],
    row3: &[u8],
    row4: &[u8],
    bands: usize,
    output_row: &mut [u8],
) {
    let mut sample_idx = 0usize;
    let len = output_row.len();
    while sample_idx < len {
        output_row[sample_idx] = row0[sample_idx]
            & row0[sample_idx + bands]
            & row0[sample_idx + 2 * bands]
            & row0[sample_idx + 3 * bands]
            & row0[sample_idx + 4 * bands]
            & row1[sample_idx]
            & row1[sample_idx + bands]
            & row1[sample_idx + 2 * bands]
            & row1[sample_idx + 3 * bands]
            & row1[sample_idx + 4 * bands]
            & row2[sample_idx]
            & row2[sample_idx + bands]
            & row2[sample_idx + 2 * bands]
            & row2[sample_idx + 3 * bands]
            & row2[sample_idx + 4 * bands]
            & row3[sample_idx]
            & row3[sample_idx + bands]
            & row3[sample_idx + 2 * bands]
            & row3[sample_idx + 3 * bands]
            & row3[sample_idx + 4 * bands]
            & row4[sample_idx]
            & row4[sample_idx + bands]
            & row4[sample_idx + 2 * bands]
            & row4[sample_idx + 3 * bands]
            & row4[sample_idx + 4 * bands];
        sample_idx += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
// SAFETY: caller must execute this only on NEON-capable aarch64 and provide three halo-extended rows plus an output row of matching logical width.
unsafe fn erode_rect_3_neon(
    row0: &[u8],
    row1: &[u8],
    row2: &[u8],
    bands: usize,
    output_row: &mut [u8],
) {
    use std::arch::aarch64::{vandq_u8, vld1q_u8, vst1q_u8};

    let chunks = output_row.len() / 16;
    let remainder_start = chunks * 16;

    for chunk in 0..chunks {
        let offset = chunk * 16;
        // SAFETY: `offset + 18 <= row{0,1,2}.len()` because each input row includes the 3×3
        // halo; `offset + 16 <= output_row.len()`; and NEON load/store intrinsics accept unaligned pointers.
        unsafe {
            let mut result = vandq_u8(
                vld1q_u8(row0.as_ptr().add(offset)),
                vld1q_u8(row0.as_ptr().add(offset + bands)),
            );
            result = vandq_u8(result, vld1q_u8(row0.as_ptr().add(offset + 2 * bands)));
            result = vandq_u8(result, vld1q_u8(row1.as_ptr().add(offset)));
            result = vandq_u8(result, vld1q_u8(row1.as_ptr().add(offset + bands)));
            result = vandq_u8(result, vld1q_u8(row1.as_ptr().add(offset + 2 * bands)));
            result = vandq_u8(result, vld1q_u8(row2.as_ptr().add(offset)));
            result = vandq_u8(result, vld1q_u8(row2.as_ptr().add(offset + bands)));
            result = vandq_u8(result, vld1q_u8(row2.as_ptr().add(offset + 2 * bands)));
            vst1q_u8(output_row.as_mut_ptr().add(offset), result);
        }
    }

    for offset in remainder_start..output_row.len() {
        output_row[offset] = row0[offset]
            & row0[offset + bands]
            & row0[offset + 2 * bands]
            & row1[offset]
            & row1[offset + bands]
            & row1[offset + 2 * bands]
            & row2[offset]
            & row2[offset + bands]
            & row2[offset + 2 * bands];
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
// SAFETY: caller must execute this only on NEON-capable aarch64 and provide five halo-extended rows plus an output row of matching logical width.
unsafe fn erode_rect_5_neon(
    row0: &[u8],
    row1: &[u8],
    row2: &[u8],
    row3: &[u8],
    row4: &[u8],
    bands: usize,
    output_row: &mut [u8],
) {
    use std::arch::aarch64::{vandq_u8, vld1q_u8, vst1q_u8};

    let chunks = output_row.len() / 16;
    let remainder_start = chunks * 16;

    for chunk in 0..chunks {
        let offset = chunk * 16;
        // SAFETY: `offset + 20 <= row{0..4}.len()` because each input row includes the 5×5
        // halo; `offset + 16 <= output_row.len()`; and NEON load/store intrinsics accept unaligned pointers.
        unsafe {
            let mut result = vandq_u8(
                vld1q_u8(row0.as_ptr().add(offset)),
                vld1q_u8(row0.as_ptr().add(offset + bands)),
            );
            result = vandq_u8(result, vld1q_u8(row0.as_ptr().add(offset + 2 * bands)));
            result = vandq_u8(result, vld1q_u8(row0.as_ptr().add(offset + 3 * bands)));
            result = vandq_u8(result, vld1q_u8(row0.as_ptr().add(offset + 4 * bands)));
            result = vandq_u8(result, vld1q_u8(row1.as_ptr().add(offset)));
            result = vandq_u8(result, vld1q_u8(row1.as_ptr().add(offset + bands)));
            result = vandq_u8(result, vld1q_u8(row1.as_ptr().add(offset + 2 * bands)));
            result = vandq_u8(result, vld1q_u8(row1.as_ptr().add(offset + 3 * bands)));
            result = vandq_u8(result, vld1q_u8(row1.as_ptr().add(offset + 4 * bands)));
            result = vandq_u8(result, vld1q_u8(row2.as_ptr().add(offset)));
            result = vandq_u8(result, vld1q_u8(row2.as_ptr().add(offset + bands)));
            result = vandq_u8(result, vld1q_u8(row2.as_ptr().add(offset + 2 * bands)));
            result = vandq_u8(result, vld1q_u8(row2.as_ptr().add(offset + 3 * bands)));
            result = vandq_u8(result, vld1q_u8(row2.as_ptr().add(offset + 4 * bands)));
            result = vandq_u8(result, vld1q_u8(row3.as_ptr().add(offset)));
            result = vandq_u8(result, vld1q_u8(row3.as_ptr().add(offset + bands)));
            result = vandq_u8(result, vld1q_u8(row3.as_ptr().add(offset + 2 * bands)));
            result = vandq_u8(result, vld1q_u8(row3.as_ptr().add(offset + 3 * bands)));
            result = vandq_u8(result, vld1q_u8(row3.as_ptr().add(offset + 4 * bands)));
            result = vandq_u8(result, vld1q_u8(row4.as_ptr().add(offset)));
            result = vandq_u8(result, vld1q_u8(row4.as_ptr().add(offset + bands)));
            result = vandq_u8(result, vld1q_u8(row4.as_ptr().add(offset + 2 * bands)));
            result = vandq_u8(result, vld1q_u8(row4.as_ptr().add(offset + 3 * bands)));
            result = vandq_u8(result, vld1q_u8(row4.as_ptr().add(offset + 4 * bands)));
            vst1q_u8(output_row.as_mut_ptr().add(offset), result);
        }
    }

    for offset in remainder_start..output_row.len() {
        output_row[offset] = row0[offset]
            & row0[offset + bands]
            & row0[offset + 2 * bands]
            & row0[offset + 3 * bands]
            & row0[offset + 4 * bands]
            & row1[offset]
            & row1[offset + bands]
            & row1[offset + 2 * bands]
            & row1[offset + 3 * bands]
            & row1[offset + 4 * bands]
            & row2[offset]
            & row2[offset + bands]
            & row2[offset + 2 * bands]
            & row2[offset + 3 * bands]
            & row2[offset + 4 * bands]
            & row3[offset]
            & row3[offset + bands]
            & row3[offset + 2 * bands]
            & row3[offset + 3 * bands]
            & row3[offset + 4 * bands]
            & row4[offset]
            & row4[offset + bands]
            & row4[offset + 2 * bands]
            & row4[offset + 3 * bands]
            & row4[offset + 4 * bands];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: caller must dispatch only when AVX2 is available and provide three halo-extended rows plus an output row of matching logical width.
// REASON: SIMD intrinsics (_mm256_loadu_si256 etc.) handle alignment internally;
// the pointer cast is intentional and safe within unsafe SIMD blocks.
#[allow(clippy::cast_ptr_alignment)]
unsafe fn erode_rect_3_avx2(
    row0: &[u8],
    row1: &[u8],
    row2: &[u8],
    bands: usize,
    output_row: &mut [u8],
) {
    use std::arch::x86_64::{__m256i, _mm256_and_si256, _mm256_loadu_si256, _mm256_storeu_si256};

    let chunks = output_row.len() / 32;
    let remainder_start = chunks * 32;

    for chunk in 0..chunks {
        let offset = chunk * 32;
        // SAFETY: `offset + 34 <= row{0,1,2}.len()` because each input row includes the 3×3
        // halo; `offset + 32 <= output_row.len()`; and AVX2 load/store intrinsics accept unaligned pointers.
        unsafe {
            let mut result = _mm256_and_si256(
                _mm256_loadu_si256(row0.as_ptr().add(offset).cast::<__m256i>()),
                _mm256_loadu_si256(row0.as_ptr().add(offset + bands).cast::<__m256i>()),
            );
            result = _mm256_and_si256(
                result,
                _mm256_loadu_si256(row0.as_ptr().add(offset + 2 * bands).cast::<__m256i>()),
            );
            result = _mm256_and_si256(
                result,
                _mm256_loadu_si256(row1.as_ptr().add(offset).cast::<__m256i>()),
            );
            result = _mm256_and_si256(
                result,
                _mm256_loadu_si256(row1.as_ptr().add(offset + bands).cast::<__m256i>()),
            );
            result = _mm256_and_si256(
                result,
                _mm256_loadu_si256(row1.as_ptr().add(offset + 2 * bands).cast::<__m256i>()),
            );
            result = _mm256_and_si256(
                result,
                _mm256_loadu_si256(row2.as_ptr().add(offset).cast::<__m256i>()),
            );
            result = _mm256_and_si256(
                result,
                _mm256_loadu_si256(row2.as_ptr().add(offset + bands).cast::<__m256i>()),
            );
            result = _mm256_and_si256(
                result,
                _mm256_loadu_si256(row2.as_ptr().add(offset + 2 * bands).cast::<__m256i>()),
            );
            _mm256_storeu_si256(
                output_row.as_mut_ptr().add(offset).cast::<__m256i>(),
                result,
            );
        }
    }

    for offset in remainder_start..output_row.len() {
        output_row[offset] = row0[offset]
            & row0[offset + bands]
            & row0[offset + 2 * bands]
            & row1[offset]
            & row1[offset + bands]
            & row1[offset + 2 * bands]
            & row2[offset]
            & row2[offset + bands]
            & row2[offset + 2 * bands];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: caller must dispatch only when AVX2 is available and provide five halo-extended rows plus an output row of matching logical width.
// REASON: SIMD intrinsics (_mm256_loadu_si256 etc.) handle alignment internally;
// the pointer cast is intentional and safe within unsafe SIMD blocks.
#[allow(clippy::cast_ptr_alignment)]
unsafe fn erode_rect_5_avx2(
    row0: &[u8],
    row1: &[u8],
    row2: &[u8],
    row3: &[u8],
    row4: &[u8],
    bands: usize,
    output_row: &mut [u8],
) {
    use std::arch::x86_64::{__m256i, _mm256_and_si256, _mm256_loadu_si256, _mm256_storeu_si256};

    let chunks = output_row.len() / 32;
    let remainder_start = chunks * 32;

    for chunk in 0..chunks {
        let offset = chunk * 32;
        // SAFETY: `offset + 36 <= row{0..4}.len()` because each input row includes the 5×5
        // halo; `offset + 32 <= output_row.len()`; and AVX2 load/store intrinsics accept unaligned pointers.
        unsafe {
            let mut result = _mm256_and_si256(
                _mm256_loadu_si256(row0.as_ptr().add(offset).cast::<__m256i>()),
                _mm256_loadu_si256(row0.as_ptr().add(offset + bands).cast::<__m256i>()),
            );
            result = _mm256_and_si256(
                result,
                _mm256_loadu_si256(row0.as_ptr().add(offset + 2 * bands).cast::<__m256i>()),
            );
            result = _mm256_and_si256(
                result,
                _mm256_loadu_si256(row0.as_ptr().add(offset + 3 * bands).cast::<__m256i>()),
            );
            result = _mm256_and_si256(
                result,
                _mm256_loadu_si256(row0.as_ptr().add(offset + 4 * bands).cast::<__m256i>()),
            );
            result = _mm256_and_si256(
                result,
                _mm256_loadu_si256(row1.as_ptr().add(offset).cast::<__m256i>()),
            );
            result = _mm256_and_si256(
                result,
                _mm256_loadu_si256(row1.as_ptr().add(offset + bands).cast::<__m256i>()),
            );
            result = _mm256_and_si256(
                result,
                _mm256_loadu_si256(row1.as_ptr().add(offset + 2 * bands).cast::<__m256i>()),
            );
            result = _mm256_and_si256(
                result,
                _mm256_loadu_si256(row1.as_ptr().add(offset + 3 * bands).cast::<__m256i>()),
            );
            result = _mm256_and_si256(
                result,
                _mm256_loadu_si256(row1.as_ptr().add(offset + 4 * bands).cast::<__m256i>()),
            );
            result = _mm256_and_si256(
                result,
                _mm256_loadu_si256(row2.as_ptr().add(offset).cast::<__m256i>()),
            );
            result = _mm256_and_si256(
                result,
                _mm256_loadu_si256(row2.as_ptr().add(offset + bands).cast::<__m256i>()),
            );
            result = _mm256_and_si256(
                result,
                _mm256_loadu_si256(row2.as_ptr().add(offset + 2 * bands).cast::<__m256i>()),
            );
            result = _mm256_and_si256(
                result,
                _mm256_loadu_si256(row2.as_ptr().add(offset + 3 * bands).cast::<__m256i>()),
            );
            result = _mm256_and_si256(
                result,
                _mm256_loadu_si256(row2.as_ptr().add(offset + 4 * bands).cast::<__m256i>()),
            );
            result = _mm256_and_si256(
                result,
                _mm256_loadu_si256(row3.as_ptr().add(offset).cast::<__m256i>()),
            );
            result = _mm256_and_si256(
                result,
                _mm256_loadu_si256(row3.as_ptr().add(offset + bands).cast::<__m256i>()),
            );
            result = _mm256_and_si256(
                result,
                _mm256_loadu_si256(row3.as_ptr().add(offset + 2 * bands).cast::<__m256i>()),
            );
            result = _mm256_and_si256(
                result,
                _mm256_loadu_si256(row3.as_ptr().add(offset + 3 * bands).cast::<__m256i>()),
            );
            result = _mm256_and_si256(
                result,
                _mm256_loadu_si256(row3.as_ptr().add(offset + 4 * bands).cast::<__m256i>()),
            );
            result = _mm256_and_si256(
                result,
                _mm256_loadu_si256(row4.as_ptr().add(offset).cast::<__m256i>()),
            );
            result = _mm256_and_si256(
                result,
                _mm256_loadu_si256(row4.as_ptr().add(offset + bands).cast::<__m256i>()),
            );
            result = _mm256_and_si256(
                result,
                _mm256_loadu_si256(row4.as_ptr().add(offset + 2 * bands).cast::<__m256i>()),
            );
            result = _mm256_and_si256(
                result,
                _mm256_loadu_si256(row4.as_ptr().add(offset + 3 * bands).cast::<__m256i>()),
            );
            result = _mm256_and_si256(
                result,
                _mm256_loadu_si256(row4.as_ptr().add(offset + 4 * bands).cast::<__m256i>()),
            );
            _mm256_storeu_si256(
                output_row.as_mut_ptr().add(offset).cast::<__m256i>(),
                result,
            );
        }
    }

    for offset in remainder_start..output_row.len() {
        output_row[offset] = row0[offset]
            & row0[offset + bands]
            & row0[offset + 2 * bands]
            & row0[offset + 3 * bands]
            & row0[offset + 4 * bands]
            & row1[offset]
            & row1[offset + bands]
            & row1[offset + 2 * bands]
            & row1[offset + 3 * bands]
            & row1[offset + 4 * bands]
            & row2[offset]
            & row2[offset + bands]
            & row2[offset + 2 * bands]
            & row2[offset + 3 * bands]
            & row2[offset + 4 * bands]
            & row3[offset]
            & row3[offset + bands]
            & row3[offset + 2 * bands]
            & row3[offset + 3 * bands]
            & row3[offset + 4 * bands]
            & row4[offset]
            & row4[offset + bands]
            & row4[offset + 2 * bands]
            & row4[offset + 3 * bands]
            & row4[offset + 4 * bands];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::image::{Region, Tile, TileMut};
    use proptest::prelude::*;

    fn mirror_rows(data: &[u8], width: usize, height: usize) -> Vec<u8> {
        let mut mirrored = vec![0u8; data.len()];
        for y in 0..height {
            for x in 0..width {
                mirrored[y * width + x] = data[y * width + (width - 1 - x)];
            }
        }
        mirrored
    }

    fn run_erode(
        mask: Vec<Vec<u8>>,
        input_region: Region,
        output_region: Region,
        bands: u32,
        input_data: &[u8],
    ) -> Vec<u8> {
        let op = Erode::new(mask).unwrap();
        let mut output_data = vec![0u8; output_region.pixel_count() * bands as usize];
        let input = Tile::<U8>::new(input_region, bands, input_data);
        let mut output = TileMut::<U8>::new(output_region, bands, &mut output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    fn naive_erode(
        mask: &[Vec<u8>],
        input_region: Region,
        output_region: Region,
        bands: u32,
        input_data: &[u8],
    ) -> Vec<u8> {
        let in_w = input_region.width as usize;
        let bands = bands as usize;
        let mut output = vec![0u8; output_region.pixel_count() * bands];

        for oy in 0..output_region.height as usize {
            for ox in 0..output_region.width as usize {
                for band in 0..bands {
                    let mut result = u8::MAX;
                    for (dy, row) in mask.iter().enumerate() {
                        for (dx, &coeff) in row.iter().enumerate() {
                            let idx = ((oy + dy) * in_w + ox + dx) * bands + band;
                            match coeff {
                                255 => result &= input_data[idx],
                                0 => result &= !input_data[idx],
                                128 => {}
                                _ => unreachable!(),
                            }
                        }
                    }
                    output[(oy * output_region.width as usize + ox) * bands + band] = result;
                }
            }
        }

        output
    }

    /// A single-element foreground mask [255] must copy input to output unchanged.
    #[test]
    fn identity_single_element_mask() {
        let op = Erode::new(vec![vec![255u8]]).unwrap();
        let region = Region::new(0, 0, 3, 3);
        let input_data: Vec<u8> = vec![0, 255, 0, 255, 0, 255, 0, 255, 0];
        let mut output_data = vec![0u8; 9];
        let input = Tile::<U8>::new(region, 1, &input_data);
        let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, input_data);
    }

    /// All-don't-care mask must produce all 255 output (AND of nothing = 255).
    #[test]
    fn all_dont_care_produces_255() {
        let op = Erode::new(vec![vec![128u8, 128u8], vec![128u8, 128u8]]).unwrap();
        let in_region = Region::new(0, 0, 3, 3);
        let out_region = Region::new(0, 0, 1, 1);
        let input_data = vec![0u8; 9];
        let mut output_data = vec![0u8; 1];
        let input = Tile::<U8>::new(in_region, 1, &input_data);
        let mut output = TileMut::<U8>::new(out_region, 1, &mut output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data[0], 255, "AND of no active elements must be 255");
    }

    /// A 3×3 all-foreground erosion of an all-255 image must produce all 255.
    #[test]
    fn erode_uniform_foreground() {
        let op = Erode::new(vec![
            vec![255u8, 255, 255],
            vec![255, 255, 255],
            vec![255, 255, 255],
        ])
        .unwrap();
        let in_region = Region::new(0, 0, 4, 4);
        let out_region = Region::new(0, 0, 2, 2);
        let input_data = vec![255u8; 16];
        let mut output_data = vec![0u8; 4];
        let input = Tile::<U8>::new(in_region, 1, &input_data);
        let mut output = TileMut::<U8>::new(out_region, 1, &mut output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
        assert!(
            output_data.iter().all(|&v| v == 255),
            "all-foreground erosion must produce 255 everywhere"
        );
    }

    /// A 3×3 all-foreground erosion of an all-zero image must produce all 0.
    #[test]
    fn erode_uniform_background() {
        let op = Erode::new(vec![
            vec![255u8, 255, 255],
            vec![255, 255, 255],
            vec![255, 255, 255],
        ])
        .unwrap();
        let in_region = Region::new(0, 0, 4, 4);
        let out_region = Region::new(0, 0, 2, 2);
        let input_data = vec![0u8; 16];
        let mut output_data = vec![99u8; 4];
        let input = Tile::<U8>::new(in_region, 1, &input_data);
        let mut output = TileMut::<U8>::new(out_region, 1, &mut output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
        assert!(
            output_data.iter().all(|&v| v == 0),
            "all-background erosion with foreground mask must produce 0"
        );
    }

    #[test]
    fn erode_three_by_three_square_to_single_white_pixel() {
        let op = Erode::rect(3).unwrap();
        let in_region = Region::new(0, 0, 5, 5);
        let out_region = Region::new(0, 0, 3, 3);
        let input_data = vec![
            0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 0, 0,
            0, 0,
        ];
        let mut output_data = vec![0u8; 9];
        let input = Tile::<U8>::new(in_region, 1, &input_data);
        let mut output = TileMut::<U8>::new(out_region, 1, &mut output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, vec![0, 0, 0, 0, 255, 0, 0, 0, 0,]);
    }

    /// Constructor must reject invalid mask values.
    #[test]
    fn rejects_invalid_mask_values() {
        assert!(Erode::new(vec![vec![1u8]]).is_err());
        assert!(Erode::new(vec![vec![127u8]]).is_err());
    }

    /// Constructor must reject empty masks.
    #[test]
    fn rejects_empty_mask() {
        assert!(Erode::new(vec![]).is_err());
        assert!(Erode::new(vec![vec![]]).is_err());
    }

    #[test]
    fn large_mask_preserves_single_foreground_pixel_under_edge_extension() {
        let op = Erode::rect(5).unwrap();
        let in_region = Region::new(-2, -2, 5, 5);
        let out_region = Region::new(0, 0, 1, 1);
        let input_data = vec![255u8; 25];
        let mut output_data = vec![0u8; 1];
        let input = Tile::<U8>::new(in_region, 1, &input_data);
        let mut output = TileMut::<U8>::new(out_region, 1, &mut output_data);

        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);

        assert_eq!(output_data[0], 255);
    }

    #[test]
    fn rect_three_by_three_erodes_rgb_bands_without_cross_talk() {
        let op = Erode::rect(3).unwrap();
        let in_region = Region::new(0, 0, 5, 5);
        let out_region = Region::new(0, 0, 3, 3);
        let input_data = vec![255u8; 5 * 5 * 3];
        let mut output_data = vec![0u8; 3 * 3 * 3];
        let input = Tile::<U8>::new(in_region, 3, &input_data);
        let mut output = TileMut::<U8>::new(out_region, 3, &mut output_data);

        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);

        for pixel in output_data.chunks_exact(3) {
            assert_eq!(pixel, &[255, 255, 255]);
        }
    }

    #[test]
    fn rejects_non_rectangular_masks() {
        assert!(Erode::new(vec![vec![255, 255], vec![255]]).is_err());
    }

    #[test]
    fn generic_path_rebuilds_offsets_and_matches_naive_for_multiband_data() {
        let mask = vec![vec![255, 0, 128], vec![128, 255, 0]];
        let op = Erode::new(mask.clone()).unwrap();
        let input_region = Region::new(0, 0, 5, 3);
        let output_region = Region::new(0, 0, 3, 2);
        let input_data = vec![
            255, 250, 240, 230, 220, 210, 200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100,
            90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 15, 25, 35,
        ];
        let input = Tile::<U8>::new(input_region, 2, &input_data);
        let mut output_data = vec![0u8; output_region.pixel_count() * 2];
        let mut output = TileMut::<U8>::new(output_region, 2, &mut output_data);
        let mut state = op.start();

        op.process_region(&mut state, &input, &mut output);

        assert_eq!(
            output_data,
            naive_erode(&mask, input_region, output_region, 2, &input_data)
        );
        assert_eq!(state.cached_bands, 2);
        assert_eq!(state.cached_row_stride, input_region.width as usize * 2);
    }

    #[test]
    fn forced_scalar_rect5_matches_naive_reference() {
        let mut op = Erode::rect(5).unwrap();
        op.rect_kernel_simd = RectKernelSimd::Scalar;
        let input_region = Region::new(0, 0, 7, 7);
        let output_region = Region::new(0, 0, 3, 3);
        let input_data: Vec<u8> = (0..49)
            .map(|idx| if idx % 4 == 0 { 0 } else { 255 })
            .collect();
        let input = Tile::<U8>::new(input_region, 1, &input_data);
        let mut output_data = vec![0u8; output_region.pixel_count()];
        let mut output = TileMut::<U8>::new(output_region, 1, &mut output_data);
        let mut state = op.start();

        op.process_region(&mut state, &input, &mut output);

        assert_eq!(
            output_data,
            naive_erode(
                &vec![vec![255; 5]; 5],
                input_region,
                output_region,
                1,
                &input_data
            )
        );
    }

    #[test]
    fn rect_three_by_three_large_width_matches_naive_reference() {
        let mask = vec![vec![255u8; 3]; 3];
        let input_region = Region::new(0, 0, 22, 4);
        let output_region = Region::new(0, 0, 20, 2);
        let input_data = (0..(input_region.pixel_count() as usize))
            .map(|index| if index % 3 == 0 { 255 } else { 0 })
            .collect::<Vec<_>>();

        let actual = run_erode(mask.clone(), input_region, output_region, 1, &input_data);
        let expected = naive_erode(&mask, input_region, output_region, 1, &input_data);

        assert_eq!(actual, expected);
    }

    #[test]
    fn rect_five_by_five_rgb_matches_naive_reference() {
        let mask = vec![vec![255u8; 5]; 5];
        let input_region = Region::new(0, 0, 24, 6);
        let output_region = Region::new(0, 0, 20, 2);
        let input_data = (0..(input_region.pixel_count() as usize * 3))
            .map(|index| ((index * 19 + 5) % 256) as u8)
            .collect::<Vec<_>>();

        let actual = run_erode(mask.clone(), input_region, output_region, 3, &input_data);
        let expected = naive_erode(&mask, input_region, output_region, 3, &input_data);

        assert_eq!(actual, expected);
    }

    #[test]
    fn mixed_foreground_and_background_mask_matches_naive_reference() {
        let mask = vec![vec![128u8, 255, 0], vec![0, 128, 255], vec![255, 0, 128]];
        let input_region = Region::new(0, 0, 7, 4);
        let output_region = Region::new(0, 0, 5, 2);
        let input_data = (0..(input_region.pixel_count() as usize * 2))
            .map(|index| ((index * 11 + 3) % 251) as u8)
            .collect::<Vec<_>>();

        let actual = run_erode(mask.clone(), input_region, output_region, 2, &input_data);
        let expected = naive_erode(&mask, input_region, output_region, 2, &input_data);

        assert_eq!(actual, expected);
    }

    proptest! {
        /// A single-element foreground mask on any U8 image must copy input to output.
        #[test]
        fn identity_mask_copies_input(
            pixels in proptest::collection::vec(0u8..=255u8, 1..=64)
        ) {
            let len = pixels.len() as u32;
            let op = Erode::new(vec![vec![255u8]]).unwrap();
            let region = Region::new(0, 0, len, 1);
            let mut output = vec![0u8; len as usize];
            let input_tile = Tile::<U8>::new(region, 1, &pixels);
            let mut output_tile = TileMut::<U8>::new(region, 1, &mut output);
            let mut state = op.start();
            op.process_region(&mut state, &input_tile, &mut output_tile);
            prop_assert_eq!(output, pixels);
        }

        /// Erosion with a background mask (coeff=0) must complement each pixel.
        #[test]
        fn background_mask_complements_input(
            pixels in proptest::collection::vec(0u8..=255u8, 1..=64)
        ) {
            let len = pixels.len() as u32;
            let op = Erode::new(vec![vec![0u8]]).unwrap();
            let region = Region::new(0, 0, len, 1);
            let mut output = vec![0u8; len as usize];
            let input_tile = Tile::<U8>::new(region, 1, &pixels);
            let mut output_tile = TileMut::<U8>::new(region, 1, &mut output);
            let mut state = op.start();
            op.process_region(&mut state, &input_tile, &mut output_tile);
            for (got, expected) in output.iter().zip(pixels.iter()) {
                prop_assert_eq!(*got, !expected, "background mask must complement pixel");
            }
        }

        #[test]
        fn symmetric_mask_commutes_with_horizontal_reflection(
            width in 1usize..5,
            height in 1usize..5,
            pixels in proptest::collection::vec(0u8..=255u8, 1..=25)
        ) {
            let pixel_count = width * height;
            prop_assume!(pixel_count <= pixels.len());
            let input_pixels = &pixels[..pixel_count];
            let mirrored = mirror_rows(input_pixels, width, height);
            let op = Erode::new(vec![vec![255u8]]).unwrap();
            let region = Region::new(0, 0, width as u32, height as u32);
            let input_tile = Tile::<U8>::new(region, 1, input_pixels);
            let mirrored_tile = Tile::<U8>::new(region, 1, &mirrored);
            let mut output = vec![0u8; pixel_count];
            let mut mirrored_output = vec![0u8; pixel_count];
            let mut output_tile = TileMut::<U8>::new(region, 1, &mut output);
            let mut mirrored_output_tile = TileMut::<U8>::new(region, 1, &mut mirrored_output);

            let mut state = op.start();
            let mut mirrored_state = op.start();
            op.process_region(&mut state, &input_tile, &mut output_tile);
            op.process_region(&mut mirrored_state, &mirrored_tile, &mut mirrored_output_tile);

            prop_assert_eq!(mirror_rows(&output, width, height), mirrored_output);
        }
    }
}

//! Median rank filter with libvips-style histogram fast path for large `U8` windows.

#![allow(clippy::struct_field_names)]
// REASON: median window fields deliberately match the neighborhood terminology used throughout the module.

use std::{cmp::Ordering, marker::PhantomData};

use crate::domain::{
    format::{BandFormatId, NumericBand},
    image::{DemandHint, Region, Tile, TileMut},
    op::{NodeSpec, Op},
    simd::SimdLevel,
};

const U8_HISTOGRAM_THRESHOLD_AREA: usize = 10;

#[derive(Clone, Copy)]
enum Median3x3Simd {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

#[inline]
fn detect_median_3x3_simd() -> Median3x3Simd {
    #[cfg(target_arch = "aarch64")]
    if SimdLevel::detect().has_neon() {
        return Median3x3Simd::Neon;
    }

    Median3x3Simd::Scalar
}

/// Represents a median state.
pub struct MedianState<T> {
    scratch: Vec<T>,
}

/// Median rank filter over a rectangular window.
///
/// The input tile must already include the halo declared by `required_input_region()`.
/// For `U8` windows larger than the libvips threshold, processing switches to a
/// 256-bin histogram path; all other cases use the generic in-place selection path.
pub struct Median<F: NumericBand> {
    window_w: u32,
    window_h: u32,
    radius_x: u32,
    radius_y: u32,
    window_area: usize,
    median_index: usize,
    median_3x3_simd: Median3x3Simd,
    _format: PhantomData<F>,
}

impl<F> Median<F>
where
    F: NumericBand,
    F::Sample: Copy + Default + PartialOrd,
{
    /// Creates a new `Median`.
    pub fn new(window_w: u32, window_h: u32) -> Result<Self, &'static str> {
        if window_w == 0 || window_h == 0 {
            return Err("Median: window dimensions must be >= 1");
        }
        if window_w.is_multiple_of(2) || window_h.is_multiple_of(2) {
            return Err("Median: window dimensions must be odd");
        }

        let window_area = window_w as usize * window_h as usize;

        Ok(Self {
            window_w,
            window_h,
            radius_x: window_w / 2,
            radius_y: window_h / 2,
            window_area,
            median_index: window_area / 2,
            median_3x3_simd: detect_median_3x3_simd(),
            _format: PhantomData,
        })
    }

    #[inline]
    fn should_use_histogram_fast_path(&self) -> bool {
        F::ID == BandFormatId::U8 && self.window_area > U8_HISTOGRAM_THRESHOLD_AREA
    }

    #[inline]
    fn process_dispatch(
        &self,
        state: &mut MedianState<F::Sample>,
        input: &Tile<F>,
        output: &mut TileMut<F>,
    ) {
        if F::ID == BandFormatId::U8 && self.window_w == 3 && self.window_h == 3 {
            // SAFETY: `BandFormat` is sealed, so `F::ID == U8` implies `F::Sample == u8`, and the cast preserves element count.
            let input_u8 = unsafe {
                std::slice::from_raw_parts(input.data.as_ptr().cast::<u8>(), input.data.len())
            };
            // SAFETY: same invariant as above for the mutable output slice.
            let output_u8 = unsafe {
                std::slice::from_raw_parts_mut(
                    output.data.as_mut_ptr().cast::<u8>(),
                    output.data.len(),
                )
            };
            self.process_region_u8_3x3(
                input_u8,
                input.region,
                input.bands,
                output_u8,
                output.region,
            );
        } else if self.should_use_histogram_fast_path() {
            // SAFETY: `BandFormat` is sealed and `F::ID == U8` can only be true when `F::Sample` is `u8`; the input/output slices keep the same element count.
            let input_u8 = unsafe {
                std::slice::from_raw_parts(input.data.as_ptr().cast::<u8>(), input.data.len())
            };
            // SAFETY: same invariant as above for the mutable output slice.
            let output_u8 = unsafe {
                std::slice::from_raw_parts_mut(
                    output.data.as_mut_ptr().cast::<u8>(),
                    output.data.len(),
                )
            };
            self.process_region_histogram_u8(
                input_u8,
                input.region,
                input.bands,
                output_u8,
                output.region,
            );
        } else {
            self.process_region_select(state.scratch.as_mut_slice(), input, output);
        }
    }

    fn process_region_u8_3x3(
        &self,
        input: &[u8],
        input_region: Region,
        bands: u32,
        output: &mut [u8],
        output_region: Region,
    ) {
        let out_h = output_region.height as usize;
        let row_stride = input_region.width as usize * bands as usize;
        let out_row_stride = output_region.width as usize * bands as usize;
        let bands = bands as usize;

        for oy in 0..out_h {
            let row0 = &input[oy * row_stride..];
            let row1 = &input[(oy + 1) * row_stride..];
            let row2 = &input[(oy + 2) * row_stride..];
            let output_row = &mut output[oy * out_row_stride..(oy + 1) * out_row_stride];

            match self.median_3x3_simd {
                #[cfg(target_arch = "aarch64")]
                Median3x3Simd::Neon => {
                    // SAFETY: `detect_median_3x3_simd()` only selects NEON on aarch64, and each row includes the two-pixel halo for every SIMD chunk below.
                    unsafe { median_3x3_u8_neon(row0, row1, row2, bands, output_row) };
                }
                Median3x3Simd::Scalar => median_3x3_u8_scalar(row0, row1, row2, bands, output_row),
            }
        }
    }

    fn process_region_select(
        &self,
        scratch: &mut [F::Sample],
        input: &Tile<F>,
        output: &mut TileMut<F>,
    ) {
        assert_eq!(scratch.len(), self.window_area);

        let out_w = output.region.width as usize;
        let out_h = output.region.height as usize;
        let in_w = input.region.width as usize;
        let bands = input.bands as usize;
        let window_w = self.window_w as usize;
        let window_h = self.window_h as usize;
        let input_row_stride = in_w * bands;
        let output_row_stride = out_w * bands;

        assert_eq!(input.data.len(), input.region.pixel_count() * bands);
        assert_eq!(output.data.len(), output.region.pixel_count() * bands);

        for (input_rows, output_row) in input
            .data
            .chunks_exact(input_row_stride)
            .zip(output.data.chunks_exact_mut(output_row_stride))
            .take(out_h)
        {
            for (ox, output_pixel) in output_row.chunks_exact_mut(bands).enumerate() {
                let column_offset = ox * bands;
                for (band, out_sample) in output_pixel.iter_mut().enumerate() {
                    for (slot, sample) in scratch.iter_mut().zip(
                        input_rows
                            .chunks_exact(input_row_stride)
                            .take(window_h)
                            .flat_map(|row| {
                                row[column_offset + band..]
                                    .iter()
                                    .step_by(bands)
                                    .take(window_w)
                            }),
                    ) {
                        *slot = *sample;
                    }
                    let (_, median, _) = scratch.select_nth_unstable_by(
                        self.median_index,
                        partial_cmp_or_equal::<F::Sample>,
                    );
                    *out_sample = *median;
                }
            }
        }
    }

    fn process_region_histogram_u8(
        &self,
        input: &[u8],
        input_region: Region,
        bands: u32,
        output: &mut [u8],
        output_region: Region,
    ) {
        let out_w = output_region.width as usize;
        let out_h = output_region.height as usize;
        let in_w = input_region.width as usize;
        let bands = bands as usize;
        let window_w = self.window_w as usize;
        let window_h = self.window_h as usize;

        for oy in 0..out_h {
            for band in 0..bands {
                let mut hist = [0u32; 256];
                for wy in 0..window_h {
                    let row_base = (oy + wy) * in_w;
                    for wx in 0..window_w {
                        let sample = input[(row_base + wx) * bands + band];
                        hist[sample as usize] += 1;
                    }
                }

                let mut median = histogram_select(&hist, self.median_index) as usize;
                let mut cumulative = histogram_cumulative(&hist, median);

                output[(oy * out_w) * bands + band] = median as u8;

                for ox in 1..out_w {
                    let remove_x = ox - 1;
                    let add_x = ox + window_w - 1;
                    for wy in 0..window_h {
                        let row_base = (oy + wy) * in_w;
                        let remove_sample = input[(row_base + remove_x) * bands + band];
                        let add_sample = input[(row_base + add_x) * bands + band];
                        hist[remove_sample as usize] -= 1;
                        if usize::from(remove_sample) <= median {
                            cumulative -= 1;
                        }
                        hist[add_sample as usize] += 1;
                        if usize::from(add_sample) <= median {
                            cumulative += 1;
                        }
                    }

                    while cumulative <= self.median_index {
                        median += 1;
                        cumulative += hist[median] as usize;
                    }

                    while cumulative - hist[median] as usize > self.median_index {
                        cumulative -= hist[median] as usize;
                        median -= 1;
                    }

                    output[(oy * out_w + ox) * bands + band] = median as u8;
                }
            }
        }
    }
}

impl<F> Op for Median<F>
where
    F: NumericBand,
    F::Sample: Copy + Default + PartialOrd,
{
    type Input = F;
    type Output = F;
    type State = MedianState<F::Sample>;

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
        MedianState {
            scratch: vec![F::Sample::default(); self.window_area],
        }
    }

    #[inline]
    fn process_region(
        &self,
        state: &mut Self::State,
        input: &Tile<Self::Input>,
        output: &mut TileMut<Self::Output>,
    ) {
        self.process_dispatch(state, input, output);
    }
}

#[inline]
fn partial_cmp_or_equal<T: PartialOrd>(lhs: &T, rhs: &T) -> Ordering {
    lhs.partial_cmp(rhs)
        .map_or(Ordering::Equal, |ordering| ordering)
}

#[inline]
fn histogram_select(hist: &[u32; 256], index: usize) -> u8 {
    let mut sum = 0usize;
    for (value, count) in hist.iter().enumerate() {
        sum += *count as usize;
        if sum > index {
            return value as u8;
        }
    }

    u8::MAX
}

#[inline]
fn histogram_cumulative(hist: &[u32; 256], value: usize) -> usize {
    hist[..=value].iter().map(|count| *count as usize).sum()
}

#[inline(always)]
const fn compare_swap_u8(lhs: &mut u8, rhs: &mut u8) {
    if *lhs > *rhs {
        std::mem::swap(lhs, rhs);
    }
}

#[inline(always)]
const fn median9_u8_scalar(
    mut a0: u8,
    mut a1: u8,
    mut a2: u8,
    mut a3: u8,
    mut a4: u8,
    mut a5: u8,
    mut a6: u8,
    mut a7: u8,
    mut a8: u8,
) -> u8 {
    compare_swap_u8(&mut a1, &mut a2);
    compare_swap_u8(&mut a4, &mut a5);
    compare_swap_u8(&mut a7, &mut a8);
    compare_swap_u8(&mut a0, &mut a1);
    compare_swap_u8(&mut a3, &mut a4);
    compare_swap_u8(&mut a6, &mut a7);
    compare_swap_u8(&mut a1, &mut a2);
    compare_swap_u8(&mut a4, &mut a5);
    compare_swap_u8(&mut a7, &mut a8);
    compare_swap_u8(&mut a0, &mut a3);
    compare_swap_u8(&mut a5, &mut a8);
    compare_swap_u8(&mut a4, &mut a7);
    compare_swap_u8(&mut a3, &mut a6);
    compare_swap_u8(&mut a1, &mut a4);
    compare_swap_u8(&mut a2, &mut a5);
    compare_swap_u8(&mut a4, &mut a7);
    compare_swap_u8(&mut a4, &mut a2);
    compare_swap_u8(&mut a6, &mut a4);
    compare_swap_u8(&mut a4, &mut a2);
    a4
}

#[inline]
fn median_3x3_u8_scalar(
    row0: &[u8],
    row1: &[u8],
    row2: &[u8],
    bands: usize,
    output_row: &mut [u8],
) {
    for (((((((((a0, a1), a2), a3), a4), a5), a6), a7), a8), out_sample) in row0
        .iter()
        .zip(row0[bands..].iter())
        .zip(row0[2 * bands..].iter())
        .zip(row1.iter())
        .zip(row1[bands..].iter())
        .zip(row1[2 * bands..].iter())
        .zip(row2.iter())
        .zip(row2[bands..].iter())
        .zip(row2[2 * bands..].iter())
        .zip(output_row.iter_mut())
    {
        *out_sample = median9_u8_scalar(*a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn compare_swap_u8x16(
    lhs: &mut std::arch::aarch64::uint8x16_t,
    rhs: &mut std::arch::aarch64::uint8x16_t,
) {
    use std::arch::aarch64::{vmaxq_u8, vminq_u8};

    // SAFETY: the NEON min intrinsic reads only the two register values passed by value.
    let lo = unsafe { vminq_u8(*lhs, *rhs) };
    // SAFETY: the NEON max intrinsic reads only the two register values passed by value.
    let hi = unsafe { vmaxq_u8(*lhs, *rhs) };
    *lhs = lo;
    *rhs = hi;
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn median9_u8x16(
    mut a0: std::arch::aarch64::uint8x16_t,
    mut a1: std::arch::aarch64::uint8x16_t,
    mut a2: std::arch::aarch64::uint8x16_t,
    mut a3: std::arch::aarch64::uint8x16_t,
    mut a4: std::arch::aarch64::uint8x16_t,
    mut a5: std::arch::aarch64::uint8x16_t,
    mut a6: std::arch::aarch64::uint8x16_t,
    mut a7: std::arch::aarch64::uint8x16_t,
    mut a8: std::arch::aarch64::uint8x16_t,
) -> std::arch::aarch64::uint8x16_t {
    // SAFETY: every compare/swap call only permutes the nine NEON registers passed by value and relies only on local mutable bindings.
    unsafe {
        compare_swap_u8x16(&mut a1, &mut a2);
        compare_swap_u8x16(&mut a4, &mut a5);
        compare_swap_u8x16(&mut a7, &mut a8);
        compare_swap_u8x16(&mut a0, &mut a1);
        compare_swap_u8x16(&mut a3, &mut a4);
        compare_swap_u8x16(&mut a6, &mut a7);
        compare_swap_u8x16(&mut a1, &mut a2);
        compare_swap_u8x16(&mut a4, &mut a5);
        compare_swap_u8x16(&mut a7, &mut a8);
        compare_swap_u8x16(&mut a0, &mut a3);
        compare_swap_u8x16(&mut a5, &mut a8);
        compare_swap_u8x16(&mut a4, &mut a7);
        compare_swap_u8x16(&mut a3, &mut a6);
        compare_swap_u8x16(&mut a1, &mut a4);
        compare_swap_u8x16(&mut a2, &mut a5);
        compare_swap_u8x16(&mut a4, &mut a7);
        compare_swap_u8x16(&mut a4, &mut a2);
        compare_swap_u8x16(&mut a6, &mut a4);
        compare_swap_u8x16(&mut a4, &mut a2);
    }
    a4
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn median_3x3_u8_neon(
    row0: &[u8],
    row1: &[u8],
    row2: &[u8],
    bands: usize,
    output_row: &mut [u8],
) {
    use std::arch::aarch64::{vld1q_u8, vst1q_u8};

    let chunks = output_row.len() / 16;
    let remainder_start = chunks * 16;

    for chunk in 0..chunks {
        let offset = chunk * 16;
        // SAFETY: each row carries the two horizontal halo pixels needed by the 3×3 kernel, so loads stay in-bounds and the store fits.
        let median = unsafe {
            median9_u8x16(
                vld1q_u8(row0.as_ptr().add(offset)),
                vld1q_u8(row0.as_ptr().add(offset + bands)),
                vld1q_u8(row0.as_ptr().add(offset + 2 * bands)),
                vld1q_u8(row1.as_ptr().add(offset)),
                vld1q_u8(row1.as_ptr().add(offset + bands)),
                vld1q_u8(row1.as_ptr().add(offset + 2 * bands)),
                vld1q_u8(row2.as_ptr().add(offset)),
                vld1q_u8(row2.as_ptr().add(offset + bands)),
                vld1q_u8(row2.as_ptr().add(offset + 2 * bands)),
            )
        };
        // SAFETY: `offset + 16 <= output_row.len()` for every SIMD chunk.
        unsafe {
            vst1q_u8(output_row.as_mut_ptr().add(offset), median);
        }
    }

    median_3x3_u8_scalar(
        &row0[remainder_start..],
        &row1[remainder_start..],
        &row2[remainder_start..],
        bands,
        &mut output_row[remainder_start..],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        format::{F32, U8},
        image::{Region, Tile, TileMut},
    };
    use proptest::prelude::*;

    fn run_median_u8_bands(
        op: &Median<U8>,
        input_region: Region,
        output_region: Region,
        bands: u32,
        input_data: &[u8],
    ) -> Vec<u8> {
        let mut state = op.start();
        let input = Tile::<U8>::new(input_region, bands, input_data);
        let mut output_data =
            vec![0u8; (output_region.width * output_region.height * bands) as usize];
        let mut output = TileMut::<U8>::new(output_region, bands, &mut output_data);
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    fn run_median_u8(
        op: &Median<U8>,
        input_region: Region,
        output_region: Region,
        input_data: &[u8],
    ) -> Vec<u8> {
        run_median_u8_bands(op, input_region, output_region, 1, input_data)
    }

    fn median_reference_sort_bands(
        input: &[u8],
        input_region: Region,
        output_region: Region,
        bands: u32,
        window_w: u32,
        window_h: u32,
    ) -> Vec<u8> {
        let out_w = output_region.width as usize;
        let out_h = output_region.height as usize;
        let in_w = input_region.width as usize;
        let bands = bands as usize;
        let window_w = window_w as usize;
        let window_h = window_h as usize;
        let median_index = (window_w * window_h) / 2;
        let mut output = vec![0u8; out_w * out_h * bands];
        let mut scratch = vec![0u8; window_w * window_h];

        for oy in 0..out_h {
            for ox in 0..out_w {
                for band in 0..bands {
                    let mut write_idx = 0usize;
                    for wy in 0..window_h {
                        let row_base = ((oy + wy) * in_w + ox) * bands + band;
                        for wx in 0..window_w {
                            scratch[write_idx] = input[row_base + wx * bands];
                            write_idx += 1;
                        }
                    }
                    let (_, median, _) =
                        scratch.select_nth_unstable_by(median_index, partial_cmp_or_equal::<u8>);
                    output[(oy * out_w + ox) * bands + band] = *median;
                }
            }
        }

        output
    }

    fn median_reference_sort(
        input: &[u8],
        input_region: Region,
        output_region: Region,
        window_w: u32,
        window_h: u32,
    ) -> Vec<u8> {
        median_reference_sort_bands(input, input_region, output_region, 1, window_w, window_h)
    }

    fn run_median_f32(
        op: &Median<F32>,
        input_region: Region,
        output_region: Region,
        input_data: &[f32],
    ) -> Vec<f32> {
        let mut state = op.start();
        let input = Tile::<F32>::new(input_region, 1, input_data);
        let mut output_data = vec![0.0f32; output_region.pixel_count()];
        let mut output = TileMut::<F32>::new(output_region, 1, &mut output_data);
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    #[test]
    fn rejects_even_windows() {
        assert!(Median::<U8>::new(2, 3).is_err());
        assert!(Median::<U8>::new(3, 2).is_err());
    }

    #[test]
    fn rejects_zero_windows_and_reports_metadata() {
        assert!(Median::<U8>::new(0, 3).is_err());
        assert!(Median::<U8>::new(3, 0).is_err());

        let op = Median::<U8>::new(5, 3).unwrap();
        let output = Region::new(10, 11, 7, 9);

        assert_eq!(op.demand_hint(), DemandHint::SmallTile);
        assert_eq!(
            op.required_input_region(&output),
            Region::new(8, 10, 11, 11)
        );
        assert_eq!(
            op.node_spec(7, 9),
            NodeSpec {
                input_tile_w: 11,
                input_tile_h: 11,
                output_tile_w: 7,
                output_tile_h: 9,
                coordinate_driven_source: None,
            }
        );
    }

    #[test]
    fn histogram_fast_path_is_enabled_for_large_u8_windows() {
        let op = Median::<U8>::new(15, 15).unwrap();
        assert!(op.should_use_histogram_fast_path());
    }

    #[test]
    fn median_3x3_preserves_uniform_max() {
        let op = Median::<U8>::new(3, 3).unwrap();
        let input_region = Region::new(0, 0, 5, 5);
        let output_region = Region::new(0, 0, 3, 3);
        let input_data = vec![u8::MAX; (input_region.width * input_region.height) as usize];
        let output = run_median_u8(&op, input_region, output_region, &input_data);
        assert!(output.iter().all(|&sample| sample == u8::MAX));
    }

    #[test]
    fn uniform_max_window_stays_max() {
        let op = Median::<U8>::new(15, 15).unwrap();
        let input_region = Region::new(0, 0, 17, 17);
        let output_region = Region::new(0, 0, 3, 3);
        let input_data = vec![u8::MAX; (input_region.width * input_region.height) as usize];
        let output = run_median_u8(&op, input_region, output_region, &input_data);
        assert!(output.iter().all(|&sample| sample == u8::MAX));
    }

    #[test]
    fn u8_3x3_large_row_matches_reference() {
        let op = Median::<U8>::new(3, 3).unwrap();
        let input_region = Region::new(0, 0, 20, 5);
        let output_region = Region::new(0, 0, 18, 3);
        let pixels = (0..(input_region.pixel_count() as usize))
            .map(|index| ((index * 37 + 11) % 251) as u8)
            .collect::<Vec<_>>();

        let fast = run_median_u8(&op, input_region, output_region, &pixels);
        let reference = median_reference_sort(&pixels, input_region, output_region, 3, 3);

        assert_eq!(fast, reference);
    }

    #[test]
    fn f32_select_path_matches_known_reference() {
        let op = Median::<F32>::new(3, 3).unwrap();
        let input_region = Region::new(0, 0, 5, 3);
        let output_region = Region::new(0, 0, 3, 1);
        let pixels = vec![
            1.0, 8.0, 3.0, 6.0, 5.0, 9.0, 2.0, 7.0, 4.0, 0.0, 5.0, 3.0, 8.0, 1.0, 6.0,
        ];

        let output = run_median_f32(&op, input_region, output_region, &pixels);

        assert_eq!(output, vec![0.0, 3.0, 5.0]);
    }

    proptest! {
        #[test]
        fn identity_window_1x1_is_noop(
            pixels in proptest::collection::vec(0u8..=255u8, 1..=64)
        ) {
            let op = Median::<U8>::new(1, 1).unwrap();
            let region = Region::new(0, 0, pixels.len() as u32, 1);
            let output = run_median_u8(&op, region, region, &pixels);
            prop_assert_eq!(output, pixels);
        }

        #[test]
        fn u8_histogram_fast_path_matches_sort_reference_large_window(
            pixels in proptest::collection::vec(0u8..=255u8, 17 * 17)
        ) {
            let op = Median::<U8>::new(15, 15).unwrap();
            let input_region = Region::new(0, 0, 17, 17);
            let output_region = Region::new(0, 0, 3, 3);
            prop_assert!(op.should_use_histogram_fast_path());

            let fast = run_median_u8(&op, input_region, output_region, &pixels);
            let reference = median_reference_sort(&pixels, input_region, output_region, 15, 15);

            prop_assert_eq!(fast, reference);
        }

        #[test]
        fn u8_3x3_fast_path_matches_sort_reference(
            pixels in proptest::collection::vec(0u8..=255u8, 25)
        ) {
            let op = Median::<U8>::new(3, 3).unwrap();
            let input_region = Region::new(0, 0, 5, 5);
            let output_region = Region::new(0, 0, 3, 3);

            let fast = run_median_u8(&op, input_region, output_region, &pixels);
            let reference = median_reference_sort(&pixels, input_region, output_region, 3, 3);

            prop_assert_eq!(fast, reference);
        }

        #[test]
        fn u8_rgb_3x3_fast_path_matches_sort_reference(
            pixels in proptest::collection::vec(0u8..=255u8, 5 * 5 * 3)
        ) {
            let op = Median::<U8>::new(3, 3).unwrap();
            let input_region = Region::new(0, 0, 5, 5);
            let output_region = Region::new(0, 0, 3, 3);

            let fast = run_median_u8_bands(&op, input_region, output_region, 3, &pixels);
            let reference =
                median_reference_sort_bands(&pixels, input_region, output_region, 3, 3, 3);

            prop_assert_eq!(fast, reference);
        }
    }
}

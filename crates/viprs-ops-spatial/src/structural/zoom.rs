#![allow(dead_code)]
// REASON: structural bridge wiring is staged for future pipeline-builder exposure.

use std::marker::PhantomData;
use viprs_core::{
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    op::{NodeSpec, Op, OperationBridge},
};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{vld1q_u8, vst1q_u8, vzipq_u8};

#[inline]
fn fill_repeated_pixel<T: Copy>(dst: &mut [T], pixel: &[T]) {
    debug_assert!(!pixel.is_empty());
    debug_assert_eq!(dst.len() % pixel.len(), 0);

    let pixel_len = pixel.len();
    dst[..pixel_len].copy_from_slice(pixel);

    let mut filled = pixel_len;
    while filled < dst.len() {
        let copy_len = filled.min(dst.len() - filled);
        dst.copy_within(0..copy_len, filled);
        filled += copy_len;
    }
}

#[inline]
fn duplicate_rows<T: Copy>(dst: &mut [T], row_len: usize, rows: usize) {
    debug_assert_eq!(dst.len(), row_len * rows);

    let mut filled_rows = 1usize;
    while filled_rows < rows {
        let rows_to_copy = filled_rows.min(rows - filled_rows);
        let copy_len = rows_to_copy * row_len;
        dst.copy_within(0..copy_len, filled_rows * row_len);
        filled_rows += rows_to_copy;
    }
}

#[inline]
fn process_zoom_partial<T: Copy>(
    xfac: u32,
    yfac: u32,
    input: &Tile<'_, impl BandFormat<Sample = T>>,
    output: &mut TileMut<'_, impl BandFormat<Sample = T>>,
) {
    let bands = input.bands as usize;
    let input_width = input.region.width as usize;
    let out_w = output.region.width as usize;
    let out_h = output.region.height as usize;
    let out_stride = out_w * bands;
    let input_stride = input_width * bands;
    let in_x0 = input.region.x as u32;
    let in_y0 = input.region.y as u32;
    let out_x0 = output.region.x as u32;
    let out_y0 = output.region.y as u32;

    let mut row = 0usize;
    while row < out_h {
        let global_y = out_y0 + row as u32;
        let src_row = (global_y / yfac - in_y0) as usize;
        let run_y = ((yfac - (global_y % yfac)) as usize).min(out_h - row);
        let src_row_start = src_row * input_stride;
        let src_row_data = &input.data[src_row_start..src_row_start + input_stride];
        let output_block_start = row * out_stride;
        let output_block_end = output_block_start + run_y * out_stride;
        let output_block = &mut output.data[output_block_start..output_block_end];
        let (first_row, repeated_rows) = output_block.split_at_mut(out_stride);

        let mut col = 0usize;
        while col < out_w {
            let global_x = out_x0 + col as u32;
            let src_col = (global_x / xfac - in_x0) as usize;
            let run_x = ((xfac - (global_x % xfac)) as usize).min(out_w - col);
            let src_start = src_col * bands;
            let dst_start = col * bands;
            let src_pixel = &src_row_data[src_start..src_start + bands];

            if bands == 1 {
                first_row[col..col + run_x].fill(src_pixel[0]);
            } else {
                let dst_end = dst_start + run_x * bands;
                let (first_pixel, repeated_pixels) =
                    first_row[dst_start..dst_end].split_at_mut(bands);
                first_pixel.copy_from_slice(src_pixel);
                for chunk in repeated_pixels.chunks_exact_mut(bands) {
                    chunk.copy_from_slice(first_pixel);
                }
            }

            col += run_x;
        }

        for chunk in repeated_rows.chunks_exact_mut(out_stride) {
            chunk.copy_from_slice(first_row);
        }

        row += run_y;
    }
}

#[inline]
fn process_zoom_whole<T: Copy>(
    xfac: u32,
    yfac: u32,
    input: &Tile<'_, impl BandFormat<Sample = T>>,
    output: &mut TileMut<'_, impl BandFormat<Sample = T>>,
) {
    let bands = input.bands as usize;
    let input_width = input.region.width as usize;
    let out_w = output.region.width as usize;
    let out_h = output.region.height as usize;
    let out_stride = out_w * bands;
    let input_stride = input_width * bands;
    let src_x0 = (output.region.x as u32 / xfac - input.region.x as u32) as usize;
    let src_y0 = (output.region.y as u32 / yfac - input.region.y as u32) as usize;
    let src_w = (output.region.width / xfac) as usize;
    let src_h = (output.region.height / yfac) as usize;
    let repeat_x = xfac as usize;
    let repeat_y = yfac as usize;

    for src_row in 0..src_h {
        let src_row_start = (src_y0 + src_row) * input_stride + src_x0 * bands;
        let src_row_data = &input.data[src_row_start..src_row_start + src_w * bands];
        let dst_row_start = src_row * repeat_y * out_stride;
        let dst_block_end = dst_row_start + repeat_y * out_stride;
        let dst_block = &mut output.data[dst_row_start..dst_block_end];
        let (first_row, _) = dst_block.split_at_mut(out_stride);

        let mut src = 0usize;
        let mut dst = 0usize;
        while src < src_row_data.len() {
            let src_pixel = &src_row_data[src..src + bands];
            fill_repeated_pixel(&mut first_row[dst..dst + repeat_x * bands], src_pixel);
            src += bands;
            dst += repeat_x * bands;
        }

        if repeat_y > 1 {
            duplicate_rows(dst_block, out_stride, repeat_y);
        }
    }

    debug_assert_eq!(src_h * repeat_y, out_h);
}

#[inline]
fn process_zoom_u8_x2(
    yfac: u32,
    input: &Tile<'_, impl BandFormat<Sample = u8>>,
    output: &mut TileMut<'_, impl BandFormat<Sample = u8>>,
) {
    let input_width = input.region.width as usize;
    let out_w = output.region.width as usize;
    let out_h = output.region.height as usize;
    let in_x0 = input.region.x as u32;
    let in_y0 = input.region.y as u32;
    let out_x0 = output.region.x as u32;
    let out_y0 = output.region.y as u32;

    let mut row = 0usize;
    while row < out_h {
        let global_y = out_y0 + row as u32;
        let src_row = (global_y / yfac - in_y0) as usize;
        let run_y = ((yfac - (global_y % yfac)) as usize).min(out_h - row);
        let src_row_start = src_row * input_width;
        let src_row_data = &input.data[src_row_start..src_row_start + input_width];
        let output_block_start = row * out_w;
        let output_block_end = output_block_start + run_y * out_w;
        let output_block = &mut output.data[output_block_start..output_block_end];
        let (first_row, repeated_rows) = output_block.split_at_mut(out_w);

        let mut dst_col = 0usize;
        let mut src_col = (out_x0 / 2 - in_x0) as usize;

        if out_x0 & 1 == 1 {
            first_row[0] = src_row_data[src_col];
            dst_col = 1;
            src_col += 1;
        }

        let pair_count = (out_w - dst_col) / 2;
        if pair_count > 0 {
            let src_pairs = &src_row_data[src_col..src_col + pair_count];
            let dst_pairs = &mut first_row[dst_col..dst_col + pair_count * 2];
            duplicate_u8_pairs(src_pairs, dst_pairs);
            dst_col += pair_count * 2;
            src_col += pair_count;
        }

        if dst_col < out_w {
            first_row[dst_col] = src_row_data[src_col];
        }

        for chunk in repeated_rows.chunks_exact_mut(out_w) {
            chunk.copy_from_slice(first_row);
        }

        row += run_y;
    }
}

#[inline]
fn expand_rgb_row_u8_x2(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len() * 2, dst.len());
    debug_assert_eq!(src.len() % 3, 0);

    let mut src_idx = 0usize;
    let mut dst_idx = 0usize;
    while src_idx < src.len() {
        let r = src[src_idx];
        let g = src[src_idx + 1];
        let b = src[src_idx + 2];
        dst[dst_idx] = r;
        dst[dst_idx + 1] = g;
        dst[dst_idx + 2] = b;
        dst[dst_idx + 3] = r;
        dst[dst_idx + 4] = g;
        dst[dst_idx + 5] = b;
        src_idx += 3;
        dst_idx += 6;
    }
}

#[inline]
fn process_zoom_u8_rgb_x2_whole(
    yfac: u32,
    input: &Tile<'_, impl BandFormat<Sample = u8>>,
    output: &mut TileMut<'_, impl BandFormat<Sample = u8>>,
) {
    let input_width = input.region.width as usize;
    let out_w = output.region.width as usize;
    let out_stride = out_w * 3;
    let input_stride = input_width * 3;
    let src_x0 = (output.region.x as u32 / 2 - input.region.x as u32) as usize;
    let src_y0 = (output.region.y as u32 / yfac - input.region.y as u32) as usize;
    let src_w = (output.region.width / 2) as usize;
    let src_h = (output.region.height / yfac) as usize;
    let repeat_y = yfac as usize;

    for src_row in 0..src_h {
        let src_row_start = (src_y0 + src_row) * input_stride + src_x0 * 3;
        let src_row_data = &input.data[src_row_start..src_row_start + src_w * 3];
        let dst_row_start = src_row * repeat_y * out_stride;
        let dst_block_end = dst_row_start + repeat_y * out_stride;
        let dst_block = &mut output.data[dst_row_start..dst_block_end];
        let (first_row, _) = dst_block.split_at_mut(out_stride);

        expand_rgb_row_u8_x2(src_row_data, first_row);

        if repeat_y > 1 {
            duplicate_rows(dst_block, out_stride, repeat_y);
        }
    }
}

#[inline]
fn duplicate_u8_pairs(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(dst.len(), src.len() * 2);

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: `dst` is exactly 2x the length of `src`, both slices are valid for the
        // duration of the call, and the helper only performs in-bounds loads/stores.
        unsafe {
            duplicate_u8_pairs_neon(src, dst);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    duplicate_u8_pairs_scalar(src, dst);
}

#[inline]
fn duplicate_u8_pairs_scalar(src: &[u8], dst: &mut [u8]) {
    let mut out = 0usize;
    for &pixel in src {
        dst[out] = pixel;
        dst[out + 1] = pixel;
        out += 2;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn duplicate_u8_pairs_neon(src: &[u8], dst: &mut [u8]) {
    let mut in_idx = 0usize;
    let mut out_idx = 0usize;

    while in_idx + 16 <= src.len() {
        // SAFETY: the loop guard guarantees 16 readable bytes from `src + in_idx`.
        let pixels = unsafe { vld1q_u8(src.as_ptr().add(in_idx)) };
        // SAFETY: aarch64 guarantees NEON, and this intrinsic only operates on registers.
        let duplicated = unsafe { vzipq_u8(pixels, pixels) };
        // SAFETY: the destination slice is exactly 2x `src`, so 32 writable bytes remain.
        unsafe {
            vst1q_u8(dst.as_mut_ptr().add(out_idx), duplicated.0);
            vst1q_u8(dst.as_mut_ptr().add(out_idx + 16), duplicated.1);
        }
        in_idx += 16;
        out_idx += 32;
    }

    duplicate_u8_pairs_scalar(&src[in_idx..], &mut dst[out_idx..]);
}

/// Integer nearest-neighbour upscale by `xfac × yfac`.
pub struct Zoom<F: BandFormat> {
    xfac: u32,
    yfac: u32,
    _format: PhantomData<F>,
}

impl<F: BandFormat + Send + Sync> Zoom<F> {
    #[must_use]
    /// Creates a new `Zoom`.
    pub fn new(xfac: u32, yfac: u32) -> Self {
        debug_assert!(xfac >= 1, "Zoom: xfac must be >= 1");
        debug_assert!(yfac >= 1, "Zoom: yfac must be >= 1");
        Self {
            xfac,
            yfac,
            _format: PhantomData,
        }
    }
}

impl<F: BandFormat> Op for Zoom<F>
where
    F::Sample: bytemuck::Pod + Copy,
{
    type Input = F;
    type Output = F;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::FatStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        let left = output.x as u32 / self.xfac;
        let top = output.y as u32 / self.yfac;
        let right = (output.x as u32 + output.width - 1) / self.xfac;
        let bottom = (output.y as u32 + output.height - 1) / self.yfac;
        Region::new(left as i32, top as i32, right - left + 1, bottom - top + 1)
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec {
            input_tile_w: tile_w.div_ceil(self.xfac) + u32::from(self.xfac > 1),
            input_tile_h: tile_h.div_ceil(self.yfac) + u32::from(self.yfac > 1),
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        let whole_aligned = (output.region.x as u32).is_multiple_of(self.xfac)
            && (output.region.y as u32).is_multiple_of(self.yfac)
            && output.region.width.is_multiple_of(self.xfac)
            && output.region.height.is_multiple_of(self.yfac);

        if input.bands == 1 && self.xfac == 2 && std::mem::size_of::<F::Sample>() == 1 {
            let input_data: &[u8] = bytemuck::cast_slice(input.data);
            let output_data: &mut [u8] = bytemuck::cast_slice_mut(output.data);
            let input_u8 = Tile::<viprs_core::format::U8>::new(input.region, 1, input_data);
            let mut output_u8 =
                TileMut::<viprs_core::format::U8>::new(output.region, 1, output_data);
            process_zoom_u8_x2(self.yfac, &input_u8, &mut output_u8);
            return;
        }

        if input.bands == 3
            && self.xfac == 2
            && whole_aligned
            && std::mem::size_of::<F::Sample>() == 1
        {
            let input_data: &[u8] = bytemuck::cast_slice(input.data);
            let output_data: &mut [u8] = bytemuck::cast_slice_mut(output.data);
            let input_u8 = Tile::<viprs_core::format::U8>::new(input.region, 3, input_data);
            let mut output_u8 =
                TileMut::<viprs_core::format::U8>::new(output.region, 3, output_data);
            process_zoom_u8_rgb_x2_whole(self.yfac, &input_u8, &mut output_u8);
            return;
        }

        if whole_aligned {
            process_zoom_whole(self.xfac, self.yfac, input, output);
        } else {
            process_zoom_partial(self.xfac, self.yfac, input, output);
        }
    }
}

pub(crate) struct ZoomBridge<F: BandFormat>
where
    F::Sample: bytemuck::Pod + Copy,
{
    inner: OperationBridge<Zoom<F>>,
}

impl<F: BandFormat> ZoomBridge<F>
where
    F::Sample: bytemuck::Pod + Copy,
{
    pub fn new(xfac: u32, yfac: u32, bands: u32) -> Self {
        Self {
            inner: OperationBridge::new(Zoom::new(xfac, yfac), bands),
        }
    }
}

impl<F: BandFormat> viprs_core::op::DynOperation for ZoomBridge<F>
where
    F::Sample: bytemuck::Pod + Copy + Send,
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
        input_w.saturating_mul(self.inner.op.xfac)
    }

    fn output_height(&self, input_h: u32) -> u32 {
        input_h.saturating_mul(self.inner.op.yfac)
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

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::{BandFormatId, U8},
        image::{Region, Tile, TileMut},
        op::DynOperation,
    };

    fn expected_zoom(
        input: &[u8],
        input_width: usize,
        input_region: Region,
        output_region: Region,
        bands: usize,
        xfac: u32,
        yfac: u32,
    ) -> Vec<u8> {
        let output_width = output_region.width as usize;
        let output_height = output_region.height as usize;
        let mut expected = vec![0u8; output_width * output_height * bands];

        for row in 0..output_height {
            let global_y = output_region.y as u32 + row as u32;
            let src_row = (global_y / yfac - input_region.y as u32) as usize;
            for col in 0..output_width {
                let global_x = output_region.x as u32 + col as u32;
                let src_col = (global_x / xfac - input_region.x as u32) as usize;
                let src = (src_row * input_width + src_col) * bands;
                let dst = (row * output_width + col) * bands;
                expected[dst..dst + bands].copy_from_slice(&input[src..src + bands]);
            }
        }

        expected
    }

    #[test]
    fn required_input_region_for_partial_tile_is_exact() {
        let op = Zoom::<U8>::new(2, 3);
        let output = Region::new(1, 2, 3, 4);
        let input = op.required_input_region(&output);
        assert_eq!(input, Region::new(0, 0, 2, 2));
    }

    #[test]
    fn output_dimensions_scale_by_factor() {
        let bridge = ZoomBridge::<U8>::new(3, 2, 1);
        assert_eq!(bridge.output_width(4), 12);
        assert_eq!(bridge.output_height(5), 10);
    }

    #[test]
    fn node_spec_requests_overlap_for_non_identity_scale() {
        let op = Zoom::<U8>::new(3, 2);
        let spec = op.node_spec(5, 6);
        assert_eq!(spec.input_tile_w, 3);
        assert_eq!(spec.input_tile_h, 4);
        assert_eq!(spec.output_tile_w, 5);
        assert_eq!(spec.output_tile_h, 6);
    }

    #[test]
    fn bridge_metadata_and_dyn_dispatch_cover_offset_tiles() {
        let bridge = ZoomBridge::<U8>::new(2, 3, 2);
        assert_eq!(bridge.input_format(), BandFormatId::U8);
        assert_eq!(bridge.output_format(), BandFormatId::U8);
        assert_eq!(bridge.bands(), 2);
        assert_eq!(bridge.demand_hint(), DemandHint::FatStrip);

        let output_region = Region::new(1, 2, 3, 4);
        let input_region = bridge.required_input_region(&output_region);
        let input = vec![0u8, 1, 2, 3, 4, 5, 6, 7];
        let mut output = vec![0u8; output_region.pixel_count() * 2];
        let mut state = bridge.dyn_start();
        bridge.dyn_process_region(
            state.as_mut(),
            &input,
            &mut output,
            input_region,
            output_region,
        );

        let expected = expected_zoom(&input, 2, input_region, output_region, 2, 2, 3);
        assert_eq!(output, expected);
    }

    #[test]
    fn process_region_repeats_nearest_neighbour() {
        let op = Zoom::<U8>::new(2, 2);
        let output_region = Region::new(0, 0, 4, 4);
        let input_region = op.required_input_region(&output_region);
        let input_data = vec![1u8, 2, 3, 4];
        let mut output_data = vec![0u8; 16];
        let input = Tile::<U8>::new(input_region, 1, &input_data);
        let mut output = TileMut::<U8>::new(output_region, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(
            output_data,
            vec![1u8, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4,]
        );
    }

    #[test]
    fn process_region_handles_odd_offset_tiles_on_fast_u8_path() {
        let op = Zoom::<U8>::new(2, 2);
        let output_region = Region::new(1, 1, 3, 3);
        let input_region = op.required_input_region(&output_region);
        let input_data = vec![1u8, 2, 3, 4];
        let mut output_data = vec![0u8; output_region.pixel_count()];
        let input = Tile::<U8>::new(input_region, 1, &input_data);
        let mut output = TileMut::<U8>::new(output_region, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);

        assert_eq!(
            output_data,
            expected_zoom(&input_data, 2, input_region, output_region, 1, 2, 2)
        );
    }

    proptest! {
        #[test]
        fn zoom_factor_1_is_identity(rows in 1usize..=6, cols in 1usize..=6) {
            let pixels: Vec<u8> = (0..(rows * cols) as u8).collect();
            let output_region = Region::new(0, 0, cols as u32, rows as u32);
            let op = Zoom::<U8>::new(1, 1);
            let input_region = op.required_input_region(&output_region);
            let mut output_data = vec![0u8; pixels.len()];
            let input = Tile::<U8>::new(input_region, 1, &pixels);
            let mut output = TileMut::<U8>::new(output_region, 1, &mut output_data);
            let mut state = ();
            op.process_region(&mut state, &input, &mut output);
            prop_assert_eq!(output_data, pixels);
        }

        #[test]
        fn zoom_single_pixel_input_expands_to_uniform_block(
            pixel in prop::collection::vec(any::<u8>(), 1..=4),
            xfac in 1u32..=8,
            yfac in 1u32..=8,
        ) {
            let bands = pixel.len() as u32;
            let op = Zoom::<U8>::new(xfac, yfac);
            let output_region = Region::new(0, 0, xfac, yfac);
            let input_region = op.required_input_region(&output_region);
            prop_assert_eq!(input_region, Region::new(0, 0, 1, 1));

            let mut output_data = vec![0u8; (xfac * yfac) as usize * pixel.len()];
            let input = Tile::<U8>::new(input_region, bands, &pixel);
            let mut output = TileMut::<U8>::new(output_region, bands, &mut output_data);
            let mut state = ();
            op.process_region(&mut state, &input, &mut output);

            let mut expected = Vec::with_capacity(output_data.len());
            for _ in 0..(xfac * yfac) {
                expected.extend_from_slice(&pixel);
            }
            prop_assert_eq!(output_data, expected);
        }

        #[test]
        fn zoom_one_pixel_wide_images_repeat_each_row_vertically(
            rows in 1usize..=8,
            bands in 1usize..=4,
            yfac in 1u32..=8,
        ) {
            let pixels = (0..rows * bands)
                .map(|idx| (idx % 251) as u8)
                .collect::<Vec<_>>();
            let op = Zoom::<U8>::new(1, yfac);
            let output_region = Region::new(0, 0, 1, (rows as u32) * yfac);
            let input_region = op.required_input_region(&output_region);
            let mut output_data = vec![0u8; output_region.pixel_count() * bands];
            let input = Tile::<U8>::new(input_region, bands as u32, &pixels);
            let mut output = TileMut::<U8>::new(output_region, bands as u32, &mut output_data);
            let mut state = ();
            op.process_region(&mut state, &input, &mut output);

            let expected = expected_zoom(&pixels, 1, input_region, output_region, bands, 1, yfac);
            prop_assert_eq!(output_data, expected);
        }

        #[test]
        fn zoom_one_pixel_tall_images_repeat_each_column_horizontally(
            cols in 1usize..=8,
            bands in 1usize..=4,
            xfac in 1u32..=8,
        ) {
            let pixels = (0..cols * bands)
                .map(|idx| (idx % 251) as u8)
                .collect::<Vec<_>>();
            let op = Zoom::<U8>::new(xfac, 1);
            let output_region = Region::new(0, 0, (cols as u32) * xfac, 1);
            let input_region = op.required_input_region(&output_region);
            let mut output_data = vec![0u8; output_region.pixel_count() * bands];
            let input = Tile::<U8>::new(input_region, bands as u32, &pixels);
            let mut output = TileMut::<U8>::new(output_region, bands as u32, &mut output_data);
            let mut state = ();
            op.process_region(&mut state, &input, &mut output);

            let expected = expected_zoom(&pixels, cols, input_region, output_region, bands, xfac, 1);
            prop_assert_eq!(output_data, expected);
        }

        #[test]
        fn zoom_factor_1_is_identity_for_multiband_images(
            rows in 1usize..=4,
            cols in 1usize..=4,
            bands in 2usize..=4,
        ) {
            let pixels = (0..rows * cols * bands)
                .map(|idx| (idx % 251) as u8)
                .collect::<Vec<_>>();
            let output_region = Region::new(0, 0, cols as u32, rows as u32);
            let op = Zoom::<U8>::new(1, 1);
            let input_region = op.required_input_region(&output_region);
            let mut output_data = vec![0u8; pixels.len()];
            let input = Tile::<U8>::new(input_region, bands as u32, &pixels);
            let mut output = TileMut::<U8>::new(output_region, bands as u32, &mut output_data);
            let mut state = ();
            op.process_region(&mut state, &input, &mut output);
            prop_assert_eq!(output_data, pixels);
        }

        #[test]
        fn zoom_large_factors_repeat_bottom_right_edge_pixels(
            xfac in 2u32..=8,
            yfac in 2u32..=8,
            bands in 1usize..=4,
        ) {
            let input_width = 2usize;
            let input_height = 2usize;
            let pixels = (0..input_width * input_height * bands)
                .map(|idx| (idx % 251) as u8)
                .collect::<Vec<_>>();
            let output_region = Region::new(0, 0, xfac + 1, yfac + 1);
            let op = Zoom::<U8>::new(xfac, yfac);
            let input_region = op.required_input_region(&output_region);
            let mut output_data = vec![0u8; output_region.pixel_count() * bands];
            let input = Tile::<U8>::new(input_region, bands as u32, &pixels);
            let mut output = TileMut::<U8>::new(output_region, bands as u32, &mut output_data);
            let mut state = ();
            op.process_region(&mut state, &input, &mut output);

            let expected = expected_zoom(
                &pixels,
                input_width,
                input_region,
                output_region,
                bands,
                xfac,
                yfac,
            );
            prop_assert_eq!(output_data, expected);
        }
    }
}

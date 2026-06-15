use crate::domain::{
    format::BandFormatId,
    image::{DemandHint, Region, clamp_i64_to_i32},
    op::{DynOperation, NodeSpec},
};
use std::any::Any;

/// Overlay a sub-image on top of a main image at `(x, y)`.
pub struct Insert {
    main_width: u32,
    main_height: u32,
    sub_width: u32,
    sub_height: u32,
    x: i32,
    y: i32,
    expand: bool,
    bands: u32,
    format: BandFormatId,
}

impl Insert {
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    /// Creates a new `Insert`.
    pub const fn new(
        main_width: u32,
        main_height: u32,
        sub_width: u32,
        sub_height: u32,
        x: i32,
        y: i32,
        expand: bool,
        bands: u32,
        format: BandFormatId,
    ) -> Self {
        Self {
            main_width,
            main_height,
            sub_width,
            sub_height,
            x,
            y,
            expand,
            bands,
            format,
        }
    }

    #[must_use]
    /// Returns or performs output width.
    pub fn output_width(&self) -> u32 {
        if self.expand {
            self.main_width
                .max((self.x.max(0) as u32).saturating_add(self.sub_width))
        } else {
            self.main_width
        }
    }

    #[must_use]
    /// Returns or performs output height.
    pub fn output_height(&self) -> u32 {
        if self.expand {
            self.main_height
                .max((self.y.max(0) as u32).saturating_add(self.sub_height))
        } else {
            self.main_height
        }
    }

    const fn placement(&self, slot: usize) -> (i32, i32, u32, u32) {
        match slot {
            0 => (0, 0, self.main_width, self.main_height),
            1 => (self.x, self.y, self.sub_width, self.sub_height),
            _ => (0, 0, 0, 0),
        }
    }
}

impl DynOperation for Insert {
    fn input_format(&self) -> BandFormatId {
        self.format
    }

    fn output_format(&self) -> BandFormatId {
        self.format
    }

    fn bands(&self) -> u32 {
        self.bands
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn input_slot_count(&self) -> usize {
        2
    }

    fn required_input_region_slot(&self, output: &Region, slot: usize) -> Region {
        let (px, py, pw, ph) = self.placement(slot);
        intersect_and_translate(*output, px, py, pw, ph)
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn output_width(&self, _input_w: u32) -> u32 {
        self.output_width()
    }

    fn output_height(&self, _input_h: u32) -> u32 {
        self.output_height()
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec::identity(tile_w, tile_h)
    }

    fn dyn_start(&self) -> Box<dyn Any + Send> {
        Box::new(())
    }

    fn dyn_process_region(
        &self,
        _state: &mut dyn Any,
        input: &[u8],
        output: &mut [u8],
        _input_region: Region,
        _output_region: Region,
    ) {
        debug_assert!(
            false,
            "Insert: dyn_process_region called on a 2-input node — use dyn_process_region_multi"
        );
        let len = output.len().min(input.len());
        output[..len].copy_from_slice(&input[..len]);
    }

    #[inline]
    fn dyn_process_region_multi(
        &self,
        _state: &mut dyn Any,
        inputs: &[&[u8]],
        output: &mut [u8],
        input_regions: &[Region],
        output_region: Region,
    ) {
        debug_assert_eq!(inputs.len(), 2, "Insert: expected exactly 2 input slices");
        debug_assert_eq!(
            input_regions.len(),
            2,
            "Insert: expected exactly 2 input regions"
        );

        output.fill(0);

        let pixel_bytes = self.bands as usize * sample_size_bytes(self.format);

        paste_region(
            inputs[0],
            input_regions[0],
            output,
            output_region,
            0,
            0,
            pixel_bytes,
        );
        paste_region(
            inputs[1],
            input_regions[1],
            output,
            output_region,
            self.x,
            self.y,
            pixel_bytes,
        );
    }
}

fn intersect_and_translate(output: Region, px: i32, py: i32, pw: u32, ph: u32) -> Region {
    let left = output.x.max(px);
    let top = output.y.max(py);
    let right = clamp_i64_to_i32(i64::from(output.x) + i64::from(output.width))
        .min(clamp_i64_to_i32(i64::from(px) + i64::from(pw)));
    let bottom = clamp_i64_to_i32(i64::from(output.y) + i64::from(output.height))
        .min(clamp_i64_to_i32(i64::from(py) + i64::from(ph)));

    if right <= left || bottom <= top {
        return Region::new(0, 0, 0, 0);
    }

    Region::new(
        left - px,
        top - py,
        (right - left) as u32,
        (bottom - top) as u32,
    )
}

fn paste_region(
    input: &[u8],
    input_region: Region,
    output: &mut [u8],
    output_region: Region,
    px: i32,
    py: i32,
    pixel_bytes: usize,
) {
    if input_region.is_empty() {
        return;
    }

    let row_bytes = input_region.width as usize * pixel_bytes;
    let output_row_bytes = output_region.width as usize * pixel_bytes;
    let out_x = (i64::from(px) + i64::from(input_region.x) - i64::from(output_region.x)) as usize;
    let out_y = (i64::from(py) + i64::from(input_region.y) - i64::from(output_region.y)) as usize;

    for row in 0..input_region.height as usize {
        let src_start = row * row_bytes;
        let dst_start = (out_y + row) * output_row_bytes + out_x * pixel_bytes;
        output[dst_start..dst_start + row_bytes]
            .copy_from_slice(&input[src_start..src_start + row_bytes]);
    }
}

const fn sample_size_bytes(format: BandFormatId) -> usize {
    match format {
        BandFormatId::U8 => 1,
        BandFormatId::U16 | BandFormatId::I16 => 2,
        BandFormatId::U32 | BandFormatId::I32 | BandFormatId::F32 => 4,
        BandFormatId::F64 => 8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::op::DynOperation;

    fn run_insert(op: &Insert, main: &[u8], sub: &[u8], output_region: Region) -> Vec<u8> {
        let inputs: &[&[u8]] = &[main, sub];
        let input_regions = [
            op.required_input_region_slot(&output_region, 0),
            op.required_input_region_slot(&output_region, 1),
        ];
        let mut output = vec![0u8; output_region.pixel_count() * op.bands() as usize];
        let mut state = op.dyn_start();
        op.dyn_process_region_multi(
            state.as_mut(),
            inputs,
            &mut output,
            &input_regions,
            output_region,
        );
        output
    }

    #[test]
    fn insert_overwrites_main_at_offset() {
        let op = Insert::new(3, 2, 2, 1, 1, 1, false, 1, BandFormatId::U8);
        let main = [1u8, 2, 3, 4, 5, 6];
        let sub = [9u8, 8];
        let output = run_insert(&op, &main, &sub, Region::new(0, 0, 3, 2));
        assert_eq!(output, vec![1u8, 2, 3, 4, 9, 8]);
    }

    #[test]
    fn expand_true_grows_canvas_and_zero_fills_background() {
        let op = Insert::new(2, 1, 1, 1, 3, 0, true, 1, BandFormatId::U8);
        let output = run_insert(&op, &[1u8, 2], &[9u8], Region::new(0, 0, 4, 1));
        assert_eq!(output, vec![1u8, 2, 0, 9]);
    }

    #[test]
    fn required_input_region_slot_clips_subimage_overlap() {
        let op = Insert::new(4, 4, 3, 3, 2, 1, false, 1, BandFormatId::U8);
        let output = Region::new(3, 2, 1, 2);
        assert_eq!(
            op.required_input_region_slot(&output, 0),
            Region::new(3, 2, 1, 2)
        );
        assert_eq!(
            op.required_input_region_slot(&output, 1),
            Region::new(1, 1, 1, 2)
        );
    }

    #[test]
    fn insert_at_origin_overlays_top_left_pixels() {
        let op = Insert::new(3, 2, 2, 2, 0, 0, false, 1, BandFormatId::U8);
        let main = [1u8, 2, 3, 4, 5, 6];
        let sub = [9u8, 8, 7, 6];
        let output = run_insert(&op, &main, &sub, Region::new(0, 0, 3, 2));
        assert_eq!(output, vec![9u8, 8, 3, 7, 6, 6]);
    }

    #[test]
    fn insert_out_of_bounds_is_a_no_op_without_expand() {
        let op = Insert::new(3, 2, 2, 2, 4, 3, false, 1, BandFormatId::U8);
        let main = [1u8, 2, 3, 4, 5, 6];
        let sub = [9u8, 8, 7, 6];
        assert_eq!(
            op.required_input_region_slot(&Region::new(0, 0, 3, 2), 1),
            Region::new(0, 0, 0, 0)
        );
        let output = run_insert(&op, &main, &sub, Region::new(0, 0, 3, 2));
        assert_eq!(output, main);
    }

    #[test]
    fn expand_with_negative_offsets_keeps_main_extent() {
        let op = Insert::new(3, 2, 2, 2, -2, -1, true, 1, BandFormatId::U8);
        assert_eq!(op.output_width(), 3);
        assert_eq!(op.output_height(), 2);
    }

    #[test]
    fn insert_dyn_metadata_matches_layout_configuration() {
        let op = Insert::new(3, 2, 1, 1, 1, 0, false, 2, BandFormatId::U16);
        let output = Region::new(0, 0, 3, 2);

        assert_eq!(op.input_format(), BandFormatId::U16);
        assert_eq!(op.output_format(), BandFormatId::U16);
        assert_eq!(op.bands(), 2);
        assert_eq!(op.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(op.input_slot_count(), 2);
        assert_eq!(op.required_input_region(&output), output);
        assert_eq!(DynOperation::output_width(&op, 3), 3);
        assert_eq!(DynOperation::output_height(&op, 2), 2);
        assert_eq!(op.node_spec(64, 16), NodeSpec::identity(64, 16));
        assert_eq!(op.output_width(), 3);
        assert_eq!(op.output_height(), 2);
    }

    #[test]
    fn insert_supports_multibyte_formats() {
        let op = Insert::new(2, 1, 1, 1, 1, 0, false, 1, BandFormatId::F32);
        let output_region = Region::new(0, 0, 2, 1);
        let inputs: &[&[u8]] = &[
            bytemuck::cast_slice(&[1.0f32, 2.0]),
            bytemuck::cast_slice(&[9.0f32]),
        ];
        let input_regions = [
            op.required_input_region_slot(&output_region, 0),
            op.required_input_region_slot(&output_region, 1),
        ];
        let mut output = [0.0f32; 2];
        let mut state = op.dyn_start();
        op.dyn_process_region_multi(
            state.as_mut(),
            inputs,
            bytemuck::cast_slice_mut(&mut output),
            &input_regions,
            output_region,
        );

        assert_eq!(output, [1.0f32, 9.0]);
    }

    #[test]
    fn intersect_and_translate_clamps_large_right_and_bottom_edges() {
        let region = intersect_and_translate(
            Region::new(i32::MAX - 2, i32::MAX - 2, 10, 10),
            i32::MAX - 2,
            i32::MAX - 2,
            10,
            10,
        );

        assert_eq!(region, Region::new(0, 0, 2, 2));
    }
}

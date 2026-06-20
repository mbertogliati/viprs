use std::any::Any;
use viprs_core::{
    format::BandFormatId,
    image::{DemandHint, Region, clamp_i64_to_i32},
    op::{DynOperation, NodeSpec},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Enumerates the available join direction values.
pub enum JoinDirection {
    /// Uses the `Horizontal` variant of `JoinDirection`.
    Horizontal,
    /// Uses the `Vertical` variant of `JoinDirection`.
    Vertical,
}

/// Concatenate two images along one axis, zero-filling uncovered background.
pub struct Join {
    direction: JoinDirection,
    a_width: u32,
    a_height: u32,
    b_width: u32,
    b_height: u32,
    bands: u32,
    format: BandFormatId,
}

impl Join {
    #[must_use]
    /// Creates a new `Join`.
    pub const fn new(
        direction: JoinDirection,
        a_width: u32,
        a_height: u32,
        b_width: u32,
        b_height: u32,
        bands: u32,
        format: BandFormatId,
    ) -> Self {
        Self {
            direction,
            a_width,
            a_height,
            b_width,
            b_height,
            bands,
            format,
        }
    }

    #[must_use]
    /// Returns or performs output width.
    pub fn output_width(&self) -> u32 {
        match self.direction {
            JoinDirection::Horizontal => clamp_join_axis_sum(self.a_width, self.b_width),
            JoinDirection::Vertical => self.a_width.max(self.b_width),
        }
    }

    #[must_use]
    /// Returns or performs output height.
    pub fn output_height(&self) -> u32 {
        match self.direction {
            JoinDirection::Horizontal => self.a_height.max(self.b_height),
            JoinDirection::Vertical => clamp_join_axis_sum(self.a_height, self.b_height),
        }
    }

    fn placement(&self, slot: usize) -> (i32, i32, u32, u32) {
        match (self.direction, slot) {
            (_, 0) => (0, 0, self.a_width, self.a_height),
            (JoinDirection::Horizontal, 1) => (
                clamp_i64_to_i32(i64::from(self.a_width)),
                0,
                self.b_width,
                self.b_height,
            ),
            (JoinDirection::Vertical, 1) => (
                0,
                clamp_i64_to_i32(i64::from(self.a_height)),
                self.b_width,
                self.b_height,
            ),
            _ => (0, 0, 0, 0),
        }
    }
}

fn clamp_join_axis_sum(lhs: u32, rhs: u32) -> u32 {
    clamp_i64_to_i32(i64::from(lhs) + i64::from(rhs)) as u32
}

impl DynOperation for Join {
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
            "Join: dyn_process_region called on a 2-input node — use dyn_process_region_multi"
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
        debug_assert_eq!(inputs.len(), 2, "Join: expected exactly 2 input slices");
        debug_assert_eq!(
            input_regions.len(),
            2,
            "Join: expected exactly 2 input regions"
        );

        output.fill(0);

        for slot in 0..2 {
            let (px, py, _, _) = self.placement(slot);
            paste_region(
                inputs[slot],
                input_regions[slot],
                output,
                output_region,
                px,
                py,
                self.bands as usize * sample_size_bytes(self.format),
            );
        }
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
    use viprs_core::op::DynOperation;

    fn run_join(op: &Join, a: &[u8], b: &[u8], output_region: Region) -> Vec<u8> {
        let inputs: &[&[u8]] = &[a, b];
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
    fn horizontal_join_concatenates_rows() {
        let op = Join::new(JoinDirection::Horizontal, 2, 1, 2, 1, 1, BandFormatId::U8);
        let output = run_join(&op, &[1u8, 2], &[3u8, 4], Region::new(0, 0, 4, 1));
        assert_eq!(output, vec![1u8, 2, 3, 4]);
    }

    #[test]
    fn vertical_join_zero_fills_uncovered_area() {
        let op = Join::new(JoinDirection::Vertical, 2, 1, 1, 1, 1, BandFormatId::U8);
        let output = run_join(&op, &[1u8, 2], &[3u8], Region::new(0, 0, 2, 2));
        assert_eq!(output, vec![1u8, 2, 3, 0]);
    }

    #[test]
    fn required_input_region_slot_clips_to_each_input() {
        let op = Join::new(JoinDirection::Horizontal, 3, 2, 2, 2, 1, BandFormatId::U8);
        let output = Region::new(2, 0, 2, 2);
        assert_eq!(
            op.required_input_region_slot(&output, 0),
            Region::new(2, 0, 1, 2)
        );
        assert_eq!(
            op.required_input_region_slot(&output, 1),
            Region::new(0, 0, 1, 2)
        );
    }

    #[test]
    fn vertical_join_concatenates_columns() {
        let op = Join::new(JoinDirection::Vertical, 1, 2, 1, 2, 1, BandFormatId::U8);
        let output = run_join(&op, &[1u8, 2], &[3u8, 4], Region::new(0, 0, 1, 4));
        assert_eq!(output, vec![1u8, 2, 3, 4]);
    }

    #[test]
    fn horizontal_join_handles_one_pixel_strips() {
        let op = Join::new(JoinDirection::Horizontal, 1, 2, 1, 2, 1, BandFormatId::U8);
        let output = run_join(&op, &[1u8, 2], &[3u8, 4], Region::new(0, 0, 2, 2));
        assert_eq!(output, vec![1u8, 3, 2, 4]);
    }

    #[test]
    fn join_output_dimensions_follow_direction() {
        let horizontal = Join::new(JoinDirection::Horizontal, 3, 5, 2, 4, 1, BandFormatId::U8);
        assert_eq!(horizontal.output_width(), 5);
        assert_eq!(horizontal.output_height(), 5);

        let vertical = Join::new(JoinDirection::Vertical, 3, 5, 2, 4, 1, BandFormatId::U8);
        assert_eq!(vertical.output_width(), 3);
        assert_eq!(vertical.output_height(), 9);
    }

    #[test]
    fn horizontal_join_output_width_clamps_large_sum_to_i32_max() {
        let op = Join::new(
            JoinDirection::Horizontal,
            u32::MAX,
            1,
            1,
            1,
            1,
            BandFormatId::U8,
        );

        assert_eq!(op.output_width(), i32::MAX as u32);
        assert_eq!(
            op.required_input_region_slot(&Region::new(-1, 0, 1, 1), 1),
            Region::new(0, 0, 0, 0)
        );
    }

    #[test]
    fn vertical_join_output_height_clamps_large_sum_to_i32_max() {
        let op = Join::new(
            JoinDirection::Vertical,
            1,
            u32::MAX,
            1,
            1,
            1,
            BandFormatId::U8,
        );

        assert_eq!(op.output_height(), i32::MAX as u32);
        assert_eq!(
            op.required_input_region_slot(&Region::new(0, -1, 1, 1), 1),
            Region::new(0, 0, 0, 0)
        );
    }

    #[test]
    fn join_dyn_metadata_matches_layout_configuration() {
        let op = Join::new(JoinDirection::Horizontal, 3, 2, 1, 2, 2, BandFormatId::U32);
        let output = Region::new(0, 0, 4, 2);

        assert_eq!(op.input_format(), BandFormatId::U32);
        assert_eq!(op.output_format(), BandFormatId::U32);
        assert_eq!(op.bands(), 2);
        assert_eq!(op.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(op.input_slot_count(), 2);
        assert_eq!(op.required_input_region(&output), output);
        assert_eq!(DynOperation::output_width(&op, 3), 4);
        assert_eq!(DynOperation::output_height(&op, 2), 2);
        assert_eq!(op.node_spec(64, 16), NodeSpec::identity(64, 16));
    }

    #[test]
    fn join_out_of_bounds_region_stays_zero_filled() {
        let op = Join::new(JoinDirection::Horizontal, 2, 1, 2, 1, 1, BandFormatId::U8);
        let output = Region::new(10, 0, 1, 1);
        assert_eq!(
            op.required_input_region_slot(&output, 0),
            Region::new(0, 0, 0, 0)
        );
        assert_eq!(
            op.required_input_region_slot(&output, 1),
            Region::new(0, 0, 0, 0)
        );
        assert_eq!(run_join(&op, &[1u8, 2], &[3u8, 4], output), vec![0u8]);
    }

    #[test]
    fn join_supports_multibyte_formats() {
        let op = Join::new(JoinDirection::Horizontal, 1, 1, 1, 1, 1, BandFormatId::F64);
        let output_region = Region::new(0, 0, 2, 1);
        let inputs: &[&[u8]] = &[
            bytemuck::cast_slice(&[1.0f64]),
            bytemuck::cast_slice(&[2.5f64]),
        ];
        let input_regions = [
            op.required_input_region_slot(&output_region, 0),
            op.required_input_region_slot(&output_region, 1),
        ];
        let mut output = [0.0f64; 2];
        let mut state = op.dyn_start();
        op.dyn_process_region_multi(
            state.as_mut(),
            inputs,
            bytemuck::cast_slice_mut(&mut output),
            &input_regions,
            output_region,
        );

        assert_eq!(output, [1.0f64, 2.5]);
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

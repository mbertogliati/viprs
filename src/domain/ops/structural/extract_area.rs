use crate::{
    domain::op::ViewOp,
    domain::{
        format::BandFormat,
        image::{DemandHint, Region, clamp_i64_to_i32},
    },
};

/// Crop: extract a rectangular sub-region of the image.
///
/// The output image is `width × height` pixels starting at `(x, y)` in the source.
/// `required_input_region` shifts every output tile by `(self.x, self.y)`, so the
/// source reads from the correct position.
///
/// `ExtractArea` is a `ViewOp` — it has no `process_region`. The scheduler passes
/// the upstream buffer directly to the downstream node without copying. The entire
/// coordinate transform is encoded in `required_input_region`.
///
/// Output pipeline dimensions: `self.width × self.height`.
pub struct ExtractArea<F: BandFormat> {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    _format: std::marker::PhantomData<F>,
}

impl<F: BandFormat> ExtractArea<F> {
    /// `x`, `y`: top-left corner in source coordinates.
    /// `width`, `height`: dimensions of the extracted region.
    #[must_use]
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            _format: std::marker::PhantomData,
        }
    }
}

impl<F: BandFormat> ViewOp for ExtractArea<F> {
    type Format = F;

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    /// Shift the output tile by (self.x, self.y) to get the source region.
    fn required_input_region(&self, output: &Region) -> Region {
        Region::new(
            clamp_i64_to_i32(i64::from(output.x) + i64::from(self.x)),
            clamp_i64_to_i32(i64::from(output.y) + i64::from(self.y)),
            output.width,
            output.height,
        )
    }

    fn valid_output_region(&self, output: &Region) -> Region {
        output.clip_to(self.width, self.height)
    }

    fn output_width(&self, _input_width: u32) -> u32 {
        self.width
    }

    fn output_height(&self, _input_height: u32) -> u32 {
        self.height
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{format::U8, image::Region};
    use proptest::prelude::*;

    #[test]
    fn required_input_region_shifts_by_offset() {
        let op = ExtractArea::<U8>::new(10, 20, 5, 5);
        let output = Region::new(0, 0, 5, 5);
        let input = op.required_input_region(&output);
        assert_eq!(input.x, 10);
        assert_eq!(input.y, 20);
        assert_eq!(input.width, 5);
        assert_eq!(input.height, 5);
    }

    #[test]
    fn required_input_region_preserves_tile_offset() {
        // When the scheduler requests output tile (8, 16, 4, 4), the input region
        // must be (8 + offset_x, 16 + offset_y, 4, 4).
        let op = ExtractArea::<U8>::new(100, 200, 50, 50);
        let output = Region::new(8, 16, 4, 4);
        let input = op.required_input_region(&output);
        assert_eq!(input.x, 108);
        assert_eq!(input.y, 216);
        assert_eq!(input.width, 4);
        assert_eq!(input.height, 4);
    }

    #[test]
    fn required_input_region_clamps_offsets_above_i32_max() {
        let op = ExtractArea::<U8>::new(u32::MAX, u32::MAX, 1, 1);
        let output = Region::new(0, 0, 1, 1);
        let input = op.required_input_region(&output);
        assert_eq!(input.x, i32::MAX);
        assert_eq!(input.y, i32::MAX);
        assert_eq!(input.width, 1);
        assert_eq!(input.height, 1);
    }

    #[test]
    fn output_dimensions_override_input() {
        let op = ExtractArea::<U8>::new(0, 0, 30, 20);
        assert_eq!(op.output_width(100), 30);
        assert_eq!(op.output_height(100), 20);
    }

    proptest! {
        /// required_input_region shifts by exactly (x, y) for any output region.
        #[test]
        fn required_input_region_shift_is_exact(
            ox in 0i32..100,
            oy in 0i32..100,
            ow in 1u32..=32,
            oh in 1u32..=32,
            crop_x in 0u32..=50,
            crop_y in 0u32..=50,
        ) {
            let op = ExtractArea::<U8>::new(crop_x, crop_y, ow, oh);
            let output = Region::new(ox, oy, ow, oh);
            let input = op.required_input_region(&output);
            prop_assert_eq!(input.x, ox + crop_x as i32);
            prop_assert_eq!(input.y, oy + crop_y as i32);
            prop_assert_eq!(input.width, ow);
            prop_assert_eq!(input.height, oh);
        }
    }
}

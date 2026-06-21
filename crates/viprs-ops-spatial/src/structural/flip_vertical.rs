use viprs_core::{
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut, clamp_i64_to_i32},
    op::Op,
};

/// Vertical flip (mirror top-bottom).
///
/// For an image of height `H`, output pixel `(ox, oy)` comes from input pixel
/// `(ox, H - 1 - oy)`. Implemented via:
/// - `required_input_region`: reflects the tile vertically within the image.
/// - `process_region`: reverses row order in the tile.
pub struct FlipVertical<F: BandFormat> {
    image_height: u32,
    _format: std::marker::PhantomData<F>,
}

impl<F: BandFormat> FlipVertical<F> {
    /// `image_height`: the full height of the image being flipped.
    #[must_use]
    pub const fn new(image_height: u32) -> Self {
        Self {
            image_height,
            _format: std::marker::PhantomData,
        }
    }
}

impl<F: BandFormat> Op for FlipVertical<F>
where
    F::Sample: Copy,
{
    type Input = F;
    type Output = F;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    /// Reflect output tile `(ox, oy, ow, oh)` to `(ox, H - oy - oh, ow, oh)`.
    fn required_input_region(&self, output: &Region) -> Region {
        let reflected_y = clamp_i64_to_i32(
            i64::from(self.image_height) - i64::from(output.y) - i64::from(output.height),
        );
        Region::new(output.x, reflected_y, output.width, output.height)
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        let bands = input.bands as usize;
        let width = input.region.width as usize;
        let height = input.region.height as usize;
        let row_samples = width * bands;

        // Reverse the row order within the tile.
        for row in 0..height {
            let src_row = height - 1 - row;
            let src_start = src_row * row_samples;
            let dst_start = row * row_samples;
            output.data[dst_start..dst_start + row_samples]
                .copy_from_slice(&input.data[src_start..src_start + row_samples]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{format::U8, image::Region};

    #[test]
    fn required_input_region_reflects_y() {
        // Image height 100. Output tile at y=10, height=20 → input at y=70.
        let op = FlipVertical::<U8>::new(100);
        let output = Region::new(0, 10, 1, 20);
        let input = op.required_input_region(&output);
        assert_eq!(input.y, 70); // 100 - 10 - 20
        assert_eq!(input.x, 0);
        assert_eq!(input.width, 1);
        assert_eq!(input.height, 20);
    }

    #[test]
    fn required_input_region_x_unchanged() {
        let op = FlipVertical::<U8>::new(64);
        let output = Region::new(5, 0, 8, 4);
        let input = op.required_input_region(&output);
        assert_eq!(input.x, 5);
        assert_eq!(input.width, 8);
    }

    #[test]
    fn required_input_region_clamps_large_image_height() {
        let op = FlipVertical::<U8>::new(u32::MAX);
        let output = Region::new(0, 0, 1, 1);
        let input = op.required_input_region(&output);
        assert_eq!(input.x, 0);
        assert_eq!(input.y, i32::MAX);
        assert_eq!(input.width, 1);
        assert_eq!(input.height, 1);
    }

    #[test]
    fn process_region_reverses_rows() {
        let op = FlipVertical::<U8>::new(3);
        let r = Region::new(0, 0, 1, 3);
        // Col 0: rows [10, 20, 30]
        let input_data = vec![10u8, 20, 30];
        let mut output_data = vec![0u8; 3];
        let input = Tile::<U8>::new(r, 1, &input_data);
        let mut output = TileMut::<U8>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, vec![30u8, 20, 10]);
    }

    #[test]
    fn process_region_reverses_multi_column_rows() {
        let op = FlipVertical::<U8>::new(2);
        let r = Region::new(0, 0, 3, 2);
        // Row 0: [1,2,3], Row 1: [4,5,6]
        let input_data = vec![1u8, 2, 3, 4, 5, 6];
        let mut output_data = vec![0u8; 6];
        let input = Tile::<U8>::new(r, 1, &input_data);
        let mut output = TileMut::<U8>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        // Row 0 → Row 1, Row 1 → Row 0
        assert_eq!(output_data, vec![4u8, 5, 6, 1, 2, 3]);
    }

    proptest! {
        /// Flipping vertically twice must be the identity.
        #[test]
        fn flip_v_twice_is_identity(
            rows in 1usize..=8usize,
            cols in 1usize..=8usize,
        ) {
            let pixels: Vec<u8> = (0..(rows * cols) as u8).collect();
            let op = FlipVertical::<U8>::new(rows as u32);
            let r = Region::new(0, 0, cols as u32, rows as u32);

            let mut mid = vec![0u8; rows * cols];
            {
                let input = Tile::<U8>::new(r, 1, &pixels);
                let mut output = TileMut::<U8>::new(r, 1, &mut mid);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
            }

            let mut result = vec![0u8; rows * cols];
            {
                let input = Tile::<U8>::new(r, 1, &mid);
                let mut output = TileMut::<U8>::new(r, 1, &mut result);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
            }

            prop_assert_eq!(result, pixels);
        }
    }
}

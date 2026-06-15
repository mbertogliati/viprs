use crate::{
    domain::op::Op,
    domain::{
        format::BandFormat,
        image::{DemandHint, Region, Tile, TileMut, clamp_i64_to_i32},
    },
};

/// Horizontal flip (mirror left-right).
///
/// For an image of width `W`, output pixel `(ox, oy)` comes from input pixel
/// `(W - 1 - ox, oy)`. Implemented via:
/// - `required_input_region`: reflects the tile horizontally within the image.
/// - `process_region`: reverses sample order within each row.
pub struct FlipHorizontal<F: BandFormat> {
    image_width: u32,
    _format: std::marker::PhantomData<F>,
}

impl<F: BandFormat> FlipHorizontal<F> {
    /// `image_width`: the full width of the image being flipped.
    #[must_use]
    pub const fn new(image_width: u32) -> Self {
        Self {
            image_width,
            _format: std::marker::PhantomData,
        }
    }
}

impl<F: BandFormat> Op for FlipHorizontal<F>
where
    F::Sample: Copy,
{
    type Input = F;
    type Output = F;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    /// Reflect output tile `(ox, oy, ow, oh)` to `(W - ox - ow, oy, ow, oh)`.
    fn required_input_region(&self, output: &Region) -> Region {
        let reflected_x = clamp_i64_to_i32(
            i64::from(self.image_width) - i64::from(output.x) - i64::from(output.width),
        );
        Region::new(reflected_x, output.y, output.width, output.height)
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        let bands = input.bands as usize;
        let width = input.region.width as usize;

        // Reverse the pixel order within each row. Samples per row = width * bands.
        for row in 0..input.region.height as usize {
            let row_start = row * width * bands;
            for col in 0..width {
                let src_col = width - 1 - col;
                for band in 0..bands {
                    output.data[row_start + col * bands + band] =
                        input.data[row_start + src_col * bands + band];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{format::U8, image::Region};
    use proptest::prelude::*;

    #[test]
    fn required_input_region_reflects_x() {
        // Image width 100. Output tile at x=10, width=20 → input at x=70.
        let op = FlipHorizontal::<U8>::new(100);
        let output = Region::new(10, 0, 20, 1);
        let input = op.required_input_region(&output);
        assert_eq!(input.x, 70); // 100 - 10 - 20
        assert_eq!(input.y, 0);
        assert_eq!(input.width, 20);
        assert_eq!(input.height, 1);
    }

    #[test]
    fn required_input_region_y_unchanged() {
        let op = FlipHorizontal::<U8>::new(64);
        let output = Region::new(0, 5, 8, 4);
        let input = op.required_input_region(&output);
        assert_eq!(input.y, 5);
        assert_eq!(input.height, 4);
    }

    #[test]
    fn required_input_region_clamps_large_image_width() {
        let op = FlipHorizontal::<U8>::new(u32::MAX);
        let output = Region::new(0, 0, 1, 1);
        let input = op.required_input_region(&output);
        assert_eq!(input.x, i32::MAX);
        assert_eq!(input.y, 0);
        assert_eq!(input.width, 1);
        assert_eq!(input.height, 1);
    }

    #[test]
    fn process_region_reverses_single_row() {
        let op = FlipHorizontal::<U8>::new(4);
        let r = Region::new(0, 0, 4, 1);
        let input_data = vec![1u8, 2, 3, 4];
        let mut output_data = vec![0u8; 4];
        let input = Tile::<U8>::new(r, 1, &input_data);
        let mut output = TileMut::<U8>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, vec![4u8, 3, 2, 1]);
    }

    #[test]
    fn process_region_reverses_two_rows() {
        let op = FlipHorizontal::<U8>::new(3);
        let r = Region::new(0, 0, 3, 2);
        // Row 0: [1,2,3], Row 1: [4,5,6]
        let input_data = vec![1u8, 2, 3, 4, 5, 6];
        let mut output_data = vec![0u8; 6];
        let input = Tile::<U8>::new(r, 1, &input_data);
        let mut output = TileMut::<U8>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        // Row 0 reversed: [3,2,1], Row 1 reversed: [6,5,4]
        assert_eq!(output_data, vec![3u8, 2, 1, 6, 5, 4]);
    }

    proptest! {
        /// Flipping horizontally twice must be the identity.
        #[test]
        fn flip_h_twice_is_identity(
            pixels in proptest::collection::vec(0u8..=255u8, 1..=64)
        ) {
            let len = pixels.len();
            let op = FlipHorizontal::<U8>::new(len as u32);
            let r = Region::new(0, 0, len as u32, 1);

            let mut mid = vec![0u8; len];
            {
                let input = Tile::<U8>::new(r, 1, &pixels);
                let mut output = TileMut::<U8>::new(r, 1, &mut mid);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
            }

            let mut result = vec![0u8; len];
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

use std::marker::PhantomData;

use viprs_core::{
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    op::{NodeSpec, Op},
};

/// Wrap image origin so input pixel `(0, 0)` appears at `(x, y)`.
pub struct Wrap<F: BandFormat> {
    image_width: u32,
    image_height: u32,
    x: i32,
    y: i32,
    _format: PhantomData<F>,
}

impl<F: BandFormat> Wrap<F> {
    #[must_use]
    /// Creates a new `Wrap`.
    pub fn new(image_width: u32, image_height: u32, x: i32, y: i32) -> Self {
        debug_assert!(image_width > 0, "Wrap: image_width must be >= 1");
        debug_assert!(image_height > 0, "Wrap: image_height must be >= 1");
        Self {
            image_width,
            image_height,
            x,
            y,
            _format: PhantomData,
        }
    }

    #[must_use]
    /// Returns or performs centered.
    pub fn centered(image_width: u32, image_height: u32) -> Self {
        Self::new(
            image_width,
            image_height,
            (image_width / 2) as i32,
            (image_height / 2) as i32,
        )
    }

    #[inline]
    fn offset(displacement: i32, size: u32) -> i64 {
        let displacement = i64::from(displacement);
        let size = i64::from(size);
        if displacement < 0 {
            -displacement % size
        } else {
            size - displacement % size
        }
    }
}

impl<F: BandFormat> Op for Wrap<F>
where
    F::Sample: Copy,
{
    type Input = F;
    type Output = F;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::SmallTile
    }

    fn required_input_region(&self, _output: &Region) -> Region {
        Region::new(0, 0, self.image_width, self.image_height)
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec {
            input_tile_w: self.image_width,
            input_tile_h: self.image_height,
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        let bands = output.bands as usize;
        let input_width = input.region.width as usize;
        let output_width = output.region.width as usize;
        let offset_x = Self::offset(self.x, self.image_width);
        let offset_y = Self::offset(self.y, self.image_height);
        let width = i64::from(self.image_width);
        let height = i64::from(self.image_height);

        for row in 0..output.region.height as usize {
            let src_y =
                (offset_y + i64::from(output.region.y) + row as i64).rem_euclid(height) as usize;
            for col in 0..output.region.width as usize {
                let src_x =
                    (offset_x + i64::from(output.region.x) + col as i64).rem_euclid(width) as usize;
                let src = (src_y * input_width + src_x) * bands;
                let dst = (row * output_width + col) * bands;
                output.data[dst..dst + bands].copy_from_slice(&input.data[src..src + bands]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::format::U8;

    fn run_wrap(op: &Wrap<U8>, pixels: &[u8], output_region: Region) -> Vec<u8> {
        let input_region = op.required_input_region(&output_region);
        let mut output = vec![0u8; output_region.pixel_count()];
        let input = Tile::<U8>::new(input_region, 1, pixels);
        let mut output_tile = TileMut::<U8>::new(output_region, 1, &mut output);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output_tile);
        output
    }

    fn expected_wrap(pixels: &[u8], width: usize, height: usize, x: i32, y: i32) -> Vec<u8> {
        let offset_x = Wrap::<U8>::offset(x, width as u32);
        let offset_y = Wrap::<U8>::offset(y, height as u32);
        let mut expected = vec![0u8; width * height];
        for row in 0..height {
            let src_y = (offset_y + row as i64).rem_euclid(height as i64) as usize;
            for col in 0..width {
                let src_x = (offset_x + col as i64).rem_euclid(width as i64) as usize;
                expected[row * width + col] = pixels[src_y * width + src_x];
            }
        }
        expected
    }

    #[test]
    fn positive_displacement_moves_origin_to_that_position() {
        let pixels = vec![1u8, 2, 3, 4, 5, 6];
        let op = Wrap::<U8>::new(3, 2, 1, 1);
        let output = run_wrap(&op, &pixels, Region::new(0, 0, 3, 2));
        assert_eq!(output[1 + 3], 1);
        assert_eq!(output, expected_wrap(&pixels, 3, 2, 1, 1));
    }

    #[test]
    fn negative_displacement_uses_clock_arithmetic() {
        let pixels = vec![1u8, 2, 3, 4];
        let op = Wrap::<U8>::new(2, 2, -1, 0);
        let output = run_wrap(&op, &pixels, Region::new(0, 0, 2, 2));
        assert_eq!(output[1], 1);
    }

    #[test]
    fn centered_wrap_moves_origin_to_image_center_and_reports_full_source_tile() {
        let op = Wrap::<U8>::centered(4, 2);
        assert_eq!(
            op.required_input_region(&Region::new(5, 6, 1, 1)),
            Region::new(0, 0, 4, 2)
        );
        let spec = op.node_spec(3, 5);
        assert_eq!(spec.input_tile_w, 4);
        assert_eq!(spec.input_tile_h, 2);
        assert_eq!(spec.output_tile_w, 3);
        assert_eq!(spec.output_tile_h, 5);
    }

    #[test]
    fn offset_normalizes_positive_and_negative_displacements() {
        assert_eq!(Wrap::<U8>::offset(0, 5), 5);
        assert_eq!(Wrap::<U8>::offset(6, 5), 4);
        assert_eq!(Wrap::<U8>::offset(-6, 5), 1);
    }

    proptest! {
        #[test]
        fn zero_displacement_is_identity(rows in 1usize..=8, cols in 1usize..=8) {
            let pixels = (0..rows * cols).map(|idx| (idx % 251) as u8).collect::<Vec<_>>();
            let op = Wrap::<U8>::new(cols as u32, rows as u32, 0, 0);
            let output = run_wrap(&op, &pixels, Region::new(0, 0, cols as u32, rows as u32));
            prop_assert_eq!(output, pixels);
        }

        #[test]
        fn wrap_matches_libvips_replicate_extract_formula(
            rows in 1usize..=6,
            cols in 1usize..=6,
            x in -12i32..=12,
            y in -12i32..=12,
        ) {
            let pixels = (0..rows * cols).map(|idx| (idx % 251) as u8).collect::<Vec<_>>();
            let op = Wrap::<U8>::new(cols as u32, rows as u32, x, y);
            let output = run_wrap(&op, &pixels, Region::new(0, 0, cols as u32, rows as u32));
            let expected = expected_wrap(&pixels, cols, rows, x, y);
            prop_assert_eq!(output, expected);
        }
    }
}

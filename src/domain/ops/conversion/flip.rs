use std::marker::PhantomData;

use crate::{
    domain::op::{NodeSpec, Op},
    domain::{
        format::BandFormat,
        image::{DemandHint, Region, Tile, TileMut},
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Enumerates the available flip direction values.
pub enum FlipDirection {
    /// Uses the `Horizontal` variant of `FlipDirection`.
    Horizontal,
    /// Uses the `Vertical` variant of `FlipDirection`.
    Vertical,
}

/// Flip an image horizontally or vertically.
pub struct Flip<F: BandFormat> {
    image_width: u32,
    image_height: u32,
    direction: FlipDirection,
    _format: PhantomData<F>,
}

impl<F: BandFormat> Flip<F> {
    #[must_use]
    /// Creates a new `Flip`.
    pub const fn new(image_width: u32, image_height: u32, direction: FlipDirection) -> Self {
        Self {
            image_width,
            image_height,
            direction,
            _format: PhantomData,
        }
    }

    #[must_use]
    /// Returns or performs horizontal.
    pub const fn horizontal(image_width: u32) -> Self {
        Self::new(image_width, 0, FlipDirection::Horizontal)
    }

    #[must_use]
    /// Returns or performs vertical.
    pub const fn vertical(image_height: u32) -> Self {
        Self::new(0, image_height, FlipDirection::Vertical)
    }
}

impl<F: BandFormat> Op for Flip<F>
where
    F::Sample: Copy,
{
    type Input = F;
    type Output = F;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        match self.direction {
            FlipDirection::Horizontal => {
                crate::domain::ops::structural::flip_horizontal::FlipHorizontal::<F>::new(
                    self.image_width,
                )
                .required_input_region(output)
            }
            FlipDirection::Vertical => {
                crate::domain::ops::structural::flip_vertical::FlipVertical::<F>::new(
                    self.image_height,
                )
                .required_input_region(output)
            }
        }
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec::identity(tile_w, tile_h)
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        match self.direction {
            FlipDirection::Horizontal => {
                crate::domain::ops::structural::flip_horizontal::FlipHorizontal::<F>::new(
                    self.image_width,
                )
                .process_region(state, input, output);
            }
            FlipDirection::Vertical => {
                crate::domain::ops::structural::flip_vertical::FlipVertical::<F>::new(
                    self.image_height,
                )
                .process_region(state, input, output);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::format::U8;
    use proptest::prelude::*;

    fn run_flip(op: &Flip<U8>, pixels: &[u8], region: Region) -> Vec<u8> {
        let mut out = vec![0u8; pixels.len()];
        let input = Tile::<U8>::new(region, 1, pixels);
        let mut output = TileMut::<U8>::new(region, 1, &mut out);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        out
    }

    #[test]
    fn horizontal_required_input_region_reflects_x() {
        let op = Flip::<U8>::horizontal(100);
        assert_eq!(
            op.required_input_region(&Region::new(10, 3, 20, 4)),
            Region::new(70, 3, 20, 4)
        );
    }

    #[test]
    fn vertical_required_input_region_reflects_y() {
        let op = Flip::<U8>::vertical(100);
        assert_eq!(
            op.required_input_region(&Region::new(3, 10, 4, 20)),
            Region::new(3, 70, 4, 20)
        );
    }

    #[test]
    fn one_pixel_boundary_is_identity_for_both_directions() {
        let region = Region::new(0, 0, 1, 1);
        let pixels = [42u8];
        assert_eq!(
            run_flip(&Flip::<U8>::horizontal(1), &pixels, region),
            pixels
        );
        assert_eq!(run_flip(&Flip::<U8>::vertical(1), &pixels, region), pixels);
    }

    proptest! {
        #[test]
        fn horizontal_flip_twice_is_identity(
            pixels in proptest::collection::vec(0u8..=255, 1..=64),
        ) {
            let region = Region::new(0, 0, pixels.len() as u32, 1);
            let op = Flip::<U8>::horizontal(pixels.len() as u32);
            let once = run_flip(&op, &pixels, region);
            let twice = run_flip(&op, &once, region);
            prop_assert_eq!(twice, pixels);
        }

        #[test]
        fn vertical_flip_twice_is_identity(rows in 1usize..=8, cols in 1usize..=8) {
            let pixels = (0..rows * cols).map(|idx| (idx % 251) as u8).collect::<Vec<_>>();
            let region = Region::new(0, 0, cols as u32, rows as u32);
            let op = Flip::<U8>::vertical(rows as u32);
            let once = run_flip(&op, &pixels, region);
            let twice = run_flip(&op, &once, region);
            prop_assert_eq!(twice, pixels);
        }
    }
}

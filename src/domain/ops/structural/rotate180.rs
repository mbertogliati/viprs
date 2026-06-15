use crate::{
    domain::op::Op,
    domain::{
        format::BandFormat,
        image::{DemandHint, Region, Tile, TileMut, clamp_i64_to_i32},
    },
};
use std::marker::PhantomData;

/// Clockwise 180° rotation.
pub struct Rotate180<F: BandFormat> {
    image_width: u32,
    image_height: u32,
    _format: PhantomData<F>,
}

impl<F: BandFormat> Rotate180<F> {
    #[must_use]
    /// Creates a new `Rotate180`.
    pub const fn new(image_width: u32, image_height: u32) -> Self {
        Self {
            image_width,
            image_height,
            _format: PhantomData,
        }
    }
}

impl<F: BandFormat> Op for Rotate180<F>
where
    F::Sample: Copy,
{
    type Input = F;
    type Output = F;
    type State = ();

    fn preferred_tile_geometry(&self) -> DemandHint {
        DemandHint::FatStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        Region::new(
            clamp_i64_to_i32(
                i64::from(self.image_width) - i64::from(output.x) - i64::from(output.width),
            ),
            clamp_i64_to_i32(
                i64::from(self.image_height) - i64::from(output.y) - i64::from(output.height),
            ),
            output.width,
            output.height,
        )
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        let bands = input.bands as usize;
        let row_len = input.region.width as usize * bands;

        if bands == 1 {
            for (src_row, dst_row) in input
                .data
                .chunks_exact(row_len)
                .rev()
                .zip(output.data.chunks_exact_mut(row_len))
            {
                for (src, dst) in src_row.iter().rev().zip(dst_row.iter_mut()) {
                    *dst = *src;
                }
            }
            return;
        }

        for (src_row, dst_row) in input
            .data
            .chunks_exact(row_len)
            .rev()
            .zip(output.data.chunks_exact_mut(row_len))
        {
            for (src_px, dst_px) in src_row
                .rchunks_exact(bands)
                .zip(dst_row.chunks_exact_mut(bands))
            {
                dst_px.copy_from_slice(src_px);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        format::U8,
        image::{Region, Tile, TileMut},
    };
    use proptest::prelude::*;

    #[test]
    fn required_input_region_reflects_both_axes() {
        let op = Rotate180::<U8>::new(8, 6);
        let output = Region::new(1, 2, 3, 2);
        let input = op.required_input_region(&output);
        assert_eq!(input, Region::new(4, 2, 3, 2));
    }

    #[test]
    fn required_input_region_clamps_large_image_dimensions() {
        let op = Rotate180::<U8>::new(u32::MAX, u32::MAX);
        let output = Region::new(0, 0, 1, 1);
        let input = op.required_input_region(&output);
        assert_eq!(input, Region::new(i32::MAX, i32::MAX, 1, 1));
    }

    #[test]
    fn process_region_rotates_2x2() {
        let op = Rotate180::<U8>::new(2, 2);
        let region = Region::new(0, 0, 2, 2);
        let input_data = vec![1u8, 2, 3, 4];
        let mut output_data = vec![0u8; 4];
        let input = Tile::<U8>::new(region, 1, &input_data);
        let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, vec![4u8, 3, 2, 1]);
    }

    #[test]
    fn preferred_tile_geometry_is_fat_strip() {
        let op = Rotate180::<U8>::new(4, 3);
        assert_eq!(op.preferred_tile_geometry(), DemandHint::FatStrip);
    }

    #[test]
    fn process_region_rotates_multiband_pixels_without_reordering_channels() {
        let op = Rotate180::<U8>::new(2, 2);
        let region = Region::new(0, 0, 2, 2);
        let input_data = vec![1u8, 2, 3, 10, 20, 30, 4, 5, 6, 40, 50, 60];
        let mut output_data = vec![0u8; input_data.len()];
        let input = Tile::<U8>::new(region, 3, &input_data);
        let mut output = TileMut::<U8>::new(region, 3, &mut output_data);

        op.process_region(&mut (), &input, &mut output);

        assert_eq!(output_data, vec![40, 50, 60, 4, 5, 6, 10, 20, 30, 1, 2, 3,]);
    }

    proptest! {
        #[test]
        fn rotate_twice_is_identity(rows in 1usize..=6, cols in 1usize..=6) {
            let pixels: Vec<u8> = (0..(rows * cols) as u8).collect();
            let region = Region::new(0, 0, cols as u32, rows as u32);
            let op = Rotate180::<U8>::new(cols as u32, rows as u32);

            let mut once = vec![0u8; pixels.len()];
            {
                let input = Tile::<U8>::new(region, 1, &pixels);
                let mut output = TileMut::<U8>::new(region, 1, &mut once);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
            }

            let mut twice = vec![0u8; pixels.len()];
            {
                let input = Tile::<U8>::new(region, 1, &once);
                let mut output = TileMut::<U8>::new(region, 1, &mut twice);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
            }

            prop_assert_eq!(twice, pixels);
        }
    }
}

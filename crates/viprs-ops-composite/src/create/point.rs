use viprs_core::{
    error::ViprsError,
    format::U8,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

/// Scatter deterministic white points over a black image.
///
/// # Examples
/// ```ignore
/// use viprs_ops_composite::create::point::PointOp;
///
/// let op = PointOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug)]
pub struct PointOp {
    width: u32,
    height: u32,
    count: usize,
    mask: Box<[u8]>,
}

impl PointOp {
    /// Creates a new `PointOp`.
    pub fn new(width: u32, height: u32, count: usize, seed: u32) -> Result<Self, ViprsError> {
        if width == 0 || height == 0 {
            return Err(ViprsError::Scheduler(format!(
                "PointOp width and height must be > 0, got {width}x{height}"
            )));
        }

        let pixel_count = width as usize * height as usize;
        let target = count.min(pixel_count);
        let mut mask = vec![0u8; pixel_count];
        let mut placed = 0usize;
        let mut state = seed;

        while placed < target {
            state = vips_random(state);
            let index = state as usize % pixel_count;
            if mask[index] == 0 {
                mask[index] = u8::MAX;
                placed += 1;
            }
        }

        Ok(Self {
            width,
            height,
            count: target,
            mask: mask.into_boxed_slice(),
        })
    }

    #[must_use]
    /// Returns or performs width.
    pub const fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    /// Returns or performs height.
    pub const fn height(&self) -> u32 {
        self.height
    }

    #[must_use]
    /// Returns or performs count.
    pub const fn count(&self) -> usize {
        self.count
    }
}

#[inline(always)]
fn vips_random_add(mut hash: u32, value: i32) -> u32 {
    for shift in [0, 8, 16, 24] {
        hash = (hash ^ ((value >> shift) as u32 & 0xff)).wrapping_mul(16_777_619);
    }
    hash
}

#[inline(always)]
fn vips_random(seed: u32) -> u32 {
    vips_random_add(2_166_136_261, seed as i32)
}

impl Op for PointOp {
    type Input = U8;
    type Output = U8;
    type State = ();

    const OUTPUT_BANDS: Option<usize> = Some(1);

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), _input: &Tile<U8>, output: &mut TileMut<U8>) {
        debug_assert_eq!(output.bands, 1, "PointOp output must be single-band");
        debug_assert!(output.region.x >= 0 && output.region.y >= 0);
        debug_assert!(output.region.x as u32 + output.region.width <= self.width);
        debug_assert!(output.region.y as u32 + output.region.height <= self.height);

        let region_width = output.region.width as usize;
        let image_width = self.width as usize;

        for row in 0..output.region.height as usize {
            let src_start =
                (output.region.y as usize + row) * image_width + output.region.x as usize;
            let src_end = src_start + region_width;
            let dst_start = row * region_width;
            let dst_end = dst_start + region_width;
            output.data[dst_start..dst_end].copy_from_slice(&self.mask[src_start..src_end]);
        }
    }
}

impl PixelLocalOp for PointOp {}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::image::{Region, Tile, TileMut};

    fn render_region(op: &PointOp, region: Region) -> Vec<u8> {
        let input_data = vec![0u8; region.pixel_count()];
        let mut output_data = vec![0u8; region.pixel_count()];
        let input = Tile::<U8>::new(region, 1, &input_data);
        let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);
        output_data
    }

    fn render_full(op: &PointOp) -> Vec<u8> {
        render_region(op, Region::new(0, 0, op.width(), op.height()))
    }

    #[test]
    fn constructor_rejects_zero_dimensions() {
        assert!(PointOp::new(0, 8, 4, 1).is_err());
        assert!(PointOp::new(8, 0, 4, 1).is_err());
    }

    #[test]
    fn output_is_deterministic_for_same_seed() {
        let first = PointOp::new(16, 16, 8, 9).unwrap();
        let second = PointOp::new(16, 16, 8, 9).unwrap();
        assert_eq!(render_full(&first), render_full(&second));
    }

    #[test]
    fn renders_exact_number_of_unique_points_up_to_capacity() {
        let op = PointOp::new(4, 4, 32, 5).unwrap();
        let output = render_full(&op);
        assert_eq!(op.count(), 16);
        assert_eq!(
            output.iter().filter(|&&sample| sample == u8::MAX).count(),
            16
        );
    }

    #[test]
    fn partial_tiles_match_full_render() {
        let op = PointOp::new(10, 10, 12, 21).unwrap();
        let full = render_full(&op);
        let region = Region::new(3, 4, 4, 3);
        let partial = render_region(&op, region);

        for row in 0..region.height as usize {
            for col in 0..region.width as usize {
                let full_index =
                    (row + region.y as usize) * op.width() as usize + col + region.x as usize;
                let partial_index = row * region.width as usize + col;
                assert_eq!(partial[partial_index], full[full_index]);
            }
        }
    }

    proptest! {
        #[test]
        fn prop_output_stays_binary(
            width in 1u32..=32,
            height in 1u32..=32,
            count in 0usize..=128,
            seed in any::<u32>(),
        ) {
            let op = PointOp::new(width, height, count, seed).unwrap();
            let output = render_full(&op);
            prop_assert_eq!(output.len(), width as usize * height as usize);
            prop_assert!(output.iter().all(|sample| *sample == 0 || *sample == u8::MAX));
            prop_assert_eq!(output.iter().filter(|&&sample| sample == u8::MAX).count(), op.count());
        }
    }
}

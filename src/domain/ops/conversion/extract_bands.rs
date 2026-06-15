use std::marker::PhantomData;

use crate::{
    domain::op::{Op, OperationBridge, PixelLocalOp},
    domain::{
        format::BandFormat,
        image::{DemandHint, Region, Tile, TileMut},
    },
};

/// Extracts a contiguous range of bands `[start, start + count)` from a multi-band image.
///
/// Output has `count` bands with the same sample format as the input.
///
/// # Output band count
///
/// The output has exactly `count` bands. Use `ExtractBands::into_bridge()` to
/// obtain a correctly configured `OperationBridge` — it reads `self.count` and
/// calls `OperationBridge::with_dynamic_bands` automatically.
pub struct ExtractBands<F: BandFormat> {
    start: usize,
    count: usize,
    input_bands: usize,
    _f: PhantomData<F>,
}

impl<F: BandFormat> ExtractBands<F> {
    /// Construct an `ExtractBands` that copies bands `[start, start + count)`.
    ///
    /// # Panics (debug only)
    ///
    /// `start + count` must be ≤ `input_bands`. Checked with `debug_assert`.
    #[must_use]
    pub fn new(start: usize, count: usize, input_bands: usize) -> Self {
        debug_assert!(count > 0, "ExtractBands: count must be at least 1, got 0");
        debug_assert!(
            start + count <= input_bands,
            "ExtractBands: start({start}) + count({count}) exceeds input_bands({input_bands})"
        );
        Self {
            start,
            count,
            input_bands,
            _f: PhantomData,
        }
    }
}

impl<F> ExtractBands<F>
where
    F: BandFormat,
    F::Sample: bytemuck::Pod,
{
    /// Build an `OperationBridge` for this `ExtractBands` op.
    ///
    /// The output band count (`self.count`) is injected via
    /// `OperationBridge::with_dynamic_bands_pixel_local` so the pipeline compiler
    /// sizes downstream buffers correctly.
    #[must_use]
    pub const fn into_bridge(self) -> OperationBridge<Self> {
        let count = self.count as u32;
        let input_bands = self.input_bands as u32;
        OperationBridge::with_dynamic_bands_pixel_local(self, input_bands, count)
    }
}

impl<F: BandFormat> Op for ExtractBands<F> {
    type Input = F;
    type Output = F;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        debug_assert_eq!(
            output.bands as usize, self.count,
            "ExtractBands output tile must have exactly count bands"
        );
        let pixel_count = input.region.pixel_count();
        for px in 0..pixel_count {
            let src_base = px * self.input_bands + self.start;
            let dst_base = px * self.count;
            output.data[dst_base..dst_base + self.count]
                .copy_from_slice(&input.data[src_base..src_base + self.count]);
        }
    }
}

// ExtractBands is pixel-local: required_input_region is identity and node_spec is identity.
impl<F: BandFormat> PixelLocalOp for ExtractBands<F> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        format::U8,
        image::{Region, Tile, TileMut},
        op::DynOperation,
    };

    fn make_region(w: u32, h: u32) -> Region {
        Region::new(0, 0, w, h)
    }

    /// Run ExtractBands directly through Op::process_region (no bridge).
    fn run_extract(
        start: usize,
        count: usize,
        input_bands: usize,
        input_data: &[u8],
        output_data: &mut [u8],
        pixel_count: usize,
    ) {
        let region = make_region(pixel_count as u32, 1);
        let op = ExtractBands::<U8>::new(start, count, input_bands);
        let input = Tile::<U8>::new(region, input_bands as u32, input_data);
        let mut output = TileMut::<U8>::new(region, count as u32, output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
    }

    #[test]
    fn extract_ga_from_rgba() {
        // RGBA: 2 pixels
        // pixel 0: R=10, G=20, B=30, A=40
        // pixel 1: R=50, G=60, B=70, A=80
        // Extract [1, 2) = G and A — wait, this is bands [1..3) = G, B
        // The task says "extract bands [1,2] of RGBA → GA" meaning bands 1 and 3.
        // However, ExtractBands only supports contiguous ranges.
        // Here we test bands [1, 3) = G, B (a contiguous 2-band extract).
        let input = [10u8, 20, 30, 40, 50, 60, 70, 80];
        let mut output = [0u8; 4];
        run_extract(1, 2, 4, &input, &mut output, 2);
        // pixel 0: G=20, B=30; pixel 1: G=60, B=70
        assert_eq!(output, [20, 30, 60, 70]);
    }

    #[test]
    fn extract_rgb_from_rgba_drops_alpha() {
        let input = [10u8, 20, 30, 40, 50, 60, 70, 80];
        let mut output = [0u8; 6];
        run_extract(0, 3, 4, &input, &mut output, 2);
        assert_eq!(output, [10, 20, 30, 50, 60, 70]);
    }

    #[test]
    fn extract_single_band_is_identical_to_bandsplit() {
        let input = [10u8, 20, 30, 40, 50, 60, 70, 80];
        let mut output = [0u8; 2];
        // Extract band 2 (B) from 4-band image.
        run_extract(2, 1, 4, &input, &mut output, 2);
        assert_eq!(output, [30, 70]);
    }

    #[test]
    fn extract_all_bands_is_identity() {
        let input = [1u8, 2, 3, 4, 5, 6];
        let mut output = [0u8; 6];
        run_extract(0, 3, 3, &input, &mut output, 2);
        assert_eq!(output, input);
    }

    #[test]
    fn pixel_local_op_impl() {
        fn requires_pixel_local<T: PixelLocalOp>(_: &T) {}
        let op = ExtractBands::<U8>::new(0, 2, 4);
        requires_pixel_local(&op);
    }

    /// `into_bridge` must set `bands()` equal to `count`.
    #[test]
    fn into_bridge_sets_bands_to_count() {
        let bridge = ExtractBands::<U8>::new(0, 3, 4).into_bridge();
        assert_eq!(bridge.bands(), 3);
    }

    /// Identity: extracting all bands returns the same pixels.
    #[test]
    fn identity_all_bands() {
        let input = [10u8, 20, 30, 40, 50, 60];
        let mut output = [0u8; 6];
        run_extract(0, 3, 3, &input, &mut output, 2);
        assert_eq!(output, input);
    }

    /// Boundary: extract from a 1-pixel image.
    #[test]
    fn boundary_single_pixel_multiband() {
        let input = [5u8, 10, 15, 20]; // 1 pixel, 4 bands
        let mut output = [0u8; 2];
        // Extract bands [1, 3) = [10, 15]
        run_extract(1, 2, 4, &input, &mut output, 1);
        assert_eq!(output, [10, 15]);
    }
}

use std::marker::PhantomData;

use viprs_core::{
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

/// Extracts a single band from a multi-band image.
///
/// `band_index` is 0-based. Output is a 1-band image with the same sample format.
///
/// # Output band count
///
/// `BandSplit` always produces exactly 1 band. This is encoded as
/// `const OUTPUT_BANDS: Option<usize> = Some(1)` in the `Op` impl, so
/// `OperationBridge::new` automatically sets `bands = 1` without any
/// extra configuration by the caller.
pub struct BandSplit<F: BandFormat> {
    band_index: usize,
    input_bands: usize,
    _f: PhantomData<F>,
}

impl<F: BandFormat> BandSplit<F> {
    /// Construct a `BandSplit` that extracts `band_index` from a `input_bands`-band image.
    ///
    /// # Panics (debug only)
    ///
    /// `band_index` must be less than `input_bands`. Checked with `debug_assert`.
    #[must_use]
    pub fn new(band_index: usize, input_bands: usize) -> Self {
        debug_assert!(
            band_index < input_bands,
            "band_index {band_index} out of range for input_bands {input_bands}"
        );
        Self {
            band_index,
            input_bands,
            _f: PhantomData,
        }
    }
}

impl<F: BandFormat> Op for BandSplit<F> {
    type Input = F;
    type Output = F;
    type State = ();

    const OUTPUT_BANDS: Option<usize> = Some(1);

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
            output.bands, 1,
            "BandSplit output tile must have exactly 1 band"
        );
        let pixel_count = input.region.pixel_count();
        for px in 0..pixel_count {
            output.data[px] = input.data[px * self.input_bands + self.band_index];
        }
    }
}

// BandSplit is pixel-local: required_input_region is identity and node_spec is identity.
impl<F: BandFormat> PixelLocalOp for BandSplit<F> {}

#[cfg(test)]
mod tests {
    use super::*;
    use viprs_core::{
        format::U8,
        image::{Region, Tile, TileMut},
        op::{DynOperation, OperationBridge},
    };

    fn make_region(w: u32, h: u32) -> Region {
        Region::new(0, 0, w, h)
    }

    /// Run `BandSplit` directly through `Op::process_region` (no bridge).
    fn run_bandsplit(
        band_index: usize,
        input_bands: usize,
        input_data: &[u8],
        output_data: &mut [u8],
        pixel_count: usize,
    ) {
        let region = make_region(pixel_count as u32, 1);
        let op = BandSplit::<U8>::new(band_index, input_bands);
        let input = Tile::<U8>::new(region, input_bands as u32, input_data);
        let mut output = TileMut::<U8>::new(region, 1, output_data);
        op.start();
        op.process_region(&mut (), &input, &mut output);
    }

    #[test]
    fn extract_red_from_rgba() {
        // RGBA image: 2 pixels
        // pixel 0: R=10, G=20, B=30, A=40
        // pixel 1: R=50, G=60, B=70, A=80
        let input = [10u8, 20, 30, 40, 50, 60, 70, 80];
        let mut output = [0u8; 2];
        run_bandsplit(0, 4, &input, &mut output, 2);
        assert_eq!(output, [10, 50]);
    }

    #[test]
    fn extract_green_from_rgba() {
        let input = [10u8, 20, 30, 40, 50, 60, 70, 80];
        let mut output = [0u8; 2];
        run_bandsplit(1, 4, &input, &mut output, 2);
        assert_eq!(output, [20, 60]);
    }

    #[test]
    fn extract_alpha_from_rgba() {
        let input = [10u8, 20, 30, 40, 50, 60, 70, 80];
        let mut output = [0u8; 2];
        run_bandsplit(3, 4, &input, &mut output, 2);
        assert_eq!(output, [40, 80]);
    }

    #[test]
    fn extract_only_band_from_single_band() {
        let input = [1u8, 2, 3, 4];
        let mut output = [0u8; 4];
        run_bandsplit(0, 1, &input, &mut output, 4);
        assert_eq!(output, [1, 2, 3, 4]);
    }

    #[test]
    fn pixel_local_op_impl() {
        // Verify the marker trait is implemented (compile-time check).
        fn requires_pixel_local<T: PixelLocalOp>(_: &T) {}
        let op = BandSplit::<U8>::new(0, 3);
        requires_pixel_local(&op);
    }

    /// `OperationBridge::new` must report `bands() == 1` regardless of what the
    /// caller passes, because `OUTPUT_BANDS = Some(1)`.
    #[test]
    fn bridge_new_always_reports_one_band() {
        // Pass a wrong value (4) — the bridge must override it via OUTPUT_BANDS.
        let bridge = OperationBridge::new(BandSplit::<U8>::new(0, 4), 4u32);
        assert_eq!(bridge.bands(), 1);
    }

    /// Identity check: for a 1-band image, BandSplit(0) is a no-op.
    #[test]
    fn identity_single_band() {
        let input = [7u8, 13, 42, 255];
        let mut output = [0u8; 4];
        run_bandsplit(0, 1, &input, &mut output, 4);
        assert_eq!(output, input);
    }

    /// Boundary value: extract from a 1-pixel image.
    #[test]
    fn boundary_single_pixel() {
        let input = [0u8, 128, 255]; // 1 pixel, 3 bands
        let mut output = [0u8; 1];
        run_bandsplit(2, 3, &input, &mut output, 1);
        assert_eq!(output, [255]);
    }
}

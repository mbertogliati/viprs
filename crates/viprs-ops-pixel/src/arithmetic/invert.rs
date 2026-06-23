use std::marker::PhantomData;

pub use viprs_core::shared_ops::invertible::Invertible;
use viprs_core::{
    format::NumericBand,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

/// Element-wise inversion of all samples in a tile.
///
/// The exact inversion semantic is type-dependent (see `Invertible` impls above).
///
/// # Examples
/// ```ignore
/// use viprs_ops_pixel::arithmetic::invert::Invert;
///
/// let op = Invert::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct Invert<F: NumericBand> {
    _format: PhantomData<F>,
}

impl<F: NumericBand> Invert<F>
where
    F::Sample: Invertible,
{
    #[must_use]
    /// Creates a new `Invert`.
    pub const fn new() -> Self {
        Self {
            _format: PhantomData,
        }
    }
}

impl<F: NumericBand> Default for Invert<F>
where
    F::Sample: Invertible,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> Op for Invert<F>
where
    F: NumericBand,
    F::Sample: Invertible,
{
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
        F::Sample::invert_bulk(input.data, output.data);
    }
}

/// `Invert` is pixel-local: it reads one sample and writes one sample with no
/// neighbourhood access and identity tile geometry. See `PixelLocalOp` for invariants.
impl<F> PixelLocalOp for Invert<F>
where
    F: NumericBand,
    F::Sample: Invertible,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{format::F32, image::Region};

    fn make_region(w: u32, h: u32) -> Region {
        Region::new(0, 0, w, h)
    }

    #[test]
    fn invert_f32_known_values() {
        let op = Invert::<F32>::new();
        let r = make_region(2, 1);
        let input_data = vec![0.0f32, 1.0];
        let mut output_data = vec![0.0f32; 2];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert!((output_data[0] - 1.0).abs() < f32::EPSILON);
        assert!((output_data[1] - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn metadata_and_all_numeric_invertible_types_match_contract() {
        use viprs_core::format::U8;

        assert_eq!(Invertible::invert(0u8), 255);
        assert_eq!(Invertible::invert(0u16), 65_535);
        assert_eq!(Invertible::invert(-7i16), 7);
        assert_eq!(Invertible::invert(-9i32), 9);
        assert_eq!(Invertible::invert(11u32), u32::MAX - 11);
        assert!((Invertible::invert(0.25f32) - 0.75).abs() < f32::EPSILON);
        assert!((Invertible::invert(0.25f64) - 0.75).abs() < f64::EPSILON);

        let op = Invert::<U8>::new();
        let region = Region::new(3, -2, 4, 5);
        assert_eq!(op.demand_hint(), viprs_core::image::DemandHint::ThinStrip);
        assert_eq!(op.required_input_region(&region), region);
        op.start();
    }

    #[test]
    fn default_constructor_matches_new() {
        use viprs_core::format::U16;

        let region = Region::new(0, 0, 3, 1);
        let input_data = [1u16, 1024, 65_535];
        let mut output_data = [0u16; 3];
        let input = Tile::<U16>::new(region, 1, &input_data);
        let mut output = TileMut::<U16>::new(region, 1, &mut output_data);
        let mut state = ();

        Invert::<U16>::default().process_region(&mut state, &input, &mut output);

        assert_eq!(output_data, [65_534, 64_511, 0]);
    }

    #[test]
    fn invert_signed_min_saturates_to_max() {
        use viprs_core::format::{I16, I32};

        let region = Region::new(0, 0, 1, 1);

        let op_i16 = Invert::<I16>::new();
        let input_i16 = [i16::MIN];
        let mut output_i16 = [0i16; 1];
        let input = Tile::<I16>::new(region, 1, &input_i16);
        let mut output = TileMut::<I16>::new(region, 1, &mut output_i16);
        let mut state = ();
        op_i16.process_region(&mut state, &input, &mut output);
        assert_eq!(output_i16, [i16::MAX]);

        let op_i32 = Invert::<I32>::new();
        let input_i32 = [i32::MIN];
        let mut output_i32 = [0i32; 1];
        let input = Tile::<I32>::new(region, 1, &input_i32);
        let mut output = TileMut::<I32>::new(region, 1, &mut output_i32);
        let mut state = ();
        op_i32.process_region(&mut state, &input, &mut output);
        assert_eq!(output_i32, [i32::MAX]);
    }

    #[test]
    fn invert_large_rows_cover_simd_dispatch_and_scalar_remainder() {
        use viprs_core::format::U8;

        let op = Invert::<U8>::new();
        let r = make_region(33, 1);
        let input_data = (0u8..33).collect::<Vec<_>>();
        let mut output_data = vec![0u8; input_data.len()];
        let input = Tile::<U8>::new(r, 1, &input_data);
        let mut output = TileMut::<U8>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        let expected: Vec<u8> = input_data.iter().map(|&sample| 255 - sample).collect();
        assert_eq!(output_data, expected);
    }

    #[test]
    fn invert_u16_bulk_path_covers_vector_tail_and_scalar_remainder() {
        use viprs_core::format::U16;

        let op = Invert::<U16>::new();
        let input_data = (0u16..43)
            .map(|sample| sample.saturating_mul(977))
            .collect::<Vec<_>>();
        let mut output_data = vec![0u16; input_data.len()];
        let region = make_region(input_data.len() as u32, 1);
        let input = Tile::<U16>::new(region, 1, &input_data);
        let mut output = TileMut::<U16>::new(region, 1, &mut output_data);
        let mut state = ();

        op.process_region(&mut state, &input, &mut output);

        let expected = input_data
            .iter()
            .map(|&sample| 65_535 - sample)
            .collect::<Vec<_>>();
        assert_eq!(output_data, expected);
    }

    #[test]
    fn invert_f32_bulk_path_covers_vector_tail_and_scalar_remainder() {
        let op = Invert::<F32>::new();
        let input_data = (0..21).map(|idx| idx as f32 / 20.0).collect::<Vec<_>>();
        let mut output_data = vec![0.0f32; input_data.len()];
        let region = make_region(input_data.len() as u32, 1);
        let input = Tile::<F32>::new(region, 1, &input_data);
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
        let mut state = ();

        op.process_region(&mut state, &input, &mut output);

        for (actual, expected) in output_data
            .iter()
            .zip(input_data.iter().map(|&sample| 1.0 - sample))
        {
            assert!((actual - expected).abs() < f32::EPSILON);
        }
    }

    proptest! {
        #[test]
        fn invert_twice_is_identity_f32(
            pixels in proptest::collection::vec(0.0f32..=1.0f32, 1..=64)
        ) {
            let len = pixels.len();
            let op = Invert::<F32>::new();
            let r = Region::new(0, 0, len as u32, 1);

            // First inversion
            let mut mid = vec![0.0f32; len];
            {
                let input = Tile::<F32>::new(r, 1, &pixels);
                let mut output = TileMut::<F32>::new(r, 1, &mut mid);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
            }

            // Second inversion
            let mut result = vec![0.0f32; len];
            {
                let input = Tile::<F32>::new(r, 1, &mid);
                let mut output = TileMut::<F32>::new(r, 1, &mut result);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
            }

            for (a, b) in pixels.iter().zip(result.iter()) {
                prop_assert!((a - b).abs() < 1e-6, "double-invert mismatch: {} vs {}", a, b);
            }
        }

        /// For u8: invert(invert(x)) == x for all x in 0..=255.
        #[test]
        fn invert_twice_is_identity_u8(
            pixels in proptest::collection::vec(0u8..=255u8, 1..=64)
        ) {
            use viprs_core::format::U8;

            let len = pixels.len();
            let op = Invert::<U8>::new();
            let r = Region::new(0, 0, len as u32, 1);

            // First inversion.
            let mut mid = vec![0u8; len];
            {
                let input = Tile::<U8>::new(r, 1, &pixels);
                let mut output = TileMut::<U8>::new(r, 1, &mut mid);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
            }

            // Second inversion.
            let mut result = vec![0u8; len];
            {
                let input = Tile::<U8>::new(r, 1, &mid);
                let mut output = TileMut::<U8>::new(r, 1, &mut result);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
            }

            prop_assert_eq!(result, pixels, "double-invert must be identity for u8");
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    proptest! {
        #[test]
        fn invert_u8_avx2_matches_scalar_reference(
            pixels in proptest::collection::vec(any::<u8>(), 33..=257)
        ) {
            if !std::arch::is_x86_feature_detected!("avx2") {
                return;
            }

            use viprs_core::format::U8;

            let len = pixels.len();
            let region = Region::new(0, 0, len as u32, 1);
            let mut output = vec![0u8; len];
            let input = Tile::<U8>::new(region, 1, &pixels);
            let mut output_tile = TileMut::<U8>::new(region, 1, &mut output);
            let mut state = ();

            Invert::<U8>::new().process_region(&mut state, &input, &mut output_tile);

            let expected: Vec<u8> = pixels.iter().map(|&sample| 255 - sample).collect();
            prop_assert_eq!(output, expected);
        }

        #[test]
        fn invert_f32_avx2_matches_scalar_reference(
            pixels in proptest::collection::vec(0.0f32..=1.0f32, 17..=129)
        ) {
            if !std::arch::is_x86_feature_detected!("avx2") {
                return;
            }

            let len = pixels.len();
            let region = Region::new(0, 0, len as u32, 1);
            let mut output = vec![0.0f32; len];
            let input = Tile::<F32>::new(region, 1, &pixels);
            let mut output_tile = TileMut::<F32>::new(region, 1, &mut output);
            let mut state = ();

            Invert::<F32>::new().process_region(&mut state, &input, &mut output_tile);

            for (actual, expected) in output.iter().zip(pixels.iter().map(|&sample| 1.0 - sample)) {
                prop_assert!((actual - expected).abs() < f32::EPSILON);
            }
        }
    }

    /// Boundary value: invert(0u8) == 255, invert(255u8) == 0.
    #[test]
    fn invert_u8_boundary_values() {
        use viprs_core::format::U8;

        let op = Invert::<U8>::new();
        let r = Region::new(0, 0, 2, 1);
        let input_data = vec![0u8, 255u8];
        let mut output_data = vec![0u8; 2];
        let input = Tile::<U8>::new(r, 1, &input_data);
        let mut output = TileMut::<U8>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data[0], 255u8, "invert(0) must be 255");
        assert_eq!(output_data[1], 0u8, "invert(255) must be 0");
    }

    /// Ported from libvips `test_arithmetic.py::test_invert`.
    ///
    /// libvips test: `~x & 0xff` for uchar format, applied per-pixel.
    /// libvips only tests invert on uchar (`fmt=[pyvips.BandFormat.UCHAR]`)
    /// because the max-value trim makes it hard to compare other formats.
    /// This test verifies the per-pixel contract for a known 3-band sRGB-like tile.
    #[test]
    fn invert_u8_multiband_pixel_contract() {
        use viprs_core::format::U8;

        // 2 pixels × 3 bands (RGB layout)
        // Pixel 0: [10, 20, 30] → invert → [245, 235, 225]
        // Pixel 1: [100, 150, 200] → invert → [155, 105, 55]
        let op = Invert::<U8>::new();
        let r = Region::new(0, 0, 2, 1);
        let input_data = vec![10u8, 20, 30, 100, 150, 200];
        let mut output_data = vec![0u8; 6];
        let input = Tile::<U8>::new(r, 3, &input_data);
        let mut output = TileMut::<U8>::new(r, 3, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data[0], 245, "band 0 px 0");
        assert_eq!(output_data[1], 235, "band 1 px 0");
        assert_eq!(output_data[2], 225, "band 2 px 0");
        assert_eq!(output_data[3], 155, "band 0 px 1");
        assert_eq!(output_data[4], 105, "band 1 px 1");
        assert_eq!(output_data[5], 55, "band 2 px 1");
    }

    /// Ported from libvips `test_arithmetic.py::test_invert`.
    ///
    /// libvips test: for u16 format, invert(x) = 65535 - x.
    /// Boundary: invert(0) = 65535, invert(65535) = 0.
    #[test]
    fn invert_u16_boundary_values() {
        use viprs_core::format::U16;

        let op = Invert::<U16>::new();
        let r = Region::new(0, 0, 3, 1);
        let input_data = vec![0u16, 32768, 65535];
        let mut output_data = vec![0u16; 3];
        let input = Tile::<U16>::new(r, 1, &input_data);
        let mut output = TileMut::<U16>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data[0], 65535u16, "invert(0u16) must be 65535");
        assert_eq!(output_data[1], 32767u16, "invert(32768u16) must be 32767");
        assert_eq!(output_data[2], 0u16, "invert(65535u16) must be 0");
    }
}

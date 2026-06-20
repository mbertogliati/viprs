use viprs_core::{
    format::F32,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

/// Applies the `Delta E 1976` colour transform to image pixels. Use it when a pipeline needs to
/// move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs::domain::crate::colour::de76::DE76;
///
/// let op = DE76;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct DE76;

#[inline(always)]
fn delta_e_76(l1: f32, a1: f32, b1: f32, l2: f32, a2: f32, b2: f32) -> f32 {
    let dl = l1 - l2;
    let da = a1 - a2;
    let db = b1 - b2;
    db.mul_add(db, da.mul_add(da, dl * dl)).sqrt()
}

impl Op for DE76 {
    type Input = F32;
    type Output = F32;
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
    fn process_region(&self, (): &mut (), input: &Tile<F32>, output: &mut TileMut<F32>) {
        for (pixel_in, pixel_out) in input.data.chunks_exact(6).zip(output.data.iter_mut()) {
            *pixel_out = delta_e_76(
                pixel_in[0],
                pixel_in[1],
                pixel_in[2],
                pixel_in[3],
                pixel_in[4],
                pixel_in[5],
            );
        }
    }
}

impl PixelLocalOp for DE76 {}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        image::{Region, Tile, TileMut},
        op::OperationBridge,
    };

    fn make_region(pixels: usize) -> Region {
        Region::new(0, 0, pixels as u32, 1)
    }

    proptest! {
        #[test]
        fn de76_identity_is_zero_proptest(
            l in 0.0f32..100.0,
            a in -128.0f32..128.0,
            b in -128.0f32..128.0,
        ) {
            let op = DE76;
            let input_data = [l, a, b, l, a, b];
            let mut output_data = [1.0f32; 1];
            let region = make_region(1);
            let input = Tile::new(region, 6, &input_data);
            let mut output = TileMut::new(region, 1, &mut output_data);
            op.process_region(&mut (), &input, &mut output);

            prop_assert!(output_data[0].abs() < 1e-6);
        }
    }

    #[test]
    fn pythagorean_distance_matches_expected_value() {
        let op = DE76;
        let input_data = [50.0_f32, 10.0, -5.0, 40.0, 10.0, -5.0];
        let mut output_data = [0.0f32; 1];
        let region = make_region(1);
        let input = Tile::new(region, 6, &input_data);
        let mut output = TileMut::new(region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);

        assert!((output_data[0] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn sharma_reference_pair_matches_known_distance() {
        let op = DE76;
        let input_data = [50.0_f32, 2.6772, -79.7751, 50.0, 0.0, -82.7485];
        let mut output_data = [0.0_f32; 1];
        let region = make_region(1);
        let input = Tile::new(region, 6, &input_data);
        let mut output = TileMut::new(region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);

        assert!((output_data[0] - 4.001_063_3).abs() < 1e-5);
    }

    #[test]
    fn demand_hint_is_any() {
        assert_eq!(DE76.demand_hint(), DemandHint::Any);
    }

    #[test]
    fn required_input_region_is_identity() {
        let region = Region::new(2, 3, 5, 7);
        assert_eq!(DE76.required_input_region(&region), region);
    }

    #[test]
    fn start_returns_unit() {
        DE76.start();
    }

    #[test]
    fn operation_bridge_forces_single_output_band() {
        let bridge = OperationBridge::new_pixel_local(DE76, 6);
        assert_eq!(bridge.bands, 1);
    }

    /// Ported from libvips test_colour.py::test_dE76.
    ///
    /// libvips reference:
    ///   reference = Lab(50, 10, 20), sample = Lab(40, -20, 10)
    ///   difference(10,10) ≈ 33.166
    ///   Verified against http://www.brucelindbloom.com
    #[test]
    fn libvips_reference_pair_de76_33_166() {
        let op = DE76;
        // Input: [ref_L, ref_a, ref_b, samp_L, samp_a, samp_b]
        let input_data = [50.0_f32, 10.0, 20.0, 40.0, -20.0, 10.0];
        let mut output_data = [0.0_f32; 1];
        let region = make_region(1);
        let input = Tile::new(region, 6, &input_data);
        let mut output = TileMut::new(region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);

        // dE76 = sqrt((50-40)^2 + (10-(-20))^2 + (20-10)^2)
        //       = sqrt(100 + 900 + 100) = sqrt(1100) ≈ 33.166
        assert!(
            (output_data[0] - 33.166).abs() < 0.01,
            "dE76={} expected ≈33.166",
            output_data[0]
        );
    }

    /// Ported from libvips test_colour.py::test_dE76.
    ///
    /// libvips test: extra band value (alpha=42) in the reference image is
    /// copied unmodified into the output. We test that the DE76 op produces
    /// only the distance value (single output band per pixel-pair).
    #[test]
    fn de76_produces_single_band_distance_ignoring_extra_channels() {
        // Two pixels: each has 6 channels (3 ref + 3 sample).
        // Pixel 0: identical → distance 0.
        // Pixel 1: dL=10, da=0, db=0 → distance 10.
        let op = DE76;
        let input_data = [
            50.0_f32, 0.0, 0.0, 50.0, 0.0, 0.0, // pixel 0: identical
            60.0_f32, 0.0, 0.0, 50.0, 0.0, 0.0, // pixel 1: dL=10
        ];
        let mut output_data = [99.0_f32; 2];
        let region = Region::new(0, 0, 2, 1);
        let input = Tile::new(region, 6, &input_data);
        let mut output = TileMut::new(region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);

        assert!(
            output_data[0].abs() < 1e-5,
            "pixel 0: dE76={}",
            output_data[0]
        );
        assert!(
            (output_data[1] - 10.0).abs() < 1e-4,
            "pixel 1: dE76={}",
            output_data[1]
        );
    }
}

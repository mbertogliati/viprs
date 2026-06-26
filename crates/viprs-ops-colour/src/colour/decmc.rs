use viprs_core::{
    format::F32,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

/// Applies the `Delta E CMC` colour transform to image pixels. Use it when a pipeline needs to
/// move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs_ops_colour::colour::decmc::DECMC;
///
/// let op = DECMC;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct DECMC;

#[inline(always)]
fn delta_e_cmc(l1: f32, c1: f32, h1: f32, l2: f32, c2: f32, h2: f32) -> f32 {
    let h1_rad = h1.to_radians();
    let h2_rad = h2.to_radians();
    let a1 = c1 * h1_rad.cos();
    let b1 = c1 * h1_rad.sin();
    let a2 = c2 * h2_rad.cos();
    let b2 = c2 * h2_rad.sin();

    let dl = l1 - l2;
    let da = a1 - a2;
    let db = b1 - b2;
    db.mul_add(db, da.mul_add(da, dl * dl)).sqrt()
}

impl Op for DECMC {
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
            *pixel_out = delta_e_cmc(
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

impl PixelLocalOp for DECMC {}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
      format::F32,
      image::{InMemoryImage, Region, Tile, TileMut},
      op::OperationBridge,
    };

    fn make_region(pixels: usize) -> Region {
        Region::new(0, 0, pixels as u32, 1)
    }

    fn run_distance(input_data: Vec<f32>) -> Vec<f32> {
        let pixels = input_data.len() / 6;
        let input_image = InMemoryImage::<F32>::from_buffer(pixels as u32, 1, 6, input_data).unwrap();
        let region = make_region(pixels);
        let input = Tile::new(region, 6, input_image.pixels());
        let mut output_data = vec![0.0_f32; pixels];
        let mut output = TileMut::new(region, 1, &mut output_data);

        DECMC.process_region(&mut (), &input, &mut output);

        output_data
    }

    proptest! {
        #[test]
        fn identical_lch_triplets_have_zero_distance_for_any_finite_values(
            lightness in -1_000.0_f32..1_000.0,
            chroma in -1_000.0_f32..1_000.0,
            hue in -720.0_f32..720.0,
        ) {
            let output = run_distance(vec![lightness, chroma, hue, lightness, chroma, hue]);
            prop_assert!(output[0].abs() < 1e-4, "distance={}", output[0]);
        }
    }

    #[test]
    fn identical_lab_triplets_have_zero_distance() {
        let op = DECMC;
        let input_data = [60.0_f32, -20.0, 10.0, 60.0, -20.0, 10.0];
        let mut output_data = [1.0f32; 1];
        let region = make_region(1);
        let input = Tile::new(region, 6, &input_data);
        let mut output = TileMut::new(region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);

        assert!(output_data[0].abs() < 1e-6);
    }

    #[test]
    fn lch_distance_matches_reference_value() {
        let op = DECMC;
        let input_data = [50.0_f32, 10.0, 0.0, 50.0, 10.0, 90.0];
        let mut output_data = [0.0f32; 1];
        let region = make_region(1);
        let input = Tile::new(region, 6, &input_data);
        let mut output = TileMut::new(region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);

        assert!(
            (output_data[0] - 14.142_136).abs() < 1e-5,
            "dECMC={}",
            output_data[0]
        );
    }

    #[test]
    fn zero_chroma_ignores_hue_difference() {
        let output = run_distance(vec![50.0, 0.0, 0.0, 50.0, 0.0, 270.0]);
        assert!(output[0].abs() < 1e-6);
    }

    #[test]
    fn demand_hint_is_any() {
        assert_eq!(DECMC.demand_hint(), DemandHint::Any);
    }

    #[test]
    fn required_input_region_is_identity() {
        let region = Region::new(2, 4, 6, 8);
        assert_eq!(DECMC.required_input_region(&region), region);
    }

    #[test]
    fn start_returns_unit() {
        DECMC.start();
    }

    #[test]
    fn operation_bridge_forces_single_output_band() {
        let bridge = OperationBridge::new_pixel_local(DECMC, 6);
        assert_eq!(bridge.bands, 1);
    }
}

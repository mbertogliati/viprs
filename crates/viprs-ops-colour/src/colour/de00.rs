use viprs_core::{
    error::{BuildError, ViprsError},
    format::F32,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

/// Applies the `Delta E 2000` colour transform to image pixels. Use it when a pipeline needs to
/// move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs_ops_colour::colour::de00::DE00;
///
/// let op = DE00;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct DE00;

const TWENTY_FIVE_POW_SEVEN: f64 = 6_103_515_625.0;

#[inline(always)]
fn ab_to_hue_degrees_f64(a: f64, b: f64) -> f64 {
    if a == 0.0 {
        if b < 0.0 {
            270.0
        } else if b == 0.0 {
            0.0
        } else {
            90.0
        }
    } else {
        let t = (b / a).atan();
        if a > 0.0 {
            if b < 0.0 {
                (t + core::f64::consts::TAU).to_degrees()
            } else {
                t.to_degrees()
            }
        } else {
            (t + core::f64::consts::PI).to_degrees()
        }
    }
}

#[inline(always)]
fn delta_e_00(l1: f32, a1: f32, b1: f32, l2: f32, a2: f32, b2: f32) -> f32 {
    let l1 = f64::from(l1);
    let a1 = f64::from(a1);
    let b1 = f64::from(b1);
    let l2 = f64::from(l2);
    let a2 = f64::from(a2);
    let b2 = f64::from(b2);

    let c1 = a1.hypot(b1);
    let c2 = a2.hypot(b2);
    let cb = f64::midpoint(c1, c2);

    let cb7 = cb.powi(7);
    let g = 0.5 * (1.0 - (cb7 / (cb7 + TWENTY_FIVE_POW_SEVEN)).sqrt());

    let l1d = l1;
    let a1d = (1.0 + g) * a1;
    let b1d = b1;
    let c1d = a1d.hypot(b1d);
    let h1d = ab_to_hue_degrees_f64(a1d, b1d);

    let l2d = l2;
    let a2d = (1.0 + g) * a2;
    let b2d = b2;
    let c2d = a2d.hypot(b2d);
    let h2d = ab_to_hue_degrees_f64(a2d, b2d);

    let ldb = f64::midpoint(l1d, l2d);
    let cdb = f64::midpoint(c1d, c2d);
    let hdb = if (h1d - h2d).abs() < 180.0 {
        f64::midpoint(h1d, h2d)
    } else {
        (h1d + h2d - 360.0).abs() / 2.0
    };

    let hdbd = (hdb - 275.0) / 25.0;
    let dtheta = 30.0 * (-(hdbd * hdbd)).exp();
    let cdb7 = cdb.powi(7);
    let rc = 2.0 * (cdb7 / (cdb7 + TWENTY_FIVE_POW_SEVEN)).sqrt();

    let rt = -(2.0 * dtheta).to_radians().sin() * rc;
    let t = 0.20f64.mul_add(
        -4.0f64.mul_add(hdb, -63.0).to_radians().cos(),
        0.32f64.mul_add(
            3.0f64.mul_add(hdb, 6.0).to_radians().cos(),
            0.24f64.mul_add(
                (2.0 * hdb).to_radians().cos(),
                0.17f64.mul_add(-(hdb - 30.0).to_radians().cos(), 1.0),
            ),
        ),
    );

    let ldb50 = ldb - 50.0;
    let sl = 1.0 + (0.015 * ldb50 * ldb50) / (20.0 + ldb50 * ldb50).sqrt();
    let sc = 0.045f64.mul_add(cdb, 1.0);
    let sh = (0.015 * cdb).mul_add(t, 1.0);

    let dhd = if (h1d - h2d).abs() < 180.0 {
        h1d - h2d
    } else {
        360.0 - (h1d - h2d)
    };

    let dld = l1d - l2d;
    let dcd = c1d - c2d;
    let dhd_term = 2.0 * (c1d * c2d).sqrt() * (dhd / 2.0).to_radians().sin();

    let nl = dld / sl;
    let nc = dcd / sc;
    let nh = dhd_term / sh;

    (rt * nc).mul_add(nh, nl * nl + nc * nc + nh * nh).sqrt() as f32
}

impl Op for DE00 {
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

    fn validate_build_contract(
        &self,
        input_bands: u32,
        output_bands: u32,
    ) -> Result<(), BuildError> {
        if input_bands == 6 && output_bands == 1 {
            Ok(())
        } else {
            Err(BuildError::InvalidOperationBands {
                op: "DE00",
                input_bands,
                output_bands,
                expected: "6 bands (two Lab triplets)",
                expected_output: "1 band",
            })
        }
    }

    fn validate_region_contract(
        &self,
        _input_region: Region,
        input_bands: u32,
        _output_region: Region,
        output_bands: u32,
    ) -> Result<(), ViprsError> {
        self.validate_build_contract(input_bands, output_bands)
            .map_err(ViprsError::from)
    }

    #[inline]
    fn process_region(&self, (): &mut (), input: &Tile<F32>, output: &mut TileMut<F32>) {
        for (pixel_in, pixel_out) in input.data.chunks_exact(6).zip(output.data.iter_mut()) {
            *pixel_out = delta_e_00(
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

impl PixelLocalOp for DE00 {}

#[cfg(all(test, feature = "_integration"))]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        error::BuildError,
        image::{Region, Tile, TileMut},
        op::OperationBridge,
    };
    use viprs_runtime::{
        pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sources::memory::MemorySource,
    };

    fn make_region(pixels: usize) -> Region {
        Region::new(0, 0, pixels as u32, 1)
    }

    fn run_distance(input_data: [f32; 6]) -> f32 {
        let op = DE00;
        let mut output_data = [0.0_f32; 1];
        let region = make_region(1);
        let input = Tile::new(region, 6, &input_data);
        let mut output = TileMut::new(region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);
        output_data[0]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn identical_lab_triplets_have_zero_distance_proptest(
            l in 0.0f32..100.0,
            a in -127.0f32..127.0,
            b in -127.0f32..127.0,
        ) {
            let distance = run_distance([l, a, b, l, a, b]);
            prop_assert!(distance.abs() < 1e-4, "distance={distance}");
        }

        #[test]
        fn de00_is_symmetric_and_non_negative_proptest(
            l1 in 0.0f32..100.0,
            a1 in -127.0f32..127.0,
            b1 in -127.0f32..127.0,
            l2 in 0.0f32..100.0,
            a2 in -127.0f32..127.0,
            b2 in -127.0f32..127.0,
        ) {
            let forward = run_distance([l1, a1, b1, l2, a2, b2]);
            let reverse = run_distance([l2, a2, b2, l1, a1, b1]);

            prop_assert!(forward >= 0.0, "forward distance={forward}");
            prop_assert!(reverse >= 0.0, "reverse distance={reverse}");
            prop_assert!((forward - reverse).abs() < 1e-4, "forward={forward} reverse={reverse}");
        }
    }

    #[test]
    fn ciede2000_reference_pair_matches_sharma_dataset() {
        let op = DE00;
        let input_data = [50.0_f32, 2.6772, -79.7751, 50.0, 0.0, -82.7485];
        let mut output_data = [0.0f32; 1];
        let region = make_region(1);
        let input = Tile::new(region, 6, &input_data);
        let mut output = TileMut::new(region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);

        assert!(
            (output_data[0] - 2.0425).abs() < 1e-3,
            "dE00={}",
            output_data[0]
        );
    }

    #[test]
    fn identical_lab_triplets_have_zero_distance() {
        let op = DE00;
        let input_data = [60.0_f32, -20.0, 10.0, 60.0, -20.0, 10.0];
        let mut output_data = [1.0f32; 1];
        let region = make_region(1);
        let input = Tile::new(region, 6, &input_data);
        let mut output = TileMut::new(region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);

        assert!(output_data[0].abs() < 1e-6);
    }

    #[test]
    fn operation_bridge_forces_single_output_band() {
        let bridge = OperationBridge::new_pixel_local(DE00, 6);
        assert_eq!(bridge.bands, 1);
    }

    #[test]
    fn pipeline_rejects_three_band_de00_input() {
        let source = MemorySource::<F32>::new(1, 1, 3, vec![0.0, 0.0, 0.0]).unwrap();
        let err = match PipelineBuilder::from_source(source)
            .then(Box::new(OperationBridge::new_pixel_local(DE00, 3)))
        {
            Ok(_) => panic!("expected DE00 build contract to reject 3-band input"),
            Err(err) => err,
        };

        match err {
            BuildError::InvalidOperationBands {
                op,
                input_bands,
                expected,
                ..
            } => {
                assert_eq!(op, "DE00");
                assert_eq!(input_bands, 3);
                assert_eq!(expected, "6 bands (two Lab triplets)");
            }
            other => panic!("expected InvalidOperationBands, got {other:?}"),
        }
    }

    #[test]
    fn pipeline_emits_non_zero_de00_for_valid_lab_pair() {
        let source =
            MemorySource::<F32>::new(1, 1, 6, vec![50.0, 10.0, 20.0, 40.0, -20.0, 10.0]).unwrap();
        let pipeline = PipelineBuilder::from_source(source)
            .then(Box::new(OperationBridge::new_pixel_local(DE00, 6)))
            .unwrap()
            .build()
            .unwrap();

        let image = pipeline
            .run_to_image::<F32, _>(&RayonScheduler::new(1).unwrap())
            .unwrap();

        assert_eq!(image.bands(), 1);
        assert!(image.pixels()[0] > 0.0, "dE00={}", image.pixels()[0]);
    }

    /// Ported from libvips test_colour.py::test_dE00.
    ///
    /// libvips reference:
    ///   reference = Lab(50, 10, 20), sample = Lab(40, -20, 10)
    ///   result at pixel (10,10) ≈ 30.238
    ///   Verified against http://www.brucelindbloom.com
    #[test]
    fn libvips_reference_pair_de00_30_238() {
        let op = DE00;
        // Input: [ref_L, ref_a, ref_b, samp_L, samp_a, samp_b]
        let input_data = [50.0_f32, 10.0, 20.0, 40.0, -20.0, 10.0];
        let mut output_data = [0.0_f32; 1];
        let region = make_region(1);
        let input = Tile::new(region, 6, &input_data);
        let mut output = TileMut::new(region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);

        assert!(
            (output_data[0] - 30.238).abs() < 0.01,
            "dE00={} expected ≈30.238",
            output_data[0]
        );
    }

    /// Ported from libvips test_colour.py::test_dE00.
    ///
    /// libvips test: the extra band (alpha=42) passed into the reference image
    /// is preserved in the output. We verify dE00 produces only the single
    /// distance band per pixel-pair.
    #[test]
    fn de00_single_output_band_per_pixel_pair() {
        // Two pixels: pixel 0 identical, pixel 1 has dL=10.
        let op = DE00;
        let input_data = [
            50.0_f32, 0.0, 0.0, 50.0, 0.0, 0.0, // pixel 0: identical → 0
            60.0_f32, 0.0, 0.0, 50.0, 0.0, 0.0, // pixel 1: dL=10
        ];
        let mut output_data = [99.0_f32; 2];
        let region = Region::new(0, 0, 2, 1);
        let input = Tile::new(region, 6, &input_data);
        let mut output = TileMut::new(region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);

        assert!(
            output_data[0].abs() < 1e-5,
            "pixel 0 dE00={}",
            output_data[0]
        );
        // dL=10 for neutral grey → roughly 10 / sl (sl ≈ 1 for L=55)
        assert!(output_data[1] > 0.0, "pixel 1 dE00 must be positive");
    }

    #[test]
    fn extreme_lab_corners_keep_distance_finite_and_non_negative() {
        for triplet in [
            [0.0_f32, -127.0, -127.0],
            [0.0, -127.0, 127.0],
            [0.0, 127.0, -127.0],
            [0.0, 127.0, 127.0],
            [100.0, -127.0, -127.0],
            [100.0, -127.0, 127.0],
            [100.0, 127.0, -127.0],
            [100.0, 127.0, 127.0],
        ] {
            let distance = run_distance([
                triplet[0], triplet[1], triplet[2], triplet[0], triplet[1], triplet[2],
            ]);
            assert!(
                distance.is_finite(),
                "distance must stay finite for {triplet:?}"
            );
            assert!(distance.abs() < 1e-4, "distance={distance} for {triplet:?}");
        }
    }

    #[test]
    fn boundary_pairs_cover_hue_wrap_and_low_chroma_paths() {
        let low_chroma = run_distance([0.0_f32, 0.0, 0.0, 100.0, 0.0, 0.0]);
        let hue_wrap_forward = run_distance([100.0_f32, 127.0, 127.0, 100.0, 127.0, -127.0]);
        let hue_wrap_reverse = run_distance([100.0_f32, 127.0, -127.0, 100.0, 127.0, 127.0]);

        assert!(low_chroma.is_finite());
        assert!(low_chroma >= 0.0);
        assert!(hue_wrap_forward.is_finite());
        assert!(hue_wrap_forward >= 0.0);
        assert!((hue_wrap_forward - hue_wrap_reverse).abs() < 1e-4);
    }
}

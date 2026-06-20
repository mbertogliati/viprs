use viprs_core::{
    colorspace::{Hsv, SRgb},
    colour::ColourConvert,
    format::{F32, U8},
    image::{DemandHint, Region, Tile, TileMut},
};

/// Applies the `hsv to srgb` colour transform to image pixels. Use it when a pipeline needs to
/// move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs::domain::crate::colour::hsv_to_srgb::HsvToSRgb;
///
/// let op = HsvToSRgb;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct HsvToSRgb;

/// Convert one HSV pixel (f32 × 3, H ∈ [0,360), S ∈ [0,1], V ∈ [0,1])
/// to sRGB (u8 × 3).
/// Standard HSV→RGB sector formula.
#[inline(always)]
fn hsv_f32_to_srgb_u8(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    if s == 0.0 {
        let val = (v * 255.0).round() as u8;
        return (val, val, val);
    }

    // Normalise H to [0, 6)
    let h_norm = h / 60.0;
    let i = h_norm.floor() as i32 % 6;
    let f = h_norm - h_norm.floor();

    let p = v * (1.0 - s);
    let q = v * s.mul_add(-f, 1.0);
    let t = v * s.mul_add(-(1.0 - f), 1.0);

    let (r, g, b) = match i {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q), // i == 5
    };

    (
        (r * 255.0).round() as u8,
        (g * 255.0).round() as u8,
        (b * 255.0).round() as u8,
    )
}

impl ColourConvert<Hsv, SRgb> for HsvToSRgb {
    type InputFormat = F32;
    type OutputFormat = U8;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn convert_region(&self, (): &mut (), input: &Tile<F32>, output: &mut TileMut<U8>) {
        for (pixel_in, pixel_out) in input
            .data
            .chunks_exact(3)
            .zip(output.data.chunks_exact_mut(3))
        {
            let (r, g, b) = hsv_f32_to_srgb_u8(pixel_in[0], pixel_in[1], pixel_in[2]);
            pixel_out[0] = r;
            pixel_out[1] = g;
            pixel_out[2] = b;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use viprs_core::image::{Region, Tile, TileMut};

    fn make_region(pixels: usize) -> Region {
        Region::new(0, 0, pixels as u32, 1)
    }

    #[test]
    fn red_hsv_to_srgb() {
        let converter = HsvToSRgb;
        let input_data: [f32; 3] = [0.0, 1.0, 1.0];
        let mut output_data = [0_u8; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);
        assert_eq!(output_data[0], 255);
        assert_eq!(output_data[1], 0);
        assert_eq!(output_data[2], 0);
    }

    #[test]
    fn green_hsv_to_srgb() {
        let converter = HsvToSRgb;
        let input_data: [f32; 3] = [120.0, 1.0, 1.0];
        let mut output_data = [0_u8; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);
        assert_eq!(output_data[0], 0);
        assert_eq!(output_data[1], 255);
        assert_eq!(output_data[2], 0);
    }

    #[test]
    fn blue_hsv_to_srgb() {
        let converter = HsvToSRgb;
        let input_data: [f32; 3] = [240.0, 1.0, 1.0];
        let mut output_data = [0_u8; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);
        assert_eq!(output_data[0], 0);
        assert_eq!(output_data[1], 0);
        assert_eq!(output_data[2], 255);
    }

    #[test]
    fn achromatic_white_hsv_to_srgb() {
        let converter = HsvToSRgb;
        let input_data: [f32; 3] = [0.0, 0.0, 1.0];
        let mut output_data = [0_u8; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);
        assert_eq!(output_data, [255, 255, 255]);
    }

    /// Round-trip sRGB → HSV → sRGB must be identity within ±1 u8.
    #[test]
    fn round_trip_srgb_hsv_srgb() {
        use super::super::srgb_to_hsv::SRgbToHsv;
        let fwd = SRgbToHsv;
        let inv = HsvToSRgb;

        let originals: &[[u8; 3]] = &[
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [128, 64, 32],
            [255, 255, 255],
            [0, 0, 0],
            [100, 200, 150],
        ];

        for orig in originals {
            let region = Region::new(0, 0, 1, 1);
            let mut hsv_buf = [0.0_f32; 3];
            let input = Tile::new(region, 3, orig.as_slice());
            let mut hsv_tile = TileMut::new(region, 3, &mut hsv_buf);
            fwd.convert_region(&mut (), &input, &mut hsv_tile);

            let mut srgb_buf = [0_u8; 3];
            let hsv_in = Tile::new(region, 3, &hsv_buf);
            let mut srgb_out = TileMut::new(region, 3, &mut srgb_buf);
            inv.convert_region(&mut (), &hsv_in, &mut srgb_out);

            let diff_r = (srgb_buf[0] as i32 - orig[0] as i32).unsigned_abs();
            let diff_g = (srgb_buf[1] as i32 - orig[1] as i32).unsigned_abs();
            let diff_b = (srgb_buf[2] as i32 - orig[2] as i32).unsigned_abs();
            assert!(
                diff_r <= 1,
                "R mismatch: {} vs {} for {:?}",
                srgb_buf[0],
                orig[0],
                orig
            );
            assert!(
                diff_g <= 1,
                "G mismatch: {} vs {} for {:?}",
                srgb_buf[1],
                orig[1],
                orig
            );
            assert!(
                diff_b <= 1,
                "B mismatch: {} vs {} for {:?}",
                srgb_buf[2],
                orig[2],
                orig
            );
        }
    }

    #[test]
    fn demand_hint_is_any() {
        assert_eq!(HsvToSRgb.demand_hint(), DemandHint::Any);
    }

    #[test]
    fn required_input_region_is_identity() {
        let r = Region::new(10, 20, 30, 40);
        assert_eq!(HsvToSRgb.required_input_region(&r), r);
    }

    #[test]
    fn start_returns_unit() {
        // Covers the `fn start(&self) {}` line.
        HsvToSRgb.start();
    }

    /// H=90° → sector i=1 (yellow-green), exercises match arm `1 => (q, v, p)`.
    #[test]
    fn sector1_yellow_green_hsv_to_srgb() {
        let converter = HsvToSRgb;
        // H=90, S=1, V=1 → chartreuse: R=128, G=255, B=0
        let input_data: [f32; 3] = [90.0, 1.0, 1.0];
        let mut output_data = [0_u8; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);
        // H=90 is mid-way between yellow (60) and green (120) → R≈128, G=255, B=0
        assert_eq!(output_data[1], 255);
        assert_eq!(output_data[2], 0);
    }

    /// H=210° → sector i=3 (azure), exercises match arm `3 => (p, q, v)`.
    #[test]
    fn sector3_azure_hsv_to_srgb() {
        let converter = HsvToSRgb;
        // H=210, S=1, V=1 → azure: R=0, G=128, B=255
        let input_data: [f32; 3] = [210.0, 1.0, 1.0];
        let mut output_data = [0_u8; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);
        // B should be max at H=210
        assert_eq!(output_data[2], 255);
        assert_eq!(output_data[0], 0);
    }

    /// H=330° → sector i=5 (magenta-rose), exercises match arm `_ => (v, p, q)`.
    #[test]
    fn sector5_magenta_hsv_to_srgb() {
        let converter = HsvToSRgb;
        // H=330, S=1, V=1 → rose: R=255, G=0, B=128
        let input_data: [f32; 3] = [330.0, 1.0, 1.0];
        let mut output_data = [0_u8; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);
        // R should be max at H=330
        assert_eq!(output_data[0], 255);
        assert_eq!(output_data[1], 0);
    }
}

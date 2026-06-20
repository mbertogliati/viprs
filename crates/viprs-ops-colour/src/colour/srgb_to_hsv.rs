use viprs_core::{
    colorspace::{Hsv, SRgb},
    colour::ColourConvert,
    format::{F32, U8},
    image::{DemandHint, Region, Tile, TileMut},
};

/// Applies the `srgb to hsv` colour transform to image pixels. Use it when a pipeline needs to
/// move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs::domain::crate::colour::srgb_to_hsv::SRgbToHsv;
///
/// let op = SRgbToHsv;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct SRgbToHsv;

/// Convert one sRGB pixel (u8 × 3) to HSV (f32 × 3).
/// H ∈ [0, 360), S ∈ [0, 1], V ∈ [0, 1].
#[inline(always)]
fn srgb_u8_to_hsv_f32(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let r = f32::from(r) / 255.0;
    let g = f32::from(g) / 255.0;
    let b = f32::from(b) / 255.0;

    let cmax = r.max(g).max(b);
    let cmin = r.min(g).min(b);
    let delta = cmax - cmin;

    let v = cmax;
    let s = if cmax == 0.0 { 0.0 } else { delta / cmax };
    let h = if delta == 0.0 {
        0.0
    } else if cmax == r {
        let seg = (g - b) / delta;
        let h = 60.0 * (seg % 6.0);
        // ensure positive; % can be negative in Rust for negative seg
        if h < 0.0 { h + 360.0 } else { h }
    } else if cmax == g {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };

    (h, s, v)
}

impl ColourConvert<SRgb, Hsv> for SRgbToHsv {
    type InputFormat = U8;
    type OutputFormat = F32;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn convert_region(&self, (): &mut (), input: &Tile<U8>, output: &mut TileMut<F32>) {
        for (pixel_in, pixel_out) in input
            .data
            .chunks_exact(3)
            .zip(output.data.chunks_exact_mut(3))
        {
            let (h, s, v) = srgb_u8_to_hsv_f32(pixel_in[0], pixel_in[1], pixel_in[2]);
            pixel_out[0] = h;
            pixel_out[1] = s;
            pixel_out[2] = v;
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
    fn red_to_hsv() {
        let converter = SRgbToHsv;
        let input_data: [u8; 3] = [255, 0, 0];
        let mut output_data = [0.0_f32; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);
        // sRGB(255,0,0) → HSV = (0.0°, 1.0, 1.0)
        let h = output_data[0];
        assert!(h.abs() < 1e-3 || (h - 360.0).abs() < 1e-3, "H={h}");
        assert!((output_data[1] - 1.0).abs() < 1e-3);
        assert!((output_data[2] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn green_to_hsv() {
        let converter = SRgbToHsv;
        let input_data: [u8; 3] = [0, 255, 0];
        let mut output_data = [0.0_f32; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);
        // H=120, S=1, V=1
        let h = output_data[0];
        assert!((h - 120.0).abs() < 1e-3, "H={h}");
        assert!((output_data[1] - 1.0).abs() < 1e-3);
        assert!((output_data[2] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn blue_to_hsv() {
        let converter = SRgbToHsv;
        let input_data: [u8; 3] = [0, 0, 255];
        let mut output_data = [0.0_f32; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);
        // H=240, S=1, V=1
        let h = output_data[0];
        assert!((h - 240.0).abs() < 1e-3, "H={h}");
        assert!((output_data[1] - 1.0).abs() < 1e-3);
        assert!((output_data[2] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn white_to_hsv() {
        let converter = SRgbToHsv;
        let input_data: [u8; 3] = [255, 255, 255];
        let mut output_data = [0.0_f32; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);
        // H=any, S=0, V=1
        assert!(output_data[1].abs() < 1e-3);
        assert!((output_data[2] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn black_to_hsv() {
        let converter = SRgbToHsv;
        let input_data: [u8; 3] = [0, 0, 0];
        let mut output_data = [0.0_f32; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);
        // H=any, S=0, V=0
        assert!(output_data[1].abs() < 1e-3);
        assert!(output_data[2].abs() < 1e-3);
    }

    #[test]
    fn demand_hint_is_any() {
        assert_eq!(SRgbToHsv.demand_hint(), DemandHint::Any);
    }

    #[test]
    fn required_input_region_is_identity() {
        let r = Region::new(10, 20, 30, 40);
        assert_eq!(SRgbToHsv.required_input_region(&r), r);
    }

    #[test]
    fn start_returns_unit() {
        // Covers the `fn start(&self) {}` line — ColourConvert::start must be called
        // before convert_region in pipeline usage.
        SRgbToHsv.start();
    }

    /// sRGB(255, 0, 100): r is max, g < b, so (g-b)/delta is negative.
    /// The `if h < 0.0 { h + 360.0 }` branch fires — H ≈ 336°.
    #[test]
    fn negative_h_correction_branch() {
        let converter = SRgbToHsv;
        // r=255, g=0, b=100: cmax=r, delta=1.0, seg=(0-100/255)/1.0 ≈ -0.392
        // h_raw = 60 * (-0.392 % 6.0) = 60 * -0.392 ≈ -23.5 → corrected to 336.5
        let input_data: [u8; 3] = [255, 0, 100];
        let mut output_data = [0.0_f32; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);
        // H should be in the red-magenta range [300°, 360°)
        let h = output_data[0];
        assert!(
            h > 300.0 && h < 360.0,
            "H={h} expected in (300,360) for rose pixel"
        );
        let s = output_data[1];
        assert!((s - 1.0).abs() < 1e-2, "S should be near 1.0: {s}");
    }
}

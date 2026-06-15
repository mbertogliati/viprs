#![allow(dead_code, clippy::large_stack_arrays)]
// REASON: colour math keeps fixed-size scratch tables for upcoming SIMD wiring and avoids heap traffic.

use std::sync::OnceLock;

const SRGB_U8_RANGE: usize = 256;
const SRGB_U8_LUT_LAST: usize = SRGB_U8_RANGE - 1;
const XYZ_TO_LAB_LUT_SIZE: usize = 100_000;
const XYZ_TO_LAB_LUT_LAST_INDEX: usize = XYZ_TO_LAB_LUT_SIZE - 2;

pub type SrgbDecodeTable = [f32; SRGB_U8_RANGE];
pub type XyzToLabCurveTable = [f32; XYZ_TO_LAB_LUT_SIZE];

#[inline(always)]
pub fn srgb_gamma_decode(x: f32) -> f32 {
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055_f32).powf(2.4)
    }
}

#[inline(always)]
pub fn srgb_gamma_encode(x: f32) -> f32 {
    if x <= 0.003_130_8 {
        12.92 * x
    } else {
        1.055f32.mul_add(x.powf(1.0 / 2.4), -0.055)
    }
}

fn srgb_u8_to_scrgb_lut() -> &'static SrgbDecodeTable {
    static LUT: OnceLock<[f32; SRGB_U8_RANGE]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut lut = [0.0f32; SRGB_U8_RANGE];
        for (index, value) in lut.iter_mut().enumerate() {
            *value = srgb_gamma_decode(index as f32 / SRGB_U8_LUT_LAST as f32);
        }
        lut
    })
}

fn scrgb_to_srgb_u8_lut() -> &'static [u8; SRGB_U8_RANGE + 1] {
    static LUT: OnceLock<[u8; SRGB_U8_RANGE + 1]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut lut = [0u8; SRGB_U8_RANGE + 1];
        for (index, value) in lut.iter_mut().take(SRGB_U8_RANGE).enumerate() {
            let encoded = srgb_gamma_encode(index as f32 / SRGB_U8_LUT_LAST as f32);
            *value = (encoded * SRGB_U8_LUT_LAST as f32).round() as u8;
        }
        lut[SRGB_U8_RANGE] = lut[SRGB_U8_LUT_LAST];
        lut
    })
}

#[inline(always)]
pub fn scrgb_to_srgb_u8_table() -> &'static [u8; SRGB_U8_RANGE + 1] {
    scrgb_to_srgb_u8_lut()
}

fn xyz_to_lab_curve_lut() -> &'static XyzToLabCurveTable {
    static LUT: OnceLock<Box<[f32; XYZ_TO_LAB_LUT_SIZE]>> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut lut = Box::new([0.0f32; XYZ_TO_LAB_LUT_SIZE]);
        for (index, value) in lut.iter_mut().enumerate() {
            let sample = index as f32 / XYZ_TO_LAB_LUT_SIZE as f32;
            *value = if sample < 0.008_856 {
                7.787f32.mul_add(sample, 16.0 / 116.0)
            } else {
                sample.cbrt()
            };
        }
        lut
    })
}

#[inline(always)]
pub fn srgb_decode_u8(sample: u8) -> f32 {
    srgb_u8_to_scrgb_lut()[sample as usize]
}

#[inline(always)]
pub fn srgb_decode_table() -> &'static SrgbDecodeTable {
    srgb_u8_to_scrgb_lut()
}

pub const D65_X0: f32 = 0.950_47;
pub const D65_Y0: f32 = 1.0;
pub const D65_Z0: f32 = 1.088_83;
const XYZ_TO_LAB_X_SCALE: f32 = XYZ_TO_LAB_LUT_SIZE as f32 / D65_X0;
const XYZ_TO_LAB_Y_SCALE: f32 = XYZ_TO_LAB_LUT_SIZE as f32 / D65_Y0;
const XYZ_TO_LAB_Z_SCALE: f32 = XYZ_TO_LAB_LUT_SIZE as f32 / D65_Z0;

#[inline(always)]
pub fn lab_to_xyz_components(lightness: f32, a: f32, b: f32) -> (f32, f32, f32) {
    let (y, cby) = if lightness < 8.0 {
        let y = (lightness * D65_Y0) / 903.3;
        (y, 7.787f32.mul_add(y / D65_Y0, 16.0 / 116.0))
    } else {
        let cby = (lightness + 16.0) / 116.0;
        (D65_Y0 * cby * cby * cby, cby)
    };

    let tmp_x = a / 500.0 + cby;
    let x = if tmp_x < 0.2069 {
        D65_X0 * (tmp_x - 0.13793) / 7.787
    } else {
        D65_X0 * tmp_x * tmp_x * tmp_x
    };

    let tmp_z = cby - b / 200.0;
    let z = if tmp_z < 0.2069 {
        D65_Z0 * (tmp_z - 0.13793) / 7.787
    } else {
        D65_Z0 * tmp_z * tmp_z * tmp_z
    };

    (x, y, z)
}

#[inline(always)]
fn xyz_to_lab_curve(value: f32) -> f32 {
    if value < 0.008_856 {
        7.787f32.mul_add(value, 16.0 / 116.0)
    } else {
        value.cbrt()
    }
}

#[inline(always)]
fn xyz_to_lab_curve_lookup(table: &[f32; XYZ_TO_LAB_LUT_SIZE], scaled: f32) -> f32 {
    let scaled = scaled.clamp(0.0, XYZ_TO_LAB_LUT_LAST_INDEX as f32);
    let index = scaled as usize;
    let fraction = scaled - index as f32;
    let base = table[index];
    fraction.mul_add(table[index + 1] - base, base)
}

#[inline(always)]
pub fn xyz_to_lab_components(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let table = xyz_to_lab_curve_lut();
    xyz_to_lab_components_with_table(table, x, y, z)
}

#[inline(always)]
pub fn xyz_to_lab_table() -> &'static XyzToLabCurveTable {
    xyz_to_lab_curve_lut()
}

#[inline(always)]
pub fn xyz_to_lab_components_with_table(
    table: &XyzToLabCurveTable,
    x: f32,
    y: f32,
    z: f32,
) -> (f32, f32, f32) {
    let cbx = xyz_to_lab_curve_lookup(table, x.max(0.0) * XYZ_TO_LAB_X_SCALE);
    let cby = xyz_to_lab_curve_lookup(table, y.max(0.0) * XYZ_TO_LAB_Y_SCALE);
    let cbz = xyz_to_lab_curve_lookup(table, z.max(0.0) * XYZ_TO_LAB_Z_SCALE);

    (
        116.0f32.mul_add(cby, -16.0),
        500.0 * (cbx - cby),
        200.0 * (cby - cbz),
    )
}

#[inline(always)]
pub fn scrgb_to_xyz_components(red: f32, green: f32, blue: f32) -> (f32, f32, f32) {
    let red = red * D65_Y0;
    let green = green * D65_Y0;
    let blue = blue * D65_Y0;

    (
        red.mul_add(0.4124, blue.mul_add(0.1805, green * 0.3576)),
        red.mul_add(0.2126, blue.mul_add(0.0722, green * 0.7152)),
        red.mul_add(0.0193, blue.mul_add(0.9505, green * 0.1192)),
    )
}

#[inline(always)]
pub fn xyz_to_scrgb_components(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let x = x / D65_Y0;
    let y = y / D65_Y0;
    let z = z / D65_Y0;

    (
        x.mul_add(3.240_625, z.mul_add(-0.498_629, y * -1.537_208)),
        x.mul_add(-0.968_931, z.mul_add(0.041_518, y * 1.875_756)),
        x.mul_add(0.055_71, z.mul_add(1.056_996, y * -0.204_021)),
    )
}

#[inline(always)]
pub fn scrgb_to_srgb_u8_component_with_table(table: &[u8; SRGB_U8_RANGE + 1], sample: f32) -> u8 {
    let sample = sample.clamp(0.0, 1.0) * SRGB_U8_LUT_LAST as f32;
    let index = sample as usize;
    let fraction = sample - index as f32;
    let base = f32::from(table[index]);
    let next = f32::from(table[index + 1]);
    fraction.mul_add(next - base, base).round() as u8
}

#[inline(always)]
fn scrgb_to_srgb_u8_component(sample: f32) -> u8 {
    scrgb_to_srgb_u8_component_with_table(scrgb_to_srgb_u8_lut(), sample)
}

#[inline(always)]
pub fn scrgb_to_srgb_u8_components(red: f32, green: f32, blue: f32) -> [u8; 3] {
    [
        scrgb_to_srgb_u8_component(red),
        scrgb_to_srgb_u8_component(green),
        scrgb_to_srgb_u8_component(blue),
    ]
}

#[inline(always)]
pub fn ab_to_hue_degrees(a: f32, b: f32) -> f32 {
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
                (t + core::f32::consts::TAU).to_degrees()
            } else {
                t.to_degrees()
            }
        } else {
            (t + core::f32::consts::PI).to_degrees()
        }
    }
}

#[inline(always)]
pub fn chroma_hue_to_ab(chroma: f32, hue: f32) -> (f32, f32) {
    let hue_rad = hue.to_radians();
    (chroma * hue_rad.cos(), chroma * hue_rad.sin())
}

#[inline(always)]
pub fn l_to_ucs(lightness: f32) -> f32 {
    if lightness < 16.0 {
        1.744 * lightness
    } else {
        0.3838f32.mul_add(lightness, 21.75 * lightness.ln()) - 38.54
    }
}

#[inline(always)]
pub fn c_to_ucs(chroma: f32) -> f32 {
    let ucs = 10.92f32.mul_add(0.07216f32.mul_add(chroma, 0.638).ln(), 0.162 * chroma) + 4.907;
    ucs.max(0.0)
}

#[inline(always)]
pub fn ch_to_hucs(chroma: f32, hue: f32) -> f32 {
    let (k4, k5, k6, k7, k8) = if hue < 49.1 {
        (133.87_f32, -134.5_f32, -0.924_f32, 1.727_f32, 340.0_f32)
    } else if hue < 110.1 {
        (11.78_f32, -12.7_f32, -0.218_f32, 2.12_f32, 333.0_f32)
    } else if hue < 269.6 {
        (13.87_f32, 10.93_f32, 0.14_f32, 1.0_f32, -83.0_f32)
    } else {
        (0.14_f32, 5.23_f32, 0.17_f32, 1.61_f32, 233.0_f32)
    };

    let p = k7.mul_add(hue, k8).to_radians().cos();
    let p_term = if p == 0.0 { 0.0 } else { p * p.abs().powf(k6) };
    let d = k5.mul_add(p_term, k4);
    let chroma4 = chroma.powi(4);
    let f = (chroma4 / (chroma4 + 1900.0)).sqrt();
    hue + d * f
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_u8_decode_lut_matches_scalar_reference() {
        for sample in [0_u8, 1, 16, 64, 118, 192, 255] {
            let scalar = srgb_gamma_decode(sample as f32 / SRGB_U8_LUT_LAST as f32);
            let lut = srgb_decode_u8(sample);
            assert!(
                (scalar - lut).abs() < 1e-7,
                "sample={sample} scalar={scalar} lut={lut}"
            );
        }
    }

    #[test]
    fn scrgb_encode_u8_lut_matches_scalar_reference() {
        for sample in [0.0_f32, 0.001, 0.018, 0.25, 0.5, 0.75, 1.0] {
            let scalar =
                (srgb_gamma_encode(sample.clamp(0.0, 1.0)) * SRGB_U8_LUT_LAST as f32).round() as u8;
            let lut = scrgb_to_srgb_u8_component(sample);
            assert!(
                (scalar as i16 - lut as i16).unsigned_abs() <= 1,
                "sample={sample} scalar={scalar} lut={lut}"
            );
        }
    }

    #[test]
    fn xyz_to_lab_lut_tracks_scalar_curve_tightly() {
        for sample in [0.0_f32, 0.001, 0.008_856, 0.1, 0.5, 1.0] {
            let scalar = xyz_to_lab_curve(sample);
            let lut = xyz_to_lab_curve_lookup(
                xyz_to_lab_curve_lut(),
                sample * XYZ_TO_LAB_LUT_SIZE as f32,
            );
            assert!(
                (scalar - lut).abs() < 1e-4,
                "sample={sample} scalar={scalar} lut={lut}"
            );
        }
    }
}

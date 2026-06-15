#![allow(clippy::large_stack_arrays, clippy::large_stack_frames)]
// REASON: the conversion keeps fixed-size stack workspaces to stay allocation-free in the pixel path.
#![allow(clippy::needless_range_loop)]
// REASON: index-based loops map directly to packed LAB/HSV band positions.

use crate::{
    domain::colour::ColourConvert,
    domain::{
        colorspace::{Lch, Ucs},
        format::F32,
        image::{DemandHint, Region, Tile, TileMut},
    },
};

use crate::domain::ops::colour::math::{c_to_ucs, ch_to_hucs, l_to_ucs};
use std::sync::OnceLock;

/// Applies the `ucs to lch` colour transform to image pixels. Use it when a pipeline needs to
/// move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::colour::ucs_to_lch::UcsToLch;
///
/// let op = UcsToLch;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct UcsToLch;

struct UcsTables {
    li: [f32; 1001],
    ci: [f32; 3001],
    hi: [[f32; 361]; 101],
}

fn build_ucs_tables() -> UcsTables {
    let mut li = [0.0f32; 1001];
    let mut ll = [0.0f32; 1001];
    for (i, value) in ll.iter_mut().enumerate() {
        *value = l_to_ucs(i as f32 / 10.0);
    }
    for (i, value) in li.iter_mut().enumerate() {
        let target = i as f32 / 10.0;
        let mut j = 0usize;
        while j < 1000 && ll[j] <= target {
            j += 1;
        }
        let base = j.saturating_sub(1);
        let denom = (ll[j] - ll[base]) * 10.0;
        *value = if denom == 0.0 {
            base as f32 / 10.0
        } else {
            base as f32 / 10.0 + (target - ll[base]) / denom
        };
    }

    let mut ci = [0.0f32; 3001];
    let mut cl = [0.0f32; 3001];
    for (i, value) in cl.iter_mut().enumerate() {
        *value = c_to_ucs(i as f32 / 10.0);
    }
    for (i, value) in ci.iter_mut().enumerate() {
        let target = i as f32 / 10.0;
        let mut j = 0usize;
        while j < 3000 && cl[j] <= target {
            j += 1;
        }
        let base = j.saturating_sub(1);
        let denom = (cl[j] - cl[base]) * 10.0;
        *value = if denom == 0.0 {
            base as f32 / 10.0
        } else {
            base as f32 / 10.0 + (target - cl[base]) / denom
        };
    }

    let mut hl = [[0.0f32; 361]; 101];
    for i in 0..361 {
        for j in 0..101 {
            hl[j][i] = ch_to_hucs(j as f32 * 2.0, i as f32);
        }
    }

    let mut hi = [[0.0f32; 361]; 101];
    for j in 0..101 {
        for i in 0..361 {
            let mut k = 1usize;
            while k < 360 && hl[j][k] <= i as f32 {
                k += 1;
            }
            hi[j][i] =
                k.saturating_sub(1) as f32 + (i as f32 - hl[j][k - 1]) / (hl[j][k] - hl[j][k - 1]);
        }
    }

    UcsTables { li, ci, hi }
}

fn tables() -> &'static UcsTables {
    static TABLES: OnceLock<UcsTables> = OnceLock::new();
    TABLES.get_or_init(build_ucs_tables)
}

#[inline(always)]
fn ucs_l_to_l(lightness_ucs: f32) -> f32 {
    let tables = tables();
    let known = (lightness_ucs * 10.0).floor().clamp(0.0, 999.0) as usize;
    (tables.li[known + 1] - tables.li[known]).mul_add(
        lightness_ucs.mul_add(10.0, -(known as f32)),
        tables.li[known],
    )
}

#[inline(always)]
fn ucs_c_to_c(chroma_ucs: f32) -> f32 {
    let tables = tables();
    let known = (chroma_ucs * 10.0).floor().clamp(0.0, 2999.0) as usize;
    (tables.ci[known + 1] - tables.ci[known])
        .mul_add(chroma_ucs.mul_add(10.0, -(known as f32)), tables.ci[known])
}

#[inline(always)]
fn ucs_h_to_h(chroma: f32, hue_ucs: f32) -> f32 {
    let tables = tables();
    let row = (f32::midpoint(chroma, 1.0) as i32).clamp(0, 99) as usize;
    let known = hue_ucs.floor().clamp(0.0, 359.0) as usize;
    let next = (known + 1).min(360);
    (tables.hi[row][next] - tables.hi[row][known])
        .mul_add(hue_ucs - known as f32, tables.hi[row][known])
}

#[inline(always)]
fn ucs_f32_to_lch_f32(lightness_ucs: f32, chroma_ucs: f32, hue_ucs: f32) -> (f32, f32, f32) {
    let chroma = ucs_c_to_c(chroma_ucs);
    (
        ucs_l_to_l(lightness_ucs),
        chroma,
        ucs_h_to_h(chroma, hue_ucs),
    )
}

impl ColourConvert<Ucs, Lch> for UcsToLch {
    type InputFormat = F32;
    type OutputFormat = F32;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {
        let _ = tables();
    }

    #[inline]
    fn convert_region(&self, (): &mut (), input: &Tile<F32>, output: &mut TileMut<F32>) {
        for (pixel_in, pixel_out) in input
            .data
            .chunks_exact(3)
            .zip(output.data.chunks_exact_mut(3))
        {
            let (lightness, chroma, hue) =
                ucs_f32_to_lch_f32(pixel_in[0], pixel_in[1], pixel_in[2]);
            pixel_out[0] = lightness;
            pixel_out[1] = chroma;
            pixel_out[2] = hue;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        colour::ColourConvert,
        image::{Region, Tile, TileMut},
        ops::colour::{LabToLch, LchToLab, LchToUcs},
    };

    fn make_region(pixels: usize) -> Region {
        Region::new(0, 0, pixels as u32, 1)
    }

    #[test]
    fn start_initializes_tables() {
        UcsToLch.start();
        assert!(tables().li[1] >= tables().li[0]);
    }

    #[test]
    fn zero_chroma_round_trip_recovers_zero_hue() {
        let converter = UcsToLch;
        let input_data = [l_to_ucs(40.0), c_to_ucs(0.0), 0.0];
        let mut output_data = [0.0_f32; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);

        assert!((output_data[0] - 40.0).abs() < 0.2);
        assert!(output_data[1].abs() < 0.2);
        assert!(output_data[2].abs() < 1e-6);
    }

    #[test]
    fn hue_wraparound_round_trip_stays_near_360_degrees() {
        let forward = LchToUcs;
        let inverse = UcsToLch;
        let input_data = [61.316_944_f32, 48.550_44, 359.155_67];
        let region = make_region(1);

        let input = Tile::new(region, 3, &input_data);
        let mut ucs_data = [0.0_f32; 3];
        let mut ucs_tile = TileMut::new(region, 3, &mut ucs_data);
        forward.convert_region(&mut (), &input, &mut ucs_tile);

        let ucs_input = Tile::new(region, 3, &ucs_data);
        let mut roundtrip_data = [0.0_f32; 3];
        let mut roundtrip_tile = TileMut::new(region, 3, &mut roundtrip_data);
        inverse.convert_region(&mut (), &ucs_input, &mut roundtrip_tile);

        assert!((roundtrip_data[0] - input_data[0]).abs() < 0.01);
        assert!((roundtrip_data[1] - input_data[1]).abs() < 0.01);
        let hue_delta = (roundtrip_data[2] - input_data[2])
            .abs()
            .min((roundtrip_data[2] - (input_data[2] - 360.0)).abs());
        assert!(hue_delta < 0.01, "hue delta={hue_delta}");
    }

    #[test]
    fn lab_cmc_lab_identity() {
        let to_lch = LabToLch;
        let to_ucs = LchToUcs;
        let to_lch_back = UcsToLch;
        let to_lab = LchToLab;
        let input_data = [61.316_944_f32, 48.548_836, -0.715_900_4];
        let region = make_region(1);

        let lab_input = Tile::new(region, 3, &input_data);
        let mut lch_data = [0.0_f32; 3];
        let mut lch_tile = TileMut::new(region, 3, &mut lch_data);
        to_lch.convert_region(&mut (), &lab_input, &mut lch_tile);

        let lch_input = Tile::new(region, 3, &lch_data);
        let mut ucs_data = [0.0_f32; 3];
        let mut ucs_tile = TileMut::new(region, 3, &mut ucs_data);
        to_ucs.convert_region(&mut (), &lch_input, &mut ucs_tile);

        let ucs_input = Tile::new(region, 3, &ucs_data);
        let mut lch_roundtrip = [0.0_f32; 3];
        let mut lch_roundtrip_tile = TileMut::new(region, 3, &mut lch_roundtrip);
        to_lch_back.convert_region(&mut (), &ucs_input, &mut lch_roundtrip_tile);

        let lch_roundtrip_input = Tile::new(region, 3, &lch_roundtrip);
        let mut lab_roundtrip = [0.0_f32; 3];
        let mut lab_roundtrip_tile = TileMut::new(region, 3, &mut lab_roundtrip);
        to_lab.convert_region(&mut (), &lch_roundtrip_input, &mut lab_roundtrip_tile);

        assert!((lab_roundtrip[0] - input_data[0]).abs() < 0.01);
        assert!((lab_roundtrip[1] - input_data[1]).abs() < 0.01);
        assert!((lab_roundtrip[2] - input_data[2]).abs() < 0.01);
    }
}

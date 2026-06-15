use std::marker::PhantomData;

use bytemuck::Pod;

use crate::{
    domain::op::{NodeSpec, Op},
    domain::{
        format::{BandFormat, F32, I16},
        image::{DemandHint, Region, Tile, TileMut},
        ops::convolution::gauss_blur::sharpen_kernel_1d,
    },
};

use super::common::ToF64;

/// Legacy per-band unsharp mask.
///
/// This keeps the existing single-op API for grayscale and format-agnostic callers.
/// `PipelineBuilder::sharpen` composes the Lab → `LabS` → Lab path for libvips-style
/// colourspace-aware sharpening.
pub struct Sharpen<F: BandFormat> {
    kernel: Vec<f64>,
    radius: usize,
    strength: f32,
    _format: PhantomData<F>,
}

/// LabS-aware sharpen core matching libvips' "sharpen L, preserve a/b" step.
pub struct LabSSharpen {
    kernel: Vec<f64>,
    radius: usize,
    lut: Vec<i32>,
}

/// Represents a sharpen state.
pub struct SharpenState {
    scratch: Vec<f32>,
}

/// Represents a lab s sharpen state.
pub struct LabSSharpenState {
    scratch: Vec<f32>,
}

const LABS_SCALE: f64 = 327.67;
const LABS_OFFSET: i32 = 32_767;

const LIBVIPS_DEFAULT_SIGMA: f32 = 0.5;
const LIBVIPS_DEFAULT_X1: f32 = 2.0;
const LIBVIPS_DEFAULT_Y2: f32 = 10.0;
const LIBVIPS_DEFAULT_Y3: f32 = 20.0;
const LIBVIPS_DEFAULT_M1: f32 = 0.0;
const LIBVIPS_DEFAULT_M2: f32 = 3.0;

impl<F: BandFormat> Sharpen<F>
where
    F::Sample: ToF64 + Pod,
{
    #[must_use]
    /// Creates a new `Sharpen`.
    pub fn new(sigma: f32, strength: f32) -> Self {
        let kernel = sharpen_kernel_1d(sigma);
        let radius = (kernel.len() - 1) / 2;
        Self {
            kernel,
            radius,
            strength,
            _format: PhantomData,
        }
    }

    const fn scratch_len_for_tile(&self, tile_w: u32, tile_h: u32, bands: u32) -> usize {
        let in_h = tile_h as usize + 2 * self.radius;
        in_h * tile_w as usize * bands as usize
    }
}

impl LabSSharpen {
    #[must_use]
    /// Creates a new `LabSSharpen`.
    pub fn new(sigma: f32, x1: f32, y2: f32, y3: f32, m1: f32, m2: f32) -> Self {
        let kernel = sharpen_kernel_1d(sigma);
        let radius = (kernel.len() - 1) / 2;
        let x1 = f64::from(x1);
        let y2 = f64::from(y2);
        let y3 = f64::from(y3);
        let m1 = f64::from(m1);
        let m2 = f64::from(m2);
        Self {
            kernel,
            radius,
            lut: build_sharpen_lut(x1, y2, y3, m1, m2),
        }
    }

    const fn scratch_len_for_tile(&self, tile_w: u32, tile_h: u32) -> usize {
        let in_h = tile_h as usize + 2 * self.radius;
        in_h * tile_w as usize
    }
}

impl Default for LabSSharpen {
    fn default() -> Self {
        Self::new(
            LIBVIPS_DEFAULT_SIGMA,
            LIBVIPS_DEFAULT_X1,
            LIBVIPS_DEFAULT_Y2,
            LIBVIPS_DEFAULT_Y3,
            LIBVIPS_DEFAULT_M1,
            LIBVIPS_DEFAULT_M2,
        )
    }
}

fn build_sharpen_lut(x1: f64, y2: f64, y3: f64, m1: f64, m2: f64) -> Vec<i32> {
    let mut lut = Vec::with_capacity(65_536);
    for i in 0..65_536 {
        let v = (f64::from(i as u32) - f64::from(LABS_OFFSET)) / LABS_SCALE;
        let mut y = if v < -x1 {
            (-x1).mul_add(m1, (v + x1) * m2)
        } else if v < x1 {
            v * m1
        } else {
            x1.mul_add(m1, (v - x1) * m2)
        };

        if y < -y3 {
            y = -y3;
        }
        if y > y2 {
            y = y2;
        }

        lut.push((y * LABS_SCALE).round_ties_even() as i32);
    }
    lut
}

impl<F> Op for Sharpen<F>
where
    F: BandFormat,
    F::Sample: ToF64 + Pod,
{
    type Input = F;
    type Output = F32;
    type State = SharpenState;

    fn demand_hint(&self) -> DemandHint {
        DemandHint::SmallTile
    }

    fn required_input_region(&self, output: &Region) -> Region {
        Region::new(
            output.x - self.radius as i32,
            output.y - self.radius as i32,
            output.width + 2 * self.radius as u32,
            output.height + 2 * self.radius as u32,
        )
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec {
            input_tile_w: tile_w + 2 * self.radius as u32,
            input_tile_h: tile_h + 2 * self.radius as u32,
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    fn start(&self) -> Self::State {
        SharpenState {
            scratch: Vec::new(),
        }
    }

    fn start_with_tile_and_bands(&self, tile_w: u32, tile_h: u32, bands: u32) -> Self::State {
        SharpenState {
            scratch: vec![0.0; self.scratch_len_for_tile(tile_w, tile_h, bands)],
        }
    }

    #[inline]
    fn process_region(&self, state: &mut Self::State, input: &Tile<F>, output: &mut TileMut<F32>) {
        let out_w = output.region.width as usize;
        let out_h = output.region.height as usize;
        let in_w = input.region.width as usize;
        let in_h = input.region.height as usize;
        let bands = input.bands as usize;
        let scratch_len = in_h * out_w * bands;
        if state.scratch.len() < scratch_len {
            state.scratch.resize(scratch_len, 0.0);
        }
        let scratch = &mut state.scratch[..scratch_len];
        let klen = self.kernel.len();

        for iy in 0..in_h {
            for ox in 0..out_w {
                for band in 0..bands {
                    let mut acc = 0.0f64;
                    for k in 0..klen {
                        let ix = ox + k;
                        let input_idx = (iy * in_w + ix) * bands + band;
                        acc = input.data[input_idx].to_f64().mul_add(self.kernel[k], acc);
                    }
                    let scratch_idx = (iy * out_w + ox) * bands + band;
                    scratch[scratch_idx] = acc as f32;
                }
            }
        }

        for oy in 0..out_h {
            for ox in 0..out_w {
                let x = ox + self.radius;
                let y = oy + self.radius;
                for band in 0..bands {
                    let input_idx = (y * in_w + x) * bands + band;
                    let original = input.data[input_idx].to_f64();
                    let mut blurred = 0.0f64;
                    for k in 0..klen {
                        let iy = oy + k;
                        let scratch_idx = (iy * out_w + ox) * bands + band;
                        blurred = f64::from(scratch[scratch_idx]).mul_add(self.kernel[k], blurred);
                    }
                    let sharpened = f64::from(self.strength).mul_add(original - blurred, original);
                    let out_idx = (oy * out_w + ox) * bands + band;
                    output.data[out_idx] = sharpened as f32;
                }
            }
        }
    }
}

impl Op for LabSSharpen {
    type Input = I16;
    type Output = I16;
    type State = LabSSharpenState;

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        Region::new(
            output.x - self.radius as i32,
            output.y - self.radius as i32,
            output.width + 2 * self.radius as u32,
            output.height + 2 * self.radius as u32,
        )
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec {
            input_tile_w: tile_w + 2 * self.radius as u32,
            input_tile_h: tile_h + 2 * self.radius as u32,
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    fn start(&self) -> Self::State {
        LabSSharpenState {
            scratch: Vec::new(),
        }
    }

    fn start_with_tile_and_bands(&self, tile_w: u32, tile_h: u32, _bands: u32) -> Self::State {
        LabSSharpenState {
            scratch: vec![0.0; self.scratch_len_for_tile(tile_w, tile_h)],
        }
    }

    #[inline]
    fn process_region(
        &self,
        state: &mut Self::State,
        input: &Tile<I16>,
        output: &mut TileMut<I16>,
    ) {
        let out_w = output.region.width as usize;
        let out_h = output.region.height as usize;
        let in_w = input.region.width as usize;
        let in_h = input.region.height as usize;
        let bands = input.bands as usize;
        let scratch_len = in_h * out_w;
        if state.scratch.len() < scratch_len {
            state.scratch.resize(scratch_len, 0.0);
        }
        let scratch = &mut state.scratch[..scratch_len];
        let klen = self.kernel.len();

        for iy in 0..in_h {
            for ox in 0..out_w {
                let mut acc = 0.0f64;
                for k in 0..klen {
                    let ix = ox + k;
                    let input_idx = (iy * in_w + ix) * bands;
                    acc = f64::from(input.data[input_idx]).mul_add(self.kernel[k], acc);
                }
                scratch[iy * out_w + ox] = acc as f32;
            }
        }

        for oy in 0..out_h {
            for ox in 0..out_w {
                let x = ox + self.radius;
                let y = oy + self.radius;
                let input_idx = (y * in_w + x) * bands;
                let original = input.data[input_idx];
                let mut blurred = 0.0f64;
                for k in 0..klen {
                    let iy = oy + k;
                    blurred = f64::from(scratch[iy * out_w + ox]).mul_add(self.kernel[k], blurred);
                }
                let blurred = blurred.clamp(0.0, f64::from(i16::MAX));
                let blurred = blurred.round_ties_even() as i32;
                let v1 = i32::from(original);
                let diff = (v1 & 0x7fff) - (blurred & 0x7fff);
                let lut_idx = (diff + 32_768) as usize;
                let sharpened = (v1 + self.lut[lut_idx]).clamp(0, i32::from(i16::MAX)) as i16;
                let out_idx = (oy * out_w + ox) * bands;
                output.data[out_idx] = sharpened;
                output.data[out_idx + 1..out_idx + bands]
                    .copy_from_slice(&input.data[input_idx + 1..input_idx + bands]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        format::{F32, I16},
        image::{Region, Tile, TileMut},
    };
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn zero_strength_is_identity(
            (width, height, input_data) in (1usize..5, 1usize..5).prop_flat_map(|(width, height)| {
                let radius = Sharpen::<F32>::new(0.5, 0.0).radius;
                let len = (width + 2 * radius) * (height + 2 * radius);
                (Just(width), Just(height), prop::collection::vec(-10.0f32..10.0, len))
            })
        ) {
            let op = Sharpen::<F32>::new(0.5, 0.0);
            let radius = op.radius;
            let in_region = Region::new(0, 0, (width + 2 * radius) as u32, (height + 2 * radius) as u32);
            let out_region = Region::new(0, 0, width as u32, height as u32);
            let input = Tile::<F32>::new(in_region, 1, &input_data);
            let mut output_data = vec![0.0f32; width * height];
            let mut output = TileMut::<F32>::new(out_region, 1, &mut output_data);

            let mut state = op.start_with_tile_and_bands(out_region.width, out_region.height, 1);
            op.process_region(&mut state, &input, &mut output);

            for oy in 0..height {
                for ox in 0..width {
                    let in_idx = (oy + radius) * (width + 2 * radius) + (ox + radius);
                    let out_idx = oy * width + ox;
                    prop_assert!((output_data[out_idx] - input_data[in_idx]).abs() < 1e-5);
                }
            }
        }

        #[test]
        fn zero_input_stays_zero(width in 1usize..5, height in 1usize..5) {
            let op = Sharpen::<F32>::new(1.0, 1.5);
            let radius = op.radius;
            let in_region = Region::new(0, 0, (width + 2 * radius) as u32, (height + 2 * radius) as u32);
            let out_region = Region::new(0, 0, width as u32, height as u32);
            let input_data = vec![0.0f32; (width + 2 * radius) * (height + 2 * radius)];
            let input = Tile::<F32>::new(in_region, 1, &input_data);
            let mut output_data = vec![1.0f32; width * height];
            let mut output = TileMut::<F32>::new(out_region, 1, &mut output_data);

            let mut state = op.start_with_tile_and_bands(out_region.width, out_region.height, 1);
            op.process_region(&mut state, &input, &mut output);

            prop_assert!(output_data.iter().all(|value| value.abs() < 1e-6));
        }

        #[test]
        fn labs_sigma_zero_is_identity(
            (width, height, input_data) in (1usize..5, 1usize..5).prop_flat_map(|(width, height)| {
                let pixels = width * height;
                prop::collection::vec((0i16..=i16::MAX, any::<i16>(), any::<i16>()), pixels)
                    .prop_map(move |samples| {
                        let mut data = Vec::with_capacity(pixels * 3);
                        for (lightness, a, b) in samples {
                            data.extend([lightness, a, b]);
                        }
                        (width, height, data)
                    })
            })
        ) {
            let op = LabSSharpen::new(0.0, 2.0, 10.0, 20.0, 0.0, 3.0);
            let region = Region::new(0, 0, width as u32, height as u32);
            let input = Tile::<I16>::new(region, 3, &input_data);
            let mut output_data = vec![0i16; width * height * 3];
            let mut output = TileMut::<I16>::new(region, 3, &mut output_data);

            let mut state = op.start_with_tile_and_bands(region.width, region.height, 3);
            op.process_region(&mut state, &input, &mut output);

            prop_assert_eq!(output_data, input_data);
        }

        #[test]
        fn labs_zero_m1_m2_is_identity(
            (width, height, input_data, radius) in (1usize..5, 1usize..5).prop_flat_map(|(width, height)| {
                let radius = LabSSharpen::new(0.5, 2.0, 10.0, 20.0, 0.0, 0.0).radius;
                let pixels = (width + 2 * radius) * (height + 2 * radius);
                prop::collection::vec((0i16..=i16::MAX, any::<i16>(), any::<i16>()), pixels)
                    .prop_map(move |samples| {
                        let mut data = Vec::with_capacity(pixels * 3);
                        for (lightness, a, b) in samples {
                            data.extend([lightness, a, b]);
                        }
                        (width, height, data, radius)
                    })
            })
        ) {
            let op = LabSSharpen::new(0.5, 2.0, 10.0, 20.0, 0.0, 0.0);
            let in_region = Region::new(
                0,
                0,
                (width + 2 * radius) as u32,
                (height + 2 * radius) as u32,
            );
            let out_region = Region::new(0, 0, width as u32, height as u32);
            let input = Tile::<I16>::new(in_region, 3, &input_data);
            let mut output_data = vec![0i16; width * height * 3];
            let mut output = TileMut::<I16>::new(out_region, 3, &mut output_data);

            let mut state = op.start_with_tile_and_bands(out_region.width, out_region.height, 3);
            op.process_region(&mut state, &input, &mut output);

            for oy in 0..height {
                for ox in 0..width {
                    let in_idx = ((oy + radius) * (width + 2 * radius) + (ox + radius)) * 3;
                    let out_idx = (oy * width + ox) * 3;
                    prop_assert_eq!(
                        &output_data[out_idx..out_idx + 3],
                        &input_data[in_idx..in_idx + 3]
                    );
                }
            }
        }

        #[test]
        fn labs_lut_respects_y2_y3_clamps(
            y2 in 0.0f32..25.0f32,
            y3 in 0.0f32..25.0f32,
            m1 in 0.0f32..5.0f32,
            m2 in 0.0f32..5.0f32,
            diff in -32767i32..32768i32,
        ) {
            let op = LabSSharpen::new(0.5, 2.0, y2, y3, m1, m2);
            let lut_value = op.lut[(diff + 32_768) as usize];
            let max_delta = (f64::from(y2) * LABS_SCALE).round_ties_even() as i32;
            let min_delta = -(f64::from(y3) * LABS_SCALE).round_ties_even() as i32;

            prop_assert!(lut_value <= max_delta);
            prop_assert!(lut_value >= min_delta);
        }
    }

    #[test]
    fn sharpen_metadata_expands_by_kernel_radius() {
        let op = Sharpen::<F32>::new(1.0, 1.5);
        let out_region = Region::new(4, 6, 3, 2);
        let radius = op.radius as i32;
        assert_eq!(
            op.demand_hint(),
            crate::domain::image::DemandHint::SmallTile
        );
        assert_eq!(
            op.required_input_region(&out_region),
            Region::new(
                out_region.x - radius,
                out_region.y - radius,
                out_region.width + 2 * op.radius as u32,
                out_region.height + 2 * op.radius as u32,
            )
        );
        let spec = op.node_spec(3, 2);
        assert_eq!(spec.input_tile_w, 3 + 2 * op.radius as u32);
        assert_eq!(spec.input_tile_h, 2 + 2 * op.radius as u32);
        assert_eq!(spec.output_tile_w, 3);
        assert_eq!(spec.output_tile_h, 2);
    }

    #[test]
    fn sharpen_preserves_constant_field() {
        let op = Sharpen::<F32>::new(1.0, 2.0);
        let radius = op.radius;
        let in_region = Region::new(0, 0, (3 + 2 * radius) as u32, (2 + 2 * radius) as u32);
        let out_region = Region::new(0, 0, 3, 2);
        let input_data = vec![7.0f32; (3 + 2 * radius) * (2 + 2 * radius)];
        let input = Tile::<F32>::new(in_region, 1, &input_data);
        let mut output_data = vec![0.0f32; 6];
        let mut output = TileMut::<F32>::new(out_region, 1, &mut output_data);

        let mut state = op.start_with_tile_and_bands(out_region.width, out_region.height, 1);
        op.process_region(&mut state, &input, &mut output);

        assert!(output_data.iter().all(|value| (*value - 7.0).abs() < 1e-4));
    }

    #[test]
    fn labs_sharpen_only_updates_l_channel() {
        let op = LabSSharpen::new(1.0, 2.0, 10.0, 20.0, 0.0, 3.0);
        let radius = op.radius;
        let in_region = Region::new(0, 0, (3 + 2 * radius) as u32, (2 + 2 * radius) as u32);
        let out_region = Region::new(0, 0, 3, 2);
        let mut input_data = vec![0i16; (3 + 2 * radius) * (2 + 2 * radius) * 3];

        for pixel in input_data.chunks_exact_mut(3) {
            pixel[0] = 16_000;
            pixel[1] = -1_024;
            pixel[2] = 2_048;
        }
        input_data[((radius + 1) * (3 + 2 * radius) + (radius + 1)) * 3] = 20_000;

        let input = Tile::<I16>::new(in_region, 3, &input_data);
        let mut output_data = vec![0i16; 3 * 2 * 3];
        let mut output = TileMut::<I16>::new(out_region, 3, &mut output_data);

        let mut state = op.start_with_tile_and_bands(out_region.width, out_region.height, 3);
        op.process_region(&mut state, &input, &mut output);

        assert!(
            output_data
                .chunks_exact(3)
                .all(|pixel| pixel[1] == -1_024 && pixel[2] == 2_048)
        );
    }

    #[test]
    fn labs_sharpen_prefers_thin_strips_for_pipeline_chains() {
        let op = LabSSharpen::default();
        assert_eq!(
            op.demand_hint(),
            crate::domain::image::DemandHint::ThinStrip
        );
    }

    #[test]
    fn labs_sharpen_clips_l_channel_to_labs_range() {
        let op = LabSSharpen::new(0.5, 2.0, 10.0, 20.0, 0.0, 20.0);
        let in_region = Region::new(0, 0, 3, 3);
        let out_region = Region::new(0, 0, 1, 1);
        let input_data = [
            0,
            -500,
            500,
            0,
            -500,
            500,
            0,
            -500,
            500,
            0,
            -500,
            500,
            i16::MAX,
            -500,
            500,
            0,
            -500,
            500,
            0,
            -500,
            500,
            0,
            -500,
            500,
            0,
            -500,
            500,
        ];
        let input = Tile::<I16>::new(in_region, 3, &input_data);
        let mut output_data = [0i16; 3];
        let mut output = TileMut::<I16>::new(out_region, 3, &mut output_data);

        let mut state = op.start_with_tile_and_bands(out_region.width, out_region.height, 3);
        op.process_region(&mut state, &input, &mut output);

        assert_eq!(output_data[0], i16::MAX);
        assert_eq!(output_data[1], -500);
        assert_eq!(output_data[2], 500);
    }

    #[test]
    fn labs_sharpen_matches_libvips_default_golden_output() {
        let op = LabSSharpen::default();
        let radius = op.radius;
        assert_eq!(radius, 1);

        let in_region = Region::new(0, 0, 5, 5);
        let out_region = Region::new(0, 0, 3, 3);
        let input_data: [i16; 75] = [
            18000, -1200, 900, 18000, -1120, 900, 18000, -1040, 900, 18000, -960, 900, 18000, -880,
            900, 18000, -1200, 830, 18000, -1120, 830, 15000, -1040, 830, 18000, -960, 830, 18000,
            -880, 830, 18000, -1200, 760, 14000, -1120, 760, 28000, -1040, 760, 22000, -960, 760,
            18000, -880, 760, 18000, -1200, 690, 18000, -1120, 690, 21000, -1040, 690, 18000, -960,
            690, 18000, -880, 690, 18000, -1200, 620, 18000, -1120, 620, 18000, -1040, 620, 18000,
            -960, 620, 18000, -880, 620,
        ];
        let expected_data: [i16; 27] = [
            18000, -1120, 830, 10633, -1040, 830, 18000, -960, 830, 8406, -1120, 760, 31277, -1040,
            760, 22272, -960, 760, 18000, -1120, 690, 21000, -1040, 690, 17707, -960, 690,
        ];

        let input = Tile::<I16>::new(in_region, 3, &input_data);
        let mut output_data = [0i16; 27];
        let mut output = TileMut::<I16>::new(out_region, 3, &mut output_data);
        let mut state = op.start_with_tile_and_bands(out_region.width, out_region.height, 3);
        op.process_region(&mut state, &input, &mut output);

        for (actual, expected) in output_data
            .chunks_exact(3)
            .zip(expected_data.chunks_exact(3))
        {
            assert!((actual[0] - expected[0]).abs() <= 3);
            assert_eq!(actual[1], expected[1]);
            assert_eq!(actual[2], expected[2]);
        }
    }

    #[test]
    fn sharpen_start_matches_pre_sized_state_output() {
        let op = Sharpen::<F32>::new(1.0, 1.5);
        let radius = op.radius;
        let in_region = Region::new(0, 0, (4 + 2 * radius) as u32, (3 + 2 * radius) as u32);
        let out_region = Region::new(0, 0, 4, 3);
        let input_data: Vec<f32> = (0..((4 + 2 * radius) * (3 + 2 * radius)))
            .map(|idx| (idx as f32 * 0.5) - 7.0)
            .collect();
        let input = Tile::<F32>::new(in_region, 1, &input_data);

        let mut expected_data = vec![0.0f32; 12];
        let mut expected_output = TileMut::<F32>::new(out_region, 1, &mut expected_data);
        let mut expected_state =
            op.start_with_tile_and_bands(out_region.width, out_region.height, 1);
        op.process_region(&mut expected_state, &input, &mut expected_output);

        let mut actual_data = vec![-999.0f32; 12];
        let mut actual_output = TileMut::<F32>::new(out_region, 1, &mut actual_data);
        let mut actual_state = op.start();
        op.process_region(&mut actual_state, &input, &mut actual_output);

        for (actual, expected) in actual_data.iter().zip(expected_data.iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn labs_sharpen_start_matches_pre_sized_state_output() {
        let op = LabSSharpen::default();
        let radius = op.radius;
        let in_region = Region::new(0, 0, (3 + 2 * radius) as u32, (2 + 2 * radius) as u32);
        let out_region = Region::new(0, 0, 3, 2);
        let mut input_data = Vec::with_capacity((3 + 2 * radius) * (2 + 2 * radius) * 3);
        for idx in 0..((3 + 2 * radius) * (2 + 2 * radius)) {
            let lightness = 12_000 + (idx as i16 * 257);
            let a = -1_500 + idx as i16 * 19;
            let b = 900 - idx as i16 * 11;
            input_data.extend([lightness, a, b]);
        }
        let input = Tile::<I16>::new(in_region, 3, &input_data);

        let mut expected_data = vec![0i16; 18];
        let mut expected_output = TileMut::<I16>::new(out_region, 3, &mut expected_data);
        let mut expected_state =
            op.start_with_tile_and_bands(out_region.width, out_region.height, 3);
        op.process_region(&mut expected_state, &input, &mut expected_output);

        let mut actual_data = vec![-1i16; 18];
        let mut actual_output = TileMut::<I16>::new(out_region, 3, &mut actual_data);
        let mut actual_state = op.start();
        op.process_region(&mut actual_state, &input, &mut actual_output);

        assert_eq!(actual_data, expected_data);
    }
}

use std::marker::PhantomData;

use bytemuck::Pod;

use viprs_core::{
    error::ViprsError,
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    op::{NodeSpec, Op},
};

use super::common::{ConvolutionMask1d, FromF64, ToF64, apply_scale_offset, validate_kernel_1d};

const DEFAULT_LAYERS: usize = 5;
const MAX_LINES: usize = 1000;
const DEFAULT_TILE_SIDE: u32 = 128;
const EPSILON: f64 = 1e-12;

#[derive(Clone, Copy)]
struct Line {
    start: usize,
    end: usize,
    factor: i32,
}

struct LinePlan {
    lines: Box<[Line]>,
    depth_scale: f64,
}

enum ConvaSepKernel {
    Lines(LinePlan),
    Exact(Box<[f64]>),
}

#[derive(Clone, Copy)]
struct SeparableGeometry {
    in_w: usize,
    in_h: usize,
    out_w: usize,
    out_h: usize,
    bands: usize,
}

/// Approximate separable convolution, matching libvips `convasep` shape.
///
/// Integer approximation keeps the input format like libvips `convasep`.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::convolution::convasep::ConvaSepOp;
///
/// let op = ConvaSepOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct ConvaSepOp<F: BandFormat> {
    kernel: ConvaSepKernel,
    radius: usize,
    scale: f64,
    offset: f64,
    _format: PhantomData<F>,
}

/// Represents a conva sep state.
pub struct ConvaSepState {
    scratch: Vec<f32>,
}

impl<F> ConvaSepOp<F>
where
    F: BandFormat,
    F::Sample: ToF64 + Pod,
{
    /// Creates a new `ConvaSepOp`.
    pub fn new(kernel: Vec<f64>) -> Result<Self, ViprsError> {
        Self::with_mask_and_layers(
            ConvolutionMask1d::from_coefficients(kernel)?,
            DEFAULT_LAYERS,
        )
    }

    /// Returns this value configured with layers.
    pub fn with_layers(kernel: Vec<f64>, layers: usize) -> Result<Self, ViprsError> {
        Self::with_mask_and_layers(ConvolutionMask1d::from_coefficients(kernel)?, layers)
    }

    /// Returns this value configured with mask.
    pub fn with_mask(mask: ConvolutionMask1d) -> Result<Self, ViprsError> {
        Self::with_mask_and_layers(mask, DEFAULT_LAYERS)
    }

    /// Returns this value configured with mask and layers.
    pub fn with_mask_and_layers(
        mask: ConvolutionMask1d,
        layers: usize,
    ) -> Result<Self, ViprsError> {
        if !(1..=MAX_LINES).contains(&layers) {
            return Err(ViprsError::Codec(
                "ConvaSepOp: layers must be in 1..=1000".to_owned(),
            ));
        }

        let radius = validate_kernel_1d("ConvaSepOp", mask.coefficients())?;
        let scale = mask.scale();
        let offset = mask.offset();
        let coefficients = mask.into_coefficients();
        let kernel = LinePlan::from_kernel(&coefficients, layers)?.map_or_else(
            || ConvaSepKernel::Exact(coefficients.into_boxed_slice()),
            ConvaSepKernel::Lines,
        );

        Ok(Self {
            kernel,
            radius,
            scale,
            offset,
            _format: PhantomData,
        })
    }

    #[must_use]
    /// Returns or performs radius.
    pub const fn radius(&self) -> usize {
        self.radius
    }
}

impl LinePlan {
    fn from_kernel(kernel: &[f64], layers: usize) -> Result<Option<Self>, ViprsError> {
        if kernel.len() == 1 {
            return Ok(None);
        }

        let max_abs = kernel
            .iter()
            .fold(0.0f64, |acc, value| acc.max(value.abs()));
        if max_abs <= EPSILON {
            return Ok(None);
        }

        let depth = max_abs / layers as f64;
        if depth <= EPSILON {
            return Ok(None);
        }

        let mut quantized = vec![0i32; kernel.len()];
        let mut max_pos = 0i32;
        let mut max_neg = 0i32;
        for (out, value) in quantized.iter_mut().zip(kernel.iter().copied()) {
            let q = (value / depth).round() as i32;
            *out = q;
            max_pos = max_pos.max(q);
            max_neg = max_neg.max(-q);
        }

        if max_pos == 0 && max_neg == 0 {
            return Ok(None);
        }

        let mut lines = Vec::new();
        add_level_lines(&quantized, max_pos, 1, &mut lines)?;
        add_level_lines(&quantized, max_neg, -1, &mut lines)?;
        combine_lines(&mut lines);

        if lines.is_empty() {
            return Ok(None);
        }

        let reconstructed_sum = lines
            .iter()
            .map(|line| f64::from(line.factor) * (line.end - line.start) as f64 * depth)
            .sum::<f64>();
        let original_sum = kernel.iter().sum::<f64>();
        let sum_scale = if original_sum.abs() > EPSILON && reconstructed_sum.abs() > EPSILON {
            original_sum / reconstructed_sum
        } else {
            1.0
        };

        Ok(Some(Self {
            lines: lines.into_boxed_slice(),
            depth_scale: depth * sum_scale,
        }))
    }
}

fn add_level_lines(
    quantized: &[i32],
    max_level: i32,
    sign: i32,
    lines: &mut Vec<Line>,
) -> Result<(), ViprsError> {
    for level in 1..=max_level {
        let mut start = None;
        for (index, value) in quantized.iter().copied().enumerate() {
            let inside = if sign > 0 {
                value >= level
            } else {
                value <= -level
            };

            match (start, inside) {
                (None, true) => start = Some(index),
                (Some(line_start), false) => {
                    push_line(lines, line_start, index, sign)?;
                    start = None;
                }
                _ => {}
            }
        }

        if let Some(line_start) = start {
            push_line(lines, line_start, quantized.len(), sign)?;
        }
    }

    Ok(())
}

fn push_line(
    lines: &mut Vec<Line>,
    start: usize,
    end: usize,
    factor: i32,
) -> Result<(), ViprsError> {
    if lines.len() >= MAX_LINES {
        return Err(ViprsError::Codec(
            "ConvaSepOp: mask too complex for line approximation".to_owned(),
        ));
    }

    lines.push(Line { start, end, factor });
    Ok(())
}

fn combine_lines(lines: &mut Vec<Line>) {
    let mut index = 0;
    while index < lines.len() {
        let mut next = index + 1;
        while next < lines.len() {
            if lines[index].start == lines[next].start && lines[index].end == lines[next].end {
                lines[index].factor += lines[next].factor;
                lines.remove(next);
            } else {
                next += 1;
            }
        }

        if lines[index].factor == 0 {
            lines.remove(index);
        } else {
            index += 1;
        }
    }
}

impl<F> Op for ConvaSepOp<F>
where
    F: BandFormat,
    F::Sample: ToF64 + FromF64 + Pod,
{
    type Input = F;
    type Output = F;
    type State = ConvaSepState;

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
        self.start_with_tile(DEFAULT_TILE_SIDE, DEFAULT_TILE_SIDE)
    }

    fn start_with_tile(&self, tile_w: u32, tile_h: u32) -> Self::State {
        let intermediate_h = tile_h as usize + 2 * self.radius;
        ConvaSepState {
            scratch: vec![0.0f32; tile_w as usize * intermediate_h],
        }
    }

    #[inline]
    fn process_region(&self, state: &mut Self::State, input: &Tile<F>, output: &mut TileMut<F>) {
        let out_w = output.region.width as usize;
        let out_h = output.region.height as usize;
        let in_w = input.region.width as usize;
        let in_h = input.region.height as usize;
        let bands = input.bands as usize;
        let geometry = SeparableGeometry {
            in_w,
            in_h,
            out_w,
            out_h,
            bands,
        };
        let scratch_len = geometry.out_w * geometry.in_h;

        if state.scratch.len() < scratch_len {
            debug_assert!(
                false,
                "ConvaSepOp scratch must be pre-sized with start_with_tile()"
            );
            return;
        }

        for band in 0..bands {
            match &self.kernel {
                ConvaSepKernel::Lines(plan) => {
                    horizontal_lines(
                        input,
                        &mut state.scratch[..scratch_len],
                        plan,
                        band,
                        geometry,
                    );
                    vertical_lines(
                        &state.scratch[..scratch_len],
                        output,
                        plan,
                        band,
                        geometry,
                        self.scale,
                        self.offset,
                    );
                }
                ConvaSepKernel::Exact(kernel) => {
                    horizontal_exact(
                        input,
                        &mut state.scratch[..scratch_len],
                        kernel,
                        band,
                        geometry,
                    );
                    vertical_exact(
                        &state.scratch[..scratch_len],
                        output,
                        kernel,
                        band,
                        geometry,
                        self.scale,
                        self.offset,
                    );
                }
            }
        }
    }
}

#[inline]
fn horizontal_lines<F>(
    input: &Tile<F>,
    scratch: &mut [f32],
    plan: &LinePlan,
    band: usize,
    geometry: SeparableGeometry,
) where
    F: BandFormat,
    F::Sample: ToF64,
{
    for y in 0..geometry.in_h {
        for ox in 0..geometry.out_w {
            let mut acc = 0.0f64;
            for line in plan.lines.iter().copied() {
                let mut line_sum = 0.0f64;
                for kx in line.start..line.end {
                    let idx = (y * geometry.in_w + ox + kx) * geometry.bands + band;
                    line_sum += input.data[idx].to_f64();
                }
                acc = f64::from(line.factor).mul_add(line_sum, acc);
            }
            scratch[y * geometry.out_w + ox] = (acc * plan.depth_scale) as f32;
        }
    }
}

#[inline]
fn vertical_lines<F>(
    scratch: &[f32],
    output: &mut TileMut<F>,
    plan: &LinePlan,
    band: usize,
    geometry: SeparableGeometry,
    scale: f64,
    offset: f64,
) where
    F: BandFormat,
    F::Sample: FromF64,
{
    for oy in 0..geometry.out_h {
        for x in 0..geometry.out_w {
            let mut acc = 0.0f64;
            for line in plan.lines.iter().copied() {
                let mut line_sum = 0.0f64;
                for ky in line.start..line.end {
                    line_sum += f64::from(scratch[(oy + ky) * geometry.out_w + x]);
                }
                acc = f64::from(line.factor).mul_add(line_sum, acc);
            }
            let out_idx = (oy * geometry.out_w + x) * geometry.bands + band;
            output.data[out_idx] =
                F::Sample::from_f64(apply_scale_offset(acc * plan.depth_scale, scale, offset));
        }
    }
}

#[inline]
fn horizontal_exact<F>(
    input: &Tile<F>,
    scratch: &mut [f32],
    kernel: &[f64],
    band: usize,
    geometry: SeparableGeometry,
) where
    F: BandFormat,
    F::Sample: ToF64,
{
    for y in 0..geometry.in_h {
        for ox in 0..geometry.out_w {
            let mut acc = 0.0f64;
            for (kx, weight) in kernel.iter().copied().enumerate() {
                let idx = (y * geometry.in_w + ox + kx) * geometry.bands + band;
                acc = input.data[idx].to_f64().mul_add(weight, acc);
            }
            scratch[y * geometry.out_w + ox] = acc as f32;
        }
    }
}

#[inline]
fn vertical_exact<F>(
    scratch: &[f32],
    output: &mut TileMut<F>,
    kernel: &[f64],
    band: usize,
    geometry: SeparableGeometry,
    scale: f64,
    offset: f64,
) where
    F: BandFormat,
    F::Sample: FromF64,
{
    for oy in 0..geometry.out_h {
        for x in 0..geometry.out_w {
            let mut acc = 0.0f64;
            for (ky, weight) in kernel.iter().copied().enumerate() {
                acc = f64::from(scratch[(oy + ky) * geometry.out_w + x]).mul_add(weight, acc);
            }
            let out_idx = (oy * geometry.out_w + x) * geometry.bands + band;
            output.data[out_idx] = F::Sample::from_f64(apply_scale_offset(acc, scale, offset));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::{F32, U16},
        image::{Region, Tile, TileMut},
    };

    fn run_f32(input: &[f32], in_region: Region, out_region: Region, kernel: Vec<f64>) -> Vec<f32> {
        let op = ConvaSepOp::<F32>::with_layers(kernel, 8).unwrap();
        let mut output = vec![0.0f32; out_region.pixel_count()];
        let input = Tile::<F32>::new(in_region, 1, input);
        let mut output_tile = TileMut::<F32>::new(out_region, 1, &mut output);
        let mut state = op.start_with_tile(out_region.width, out_region.height);
        op.process_region(&mut state, &input, &mut output_tile);
        output
    }

    fn run_f32_mask(
        input: &[f32],
        in_region: Region,
        out_region: Region,
        mask: ConvolutionMask1d,
    ) -> Vec<f32> {
        let op = ConvaSepOp::<F32>::with_mask_and_layers(mask, 8).unwrap();
        let mut output = vec![0.0f32; out_region.pixel_count()];
        let input = Tile::<F32>::new(in_region, 1, input);
        let mut output_tile = TileMut::<F32>::new(out_region, 1, &mut output);
        let mut state = op.start_with_tile(out_region.width, out_region.height);
        op.process_region(&mut state, &input, &mut output_tile);
        output
    }

    #[test]
    fn metadata_expands_by_kernel_radius() {
        let op = ConvaSepOp::<F32>::new(vec![0.25, 0.5, 0.25]).unwrap();
        let output = Region::new(4, 5, 7, 8);
        assert_eq!(op.radius(), 1);
        assert_eq!(op.required_input_region(&output), Region::new(3, 4, 9, 10));
        let spec = op.node_spec(7, 8);
        assert_eq!(spec.input_tile_w, 9);
        assert_eq!(spec.input_tile_h, 10);
        assert_eq!(spec.output_tile_w, 7);
        assert_eq!(spec.output_tile_h, 8);
    }

    #[test]
    fn identity_kernel_preserves_multiband_u16_samples_as_u16() {
        let op = ConvaSepOp::<U16>::new(vec![1.0]).unwrap();
        fn output_is_u16<O: Op<Input = U16, Output = U16>>(_: &O) {}
        output_is_u16(&op);
        let region = Region::new(0, 0, 2, 2);
        let input_data = vec![1u16, 10, 2, 20, 3, 30, 4, 40];
        let mut output = vec![0u16; input_data.len()];
        let input = Tile::<U16>::new(region, 2, &input_data);
        let mut output_tile = TileMut::<U16>::new(region, 2, &mut output);
        let mut state = op.start_with_tile(region.width, region.height);

        op.process_region(&mut state, &input, &mut output_tile);

        assert_eq!(output, input_data);
    }

    proptest! {
        #[test]
        fn identity_kernel_round_trips_samples(samples in prop::collection::vec(-10.0f32..10.0, 1..32)) {
            let region = Region::new(0, 0, samples.len() as u32, 1);
            let output = run_f32(&samples, region, region, vec![1.0]);

            for (actual, expected) in output.iter().zip(samples.iter()) {
                prop_assert!((actual - expected).abs() < 1e-6);
            }
        }

        #[test]
        fn zero_input_stays_zero(width in 1usize..6, height in 1usize..6) {
            let input_region = Region::new(0, 0, (width + 2) as u32, (height + 2) as u32);
            let output_region = Region::new(0, 0, width as u32, height as u32);
            let input = vec![0.0f32; (width + 2) * (height + 2)];
            let output = run_f32(&input, input_region, output_region, vec![0.25, 0.5, 0.25]);

            prop_assert!(output.iter().all(|sample| sample.abs() < 1e-6));
        }
    }

    #[test]
    fn mask_scale_and_offset_apply_after_both_passes() {
        let region = Region::new(0, 0, 3, 1);
        let input = vec![10.0, 20.0, 30.0];
        let mask = ConvolutionMask1d::new(vec![2.0], 4.0, 5.0).unwrap();

        assert_eq!(
            run_f32_mask(&input, region, region, mask),
            vec![15.0, 25.0, 35.0]
        );
    }
}

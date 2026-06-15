use std::marker::PhantomData;

use bytemuck::Pod;

use crate::{
    domain::op::{NodeSpec, Op},
    domain::{
        error::ViprsError,
        format::{BandFormat, F32},
        image::{DemandHint, Region, Tile, TileMut},
    },
};

use super::{
    common::{ConvolutionMask2d, ToF64, validate_kernel_2d},
    conv2d::{Conv2d, ToF64 as ConvToF64},
};

/// Type alias for mask.
pub type Mask = Vec<Vec<f64>>;

const DEFAULT_LAYERS: usize = 5;
const DEFAULT_CLUSTER: usize = 1;
const DEFAULT_TILE_SIDE: u32 = 128;
const MAX_LINES: usize = 1000;
const MAX_EDGES: usize = 1000;
const MAX_HLINES: usize = 150;
const EPSILON: f64 = 1e-12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HLine {
    start: usize,
    end: usize,
    weight: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VElement {
    band: usize,
    row: usize,
    factor: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VLine {
    band: usize,
    factor: i32,
    start: usize,
    end: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Edge {
    a: usize,
    b: usize,
    distance: usize,
}

struct ApproxPlan {
    hlines: Box<[HLine]>,
    vlines: Box<[VLine]>,
    divisor: f32,
    rounding: f32,
    offset: f32,
    radius_x: u32,
    radius_y: u32,
}

enum ConvaMode<F: BandFormat> {
    Approx { plan: ApproxPlan },
    Exact(Conv2d<F>),
}

/// Applies the `conva` convolution-style filter to the image. It evaluates each output pixel
/// from a local neighbourhood of input samples.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::convolution::conva::ConvaOp;
///
/// let op = ConvaOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct ConvaOp<F: BandFormat> {
    mode: ConvaMode<F>,
    _format: PhantomData<F>,
}

/// Enumerates the available conva state values.
pub enum ConvaState {
    /// Uses the `Approx` variant of `ConvaState`.
    Approx {
        /// Stores the `horizontal` value for this item.
        horizontal: Vec<f32>,
        /// Stores the `rolling` value for this item.
        rolling: Vec<f32>,
    },
    /// Uses the `Exact` variant of `ConvaState`.
    Exact(()),
}

impl<F> ConvaOp<F>
where
    F: BandFormat,
    F::Sample: ToF64 + ConvToF64 + Pod,
{
    /// Creates a new `ConvaOp`.
    pub fn new(mask: Mask) -> Result<Self, ViprsError> {
        Self::with_mask(ConvolutionMask2d::from_coefficients(mask)?)
    }

    /// Returns this value configured with mask.
    #[allow(clippy::needless_pass_by_value)]
    // REASON: public API stability for callers that already own the validated mask.
    pub fn with_mask(mask: ConvolutionMask2d) -> Result<Self, ViprsError> {
        let exact = Conv2d::<F>::with_mask(mask.clone())?;
        let mode = ApproxPlan::from_mask(&mask)
            .map_or(ConvaMode::Exact(exact), |plan| ConvaMode::Approx { plan });

        Ok(Self {
            mode,
            _format: PhantomData,
        })
    }

    #[cfg(test)]
    fn is_approximate(&self) -> bool {
        matches!(self.mode, ConvaMode::Approx { .. })
    }

    fn exact_process(&self, input: &Tile<F>, output: &mut TileMut<F32>) {
        match &self.mode {
            ConvaMode::Approx { .. } => {
                debug_assert!(false, "ConvaOp::exact_process must only run in exact mode");
                output.data.fill(0.0);
            }
            ConvaMode::Exact(exact) => {
                let mut state = ();
                exact.process_region(&mut state, input, output);
            }
        }
    }
}

impl ApproxPlan {
    fn from_mask(mask: &ConvolutionMask2d) -> Option<Self> {
        let coefficients = mask.coefficients();
        let (width, height) = validate_kernel_2d("ConvaOp", coefficients).ok()?;
        if width <= 5 || height <= 5 {
            return None;
        }
        let sum_abs = coefficients
            .iter()
            .flatten()
            .map(|value| value.abs())
            .sum::<f64>();
        if sum_abs <= EPSILON {
            return None;
        }

        let (mut hlines, mut velements) = decompose_hlines(coefficients, DEFAULT_LAYERS)?;
        cluster_hlines(&mut hlines, &mut velements, DEFAULT_CLUSTER);
        renumber_hlines(&mut hlines, &mut velements);
        if hlines.is_empty() || hlines.len() > MAX_HLINES {
            return None;
        }

        let mut vlines = build_vlines(&mut velements);
        if vlines.is_empty() {
            return None;
        }

        let mut area = velements
            .iter()
            .map(|element| {
                let hline = hlines[element.band];
                f64::from(element.factor.abs()) * (hline.end - hline.start) as f64
            })
            .sum::<f64>();

        let common = velements
            .iter()
            .map(|element| element.factor.abs())
            .reduce(gcd)
            .unwrap_or(1)
            .max(1);
        if common > 1 {
            for element in &mut velements {
                element.factor /= common;
            }
            for vline in &mut vlines {
                vline.factor /= common;
            }
            area *= f64::from(common);
        }

        let divisor = (area * mask.scale() / sum_abs).round().max(1.0) as f32;
        Some(Self {
            hlines: hlines.into_boxed_slice(),
            vlines: vlines.into_boxed_slice(),
            divisor,
            rounding: ((divisor as i32 + 1) / 2) as f32,
            offset: mask.offset() as f32,
            radius_x: (width / 2) as u32,
            radius_y: (height / 2) as u32,
        })
    }
}

impl<F> Op for ConvaOp<F>
where
    F: BandFormat,
    F::Sample: ToF64 + ConvToF64 + Pod,
{
    type Input = F;
    type Output = F32;
    type State = ConvaState;

    fn demand_hint(&self) -> DemandHint {
        DemandHint::SmallTile
    }

    fn required_input_region(&self, output: &Region) -> Region {
        match &self.mode {
            ConvaMode::Approx { plan } => Region::new(
                output.x - plan.radius_x as i32,
                output.y - plan.radius_y as i32,
                output.width + 2 * plan.radius_x,
                output.height + 2 * plan.radius_y,
            ),
            ConvaMode::Exact(exact) => exact.required_input_region(output),
        }
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        match &self.mode {
            ConvaMode::Approx { plan } => NodeSpec {
                input_tile_w: tile_w + 2 * plan.radius_x,
                input_tile_h: tile_h + 2 * plan.radius_y,
                output_tile_w: tile_w,
                output_tile_h: tile_h,
                coordinate_driven_source: None,
            },
            ConvaMode::Exact(exact) => exact.node_spec(tile_w, tile_h),
        }
    }

    fn start(&self) -> Self::State {
        self.start_with_tile(DEFAULT_TILE_SIDE, DEFAULT_TILE_SIDE)
    }

    fn start_with_tile(&self, tile_w: u32, tile_h: u32) -> Self::State {
        match &self.mode {
            ConvaMode::Approx { plan } => {
                let spec = self.node_spec(tile_w, tile_h);
                let horizontal_len =
                    spec.output_tile_w as usize * spec.input_tile_h as usize * plan.hlines.len();
                ConvaState::Approx {
                    horizontal: vec![0.0; horizontal_len],
                    rolling: vec![0.0; plan.vlines.len()],
                }
            }
            ConvaMode::Exact(_) => ConvaState::Exact(()),
        }
    }

    #[inline]
    fn process_region(&self, state: &mut Self::State, input: &Tile<F>, output: &mut TileMut<F32>) {
        match (&self.mode, state) {
            (ConvaMode::Exact(exact), ConvaState::Exact(exact_state)) => {
                exact.process_region(exact_state, input, output);
            }
            (
                ConvaMode::Approx { plan },
                ConvaState::Approx {
                    horizontal,
                    rolling,
                },
            ) => {
                let in_w = input.region.width as usize;
                let in_h = input.region.height as usize;
                let out_w = output.region.width as usize;
                let out_h = output.region.height as usize;
                let bands = input.bands as usize;
                let hline_count = plan.hlines.len();

                let horizontal_required = out_w * in_h * hline_count;
                assert!(
                    horizontal.len() >= horizontal_required,
                    "ConvaOp horizontal scratch must be pre-sized with start_with_tile() for {}x{} output tiles",
                    output.region.width,
                    output.region.height
                );
                assert!(
                    rolling.len() >= plan.vlines.len(),
                    "ConvaOp rolling scratch must be pre-sized with start_with_tile()"
                );

                for band in 0..bands {
                    horizontal_pass::<F>(
                        input,
                        band,
                        &plan.hlines,
                        &mut horizontal[..horizontal_required],
                        in_w,
                        in_h,
                        out_w,
                    );
                    vertical_pass(
                        &horizontal[..horizontal_required],
                        &plan.hlines,
                        &plan.vlines,
                        &mut rolling[..plan.vlines.len()],
                        output.data,
                        band,
                        bands,
                        out_w,
                        out_h,
                        plan.divisor,
                        plan.rounding,
                        plan.offset,
                    );
                }
            }
            _ => self.exact_process(input, output),
        }
    }
}

fn decompose_hlines(
    mask: &[Vec<f64>],
    layers_requested: usize,
) -> Option<(Vec<HLine>, Vec<VElement>)> {
    let height = mask.len();
    let width = mask.first()?.len();

    let mut max = 0.0f64;
    let mut min = 0.0f64;
    for row in mask {
        for &value in row {
            max = max.max(value);
            min = min.min(value);
        }
    }

    let mut depth = (max - min) / layers_requested as f64;
    if depth <= EPSILON || max <= EPSILON {
        return None;
    }

    let layers_above = (max / depth).ceil() as i32;
    if layers_above <= 0 {
        return None;
    }
    depth = max / f64::from(layers_above);
    if depth <= EPSILON {
        return None;
    }
    let layers_below = (min / depth).floor() as i32;
    let layers = (layers_above - layers_below).clamp(1, MAX_LINES as i32) as usize;

    let mut hlines = Vec::new();
    let mut velements = Vec::new();

    for z in 0..layers {
        let z_ph = (z as f64 + 1.0).mul_add(-depth, max) + depth / 2.0;
        let positive = (z as i32) < layers_above;

        for (row, coefficients) in mask.iter().enumerate().take(height) {
            let mut start = None;
            for (x, &coefficient) in coefficients.iter().enumerate().take(width) {
                let inside = if positive {
                    coefficient >= z_ph
                } else {
                    coefficient <= z_ph
                };
                match (start, inside) {
                    (None, true) => start = Some(x),
                    (Some(line_start), false) => {
                        push_hline(
                            &mut hlines,
                            &mut velements,
                            line_start,
                            x,
                            row,
                            if positive { 1 } else { -1 },
                        )
                        .ok()?;
                        start = None;
                    }
                    _ => {}
                }
            }

            if let Some(line_start) = start {
                push_hline(
                    &mut hlines,
                    &mut velements,
                    line_start,
                    width,
                    row,
                    if positive { 1 } else { -1 },
                )
                .ok()?;
            }
        }
    }

    if hlines.is_empty() {
        return None;
    }

    Some((hlines, velements))
}

fn push_hline(
    hlines: &mut Vec<HLine>,
    velements: &mut Vec<VElement>,
    start: usize,
    end: usize,
    row: usize,
    factor: i32,
) -> Result<(), ()> {
    if hlines.len() >= MAX_LINES || velements.len() >= MAX_LINES {
        return Err(());
    }

    let band = hlines.len();
    hlines.push(HLine {
        start,
        end,
        weight: 1,
    });
    velements.push(VElement { band, row, factor });
    Ok(())
}

fn cluster_hlines(hlines: &mut [HLine], velements: &mut [VElement], threshold: usize) {
    loop {
        let mut edges = collect_edges(hlines);
        if edges.is_empty() {
            break;
        }
        if edges.len() > MAX_EDGES {
            edges.truncate(MAX_EDGES);
        }

        let mut merged = false;
        let mut invalid = vec![false; hlines.len()];
        for edge in edges {
            if edge.distance > threshold || invalid[edge.a] || invalid[edge.b] {
                continue;
            }
            merge_hlines(hlines, velements, edge.a, edge.b);
            invalid[edge.a] = true;
            invalid[edge.b] = true;
            merged = true;
        }

        if !merged {
            break;
        }
    }
}

fn collect_edges(hlines: &[HLine]) -> Vec<Edge> {
    let mut edges = Vec::new();
    for a in 0..hlines.len() {
        if hlines[a].weight <= 0 {
            continue;
        }
        for b in (a + 1)..hlines.len() {
            if hlines[b].weight <= 0 {
                continue;
            }
            edges.push(Edge {
                a,
                b,
                distance: hline_distance(hlines[a], hlines[b]),
            });
        }
    }
    edges.sort_unstable_by_key(|edge| edge.distance);
    edges
}

const fn hline_distance(a: HLine, b: HLine) -> usize {
    a.start.abs_diff(b.start) + a.end.abs_diff(b.end)
}

fn merge_hlines(hlines: &mut [HLine], velements: &mut [VElement], a: usize, b: usize) {
    let fa = hlines[a].weight;
    let fb = hlines[b].weight;
    let total = fa + fb;
    if total <= 0 {
        return;
    }

    let w = f64::from(fb) / f64::from(total);
    hlines[a].start = w.mul_add(
        hlines[b].start as f64 - hlines[a].start as f64,
        hlines[a].start as f64,
    ) as usize;
    hlines[a].end = w.mul_add(
        hlines[b].end as f64 - hlines[a].end as f64,
        hlines[a].end as f64,
    ) as usize;
    hlines[a].weight = total;

    for element in velements {
        if element.band == b {
            element.band = a;
        }
    }

    hlines[b].weight = 0;
}

fn renumber_hlines(hlines: &mut Vec<HLine>, velements: &mut [VElement]) {
    let mut mapping = vec![usize::MAX; hlines.len()];
    let mut compact = Vec::with_capacity(hlines.len());

    for (index, hline) in hlines.iter().copied().enumerate() {
        if hline.weight > 0 {
            mapping[index] = compact.len();
            compact.push(hline);
        }
    }

    for element in velements {
        element.band = mapping[element.band];
    }

    *hlines = compact;
}

fn build_vlines(velements: &mut Vec<VElement>) -> Vec<VLine> {
    velements.sort_unstable_by(|left, right| {
        left.band
            .cmp(&right.band)
            .then(left.factor.cmp(&right.factor))
            .then(left.row.cmp(&right.row))
    });

    let mut compact = Vec::with_capacity(velements.len());
    let mut index = 0;
    while index < velements.len() {
        let current = velements[index];
        let mut factor = current.factor;
        let mut next = index + 1;
        while next < velements.len()
            && velements[next].band == current.band
            && velements[next].row == current.row
        {
            factor += velements[next].factor;
            next += 1;
        }
        if factor != 0 {
            compact.push(VElement {
                band: current.band,
                row: current.row,
                factor,
            });
        }
        index = next;
    }
    *velements = compact;

    velements.sort_unstable_by(|left, right| {
        left.band
            .cmp(&right.band)
            .then(left.factor.cmp(&right.factor))
            .then(left.row.cmp(&right.row))
    });

    let mut vlines = Vec::with_capacity(velements.len());
    let mut row = 0;
    while row < velements.len() {
        let start = velements[row];
        let mut end = row + 1;
        while end < velements.len()
            && velements[end].band == start.band
            && velements[end].factor == start.factor
            && velements[end].row == start.row + (end - row)
        {
            end += 1;
        }
        vlines.push(VLine {
            band: start.band,
            factor: start.factor,
            start: start.row,
            end: velements[end - 1].row + 1,
        });
        row = end;
    }

    vlines
}

fn gcd(a: i32, b: i32) -> i32 {
    if b == 0 { a.abs() } else { gcd(b, a % b) }
}

#[inline(always)]
fn horizontal_pass<F: BandFormat>(
    input: &Tile<F>,
    band: usize,
    hlines: &[HLine],
    horizontal: &mut [f32],
    in_w: usize,
    in_h: usize,
    out_w: usize,
) where
    F::Sample: ToF64,
{
    let bands = input.bands as usize;
    let hline_count = hlines.len();

    for y in 0..in_h {
        for (hline_index, hline) in hlines.iter().copied().enumerate() {
            let base = (y * out_w) * hline_count + hline_index;
            let mut sum = 0.0f32;
            for x in hline.start..hline.end {
                let source_index = (y * in_w + x) * bands + band;
                sum += input.data[source_index].to_f64() as f32;
            }
            horizontal[base] = sum;

            for ox in 1..out_w {
                let add_index = (y * in_w + ox + hline.end - 1) * bands + band;
                let sub_index = (y * in_w + ox + hline.start - 1) * bands + band;
                sum += input.data[add_index].to_f64() as f32;
                sum -= input.data[sub_index].to_f64() as f32;
                horizontal[(y * out_w + ox) * hline_count + hline_index] = sum;
            }
        }
    }
}

#[inline(always)]
fn vertical_pass(
    horizontal: &[f32],
    hlines: &[HLine],
    vlines: &[VLine],
    rolling: &mut [f32],
    output_data: &mut [f32],
    band: usize,
    bands: usize,
    out_w: usize,
    out_h: usize,
    divisor: f32,
    rounding: f32,
    offset: f32,
) {
    let _ = hlines;
    let hline_count = hlines.len();

    for x in 0..out_w {
        let mut total = 0.0f32;
        for (index, vline) in vlines.iter().copied().enumerate() {
            let mut sum = 0.0f32;
            for row in vline.start..vline.end {
                sum += horizontal[(row * out_w + x) * hline_count + vline.band];
            }
            rolling[index] = sum;
            total = (vline.factor as f32).mul_add(sum, total);
        }
        output_data[x * bands + band] = (total + rounding) / divisor + offset;

        for oy in 1..out_h {
            let mut total = 0.0f32;
            for (index, vline) in vlines.iter().copied().enumerate() {
                let add_row = oy + vline.end - 1;
                let sub_row = oy + vline.start - 1;
                rolling[index] += horizontal[(add_row * out_w + x) * hline_count + vline.band];
                rolling[index] -= horizontal[(sub_row * out_w + x) * hline_count + vline.band];
                total = (vline.factor as f32).mul_add(rolling[index], total);
            }
            output_data[(oy * out_w + x) * bands + band] = (total + rounding) / divisor + offset;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        format::F32,
        image::{Region, Tile, TileMut},
        ops::convolution::{conv2d::Conv2d, gauss_blur::gaussian_kernel_1d},
    };
    use proptest::prelude::*;

    fn gaussian_kernel_2d(size: usize) -> Mask {
        let sigma = size as f32 / 6.0;
        let kernel_1d = gaussian_kernel_1d(sigma);
        let actual_size = kernel_1d.len();
        let mut kernel = vec![vec![0.0; actual_size]; actual_size];
        for y in 0..actual_size {
            for x in 0..actual_size {
                kernel[y][x] = kernel_1d[y] * kernel_1d[x];
            }
        }
        kernel
    }

    fn identity_kernel() -> Mask {
        vec![vec![1.0]]
    }

    fn extract_region(full: &[f32], full_width: usize, region: Region) -> Vec<f32> {
        let mut extracted = vec![0.0f32; region.pixel_count()];
        for y in 0..region.height as usize {
            for x in 0..region.width as usize {
                let src_x = region.x as usize + x;
                let src_y = region.y as usize + y;
                extracted[y * region.width as usize + x] = full[src_y * full_width + src_x];
            }
        }
        extracted
    }

    fn run_op<O: Op<Input = F32, Output = F32, State = S>, S>(
        op: &O,
        input_region: Region,
        output_region: Region,
        input_data: &[f32],
    ) -> Vec<f32> {
        let input = Tile::<F32>::new(input_region, 1, input_data);
        let mut output_data = vec![0.0f32; output_region.pixel_count()];
        let mut output = TileMut::<F32>::new(output_region, 1, &mut output_data);
        let mut state = op.start_with_tile(output_region.width, output_region.height);
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    fn centered_output_region(
        image_side: usize,
        output_side: u32,
        conva: &ConvaOp<F32>,
        exact: &Conv2d<F32>,
    ) -> Region {
        let probe = Region::new(0, 0, output_side, output_side);
        let conva_input_region = conva.required_input_region(&probe);
        let exact_input_region = exact.required_input_region(&probe);
        let halo_x =
            ((conva_input_region.width.max(exact_input_region.width) - output_side) / 2) as usize;
        let halo_y =
            ((conva_input_region.height.max(exact_input_region.height) - output_side) / 2) as usize;
        let output_side = output_side as usize;

        let x = halo_x + (image_side - output_side - 2 * halo_x) / 2;
        let y = halo_y + (image_side - output_side - 2 * halo_y) / 2;

        Region::new(x as i32, y as i32, output_side as u32, output_side as u32)
    }

    fn mean_and_max_error(left: &[f32], right: &[f32]) -> (f32, f32) {
        let mut total = 0.0f32;
        let mut max_error = 0.0f32;
        for (lhs, rhs) in left.iter().zip(right.iter()) {
            let error = (lhs - rhs).abs();
            total += error;
            max_error = max_error.max(error);
        }
        (total / left.len() as f32, max_error)
    }

    #[test]
    fn non_separable_kernel_falls_back_to_exact_convolution() {
        let kernel = vec![
            vec![0.0, 1.0, 0.0],
            vec![1.0, 2.0, 1.0],
            vec![0.0, 0.0, 1.0],
        ];
        let conva = ConvaOp::<F32>::new(kernel.clone()).unwrap();
        let exact = Conv2d::<F32>::new(kernel).unwrap();
        assert!(!conva.is_approximate());
        let output_region = Region::new(4, 4, 3, 3);
        let conva_input_region = conva.required_input_region(&output_region);
        let exact_input_region = exact.required_input_region(&output_region);
        let full = vec![2.0f32; 16 * 16];
        let conva_input = extract_region(&full, 16, conva_input_region);
        let exact_input = extract_region(&full, 16, exact_input_region);

        let conva_result = run_op(&conva, conva_input_region, output_region, &conva_input);
        let exact_result = run_op(&exact, exact_input_region, output_region, &exact_input);

        assert_eq!(conva_result, exact_result);
    }

    #[test]
    fn gaussian_kernel_uses_approximate_box_decomposition() {
        let conva = ConvaOp::<F32>::new(gaussian_kernel_2d(21)).unwrap();
        assert!(conva.is_approximate());
    }

    #[test]
    fn gaussian_kernel_stays_close_to_exact_convolution() {
        let kernel = gaussian_kernel_2d(21);
        let conva = ConvaOp::<F32>::new(kernel.clone()).unwrap();
        let exact = Conv2d::<F32>::new(kernel).unwrap();
        let full_width = 128usize;
        let full_height = 128usize;
        let output_region = centered_output_region(full_width, 32, &conva, &exact);
        let full = (0..(full_width * full_height))
            .map(|index| ((index * 37) % 251) as f32)
            .collect::<Vec<_>>();
        let conva_input_region = conva.required_input_region(&output_region);
        let exact_input_region = exact.required_input_region(&output_region);

        let conva_input = extract_region(&full, full_width, conva_input_region);
        let exact_input = extract_region(&full, full_width, exact_input_region);
        let conva_result = run_op(&conva, conva_input_region, output_region, &conva_input);
        let exact_result = run_op(&exact, exact_input_region, output_region, &exact_input);
        let (mean_error, max_error) = mean_and_max_error(&conva_result, &exact_result);

        assert!(
            mean_error <= 4.0 && max_error <= 12.0,
            "expected conva close to exact convolution: mean_error={mean_error}, max_error={max_error}"
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]
        #[test]
        fn identity_kernel_matches_input(
            pixels in proptest::collection::vec(0.0f32..=255.0f32, 24 * 24)
        ) {
            let full_width = 24usize;
            let output_region = Region::new(10, 10, 4, 4);
            let conva = ConvaOp::<F32>::new(identity_kernel()).unwrap();
            let input_region = conva.required_input_region(&output_region);
            let input_data = extract_region(&pixels, full_width, input_region);
            let output = run_op(&conva, input_region, output_region, &input_data);
            let expected = extract_region(&pixels, full_width, output_region);

            for (got, want) in output.iter().zip(expected.iter()) {
                prop_assert!((got - want).abs() <= 1e-5, "expected {want}, got {got}");
            }
        }

        #[test]
        fn gaussian_error_stays_bounded_vs_conv2d(
            pixels in prop_oneof![
                proptest::collection::vec(0.0f32..=255.0f32, 64 * 64),
                proptest::collection::vec(0.0f32..=255.0f32, 128 * 128),
            ]
        ) {
            let kernel = gaussian_kernel_2d(21);
            let conva = ConvaOp::<F32>::new(kernel.clone()).unwrap();
            let exact = Conv2d::<F32>::new(kernel).unwrap();
            let full_width = match pixels.len() {
                4_096 => 64usize,
                16_384 => 128usize,
                other => panic!("unexpected generated image size: {other}"),
            };
            let output_region = centered_output_region(full_width, 8, &conva, &exact);
            let conva_input_region = conva.required_input_region(&output_region);
            let exact_input_region = exact.required_input_region(&output_region);

            let conva_input = extract_region(&pixels, full_width, conva_input_region);
            let exact_input = extract_region(&pixels, full_width, exact_input_region);
            let conva_result = run_op(&conva, conva_input_region, output_region, &conva_input);
            let exact_result = run_op(&exact, exact_input_region, output_region, &exact_input);
            let (mean_error, max_error) = mean_and_max_error(&conva_result, &exact_result);

            prop_assert!(
                mean_error <= 4.0 && max_error <= 12.0,
                "expected conva close to exact convolution: mean_error={mean_error}, max_error={max_error}"
            );
        }
    }
}

#![allow(clippy::while_float)]
// REASON: the stepping loop mirrors libvips' floating-point convergence logic exactly.

use viprs_core::{
    format::{F32, U8},
    image::{DemandHint, Region, Tile, TileMut},
    op::Op,
};

/// Represents a nearest state.
pub struct NearestState {
    line_input: Vec<f32>,
    line_output: Vec<f32>,
    lower_envelope_sites: Vec<usize>,
    lower_envelope_breaks: Vec<f32>,
}

/// Applies the `nearest` morphological operation to the image. Use it for neighbourhood-based
/// shape filtering and mask analysis.
///
/// # Examples
/// ```ignore
/// use viprs_ops_spatial::morphology::nearest::NearestOp;
///
/// let op = NearestOp { /* operation parameters */ };
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct NearestOp {
    width: u32,
    height: u32,
}

impl NearestOp {
    #[must_use]
    /// Creates a new `NearestOp`.
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

impl Op for NearestOp {
    type Input = U8;
    type Output = F32;
    type State = NearestState;

    const OUTPUT_BANDS: Option<usize> = Some(1);

    fn demand_hint(&self) -> DemandHint {
        // The squared-distance transform needs the whole image: each column pass depends on
        // the horizontal transform from every row. We therefore keep FullImage and size the
        // line scratch once from the constructor-provided image dimensions.
        DemandHint::FullImage
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) -> Self::State {
        let max_dim = self.width.max(self.height) as usize;
        NearestState {
            line_input: vec![0.0; max_dim],
            line_output: vec![0.0; max_dim],
            lower_envelope_sites: vec![0; max_dim],
            lower_envelope_breaks: vec![0.0; max_dim + 1],
        }
    }

    #[inline]
    fn process_region(&self, state: &mut Self::State, input: &Tile<U8>, output: &mut TileMut<F32>) {
        let width = input.region.width as usize;
        let height = input.region.height as usize;
        let bands = input.bands as usize;
        let inf = max_squared_distance(width, height);

        debug_assert_eq!(width, self.width as usize);
        debug_assert_eq!(height, self.height as usize);
        debug_assert!(width <= state.line_input.len());
        debug_assert!(height <= state.line_input.len());

        for y in 0..height {
            for x in 0..width {
                let pixel = (y * width + x) * bands;
                let active = input.data[pixel..pixel + bands]
                    .iter()
                    .any(|&sample| sample != 0);
                state.line_input[x] = if active { 0.0 } else { inf };
            }

            squared_distance_transform_1d(
                &state.line_input[..width],
                &mut state.line_output[..width],
                &mut state.lower_envelope_sites[..width],
                &mut state.lower_envelope_breaks[..=width],
                inf,
            );

            let row_start = y * width;
            output.data[row_start..row_start + width].copy_from_slice(&state.line_output[..width]);
        }

        for x in 0..width {
            for y in 0..height {
                state.line_input[y] = output.data[y * width + x];
            }

            squared_distance_transform_1d(
                &state.line_input[..height],
                &mut state.line_output[..height],
                &mut state.lower_envelope_sites[..height],
                &mut state.lower_envelope_breaks[..=height],
                inf,
            );

            for y in 0..height {
                let value = state.line_output[y];
                output.data[y * width + x] = if value >= inf {
                    f32::INFINITY
                } else {
                    value.sqrt()
                };
            }
        }
    }
}

fn max_squared_distance(width: usize, height: usize) -> f32 {
    let max_dx = width.saturating_sub(1) as f32;
    let max_dy = height.saturating_sub(1) as f32;
    max_dx * max_dx + max_dy * max_dy + 1.0
}

fn squared_distance_transform_1d(
    input: &[f32],
    output: &mut [f32],
    lower_envelope_sites: &mut [usize],
    lower_envelope_breaks: &mut [f32],
    inf: f32,
) {
    let len = input.len();
    if len == 0 {
        return;
    }

    let mut envelope_size = 0usize;
    lower_envelope_sites[0] = 0;
    lower_envelope_breaks[0] = f32::NEG_INFINITY;
    lower_envelope_breaks[1] = f32::INFINITY;

    for q in 1..len {
        let mut breakpoint = intersection(q, lower_envelope_sites[envelope_size], input, inf);

        while envelope_size > 0 && breakpoint <= lower_envelope_breaks[envelope_size] {
            envelope_size -= 1;
            breakpoint = intersection(q, lower_envelope_sites[envelope_size], input, inf);
        }

        envelope_size += 1;
        lower_envelope_sites[envelope_size] = q;
        lower_envelope_breaks[envelope_size] = breakpoint;
        lower_envelope_breaks[envelope_size + 1] = f32::INFINITY;
    }

    let mut envelope_idx = 0usize;
    for (q, slot) in output.iter_mut().enumerate() {
        while lower_envelope_breaks[envelope_idx + 1] < q as f32 {
            envelope_idx += 1;
        }

        let site = lower_envelope_sites[envelope_idx];
        let dx = q as f32 - site as f32;
        *slot = dx.mul_add(dx, input[site]);
    }
}

fn intersection(q: usize, candidate: usize, input: &[f32], inf: f32) -> f32 {
    let q_value = input[q];
    let candidate_value = input[candidate];

    if q_value >= inf && candidate_value >= inf {
        return f32::INFINITY;
    }
    if q_value >= inf {
        return f32::INFINITY;
    }
    if candidate_value >= inf {
        return f32::NEG_INFINITY;
    }

    ((q_value + (q * q) as f32) - (candidate_value + (candidate * candidate) as f32))
        / (2.0 * (q as f32 - candidate as f32))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn run_op(width: u32, height: u32, data: &[u8]) -> Vec<f32> {
        let op = NearestOp::new(width, height);
        let region = Region::new(0, 0, width, height);
        let input = Tile::<U8>::new(region, 1, data);
        let mut output_data = vec![0.0f32; width as usize * height as usize];
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    fn reference_nearest(width: u32, height: u32, data: &[u8]) -> Vec<f32> {
        let width = width as usize;
        let height = height as usize;
        let mut seeds = Vec::new();
        let mut output = vec![0.0f32; width * height];

        for y in 0..height {
            for x in 0..width {
                if data[y * width + x] != 0 {
                    seeds.push((x as i32, y as i32));
                }
            }
        }

        if seeds.is_empty() {
            output.fill(f32::INFINITY);
            return output;
        }

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                if data[idx] != 0 {
                    output[idx] = 0.0;
                    continue;
                }

                output[idx] = seeds
                    .iter()
                    .map(|&(seed_x, seed_y)| {
                        let dx = seed_x - x as i32;
                        let dy = seed_y - y as i32;
                        (dx * dx + dy * dy) as f32
                    })
                    .fold(f32::INFINITY, f32::min)
                    .sqrt();
            }
        }

        output
    }

    #[test]
    fn nearest_single_pixel_cases_match_expectations() {
        assert_eq!(run_op(1, 1, &[1]), vec![0.0]);
        assert!(run_op(1, 1, &[0])[0].is_infinite());
    }

    #[test]
    fn nearest_from_corner_seed_matches_euclidean_distance() {
        let output = run_op(3, 3, &[1, 0, 0, 0, 0, 0, 0, 0, 0]);

        assert_eq!(output[0], 0.0);
        assert!((output[1] - 1.0).abs() < f32::EPSILON);
        assert!((output[4] - 2.0_f32.sqrt()).abs() < 1e-6);
        assert!((output[8] - 8.0_f32.sqrt()).abs() < 1e-6);
        assert!(output[1] < output[4]);
        assert!(output[4] < output[8]);
    }

    #[test]
    fn nearest_without_foreground_returns_infinity() {
        let output = run_op(2, 2, &[0, 0, 0, 0]);
        assert!(output.iter().all(|value| value.is_infinite()));
    }

    #[test]
    fn nearest_multiple_regions_match_reference() {
        let data = [
            0, 1, 0, 0, //
            0, 0, 0, 1, //
            1, 0, 0, 0, //
        ];

        let output = run_op(4, 3, &data);
        let expected = reference_nearest(4, 3, &data);

        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn nearest_row_without_seeds_uses_vertical_pass() {
        let output = run_op(3, 3, &[0, 0, 0, 1, 0, 0, 0, 0, 0]);

        assert!((output[0] - 1.0).abs() < 1e-6);
        assert!((output[1] - 2.0_f32.sqrt()).abs() < 1e-6);
        assert!((output[2] - 5.0_f32.sqrt()).abs() < 1e-6);
    }

    proptest! {
        #[test]
        fn nearest_random_inputs_match_reference_and_do_not_panic(
            width in 1u32..=8,
            height in 1u32..=8,
            data in proptest::collection::vec(any::<u8>(), 1..=64),
        ) {
            let pixel_count = width as usize * height as usize;
            prop_assume!(pixel_count <= data.len());
            let input = &data[..pixel_count];

            let output = run_op(width, height, input);
            let expected = reference_nearest(width, height, input);

            for (actual, expected) in output.iter().zip(expected.iter()) {
                if expected.is_infinite() {
                    prop_assert!(actual.is_infinite());
                } else {
                    prop_assert!((actual - expected).abs() < 1e-6);
                }
            }
        }
    }
}

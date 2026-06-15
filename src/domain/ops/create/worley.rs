#![allow(clippy::struct_field_names)]
// REASON: Worley configuration field names follow the algorithm's published terminology.

use crate::domain::{
    error::ViprsError,
    format::F32,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
    ops::resample::sample_conv::FromF64,
};

const MAX_FEATURES: usize = 10;

#[derive(Clone, Copy, Debug)]
struct Cell {
    feature_count: u8,
    feature_x: [i32; MAX_FEATURES],
    feature_y: [i32; MAX_FEATURES],
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            feature_count: 0,
            feature_x: [0; MAX_FEATURES],
            feature_y: [0; MAX_FEATURES],
        }
    }
}

/// Generate deterministic Worley / cellular noise using precomputed feature points.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::create::worley::WorleyOp;
///
/// let op = WorleyOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug)]
pub struct WorleyOp {
    width: u32,
    height: u32,
    cell_size: u32,
    cells_across: u32,
    cells_down: u32,
    cells: Box<[Cell]>,
}

impl WorleyOp {
    /// Associated constant for default cell size.
    pub const DEFAULT_CELL_SIZE: u32 = 256;

    /// Creates a new `WorleyOp`.
    pub fn new(width: u32, height: u32, cell_size: u32, seed: u32) -> Result<Self, ViprsError> {
        if width == 0 || height == 0 {
            return Err(ViprsError::Scheduler(format!(
                "WorleyOp width and height must be > 0, got {width}x{height}"
            )));
        }
        if cell_size == 0 {
            return Err(ViprsError::Scheduler(
                "WorleyOp cell_size must be > 0".to_owned(),
            ));
        }

        let cells_across = width.div_ceil(cell_size);
        let cells_down = height.div_ceil(cell_size);
        let mut cells = Vec::with_capacity(cells_across as usize * cells_down as usize);

        for cell_y in 0..cells_down {
            for cell_x in 0..cells_across {
                cells.push(build_cell(
                    cell_x as i32,
                    cell_y as i32,
                    cells_across as i32,
                    cells_down as i32,
                    cell_size as i32,
                    seed,
                ));
            }
        }

        Ok(Self {
            width,
            height,
            cell_size,
            cells_across,
            cells_down,
            cells: cells.into_boxed_slice(),
        })
    }

    #[must_use]
    /// Returns or performs width.
    pub const fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    /// Returns or performs height.
    pub const fn height(&self) -> u32 {
        self.height
    }

    #[inline(always)]
    fn cell_at(&self, cell_x: i32, cell_y: i32) -> &Cell {
        let wrapped_x = cell_x.rem_euclid(self.cells_across as i32) as u32;
        let wrapped_y = cell_y.rem_euclid(self.cells_down as i32) as u32;
        &self.cells[(wrapped_y * self.cells_across + wrapped_x) as usize]
    }
}

#[inline(always)]
fn vips_random_add(mut hash: u32, value: i32) -> u32 {
    for shift in [0, 8, 16, 24] {
        hash = (hash ^ ((value >> shift) as u32 & 0xff)).wrapping_mul(16_777_619);
    }
    hash
}

#[inline(always)]
fn vips_random(seed: u32) -> u32 {
    vips_random_add(2_166_136_261, seed as i32)
}

fn build_cell(
    cell_x: i32,
    cell_y: i32,
    cells_across: i32,
    cells_down: i32,
    cell_size: i32,
    seed: u32,
) -> Cell {
    let wrapped_x = cell_x.rem_euclid(cells_across);
    let wrapped_y = cell_y.rem_euclid(cells_down);
    let mut mixed = vips_random_add(seed, wrapped_x);
    mixed = vips_random_add(mixed, wrapped_y);

    let mut cell = Cell {
        feature_count: (mixed % (MAX_FEATURES as u32 - 1) + 1) as u8,
        ..Cell::default()
    };

    for feature_index in 0..usize::from(cell.feature_count) {
        mixed = vips_random(mixed);
        cell.feature_x[feature_index] = cell_x * cell_size + (mixed % cell_size as u32) as i32;
        mixed = vips_random(mixed);
        cell.feature_y[feature_index] = cell_y * cell_size + (mixed % cell_size as u32) as i32;
    }

    cell
}

impl Op for WorleyOp {
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
    fn process_region(&self, _state: &mut (), _input: &Tile<F32>, output: &mut TileMut<F32>) {
        debug_assert_eq!(output.bands, 1, "WorleyOp output must be single-band");
        debug_assert!(output.region.x >= 0 && output.region.y >= 0);
        debug_assert!(output.region.x as u32 + output.region.width <= self.width);
        debug_assert!(output.region.y as u32 + output.region.height <= self.height);

        let region_width = output.region.width as usize;
        let max_distance = self.cell_size as f32 * 1.5;

        for row in 0..output.region.height as usize {
            let y = output.region.y + row as i32;

            for col in 0..region_width {
                let x = output.region.x + col as i32;
                let cell_x = x / self.cell_size as i32;
                let cell_y = y / self.cell_size as i32;
                let mut nearest = max_distance;

                for neighbor_y in cell_y - 1..=cell_y + 1 {
                    for neighbor_x in cell_x - 1..=cell_x + 1 {
                        let cell = self.cell_at(neighbor_x, neighbor_y);
                        for feature_index in 0..usize::from(cell.feature_count) {
                            let dx = x - cell.feature_x[feature_index];
                            let dy = y - cell.feature_y[feature_index];
                            let distance = ((dx * dx + dy * dy) as f32).sqrt();
                            nearest = nearest.min(distance);
                        }
                    }
                }

                output.data[row * region_width + col] =
                    f32::from_f64((f64::from(nearest) / f64::from(max_distance)).clamp(0.0, 1.0));
            }
        }
    }
}

impl PixelLocalOp for WorleyOp {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::image::{Region, Tile, TileMut};
    use proptest::prelude::*;

    fn render_region(op: &WorleyOp, region: Region) -> Vec<f32> {
        let input_data = vec![0.0f32; region.pixel_count()];
        let mut output_data = vec![0.0f32; region.pixel_count()];
        let input = Tile::<F32>::new(region, 1, &input_data);
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);
        output_data
    }

    fn render_full(op: &WorleyOp) -> Vec<f32> {
        render_region(op, Region::new(0, 0, op.width(), op.height()))
    }

    #[test]
    fn constructor_rejects_invalid_geometry() {
        assert!(WorleyOp::new(0, 8, 4, 1).is_err());
        assert!(WorleyOp::new(8, 0, 4, 1).is_err());
        assert!(WorleyOp::new(8, 8, 0, 1).is_err());
    }

    #[test]
    fn output_is_deterministic_for_same_seed() {
        let first = WorleyOp::new(32, 16, 8, 17).unwrap();
        let second = WorleyOp::new(32, 16, 8, 17).unwrap();
        assert_eq!(render_full(&first), render_full(&second));
    }

    #[test]
    fn partial_tiles_match_full_render() {
        let op = WorleyOp::new(24, 24, 6, 13).unwrap();
        let full = render_full(&op);
        let region = Region::new(7, 5, 8, 6);
        let partial = render_region(&op, region);

        for row in 0..region.height as usize {
            for col in 0..region.width as usize {
                let full_index =
                    (row + region.y as usize) * op.width() as usize + col + region.x as usize;
                let partial_index = row * region.width as usize + col;
                assert_eq!(partial[partial_index], full[full_index]);
            }
        }
    }

    proptest! {
        #[test]
        fn prop_output_stays_in_unit_interval(
            width in 1u32..=32,
            height in 1u32..=32,
            cell_size in 1u32..=16,
            seed in any::<u32>(),
        ) {
            let op = WorleyOp::new(width, height, cell_size, seed).unwrap();
            let output = render_full(&op);
            prop_assert_eq!(output.len(), width as usize * height as usize);
            prop_assert!(output.iter().all(|sample| sample.is_finite()));
            prop_assert!(output.iter().all(|sample| (0.0..=1.0).contains(sample)));
        }
    }
}

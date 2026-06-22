#![allow(clippy::items_after_statements)]
// REASON: local helper declarations stay close to the solver logic they support.

use viprs_core::{
    error::ViprsError,
    format::BandFormat,
    image::{Region, Tile},
    reducer::TileReducer,
    shared_ops::sample_conv::ToF64,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Represents a tile overlap.
pub struct TileOverlap {
    /// Stores the `lhs` value for this item.
    pub lhs: usize,
    /// Stores the `rhs` value for this item.
    pub rhs: usize,
    /// Stores the `region` value for this item.
    pub region: Region,
}

#[derive(Debug, Clone, PartialEq)]
/// Represents a global balance solution.
pub struct GlobalBalanceSolution {
    /// Stores the `gains` value for this item.
    pub gains: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Represents an overlap measurement.
pub struct OverlapMeasurement {
    overlap_index: usize,
    tile_index: usize,
    mean: f64,
    count: usize,
}

/// Represents a global balance reducer.
pub struct GlobalBalanceReducer {
    tile_regions: Box<[Region]>,
    overlaps: Box<[TileOverlap]>,
    bands: usize,
    gamma: f64,
    max_iterations: usize,
    tolerance: f64,
}

impl GlobalBalanceReducer {
    /// Creates a new `GlobalBalanceReducer`.
    pub fn new(
        tile_regions: Vec<Region>,
        overlaps: Vec<TileOverlap>,
        bands: u32,
    ) -> Result<Self, ViprsError> {
        if tile_regions.is_empty() {
            return Err(ViprsError::Scheduler(
                "GlobalBalanceReducer requires at least one tile".into(),
            ));
        }
        if bands == 0 {
            return Err(ViprsError::Scheduler(
                "GlobalBalanceReducer requires at least one band".into(),
            ));
        }
        for overlap in &overlaps {
            if overlap.lhs >= tile_regions.len() || overlap.rhs >= tile_regions.len() {
                return Err(ViprsError::Scheduler(format!(
                    "GlobalBalanceReducer overlap references tile {} or {} but there are only {} tiles",
                    overlap.lhs,
                    overlap.rhs,
                    tile_regions.len()
                )));
            }
            if intersect_regions(tile_regions[overlap.lhs], overlap.region).is_none()
                || intersect_regions(tile_regions[overlap.rhs], overlap.region).is_none()
            {
                return Err(ViprsError::Scheduler(
                    "GlobalBalanceReducer overlap must intersect both tiles".into(),
                ));
            }
        }

        Ok(Self {
            tile_regions: tile_regions.into_boxed_slice(),
            overlaps: overlaps.into_boxed_slice(),
            bands: bands as usize,
            gamma: 1.0,
            max_iterations: 64,
            tolerance: 1e-9,
        })
    }

    /// Returns this value configured with gamma.
    pub fn with_gamma(mut self, gamma: f64) -> Result<Self, ViprsError> {
        if !gamma.is_finite() || gamma <= 0.0 {
            return Err(ViprsError::Scheduler(
                "GlobalBalanceReducer gamma must be finite and > 0".into(),
            ));
        }
        self.gamma = gamma;
        Ok(self)
    }

    /// Returns this value configured with solver.
    pub fn with_solver(
        mut self,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<Self, ViprsError> {
        if max_iterations == 0 {
            return Err(ViprsError::Scheduler(
                "GlobalBalanceReducer solver needs at least one iteration".into(),
            ));
        }
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(ViprsError::Scheduler(
                "GlobalBalanceReducer tolerance must be finite and > 0".into(),
            ));
        }
        self.max_iterations = max_iterations;
        self.tolerance = tolerance;
        Ok(self)
    }

    fn tile_index_for_region(&self, region: &Region) -> Option<usize> {
        self.tile_regions
            .iter()
            .position(|candidate| candidate == region)
    }

    fn mean_overlap<F>(&self, tile: &Tile<F>, overlap: Region) -> Option<(f64, usize)>
    where
        F: BandFormat,
        F::Sample: ToF64,
    {
        if overlap.is_empty() {
            return None;
        }

        let local_x0 = usize::try_from(overlap.x - tile.region.x).ok()?;
        let local_y0 = usize::try_from(overlap.y - tile.region.y).ok()?;
        let width = tile.region.width as usize;
        let overlap_width = overlap.width as usize;
        let overlap_height = overlap.height as usize;
        let mut sum = 0.0f64;
        let mut count = 0usize;

        for row in 0..overlap_height {
            let row_start = ((local_y0 + row) * width + local_x0) * self.bands;
            let row_end = row_start + overlap_width * self.bands;
            for sample in &tile.data[row_start..row_end] {
                sum += sample.to_f64();
                count += 1;
            }
        }

        if count == 0 {
            None
        } else {
            Some((sum / count as f64, count))
        }
    }

    fn gamma_corrected_mean(&self, mean: f64) -> f64 {
        if (self.gamma - 1.0).abs() < f64::EPSILON {
            mean
        } else {
            mean.powf(1.0 / self.gamma)
        }
    }

    fn solve_log_gains(&self, ata: &[f64], atb: &[f64]) -> Vec<f64> {
        let dim = atb.len();
        let mut solution = vec![0.0f64; dim];

        for _ in 0..self.max_iterations {
            let mut max_delta = 0.0f64;
            for row in 0..dim {
                let diagonal = ata[row * dim + row];
                if diagonal <= 1e-12 {
                    continue;
                }

                let mut rhs = atb[row];
                for col in 0..dim {
                    if col != row {
                        rhs = ata[row * dim + col].mul_add(-solution[col], rhs);
                    }
                }

                let next = rhs / diagonal;
                max_delta = max_delta.max((next - solution[row]).abs());
                solution[row] = next;
            }

            if max_delta <= self.tolerance {
                break;
            }
        }

        solution
    }

    fn tile_measurement_capacity(&self) -> usize {
        self.overlaps.len().min(4)
    }

    fn partial_capacity(&self) -> usize {
        self.overlaps.len().saturating_mul(2)
    }
}

impl<F> TileReducer<F> for GlobalBalanceReducer
where
    F: BandFormat,
    F::Sample: ToF64,
{
    type Partial = Vec<OverlapMeasurement>;
    type Output = GlobalBalanceSolution;
    /// Per-thread reusable measurement buffer for one tile. The thread-local partial is
    /// also pre-sized once, so both staging and accumulation avoid per-tile heap churn.
    type Scratch = Vec<OverlapMeasurement>;

    fn reduce_tile(&self, tile: &Tile<F>, region: &Region) -> Self::Partial {
        debug_assert_eq!(tile.bands as usize, self.bands);
        let Some(tile_index) = self.tile_index_for_region(region) else {
            debug_assert!(
                false,
                "GlobalBalanceReducer received an unknown tile region: {region:?}"
            );
            return Vec::new();
        };

        let mut measurements = Vec::with_capacity(self.tile_measurement_capacity());
        for (overlap_index, overlap) in self.overlaps.iter().copied().enumerate() {
            if overlap.lhs != tile_index && overlap.rhs != tile_index {
                continue;
            }
            let Some(shared) = intersect_regions(*region, overlap.region) else {
                continue;
            };
            let Some((mean, count)) = self.mean_overlap(tile, shared) else {
                continue;
            };
            if mean <= 0.0 {
                continue;
            }
            measurements.push(OverlapMeasurement {
                overlap_index,
                tile_index,
                mean,
                count,
            });
        }

        measurements
    }

    fn accumulate_into(
        &self,
        tile: &Tile<F>,
        region: &Region,
        scratch: &mut Self::Scratch,
        partial: &mut Option<Self::Partial>,
    ) {
        debug_assert_eq!(tile.bands as usize, self.bands);
        let Some(tile_index) = self.tile_index_for_region(region) else {
            debug_assert!(
                false,
                "GlobalBalanceReducer received an unknown tile region: {region:?}"
            );
            return;
        };

        if scratch.capacity() < self.tile_measurement_capacity() {
            scratch.reserve(self.tile_measurement_capacity() - scratch.capacity());
        }
        scratch.clear();

        for (overlap_index, overlap) in self.overlaps.iter().copied().enumerate() {
            if overlap.lhs != tile_index && overlap.rhs != tile_index {
                continue;
            }
            let Some(shared) = intersect_regions(*region, overlap.region) else {
                continue;
            };
            let Some((mean, count)) = self.mean_overlap(tile, shared) else {
                continue;
            };
            if mean <= 0.0 {
                continue;
            }
            scratch.push(OverlapMeasurement {
                overlap_index,
                tile_index,
                mean,
                count,
            });
        }

        let accumulated =
            partial.get_or_insert_with(|| Vec::with_capacity(self.partial_capacity()));
        accumulated.extend_from_slice(scratch);
    }

    fn combine(&self, mut a: Self::Partial, mut b: Self::Partial) -> Self::Partial {
        a.append(&mut b);
        a
    }

    fn finalize(&self, combined: Self::Partial) -> Self::Output {
        if self.tile_regions.len() == 1 {
            return GlobalBalanceSolution { gains: vec![1.0] };
        }

        #[derive(Clone, Copy, Default)]
        struct PairStats {
            lhs_mean: Option<f64>,
            rhs_mean: Option<f64>,
            weight: f64,
        }

        let mut pair_stats = vec![PairStats::default(); self.overlaps.len()];
        for measurement in combined {
            let overlap = self.overlaps[measurement.overlap_index];
            let stats = &mut pair_stats[measurement.overlap_index];
            stats.weight = stats.weight.max(measurement.count as f64);
            if measurement.tile_index == overlap.lhs {
                stats.lhs_mean = Some(measurement.mean);
            } else if measurement.tile_index == overlap.rhs {
                stats.rhs_mean = Some(measurement.mean);
            }
        }

        let dim = self.tile_regions.len() - 1;
        let mut ata = vec![0.0f64; dim * dim];
        let mut atb = vec![0.0f64; dim];
        let mut equations = 0usize;

        for (stats, overlap) in pair_stats.iter().zip(self.overlaps.iter().copied()) {
            let (Some(lhs_mean), Some(rhs_mean)) = (stats.lhs_mean, stats.rhs_mean) else {
                continue;
            };
            if lhs_mean <= 0.0 || rhs_mean <= 0.0 {
                continue;
            }

            let lhs_linear = self.gamma_corrected_mean(lhs_mean);
            let rhs_linear = self.gamma_corrected_mean(rhs_mean);
            let b = lhs_linear.ln() - rhs_linear.ln();
            let weight = stats.weight.max(1.0);
            let mut entries = [(0usize, 0.0f64); 2];
            let mut count = 0usize;

            if overlap.lhs != 0 {
                entries[count] = (overlap.lhs - 1, -1.0);
                count += 1;
            }
            if overlap.rhs != 0 {
                entries[count] = (overlap.rhs - 1, 1.0);
                count += 1;
            }
            if count == 0 {
                continue;
            }

            for i in 0..count {
                let (row, coeff_row) = entries[i];
                atb[row] = (weight * coeff_row).mul_add(b, atb[row]);
                for (col, coeff_col) in entries.iter().take(count).copied() {
                    ata[row * dim + col] =
                        (weight * coeff_row).mul_add(coeff_col, ata[row * dim + col]);
                }
            }
            equations += 1;
        }

        if equations == 0 {
            return GlobalBalanceSolution {
                gains: vec![1.0; self.tile_regions.len()],
            };
        }

        for diagonal in 0..dim {
            ata[diagonal * dim + diagonal] += 1e-9;
        }

        let mut logs = vec![0.0f64; self.tile_regions.len()];
        let solved = self.solve_log_gains(&ata, &atb);
        logs[1..].copy_from_slice(&solved);

        let mean_log = logs.iter().sum::<f64>() / logs.len() as f64;
        for value in &mut logs {
            *value -= mean_log;
        }

        GlobalBalanceSolution {
            gains: logs.into_iter().map(f64::exp).collect(),
        }
    }
}

fn intersect_regions(lhs: Region, rhs: Region) -> Option<Region> {
    let x0 = lhs.x.max(rhs.x);
    let y0 = lhs.y.max(rhs.y);
    let x1 = (lhs.x + lhs.width as i32).min(rhs.x + rhs.width as i32);
    let y1 = (lhs.y + lhs.height as i32).min(rhs.y + rhs.height as i32);
    if x1 <= x0 || y1 <= y0 {
        None
    } else {
        Some(Region::new(x0, y0, (x1 - x0) as u32, (y1 - y0) as u32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{format::U8, reducer::TileReducer};

    fn simple_reducer() -> GlobalBalanceReducer {
        let left = Region::new(0, 0, 4, 4);
        let right = Region::new(2, 0, 4, 4);
        GlobalBalanceReducer::new(
            vec![left, right],
            vec![TileOverlap {
                lhs: 0,
                rhs: 1,
                region: Region::new(2, 0, 2, 4),
            }],
            1,
        )
        .unwrap()
    }

    #[test]
    fn solves_two_tile_overlap_gains() {
        let reducer = simple_reducer();
        let left_region = Region::new(0, 0, 4, 4);
        let right_region = Region::new(2, 0, 4, 4);
        let left_data = vec![10u8; 16];
        let right_data = vec![20u8; 16];
        let left_tile = Tile::<U8>::new(left_region, 1, &left_data);
        let right_tile = Tile::<U8>::new(right_region, 1, &right_data);

        let combined = <GlobalBalanceReducer as TileReducer<U8>>::combine(
            &reducer,
            reducer.reduce_tile(&left_tile, &left_region),
            reducer.reduce_tile(&right_tile, &right_region),
        );
        let solution = <GlobalBalanceReducer as TileReducer<U8>>::finalize(&reducer, combined);

        assert_eq!(solution.gains.len(), 2);
        let balanced_left = solution.gains[0] * 10.0;
        let balanced_right = solution.gains[1] * 20.0;
        assert!((balanced_left - balanced_right).abs() < 1e-6);
        assert!(solution.gains[0].mul_add(solution.gains[1], -1.0).abs() < 1e-6);
    }

    #[test]
    fn identical_tiles_keep_unit_balance() {
        let reducer = simple_reducer();
        let left_region = Region::new(0, 0, 4, 4);
        let right_region = Region::new(2, 0, 4, 4);
        let left_data = vec![17u8; 16];
        let right_data = vec![17u8; 16];
        let left_tile = Tile::<U8>::new(left_region, 1, &left_data);
        let right_tile = Tile::<U8>::new(right_region, 1, &right_data);

        let combined = <GlobalBalanceReducer as TileReducer<U8>>::combine(
            &reducer,
            reducer.reduce_tile(&left_tile, &left_region),
            reducer.reduce_tile(&right_tile, &right_region),
        );
        let solution = <GlobalBalanceReducer as TileReducer<U8>>::finalize(&reducer, combined);

        assert!(solution.gains.iter().all(|gain| (gain - 1.0).abs() < 1e-6));
    }

    #[test]
    fn accumulate_into_reuses_scratch_and_matches_reduce_tile_path() {
        let reducer = simple_reducer();
        let left_region = Region::new(0, 0, 4, 4);
        let right_region = Region::new(2, 0, 4, 4);
        let left_data = vec![10u8; 16];
        let right_data = vec![20u8; 16];
        let left_tile = Tile::<U8>::new(left_region, 1, &left_data);
        let right_tile = Tile::<U8>::new(right_region, 1, &right_data);

        let expected = <GlobalBalanceReducer as TileReducer<U8>>::combine(
            &reducer,
            reducer.reduce_tile(&left_tile, &left_region),
            reducer.reduce_tile(&right_tile, &right_region),
        );

        let mut partial = None;
        let mut scratch = Vec::with_capacity(reducer.tile_measurement_capacity());
        let scratch_ptr = scratch.as_ptr();
        reducer.accumulate_into(&left_tile, &left_region, &mut scratch, &mut partial);
        reducer.accumulate_into(&right_tile, &right_region, &mut scratch, &mut partial);

        assert_eq!(scratch_ptr, scratch.as_ptr());
        assert_eq!(
            partial.expect("partial should exist after accumulate_into"),
            expected
        );
    }

    #[test]
    fn constructor_and_solver_reject_invalid_inputs() {
        let region = Region::new(0, 0, 4, 4);
        let overlap = TileOverlap {
            lhs: 0,
            rhs: 1,
            region,
        };

        assert!(GlobalBalanceReducer::new(Vec::new(), Vec::new(), 1).is_err());
        assert!(GlobalBalanceReducer::new(vec![region], Vec::new(), 0).is_err());
        assert!(GlobalBalanceReducer::new(vec![region], vec![overlap], 1).is_err());
        assert!(
            GlobalBalanceReducer::new(
                vec![region, Region::new(8, 0, 4, 4)],
                vec![TileOverlap {
                    lhs: 0,
                    rhs: 1,
                    region,
                }],
                1,
            )
            .is_err()
        );
        assert!(simple_reducer().with_gamma(0.0).is_err());
        assert!(simple_reducer().with_gamma(f64::NAN).is_err());
        assert!(simple_reducer().with_solver(0, 1e-6).is_err());
        assert!(simple_reducer().with_solver(8, f64::INFINITY).is_err());
    }

    #[test]
    fn finalize_handles_single_tile_and_missing_measurements() {
        let single =
            GlobalBalanceReducer::new(vec![Region::new(0, 0, 2, 2)], Vec::new(), 1).unwrap();
        let single_solution =
            <GlobalBalanceReducer as TileReducer<U8>>::finalize(&single, Vec::new());
        assert_eq!(single_solution.gains, vec![1.0]);

        let missing_measurements =
            <GlobalBalanceReducer as TileReducer<U8>>::finalize(&simple_reducer(), Vec::new());
        assert_eq!(missing_measurements.gains, vec![1.0, 1.0]);
    }

    #[test]
    fn reduce_tile_ignores_non_positive_means() {
        let reducer = simple_reducer();
        let region = Region::new(0, 0, 4, 4);
        let zero_data = vec![0u8; 16];
        let tile = Tile::<U8>::new(region, 1, &zero_data);

        assert!(reducer.reduce_tile(&tile, &region).is_empty());
    }

    #[test]
    fn gamma_balancing_uses_gamma_corrected_overlap_means() {
        let reducer = simple_reducer().with_gamma(2.0).unwrap();
        let left_region = Region::new(0, 0, 4, 4);
        let right_region = Region::new(2, 0, 4, 4);
        let left_data = vec![16u8; 16];
        let right_data = vec![81u8; 16];
        let left_tile = Tile::<U8>::new(left_region, 1, &left_data);
        let right_tile = Tile::<U8>::new(right_region, 1, &right_data);
        let combined = <GlobalBalanceReducer as TileReducer<U8>>::combine(
            &reducer,
            reducer.reduce_tile(&left_tile, &left_region),
            reducer.reduce_tile(&right_tile, &right_region),
        );

        let solution = <GlobalBalanceReducer as TileReducer<U8>>::finalize(&reducer, combined);

        assert!(
            solution.gains[1]
                .mul_add(-9.0, solution.gains[0] * 4.0)
                .abs()
                < 1e-6
        );
    }

    proptest! {
        #[test]
        fn equal_overlap_means_produce_unit_gains(value in 1u8..=255) {
            let reducer = simple_reducer();
            let left_region = Region::new(0, 0, 4, 4);
            let right_region = Region::new(2, 0, 4, 4);
            let left_data = vec![value; 16];
            let right_data = vec![value; 16];
            let left_tile = Tile::<U8>::new(left_region, 1, &left_data);
            let right_tile = Tile::<U8>::new(right_region, 1, &right_data);

            let combined = <GlobalBalanceReducer as TileReducer<U8>>::combine(
                &reducer,
                reducer.reduce_tile(&left_tile, &left_region),
                reducer.reduce_tile(&right_tile, &right_region),
            );
            let solution = <GlobalBalanceReducer as TileReducer<U8>>::finalize(&reducer, combined);

            prop_assert!(solution.gains.iter().all(|gain| (gain - 1.0).abs() < 1e-6));
        }
    }
}

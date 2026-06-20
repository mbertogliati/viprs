use std::marker::PhantomData;

use viprs_core::{
    error::ViprsError,
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
    shared_ops::sample_conv::FromF64,
};

#[derive(Clone, Copy, Debug)]
struct ControlPoint {
    x: f64,
    y: f64,
}

/// Build a one-band LUT by linearly interpolating between control points.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::create::buildlut::BuildlutOp;
///
/// let op = BuildlutOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Clone, Debug)]
pub struct BuildlutOp<F: BandFormat> {
    points: Vec<ControlPoint>,
    output_size: usize,
    _format: PhantomData<F>,
}

impl<F: BandFormat> BuildlutOp<F> {
    /// Creates a new `BuildlutOp`.
    pub fn new(points: Vec<(f64, f64)>, output_size: usize) -> Result<Self, ViprsError> {
        if output_size == 0 {
            return Err(ViprsError::Scheduler(
                "BuildlutOp output_size must be > 0".to_owned(),
            ));
        }
        if points.len() < 2 {
            return Err(ViprsError::Scheduler(
                "BuildlutOp requires at least two control points".to_owned(),
            ));
        }

        let mut normalized = Vec::with_capacity(points.len());
        for (x, y) in points {
            if !x.is_finite() || !y.is_finite() {
                return Err(ViprsError::Scheduler(
                    "BuildlutOp control points must be finite".to_owned(),
                ));
            }
            normalized.push(ControlPoint { x, y });
        }
        normalized.sort_by(|left, right| left.x.total_cmp(&right.x));

        for pair in normalized.windows(2) {
            if (pair[1].x - pair[0].x).abs() <= f64::EPSILON {
                return Err(ViprsError::Scheduler(
                    "BuildlutOp control point x values must be unique".to_owned(),
                ));
            }
        }

        Ok(Self {
            points: normalized,
            output_size,
            _format: PhantomData,
        })
    }

    #[must_use]
    /// Returns or performs width.
    pub const fn width(&self) -> u32 {
        self.output_size as u32
    }

    #[must_use]
    /// Returns or performs height.
    pub const fn height(&self) -> u32 {
        1
    }

    #[inline]
    fn sample_at(&self, x: f64) -> f64 {
        let upper = self.points.partition_point(|point| point.x <= x);
        if upper == 0 {
            return self.points[0].y;
        }
        if upper == self.points.len() {
            return self.points[self.points.len() - 1].y;
        }

        let left = self.points[upper - 1];
        let right = self.points[upper];
        let t = (x - left.x) / (right.x - left.x);
        t.mul_add(right.y - left.y, left.y)
    }
}

impl<F> Op for BuildlutOp<F>
where
    F: BandFormat,
    F::Sample: FromF64,
{
    type Input = F;
    type Output = F;
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
    fn process_region(&self, _state: &mut (), _input: &Tile<F>, output: &mut TileMut<F>) {
        debug_assert_eq!(output.bands, 1, "BuildlutOp output must be single-band");
        debug_assert!(output.region.x >= 0 && output.region.y >= 0);
        debug_assert!(output.region.x as u32 + output.region.width <= self.width());
        debug_assert!(output.region.y as u32 + output.region.height <= self.height());

        let region_width = output.region.width as usize;
        for row in 0..output.region.height as usize {
            for col in 0..region_width {
                let x = f64::from(output.region.x as u32 + col as u32);
                let value = self.sample_at(x);
                output.data[row * region_width + col] = F::Sample::from_f64(value);
            }
        }
    }
}

impl<F> PixelLocalOp for BuildlutOp<F>
where
    F: BandFormat,
    F::Sample: FromF64,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::F32,
        image::{Region, Tile, TileMut},
    };

    fn render(op: &BuildlutOp<F32>) -> Vec<f32> {
        let region = Region::new(0, 0, op.width(), op.height());
        let input_data = vec![0.0f32; region.pixel_count()];
        let mut output_data = vec![0.0f32; region.pixel_count()];
        let input = Tile::<F32>::new(region, 1, &input_data);
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);
        output_data
    }

    #[test]
    fn constructor_rejects_invalid_parameters() {
        assert!(BuildlutOp::<F32>::new(vec![], 4).is_err());
        assert!(BuildlutOp::<F32>::new(vec![(0.0, 0.0)], 4).is_err());
        assert!(BuildlutOp::<F32>::new(vec![(0.0, 0.0), (0.0, 1.0)], 4).is_err());
        assert!(BuildlutOp::<F32>::new(vec![(0.0, f64::NAN), (1.0, 1.0)], 4).is_err());
        assert!(BuildlutOp::<F32>::new(vec![(0.0, 0.0), (1.0, 1.0)], 0).is_err());
    }

    #[test]
    fn linearly_interpolates_between_two_points() {
        let op = BuildlutOp::<F32>::new(vec![(0.0, 0.0), (3.0, 30.0)], 4).unwrap();
        assert_eq!(render(&op), vec![0.0, 10.0, 20.0, 30.0]);
    }

    #[test]
    fn unsorted_points_are_sorted_before_interpolation() {
        let op = BuildlutOp::<F32>::new(vec![(3.0, 30.0), (0.0, 0.0), (1.0, 10.0)], 4).unwrap();
        assert_eq!(render(&op), vec![0.0, 10.0, 20.0, 30.0]);
    }

    #[test]
    fn values_before_and_after_control_point_range_are_clamped() {
        let op = BuildlutOp::<F32>::new(vec![(2.0, 10.0), (4.0, 20.0)], 6).unwrap();
        assert_eq!(render(&op), vec![10.0, 10.0, 10.0, 15.0, 20.0, 20.0]);
    }

    proptest! {
        #[test]
        fn prop_output_has_expected_dimensions_and_range(
            points in proptest::collection::vec((0u8..64, -100.0f64..=100.0), 2..=8),
        ) {
            let mut sorted = points
                .into_iter()
                .map(|(x, y)| (f64::from(x), y))
                .collect::<Vec<_>>();
            sorted.sort_by(|left, right| left.0.total_cmp(&right.0));
            sorted.dedup_by(|left, right| (left.0 - right.0).abs() <= f64::EPSILON);
            prop_assume!(sorted.len() >= 2);

            let op = BuildlutOp::<F32>::new(sorted.clone(), 64).unwrap();
            let samples = render(&op);
            let min_y = sorted.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min) as f32;
            let max_y = sorted.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max) as f32;

            prop_assert_eq!(samples.len(), 64);
            prop_assert!(samples.iter().all(|sample| sample.is_finite()));
            prop_assert!(samples.iter().all(|sample| *sample >= min_y - 1e-4 && *sample <= max_y + 1e-4));
        }
    }
}

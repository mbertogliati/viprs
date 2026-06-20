#![allow(clippy::needless_range_loop)]
// REASON: indexed loops keep patch matching aligned with packed tile buffers.

use std::marker::PhantomData;

use viprs_core::{
    error::{MosaicingError, ViprsError},
    format::BandFormat,
    image::Region,
};

const SINGULAR_EPSILON: f64 = 1e-12;

#[derive(Debug, Clone, Copy, PartialEq)]
/// Represents a tie point.
pub struct TiePoint {
    /// Horizontal factor associated with this condition.
    pub x: f64,
    /// Vertical factor associated with this condition.
    pub y: f64,
}

impl TiePoint {
    #[must_use]
    /// Creates a new `TiePoint`.
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Represents a tie point pair.
pub struct TiePointPair {
    /// Stores the `reference` value for this item.
    pub reference: TiePoint,
    /// Stores the `secondary` value for this item.
    pub secondary: TiePoint,
}

impl TiePointPair {
    #[must_use]
    /// Creates a new `TiePointPair`.
    pub const fn new(reference: TiePoint, secondary: TiePoint) -> Self {
        Self {
            reference,
            secondary,
        }
    }

    #[must_use]
    /// Creates this value from xy.
    pub const fn from_xy(xr: f64, yr: f64, xs: f64, ys: f64) -> Self {
        Self::new(TiePoint::new(xr, yr), TiePoint::new(xs, ys))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Represents an affine transform.
pub struct AffineTransform {
    /// Matrix associated with this condition.
    pub matrix: [f64; 4],
    /// Stores the `tx` value for this item.
    pub tx: f64,
    /// Stores the `ty` value for this item.
    pub ty: f64,
}

impl AffineTransform {
    /// Associated constant for identity.
    pub const IDENTITY: Self = Self {
        matrix: [1.0, 0.0, 0.0, 1.0],
        tx: 0.0,
        ty: 0.0,
    };

    #[must_use]
    /// Creates a new `AffineTransform`.
    pub const fn new(matrix: [f64; 4], tx: f64, ty: f64) -> Self {
        Self { matrix, tx, ty }
    }

    #[inline]
    #[must_use]
    /// Returns or performs map reference to secondary.
    pub fn map_reference_to_secondary(self, point: TiePoint) -> TiePoint {
        let [a, b, c, d] = self.matrix;
        TiePoint {
            x: b.mul_add(point.y, a * point.x) + self.tx,
            y: d.mul_add(point.y, c * point.x) + self.ty,
        }
    }

    /// Returns or performs inverse.
    pub fn inverse(self) -> Result<Self, ViprsError> {
        let [a, b, c, d] = self.matrix;
        let determinant = b.mul_add(-c, a * d);
        if determinant.abs() <= SINGULAR_EPSILON {
            return Err(MosaicingError::SingularAffineTransform.into());
        }

        let inv_det = 1.0 / determinant;
        let inv_matrix = [d * inv_det, -b * inv_det, -c * inv_det, a * inv_det];
        let inv_tx = -inv_matrix[1].mul_add(self.ty, inv_matrix[0] * self.tx);
        let inv_ty = -inv_matrix[3].mul_add(self.ty, inv_matrix[2] * self.tx);

        Ok(Self::new(inv_matrix, inv_tx, inv_ty))
    }

    #[inline]
    #[must_use]
    /// Returns or performs residual.
    pub fn residual(self, pair: TiePointPair) -> f64 {
        let predicted = self.map_reference_to_secondary(pair.reference);
        let dx = predicted.x - pair.secondary.x;
        let dy = predicted.y - pair.secondary.y;
        dx.hypot(dy)
    }
}

/// Applies the `match` mosaicing operation to related images. Use it when matching, aligning,
/// or merging overlapping image content.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::mosaicing::match_op::MatchOp;
///
/// let op = MatchOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone, Copy)]
pub struct MatchOp<F: BandFormat> {
    reference_region: Region,
    secondary_region: Region,
    _format: PhantomData<F>,
}

impl<F: BandFormat> MatchOp<F> {
    #[must_use]
    /// Creates a new `MatchOp`.
    pub const fn new(reference_region: Region, secondary_region: Region) -> Self {
        Self {
            reference_region,
            secondary_region,
            _format: PhantomData,
        }
    }

    /// Returns or performs fit.
    pub fn fit(&self, pairs: &[TiePointPair]) -> Result<AffineTransform, ViprsError> {
        validate_pairs(pairs, self.reference_region, self.secondary_region)?;
        fit_affine_transform(pairs)
    }
}

pub(crate) fn fit_affine_transform(pairs: &[TiePointPair]) -> Result<AffineTransform, ViprsError> {
    match pairs {
        [] => Err(MosaicingError::NotEnoughTiePoints {
            minimum: 1,
            actual: 0,
        }
        .into()),
        [pair] => Ok(AffineTransform::new(
            AffineTransform::IDENTITY.matrix,
            pair.secondary.x - pair.reference.x,
            pair.secondary.y - pair.reference.y,
        )),
        [first, second] => fit_similarity([*first, *second]),
        _ => fit_least_squares(pairs),
    }
}

fn validate_pairs(
    pairs: &[TiePointPair],
    reference_region: Region,
    secondary_region: Region,
) -> Result<(), ViprsError> {
    if pairs.is_empty() {
        return Err(MosaicingError::NotEnoughTiePoints {
            minimum: 1,
            actual: 0,
        }
        .into());
    }

    for pair in pairs {
        ensure_point_in_region("reference", pair.reference, reference_region)?;
        ensure_point_in_region("secondary", pair.secondary, secondary_region)?;
    }

    Ok(())
}

fn ensure_point_in_region(
    label: &'static str,
    point: TiePoint,
    region: Region,
) -> Result<(), ViprsError> {
    if point_in_region(point, region) {
        return Ok(());
    }

    Err(ViprsError::RegionOutOfBounds {
        requested: format!(
            "{label} tie-point ({:.3}, {:.3}) is outside region {:?}",
            point.x, point.y, region
        ),
        width: region.width,
        height: region.height,
    })
}

#[inline]
fn point_in_region(point: TiePoint, region: Region) -> bool {
    let left = f64::from(region.x);
    let top = f64::from(region.y);
    let right = left + f64::from(region.width);
    let bottom = top + f64::from(region.height);

    point.x.is_finite()
        && point.y.is_finite()
        && point.x >= left
        && point.y >= top
        && point.x < right
        && point.y < bottom
}

fn fit_similarity(pairs: [TiePointPair; 2]) -> Result<AffineTransform, ViprsError> {
    let ref_dx = pairs[1].reference.x - pairs[0].reference.x;
    let ref_dy = pairs[1].reference.y - pairs[0].reference.y;
    let sec_dx = pairs[1].secondary.x - pairs[0].secondary.x;
    let sec_dy = pairs[1].secondary.y - pairs[0].secondary.y;
    let denominator = ref_dy.mul_add(ref_dy, ref_dx * ref_dx);
    if denominator <= SINGULAR_EPSILON {
        return Err(MosaicingError::DegenerateTiePointConfiguration.into());
    }

    let a = sec_dy.mul_add(ref_dy, sec_dx * ref_dx) / denominator;
    let b = sec_dx.mul_add(-ref_dy, sec_dy * ref_dx) / denominator;
    let tx = pairs[0].secondary.x - b.mul_add(-pairs[0].reference.y, a * pairs[0].reference.x);
    let ty = pairs[0].secondary.y - a.mul_add(pairs[0].reference.y, b * pairs[0].reference.x);

    Ok(AffineTransform::new([a, -b, b, a], tx, ty))
}

fn fit_least_squares(pairs: &[TiePointPair]) -> Result<AffineTransform, ViprsError> {
    let ref_norm = PointNormalization::from_points(pairs.iter().map(|pair| pair.reference));
    let sec_norm = PointNormalization::from_points(pairs.iter().map(|pair| pair.secondary));
    let mut ata = [[0.0f64; 6]; 6];
    let mut atb = [0.0f64; 6];

    for pair in pairs {
        let reference = ref_norm.normalize(pair.reference);
        let secondary = sec_norm.normalize(pair.secondary);
        let xr = reference.x;
        let yr = reference.y;
        let xs = secondary.x;
        let ys = secondary.y;
        let x_row = [xr, yr, 1.0, 0.0, 0.0, 0.0];
        let y_row = [0.0, 0.0, 0.0, xr, yr, 1.0];
        accumulate_normal_equation(&mut ata, &mut atb, x_row, xs);
        accumulate_normal_equation(&mut ata, &mut atb, y_row, ys);
    }

    let solution = solve_linear_system(ata, atb)?;
    let scale_ratio = sec_norm.scale / ref_norm.scale;
    let matrix = [
        solution[0] * scale_ratio,
        solution[1] * scale_ratio,
        solution[3] * scale_ratio,
        solution[4] * scale_ratio,
    ];
    let tx = sec_norm.scale.mul_add(solution[2], sec_norm.cx)
        - matrix[1].mul_add(ref_norm.cy, matrix[0] * ref_norm.cx);
    let ty = sec_norm.scale.mul_add(solution[5], sec_norm.cy)
        - matrix[3].mul_add(ref_norm.cy, matrix[2] * ref_norm.cx);

    Ok(AffineTransform::new(matrix, tx, ty))
}

#[derive(Debug, Clone, Copy)]
struct PointNormalization {
    cx: f64,
    cy: f64,
    scale: f64,
}

impl PointNormalization {
    fn from_points(points: impl Iterator<Item = TiePoint> + Clone) -> Self {
        let mut count = 0usize;
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;

        for point in points.clone() {
            sum_x += point.x;
            sum_y += point.y;
            count += 1;
        }

        let cx = sum_x / count as f64;
        let cy = sum_y / count as f64;
        let mut mean_distance = 0.0f64;
        for point in points {
            mean_distance += (point.x - cx).hypot(point.y - cy);
        }
        mean_distance /= count as f64;

        Self {
            cx,
            cy,
            scale: mean_distance.max(1.0),
        }
    }

    #[inline]
    fn normalize(self, point: TiePoint) -> TiePoint {
        TiePoint::new(
            (point.x - self.cx) / self.scale,
            (point.y - self.cy) / self.scale,
        )
    }
}

fn accumulate_normal_equation(
    ata: &mut [[f64; 6]; 6],
    atb: &mut [f64; 6],
    row: [f64; 6],
    target: f64,
) {
    for r in 0..6 {
        atb[r] = row[r].mul_add(target, atb[r]);
        for c in 0..6 {
            ata[r][c] = row[r].mul_add(row[c], ata[r][c]);
        }
    }
}

fn solve_linear_system<const N: usize>(
    mut matrix: [[f64; N]; N],
    mut rhs: [f64; N],
) -> Result<[f64; N], ViprsError> {
    for pivot in 0..N {
        let mut best_row = pivot;
        let mut best_value = matrix[pivot][pivot].abs();
        for row in (pivot + 1)..N {
            let candidate = matrix[row][pivot].abs();
            if candidate > best_value {
                best_row = row;
                best_value = candidate;
            }
        }

        if best_value <= SINGULAR_EPSILON {
            return Err(MosaicingError::SingularAffineTransform.into());
        }

        if best_row != pivot {
            matrix.swap(pivot, best_row);
            rhs.swap(pivot, best_row);
        }

        let pivot_value = matrix[pivot][pivot];
        for col in pivot..N {
            matrix[pivot][col] /= pivot_value;
        }
        rhs[pivot] /= pivot_value;

        for row in 0..N {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            if factor.abs() <= SINGULAR_EPSILON {
                continue;
            }
            for col in pivot..N {
                matrix[row][col] = factor.mul_add(-matrix[pivot][col], matrix[row][col]);
            }
            rhs[row] = factor.mul_add(-rhs[pivot], rhs[row]);
        }
    }

    Ok(rhs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{format::U8, image::Region};

    fn approx_eq(lhs: f64, rhs: f64) {
        assert!((lhs - rhs).abs() <= 1e-9, "expected {lhs} ≈ {rhs}");
    }

    fn assert_transform_close(actual: AffineTransform, expected: AffineTransform) {
        for (lhs, rhs) in actual.matrix.iter().zip(expected.matrix) {
            approx_eq(*lhs, rhs);
        }
        approx_eq(actual.tx, expected.tx);
        approx_eq(actual.ty, expected.ty);
    }

    #[test]
    fn aligned_tie_points_fit_identity_transform() {
        let op = MatchOp::<U8>::new(Region::new(0, 0, 32, 32), Region::new(0, 0, 32, 32));
        let pairs = [
            TiePointPair::from_xy(4.0, 6.0, 4.0, 6.0),
            TiePointPair::from_xy(20.0, 6.0, 20.0, 6.0),
            TiePointPair::from_xy(8.0, 24.0, 8.0, 24.0),
        ];

        let transform = op.fit(&pairs).unwrap();

        assert_transform_close(transform, AffineTransform::IDENTITY);
    }

    #[test]
    fn two_point_fit_matches_similarity_reference_model() {
        let op = MatchOp::<U8>::new(Region::new(0, 0, 32, 32), Region::new(0, 0, 32, 32));
        let pairs = [
            TiePointPair::from_xy(0.0, 0.0, 3.0, 2.0),
            TiePointPair::from_xy(1.0, 0.0, 3.0, 3.0),
        ];

        let transform = op.fit(&pairs).unwrap();
        let mapped = transform.map_reference_to_secondary(TiePoint::new(0.0, 2.0));

        assert_transform_close(
            transform,
            AffineTransform::new([0.0, -1.0, 1.0, 0.0], 3.0, 2.0),
        );
        approx_eq(mapped.x, 1.0);
        approx_eq(mapped.y, 2.0);
    }

    #[test]
    fn single_pair_reduces_to_translation() {
        let op = MatchOp::<U8>::new(Region::new(0, 0, 16, 16), Region::new(0, 0, 16, 16));
        let transform = op
            .fit(&[TiePointPair::from_xy(2.0, 3.0, 7.0, 11.0)])
            .unwrap();

        assert_transform_close(
            transform,
            AffineTransform::new([1.0, 0.0, 0.0, 1.0], 5.0, 8.0),
        );
    }

    #[test]
    fn inverse_round_trips_points() {
        let transform = AffineTransform::new([1.1, -0.2, 0.15, 0.9], 3.0, -4.0);
        let inverse = transform.inverse().unwrap();
        let point = TiePoint::new(11.0, 7.0);
        let mapped = transform.map_reference_to_secondary(point);
        let restored = inverse.map_reference_to_secondary(mapped);

        approx_eq(restored.x, point.x);
        approx_eq(restored.y, point.y);
    }

    #[test]
    fn rejects_out_of_bounds_tie_points() {
        let op = MatchOp::<U8>::new(Region::new(0, 0, 8, 8), Region::new(0, 0, 8, 8));
        let error = op
            .fit(&[TiePointPair::from_xy(9.0, 1.0, 1.0, 1.0)])
            .unwrap_err();

        assert!(matches!(error, ViprsError::RegionOutOfBounds { .. }));
    }

    #[test]
    fn rejects_degenerate_similarity_configuration() {
        let op = MatchOp::<U8>::new(Region::new(0, 0, 16, 16), Region::new(0, 0, 16, 16));
        let error = op
            .fit(&[
                TiePointPair::from_xy(4.0, 4.0, 4.0, 4.0),
                TiePointPair::from_xy(4.0, 4.0, 8.0, 8.0),
            ])
            .unwrap_err();

        assert!(matches!(
            error,
            ViprsError::Mosaicing(MosaicingError::DegenerateTiePointConfiguration)
        ));
    }

    #[test]
    fn rejects_singular_least_squares_system() {
        let op = MatchOp::<U8>::new(Region::new(0, 0, 16, 16), Region::new(0, 0, 16, 16));
        let error = op
            .fit(&[
                TiePointPair::from_xy(1.0, 1.0, 3.0, 3.0),
                TiePointPair::from_xy(2.0, 2.0, 4.0, 4.0),
                TiePointPair::from_xy(3.0, 3.0, 5.0, 5.0),
            ])
            .unwrap_err();

        assert!(matches!(
            error,
            ViprsError::Mosaicing(MosaicingError::SingularAffineTransform)
        ));
    }

    #[test]
    fn inverse_rejects_singular_matrix() {
        let error = AffineTransform::new([1.0, 2.0, 2.0, 4.0], 0.0, 0.0)
            .inverse()
            .unwrap_err();

        assert!(matches!(
            error,
            ViprsError::Mosaicing(MosaicingError::SingularAffineTransform)
        ));
    }

    #[test]
    fn least_squares_fit_stays_stable_for_large_coordinates() {
        let region = Region::new(1_000_000_000, 1_000_000_000, 512, 512);
        let op = MatchOp::<U8>::new(region, region);
        let expected = AffineTransform::new([1.0, 0.0, 0.0, 1.0], 0.125, -0.375);
        let pairs = [
            TiePointPair::from_xy(
                1_000_000_032.0,
                1_000_000_040.0,
                1_000_000_032.125,
                1_000_000_039.625,
            ),
            TiePointPair::from_xy(
                1_000_000_200.0,
                1_000_000_064.0,
                1_000_000_200.125,
                1_000_000_063.625,
            ),
            TiePointPair::from_xy(
                1_000_000_080.0,
                1_000_000_220.0,
                1_000_000_080.125,
                1_000_000_219.625,
            ),
            TiePointPair::from_xy(
                1_000_000_240.0,
                1_000_000_256.0,
                1_000_000_240.125,
                1_000_000_255.625,
            ),
        ];

        let transform = op.fit(&pairs).unwrap();

        assert_transform_close(transform, expected);
    }

    proptest! {
        #[test]
        fn least_squares_recovers_consistent_affine_transforms(
            a in 0.8f64..1.2,
            b in -0.2f64..0.2,
            c in -0.2f64..0.2,
            d in 0.8f64..1.2,
            tx in 10.0f64..30.0,
            ty in 12.0f64..28.0,
        ) {
            prop_assume!((a * d - b * c).abs() > 0.1);
            let expected = AffineTransform::new([a, b, c, d], tx, ty);
            let reference_region = Region::new(0, 0, 128, 128);
            let secondary_region = Region::new(0, 0, 128, 128);
            let control = [
                TiePoint::new(5.0, 5.0),
                TiePoint::new(24.0, 7.0),
                TiePoint::new(8.0, 25.0),
                TiePoint::new(28.0, 26.0),
            ];
            let pairs: Vec<_> = control
                .into_iter()
                .map(|point| TiePointPair::new(point, expected.map_reference_to_secondary(point)))
                .collect();
            let op = MatchOp::<U8>::new(reference_region, secondary_region);
            let transform = op.fit(&pairs).unwrap();

            for pair in pairs {
                let mapped = transform.map_reference_to_secondary(pair.reference);
                prop_assert!((mapped.x - pair.secondary.x).abs() <= 1e-8);
                prop_assert!((mapped.y - pair.secondary.y).abs() <= 1e-8);
            }
        }
    }
}

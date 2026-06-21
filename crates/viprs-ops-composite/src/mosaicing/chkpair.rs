#![allow(clippy::unreadable_literal)]
// REASON: the literal constants mirror the original mosaic heuristics for easier parity checks.

use std::collections::HashSet;

use viprs_core::error::{MosaicingError, ViprsError};

use super::match_op::{AffineTransform, TiePointPair, fit_affine_transform};

const DEFAULT_MAX_HYPOTHESES: usize = 256;

/// Applies the `chkpair` mosaicing operation to related images. Use it when matching, aligning,
/// or merging overlapping image content.
///
/// # Examples
/// ```ignore
/// use viprs_ops_composite::mosaicing::chkpair::ChkpairOp;
///
/// let op = ChkpairOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone)]
pub struct ChkpairOp {
    residual_threshold: f64,
    max_hypotheses: usize,
}

impl ChkpairOp {
    /// Creates a new `ChkpairOp`.
    pub fn new(residual_threshold: f64) -> Result<Self, ViprsError> {
        validate_residual_threshold(residual_threshold)?;
        Ok(Self {
            residual_threshold,
            max_hypotheses: DEFAULT_MAX_HYPOTHESES,
        })
    }

    /// Returns this value configured with max hypotheses.
    pub fn with_max_hypotheses(mut self, max_hypotheses: usize) -> Result<Self, ViprsError> {
        if max_hypotheses == 0 {
            return Err(MosaicingError::InvalidHypothesisCount.into());
        }
        self.max_hypotheses = max_hypotheses;
        Ok(self)
    }

    /// Returns or performs filter.
    pub fn filter(&self, pairs: &[TiePointPair]) -> Result<Vec<TiePointPair>, ViprsError> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }

        if pairs.len() < 3 {
            let transform = fit_affine_transform(pairs)?;
            return Ok(collect_inliers(transform, pairs, self.residual_threshold));
        }

        let mut best: Option<ConsensusCandidate> = None;
        for hypothesis in hypothesis_triplets(pairs.len(), self.max_hypotheses) {
            let sample = [
                pairs[hypothesis[0]],
                pairs[hypothesis[1]],
                pairs[hypothesis[2]],
            ];
            let Ok(transform) = fit_affine_transform(&sample) else {
                continue;
            };
            let candidate = consensus_candidate(transform, pairs, self.residual_threshold);
            if candidate.inlier_count == 0 {
                continue;
            }

            if best.as_ref().is_none_or(|current| {
                candidate.inlier_count > current.inlier_count
                    || (candidate.inlier_count == current.inlier_count
                        && candidate.mean_residual < current.mean_residual)
            }) {
                best = Some(candidate);
            }
        }

        let seed = best.map_or(fit_affine_transform(pairs)?, |candidate| {
            candidate.transform
        });
        let initial_inliers = collect_inliers(seed, pairs, self.residual_threshold);
        if initial_inliers.is_empty() {
            return Ok(Vec::new());
        }

        let refined = fit_affine_transform(&initial_inliers).unwrap_or(seed);
        Ok(collect_inliers(refined, pairs, self.residual_threshold))
    }
}

#[derive(Debug, Clone, Copy)]
struct ConsensusCandidate {
    transform: AffineTransform,
    inlier_count: usize,
    mean_residual: f64,
}

fn validate_residual_threshold(threshold: f64) -> Result<(), ViprsError> {
    if threshold.is_finite() && threshold > 0.0 {
        Ok(())
    } else {
        Err(MosaicingError::InvalidResidualThreshold { threshold }.into())
    }
}

fn consensus_candidate(
    transform: AffineTransform,
    pairs: &[TiePointPair],
    threshold: f64,
) -> ConsensusCandidate {
    let mut count = 0usize;
    let mut residual_sum = 0.0f64;
    for pair in pairs {
        let residual = transform.residual(*pair);
        if residual <= threshold {
            count += 1;
            residual_sum += residual;
        }
    }

    ConsensusCandidate {
        transform,
        inlier_count: count,
        mean_residual: if count == 0 {
            f64::INFINITY
        } else {
            residual_sum / count as f64
        },
    }
}

fn collect_inliers(
    transform: AffineTransform,
    pairs: &[TiePointPair],
    threshold: f64,
) -> Vec<TiePointPair> {
    pairs
        .iter()
        .copied()
        .filter(|pair| transform.residual(*pair) <= threshold)
        .collect()
}

fn hypothesis_triplets(count: usize, max_hypotheses: usize) -> Vec<[usize; 3]> {
    if count < 3 {
        return Vec::new();
    }

    let total = count
        .saturating_mul(count.saturating_sub(1))
        .saturating_mul(count.saturating_sub(2))
        / 6;
    if total <= max_hypotheses {
        let mut exhaustive = Vec::with_capacity(total);
        for i in 0..(count - 2) {
            for j in (i + 1)..(count - 1) {
                for k in (j + 1)..count {
                    exhaustive.push([i, j, k]);
                }
            }
        }
        return exhaustive;
    }

    let target = max_hypotheses.min(total).max(1);
    let mut samples = Vec::with_capacity(target);
    samples.push([0, 1, 2]);
    let mut seen = HashSet::with_capacity(target * 2);
    seen.insert(encode_triplet([0, 1, 2]));
    let mut state = 0x9E37_79B9_7F4A_7C15_u64 ^ count as u64;
    let mut attempts = 0usize;

    while samples.len() < target && attempts < target * 64 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut indices = [0usize; 3];
        for index in &mut indices {
            state = state.wrapping_mul(1442695040888963407).wrapping_add(1);
            *index = (state as usize) % count;
        }
        if indices[0] == indices[1] || indices[0] == indices[2] || indices[1] == indices[2] {
            attempts += 1;
            continue;
        }
        indices.sort_unstable();
        let key = encode_triplet(indices);
        if seen.insert(key) {
            samples.push(indices);
        }
        attempts += 1;
    }

    samples
}

const fn encode_triplet(indices: [usize; 3]) -> u128 {
    (indices[0] as u128) << 84 | (indices[1] as u128) << 42 | indices[2] as u128
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn translated_pairs(dx: f64, dy: f64) -> Vec<TiePointPair> {
        vec![
            TiePointPair::from_xy(4.0, 4.0, 4.0 + dx, 4.0 + dy),
            TiePointPair::from_xy(20.0, 5.0, 20.0 + dx, 5.0 + dy),
            TiePointPair::from_xy(7.0, 22.0, 7.0 + dx, 22.0 + dy),
            TiePointPair::from_xy(24.0, 24.0, 24.0 + dx, 24.0 + dy),
        ]
    }

    #[test]
    fn retains_all_consistent_pairs() {
        let pairs = translated_pairs(3.0, -2.0);
        let filtered = ChkpairOp::new(0.25).unwrap().filter(&pairs).unwrap();

        assert_eq!(filtered, pairs);
    }

    #[test]
    fn removes_single_outlier_pair() {
        let mut pairs = translated_pairs(2.0, 3.0);
        let outlier = TiePointPair::from_xy(18.0, 11.0, 99.0, 105.0);
        pairs.push(outlier);

        let filtered = ChkpairOp::new(0.5).unwrap().filter(&pairs).unwrap();

        assert_eq!(filtered.len(), pairs.len() - 1);
        assert!(!filtered.contains(&outlier));
        assert!(
            filtered
                .iter()
                .all(|pair| pair.secondary.x - pair.reference.x == 2.0)
        );
    }

    #[test]
    fn empty_input_returns_empty_output() {
        let filtered = ChkpairOp::new(0.5).unwrap().filter(&[]).unwrap();

        assert!(filtered.is_empty());
    }

    #[test]
    fn validates_threshold_and_hypothesis_count() {
        assert!(matches!(
            ChkpairOp::new(0.0).unwrap_err(),
            ViprsError::Mosaicing(MosaicingError::InvalidResidualThreshold { .. })
        ));
        assert!(matches!(
            ChkpairOp::new(1.0)
                .unwrap()
                .with_max_hypotheses(0)
                .unwrap_err(),
            ViprsError::Mosaicing(MosaicingError::InvalidHypothesisCount)
        ));
    }

    #[test]
    fn two_pair_input_uses_direct_fit_path() {
        let pairs = vec![
            TiePointPair::from_xy(2.0, 2.0, 4.0, 5.0),
            TiePointPair::from_xy(5.0, 2.0, 7.0, 5.0),
        ];

        let filtered = ChkpairOp::new(1e-6).unwrap().filter(&pairs).unwrap();

        assert_eq!(filtered, pairs);
    }

    #[test]
    fn exhaustive_hypotheses_cover_all_triplets() {
        let hypotheses = hypothesis_triplets(4, 8);

        assert_eq!(hypotheses, vec![[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]);
    }

    #[test]
    fn sampled_hypotheses_are_sorted_and_bounded() {
        let hypotheses = hypothesis_triplets(12, 3);

        assert_eq!(hypotheses.len(), 3);
        assert_eq!(hypotheses[0], [0, 1, 2]);
        assert!(hypotheses.iter().all(|triplet| {
            triplet[0] < triplet[1] && triplet[1] < triplet[2] && triplet[2] < 12
        }));
    }

    #[test]
    fn degenerate_samples_fall_back_to_global_fit_error() {
        let pairs = vec![
            TiePointPair::from_xy(1.0, 1.0, 3.0, 3.0),
            TiePointPair::from_xy(2.0, 2.0, 4.0, 4.0),
            TiePointPair::from_xy(3.0, 3.0, 5.0, 5.0),
            TiePointPair::from_xy(4.0, 4.0, 6.0, 6.0),
        ];

        let error = ChkpairOp::new(0.5)
            .unwrap()
            .with_max_hypotheses(1)
            .unwrap()
            .filter(&pairs)
            .unwrap_err();

        assert!(matches!(
            error,
            ViprsError::Mosaicing(MosaicingError::SingularAffineTransform)
        ));
    }

    proptest! {
        #[test]
        fn consistent_translations_survive_ransac(dx in -6.0f64..6.0, dy in -6.0f64..6.0) {
            let pairs = translated_pairs(dx, dy);
            let filtered = ChkpairOp::new(1e-6).unwrap().filter(&pairs).unwrap();
            prop_assert_eq!(filtered, pairs);
        }
    }
}

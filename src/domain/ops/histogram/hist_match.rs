use std::marker::PhantomData;

use crate::domain::{
    error::ViprsError,
    format::{BandFormat, U8},
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

/// Apply histogram matching through a precomputed 256-entry LUT.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::histogram::hist_match::HistMatchOp;
///
/// let op = HistMatchOp { /* operation parameters */ };
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct HistMatchOp<F: BandFormat> {
    lut: [u8; 256],
    _phantom: PhantomData<F>,
}

impl<F: BandFormat> HistMatchOp<F> {
    /// Build the LUT from source and reference cumulative histograms.
    pub fn from_cumulative_hists(cum_src: &[u64], cum_ref: &[u64]) -> Result<Self, ViprsError> {
        if cum_src.len() != 256 || cum_ref.len() != 256 {
            return Err(ViprsError::Scheduler(
                "HistMatchOp requires 256-entry cumulative histograms".into(),
            ));
        }

        let mut lut = [0u8; 256];
        let mut ref_idx = 0usize;

        for (src_idx, &src_value) in cum_src.iter().enumerate() {
            while ref_idx + 1 < cum_ref.len() && cum_ref[ref_idx] < src_value {
                ref_idx += 1;
            }

            let mapped = if ref_idx == 0 {
                0usize
            } else {
                let prev = cum_ref[ref_idx - 1];
                let curr = cum_ref[ref_idx];
                if src_value.saturating_sub(prev) <= curr.saturating_sub(src_value) {
                    ref_idx - 1
                } else {
                    ref_idx
                }
            };

            lut[src_idx] = mapped as u8;
        }

        Ok(Self {
            lut,
            _phantom: PhantomData,
        })
    }

    #[must_use]
    /// Returns or performs lut.
    pub const fn lut(&self) -> &[u8; 256] {
        &self.lut
    }
}

impl Op for HistMatchOp<U8> {
    type Input = U8;
    type Output = U8;
    type State = ();

    fn preferred_tile_geometry(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<U8>, output: &mut TileMut<U8>) {
        for (src, dst) in input.data.iter().zip(output.data.iter_mut()) {
            *dst = self.lut[*src as usize];
        }
    }
}

impl PixelLocalOp for HistMatchOp<U8> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::image::Region;
    use proptest::prelude::*;

    fn identity_cdf() -> Vec<u64> {
        (0u64..256u64).collect()
    }

    fn cumulative_histogram(data: &[u8]) -> Vec<u64> {
        let mut hist = [0u64; 256];
        for &sample in data {
            hist[sample as usize] += 1;
        }

        let mut cumulative = Vec::with_capacity(256);
        let mut sum = 0u64;
        for bin in hist {
            sum += bin;
            cumulative.push(sum);
        }
        cumulative
    }

    fn apply_hist_match(input_data: &[u8], reference_data: &[u8]) -> Vec<u8> {
        let cum_src = cumulative_histogram(input_data);
        let cum_ref = cumulative_histogram(reference_data);
        let op = HistMatchOp::<U8>::from_cumulative_hists(&cum_src, &cum_ref).unwrap();
        let region = Region::new(0, 0, input_data.len() as u32, 1);
        let input = Tile::<U8>::new(region, 1, input_data);
        let mut output_data = vec![0u8; input_data.len()];
        let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    #[test]
    fn hist_match_identity_lut_for_same_histogram() {
        let cdf = identity_cdf();
        let op = HistMatchOp::<U8>::from_cumulative_hists(&cdf, &cdf).unwrap();
        for (idx, &mapped) in op.lut().iter().enumerate() {
            assert_eq!(mapped, idx as u8);
        }
    }

    #[test]
    fn hist_match_process_region_applies_lut() {
        let cdf = identity_cdf();
        let op = HistMatchOp::<U8>::from_cumulative_hists(&cdf, &cdf).unwrap();
        let region = Region::new(0, 0, 4, 1);
        let input_data = vec![0u8, 64, 128, 255];
        let mut output_data = vec![0u8; 4];
        let input = Tile::<U8>::new(region, 1, &input_data);
        let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, input_data);
    }

    #[test]
    fn hist_match_dark_image_to_bright_reference_brightens_output() {
        let input = vec![8u8, 16, 24, 32, 40, 48, 56, 64];
        let reference = vec![160u8, 176, 192, 208, 224, 240, 248, 255];

        let output = apply_hist_match(&input, &reference);

        let input_sum: u32 = input.iter().map(|&value| u32::from(value)).sum();
        let output_sum: u32 = output.iter().map(|&value| u32::from(value)).sum();

        assert!(
            output_sum > input_sum,
            "hist_match should brighten toward the bright reference"
        );
        assert!(
            output
                .iter()
                .zip(input.iter())
                .all(|(&out, &src)| out >= src)
        );
    }

    #[test]
    fn hist_match_identical_images_preserve_pixels() {
        let input = vec![0u8, 32, 32, 64, 96, 160, 224, 255];
        let output = apply_hist_match(&input, &input);
        assert_eq!(output, input);
    }

    #[test]
    fn hist_match_rejects_non_256_entry_cumulative_histograms() {
        assert!(HistMatchOp::<U8>::from_cumulative_hists(&[0; 255], &[0; 256]).is_err());
        assert!(HistMatchOp::<U8>::from_cumulative_hists(&[0; 256], &[0; 255]).is_err());
    }

    #[test]
    fn hist_match_prefers_previous_bin_when_distances_tie() {
        let mut src = vec![0u64; 256];
        let mut reference = vec![0u64; 256];
        for value in src.iter_mut().skip(5) {
            *value = 10;
        }
        for value in reference.iter_mut().skip(4) {
            *value = 8;
        }
        for value in reference.iter_mut().skip(5) {
            *value = 12;
        }

        let op = HistMatchOp::<U8>::from_cumulative_hists(&src, &reference).unwrap();

        assert_eq!(op.lut()[5], 4);
        assert_eq!(op.preferred_tile_geometry(), DemandHint::ThinStrip);
        assert_eq!(
            op.required_input_region(&Region::new(2, 3, 4, 1)),
            Region::new(2, 3, 4, 1)
        );
    }

    proptest! {
        #[test]
        fn hist_match_identity_prop(pixels in proptest::collection::vec(0u8..=255u8, 1..=128)) {
            let cdf = identity_cdf();
            let op = HistMatchOp::<U8>::from_cumulative_hists(&cdf, &cdf).unwrap();
            let region = Region::new(0, 0, pixels.len() as u32, 1);
            let input = Tile::<U8>::new(region, 1, &pixels);
            let mut output_data = vec![0u8; pixels.len()];
            let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
            let mut state = ();
            op.process_region(&mut state, &input, &mut output);
            prop_assert_eq!(output_data, pixels);
        }
    }
}

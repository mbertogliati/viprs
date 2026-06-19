use std::{cmp::Ordering, marker::PhantomData};

use crate::{
    domain::op::{Op, OperationBridge, PixelLocalOp},
    domain::{
        format::BandFormat,
        image::{DemandHint, Region, Tile, TileMut},
    },
};

/// Select a rank-ordered band from each pixel.
///
/// `BandRank::new(input_bands)` selects the median (`input_bands / 2`), matching
/// the common libvips use for band-wise median selection. Use `with_rank` to
/// request a different rank explicitly.
pub struct BandRank<F: BandFormat> {
    rank_index: usize,
    input_bands: usize,
    _f: PhantomData<F>,
}

impl<F: BandFormat> BandRank<F> {
    /// Construct a median `BandRank`.
    #[must_use]
    pub fn new(input_bands: usize) -> Self {
        debug_assert!(input_bands > 0, "BandRank: input_bands must be at least 1");
        Self::with_rank(input_bands / 2, input_bands)
    }

    /// Construct a `BandRank` that selects `rank_index` from the sorted band list.
    #[must_use]
    pub fn with_rank(rank_index: usize, input_bands: usize) -> Self {
        debug_assert!(input_bands > 0, "BandRank: input_bands must be at least 1");
        debug_assert!(
            rank_index < input_bands,
            "BandRank: rank_index {rank_index} out of range for input_bands {input_bands}"
        );
        Self {
            rank_index,
            input_bands,
            _f: PhantomData,
        }
    }
}

impl<F> BandRank<F>
where
    F: BandFormat,
    F::Sample: Copy + PartialOrd + bytemuck::Pod,
{
    /// Build an `OperationBridge` configured with the fixed 1-band output.
    #[must_use]
    pub fn into_bridge(self) -> OperationBridge<Self> {
        let input_bands = self.input_bands as u32;
        OperationBridge::new_pixel_local(self, input_bands)
    }
}

impl<F> Op for BandRank<F>
where
    F: BandFormat,
    F::Sample: Copy + PartialOrd,
{
    type Input = F;
    type Output = F;
    type State = Vec<F::Sample>;

    const OUTPUT_BANDS: Option<usize> = Some(1);

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) -> Self::State {
        Vec::with_capacity(self.input_bands)
    }

    #[inline]
    fn process_region(&self, state: &mut Self::State, input: &Tile<F>, output: &mut TileMut<F>) {
        debug_assert_eq!(
            input.bands as usize, self.input_bands,
            "BandRank input tile band count must match constructor input_bands"
        );
        debug_assert_eq!(
            output.bands, 1,
            "BandRank output tile must have exactly 1 band"
        );

        let pixel_count = input.region.pixel_count();
        for px in 0..pixel_count {
            let src_base = px * self.input_bands;
            let samples = &input.data[src_base..src_base + self.input_bands];

            let value = if self.rank_index == 0 {
                min_partial(samples)
            } else if self.rank_index + 1 == self.input_bands {
                max_partial(samples)
            } else {
                state.clear();
                state.extend_from_slice(samples);
                let (_, nth, _) = state
                    .select_nth_unstable_by(self.rank_index, partial_cmp_or_equal::<F::Sample>);
                *nth
            };

            output.data[px] = value;
        }
    }
}

impl<F> PixelLocalOp for BandRank<F>
where
    F: BandFormat,
    F::Sample: Copy + PartialOrd,
{
}

#[inline]
fn partial_cmp_or_equal<T: PartialOrd>(lhs: &T, rhs: &T) -> Ordering {
    lhs.partial_cmp(rhs).unwrap_or(Ordering::Equal)
}

#[inline]
fn min_partial<T: Copy + PartialOrd>(samples: &[T]) -> T {
    let mut current = samples[0];
    for sample in &samples[1..] {
        if matches!(sample.partial_cmp(&current), Some(Ordering::Less)) {
            current = *sample;
        }
    }
    current
}

#[inline]
fn max_partial<T: Copy + PartialOrd>(samples: &[T]) -> T {
    let mut current = samples[0];
    for sample in &samples[1..] {
        if matches!(sample.partial_cmp(&current), Some(Ordering::Greater)) {
            current = *sample;
        }
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        format::U8,
        image::{Region, Tile, TileMut},
        op::DynOperation,
    };
    use proptest::prelude::*;

    fn make_region(w: u32, h: u32) -> Region {
        Region::new(0, 0, w, h)
    }

    fn run_bandrank_u8(op: BandRank<U8>, input_data: &[u8], output_data: &mut [u8], pixels: usize) {
        let region = make_region(pixels as u32, 1);
        let input_bands = op.input_bands as u32;
        let input = Tile::<U8>::new(region, input_bands, input_data);
        let mut output = TileMut::<U8>::new(region, 1, output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
    }

    #[test]
    fn into_bridge_reports_one_band() {
        let bridge = BandRank::<U8>::new(5).into_bridge();
        assert_eq!(bridge.bands(), 1);
    }

    #[test]
    fn median_single_pixel() {
        let input = [9u8, 2, 7, 4, 5];
        let mut output = [0u8; 1];
        run_bandrank_u8(BandRank::<U8>::new(5), &input, &mut output, 1);
        assert_eq!(output, [5]);
    }

    #[test]
    fn explicit_min_and_max_rank_work() {
        let input = [9u8, 2, 7, 4, 5];
        let mut min_output = [0u8; 1];
        let mut max_output = [0u8; 1];
        run_bandrank_u8(BandRank::<U8>::with_rank(0, 5), &input, &mut min_output, 1);
        run_bandrank_u8(BandRank::<U8>::with_rank(4, 5), &input, &mut max_output, 1);
        assert_eq!(min_output, [2]);
        assert_eq!(max_output, [9]);
    }

    proptest! {
        #[test]
        fn median_of_identical_bands_is_identity(
            pixels in proptest::collection::vec(0u8..=255u8, 1..=64),
            bands in 1usize..=8,
        ) {
            let mut input = Vec::with_capacity(pixels.len() * bands);
            for &pixel in &pixels {
                for _ in 0..bands {
                    input.push(pixel);
                }
            }

            let mut output = vec![0u8; pixels.len()];
            run_bandrank_u8(BandRank::<U8>::new(bands), &input, &mut output, pixels.len());
            prop_assert_eq!(output, pixels);
        }
    }
}

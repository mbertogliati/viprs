use std::{cmp::Ordering, marker::PhantomData};

use crate::domain::{
    format::{BandFormatId, NumericBand},
    image::{DemandHint, Region, Tile, TileMut},
    op::{NodeSpec, Op},
};

const U8_HISTOGRAM_THRESHOLD_AREA: usize = 10;

/// Represents a rank state.
pub struct RankState<T> {
    scratch: Vec<T>,
}

/// Applies the `rank` morphological operation to the image. Use it for neighbourhood-based
/// shape filtering and mask analysis.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::morphology::rank::RankOp;
///
/// let op = RankOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct RankOp<F: NumericBand> {
    width: u32,
    height: u32,
    radius_x: u32,
    radius_y: u32,
    window_area: usize,
    rank: usize,
    _phantom: PhantomData<F>,
}

impl<F> RankOp<F>
where
    F: NumericBand,
    F::Sample: Copy + Default + PartialOrd,
{
    /// Creates a new `RankOp`.
    pub const fn new(width: u32, height: u32, rank: u32) -> Result<Self, &'static str> {
        if width == 0 || height == 0 {
            return Err("RankOp: window dimensions must be >= 1");
        }

        let window_area = width as usize * height as usize;
        if rank as usize >= window_area {
            return Err("RankOp: rank must be within the window area");
        }

        Ok(Self {
            width,
            height,
            radius_x: width / 2,
            radius_y: height / 2,
            window_area,
            rank: rank as usize,
            _phantom: PhantomData,
        })
    }

    #[inline]
    fn should_use_histogram_fast_path(&self) -> bool {
        F::ID == BandFormatId::U8 && self.window_area > U8_HISTOGRAM_THRESHOLD_AREA
    }

    #[inline]
    fn process_dispatch(
        &self,
        state: &mut RankState<F::Sample>,
        input: &Tile<F>,
        output: &mut TileMut<F>,
    ) {
        if self.should_use_histogram_fast_path() {
            // SAFETY: F::ID == U8 implies F::Sample == u8 because BandFormat is sealed.
            let input_u8 = unsafe {
                std::slice::from_raw_parts(input.data.as_ptr().cast::<u8>(), input.data.len())
            };
            // SAFETY: same invariant as above for the mutable output slice.
            let output_u8 = unsafe {
                std::slice::from_raw_parts_mut(
                    output.data.as_mut_ptr().cast::<u8>(),
                    output.data.len(),
                )
            };
            self.process_region_histogram_u8(
                input_u8,
                input.region,
                input.bands,
                output_u8,
                output.region,
            );
        } else {
            self.process_region_select(state.scratch.as_mut_slice(), input, output);
        }
    }

    fn process_region_select(
        &self,
        scratch: &mut [F::Sample],
        input: &Tile<F>,
        output: &mut TileMut<F>,
    ) {
        let out_w = output.region.width as usize;
        let out_h = output.region.height as usize;
        let in_w = input.region.width as usize;
        let bands = input.bands as usize;
        let window_w = self.width as usize;
        let window_h = self.height as usize;

        for oy in 0..out_h {
            for ox in 0..out_w {
                for band in 0..bands {
                    let mut write_idx = 0usize;
                    for wy in 0..window_h {
                        let row_base = ((oy + wy) * in_w + ox) * bands + band;
                        for wx in 0..window_w {
                            scratch[write_idx] = input.data[row_base + wx * bands];
                            write_idx += 1;
                        }
                    }

                    let (_, rank_value, _) = scratch
                        .select_nth_unstable_by(self.rank, partial_cmp_or_equal::<F::Sample>);
                    output.data[(oy * out_w + ox) * bands + band] = *rank_value;
                }
            }
        }
    }

    fn process_region_histogram_u8(
        &self,
        input: &[u8],
        input_region: Region,
        bands: u32,
        output: &mut [u8],
        output_region: Region,
    ) {
        let out_w = output_region.width as usize;
        let out_h = output_region.height as usize;
        let in_w = input_region.width as usize;
        let bands = bands as usize;
        let window_w = self.width as usize;
        let window_h = self.height as usize;

        for oy in 0..out_h {
            for band in 0..bands {
                let mut hist = [0u32; 256];
                for wy in 0..window_h {
                    let row_base = (oy + wy) * in_w;
                    for wx in 0..window_w {
                        let sample = input[(row_base + wx) * bands + band];
                        hist[sample as usize] += 1;
                    }
                }

                output[(oy * out_w) * bands + band] = histogram_select(&hist, self.rank);

                for ox in 1..out_w {
                    let remove_x = ox - 1;
                    let add_x = ox + window_w - 1;
                    for wy in 0..window_h {
                        let row_base = (oy + wy) * in_w;
                        let remove_sample = input[(row_base + remove_x) * bands + band];
                        let add_sample = input[(row_base + add_x) * bands + band];
                        hist[remove_sample as usize] -= 1;
                        hist[add_sample as usize] += 1;
                    }

                    output[(oy * out_w + ox) * bands + band] = histogram_select(&hist, self.rank);
                }
            }
        }
    }
}

impl<F> Op for RankOp<F>
where
    F: NumericBand,
    F::Sample: Copy + Default + PartialOrd,
{
    type Input = F;
    type Output = F;
    type State = RankState<F::Sample>;

    fn demand_hint(&self) -> DemandHint {
        DemandHint::SmallTile
    }

    fn required_input_region(&self, output: &Region) -> Region {
        Region::new(
            output.x - self.radius_x as i32,
            output.y - self.radius_y as i32,
            output.width + 2 * self.radius_x,
            output.height + 2 * self.radius_y,
        )
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec {
            input_tile_w: tile_w + 2 * self.radius_x,
            input_tile_h: tile_h + 2 * self.radius_y,
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    fn start(&self) -> Self::State {
        RankState {
            scratch: vec![F::Sample::default(); self.window_area],
        }
    }

    #[inline]
    fn process_region(
        &self,
        state: &mut Self::State,
        input: &Tile<Self::Input>,
        output: &mut TileMut<Self::Output>,
    ) {
        self.process_dispatch(state, input, output);
    }
}

#[inline]
fn partial_cmp_or_equal<T: PartialOrd>(lhs: &T, rhs: &T) -> Ordering {
    lhs.partial_cmp(rhs).unwrap_or(Ordering::Equal)
}

#[inline]
fn histogram_select(hist: &[u32; 256], index: usize) -> u8 {
    let mut sum = 0usize;
    for (value, count) in hist.iter().enumerate() {
        sum += *count as usize;
        if sum > index {
            return value as u8;
        }
    }
    u8::MAX
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        format::{F32, U8},
        image::{Tile, TileMut},
        ops::morphology::{Dilate, Erode},
    };
    use proptest::prelude::*;
    use std::cmp::Ordering;

    fn edge_extend_scanline(samples: &[u8], radius: usize) -> Vec<u8> {
        let mut extended = Vec::with_capacity(samples.len() + 2 * radius);
        for x in 0..(samples.len() + 2 * radius) {
            let src_x = (x as i32 - radius as i32).clamp(0, samples.len() as i32 - 1) as usize;
            extended.push(samples[src_x]);
        }
        extended
    }

    #[test]
    fn rank_zero_matches_binary_erosion() {
        let rank = RankOp::<U8>::new(3, 3, 0).unwrap();
        let erode = Erode::rect(3).unwrap();
        let input_region = Region::new(0, 0, 5, 5);
        let output_region = Region::new(0, 0, 3, 3);
        let input_data = vec![
            255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255,
        ];

        let input = Tile::<U8>::new(input_region, 1, &input_data);
        let mut rank_output = vec![0u8; 9];
        let mut erode_output = vec![0u8; 9];
        let mut rank_tile = TileMut::<U8>::new(output_region, 1, &mut rank_output);
        let mut erode_tile = TileMut::<U8>::new(output_region, 1, &mut erode_output);

        let mut rank_state = rank.start();
        rank.process_region(&mut rank_state, &input, &mut rank_tile);
        let mut erode_state = erode.start();
        erode.process_region(&mut erode_state, &input, &mut erode_tile);

        assert_eq!(rank_output, erode_output);
    }

    #[test]
    fn rank_single_pixel_window_is_identity() {
        let op = RankOp::<U8>::new(1, 1, 0).unwrap();
        let region = Region::new(0, 0, 4, 1);
        let input = Tile::<U8>::new(region, 1, &[3, 1, 4, 1]);
        let mut output_data = vec![0u8; 4];
        let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);

        assert_eq!(output_data, vec![3, 1, 4, 1]);
    }

    #[test]
    fn constructor_rejects_invalid_windows() {
        assert!(RankOp::<U8>::new(0, 1, 0).is_err());
        assert!(RankOp::<U8>::new(1, 0, 0).is_err());
        assert!(RankOp::<U8>::new(2, 2, 4).is_err());
    }

    #[test]
    fn metadata_reports_radius_and_allocated_scratch() {
        let op = RankOp::<U8>::new(5, 3, 7).unwrap();
        let output = Region::new(3, 4, 6, 7);
        let state = op.start();

        assert!(op.should_use_histogram_fast_path());
        assert_eq!(state.scratch.len(), 15);
        assert_eq!(op.demand_hint(), DemandHint::SmallTile);
        assert_eq!(op.required_input_region(&output), Region::new(1, 3, 10, 9));
        assert_eq!(
            op.node_spec(6, 7),
            NodeSpec {
                input_tile_w: 10,
                input_tile_h: 9,
                output_tile_w: 6,
                output_tile_h: 7,
                coordinate_driven_source: None,
            }
        );
    }

    #[test]
    fn rank_max_matches_binary_dilation() {
        let rank = RankOp::<U8>::new(3, 3, 8).unwrap();
        let dilate = Dilate::rect(3).unwrap();
        let input_region = Region::new(0, 0, 5, 5);
        let output_region = Region::new(0, 0, 3, 3);
        let input_data = vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];

        let input = Tile::<U8>::new(input_region, 1, &input_data);
        let mut rank_output = vec![0u8; 9];
        let mut dilate_output = vec![0u8; 9];
        let mut rank_tile = TileMut::<U8>::new(output_region, 1, &mut rank_output);
        let mut dilate_tile = TileMut::<U8>::new(output_region, 1, &mut dilate_output);

        let mut rank_state = rank.start();
        rank.process_region(&mut rank_state, &input, &mut rank_tile);
        let mut dilate_state = dilate.start();
        dilate.process_region(&mut dilate_state, &input, &mut dilate_tile);

        assert_eq!(rank_output, dilate_output);
    }

    #[test]
    fn rank_median_on_odd_window_selects_middle_value() {
        let rank = RankOp::<U8>::new(3, 3, 4).unwrap();
        let input_region = Region::new(0, 0, 3, 3);
        let output_region = Region::new(0, 0, 1, 1);
        let input_data = vec![9, 1, 5, 3, 7, 2, 8, 4, 6];
        let input = Tile::<U8>::new(input_region, 1, &input_data);
        let mut output_data = vec![0u8; 1];
        let mut output = TileMut::<U8>::new(output_region, 1, &mut output_data);

        let mut state = rank.start();
        rank.process_region(&mut state, &input, &mut output);

        assert_eq!(output_data, vec![5]);
    }

    #[test]
    fn rank_histogram_path_matches_expected_median() {
        let rank = RankOp::<U8>::new(5, 3, 7).unwrap();
        let input_region = Region::new(0, 0, 5, 3);
        let output_region = Region::new(0, 0, 1, 1);
        let input_data = vec![15, 2, 9, 7, 12, 6, 1, 14, 3, 11, 10, 5, 4, 13, 8];
        let input = Tile::<U8>::new(input_region, 1, &input_data);
        let mut output_data = vec![0u8; 1];
        let mut output = TileMut::<U8>::new(output_region, 1, &mut output_data);

        let mut state = rank.start();
        rank.process_region(&mut state, &input, &mut output);

        assert_eq!(output_data, vec![8]);
    }

    #[test]
    fn histogram_helpers_cover_boundary_cases() {
        let mut hist = [0u32; 256];
        hist[0] = 1;
        hist[255] = 2;
        assert_eq!(histogram_select(&hist, 0), 0);
        assert_eq!(histogram_select(&hist, 1), 255);
        assert_eq!(histogram_select(&hist, 2), 255);
        assert_eq!(partial_cmp_or_equal(&f32::NAN, &1.0), Ordering::Equal);
    }

    #[test]
    fn f32_rank_uses_selection_path_for_non_u8_inputs() {
        let rank = RankOp::<F32>::new(3, 1, 1).unwrap();
        let input_region = Region::new(0, 0, 3, 1);
        let output_region = Region::new(0, 0, 1, 1);
        let input_data = vec![3.0f32, 1.0, 2.0];
        let input = Tile::<F32>::new(input_region, 1, &input_data);
        let mut output_data = vec![0.0f32; 1];
        let mut output = TileMut::<F32>::new(output_region, 1, &mut output_data);
        let mut state = rank.start();

        assert!(!rank.should_use_histogram_fast_path());
        rank.process_region(&mut state, &input, &mut output);

        assert_eq!(output_data, vec![2.0]);
    }

    #[test]
    fn histogram_path_handles_multiple_bands() {
        let rank = RankOp::<U8>::new(5, 3, 7).unwrap();
        let input_region = Region::new(0, 0, 5, 3);
        let output_region = Region::new(0, 0, 1, 1);
        let input_data = vec![
            15, 115, 2, 102, 9, 109, 7, 107, 12, 112, //
            6, 106, 1, 101, 14, 114, 3, 103, 11, 111, //
            10, 110, 5, 105, 4, 104, 13, 113, 8, 108,
        ];
        let input = Tile::<U8>::new(input_region, 2, &input_data);
        let mut output_data = vec![0u8; 2];
        let mut output = TileMut::<U8>::new(output_region, 2, &mut output_data);
        let mut state = rank.start();

        rank.process_region(&mut state, &input, &mut output);

        assert_eq!(output_data, vec![8, 108]);
    }

    #[test]
    fn even_window_rank_matches_sorted_index() {
        let rank = RankOp::<U8>::new(4, 4, 7).unwrap();
        let input_region = Region::new(0, 0, 4, 4);
        let output_region = Region::new(0, 0, 1, 1);
        let input_data = vec![16, 1, 12, 4, 9, 7, 14, 2, 15, 3, 10, 6, 13, 5, 11, 8];
        let input = Tile::<U8>::new(input_region, 1, &input_data);
        let mut output_data = vec![0u8; 1];
        let mut output = TileMut::<U8>::new(output_region, 1, &mut output_data);
        let mut state = rank.start();

        rank.process_region(&mut state, &input, &mut output);

        let mut sorted = input_data.clone();
        sorted.sort_unstable();
        assert_eq!(output_data[0], sorted[7]);
    }

    proptest! {
        #[test]
        fn horizontal_reflection_preserves_rank_for_symmetric_window(
            samples in prop::collection::vec(0u8..=255u8, 1..16)
        ) {
            let rank = RankOp::<U8>::new(3, 1, 1).unwrap();
            let radius = 1usize;
            let input = edge_extend_scanline(&samples, radius);
            let mirrored_samples: Vec<u8> = samples.iter().copied().rev().collect();
            let mirrored_input = edge_extend_scanline(&mirrored_samples, radius);
            let in_region = Region::new(-(radius as i32), 0, input.len() as u32, 1);
            let out_region = Region::new(0, 0, samples.len() as u32, 1);
            let input_tile = Tile::<U8>::new(in_region, 1, &input);
            let mirrored_input_tile = Tile::<U8>::new(in_region, 1, &mirrored_input);
            let mut output = vec![0u8; samples.len()];
            let mut mirrored_output = vec![0u8; samples.len()];
            let mut output_tile = TileMut::<U8>::new(out_region, 1, &mut output);
            let mut mirrored_output_tile = TileMut::<U8>::new(out_region, 1, &mut mirrored_output);
            let mut state = rank.start();
            let mut mirrored_state = rank.start();

            rank.process_region(&mut state, &input_tile, &mut output_tile);
            rank.process_region(&mut mirrored_state, &mirrored_input_tile, &mut mirrored_output_tile);

            for (lhs, rhs) in output.iter().zip(mirrored_output.iter().rev()) {
                prop_assert_eq!(lhs, rhs);
            }
        }
    }
}

#![allow(clippy::unused_self)]
// REASON: region-labelling helpers remain instance methods for API consistency.

use crate::domain::{
    error::ViprsError,
    format::{U8, U32},
    image::{DemandHint, Region, Tile, TileMut},
    op::Op,
};

const INTERNAL_TILE_EDGE: usize = 128;

/// Represents a label regions state.
pub struct LabelRegionsState {
    parent: Vec<usize>,
    rank: Vec<u8>,
    scratch_overflowed: bool,
}

impl LabelRegionsState {
    fn with_capacity(pixel_count: usize) -> Self {
        Self {
            parent: vec![0; pixel_count],
            rank: vec![0; pixel_count],
            scratch_overflowed: false,
        }
    }

    const fn overflowed() -> Self {
        Self {
            parent: Vec::new(),
            rank: Vec::new(),
            scratch_overflowed: true,
        }
    }
}

/// Applies the `labelregions` morphological operation to the image. Use it for
/// neighbourhood-based shape filtering and mask analysis.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::morphology::labelregions::LabelRegionsOp;
///
/// let op = LabelRegionsOp;
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct LabelRegionsOp;

impl LabelRegionsOp {
    #[must_use]
    /// Creates a new `LabelRegionsOp`.
    pub const fn new() -> Self {
        Self
    }

    #[inline]
    fn image_too_large(width: u32, height: u32, details: &'static str) -> ViprsError {
        ViprsError::ImageTooLarge {
            width,
            height,
            bands: 1,
            bytes: u128::from(width) * u128::from(height),
            limit_bytes: usize::MAX as u128,
            details,
        }
    }

    #[inline]
    fn checked_state_len_for_tile(tile_w: u32, tile_h: u32) -> Result<usize, ViprsError> {
        let Some(pixel_count) = tile_w.checked_mul(tile_h) else {
            return Err(Self::image_too_large(
                tile_w,
                tile_h,
                "labelregions scratch exceeds addressable memory",
            ));
        };

        usize::try_from(pixel_count).map_err(|_| {
            Self::image_too_large(
                tile_w,
                tile_h,
                "labelregions scratch exceeds addressable memory",
            )
        })
    }

    #[inline]
    fn checked_output_pixel_count(region: Region) -> Result<usize, ViprsError> {
        let Some(_) = region.width.checked_mul(region.height) else {
            return Err(Self::image_too_large(
                region.width,
                region.height,
                "labelregions output pixel count exceeds addressable memory",
            ));
        };

        region.checked_pixel_count().ok_or_else(|| {
            Self::image_too_large(
                region.width,
                region.height,
                "labelregions output pixel count exceeds addressable memory",
            )
        })
    }

    fn process_region_checked(
        self,
        state: &mut LabelRegionsState,
        input: &Tile<U8>,
        output: &mut TileMut<U32>,
    ) -> Result<(), ViprsError> {
        let pixel_count = Self::checked_output_pixel_count(output.region)?;
        let width = output.region.width as usize;
        let height = output.region.height as usize;
        let bands = input.bands as usize;

        if state.scratch_overflowed {
            return Err(Self::image_too_large(
                output.region.width,
                output.region.height,
                "labelregions scratch exceeds addressable memory",
            ));
        }
        if state.parent.len() != pixel_count || state.rank.len() != pixel_count {
            return Err(ViprsError::Scheduler(
                "LabelRegionsOp state must be pre-sized with start_with_tile_and_bands".into(),
            ));
        }

        output.data.fill(0);
        for (idx, parent) in state.parent.iter_mut().enumerate() {
            *parent = idx;
        }
        state.rank.fill(0);

        label_internal_tiles(
            &mut state.parent,
            &mut state.rank,
            input.data,
            width,
            height,
            bands,
        );
        merge_internal_tile_boundaries(
            &mut state.parent,
            &mut state.rank,
            input.data,
            width,
            height,
            bands,
        );

        let mut next_label = 1u32;
        for idx in 0..pixel_count {
            let root = find_root(&mut state.parent, idx);
            if output.data[root] == 0 {
                output.data[root] = next_label;
                next_label += 1;
            }
            output.data[idx] = output.data[root];
        }

        Ok(())
    }
}

impl Op for LabelRegionsOp {
    type Input = U8;
    type Output = U32;
    type State = LabelRegionsState;

    const OUTPUT_BANDS: Option<usize> = Some(1);

    fn demand_hint(&self) -> DemandHint {
        DemandHint::FullImage
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) -> Self::State {
        LabelRegionsState::with_capacity(0)
    }

    fn start_with_tile_and_bands(&self, tile_w: u32, tile_h: u32, _bands: u32) -> Self::State {
        Self::checked_state_len_for_tile(tile_w, tile_h).map_or_else(
            |_| LabelRegionsState::overflowed(),
            LabelRegionsState::with_capacity,
        )
    }

    fn validate_region_contract(
        &self,
        input_region: Region,
        input_bands: u32,
        output_region: Region,
        output_bands: u32,
    ) -> Result<(), ViprsError> {
        let _ = (input_region, input_bands, output_bands);
        Self::checked_output_pixel_count(output_region).map(|_| ())
    }

    #[inline]
    fn process_region(&self, state: &mut Self::State, input: &Tile<U8>, output: &mut TileMut<U32>) {
        // Label numbering must be global, so we keep FullImage and do the intra-tile
        // labeling plus cross-tile boundary merge inside the pre-sized per-thread state.
        debug_assert_eq!(input.region, output.region);
        debug_assert_eq!(output.bands, 1);

        if self.process_region_checked(state, input, output).is_err() {
            output.data.fill(0);
        }
    }
}

#[inline(always)]
fn pixels_are_equal(data: &[u8], lhs_base: usize, rhs_base: usize, bands: usize) -> bool {
    data[lhs_base..lhs_base + bands] == data[rhs_base..rhs_base + bands]
}

fn label_internal_tiles(
    parent: &mut [usize],
    rank: &mut [u8],
    data: &[u8],
    width: usize,
    height: usize,
    bands: usize,
) {
    for tile_y in (0..height).step_by(INTERNAL_TILE_EDGE) {
        let tile_end_y = (tile_y + INTERNAL_TILE_EDGE).min(height);
        for tile_x in (0..width).step_by(INTERNAL_TILE_EDGE) {
            let tile_end_x = (tile_x + INTERNAL_TILE_EDGE).min(width);

            for y in tile_y..tile_end_y {
                for x in tile_x..tile_end_x {
                    let idx = y * width + x;
                    let pixel_base = idx * bands;

                    if x > tile_x && pixels_are_equal(data, pixel_base, pixel_base - bands, bands) {
                        union(parent, rank, idx, idx - 1);
                    }
                    if y > tile_y
                        && pixels_are_equal(data, pixel_base, pixel_base - (width * bands), bands)
                    {
                        union(parent, rank, idx, idx - width);
                    }
                }
            }
        }
    }
}

fn merge_internal_tile_boundaries(
    parent: &mut [usize],
    rank: &mut [u8],
    data: &[u8],
    width: usize,
    height: usize,
    bands: usize,
) {
    for boundary_x in (INTERNAL_TILE_EDGE..width).step_by(INTERNAL_TILE_EDGE) {
        let left_x = boundary_x - 1;
        for y in 0..height {
            let left = y * width + left_x;
            let right = left + 1;
            if pixels_are_equal(data, left * bands, right * bands, bands) {
                union(parent, rank, left, right);
            }
        }
    }

    for boundary_y in (INTERNAL_TILE_EDGE..height).step_by(INTERNAL_TILE_EDGE) {
        let top_row = (boundary_y - 1) * width;
        let bottom_row = boundary_y * width;
        for x in 0..width {
            let top = top_row + x;
            let bottom = bottom_row + x;
            if pixels_are_equal(data, top * bands, bottom * bands, bands) {
                union(parent, rank, top, bottom);
            }
        }
    }
}

fn find_root(parent: &mut [usize], idx: usize) -> usize {
    let mut root = idx;
    while parent[root] != root {
        root = parent[root];
    }

    let mut current = idx;
    while parent[current] != root {
        let next = parent[current];
        parent[current] = root;
        current = next;
    }

    root
}

fn union(parent: &mut [usize], rank: &mut [u8], lhs: usize, rhs: usize) {
    let lhs_root = find_root(parent, lhs);
    let rhs_root = find_root(parent, rhs);
    if lhs_root == rhs_root {
        return;
    }

    match rank[lhs_root].cmp(&rank[rhs_root]) {
        std::cmp::Ordering::Less => parent[lhs_root] = rhs_root,
        std::cmp::Ordering::Greater => parent[rhs_root] = lhs_root,
        std::cmp::Ordering::Equal => {
            parent[rhs_root] = lhs_root;
            rank[lhs_root] += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use std::collections::{BTreeSet, VecDeque};

    fn run_op(width: u32, height: u32, data: &[u8]) -> Vec<u32> {
        let op = LabelRegionsOp::default();
        let region = Region::new(0, 0, width, height);
        let input = Tile::<U8>::new(region, 1, data);
        let mut output_data = vec![0u32; width as usize * height as usize];
        let mut output = TileMut::<U32>::new(region, 1, &mut output_data);
        let mut state = op.start_with_tile_and_bands(width, height, 1);
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    fn unique_labels(labels: &[u32]) -> BTreeSet<u32> {
        labels.iter().copied().collect()
    }

    fn reference_labels(width: u32, height: u32, data: &[u8]) -> Vec<u32> {
        let width = width as usize;
        let height = height as usize;
        let mut labels = vec![0u32; width * height];
        let mut next_label = 1u32;
        let mut queue = VecDeque::new();

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                if labels[idx] != 0 {
                    continue;
                }

                let pixel_value = data[idx];
                labels[idx] = next_label;
                queue.push_back(idx);

                while let Some(current) = queue.pop_front() {
                    let cx = current % width;
                    let cy = current / width;

                    if cx > 0 {
                        let left = current - 1;
                        if data[left] == pixel_value && labels[left] == 0 {
                            labels[left] = next_label;
                            queue.push_back(left);
                        }
                    }
                    if cx + 1 < width {
                        let right = current + 1;
                        if data[right] == pixel_value && labels[right] == 0 {
                            labels[right] = next_label;
                            queue.push_back(right);
                        }
                    }
                    if cy > 0 {
                        let up = current - width;
                        if data[up] == pixel_value && labels[up] == 0 {
                            labels[up] = next_label;
                            queue.push_back(up);
                        }
                    }
                    if cy + 1 < height {
                        let down = current + width;
                        if data[down] == pixel_value && labels[down] == 0 {
                            labels[down] = next_label;
                            queue.push_back(down);
                        }
                    }
                }

                next_label += 1;
            }
        }

        labels
    }

    #[test]
    fn validate_region_contract_rejects_overflowing_full_image_scratch() {
        let op = LabelRegionsOp::default();
        let huge = Region::new(0, 0, u32::MAX, u32::MAX);

        let err = op.validate_region_contract(huge, 1, huge, 1).unwrap_err();

        assert!(matches!(
            err,
            ViprsError::ImageTooLarge {
                width: u32::MAX,
                height: u32::MAX,
                bands: 1,
                ..
            }
        ));
    }

    #[test]
    fn op_reports_full_image_demand_and_identity_input_region() {
        let op = LabelRegionsOp::new();
        let region = Region::new(3, 4, 5, 6);

        assert_eq!(op.demand_hint(), DemandHint::FullImage);
        assert_eq!(op.required_input_region(&region), region);
    }

    #[test]
    fn checked_state_len_for_tile_rejects_overflowing_dimensions_with_typed_error() {
        let err = LabelRegionsOp::checked_state_len_for_tile(u32::MAX, 2).unwrap_err();

        assert!(matches!(
            err,
            ViprsError::ImageTooLarge {
                width: u32::MAX,
                height: 2,
                bands: 1,
                details: "labelregions scratch exceeds addressable memory",
                ..
            }
        ));
    }

    #[test]
    fn process_region_returns_typed_error_when_state_marks_scratch_overflow() {
        let op = LabelRegionsOp::default();
        let region = Region::new(0, 0, 1, 1);
        let input = Tile::<U8>::new(region, 1, &[0]);
        let mut output_data = [0u32; 1];
        let mut output = TileMut::<U32>::new(region, 1, &mut output_data);
        let mut state = LabelRegionsState::overflowed();

        let err = op
            .process_region_checked(&mut state, &input, &mut output)
            .unwrap_err();

        assert!(matches!(
            err,
            ViprsError::ImageTooLarge {
                width: 1,
                height: 1,
                bands: 1,
                details: "labelregions scratch exceeds addressable memory",
                ..
            }
        ));
    }

    #[test]
    fn start_with_tile_and_bands_overflow_state_returns_error_when_processing_any_tile() {
        let op = LabelRegionsOp::default();
        let region = Region::new(0, 0, 1, 1);
        let input = Tile::<U8>::new(region, 1, &[0]);
        let mut output_data = [0u32; 1];
        let mut output = TileMut::<U32>::new(region, 1, &mut output_data);
        let mut state = op.start_with_tile_and_bands(u32::MAX, 2, 1);

        let err = op
            .process_region_checked(&mut state, &input, &mut output)
            .unwrap_err();

        assert!(matches!(
            err,
            ViprsError::ImageTooLarge {
                width: 1,
                height: 1,
                bands: 1,
                details: "labelregions scratch exceeds addressable memory",
                ..
            }
        ));
    }

    #[test]
    fn process_region_returns_scheduler_error_when_state_is_not_pre_sized_for_the_tile() {
        let op = LabelRegionsOp::default();
        let region = Region::new(0, 0, 1, 1);
        let input = Tile::<U8>::new(region, 1, &[0]);
        let mut output_data = [0u32; 1];
        let mut output = TileMut::<U32>::new(region, 1, &mut output_data);
        let mut state = op.start();

        let err = op
            .process_region_checked(&mut state, &input, &mut output)
            .unwrap_err();

        assert!(matches!(
            err,
            ViprsError::Scheduler(message)
                if message == "LabelRegionsOp state must be pre-sized with start_with_tile_and_bands"
        ));
    }

    #[test]
    fn process_region_returns_scheduler_error_when_only_the_rank_buffer_is_mismatched() {
        let op = LabelRegionsOp::default();
        let region = Region::new(0, 0, 1, 1);
        let input = Tile::<U8>::new(region, 1, &[0]);
        let mut output_data = [0u32; 1];
        let mut output = TileMut::<U32>::new(region, 1, &mut output_data);
        let mut state = LabelRegionsState {
            parent: vec![0],
            rank: Vec::new(),
            scratch_overflowed: false,
        };

        let err = op
            .process_region_checked(&mut state, &input, &mut output)
            .unwrap_err();

        assert!(matches!(
            err,
            ViprsError::Scheduler(message)
                if message == "LabelRegionsOp state must be pre-sized with start_with_tile_and_bands"
        ));
    }

    #[test]
    fn process_region_returns_typed_error_when_output_region_pixel_count_overflows() {
        let op = LabelRegionsOp::default();
        let huge = Region::new(0, 0, u32::MAX, u32::MAX);
        let input = Tile::<U8> {
            region: huge,
            bands: 1,
            data: &[],
        };
        let mut output = TileMut::<U32> {
            region: huge,
            bands: 1,
            data: &mut [],
        };
        let mut state = op.start_with_tile_and_bands(1, 1, 1);

        let err = op
            .process_region_checked(&mut state, &input, &mut output)
            .unwrap_err();

        assert!(matches!(
            err,
            ViprsError::ImageTooLarge {
                width: u32::MAX,
                height: u32::MAX,
                bands: 1,
                details: "labelregions output pixel count exceeds addressable memory",
                ..
            }
        ));
    }

    #[test]
    fn differing_pixels_across_horizontal_internal_tile_boundary_remain_separate_regions() {
        let width = INTERNAL_TILE_EDGE as u32 + 1;
        let mut data = vec![0u8; width as usize];
        data[INTERNAL_TILE_EDGE - 1] = 7;
        data[INTERNAL_TILE_EDGE] = 8;

        let labels = run_op(width, 1, &data);

        assert_ne!(labels[INTERNAL_TILE_EDGE - 1], labels[INTERNAL_TILE_EDGE]);
        assert_ne!(labels[INTERNAL_TILE_EDGE - 1], labels[0]);
        assert_ne!(labels[INTERNAL_TILE_EDGE], labels[0]);
    }

    #[test]
    fn differing_pixels_across_vertical_internal_tile_boundary_remain_separate_regions() {
        let height = INTERNAL_TILE_EDGE as u32 + 1;
        let mut data = vec![0u8; height as usize];
        data[INTERNAL_TILE_EDGE - 1] = 7;
        data[INTERNAL_TILE_EDGE] = 8;

        let labels = run_op(1, height, &data);

        assert_ne!(labels[INTERNAL_TILE_EDGE - 1], labels[INTERNAL_TILE_EDGE]);
        assert_ne!(labels[INTERNAL_TILE_EDGE - 1], labels[0]);
        assert_ne!(labels[INTERNAL_TILE_EDGE], labels[0]);
    }

    #[test]
    fn component_crossing_only_horizontal_internal_tile_boundary_keeps_one_label() {
        let width = INTERNAL_TILE_EDGE as u32 + 1;
        let mut data = vec![0u8; width as usize];
        data[INTERNAL_TILE_EDGE - 1] = 7;
        data[INTERNAL_TILE_EDGE] = 7;

        let labels = run_op(width, 1, &data);

        assert_eq!(labels[INTERNAL_TILE_EDGE - 1], labels[INTERNAL_TILE_EDGE]);
        assert_ne!(labels[INTERNAL_TILE_EDGE - 1], labels[0]);
    }

    #[test]
    fn component_crossing_only_vertical_internal_tile_boundary_keeps_one_label() {
        let height = INTERNAL_TILE_EDGE as u32 + 1;
        let mut data = vec![0u8; height as usize];
        data[INTERNAL_TILE_EDGE - 1] = 7;
        data[INTERNAL_TILE_EDGE] = 7;

        let labels = run_op(1, height, &data);

        assert_eq!(labels[INTERNAL_TILE_EDGE - 1], labels[INTERNAL_TILE_EDGE]);
        assert_ne!(labels[INTERNAL_TILE_EDGE - 1], labels[0]);
    }

    proptest! {
        #[test]
        fn all_zero_image_is_single_region(width in 1u32..=16, height in 1u32..=16) {
            let data = vec![0u8; width as usize * height as usize];
            let labels = run_op(width, height, &data);
            prop_assert!(labels.iter().all(|&label| label == 1));
        }

        #[test]
        fn all_equal_image_is_single_region(
            width in 1u32..=16,
            height in 1u32..=16,
            pixel in any::<u8>(),
        ) {
            let data = vec![pixel; width as usize * height as usize];
            let labels = run_op(width, height, &data);
            prop_assert!(labels.iter().all(|&label| label == 1));
        }

        #[test]
        fn same_value_contiguous_pixels_share_a_label_and_adjacent_different_values_do_not(
            left_value in any::<u8>(),
            delta in 1u8..=u8::MAX,
        ) {
            let right_value = left_value.wrapping_add(delta);
            let labels = run_op(3, 1, &[left_value, left_value, right_value]);

            prop_assert_eq!(labels[0], labels[1]);
            prop_assert_ne!(labels[1], labels[2]);
        }

        #[test]
        fn two_separated_equal_value_blocks_are_not_merged_across_a_different_region(
            left_width in 1u32..=5,
            gap_width in 1u32..=4,
            right_width in 1u32..=5,
            height in 1u32..=5,
        ) {
            let width = left_width + gap_width + right_width;
            let mut data = vec![0u8; width as usize * height as usize];

            for y in 0..height as usize {
                let row = y * width as usize;
                data[row..row + left_width as usize].fill(1);
                data[row + (left_width + gap_width) as usize..row + width as usize].fill(1);
            }

            let labels = run_op(width, height, &data);
            let unique: Vec<u32> = unique_labels(&labels).into_iter().collect();
            let left_label = labels[0];
            let gap_label = labels[left_width as usize];
            let right_label = labels[(left_width + gap_width) as usize];

            prop_assert_eq!(unique, vec![1, 2, 3]);
            prop_assert_ne!(left_label, gap_label);
            prop_assert_ne!(gap_label, right_label);
            prop_assert_ne!(left_label, right_label);
        }

        #[test]
        fn labels_are_contiguous_without_gaps(
            (width, height, pixels) in (1u32..=8, 1u32..=8)
                .prop_flat_map(|(width, height)| {
                    let pixel_count = (width * height) as usize;
                    (Just(width), Just(height), vec(any::<u8>(), pixel_count))
                }),
        ) {
            let labels = run_op(width, height, &pixels);
            let unique: Vec<u32> = unique_labels(&labels).into_iter().collect();
            let expected: Vec<u32> = (1..=unique.len() as u32).collect();
            prop_assert_eq!(unique, expected);
        }

        #[test]
        fn labels_match_reference_connected_components(
            (width, height, pixels) in (1u32..=16, 1u32..=16)
                .prop_flat_map(|(width, height)| {
                    let pixel_count = (width * height) as usize;
                    (Just(width), Just(height), vec(any::<u8>(), pixel_count))
                }),
        ) {
            let labels = run_op(width, height, &pixels);
            let expected = reference_labels(width, height, &pixels);
            prop_assert_eq!(labels, expected);
        }

        #[test]
        fn component_crossing_internal_tile_edges_keeps_one_label(
            extension in 1u32..=8,
            pad in 0u32..=8,
        ) {
            let edge = INTERNAL_TILE_EDGE as u32;
            let width = edge + extension + pad;
            let height = edge + extension + pad;
            let mut data = vec![0u8; width as usize * height as usize];

            for y in edge - 1..=edge {
                for x in edge - 1..=edge {
                    data[y as usize * width as usize + x as usize] = 7;
                }
            }

            let labels = run_op(width, height, &data);
            let top_left = labels[(edge - 1) as usize * width as usize + (edge - 1) as usize];
            let top_right = labels[(edge - 1) as usize * width as usize + edge as usize];
            let bottom_left = labels[edge as usize * width as usize + (edge - 1) as usize];
            let bottom_right = labels[edge as usize * width as usize + edge as usize];

            prop_assert_eq!(top_left, top_right);
            prop_assert_eq!(top_left, bottom_left);
            prop_assert_eq!(top_left, bottom_right);
            prop_assert_ne!(labels[0], top_left);
        }
    }
}

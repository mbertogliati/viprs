#![allow(dead_code)]
// REASON: the bridge glue is staged for future pipeline-builder exposure.

use std::{any::Any, marker::PhantomData};

use viprs_core::{
    format::{BandFormat, BandFormatId},
    image::{DemandHint, Region, Tile, TileMut},
    op::{DynOperation, NodeSpec, Op, OperationBridge},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Enumerates the available angle45 values.
pub enum Angle45 {
    /// Uses the `D0` variant of `Angle45`.
    D0,
    /// Uses the `D45` variant of `Angle45`.
    D45,
    /// Uses the `D90` variant of `Angle45`.
    D90,
    /// Uses the `D135` variant of `Angle45`.
    D135,
    /// Uses the `D180` variant of `Angle45`.
    D180,
    /// Uses the `D225` variant of `Angle45`.
    D225,
    /// Uses the `D270` variant of `Angle45`.
    D270,
    /// Uses the `D315` variant of `Angle45`.
    D315,
}

impl Angle45 {
    const fn steps(self) -> u8 {
        match self {
            Self::D0 => 0,
            Self::D45 => 1,
            Self::D90 => 2,
            Self::D135 => 3,
            Self::D180 => 4,
            Self::D225 => 5,
            Self::D270 => 6,
            Self::D315 => 7,
        }
    }
}

/// Lossless libvips-style 45 degree rotation for odd square masks.
///
/// # Examples
/// ```ignore
/// use viprs_ops_composite::conversion::rot45::Rot45Op;
///
/// let op = Rot45Op::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct Rot45Op<F: BandFormat> {
    size: u32,
    angle: Angle45,
    _format: PhantomData<F>,
}

impl<F: BandFormat + Send + Sync> Rot45Op<F> {
    #[must_use]
    /// Creates a new `Rot45Op`.
    pub fn new(size: u32, angle: Angle45) -> Self {
        debug_assert!(size % 2 == 1, "Rot45Op requires an odd square image");
        Self {
            size,
            angle,
            _format: PhantomData,
        }
    }

    #[must_use]
    /// Returns or performs angle.
    pub const fn angle(&self) -> Angle45 {
        self.angle
    }
}

impl<F> Op for Rot45Op<F>
where
    F: BandFormat,
    F::Sample: Copy,
{
    type Input = F;
    type Output = F;
    type State = Vec<F::Sample>;

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, _output: &Region) -> Region {
        Region::new(0, 0, self.size, self.size)
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        let _ = (tile_w, tile_h);
        NodeSpec {
            input_tile_w: self.size,
            input_tile_h: self.size,
            output_tile_w: self.size,
            output_tile_h: self.size,
            coordinate_driven_source: None,
        }
    }

    fn start(&self) -> Vec<F::Sample> {
        Vec::new()
    }

    fn start_with_tile_and_bands(&self, tile_w: u32, tile_h: u32, bands: u32) -> Vec<F::Sample> {
        let scratch_w = tile_w.max(self.size);
        let scratch_h = tile_h.max(self.size);
        let len = scratch_w as usize * scratch_h as usize * bands as usize;
        vec![input_zero::<F>(); len]
    }

    #[inline]
    fn process_region(&self, state: &mut Vec<F::Sample>, input: &Tile<F>, output: &mut TileMut<F>) {
        debug_assert_eq!(input.region.width, self.size);
        debug_assert_eq!(input.region.height, self.size);
        debug_assert_eq!(output.region.width, self.size);
        debug_assert_eq!(output.region.height, self.size);

        let len = input.data.len();
        if self.angle.steps() == 0 {
            output.data.copy_from_slice(input.data);
            return;
        }

        debug_assert!(state.len() >= len, "rot45 scratch must be preallocated");

        let scratch = &mut state[..len];
        apply_rot45_once(self.size, output.bands, input.data, output.data);
        let mut last_wrote_to_scratch = false;

        for _ in 1..self.angle.steps() {
            if last_wrote_to_scratch {
                apply_rot45_once(self.size, output.bands, scratch, output.data);
            } else {
                apply_rot45_once(self.size, output.bands, output.data, scratch);
            }
            last_wrote_to_scratch = !last_wrote_to_scratch;
        }

        if last_wrote_to_scratch {
            output.data.copy_from_slice(scratch);
        }
    }
}

#[inline(always)]
fn input_zero<F: BandFormat>() -> F::Sample {
    bytemuck::Zeroable::zeroed()
}

#[inline(always)]
const fn pixel_base(size: usize, bands: usize, x: usize, y: usize) -> usize {
    (y * size + x) * bands
}

#[inline(always)]
fn copy_pixel<T: Copy>(
    src: &[T],
    dst: &mut [T],
    src_x: usize,
    src_y: usize,
    dst_x: usize,
    dst_y: usize,
    size: usize,
    bands: usize,
) {
    let src_base = pixel_base(size, bands, src_x, src_y);
    let dst_base = pixel_base(size, bands, dst_x, dst_y);
    dst[dst_base..dst_base + bands].copy_from_slice(&src[src_base..src_base + bands]);
}

fn apply_rot45_once<T: Copy>(size: u32, bands: u32, src: &[T], dst: &mut [T]) {
    let size = size as usize;
    let half = size / 2;
    let bands = bands as usize;

    for y in 0..half {
        for x in y..half {
            copy_pixel(src, dst, y, half - (x - y), x, y, size, bands);
            copy_pixel(src, dst, y, size - 1 - x, y, half - (x - y), size, bands);
            copy_pixel(
                src,
                dst,
                half - (x - y),
                size - 1 - y,
                y,
                size - 1 - x,
                size,
                bands,
            );
            copy_pixel(
                src,
                dst,
                size - 1 - x,
                size - 1 - y,
                half - (x - y),
                size - 1 - y,
                size,
                bands,
            );
            copy_pixel(
                src,
                dst,
                size - 1 - y,
                half + (x - y),
                size - 1 - x,
                size - 1 - y,
                size,
                bands,
            );
            copy_pixel(
                src,
                dst,
                size - 1 - y,
                x,
                size - 1 - y,
                half + (x - y),
                size,
                bands,
            );
            copy_pixel(src, dst, half + (x - y), y, size - 1 - y, x, size, bands);
            copy_pixel(src, dst, x, y, half + (x - y), y, size, bands);
        }
    }

    copy_pixel(src, dst, half, half, half, half, size, bands);
}

pub(crate) struct Rot45Bridge<F: BandFormat>
where
    F::Sample: bytemuck::Pod + Copy,
{
    inner: OperationBridge<Rot45Op<F>>,
}

impl<F: BandFormat> Rot45Bridge<F>
where
    F::Sample: bytemuck::Pod + Copy,
{
    pub fn new(size: u32, angle: Angle45, bands: u32) -> Self {
        Self {
            inner: OperationBridge::new(Rot45Op::new(size, angle), bands),
        }
    }
}

impl<F: BandFormat> DynOperation for Rot45Bridge<F>
where
    F::Sample: bytemuck::Pod + Copy + Send,
{
    fn input_format(&self) -> BandFormatId {
        self.inner.input_format()
    }

    fn output_format(&self) -> BandFormatId {
        self.inner.output_format()
    }

    fn bands(&self) -> u32 {
        self.inner.bands()
    }

    fn demand_hint(&self) -> DemandHint {
        self.inner.demand_hint()
    }

    fn required_input_region(&self, output: &Region) -> Region {
        self.inner.required_input_region(output)
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        self.inner.node_spec(tile_w, tile_h)
    }

    fn dyn_start(&self) -> Box<dyn Any + Send> {
        self.inner.dyn_start()
    }

    fn dyn_start_with_tile(&self, tile_w: u32, tile_h: u32) -> Box<dyn Any + Send> {
        self.inner.dyn_start_with_tile(tile_w, tile_h)
    }

    fn dyn_start_with_tile_and_bands(
        &self,
        tile_w: u32,
        tile_h: u32,
        bands: u32,
    ) -> Box<dyn Any + Send> {
        self.inner
            .dyn_start_with_tile_and_bands(tile_w, tile_h, bands)
    }

    fn dyn_process_region(
        &self,
        state: &mut dyn Any,
        input: &[u8],
        output: &mut [u8],
        input_region: Region,
        output_region: Region,
    ) {
        self.inner
            .dyn_process_region(state, input, output, input_region, output_region);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{format::U8, op::DynOperation};

    fn run_rot45_with_bands(size: u32, angle: Angle45, bands: u32, pixels: &[u8]) -> Vec<u8> {
        let op = Rot45Op::<U8>::new(size, angle);
        let output_region = Region::new(0, 0, size, size);
        let input_region = op.required_input_region(&output_region);
        let input = Tile::<U8>::new(input_region, bands, pixels);
        let mut output = vec![0u8; output_region.pixel_count() * bands as usize];
        let mut output_tile = TileMut::<U8>::new(output_region, bands, &mut output);
        let mut state = op.start_with_tile_and_bands(size, size, bands);
        op.process_region(&mut state, &input, &mut output_tile);
        output
    }

    fn run_rot45(size: u32, angle: Angle45, pixels: &[u8]) -> Vec<u8> {
        run_rot45_with_bands(size, angle, 1, pixels)
    }

    #[test]
    fn d45_matches_libvips_triangle_mapping_for_5x5() {
        let input = (0u8..25).collect::<Vec<_>>();
        let output = run_rot45(5, Angle45::D45, &input);
        assert_eq!(
            output,
            vec![
                10, 5, 0, 1, 2, 15, 11, 6, 7, 3, 20, 16, 12, 8, 4, 21, 17, 18, 13, 9, 22, 23, 24,
                19, 14,
            ]
        );
    }

    #[test]
    fn d0_is_identity_boundary_single_pixel() {
        assert_eq!(run_rot45(1, Angle45::D0, &[255]), vec![255]);
    }

    #[test]
    fn bridge_keeps_dimensions_and_format() {
        let bridge = Rot45Bridge::<U8>::new(5, Angle45::D45, 3);
        assert_eq!(bridge.output_width(5), 5);
        assert_eq!(bridge.output_height(5), 5);
        assert_eq!(bridge.input_format(), BandFormatId::U8);
        assert_eq!(bridge.output_format(), BandFormatId::U8);
    }

    #[test]
    fn scratch_preallocation_covers_odd_rectangular_state_regions() {
        let op = Rot45Op::<U8>::new(5, Angle45::D45);

        for (tile_w, tile_h) in [(7, 5), (11, 9)] {
            let state = op.start_with_tile_and_bands(tile_w, tile_h, 3);
            assert!(state.len() >= (op.size as usize * op.size as usize * 3));
        }
    }

    #[test]
    fn rot45_odd_sized_does_not_panic() {
        for size in [5, 11] {
            let len = (size * size * 3) as usize;
            let input = (0..len).map(|idx| (idx % 251) as u8).collect::<Vec<_>>();
            let output = run_rot45_with_bands(size, Angle45::D45, 3, &input);
            assert_eq!(output.len(), len);
            assert!(output.iter().any(|&sample| sample != 0));
        }
    }

    #[test]
    fn angle_variants_cover_all_step_counts_and_node_spec_stays_square() {
        let cases = [
            (Angle45::D0, 0),
            (Angle45::D45, 1),
            (Angle45::D90, 2),
            (Angle45::D135, 3),
            (Angle45::D180, 4),
            (Angle45::D225, 5),
            (Angle45::D270, 6),
            (Angle45::D315, 7),
        ];
        for (angle, steps) in cases {
            assert_eq!(angle.steps(), steps);
            assert_eq!(Rot45Op::<U8>::new(5, angle).angle(), angle);
        }

        let spec = Rot45Op::<U8>::new(7, Angle45::D270).node_spec(2, 3);
        assert_eq!(spec.input_tile_w, 7);
        assert_eq!(spec.input_tile_h, 7);
        assert_eq!(spec.output_tile_w, 7);
        assert_eq!(spec.output_tile_h, 7);
    }

    #[test]
    fn d180_matches_four_successive_d45_turns_for_multiband_tiles() {
        let size = 5;
        let input = (0..(size * size * 3) as usize)
            .map(|idx| (idx % 251) as u8)
            .collect::<Vec<_>>();
        let mut iterated = input.clone();
        for _ in 0..4 {
            iterated = run_rot45_with_bands(size, Angle45::D45, 3, &iterated);
        }
        assert_eq!(
            run_rot45_with_bands(size, Angle45::D180, 3, &input),
            iterated
        );
    }

    #[test]
    fn helper_functions_preserve_pixel_coordinates_and_empty_state() {
        assert_eq!(input_zero::<U8>(), 0);
        assert_eq!(pixel_base(5, 3, 2, 1), 21);

        let src = (0u8..9).collect::<Vec<_>>();
        let mut dst = vec![0u8; 9];
        copy_pixel(&src, &mut dst, 1, 1, 0, 0, 3, 1);
        assert_eq!(dst[0], 4);

        let mut rotated = vec![0u8; 9];
        apply_rot45_once(3, 1, &src, &mut rotated);
        assert_eq!(rotated, vec![3, 0, 1, 6, 4, 2, 7, 8, 5]);

        let op = Rot45Op::<U8>::new(3, Angle45::D45);
        assert!(op.start().is_empty());
        assert_eq!(op.demand_hint(), DemandHint::Any);
    }

    #[test]
    fn dyn_bridge_start_with_tile_and_bands_preserves_output_size() {
        let bridge = Rot45Bridge::<U8>::new(5, Angle45::D225, 2);
        let mut state = bridge.dyn_start_with_tile_and_bands(3, 4, 2);
        let input_region = Region::new(0, 0, 5, 5);
        let output_region = Region::new(0, 0, 5, 5);
        let input = (0..50u8).collect::<Vec<_>>();
        let mut output = vec![0u8; 50];
        bridge.dyn_process_region(
            state.as_mut(),
            &input,
            &mut output,
            input_region,
            output_region,
        );
        assert_eq!(output.len(), input.len());
        assert_ne!(output, input);
    }

    proptest! {
        #[test]
        fn eight_d45_turns_are_identity(size_half in 0u32..=4) {
            let size = size_half * 2 + 1;
            let len = (size * size) as usize;
            let input = (0..len).map(|idx| (idx % 251) as u8).collect::<Vec<_>>();
            let mut image = input.clone();
            for _ in 0..8 {
                image = run_rot45(size, Angle45::D45, &image);
            }
            prop_assert_eq!(image, input);
        }

        #[test]
        fn required_region_is_full_odd_square(size_half in 0u32..=8) {
            let size = size_half * 2 + 1;
            let op = Rot45Op::<U8>::new(size, Angle45::D45);
            prop_assert_eq!(
                op.required_input_region(&Region::new(1, 1, 1, 1)),
                Region::new(0, 0, size, size)
            );
        }
    }
}

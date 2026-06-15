//! Integer-factor shrink wrapper.
//!
//! `Shrink` exposes the libvips-style integer shrink entry point by sequencing
//! `ShrinkH` then `ShrinkV` with floor-divided output geometry on both passes.

#![allow(private_bounds)]
// REASON: shrink sample conversion stays crate-private while the public op remains typed.
#![allow(clippy::needless_range_loop)]
// REASON: indexed loops keep the shrink kernel aligned with packed tile buffers.

use crate::domain::{
    error::{BuildError, ViprsError},
    format::{BandFormat, BandFormatId},
    image::{DemandHint, Region, Tile, TileMut},
    op::{DynOperation, NodeSpec, Op, OperationBridge},
    resample::ResampleOp,
};

use super::{
    shrinkh::{ShrinkH, ShrinkSample},
    shrinkv::ShrinkV,
};

/// Per-thread scratch storage for [`Shrink`].
pub struct ShrinkState<T> {
    scratch: Vec<T>,
    vertical_scratch: Vec<u32>,
}

/// Integer-factor downscale wrapper that sequences [`ShrinkH`] then [`ShrinkV`].
///
/// Output dimensions follow libvips `shrink`: `floor(width / hshrink)` and
/// `floor(height / vshrink)`. Factors must be at least 1.
pub struct Shrink<F: BandFormat>
where
    F::Sample: ShrinkSample + bytemuck::Pod,
{
    hshrink: u32,
    vshrink: u32,
    horizontal: ShrinkH<F>,
    vertical: ShrinkV<F>,
}

impl<F: BandFormat + Send + Sync> Shrink<F>
where
    F::Sample: ShrinkSample + bytemuck::Pod,
{
    /// Creates a new `Shrink`.
    pub fn new(hshrink: usize, vshrink: usize) -> Result<Self, BuildError> {
        let hshrink = validate_shrink_factor("hshrink", hshrink)?;
        let vshrink = validate_shrink_factor("vshrink", vshrink)?;

        Ok(Self {
            hshrink,
            vshrink,
            horizontal: ShrinkH::new(hshrink)?,
            vertical: ShrinkV::new(vshrink)?,
        })
    }

    #[inline]
    fn intermediate_region(&self, output: &Region) -> Region {
        self.vertical.required_input_region(output)
    }

    #[inline]
    fn scratch_len_for_tile(&self, tile_w: u32, tile_h: u32, bands: u32) -> usize {
        let spec = self.node_spec(tile_w, tile_h);
        spec.output_tile_w as usize * spec.input_tile_h as usize * bands as usize
    }

    #[inline]
    fn uses_direct_u8_2x2(&self) -> bool {
        F::ID == BandFormatId::U8 && self.hshrink == 2 && self.vshrink == 2
    }

    #[inline]
    fn checked_scratch_len(region: Region, bands: u32) -> Result<usize, ViprsError> {
        region
            .checked_pixel_count()
            .and_then(|n| n.checked_mul(bands as usize))
            .ok_or_else(|| ViprsError::ImageTooLarge {
                width: region.width,
                height: region.height,
                bands,
                bytes: u128::from(region.width) * u128::from(region.height) * u128::from(bands),
                limit_bytes: usize::MAX as u128,
                details: "shrink scratch exceeds addressable memory",
            })
    }
}

fn validate_shrink_factor(name: &'static str, factor: usize) -> Result<u32, BuildError> {
    if !(1..=u32::MAX as usize).contains(&factor) {
        return Err(BuildError::SourceHint {
            context: "shrink",
            message: format!("{name} must be in 1..=u32::MAX"),
        });
    }

    Ok(factor as u32)
}

impl<F> Op for Shrink<F>
where
    F: BandFormat,
    F::Sample: ShrinkSample + bytemuck::Pod,
{
    type Input = F;
    type Output = F;
    type State = ShrinkState<F::Sample>;

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        let vertical_input = self.vertical.required_input_region(output);
        self.horizontal.required_input_region(&vertical_input)
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec {
            input_tile_w: tile_w.saturating_mul(self.hshrink),
            input_tile_h: tile_h.saturating_mul(self.vshrink),
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    fn start(&self) -> Self::State {
        if self.uses_direct_u8_2x2() {
            return ShrinkState {
                scratch: Vec::new(),
                vertical_scratch: Vec::new(),
            };
        }

        ShrinkState {
            scratch: Vec::new(),
            vertical_scratch: Vec::new(),
        }
    }

    fn start_with_tile_and_bands(&self, tile_w: u32, tile_h: u32, bands: u32) -> Self::State {
        if self.uses_direct_u8_2x2() {
            return ShrinkState {
                scratch: Vec::new(),
                vertical_scratch: Vec::new(),
            };
        }

        let scratch_len = self.scratch_len_for_tile(tile_w, tile_h, bands);
        ShrinkState {
            scratch: vec![F::Sample::from_f64_clamped(0.0); scratch_len],
            vertical_scratch: vec![0; tile_w as usize * bands as usize],
        }
    }

    fn validate_region_contract(
        &self,
        input_region: Region,
        input_bands: u32,
        output_region: Region,
        output_bands: u32,
    ) -> Result<(), ViprsError> {
        let _ = (input_region, output_bands);
        if self.uses_direct_u8_2x2() {
            return Ok(());
        }
        let intermediate_region = self.intermediate_region(&output_region);
        Self::checked_scratch_len(intermediate_region, input_bands).map(|_| ())
    }

    #[inline]
    fn process_region(&self, state: &mut Self::State, input: &Tile<F>, output: &mut TileMut<F>) {
        if self.uses_direct_u8_2x2() {
            shrink_u8_2x2(
                bytemuck::cast_slice(input.data),
                input.bands as usize,
                output.region.width as usize,
                output.region.height as usize,
                bytemuck::cast_slice_mut(output.data),
            );
            return;
        }

        let intermediate_region = self.intermediate_region(&output.region);
        let Ok(scratch_len) = Self::checked_scratch_len(intermediate_region, input.bands) else {
            debug_assert!(false, "Shrink scratch overflow");
            return;
        };
        if state.scratch.len() < scratch_len {
            debug_assert!(
                false,
                "Shrink scratch must be pre-sized with start_with_tile_and_bands()"
            );
            return;
        }

        {
            let mut intermediate = TileMut::<F>::new(
                intermediate_region,
                input.bands,
                &mut state.scratch[..scratch_len],
            );
            self.horizontal
                .process_region(&mut (), input, &mut intermediate);
        }

        let intermediate = Tile::<F>::new(
            intermediate_region,
            input.bands,
            &state.scratch[..scratch_len],
        );
        self.vertical
            .process_region(&mut state.vertical_scratch, &intermediate, output);
    }
}

#[inline]
fn shrink_u8_2x2(input: &[u8], bands: usize, out_w: usize, out_h: usize, output: &mut [u8]) {
    let input_row_len = out_w * 2 * bands;
    let output_row_len = out_w * bands;

    for y_out in 0..out_h {
        let row0 = &input[(y_out * 2) * input_row_len..(y_out * 2 + 1) * input_row_len];
        let row1 = &input[(y_out * 2 + 1) * input_row_len..(y_out * 2 + 2) * input_row_len];
        let output_row = &mut output[y_out * output_row_len..(y_out + 1) * output_row_len];

        match bands {
            1 => {
                for x_out in 0..out_w {
                    let base = x_out * 2;
                    let top = (u16::from(row0[base]) + u16::from(row0[base + 1]) + 1) >> 1;
                    let bottom = (u16::from(row1[base]) + u16::from(row1[base + 1]) + 1) >> 1;
                    output_row[x_out] = ((top + bottom + 1) >> 1) as u8;
                }
            }
            3 => {
                for x_out in 0..out_w {
                    let base = x_out * 6;
                    let out_base = x_out * 3;
                    for band in 0..3 {
                        let top =
                            (u16::from(row0[base + band]) + u16::from(row0[base + 3 + band]) + 1)
                                >> 1;
                        let bottom =
                            (u16::from(row1[base + band]) + u16::from(row1[base + 3 + band]) + 1)
                                >> 1;
                        output_row[out_base + band] = ((top + bottom + 1) >> 1) as u8;
                    }
                }
            }
            4 => {
                for x_out in 0..out_w {
                    let base = x_out * 8;
                    let out_base = x_out * 4;
                    for band in 0..4 {
                        let top =
                            (u16::from(row0[base + band]) + u16::from(row0[base + 4 + band]) + 1)
                                >> 1;
                        let bottom =
                            (u16::from(row1[base + band]) + u16::from(row1[base + 4 + band]) + 1)
                                >> 1;
                        output_row[out_base + band] = ((top + bottom + 1) >> 1) as u8;
                    }
                }
            }
            _ => {
                for x_out in 0..out_w {
                    let base = x_out * 2 * bands;
                    let out_base = x_out * bands;
                    for band in 0..bands {
                        let top = (u16::from(row0[base + band])
                            + u16::from(row0[base + bands + band])
                            + 1)
                            >> 1;
                        let bottom = (u16::from(row1[base + band])
                            + u16::from(row1[base + bands + band])
                            + 1)
                            >> 1;
                        output_row[out_base + band] = ((top + bottom + 1) >> 1) as u8;
                    }
                }
            }
        }
    }
}

impl<F> ResampleOp for Shrink<F>
where
    F: BandFormat,
    F::Sample: ShrinkSample + bytemuck::Pod,
{
    fn output_width(&self, input_w: u32) -> u32 {
        input_w / self.hshrink
    }

    fn output_height(&self, input_h: u32) -> u32 {
        input_h / self.vshrink
    }
}

pub(crate) struct ShrinkBridge<F: BandFormat>
where
    F::Sample: bytemuck::Pod + ShrinkSample,
{
    inner: OperationBridge<Shrink<F>>,
}

impl<F: BandFormat> ShrinkBridge<F>
where
    F::Sample: bytemuck::Pod + ShrinkSample,
{
    pub fn new(hshrink: usize, vshrink: usize, bands: u32) -> Result<Self, BuildError> {
        Ok(Self {
            inner: OperationBridge::new(Shrink::new(hshrink, vshrink)?, bands),
        })
    }
}

impl<F: BandFormat> DynOperation for ShrinkBridge<F>
where
    F::Sample: bytemuck::Pod + ShrinkSample + Send,
{
    fn input_format(&self) -> crate::domain::format::BandFormatId {
        self.inner.input_format()
    }

    fn output_format(&self) -> crate::domain::format::BandFormatId {
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

    fn output_width(&self, input_w: u32) -> u32 {
        self.inner.op.output_width(input_w)
    }

    fn output_height(&self, input_h: u32) -> u32 {
        self.inner.op.output_height(input_h)
    }

    fn dyn_start(&self) -> Box<dyn std::any::Any + Send> {
        self.inner.dyn_start()
    }

    fn dyn_start_with_tile(&self, tile_w: u32, tile_h: u32) -> Box<dyn std::any::Any + Send> {
        self.inner.dyn_start_with_tile(tile_w, tile_h)
    }

    fn dyn_start_with_tile_and_bands(
        &self,
        tile_w: u32,
        tile_h: u32,
        bands: u32,
    ) -> Box<dyn std::any::Any + Send> {
        self.inner
            .dyn_start_with_tile_and_bands(tile_w, tile_h, bands)
    }

    fn validate_region_contract(
        &self,
        input_region: Region,
        input_bands: u32,
        output_region: Region,
        output_bands: u32,
    ) -> Result<(), crate::domain::error::ViprsError> {
        self.inner
            .validate_region_contract(input_region, input_bands, output_region, output_bands)
    }

    fn dyn_process_region(
        &self,
        state: &mut dyn std::any::Any,
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
    use crate::{
        adapters::sources::memory::MemorySource,
        domain::{
            error::{BuildError, ViprsError},
            format::{BandFormatId, U8},
            image::{Region, Tile, TileMut},
            op::DynOperation,
            resample::ResampleOp,
        },
        ports::source::ImageSource,
    };
    use proptest::prelude::*;

    fn run_shrink<F>(
        hshrink: usize,
        vshrink: usize,
        input_data: &[F::Sample],
        in_w: u32,
        in_h: u32,
        bands: u32,
    ) -> (Region, Vec<F::Sample>)
    where
        F: BandFormat,
        F::Sample: ShrinkSample + bytemuck::Pod,
    {
        let source = MemorySource::<F>::new(in_w, in_h, bands, input_data.to_vec()).unwrap();
        let op = Shrink::<F>::new(hshrink, vshrink).unwrap();
        let out_region = Region::new(0, 0, op.output_width(in_w), op.output_height(in_h));
        let in_region = op.required_input_region(&out_region);
        let mut prepared_input =
            vec![F::Sample::from_f64_clamped(0.0); in_region.pixel_count() * bands as usize];
        source
            .read_region(in_region, bytemuck::cast_slice_mut(&mut prepared_input))
            .unwrap();
        let mut output_data =
            vec![F::Sample::from_f64_clamped(0.0); out_region.pixel_count() * bands as usize];
        let input = Tile::<F>::new(in_region, bands, &prepared_input);
        let mut output = TileMut::<F>::new(out_region, bands, &mut output_data);
        let mut state = op.start_with_tile_and_bands(out_region.width, out_region.height, bands);
        op.process_region(&mut state, &input, &mut output);
        (out_region, output_data)
    }

    #[test]
    fn shrink_sequences_horizontal_then_vertical() {
        let input = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let (region, output) = run_shrink::<U8>(2, 2, &input, 4, 4, 1);
        assert_eq!(region, Region::new(0, 0, 2, 2));
        assert_eq!(output, vec![4u8, 6, 12, 14]);
    }

    #[test]
    fn shrink_factor_one_is_identity() {
        let input: Vec<u8> = (0u8..12).collect();
        let (region, output) = run_shrink::<U8>(1, 1, &input, 4, 3, 1);
        assert_eq!(region, Region::new(0, 0, 4, 3));
        assert_eq!(output, input);
    }

    #[test]
    fn shrink_output_dimensions_follow_floor_division() {
        let op = Shrink::<U8>::new(2, 3).unwrap();
        assert_eq!(op.output_size(5, 7), (2, 2));
    }

    #[test]
    fn shrink_factor_3_matches_libvips_floor_geometry_for_odd_and_even_inputs() {
        let op = Shrink::<U8>::new(3, 3).unwrap();
        assert_eq!(op.output_size(61, 53), (20, 17));
        assert_eq!(op.output_size(60, 54), (20, 18));
    }

    #[test]
    fn shrink_factor_5_matches_libvips_floor_geometry_for_odd_and_even_inputs() {
        let op = Shrink::<U8>::new(5, 5).unwrap();
        assert_eq!(op.output_size(27, 23), (5, 4));
        assert_eq!(op.output_size(28, 24), (5, 4));
    }

    #[test]
    fn shrink_required_input_region_matches_combined_factors() {
        let op = Shrink::<U8>::new(3, 2).unwrap();
        assert_eq!(
            op.required_input_region(&Region::new(4, 5, 6, 7)),
            Region::new(12, 10, 18, 14)
        );
    }

    #[test]
    fn start_with_tile_and_bands_presizes_scratch_from_node_spec() {
        let op = Shrink::<U8>::new(2, 3).unwrap();
        let tile_w = 8;
        let tile_h = 5;
        let bands = 4;
        let expected = op.scratch_len_for_tile(tile_w, tile_h, bands);
        let state = op.start_with_tile_and_bands(tile_w, tile_h, bands);
        assert_eq!(state.scratch.len(), expected);
    }

    #[test]
    fn shrink_node_spec_saturates_huge_factors() {
        let op = Shrink::<U8>::new(u32::MAX as usize, u32::MAX as usize).unwrap();
        let spec = op.node_spec(3, 2);

        assert_eq!(spec.input_tile_w, u32::MAX);
        assert_eq!(spec.input_tile_h, u32::MAX);
        assert_eq!(spec.output_tile_w, 3);
        assert_eq!(spec.output_tile_h, 2);
    }

    #[test]
    fn shrink_2x2_u8_presizes_no_intermediate_scratch() {
        let state = Shrink::<U8>::new(2, 2)
            .unwrap()
            .start_with_tile_and_bands(8, 5, 3);
        assert!(state.scratch.is_empty());
        assert!(state.vertical_scratch.is_empty());
    }

    #[test]
    fn shrink_2x2_u8_matches_separable_rounding() {
        let input = vec![1u8, 1, 1, 2];
        let (region, output) = run_shrink::<U8>(2, 2, &input, 2, 2, 1);
        assert_eq!(region, Region::new(0, 0, 1, 1));
        assert_eq!(output, vec![2u8]);
    }

    #[test]
    fn shrink_boundary_tile_drops_incomplete_trailing_rows_and_columns() {
        let input = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9];
        let (region, output) = run_shrink::<U8>(2, 2, &input, 3, 3, 1);
        assert_eq!(region, Region::new(0, 0, 1, 1));
        assert_eq!(output, vec![4u8]);
    }

    #[test]
    fn shrink_u8_2x2_covers_specialized_and_generic_band_counts() {
        let mut rgb_output = vec![0u8; 6];
        shrink_u8_2x2(
            &[
                10, 20, 30, 20, 30, 40, 30, 40, 50, 40, 50, 60, 100, 110, 120, 110, 120, 130, 120,
                130, 140, 130, 140, 150,
            ],
            3,
            2,
            1,
            &mut rgb_output,
        );
        assert_eq!(rgb_output, vec![60, 70, 80, 80, 90, 100]);

        let mut rgba_output = vec![0u8; 4];
        shrink_u8_2x2(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            4,
            1,
            1,
            &mut rgba_output,
        );
        assert_eq!(rgba_output, vec![7, 8, 9, 10]);

        let mut generic_output = vec![0u8; 2];
        shrink_u8_2x2(&[1, 10, 3, 30, 5, 50, 7, 70], 2, 1, 1, &mut generic_output);
        assert_eq!(generic_output, vec![4, 40]);
    }

    #[test]
    fn shrink_start_fast_path_keeps_scratch_empty() {
        let state = Shrink::<U8>::new(2, 2).unwrap().start();
        assert!(state.scratch.is_empty());
        assert!(state.vertical_scratch.is_empty());
    }

    #[test]
    fn shrink_new_rejects_zero_horizontal_factor() {
        let result = Shrink::<U8>::new(0, 1);
        assert!(matches!(
            result,
            Err(BuildError::SourceHint {
                context: "shrink",
                message,
            }) if message == "hshrink must be in 1..=u32::MAX"
        ));
    }

    #[test]
    fn validate_region_contract_rejects_overflowing_scratch() {
        let op = Shrink::<U8>::new(1, 1).unwrap();
        let huge = Region::new(0, 0, u32::MAX, u32::MAX);

        let err = op.validate_region_contract(huge, 2, huge, 2).unwrap_err();

        assert!(matches!(
            err,
            ViprsError::ImageTooLarge {
                width: u32::MAX,
                height: u32::MAX,
                bands: 2,
                ..
            }
        ));
    }

    #[test]
    fn shrink_new_rejects_zero_vertical_factor() {
        let result = Shrink::<U8>::new(1, 0);
        assert!(matches!(
            result,
            Err(BuildError::SourceHint {
                context: "shrink",
                message,
            }) if message == "vshrink must be in 1..=u32::MAX"
        ));
    }

    #[test]
    fn shrink_new_accepts_u32_max_factor() {
        let shrink = Shrink::<U8>::new(u32::MAX as usize, 1).unwrap();
        assert_eq!(shrink.hshrink, u32::MAX);
        assert_eq!(shrink.vshrink, 1);
    }

    #[test]
    fn shrink_new_rejects_factor_above_u32_max() {
        let Some(overflow_factor) = (u32::MAX as usize).checked_add(1) else {
            return;
        };
        let result = Shrink::<U8>::new(overflow_factor, 1);
        assert!(matches!(
            result,
            Err(BuildError::SourceHint {
                context: "shrink",
                message,
            }) if message == "hshrink must be in 1..=u32::MAX"
        ));
    }

    #[test]
    fn shrink_bridge_new_rejects_zero_horizontal_factor() {
        let result = ShrinkBridge::<U8>::new(0, 1, 1);
        assert!(matches!(
            result,
            Err(BuildError::SourceHint {
                context: "shrink",
                message,
            }) if message == "hshrink must be in 1..=u32::MAX"
        ));
    }

    #[test]
    fn shrink_bridge_new_rejects_zero_vertical_factor() {
        let result = ShrinkBridge::<U8>::new(1, 0, 1);
        assert!(matches!(
            result,
            Err(BuildError::SourceHint {
                context: "shrink",
                message,
            }) if message == "vshrink must be in 1..=u32::MAX"
        ));
    }

    #[test]
    fn shrink_bridge_new_accepts_u32_max_factor() {
        let bridge = ShrinkBridge::<U8>::new(u32::MAX as usize, 1, 3).unwrap();
        assert_eq!(bridge.output_width(u32::MAX), 1);
        assert_eq!(bridge.output_height(7), 7);
        assert_eq!(bridge.bands(), 3);
    }

    #[test]
    fn shrink_bridge_new_rejects_factor_above_u32_max() {
        let Some(overflow_factor) = (u32::MAX as usize).checked_add(1) else {
            return;
        };
        let result = ShrinkBridge::<U8>::new(1, overflow_factor, 1);
        assert!(matches!(
            result,
            Err(BuildError::SourceHint {
                context: "shrink",
                message,
            }) if message == "vshrink must be in 1..=u32::MAX"
        ));
    }

    #[test]
    fn shrink_bridge_exposes_dyn_operation_contract() {
        let bridge = ShrinkBridge::<U8>::new(2, 2, 1).unwrap();
        assert_eq!(bridge.input_format(), BandFormatId::U8);
        assert_eq!(bridge.output_format(), BandFormatId::U8);
        assert_eq!(bridge.bands(), 1);
        assert_eq!(bridge.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(bridge.output_width(4), 2);
        assert_eq!(bridge.output_height(4), 2);
        assert_eq!(
            bridge.node_spec(2, 2),
            Shrink::<U8>::new(2, 2).unwrap().node_spec(2, 2)
        );

        let source = MemorySource::<U8>::new(
            4,
            4,
            1,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        )
        .unwrap();
        let out_region = Region::new(0, 0, 2, 2);
        let input_region = bridge.required_input_region(&out_region);
        let mut input_bytes = vec![0u8; input_region.pixel_count()];
        source.read_region(input_region, &mut input_bytes).unwrap();
        let mut output_bytes = vec![0u8; out_region.pixel_count()];
        let mut state =
            bridge.dyn_start_with_tile_and_bands(out_region.width, out_region.height, 1);
        bridge.dyn_process_region(
            &mut *state,
            &input_bytes,
            &mut output_bytes,
            input_region,
            out_region,
        );
        assert_eq!(output_bytes, vec![4u8, 6, 12, 14]);
    }

    proptest! {
        #[test]
        fn shrink_factor_one_is_identity_prop(
            (width, height, bands, pixels) in (1u32..=16, 1u32..=16, 1u32..=4).prop_flat_map(|(width, height, bands)| {
                (
                    Just(width),
                    Just(height),
                    Just(bands),
                    prop::collection::vec(any::<u8>(), (width * height * bands) as usize),
                )
            }),
        ) {
            let (_, output) = run_shrink::<U8>(1, 1, &pixels, width, height, bands);
            prop_assert_eq!(output, pixels);
        }
    }
}

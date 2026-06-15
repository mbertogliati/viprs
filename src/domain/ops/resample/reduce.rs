use std::marker::PhantomData;

use crate::domain::{
    error::ViprsError,
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    kernel::InterpolationKernel,
    op::{DynOperation, NodeSpec, Op, OperationBridge},
    resample::{ReduceConfig, ResampleOp},
};

use super::{
    reduce_common::{
        ReduceKernel, clamp_axis, validate_reduce_factors, validate_reduce_kernel,
        validate_reduce_tap_limits,
    },
    sample_conv::{FromF64, ToF64},
};

/// Represents a reduce state.
pub struct ReduceState<T> {
    scratch: Vec<T>,
}

/// Convenience wrapper that presents `reduceh |> reducev` as one operation.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::resample::reduce::ReduceOp;
///
/// let op = ReduceOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct ReduceOp<F: BandFormat> {
    /// Stores the `h_factor` value for this item.
    pub h_factor: f64,
    /// Stores the `v_factor` value for this item.
    pub v_factor: f64,
    /// Stores the `kernel` value for this item.
    pub kernel: InterpolationKernel,
    h_filter: ReduceKernel,
    v_filter: ReduceKernel,
    _phantom: PhantomData<F>,
}

impl<F: BandFormat + Send + Sync> ReduceOp<F> {
    /// Creates a new `ReduceOp`.
    pub fn new(
        h_factor: f64,
        v_factor: f64,
        kernel: InterpolationKernel,
    ) -> Result<Self, crate::domain::error::BuildError> {
        validate_reduce_factors(h_factor, v_factor)?;
        validate_reduce_kernel("reduce", kernel)?;
        validate_reduce_tap_limits(h_factor, v_factor, kernel)?;
        Ok(Self {
            h_factor,
            v_factor,
            kernel,
            h_filter: ReduceKernel::new(h_factor, kernel)?,
            v_filter: ReduceKernel::new(v_factor, kernel)?,
            _phantom: PhantomData,
        })
    }

    #[inline]
    const fn h_config(&self) -> ReduceConfig {
        self.h_filter.config()
    }

    #[inline]
    const fn v_config(&self) -> ReduceConfig {
        self.v_filter.config()
    }

    #[inline]
    fn scratch_len_for_tile(&self, tile_w: u32, tile_h: u32, bands: u32) -> usize
    where
        F::Sample: ToF64 + FromF64,
    {
        let spec = self.node_spec(tile_w, tile_h);
        spec.output_tile_w as usize * spec.input_tile_h as usize * bands as usize
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
                details: "reduce scratch exceeds addressable memory",
            })
    }

    #[inline]
    fn process_horizontal(&self, input: &Tile<F>, output: &mut TileMut<F>)
    where
        F::Sample: ToF64 + FromF64,
    {
        let out_w = output.region.width as usize;
        let out_h = output.region.height as usize;
        let in_w = input.region.width as usize;
        let bands = input.bands as usize;

        for y in 0..out_h {
            for x_out in 0..out_w {
                let source_x = self
                    .h_filter
                    .source_position(f64::from(output.region.x) + x_out as f64);
                let (start_x, weights) = self.h_filter.taps_for_f64(source_x);

                for band in 0..bands {
                    let mut acc = 0.0;
                    for (tap, weight) in weights.iter().copied().enumerate() {
                        let tile_x = clamp_axis(start_x + tap as i64, input.region.x, in_w);
                        let idx = (y * in_w + tile_x) * bands + band;
                        acc = input.data[idx].to_f64().mul_add(weight, acc);
                    }

                    let out_idx = (y * out_w + x_out) * bands + band;
                    output.data[out_idx] = F::Sample::from_f64(acc);
                }
            }
        }
    }

    #[inline]
    fn process_vertical(&self, input: &Tile<F>, output: &mut TileMut<F>)
    where
        F::Sample: ToF64 + FromF64,
    {
        let out_w = output.region.width as usize;
        let out_h = output.region.height as usize;
        let in_w = input.region.width as usize;
        let in_h = input.region.height as usize;
        let bands = input.bands as usize;

        for y_out in 0..out_h {
            let source_y = self
                .v_filter
                .source_position(f64::from(output.region.y) + y_out as f64);
            let (start_y, weights) = self.v_filter.taps_for_f64(source_y);

            for x in 0..out_w {
                for band in 0..bands {
                    let mut acc = 0.0;
                    for (tap, weight) in weights.iter().copied().enumerate() {
                        let tile_y = clamp_axis(start_y + tap as i64, input.region.y, in_h);
                        let idx = (tile_y * in_w + x) * bands + band;
                        acc = input.data[idx].to_f64().mul_add(weight, acc);
                    }

                    let out_idx = (y_out * out_w + x) * bands + band;
                    output.data[out_idx] = F::Sample::from_f64(acc);
                }
            }
        }
    }
}

impl<F: BandFormat> Op for ReduceOp<F>
where
    F::Sample: ToF64 + FromF64,
{
    type Input = F;
    type Output = F;
    type State = ReduceState<F::Sample>;

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        let intermediate = self.v_config().required_input_region_v(output);
        self.h_config().required_input_region_h(&intermediate)
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        let h_cfg = self.h_config();
        let v_cfg = self.v_config();
        NodeSpec {
            input_tile_w: (f64::from(tile_w) * h_cfg.factor).ceil() as u32 + h_cfg.taps,
            input_tile_h: (f64::from(tile_h) * v_cfg.factor).ceil() as u32 + v_cfg.taps,
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    fn start(&self) -> Self::State {
        ReduceState {
            scratch: Vec::new(),
        }
    }

    fn start_with_tile_and_bands(&self, tile_w: u32, tile_h: u32, bands: u32) -> Self::State {
        let scratch_len = self.scratch_len_for_tile(tile_w, tile_h, bands);
        ReduceState {
            scratch: vec![F::Sample::from_f64(0.0); scratch_len],
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
        let intermediate_region = self.v_config().required_input_region_v(&output_region);
        Self::checked_scratch_len(intermediate_region, input_bands).map(|_| ())
    }

    #[inline]
    fn process_region(&self, state: &mut Self::State, input: &Tile<F>, output: &mut TileMut<F>) {
        let intermediate_region = self.v_config().required_input_region_v(&output.region);
        let Ok(scratch_len) = Self::checked_scratch_len(intermediate_region, input.bands) else {
            debug_assert!(false, "ReduceOp scratch overflow");
            return;
        };
        if state.scratch.len() < scratch_len {
            debug_assert!(
                false,
                "ReduceOp scratch must be pre-sized with start_with_tile_and_bands()"
            );
            return;
        }
        let scratch = &mut state.scratch[..scratch_len];

        {
            let mut intermediate = TileMut::<F>::new(intermediate_region, input.bands, scratch);
            self.process_horizontal(input, &mut intermediate);
        }

        let intermediate = Tile::<F>::new(
            intermediate_region,
            input.bands,
            &state.scratch[..scratch_len],
        );
        self.process_vertical(&intermediate, output);
    }
}

impl<F: BandFormat> ResampleOp for ReduceOp<F>
where
    F::Sample: ToF64 + FromF64,
{
    fn output_width(&self, input_w: u32) -> u32 {
        self.h_config().output_width(input_w)
    }

    fn output_height(&self, input_h: u32) -> u32 {
        self.v_config().output_height(input_h)
    }
}

pub(crate) struct ReduceBridge<F: BandFormat>
where
    F::Sample: bytemuck::Pod + ToF64 + FromF64,
{
    inner: OperationBridge<ReduceOp<F>>,
}

impl<F: BandFormat> ReduceBridge<F>
where
    F::Sample: bytemuck::Pod + ToF64 + FromF64,
{
    pub fn new(
        h_factor: f64,
        v_factor: f64,
        kernel: InterpolationKernel,
        bands: u32,
    ) -> Result<Self, crate::domain::error::BuildError> {
        Ok(Self {
            inner: OperationBridge::new(ReduceOp::new(h_factor, v_factor, kernel)?, bands),
        })
    }
}

impl<F: BandFormat> DynOperation for ReduceBridge<F>
where
    F::Sample: bytemuck::Pod + ToF64 + FromF64 + Send,
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
    use super::super::{reduceh::ReduceH, reducev::ReduceV};
    use super::*;
    use crate::{
        adapters::{
            pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
            sinks::memory::MemorySink, sources::memory::MemorySource,
        },
        domain::{
            error::ViprsError,
            format::U8,
            image::{Image, Region, Tile, TileMut},
            resample::ResampleOp,
        },
        ports::scheduler::TileScheduler,
    };
    use proptest::prelude::*;

    fn patterned_u8_image(width: u32, height: u32, bands: u32) -> Image<U8> {
        let mut pixels = Vec::with_capacity(width as usize * height as usize * bands as usize);
        for y in 0..height {
            for x in 0..width {
                for band in 0..bands {
                    pixels.push(((x * 31 + y * 17 + band * 53 + 11) % 256) as u8);
                }
            }
        }

        Image::from_buffer(width, height, bands, pixels).unwrap()
    }

    fn read_region_with_edge_copy(image: &Image<U8>, region: Region) -> Vec<u8> {
        let width = image.width() as i64;
        let height = image.height() as i64;
        let bands = image.bands() as usize;
        let mut prepared = vec![0u8; region.pixel_count() * bands];

        for y in 0..region.height as usize {
            let source_y = (region.y as i64 + y as i64).clamp(0, height - 1) as usize;
            for x in 0..region.width as usize {
                let source_x = (region.x as i64 + x as i64).clamp(0, width - 1) as usize;
                let source_idx = (source_y * image.width() as usize + source_x) * bands;
                let target_idx = (y * region.width as usize + x) * bands;
                prepared[target_idx..target_idx + bands]
                    .copy_from_slice(&image.pixels()[source_idx..source_idx + bands]);
            }
        }

        prepared
    }

    fn run_reduce(image: &Image<U8>, h_factor: f64, v_factor: f64) -> (Region, Vec<u8>) {
        let op = ReduceOp::<U8>::new(h_factor, v_factor, InterpolationKernel::Bilinear).unwrap();
        let out_region = Region::new(
            0,
            0,
            op.output_width(image.width()),
            op.output_height(image.height()),
        );
        let in_region = op.required_input_region(&out_region);
        let prepared_input = read_region_with_edge_copy(image, in_region);
        let mut output_data = vec![0u8; out_region.pixel_count() * image.bands() as usize];
        let input = Tile::<U8>::new(in_region, image.bands(), &prepared_input);
        let mut output = TileMut::<U8>::new(out_region, image.bands(), &mut output_data);
        let mut state =
            op.start_with_tile_and_bands(out_region.width, out_region.height, image.bands());
        op.process_region(&mut state, &input, &mut output);
        (out_region, output_data)
    }

    fn run_reduce_reference(image: &Image<U8>, h_factor: f64, v_factor: f64) -> (Region, Vec<u8>) {
        let h_op = ReduceH::<U8>::new(h_factor, InterpolationKernel::Bilinear).unwrap();
        let h_out_region = Region::new(0, 0, h_op.output_width(image.width()), image.height());
        let h_in_region = h_op.required_input_region(&h_out_region);
        let h_input = read_region_with_edge_copy(image, h_in_region);
        let mut h_output = vec![0u8; h_out_region.pixel_count() * image.bands() as usize];
        let h_tile = Tile::<U8>::new(h_in_region, image.bands(), &h_input);
        let mut h_target = TileMut::<U8>::new(h_out_region, image.bands(), &mut h_output);
        let mut h_state = h_op.start_with_tile(h_out_region.width, h_out_region.height);
        h_op.process_region(&mut h_state, &h_tile, &mut h_target);

        let intermediate = Image::<U8>::from_buffer(
            h_out_region.width,
            h_out_region.height,
            image.bands(),
            h_output,
        )
        .unwrap();

        let v_op = ReduceV::<U8>::new(v_factor, InterpolationKernel::Bilinear).unwrap();
        let v_out_region = Region::new(
            0,
            0,
            intermediate.width(),
            v_op.output_height(intermediate.height()),
        );
        let v_in_region = v_op.required_input_region(&v_out_region);
        let v_input = read_region_with_edge_copy(&intermediate, v_in_region);
        let mut v_output = vec![0u8; v_out_region.pixel_count() * intermediate.bands() as usize];
        let v_tile = Tile::<U8>::new(v_in_region, intermediate.bands(), &v_input);
        let mut v_target = TileMut::<U8>::new(v_out_region, intermediate.bands(), &mut v_output);
        let mut v_state = v_op.start_with_tile(v_out_region.width, v_out_region.height);
        v_op.process_region(&mut v_state, &v_tile, &mut v_target);

        (v_out_region, v_output)
    }

    fn assert_pixels_close(actual: &[u8], expected: &[u8]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                actual.abs_diff(*expected) <= 1,
                "pixel {index} differed: actual={actual}, expected={expected}"
            );
        }
    }

    prop_compose! {
        fn mixed_reduce_case()(
            width in 2u32..=6,
            height in 2u32..=6,
            bands in 1u32..=3,
            h_factor in prop_oneof![Just(1.5), Just(2.0), Just(3.0)],
            v_factor in prop_oneof![Just(1.5), Just(2.0), Just(3.0)],
        )(
            input in prop::collection::vec(any::<u8>(), (width * height * bands) as usize),
            width in Just(width),
            height in Just(height),
            bands in Just(bands),
            h_factor in Just(h_factor),
            v_factor in Just(v_factor),
        ) -> (u32, u32, u32, f64, f64, Vec<u8>) {
            (width, height, bands, h_factor, v_factor, input)
        }
    }

    #[test]
    fn factor_one_is_identity() {
        let image = Image::<U8>::from_buffer(4, 3, 1, (0u8..12).collect()).unwrap();
        let (region, output) = run_reduce(&image, 1.0, 1.0);
        assert_eq!(region, Region::new(0, 0, image.width(), image.height()));
        assert_eq!(output, image.pixels());
    }

    #[test]
    fn factor_two_halves_dimensions() {
        let op = ReduceOp::<U8>::new(2.0, 2.0, InterpolationKernel::Bilinear).unwrap();
        assert_eq!(op.output_width(8), 4);
        assert_eq!(op.output_height(6), 3);
    }

    #[test]
    fn start_with_tile_and_bands_presizes_scratch_from_node_spec() {
        let op = ReduceOp::<U8>::new(2.0, 1.5, InterpolationKernel::Bilinear).unwrap();
        let tile_w = 8;
        let tile_h = 5;
        let bands = 3;
        let expected = op.scratch_len_for_tile(tile_w, tile_h, bands);
        let state = op.start_with_tile_and_bands(tile_w, tile_h, bands);
        assert_eq!(state.scratch.len(), expected);
    }

    #[test]
    fn process_region_reuses_presized_scratch_for_boundary_tiles() {
        let op = ReduceOp::<U8>::new(2.0, 2.0, InterpolationKernel::Bilinear).unwrap();
        let bands = 1;
        let mut state = op.start_with_tile_and_bands(8, 8, bands);
        let initial_len = state.scratch.len();

        let full_out = Region::new(0, 0, 8, 8);
        let full_in = op.required_input_region(&full_out);
        let full_in_data = vec![11u8; full_in.pixel_count()];
        let mut full_out_data = vec![0u8; full_out.pixel_count()];
        op.process_region(
            &mut state,
            &Tile::<U8>::new(full_in, bands, &full_in_data),
            &mut TileMut::<U8>::new(full_out, bands, &mut full_out_data),
        );
        assert_eq!(state.scratch.len(), initial_len);

        let edge_out = Region::new(0, 5, 8, 3);
        let edge_in = op.required_input_region(&edge_out);
        let edge_in_data = vec![29u8; edge_in.pixel_count()];
        let mut edge_out_data = vec![0u8; edge_out.pixel_count()];
        op.process_region(
            &mut state,
            &Tile::<U8>::new(edge_in, bands, &edge_in_data),
            &mut TileMut::<U8>::new(edge_out, bands, &mut edge_out_data),
        );
        assert_eq!(state.scratch.len(), initial_len);
    }

    #[test]
    fn mixed_multiband_pixels_match_separable_reference_at_factor_two() {
        let image = Image::<U8>::from_buffer(
            4,
            4,
            3,
            vec![
                0, 10, 20, 30, 45, 60, 90, 60, 30, 120, 90, 40, 15, 35, 55, 45, 75, 105, 95, 65,
                35, 125, 95, 45, 25, 50, 75, 55, 85, 115, 105, 70, 40, 135, 100, 50, 35, 60, 85,
                65, 95, 125, 115, 80, 45, 145, 105, 55,
            ],
        )
        .unwrap();

        let (region, output) = run_reduce(&image, 2.0, 2.0);
        let (expected_region, expected) = run_reduce_reference(&image, 2.0, 2.0);

        assert_eq!(region, Region::new(0, 0, 2, 2));
        assert_eq!(region, expected_region);
        assert_pixels_close(&output, &expected);
    }

    #[test]
    fn mixed_pixels_match_separable_reference_for_anisotropic_shrink() {
        let image = Image::<U8>::from_buffer(
            5,
            4,
            1,
            vec![
                0, 20, 80, 120, 200, 15, 35, 95, 135, 215, 25, 55, 110, 150, 230, 40, 70, 125, 165,
                245,
            ],
        )
        .unwrap();

        let (region, output) = run_reduce(&image, 1.5, 2.0);
        let (expected_region, expected) = run_reduce_reference(&image, 1.5, 2.0);

        assert_eq!(region, Region::new(0, 0, 3, 2));
        assert_eq!(region, expected_region);
        assert_pixels_close(&output, &expected);
    }

    #[test]
    fn reduce_bridge_presizes_scratch_for_fractional_factors() {
        let bridge = ReduceBridge::<U8>::new(1.5, 2.5, InterpolationKernel::Lanczos3, 3).unwrap();
        let state = bridge.dyn_start_with_tile_and_bands(64, 32, 3);
        let state = state
            .downcast::<ReduceState<u8>>()
            .expect("ReduceBridge should allocate ReduceState");

        assert_eq!(
            state.scratch.len(),
            ReduceOp::<U8>::new(1.5, 2.5, InterpolationKernel::Lanczos3)
                .unwrap()
                .scratch_len_for_tile(64, 32, 3)
        );
    }

    #[test]
    fn fractional_reduce_pipeline_matches_declared_dimensions() {
        let image = patterned_u8_image(777, 333, 3);
        let pipeline = PipelineBuilder::from_source(
            MemorySource::<U8>::new(
                image.width(),
                image.height(),
                image.bands(),
                image.pixels().to_vec(),
            )
            .unwrap(),
        )
        .reduce(1.5, 2.5, InterpolationKernel::Lanczos3)
        .unwrap()
        .build()
        .unwrap();

        let expected_len =
            pipeline.width as usize * pipeline.height as usize * pipeline.output_bands as usize;
        let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
        RayonScheduler::new(2)
            .unwrap()
            .run(&pipeline, &mut sink)
            .unwrap();

        assert_eq!(sink.into_buffer().len(), expected_len);
    }

    #[test]
    fn validate_region_contract_rejects_overflowing_scratch() {
        let huge = Region::new(0, 0, u32::MAX, u32::MAX);

        let err = ReduceOp::<U8>::checked_scratch_len(huge, 2).unwrap_err();

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

    proptest! {
        #[test]
        fn factor_one_preserves_uniform_tiles(
            width in 1u32..=8,
            height in 1u32..=8,
            value in any::<u8>(),
        ) {
            let image = Image::<U8>::from_buffer(width, height, 1, vec![value; (width * height) as usize]).unwrap();
            let (_, output) = run_reduce(&image, 1.0, 1.0);
            prop_assert_eq!(output, image.pixels());
        }

        #[test]
        fn shrink_above_one_matches_separable_reference_for_mixed_pixels(
            (width, height, bands, h_factor, v_factor, input) in mixed_reduce_case(),
        ) {
            prop_assume!(input.iter().any(|&value| value != input[0]));

            let image = Image::<U8>::from_buffer(width, height, bands, input).unwrap();
            let (region, output) = run_reduce(&image, h_factor, v_factor);
            let (expected_region, expected) = run_reduce_reference(&image, h_factor, v_factor);

            prop_assert_eq!(region, expected_region);
            prop_assert!(
                output.iter().zip(expected.iter()).all(|(actual, expected)| actual.abs_diff(*expected) <= 2),
                "ReduceOp diverged from separable reference beyond +/-2: output={output:?} expected={expected:?}",
            );
        }
    }
}

#![allow(missing_docs)]
// REASON: these bridge helpers are public only for cross-crate workspace wiring, not end-user API.

//! Vertical reduction (downscale along the y-axis).
//!
//! `ReduceV` mirrors libvips `reducev`: the op precomputes 64 subpixel
//! coefficient tables up front and reuses them during column filtering.

use std::marker::PhantomData;

use viprs_core::{
    error::BuildError,
    format::{BandFormat, BandFormatId},
    image::{DemandHint, Region, Tile, TileMut},
    kernel::InterpolationKernel,
    op::{NodeSpec, Op},
    resample::{ReduceConfig, ResampleOp},
};

use super::{
    reduce_common::{
        ReduceKernel, validate_reduce_factors, validate_reduce_kernel, validate_reduce_tap_limits,
    },
    reduce_simd::{self, reduce_v_scalar},
    sample_conv::ReduceSample,
};

/// Vertical 1-D reduction by a rational factor using a precomputed kernel table.
pub struct ReduceV<F: BandFormat> {
    filter: ReduceKernel,
    _format: PhantomData<F>,
}

/// Represents a reduce v state.
pub struct ReduceVState {
    starts: Vec<i64>,
    phases: Vec<u8>,
}

impl<F: BandFormat + Send + Sync> ReduceV<F> {
    /// Create a new `ReduceV` that reduces the image height by `factor`.
    pub fn new(factor: f64, kernel: InterpolationKernel) -> Result<Self, BuildError> {
        validate_reduce_factors(1.0, factor)?;
        validate_reduce_kernel("reducev", kernel)?;
        validate_reduce_tap_limits(1.0, factor, kernel)?;
        Ok(Self {
            filter: ReduceKernel::new(factor, kernel)?,
            _format: PhantomData,
        })
    }

    /// Bind the full input height so the resampling centre matches libvips.
    #[must_use]
    pub fn with_input_height(mut self, input_h: u32) -> Self {
        self.filter.bind_input_len(input_h);
        self
    }

    #[inline]
    const fn config(&self) -> ReduceConfig {
        self.filter.config()
    }
}

impl<F: BandFormat> Op for ReduceV<F>
where
    F::Sample: ReduceSample,
{
    type Input = F;
    type Output = F;
    type State = ReduceVState;

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        self.filter.required_input_region_v(output)
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        self.config().node_spec_v(tile_w, tile_h)
    }

    fn start(&self) -> Self::State {
        ReduceVState {
            starts: Vec::new(),
            phases: Vec::new(),
        }
    }

    fn start_with_tile(&self, _tile_w: u32, tile_h: u32) -> Self::State {
        ReduceVState {
            starts: vec![0; tile_h as usize],
            phases: vec![0; tile_h as usize],
        }
    }

    #[inline]
    fn process_region(&self, state: &mut Self::State, input: &Tile<F>, output: &mut TileMut<F>) {
        if F::ID == BandFormatId::U8 {
            let out_h = output.region.height as usize;
            if state.starts.len() < out_h {
                state.starts.resize(out_h, 0);
            }
            if state.phases.len() < out_h {
                state.phases.resize(out_h, 0);
            }
            let starts = &mut state.starts[..out_h];
            let phases = &mut state.phases[..out_h];
            let step = self.filter.config().factor;
            let mut source_y = self.filter.source_position(f64::from(output.region.y));
            for y_out in 0..out_h {
                let (start_y, phase) = self.filter.plan_i16(source_y);
                starts[y_out] = start_y;
                phases[y_out] = phase as u8;
                source_y += step;
            }
            reduce_simd::reduce_v_u8(
                &self.filter,
                &input.region,
                bytemuck::cast_slice(input.data),
                input.bands,
                &output.region,
                starts,
                phases,
                bytemuck::cast_slice_mut(output.data),
            );
            return;
        }
        if F::ID == BandFormatId::U16 {
            reduce_simd::reduce_v_u16(
                &self.filter,
                &input.region,
                bytemuck::cast_slice(input.data),
                input.bands,
                &output.region,
                bytemuck::cast_slice_mut(output.data),
            );
            return;
        }
        if F::ID == BandFormatId::F32 {
            reduce_simd::reduce_v_f32(
                &self.filter,
                &input.region,
                bytemuck::cast_slice(input.data),
                input.bands,
                &output.region,
                bytemuck::cast_slice_mut(output.data),
            );
            return;
        }

        reduce_v_scalar(
            &self.filter,
            &input.region,
            input.data,
            input.bands,
            &output.region,
            output.data,
        );
    }
}

impl<F: BandFormat> ResampleOp for ReduceV<F>
where
    F::Sample: ReduceSample,
{
    fn output_width(&self, input_w: u32) -> u32 {
        input_w
    }

    fn output_height(&self, input_h: u32) -> u32 {
        self.config().output_height(input_h)
    }
}

pub struct ReduceVBridge<F: BandFormat>
where
    F::Sample: bytemuck::Pod + ReduceSample,
{
    inner: viprs_core::op::OperationBridge<ReduceV<F>>,
    demand_hint: DemandHint,
}

impl<F: BandFormat> ReduceVBridge<F>
where
    F::Sample: bytemuck::Pod + ReduceSample,
{
    pub fn new(
        factor: f64,
        kernel: InterpolationKernel,
        bands: u32,
        input_h: u32,
        demand_hint: DemandHint,
    ) -> Result<Self, BuildError> {
        Ok(Self {
            inner: viprs_core::op::OperationBridge::new(
                ReduceV::new(factor, kernel)?.with_input_height(input_h),
                bands,
            ),
            demand_hint,
        })
    }
}

impl<F: BandFormat> viprs_core::op::DynOperation for ReduceVBridge<F>
where
    F::Sample: bytemuck::Pod + ReduceSample + Send,
{
    fn input_format(&self) -> viprs_core::format::BandFormatId {
        self.inner.input_format()
    }

    fn output_format(&self) -> viprs_core::format::BandFormatId {
        self.inner.output_format()
    }

    fn bands(&self) -> u32 {
        self.inner.bands()
    }

    fn demand_hint(&self) -> DemandHint {
        self.demand_hint
    }

    fn required_input_region(&self, output: &Region) -> Region {
        self.inner.required_input_region(output)
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        self.inner.node_spec(tile_w, tile_h)
    }

    fn output_width(&self, input_w: u32) -> u32 {
        input_w
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

#[cfg(all(test, feature = "_integration"))]
mod tests {
    use super::*;
    use crate::{adapters::sources::memory::MemorySource, ports::source::ImageSource};
    use proptest::prelude::*;
    use viprs_core::{
        format::{BandFormatId, F32, U8, U16},
        image::{Region, Tile, TileMut},
        kernel::InterpolationKernel,
        op::DynOperation,
        ops::resample::reduce_simd,
        resample::ResampleOp,
    };

    #[test]
    fn new_rejects_extreme_finite_factor() {
        let result = ReduceV::<U8>::new(f64::MAX / 2.0, InterpolationKernel::Lanczos3);

        assert!(matches!(
            result,
            Err(BuildError::InvalidReduceParameters { .. })
        ));
    }

    fn run_reduce_v_u8(input_data: &[u8], factor: f64, kernel: InterpolationKernel) -> Vec<u8> {
        let source =
            MemorySource::<U8>::new(1, input_data.len() as u32, 1, input_data.to_vec()).unwrap();
        let op = ReduceV::<U8>::new(factor, kernel)
            .unwrap()
            .with_input_height(input_data.len() as u32);
        let out_region = Region::new(0, 0, 1, op.output_height(input_data.len() as u32));
        let in_region = op.required_input_region(&out_region);
        let mut prepared_input = vec![0u8; in_region.pixel_count()];
        source
            .read_region(in_region, bytemuck::cast_slice_mut(&mut prepared_input))
            .unwrap();
        let mut output_data = vec![0u8; out_region.pixel_count()];
        let input = Tile::<U8>::new(in_region, 1, &prepared_input);
        let mut output = TileMut::<U8>::new(out_region, 1, &mut output_data);
        let mut state = op.start_with_tile(out_region.width, out_region.height);
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    fn run_reduce_v_f32(input_data: &[f32], factor: f64, kernel: InterpolationKernel) -> Vec<f32> {
        let source =
            MemorySource::<F32>::new(1, input_data.len() as u32, 1, input_data.to_vec()).unwrap();
        let op = ReduceV::<F32>::new(factor, kernel)
            .unwrap()
            .with_input_height(input_data.len() as u32);
        let out_region = Region::new(0, 0, 1, op.output_height(input_data.len() as u32));
        let in_region = op.required_input_region(&out_region);
        let mut prepared_input = vec![0.0f32; in_region.pixel_count()];
        source
            .read_region(in_region, bytemuck::cast_slice_mut(&mut prepared_input))
            .unwrap();
        let mut output_data = vec![0.0f32; out_region.pixel_count()];
        let input = Tile::<F32>::new(in_region, 1, &prepared_input);
        let mut output = TileMut::<F32>::new(out_region, 1, &mut output_data);
        let mut state = op.start_with_tile(out_region.width, out_region.height);
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    fn run_reduce_v_u16(input_data: &[u16], factor: f64, kernel: InterpolationKernel) -> Vec<u16> {
        let source =
            MemorySource::<U16>::new(1, input_data.len() as u32, 1, input_data.to_vec()).unwrap();
        let op = ReduceV::<U16>::new(factor, kernel)
            .unwrap()
            .with_input_height(input_data.len() as u32);
        let out_region = Region::new(0, 0, 1, op.output_height(input_data.len() as u32));
        let in_region = op.required_input_region(&out_region);
        let mut prepared_input = vec![0u16; in_region.pixel_count()];
        source
            .read_region(in_region, bytemuck::cast_slice_mut(&mut prepared_input))
            .unwrap();
        let mut output_data = vec![0u16; out_region.pixel_count()];
        let input = Tile::<U16>::new(in_region, 1, &prepared_input);
        let mut output = TileMut::<U16>::new(out_region, 1, &mut output_data);
        let mut state = op.start_with_tile(out_region.width, out_region.height);
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    fn run_reduce_v_u8_scalar(
        input_data: &[u8],
        factor: f64,
        kernel: InterpolationKernel,
    ) -> Vec<u8> {
        let source =
            MemorySource::<U8>::new(1, input_data.len() as u32, 1, input_data.to_vec()).unwrap();
        let op = ReduceV::<U8>::new(factor, kernel)
            .unwrap()
            .with_input_height(input_data.len() as u32);
        let out_region = Region::new(0, 0, 1, op.output_height(input_data.len() as u32));
        let in_region = op.required_input_region(&out_region);
        let mut prepared_input = vec![0u8; in_region.pixel_count()];
        source
            .read_region(in_region, bytemuck::cast_slice_mut(&mut prepared_input))
            .unwrap();
        let mut output_data = vec![0u8; out_region.pixel_count()];
        reduce_simd::reduce_v_scalar(
            &op.filter,
            &in_region,
            &prepared_input,
            1,
            &out_region,
            &mut output_data,
        );
        output_data
    }

    fn run_reduce_v_u16_scalar(
        input_data: &[u16],
        factor: f64,
        kernel: InterpolationKernel,
    ) -> Vec<u16> {
        let source =
            MemorySource::<U16>::new(1, input_data.len() as u32, 1, input_data.to_vec()).unwrap();
        let op = ReduceV::<U16>::new(factor, kernel)
            .unwrap()
            .with_input_height(input_data.len() as u32);
        let out_region = Region::new(0, 0, 1, op.output_height(input_data.len() as u32));
        let in_region = op.required_input_region(&out_region);
        let mut prepared_input = vec![0u16; in_region.pixel_count()];
        source
            .read_region(in_region, bytemuck::cast_slice_mut(&mut prepared_input))
            .unwrap();
        let mut output_data = vec![0u16; out_region.pixel_count()];
        reduce_simd::reduce_v_scalar(
            &op.filter,
            &in_region,
            &prepared_input,
            1,
            &out_region,
            &mut output_data,
        );
        output_data
    }

    fn run_reduce_v_f32_scalar(
        input_data: &[f32],
        factor: f64,
        kernel: InterpolationKernel,
    ) -> Vec<f32> {
        let source =
            MemorySource::<F32>::new(1, input_data.len() as u32, 1, input_data.to_vec()).unwrap();
        let op = ReduceV::<F32>::new(factor, kernel)
            .unwrap()
            .with_input_height(input_data.len() as u32);
        let out_region = Region::new(0, 0, 1, op.output_height(input_data.len() as u32));
        let in_region = op.required_input_region(&out_region);
        let mut prepared_input = vec![0.0f32; in_region.pixel_count()];
        source
            .read_region(in_region, bytemuck::cast_slice_mut(&mut prepared_input))
            .unwrap();
        let mut output_data = vec![0.0f32; out_region.pixel_count()];
        reduce_simd::reduce_v_scalar(
            &op.filter,
            &in_region,
            &prepared_input,
            1,
            &out_region,
            &mut output_data,
        );
        output_data
    }

    fn run_reduce_v_image_u8(
        input_data: &[u8],
        width: u32,
        height: u32,
        bands: u32,
        factor: f64,
        kernel: InterpolationKernel,
    ) -> Vec<u8> {
        let source = MemorySource::<U8>::new(width, height, bands, input_data.to_vec()).unwrap();
        let op = ReduceV::<U8>::new(factor, kernel)
            .unwrap()
            .with_input_height(height);
        let out_region = Region::new(0, 0, width, op.output_height(height));
        let in_region = op.required_input_region(&out_region);
        let mut prepared_input = vec![0u8; in_region.pixel_count() * bands as usize];
        source
            .read_region(in_region, bytemuck::cast_slice_mut(&mut prepared_input))
            .unwrap();
        let mut output_data = vec![0u8; out_region.pixel_count() * bands as usize];
        let input = Tile::<U8>::new(in_region, bands, &prepared_input);
        let mut output = TileMut::<U8>::new(out_region, bands, &mut output_data);
        let mut state = op.start_with_tile(out_region.width, out_region.height);
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    fn kernel_strategy() -> impl Strategy<Value = InterpolationKernel> {
        prop_oneof![
            Just(InterpolationKernel::Nearest),
            Just(InterpolationKernel::Bilinear),
            Just(InterpolationKernel::Bicubic),
            Just(InterpolationKernel::CatmullRom),
            Just(InterpolationKernel::Lanczos2),
            Just(InterpolationKernel::Lanczos3),
        ]
    }

    #[test]
    fn factor2_gradient_stays_monotone_for_bilinear() {
        let input: Vec<f32> = (0..16).map(|i| i as f32 * 8.0).collect();
        let output = run_reduce_v_f32(&input, 2.0, InterpolationKernel::Bilinear);
        for window in output.windows(2) {
            assert!(window[1] >= window[0], "gradient flipped: {output:?}");
        }
    }

    #[test]
    fn factor2_single_pixel_uses_edge_copy() {
        let output = run_reduce_v_u8(&[0], 2.0, InterpolationKernel::Lanczos3);
        assert_eq!(output, vec![0]);
    }

    #[test]
    fn factor2_single_pixel_uses_edge_copy_for_all_supported_kernels() {
        for kernel in [
            InterpolationKernel::Nearest,
            InterpolationKernel::Bilinear,
            InterpolationKernel::Bicubic,
            InterpolationKernel::CatmullRom,
            InterpolationKernel::Lanczos2,
            InterpolationKernel::Lanczos3,
        ] {
            let output = run_reduce_v_u8(&[91], 2.0, kernel);
            assert_eq!(output, vec![91], "kernel={kernel:?}");
        }
    }

    #[test]
    fn factor2_matches_libvips_centering_on_gradient() {
        let output = run_reduce_v_u8(
            &(0u8..8).collect::<Vec<_>>(),
            2.0,
            InterpolationKernel::Bilinear,
        );
        assert_eq!(output, vec![1, 3, 5, 6]);
    }

    #[test]
    fn small_multiband_image_preserves_edge_copy() {
        let input = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let output = run_reduce_v_image_u8(&input, 1, 4, 3, 2.0, InterpolationKernel::Bilinear);
        assert_eq!(output, vec![29, 39, 49, 81, 91, 101]);
    }

    #[test]
    fn lanczos3_factor1_5_matches_libvips_on_grayscale_fixture() {
        let width = 6;
        let height = 8;
        let mut source = Vec::with_capacity(64);
        for y in 0..8 {
            for x in 0..8 {
                source.push(((x * 17 + y * 13 + 5) % 256) as u8);
            }
        }
        let input = source[..(width * height) as usize].to_vec();

        let output =
            run_reduce_v_image_u8(&input, width, height, 1, 1.5, InterpolationKernel::Lanczos3);

        assert_eq!(
            output,
            vec![
                57, 74, 25, 42, 59, 76, 98, 115, 100, 117, 51, 68, 47, 64, 89, 106, 130, 147, 126,
                143, 77, 94, 79, 96, 118, 135, 152, 169, 120, 137,
            ]
        );
    }

    #[test]
    fn reducev_bridge_exposes_dyn_operation_contract() {
        let bridge = ReduceVBridge::<U8>::new(
            2.0,
            InterpolationKernel::Bilinear,
            1,
            4,
            DemandHint::ThinStrip,
        )
        .unwrap();
        assert_eq!(bridge.input_format(), BandFormatId::U8);
        assert_eq!(bridge.output_format(), BandFormatId::U8);
        assert_eq!(bridge.bands(), 1);
        assert_eq!(bridge.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(bridge.output_width(1), 1);
        assert_eq!(bridge.output_height(4), 2);
        assert_eq!(
            bridge.node_spec(1, 2),
            ReduceV::<U8>::new(2.0, InterpolationKernel::Bilinear)
                .unwrap()
                .with_input_height(4)
                .node_spec(1, 2)
        );

        let source = MemorySource::<U8>::new(1, 4, 1, vec![0, 1, 2, 3]).unwrap();
        let out_region = Region::new(0, 0, 1, 2);
        let input_region = bridge.required_input_region(&out_region);
        let mut input_bytes = vec![0u8; input_region.pixel_count()];
        source.read_region(input_region, &mut input_bytes).unwrap();
        let mut output_bytes = vec![0u8; out_region.pixel_count()];
        let mut state = bridge.dyn_start();
        bridge.dyn_process_region(
            &mut *state,
            &input_bytes,
            &mut output_bytes,
            input_region,
            out_region,
        );
        assert_eq!(output_bytes, vec![1, 2]);
    }

    #[test]
    fn reducev_rejects_vsqbs_kernel() {
        let err = match ReduceV::<U8>::new(2.0, InterpolationKernel::Vsqbs) {
            Ok(_) => panic!("vsqbs must be rejected for reducev"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            BuildError::InvalidKernel {
                op: "reducev",
                kernel: InterpolationKernel::Vsqbs,
                ..
            }
        ));
        assert!(err.to_string().contains("non-separable"));
    }

    #[test]
    fn reducev_rejects_lbb_kernel() {
        let err = match ReduceV::<U8>::new(2.0, InterpolationKernel::Lbb) {
            Ok(_) => panic!("lbb must be rejected for reducev"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            BuildError::InvalidKernel {
                op: "reducev",
                kernel: InterpolationKernel::Lbb,
                ..
            }
        ));
        assert!(err.to_string().contains("nonlinear 2-D affine"));
    }

    #[test]
    fn reducev_rejects_nohalo_kernel() {
        let err = match ReduceV::<U8>::new(2.0, InterpolationKernel::Nohalo) {
            Ok(_) => panic!("nohalo must be rejected for reducev"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            BuildError::InvalidKernel {
                op: "reducev",
                kernel: InterpolationKernel::Nohalo,
                ..
            }
        ));
        assert!(err.to_string().contains("no separable reduce kernel"));
        assert!(
            err.to_string()
                .contains("Nearest, Bilinear, Bicubic/CatmullRom, Lanczos2, Lanczos3")
        );
    }

    proptest! {
        #[test]
        fn factor1_is_identity(
            column in prop::collection::vec(any::<u8>(), 1..=32),
            kernel in kernel_strategy(),
        ) {
            let output = run_reduce_v_u8(&column, 1.0, kernel);
            prop_assert_eq!(output, column);
        }

        #[test]
        fn factor1_multiband_2x2_is_identity(
            pixels in prop::collection::vec(any::<u8>(), 12),
        ) {
            let output = run_reduce_v_image_u8(
                &pixels,
                2,
                2,
                3,
                1.0,
                InterpolationKernel::Bilinear,
            );
            prop_assert_eq!(output, pixels);
        }

        #[test]
        fn uniform_columns_stay_uniform_for_integer_factors(
            value in any::<u8>(),
            factor in prop_oneof![Just(2.0), Just(4.0), Just(8.0)],
            kernel in kernel_strategy(),
        ) {
            let input = vec![value; 64];
            let output = run_reduce_v_u8(&input, factor, kernel);
            prop_assert!(output.iter().all(|sample| *sample == value));
        }

        #[test]
        fn uniform_columns_stay_uniform_for_non_integer_factors(
            value in any::<u8>(),
            factor in prop_oneof![Just(1.5), Just(2.5), Just(3.5)],
            kernel in kernel_strategy(),
        ) {
            let input = vec![value; 64];
            let output = run_reduce_v_u8(&input, factor, kernel);
            prop_assert!(output.iter().all(|sample| *sample == value));
        }

        #[test]
        fn one_pixel_tall_images_preserve_constant_edges(
            value in any::<u8>(),
            width in 1u32..=8,
            kernel in kernel_strategy(),
        ) {
            let input = vec![value; width as usize];
            let output = run_reduce_v_image_u8(&input, width, 1, 1, 2.0, kernel);
            prop_assert_eq!(output, input);
        }
    }

    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx2")
    ))]
    proptest! {
        #[test]
        fn scalar_matches_simd_u8(
            column in prop::collection::vec(any::<u8>(), 1..=128),
            factor in prop_oneof![Just(1.5), Just(2.0), Just(3.0), Just(4.0)],
            kernel in kernel_strategy(),
        ) {
            let scalar = run_reduce_v_u8_scalar(&column, factor, kernel);
            let simd = run_reduce_v_u8(&column, factor, kernel);
            prop_assert_eq!(simd, scalar);
        }

        #[test]
        fn scalar_matches_simd_u16(
            column in prop::collection::vec(any::<u16>(), 1..=128),
            factor in prop_oneof![Just(1.5), Just(2.0), Just(3.0), Just(4.0)],
            kernel in kernel_strategy(),
        ) {
            let scalar = run_reduce_v_u16_scalar(&column, factor, kernel);
            let simd = run_reduce_v_u16(&column, factor, kernel);
            prop_assert_eq!(simd, scalar);
        }

        #[test]
        fn scalar_matches_simd_f32(
            column in prop::collection::vec(-10_000.0f32..10_000.0f32, 1..=128),
            factor in prop_oneof![Just(1.5), Just(2.0), Just(3.0), Just(4.0)],
            kernel in kernel_strategy(),
        ) {
            let scalar = run_reduce_v_f32_scalar(&column, factor, kernel);
            let simd = run_reduce_v_f32(&column, factor, kernel);
            prop_assert_eq!(simd.len(), scalar.len());
            for (lhs, rhs) in simd.iter().zip(scalar.iter()) {
                prop_assert!((lhs - rhs).abs() <= 1e-3);
            }
        }
    }
}

//! Horizontal reduction (downscale along the x-axis).
//!
//! `ReduceH` mirrors libvips `reduceh`: it precomputes 64 subpixel coefficient
//! tables at construction time and reuses them in the hot path, avoiding any
//! heap allocation per pixel or tile.

use std::marker::PhantomData;

use crate::domain::{
    error::BuildError,
    format::{BandFormat, BandFormatId},
    image::{DemandHint, Region, Tile, TileMut},
    kernel::InterpolationKernel,
    op::{NodeSpec, Op},
    resample::{ReduceConfig, ResampleOp},
};

pub use super::sample_conv::ReduceSample;
use super::{
    reduce_common::{
        ReduceKernel, validate_reduce_factors, validate_reduce_kernel, validate_reduce_tap_limits,
    },
    reduce_simd::{self, reduce_h_scalar},
};

/// Horizontal 1-D reduction by a rational factor using a precomputed kernel table.
pub struct ReduceH<F: BandFormat> {
    filter: ReduceKernel,
    _format: PhantomData<F>,
}

/// Represents a reduce h state.
pub struct ReduceHState {
    starts: Vec<i64>,
    phases: Vec<u8>,
}

impl<F: BandFormat + Send + Sync> ReduceH<F> {
    /// Create a new `ReduceH` that reduces the image width by `factor`.
    pub fn new(factor: f64, kernel: InterpolationKernel) -> Result<Self, BuildError> {
        validate_reduce_factors(factor, 1.0)?;
        validate_reduce_kernel("reduceh", kernel)?;
        validate_reduce_tap_limits(factor, 1.0, kernel)?;
        Ok(Self {
            filter: ReduceKernel::new(factor, kernel)?,
            _format: PhantomData,
        })
    }

    /// Bind the full input width so the resampling centre matches libvips.
    #[must_use]
    pub fn with_input_width(mut self, input_w: u32) -> Self {
        self.filter.bind_input_len(input_w);
        self
    }

    #[inline]
    const fn config(&self) -> ReduceConfig {
        self.filter.config()
    }
}

impl<F: BandFormat> Op for ReduceH<F>
where
    F::Sample: ReduceSample,
{
    type Input = F;
    type Output = F;
    type State = ReduceHState;

    fn preferred_tile_geometry(&self) -> DemandHint {
        DemandHint::FatStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        self.filter.required_input_region_h(output)
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        self.config().node_spec_h(tile_w, tile_h)
    }

    fn start(&self) -> Self::State {
        ReduceHState {
            starts: Vec::new(),
            phases: Vec::new(),
        }
    }

    fn start_with_tile(&self, tile_w: u32, _tile_h: u32) -> Self::State {
        ReduceHState {
            starts: vec![0; tile_w as usize],
            phases: vec![0; tile_w as usize],
        }
    }

    #[inline]
    fn process_region(&self, state: &mut Self::State, input: &Tile<F>, output: &mut TileMut<F>) {
        if F::ID == BandFormatId::U8 {
            let out_w = output.region.width as usize;
            if state.starts.len() < out_w {
                state.starts.resize(out_w, 0);
            }
            if state.phases.len() < out_w {
                state.phases.resize(out_w, 0);
            }
            let starts = &mut state.starts[..out_w];
            let phases = &mut state.phases[..out_w];
            let step = self.filter.config().factor;
            let mut source_x = self.filter.source_position(f64::from(output.region.x));
            for x_out in 0..out_w {
                let (start_x, phase) = self.filter.plan_i16(source_x);
                starts[x_out] = start_x;
                phases[x_out] = phase as u8;
                source_x += step;
            }
            reduce_simd::reduce_h_u8_planned(
                &self.filter,
                &input.region,
                bytemuck::cast_slice(input.data),
                input.bands,
                &output.region,
                bytemuck::cast_slice_mut(output.data),
                starts,
                phases,
            );
            return;
        }
        if F::ID == BandFormatId::U16 {
            reduce_simd::reduce_h_u16(
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
            reduce_simd::reduce_h_f32(
                &self.filter,
                &input.region,
                bytemuck::cast_slice(input.data),
                input.bands,
                &output.region,
                bytemuck::cast_slice_mut(output.data),
            );
            return;
        }

        reduce_h_scalar(
            &self.filter,
            &input.region,
            input.data,
            input.bands,
            &output.region,
            output.data,
        );
    }
}

impl<F: BandFormat> ResampleOp for ReduceH<F>
where
    F::Sample: ReduceSample,
{
    fn output_width(&self, input_w: u32) -> u32 {
        self.config().output_width(input_w)
    }

    fn output_height(&self, input_h: u32) -> u32 {
        input_h
    }
}

pub(crate) struct ReduceHBridge<F: BandFormat>
where
    F::Sample: bytemuck::Pod + ReduceSample,
{
    inner: crate::domain::op::OperationBridge<ReduceH<F>>,
    demand_hint: DemandHint,
}

impl<F: BandFormat> ReduceHBridge<F>
where
    F::Sample: bytemuck::Pod + ReduceSample,
{
    pub fn new(
        factor: f64,
        kernel: InterpolationKernel,
        bands: u32,
        input_w: u32,
        demand_hint: DemandHint,
    ) -> Result<Self, BuildError> {
        Ok(Self {
            inner: crate::domain::op::OperationBridge::new(
                ReduceH::new(factor, kernel)?.with_input_width(input_w),
                bands,
            ),
            demand_hint,
        })
    }
}

impl<F: BandFormat> crate::domain::op::DynOperation for ReduceHBridge<F>
where
    F::Sample: bytemuck::Pod + ReduceSample + Send,
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
        self.demand_hint
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
        input_h
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        adapters::sources::memory::MemorySource,
        domain::{
            format::{BandFormatId, F32, I16, U8, U16},
            image::{Region, Tile, TileMut},
            kernel::InterpolationKernel,
            op::DynOperation,
            ops::resample::{reduce_simd, shrinkh::ShrinkH},
            resample::ResampleOp,
        },
        ports::source::ImageSource,
    };
    use proptest::prelude::*;

    #[test]
    fn new_rejects_extreme_finite_factor() {
        let result = ReduceH::<U8>::new(f64::MAX / 2.0, InterpolationKernel::Lanczos3);

        assert!(matches!(
            result,
            Err(BuildError::InvalidReduceParameters { .. })
        ));
    }

    fn run_reduce_h_u8(input_data: &[u8], factor: f64, kernel: InterpolationKernel) -> Vec<u8> {
        let source =
            MemorySource::<U8>::new(input_data.len() as u32, 1, 1, input_data.to_vec()).unwrap();
        let op = ReduceH::<U8>::new(factor, kernel)
            .unwrap()
            .with_input_width(input_data.len() as u32);
        let out_region = Region::new(0, 0, op.output_width(input_data.len() as u32), 1);
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

    fn run_reduce_h_f32(input_data: &[f32], factor: f64, kernel: InterpolationKernel) -> Vec<f32> {
        let source =
            MemorySource::<F32>::new(input_data.len() as u32, 1, 1, input_data.to_vec()).unwrap();
        let op = ReduceH::<F32>::new(factor, kernel)
            .unwrap()
            .with_input_width(input_data.len() as u32);
        let out_region = Region::new(0, 0, op.output_width(input_data.len() as u32), 1);
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

    fn run_reduce_h_u16(input_data: &[u16], factor: f64, kernel: InterpolationKernel) -> Vec<u16> {
        let source =
            MemorySource::<U16>::new(input_data.len() as u32, 1, 1, input_data.to_vec()).unwrap();
        let op = ReduceH::<U16>::new(factor, kernel)
            .unwrap()
            .with_input_width(input_data.len() as u32);
        let out_region = Region::new(0, 0, op.output_width(input_data.len() as u32), 1);
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

    fn run_reduce_h_u8_scalar(
        input_data: &[u8],
        factor: f64,
        kernel: InterpolationKernel,
    ) -> Vec<u8> {
        let source =
            MemorySource::<U8>::new(input_data.len() as u32, 1, 1, input_data.to_vec()).unwrap();
        let op = ReduceH::<U8>::new(factor, kernel)
            .unwrap()
            .with_input_width(input_data.len() as u32);
        let out_region = Region::new(0, 0, op.output_width(input_data.len() as u32), 1);
        let in_region = op.required_input_region(&out_region);
        let mut prepared_input = vec![0u8; in_region.pixel_count()];
        source
            .read_region(in_region, bytemuck::cast_slice_mut(&mut prepared_input))
            .unwrap();
        let mut output_data = vec![0u8; out_region.pixel_count()];
        reduce_simd::reduce_h_scalar(
            &op.filter,
            &in_region,
            &prepared_input,
            1,
            &out_region,
            &mut output_data,
        );
        output_data
    }

    fn run_reduce_h_u16_scalar(
        input_data: &[u16],
        factor: f64,
        kernel: InterpolationKernel,
    ) -> Vec<u16> {
        let source =
            MemorySource::<U16>::new(input_data.len() as u32, 1, 1, input_data.to_vec()).unwrap();
        let op = ReduceH::<U16>::new(factor, kernel)
            .unwrap()
            .with_input_width(input_data.len() as u32);
        let out_region = Region::new(0, 0, op.output_width(input_data.len() as u32), 1);
        let in_region = op.required_input_region(&out_region);
        let mut prepared_input = vec![0u16; in_region.pixel_count()];
        source
            .read_region(in_region, bytemuck::cast_slice_mut(&mut prepared_input))
            .unwrap();
        let mut output_data = vec![0u16; out_region.pixel_count()];
        reduce_simd::reduce_h_scalar(
            &op.filter,
            &in_region,
            &prepared_input,
            1,
            &out_region,
            &mut output_data,
        );
        output_data
    }

    fn run_reduce_h_f32_scalar(
        input_data: &[f32],
        factor: f64,
        kernel: InterpolationKernel,
    ) -> Vec<f32> {
        let source =
            MemorySource::<F32>::new(input_data.len() as u32, 1, 1, input_data.to_vec()).unwrap();
        let op = ReduceH::<F32>::new(factor, kernel)
            .unwrap()
            .with_input_width(input_data.len() as u32);
        let out_region = Region::new(0, 0, op.output_width(input_data.len() as u32), 1);
        let in_region = op.required_input_region(&out_region);
        let mut prepared_input = vec![0.0f32; in_region.pixel_count()];
        source
            .read_region(in_region, bytemuck::cast_slice_mut(&mut prepared_input))
            .unwrap();
        let mut output_data = vec![0.0f32; out_region.pixel_count()];
        reduce_simd::reduce_h_scalar(
            &op.filter,
            &in_region,
            &prepared_input,
            1,
            &out_region,
            &mut output_data,
        );
        output_data
    }

    fn run_reduce_h_image_u8(
        input_data: &[u8],
        width: u32,
        height: u32,
        bands: u32,
        factor: f64,
        kernel: InterpolationKernel,
    ) -> Vec<u8> {
        let source = MemorySource::<U8>::new(width, height, bands, input_data.to_vec()).unwrap();
        let op = ReduceH::<U8>::new(factor, kernel)
            .unwrap()
            .with_input_width(width);
        let out_region = Region::new(0, 0, op.output_width(width), height);
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

    fn run_shrink_h_u8(input_data: &[u8], factor: u32) -> Vec<u8> {
        let op = ShrinkH::<U8>::new(factor).unwrap();
        let in_region = Region::new(0, 0, input_data.len() as u32, 1);
        let out_region = Region::new(0, 0, op.output_width(input_data.len() as u32), 1);
        let input = Tile::<U8>::new(in_region, 1, input_data);
        let mut output_data = vec![0u8; out_region.pixel_count()];
        let mut output = TileMut::<U8>::new(out_region, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    fn run_reduce_h_i16(input_data: &[i16], factor: f64, kernel: InterpolationKernel) -> Vec<i16> {
        let source =
            MemorySource::<I16>::new(input_data.len() as u32, 1, 1, input_data.to_vec()).unwrap();
        let op = ReduceH::<I16>::new(factor, kernel)
            .unwrap()
            .with_input_width(input_data.len() as u32);
        let out_region = Region::new(0, 0, op.output_width(input_data.len() as u32), 1);
        let in_region = op.required_input_region(&out_region);
        let mut prepared_input = vec![0i16; in_region.pixel_count()];
        source
            .read_region(in_region, bytemuck::cast_slice_mut(&mut prepared_input))
            .unwrap();
        let mut output_data = vec![0i16; out_region.pixel_count()];
        let input = Tile::<I16>::new(in_region, 1, &prepared_input);
        let mut output = TileMut::<I16>::new(out_region, 1, &mut output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    fn run_reduce_h_i16_scalar(
        input_data: &[i16],
        factor: f64,
        kernel: InterpolationKernel,
    ) -> Vec<i16> {
        let source =
            MemorySource::<I16>::new(input_data.len() as u32, 1, 1, input_data.to_vec()).unwrap();
        let op = ReduceH::<I16>::new(factor, kernel)
            .unwrap()
            .with_input_width(input_data.len() as u32);
        let out_region = Region::new(0, 0, op.output_width(input_data.len() as u32), 1);
        let in_region = op.required_input_region(&out_region);
        let mut prepared_input = vec![0i16; in_region.pixel_count()];
        source
            .read_region(in_region, bytemuck::cast_slice_mut(&mut prepared_input))
            .unwrap();
        let mut output_data = vec![0i16; out_region.pixel_count()];
        reduce_h_scalar(
            &op.filter,
            &in_region,
            &prepared_input,
            1,
            &out_region,
            &mut output_data,
        );
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
        let output = run_reduce_h_f32(&input, 2.0, InterpolationKernel::Bilinear);
        for window in output.windows(2) {
            assert!(window[1] >= window[0], "gradient flipped: {output:?}");
        }
    }

    #[test]
    fn factor2_single_pixel_uses_edge_copy() {
        let output = run_reduce_h_u8(&[255], 2.0, InterpolationKernel::Lanczos3);
        assert_eq!(output, vec![255]);
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
            let output = run_reduce_h_u8(&[173], 2.0, kernel);
            assert_eq!(output, vec![173], "kernel={kernel:?}");
        }
    }

    #[test]
    fn factor2_matches_libvips_centering_on_gradient() {
        let output = run_reduce_h_u8(
            &(0u8..8).collect::<Vec<_>>(),
            2.0,
            InterpolationKernel::Bilinear,
        );
        assert_eq!(output, vec![1, 3, 5, 6]);
    }

    #[test]
    fn small_multiband_image_preserves_edge_copy() {
        let input = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let output = run_reduce_h_image_u8(&input, 2, 2, 3, 2.0, InterpolationKernel::Bilinear);
        assert_eq!(output, vec![25, 35, 45, 85, 95, 105]);
    }

    #[test]
    fn reduceh_bridge_exposes_dyn_operation_contract() {
        let bridge = ReduceHBridge::<U8>::new(
            2.0,
            InterpolationKernel::Bilinear,
            1,
            4,
            DemandHint::FatStrip,
        )
        .unwrap();
        assert_eq!(bridge.input_format(), BandFormatId::U8);
        assert_eq!(bridge.output_format(), BandFormatId::U8);
        assert_eq!(bridge.bands(), 1);
        assert_eq!(bridge.demand_hint(), DemandHint::FatStrip);
        assert_eq!(bridge.output_width(4), 2);
        assert_eq!(bridge.output_height(1), 1);
        assert_eq!(
            bridge.node_spec(2, 1),
            ReduceH::<U8>::new(2.0, InterpolationKernel::Bilinear)
                .unwrap()
                .with_input_width(4)
                .node_spec(2, 1)
        );

        let source = MemorySource::<U8>::new(4, 1, 1, vec![0, 1, 2, 3]).unwrap();
        let out_region = Region::new(0, 0, 2, 1);
        let input_region = bridge.required_input_region(&out_region);
        let mut input_bytes = vec![0u8; input_region.pixel_count()];
        source.read_region(input_region, &mut input_bytes).unwrap();
        let mut output_bytes = vec![0u8; out_region.pixel_count()];
        let mut state = bridge.dyn_start_with_tile(out_region.width, out_region.height);
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
    fn reduceh_rejects_vsqbs_kernel() {
        let err = match ReduceH::<U8>::new(2.0, InterpolationKernel::Vsqbs) {
            Ok(_) => panic!("vsqbs must be rejected for reduceh"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            BuildError::InvalidKernel {
                op: "reduceh",
                kernel: InterpolationKernel::Vsqbs,
                ..
            }
        ));
        assert!(err.to_string().contains("non-separable"));
    }

    #[test]
    fn reduceh_rejects_lbb_kernel() {
        let err = match ReduceH::<U8>::new(2.0, InterpolationKernel::Lbb) {
            Ok(_) => panic!("lbb must be rejected for reduceh"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            BuildError::InvalidKernel {
                op: "reduceh",
                kernel: InterpolationKernel::Lbb,
                ..
            }
        ));
        assert!(err.to_string().contains("nonlinear 2-D affine"));
    }

    #[test]
    fn reduceh_rejects_nohalo_kernel() {
        let err = match ReduceH::<U8>::new(2.0, InterpolationKernel::Nohalo) {
            Ok(_) => panic!("nohalo must be rejected for reduceh"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            BuildError::InvalidKernel {
                op: "reduceh",
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

    #[test]
    fn reduceh_then_shrinkh_differs_from_single_reduceh_by_design() {
        // libvips documents shrink as a box filter and reduce as a kernel-based
        // resampler, so these pipelines are intentionally not equivalent.
        let input: Vec<u8> = (0..96).map(|x| ((x * 37 + 11) % 256) as u8).collect();

        let combined = run_shrink_h_u8(
            &run_reduce_h_u8(&input, 2.0, InterpolationKernel::Lanczos3),
            2,
        );
        let direct = run_reduce_h_u8(&input, 4.0, InterpolationKernel::Lanczos3);

        assert_eq!(combined.len(), direct.len());
        assert_ne!(combined, direct);
    }

    #[test]
    fn reduceh_metadata_and_empty_state_resize_match_runtime_needs() {
        let source = MemorySource::<U8>::new(8, 1, 1, (0u8..8).collect()).unwrap();
        let op = ReduceH::<U8>::new(2.0, InterpolationKernel::Bilinear)
            .unwrap()
            .with_input_width(8);
        let out_region = Region::new(0, 0, op.output_width(8), 1);
        let in_region = op.required_input_region(&out_region);
        let mut prepared_input = vec![0u8; in_region.pixel_count()];
        source.read_region(in_region, &mut prepared_input).unwrap();
        let input = Tile::<U8>::new(in_region, 1, &prepared_input);
        let mut output_data = vec![0u8; out_region.pixel_count()];
        let mut output = TileMut::<U8>::new(out_region, 1, &mut output_data);
        let mut state = op.start();

        op.process_region(&mut state, &input, &mut output);

        assert_eq!(op.preferred_tile_geometry(), DemandHint::FatStrip);
        assert_eq!(op.output_height(5), 5);
        assert_eq!(state.starts.len(), out_region.width as usize);
        assert_eq!(state.phases.len(), out_region.width as usize);
    }

    #[test]
    fn reduceh_i16_uses_scalar_path_and_matches_scalar_reference() {
        let input = vec![-300i16, -100, 0, 100, 300, 600, 900, 1200];
        let scalar = run_reduce_h_i16_scalar(&input, 2.0, InterpolationKernel::Bicubic);
        let output = run_reduce_h_i16(&input, 2.0, InterpolationKernel::Bicubic);

        assert_eq!(output, scalar);
    }

    #[test]
    fn reduceh_bridge_dyn_start_without_tile_runs_successfully() {
        let bridge = ReduceHBridge::<U8>::new(
            2.0,
            InterpolationKernel::Bilinear,
            1,
            4,
            DemandHint::FatStrip,
        )
        .unwrap();
        let source = MemorySource::<U8>::new(4, 1, 1, vec![0, 1, 2, 3]).unwrap();
        let out_region = Region::new(0, 0, 2, 1);
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

    proptest! {
        #[test]
        fn factor1_is_identity(
            row in prop::collection::vec(any::<u8>(), 1..=32),
            kernel in kernel_strategy(),
        ) {
            let output = run_reduce_h_u8(&row, 1.0, kernel);
            prop_assert_eq!(output, row);
        }

        #[test]
        fn factor1_multiband_2x2_is_identity(
            pixels in prop::collection::vec(any::<u8>(), 12),
        ) {
            let output = run_reduce_h_image_u8(
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
        fn uniform_rows_stay_uniform_for_integer_factors(
            value in any::<u8>(),
            factor in prop_oneof![Just(2.0), Just(4.0), Just(8.0)],
            kernel in kernel_strategy(),
        ) {
            let input = vec![value; 64];
            let output = run_reduce_h_u8(&input, factor, kernel);
            prop_assert!(output.iter().all(|sample| *sample == value));
        }

        #[test]
        fn uniform_rows_stay_uniform_for_non_integer_factors(
            value in any::<u8>(),
            factor in prop_oneof![Just(1.5), Just(2.5), Just(3.5)],
            kernel in kernel_strategy(),
        ) {
            let input = vec![value; 64];
            let output = run_reduce_h_u8(&input, factor, kernel);
            prop_assert!(output.iter().all(|sample| *sample == value));
        }

        #[test]
        fn one_pixel_wide_images_preserve_constant_edges(
            value in any::<u8>(),
            height in 1u32..=8,
            kernel in kernel_strategy(),
        ) {
            let input = vec![value; height as usize];
            let output = run_reduce_h_image_u8(&input, 1, height, 1, 2.0, kernel);
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
            row in prop::collection::vec(any::<u8>(), 1..=128),
            factor in prop_oneof![Just(1.5), Just(2.0), Just(3.0), Just(4.0)],
            kernel in kernel_strategy(),
        ) {
            let scalar = run_reduce_h_u8_scalar(&row, factor, kernel);
            let simd = run_reduce_h_u8(&row, factor, kernel);
            prop_assert_eq!(simd, scalar);
        }

        #[test]
        fn scalar_matches_simd_u16(
            row in prop::collection::vec(any::<u16>(), 1..=128),
            factor in prop_oneof![Just(1.5), Just(2.0), Just(3.0), Just(4.0)],
            kernel in kernel_strategy(),
        ) {
            let scalar = run_reduce_h_u16_scalar(&row, factor, kernel);
            let simd = run_reduce_h_u16(&row, factor, kernel);
            prop_assert_eq!(simd, scalar);
        }

        #[test]
        fn scalar_matches_simd_f32(
            row in prop::collection::vec(-10_000.0f32..10_000.0f32, 1..=128),
            factor in prop_oneof![Just(1.5), Just(2.0), Just(3.0), Just(4.0)],
            kernel in kernel_strategy(),
        ) {
            let scalar = run_reduce_h_f32_scalar(&row, factor, kernel);
            let simd = run_reduce_h_f32(&row, factor, kernel);
            prop_assert_eq!(simd.len(), scalar.len());
            for (lhs, rhs) in simd.iter().zip(scalar.iter()) {
                prop_assert!((lhs - rhs).abs() <= 1e-3);
            }
        }
    }
}

#![allow(clippy::neg_cmp_op_on_partial_ord)]
// REASON: the explicit floating-point guards mirror the reference validation logic verbatim.

use std::marker::PhantomData;

use viprs_core::{
    format::{BandFormat, F32, U8},
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
    shared_ops::sample_conv::ToF64,
    stats::ImageStats,
};

/// Constant value for default log exponent.
pub const DEFAULT_LOG_EXPONENT: f64 = 0.25;

/// Output transfer curve for [`ScaleOp`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScaleMode {
    /// Uses the `Linear` variant of `ScaleMode`.
    Linear,
    /// Logarithmic scaling using the provided exponent.
    Log {
        /// Exponent applied before normalization.
        exponent: f64,
    },
}

impl ScaleMode {
    #[must_use]
    /// Returns or performs log default.
    pub const fn log_default() -> Self {
        Self::Log {
            exponent: DEFAULT_LOG_EXPONENT,
        }
    }
}

/// Output sample types supported by [`ScaleOp`].
pub trait ScaleOutputSample: Copy {
    /// Creates this value from unit interval.
    fn from_unit_interval(value: f64) -> Self;
}

impl ScaleOutputSample for u8 {
    #[inline(always)]
    fn from_unit_interval(value: f64) -> Self {
        value.clamp(0.0, 1.0).mul_add(255.0, 0.5).floor() as Self
    }
}

impl ScaleOutputSample for f32 {
    #[inline(always)]
    fn from_unit_interval(value: f64) -> Self {
        value.clamp(0.0, 1.0) as Self
    }
}

#[derive(Debug, Clone, Copy)]
enum ScaleTransfer {
    ZeroRange,
    Linear { scale: f64, offset: f64 },
    Log { denominator_inv: f64, exponent: f64 },
}

/// Low-level display scaling with reducer-fed global min/max.
///
/// libvips computes one global min/max pair with `vips_stats`, then applies either:
/// - linear scaling: `(x - min) / (max - min)`
/// - log scaling: `log10(1 + pow(x, exp)) / log10(1 + pow(max, exp))`
///
/// Callers that need full pipeline parity should first obtain [`ImageStats`] through
/// `StatsReducer`, then build this op with [`ScaleOp::from_stats`].
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::conversion::scale::ScaleOp;
///
/// let op = ScaleOp { /* operation parameters */ };
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct ScaleOp<In: BandFormat, Out: BandFormat> {
    min: f64,
    max: f64,
    mode: ScaleMode,
    transfer: ScaleTransfer,
    _input: PhantomData<In>,
    _output: PhantomData<Out>,
}

impl<In, Out> ScaleOp<In, Out>
where
    In: BandFormat,
    Out: BandFormat,
    In::Sample: ToF64,
    Out::Sample: ScaleOutputSample,
{
    #[must_use]
    /// Returns or performs new linear.
    pub fn new_linear(min: f64, max: f64) -> Self {
        debug_assert!(
            min.is_finite() && max.is_finite(),
            "ScaleOp requires finite min/max"
        );
        Self::new(min, max, ScaleMode::Linear)
    }

    #[must_use]
    /// Returns or performs new log.
    pub fn new_log(min: f64, max: f64, exponent: f64) -> Self {
        debug_assert!(
            min.is_finite() && max.is_finite() && exponent.is_finite() && exponent > 0.0,
            "ScaleOp log mode requires finite min/max and exponent > 0"
        );
        Self::new(min, max, ScaleMode::Log { exponent })
    }

    #[must_use]
    /// Creates this value from stats.
    pub fn from_stats(stats: &ImageStats, mode: ScaleMode) -> Self {
        let (min, max) = Self::global_min_max(stats);
        Self::new(min, max, mode)
    }

    #[must_use]
    /// Creates this value from stats log.
    pub fn from_stats_log(stats: &ImageStats) -> Self {
        Self::from_stats(stats, ScaleMode::log_default())
    }

    #[must_use]
    /// Returns or performs min.
    pub const fn min(&self) -> f64 {
        self.min
    }

    #[must_use]
    /// Returns or performs max.
    pub const fn max(&self) -> f64 {
        self.max
    }

    #[must_use]
    /// Returns or performs mode.
    pub const fn mode(&self) -> ScaleMode {
        self.mode
    }

    #[must_use]
    fn new(min: f64, max: f64, mode: ScaleMode) -> Self {
        let transfer = match mode {
            ScaleMode::Linear => Self::build_linear_transfer(min, max),
            ScaleMode::Log { exponent } => Self::build_log_transfer(min, max, exponent),
        };

        Self {
            min,
            max,
            mode,
            transfer,
            _input: PhantomData,
            _output: PhantomData,
        }
    }

    #[must_use]
    fn global_min_max(stats: &ImageStats) -> (f64, f64) {
        let min = stats.min.iter().copied().fold(f64::INFINITY, f64::min);
        let max = stats.max.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        if min.is_finite() && max.is_finite() {
            (min, max)
        } else {
            (0.0, 0.0)
        }
    }

    #[must_use]
    fn build_linear_transfer(min: f64, max: f64) -> ScaleTransfer {
        if !(min < max) {
            return ScaleTransfer::ZeroRange;
        }

        let range = max - min;
        ScaleTransfer::Linear {
            scale: 1.0 / range,
            offset: -min / range,
        }
    }

    #[must_use]
    fn build_log_transfer(min: f64, max: f64, exponent: f64) -> ScaleTransfer {
        if !(min < max) {
            return ScaleTransfer::ZeroRange;
        }

        let denominator = (1.0 + max.powf(exponent)).log10();
        if denominator.is_finite() && denominator > 0.0 {
            ScaleTransfer::Log {
                denominator_inv: 1.0 / denominator,
                exponent,
            }
        } else {
            ScaleTransfer::ZeroRange
        }
    }

    #[inline(always)]
    fn normalize(&self, value: f64) -> f64 {
        let normalized = match self.transfer {
            ScaleTransfer::ZeroRange => 0.0,
            ScaleTransfer::Linear { scale, offset } => value.mul_add(scale, offset),
            ScaleTransfer::Log {
                denominator_inv,
                exponent,
            } => {
                let transformed = (1.0 + value.powf(exponent)).log10();
                if transformed.is_finite() {
                    transformed * denominator_inv
                } else {
                    0.0
                }
            }
        };

        if normalized.is_finite() {
            normalized.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

impl<In, Out> Op for ScaleOp<In, Out>
where
    In: BandFormat,
    Out: BandFormat,
    In::Sample: ToF64,
    Out::Sample: ScaleOutputSample,
{
    type Input = In;
    type Output = Out;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<In>, output: &mut TileMut<Out>) {
        for (src, dst) in input.data.iter().zip(output.data.iter_mut()) {
            *dst = Out::Sample::from_unit_interval(self.normalize(src.to_f64()));
        }
    }
}

impl<In, Out> PixelLocalOp for ScaleOp<In, Out>
where
    In: BandFormat,
    Out: BandFormat,
    In::Sample: ToF64,
    Out::Sample: ScaleOutputSample,
{
}

/// Type alias for scale to u8 op.
pub type ScaleToU8Op<F> = ScaleOp<F, U8>;
/// Type alias for scale to f32 op.
pub type ScaleToF32Op<F> = ScaleOp<F, F32>;

#[cfg(all(test, feature = "_integration"))]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::{U8, U16},
        image::{Region, Tile, TileMut},
        op::{Op, OperationBridge},
        stats::ImageStats,
    };
    use viprs_ports::scheduler::TileScheduler;
    use viprs_runtime::{
        pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    };

    struct PassThroughU16;

    impl Op for PassThroughU16 {
        type Input = U16;
        type Output = U16;
        type State = ();

        fn demand_hint(&self) -> DemandHint {
            DemandHint::ThinStrip
        }

        fn required_input_region(&self, output: &Region) -> Region {
            *output
        }

        fn start(&self) {}

        #[inline]
        fn process_region(&self, _state: &mut (), input: &Tile<U16>, output: &mut TileMut<U16>) {
            output.data.copy_from_slice(input.data);
        }
    }

    fn run_scale_u8<F: BandFormat>(op: ScaleOp<F, U8>, input_data: &[F::Sample]) -> Vec<u8>
    where
        F::Sample: ToF64,
    {
        let region = Region::new(0, 0, input_data.len() as u32, 1);
        let input = Tile::<F>::new(region, 1, input_data);
        let mut output_data = vec![0u8; input_data.len()];
        let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    fn run_scale_f32<F: BandFormat>(op: ScaleOp<F, F32>, input_data: &[F::Sample]) -> Vec<f32>
    where
        F::Sample: ToF64,
    {
        let region = Region::new(0, 0, input_data.len() as u32, 1);
        let input = Tile::<F>::new(region, 1, input_data);
        let mut output_data = vec![0.0f32; input_data.len()];
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    fn bytes_to_u16(bytes: &[u8]) -> Vec<u16> {
        bytemuck::cast_slice(bytes).to_vec()
    }

    fn stats_from_u16_samples(samples: &[u16]) -> ImageStats {
        let count = samples.len() as f64;
        let min = f64::from(*samples.iter().min().unwrap());
        let max = f64::from(*samples.iter().max().unwrap());
        let mean = samples.iter().map(|&sample| f64::from(sample)).sum::<f64>() / count;
        let variance = samples
            .iter()
            .map(|&sample| {
                let delta = f64::from(sample) - mean;
                delta * delta
            })
            .sum::<f64>()
            / count;

        ImageStats {
            bands: 1,
            min: vec![min],
            max: vec![max],
            mean: vec![mean],
            stddev: vec![variance.sqrt()],
        }
    }

    fn run_reducer_driven_scale(mode: ScaleMode, pixels: Vec<u16>) -> (ImageStats, Vec<u8>) {
        let width = pixels.len() as u32;
        let source = MemorySource::<U16>::new(width, 1, 1, pixels).unwrap();
        let input_pipeline = PipelineBuilder::from_source(source)
            .then(Box::new(OperationBridge::new(PassThroughU16, 1)))
            .unwrap()
            .build()
            .unwrap();
        let scheduler = RayonScheduler::new(2).unwrap();
        let mut stats_sink = MemorySink::for_pipeline(&input_pipeline).unwrap();
        scheduler.run(&input_pipeline, &mut stats_sink).unwrap();
        let intermediate = bytes_to_u16(&stats_sink.into_buffer());
        let stats = stats_from_u16_samples(&intermediate);

        let scale_source = MemorySource::<U16>::new(width, 1, 1, intermediate).unwrap();
        let scale_op: Box<dyn viprs_core::op::DynOperation> = Box::new(
            OperationBridge::new_pixel_local(ScaleOp::<U16, U8>::from_stats(&stats, mode), 1),
        );
        let scale_pipeline = PipelineBuilder::from_source(scale_source)
            .then(scale_op)
            .unwrap()
            .build()
            .unwrap();
        let mut output_sink = MemorySink::for_pipeline(&scale_pipeline).unwrap();
        scheduler.run(&scale_pipeline, &mut output_sink).unwrap();

        (stats, output_sink.into_buffer())
    }

    #[test]
    fn from_stats_uses_global_min_and_max_across_bands() {
        let stats = ImageStats {
            bands: 2,
            min: vec![10.0, -5.0],
            max: vec![30.0, 40.0],
            mean: vec![20.0, 17.5],
            stddev: vec![5.0, 11.0],
        };

        let op = ScaleOp::<U16, U8>::from_stats(&stats, ScaleMode::Linear);

        assert_eq!(op.min(), -5.0);
        assert_eq!(op.max(), 40.0);
    }

    #[test]
    fn linear_u8_maps_range_to_full_display() {
        let output = run_scale_u8(ScaleOp::<U16, U8>::new_linear(10.0, 30.0), &[10, 20, 30]);
        assert_eq!(output, vec![0, 128, 255]);
    }

    #[test]
    fn linear_f32_maps_range_to_unit_interval() {
        let output = run_scale_f32(ScaleOp::<U16, F32>::new_linear(10.0, 30.0), &[10, 20, 30]);
        assert!((output[0] - 0.0).abs() < 1e-6);
        assert!((output[1] - 0.5).abs() < 1e-6);
        assert!((output[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn zero_range_returns_black() {
        let output = run_scale_u8(ScaleOp::<U8, U8>::new_linear(7.0, 7.0), &[7, 7, 7]);
        assert_eq!(output, vec![0, 0, 0]);
    }

    #[test]
    fn log_mode_maps_zero_and_max_to_display_endpoints() {
        let output = run_scale_u8(
            ScaleOp::<U16, U8>::new_log(0.0, 255.0, DEFAULT_LOG_EXPONENT),
            &[0, 255],
        );
        assert_eq!(output, vec![0, 255]);
    }

    #[test]
    fn reducer_driven_linear_pipeline_maps_min_mid_and_max() {
        let (stats, output) = run_reducer_driven_scale(ScaleMode::Linear, vec![10, 20, 30]);

        assert_eq!(stats.min, vec![10.0]);
        assert_eq!(stats.max, vec![30.0]);
        assert_eq!(output, vec![0, 128, 255]);
    }

    #[test]
    fn reducer_driven_log_pipeline_matches_libvips_default_curve() {
        let (stats, output) = run_reducer_driven_scale(ScaleMode::log_default(), vec![0, 15, 255]);

        let expected_mid = (((1.0 + 15.0_f64.powf(DEFAULT_LOG_EXPONENT)).log10()
            / (1.0 + 255.0_f64.powf(DEFAULT_LOG_EXPONENT)).log10())
            * 255.0
            + 0.5)
            .floor() as u8;

        assert_eq!(stats.min, vec![0.0]);
        assert_eq!(stats.max, vec![255.0]);
        assert_eq!(output, vec![0, expected_mid, 255]);
    }

    proptest! {
        #[test]
        fn uniform_midpoint_preserves_display_value(len in 1usize..=128) {
            let samples = vec![127u8; len];
            let output = run_scale_u8(ScaleOp::<U8, U8>::new_linear(0.0, 255.0), &samples);
            prop_assert!(output.iter().all(|&value| value == 127));
        }

        #[test]
        fn uniform_maximum_preserves_display_value(len in 1usize..=128) {
            let samples = vec![255u8; len];
            let output = run_scale_u8(ScaleOp::<U8, U8>::new_linear(0.0, 255.0), &samples);
            prop_assert!(output.iter().all(|&value| value == 255));
        }

        #[test]
        fn boundary_values_map_to_display_endpoints(min in 0u16..=1024, delta in 1u16..=1024) {
            let max = min.saturating_add(delta);
            let output = run_scale_u8(ScaleOp::<U16, U8>::new_linear(f64::from(min), f64::from(max)), &[min, max]);
            prop_assert_eq!(output, vec![0, 255]);
        }
    }
}

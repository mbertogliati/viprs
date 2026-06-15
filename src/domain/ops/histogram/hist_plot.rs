use std::{any::Any, marker::PhantomData};

use bytemuck::{Pod, cast_slice, cast_slice_mut};

use crate::domain::{
    error::ViprsError,
    format::{BandFormat, BandFormatId},
    image::{DemandHint, Region},
    op::{DynOperation, NodeSpec},
};

/// Render a histogram image as a bar plot, matching libvips `hist_plot`.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::histogram::hist_plot::HistPlotOp;
///
/// let op = HistPlotOp { /* operation parameters */ };
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct HistPlotOp<F: BandFormat> {
    bands: u32,
    plot_width: u32,
    plot_height: u32,
    vertical: bool,
    render_black: bool,
    transform: HistPlotTransform,
    _phantom: PhantomData<F>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum HistPlotTransform {
    Identity,
    Offset(f64),
    ScaleOffset { scale: f64, offset: f64 },
    Black,
}

impl<F> HistPlotOp<F>
where
    F: BandFormat,
    F::Sample: HistPlotValue + Pod,
{
    /// Creates this value from histogram.
    pub fn from_histogram(
        width: u32,
        height: u32,
        bands: u32,
        data: &[F::Sample],
    ) -> Result<Self, ViprsError> {
        if bands == 0 {
            return Err(ViprsError::Scheduler(
                "hist_plot requires at least one band".into(),
            ));
        }

        let pixel_count = u64::from(width)
            .checked_mul(u64::from(height))
            .ok_or_else(|| hist_plot_image_too_large(width, height, bands))?;
        if matches!(F::ID, BandFormatId::F32 | BandFormatId::F64)
            && pixel_count > u64::from(u32::MAX)
        {
            return Err(hist_plot_image_too_large(width, height, bands));
        }

        let expected = usize::try_from(pixel_count)
            .ok()
            .and_then(|pixels| pixels.checked_mul(bands as usize))
            .ok_or_else(|| ViprsError::ImageTooLarge {
                width,
                height,
                bands,
                bytes: u128::from(width)
                    * u128::from(height)
                    * u128::from(bands)
                    * std::mem::size_of::<F::Sample>() as u128,
                limit_bytes: usize::MAX as u128,
                details: "hist_plot histogram dimensions exceed addressable memory",
            })?;
        if data.len() != expected {
            return Err(ViprsError::Scheduler(format!(
                "hist_plot histogram data length {} does not match {}x{}x{}={expected}",
                data.len(),
                width,
                height,
                bands
            )));
        }

        let all_zero = data.iter().all(|&sample| sample.hist_plot_value() == 0.0);

        let (transform, max_value) = match F::ID {
            BandFormatId::I16 | BandFormatId::I32 => {
                let min = data
                    .iter()
                    .map(|&sample| sample.hist_plot_value())
                    .fold(f64::INFINITY, f64::min);
                let max = data
                    .iter()
                    .map(|&sample| sample.hist_plot_value() - min)
                    .fold(0.0, f64::max);
                (HistPlotTransform::Offset(-min), max)
            }
            BandFormatId::F32 | BandFormatId::F64 => {
                let min = data
                    .iter()
                    .map(|&sample| sample.hist_plot_value())
                    .fold(f64::INFINITY, f64::min);
                let max = data
                    .iter()
                    .map(|&sample| sample.hist_plot_value())
                    .fold(f64::NEG_INFINITY, f64::max);
                if (max - min).abs() > 0.01 {
                    let any = pixel_count as f64;
                    let scale = any / (max - min);
                    (
                        HistPlotTransform::ScaleOffset {
                            scale,
                            offset: -min * scale,
                        },
                        any,
                    )
                } else {
                    (HistPlotTransform::Black, 0.0)
                }
            }
            BandFormatId::U8 => (HistPlotTransform::Identity, f64::from(u8::MAX)),
            _ => {
                let max = data
                    .iter()
                    .map(|&sample| sample.hist_plot_value())
                    .fold(0.0, f64::max);
                (HistPlotTransform::Identity, max)
            }
        };

        let mut tsize = if F::ID == BandFormatId::U8 {
            256
        } else {
            max_value.ceil() as u32
        };
        if tsize == 0 {
            tsize = 1;
        }

        let vertical = width == 1;
        let plot_width = if vertical { tsize } else { width };
        let plot_height = if vertical { height } else { tsize };

        Ok(Self {
            bands,
            plot_width,
            plot_height,
            vertical,
            render_black: all_zero || max_value == 0.0,
            transform,
            _phantom: PhantomData,
        })
    }

    #[must_use]
    /// Returns or performs plot width.
    pub const fn plot_width(&self) -> u32 {
        self.plot_width
    }

    #[must_use]
    /// Returns or performs plot height.
    pub const fn plot_height(&self) -> u32 {
        self.plot_height
    }

    fn value(&self, sample: F::Sample) -> f64 {
        let value = sample.hist_plot_value();
        match self.transform {
            HistPlotTransform::Identity => value,
            HistPlotTransform::Offset(offset) => value + offset,
            HistPlotTransform::ScaleOffset { scale, offset } => value.mul_add(scale, offset),
            HistPlotTransform::Black => 0.0,
        }
    }
}

fn hist_plot_image_too_large(width: u32, height: u32, bands: u32) -> ViprsError {
    ViprsError::ImageTooLarge {
        width,
        height,
        bands,
        bytes: u128::from(width) * u128::from(height) * u128::from(bands),
        limit_bytes: u128::from(u32::MAX),
        details: "hist_plot scaling area exceeds the u32::MAX pixel limit",
    }
}

impl<F> DynOperation for HistPlotOp<F>
where
    F: BandFormat,
    F::Sample: HistPlotValue + Pod,
{
    fn input_format(&self) -> BandFormatId {
        F::ID
    }

    fn output_format(&self) -> BandFormatId {
        BandFormatId::U8
    }

    fn bands(&self) -> u32 {
        self.bands
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        if self.vertical {
            Region::new(0, output.y, 1, output.height)
        } else {
            Region::new(output.x, 0, output.width, 1)
        }
    }

    fn output_width(&self, _input_w: u32) -> u32 {
        self.plot_width
    }

    fn output_height(&self, _input_h: u32) -> u32 {
        self.plot_height
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        if self.vertical {
            NodeSpec {
                input_tile_w: 1,
                input_tile_h: tile_h,
                output_tile_w: tile_w,
                output_tile_h: tile_h,
                coordinate_driven_source: None,
            }
        } else {
            NodeSpec {
                input_tile_w: tile_w,
                input_tile_h: 1,
                output_tile_w: tile_w,
                output_tile_h: tile_h,
                coordinate_driven_source: None,
            }
        }
    }

    fn dyn_start(&self) -> Box<dyn Any + Send> {
        Box::new(())
    }

    #[inline]
    fn dyn_process_region(
        &self,
        _state: &mut dyn Any,
        input: &[u8],
        output: &mut [u8],
        _input_region: Region,
        output_region: Region,
    ) {
        let input_samples = cast_slice::<u8, F::Sample>(input);
        let output_samples = cast_slice_mut::<u8, u8>(output);

        if self.render_black {
            output_samples.fill(0);
            return;
        }

        let out_width = output_region.width as usize;
        let out_height = output_region.height as usize;
        let bands = self.bands as usize;

        if self.vertical {
            for row in 0..out_height {
                let src_base = row * bands;
                for col in 0..out_width {
                    let x = (output_region.x + col as i32).max(0) as u32;
                    let threshold = f64::from(x);
                    let dst_base = (row * out_width + col) * bands;
                    for band in 0..bands {
                        output_samples[dst_base + band] =
                            if self.value(input_samples[src_base + band]) < threshold {
                                0
                            } else {
                                255
                            };
                    }
                }
            }
        } else {
            for row in 0..out_height {
                let y = (output_region.y + row as i32).max(0) as u32;
                let threshold = f64::from(self.plot_height.saturating_sub(y));
                for col in 0..out_width {
                    let src_base = col * bands;
                    let dst_base = (row * out_width + col) * bands;
                    for band in 0..bands {
                        output_samples[dst_base + band] =
                            if self.value(input_samples[src_base + band]) < threshold {
                                0
                            } else {
                                255
                            };
                    }
                }
            }
        }
    }
}

/// Defines the contract for hist plot value.
pub trait HistPlotValue {
    /// Returns or performs hist plot value.
    fn hist_plot_value(self) -> f64;
}

impl HistPlotValue for u8 {
    fn hist_plot_value(self) -> f64 {
        f64::from(self)
    }
}

impl HistPlotValue for u16 {
    fn hist_plot_value(self) -> f64 {
        f64::from(self)
    }
}

impl HistPlotValue for i16 {
    fn hist_plot_value(self) -> f64 {
        f64::from(self)
    }
}

impl HistPlotValue for u32 {
    fn hist_plot_value(self) -> f64 {
        f64::from(self)
    }
}

impl HistPlotValue for i32 {
    fn hist_plot_value(self) -> f64 {
        f64::from(self)
    }
}

impl HistPlotValue for f32 {
    fn hist_plot_value(self) -> f64 {
        f64::from(self)
    }
}

impl HistPlotValue for f64 {
    fn hist_plot_value(self) -> f64 {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::format::{F32, I16, U8, U16, U32};
    use proptest::prelude::*;

    fn run_horizontal_u32(histogram: &[u32]) -> (HistPlotOp<U32>, Vec<u8>) {
        let op =
            HistPlotOp::<U32>::from_histogram(histogram.len() as u32, 1, 1, histogram).unwrap();
        let region = Region::new(0, 0, histogram.len() as u32, 1);
        let output_region = Region::new(0, 0, op.plot_width(), op.plot_height());
        let mut output_data = vec![0u8; output_region.pixel_count()];
        let mut state = op.dyn_start();
        op.dyn_process_region(
            state.as_mut(),
            cast_slice(histogram),
            &mut output_data,
            region,
            output_region,
        );
        (op, output_data)
    }

    #[test]
    fn horizontal_plot_renders_bottom_aligned_bars() {
        let (op, output) = run_horizontal_u32(&[0, 2, 4]);

        assert_eq!(op.plot_width(), 3);
        assert_eq!(op.plot_height(), 4);
        assert_eq!(
            output,
            vec![
                0, 0, 255, //
                0, 0, 255, //
                0, 255, 255, //
                0, 255, 255,
            ]
        );
    }

    #[test]
    fn float_histogram_with_zero_range_returns_black_plot() {
        let histogram = [5.0f32, 5.0];
        let op = HistPlotOp::<F32>::from_histogram(2, 1, 1, &histogram).unwrap();
        let region = Region::new(0, 0, 2, 1);
        let output_region = Region::new(0, 0, op.plot_width(), op.plot_height());
        let mut output_data = vec![255u8; output_region.pixel_count()];
        let mut state = op.dyn_start();

        op.dyn_process_region(
            state.as_mut(),
            cast_slice(&histogram),
            &mut output_data,
            region,
            output_region,
        );

        assert_eq!(op.plot_height(), 1);
        assert_eq!(op.value(histogram[0]), 0.0);
        assert!(output_data.iter().all(|&value| value == 0));
    }

    #[test]
    fn single_bin_histogram_renders_a_one_pixel_vertical_plot() {
        let histogram = [1u32];
        let op = HistPlotOp::<U32>::from_histogram(1, 1, 1, &histogram).unwrap();
        let region = Region::new(0, 0, 1, 1);
        let output_region = Region::new(0, 0, op.plot_width(), op.plot_height());
        let mut output_data = vec![0u8; output_region.pixel_count()];
        let mut state = op.dyn_start();

        op.dyn_process_region(
            state.as_mut(),
            cast_slice(&histogram),
            &mut output_data,
            region,
            output_region,
        );

        assert_eq!(op.plot_width(), 1);
        assert_eq!(op.plot_height(), 1);
        assert_eq!(output_data, vec![255]);
    }

    #[test]
    fn vertical_plot_uses_column_thresholds() {
        let histogram = [0u16, 3];
        let op = HistPlotOp::<U16>::from_histogram(1, 2, 1, &histogram).unwrap();
        let region = Region::new(0, 0, 1, 2);
        let output_region = Region::new(0, 0, op.plot_width(), op.plot_height());
        let mut output_data = vec![0u8; output_region.pixel_count()];
        let mut state = op.dyn_start();

        op.dyn_process_region(
            state.as_mut(),
            cast_slice(&histogram),
            &mut output_data,
            region,
            output_region,
        );

        assert_eq!(op.plot_width(), 3);
        assert_eq!(op.plot_height(), 2);
        assert_eq!(output_data, vec![255, 0, 0, 255, 255, 255]);
    }

    #[test]
    fn vertical_zero_histogram_renders_black() {
        let histogram = [0u32, 0];
        let op = HistPlotOp::<U32>::from_histogram(1, 2, 1, &histogram).unwrap();
        let region = Region::new(0, 0, 1, 2);
        let output_region = Region::new(0, 0, op.plot_width(), op.plot_height());
        let mut output_data = vec![255u8; output_region.pixel_count()];
        let mut state = op.dyn_start();

        op.dyn_process_region(
            state.as_mut(),
            cast_slice(&histogram),
            &mut output_data,
            region,
            output_region,
        );

        assert_eq!(op.plot_width(), 1);
        assert_eq!(op.plot_height(), 2);
        assert_eq!(output_data, vec![0, 0]);
    }

    #[test]
    fn dyn_metadata_reports_plot_geometry() {
        let histogram = [0u32, 2, 4];
        let op = HistPlotOp::<U32>::from_histogram(3, 1, 1, &histogram).unwrap();

        assert_eq!(op.demand_hint(), DemandHint::Any);
        assert_eq!(op.output_width(3), 3);
        assert_eq!(op.output_height(1), 4);
        assert_eq!(op.output_format(), BandFormatId::U8);
        assert_eq!(
            op.required_input_region(&Region::new(1, 2, 2, 3)),
            Region::new(1, 0, 2, 1)
        );
    }

    #[test]
    fn signed_histogram_offsets_negative_values_before_plotting() {
        let histogram = [-2i16, 1, 3];
        let op = HistPlotOp::<I16>::from_histogram(3, 1, 1, &histogram).unwrap();
        let region = Region::new(0, 0, 3, 1);
        let output_region = Region::new(0, 0, op.plot_width(), op.plot_height());
        let mut output_data = vec![0u8; output_region.pixel_count()];
        let mut state = op.dyn_start();

        op.dyn_process_region(
            state.as_mut(),
            cast_slice(&histogram),
            &mut output_data,
            region,
            output_region,
        );

        assert_eq!(op.plot_height(), 5);
        assert_eq!(op.value(histogram[0]), 0.0);
        assert_eq!(op.value(histogram[2]), 5.0);
        assert!(output_data.iter().any(|&sample| sample == 255));
    }

    #[test]
    fn floating_histogram_scales_into_output_extent() {
        let histogram = [2.0f32, 4.0, 6.0];
        let op = HistPlotOp::<F32>::from_histogram(3, 1, 1, &histogram).unwrap();

        assert_eq!(op.plot_height(), 3);
        assert!((op.value(histogram[0]) - 0.0).abs() < 1e-6);
        assert!((op.value(histogram[2]) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn floating_histogram_rejects_scaling_area_that_overflows_u32() {
        let err = match HistPlotOp::<F32>::from_histogram(65_536, 65_536, 1, &[]) {
            Ok(_) => panic!("hist_plot must reject scaling areas that overflow u32"),
            Err(err) => err,
        };

        assert!(matches!(err, ViprsError::ImageTooLarge { .. }));
    }

    #[test]
    fn histogram_rejects_dimensions_that_exceed_addressable_memory() {
        let err = match HistPlotOp::<U32>::from_histogram(u32::MAX, u32::MAX, u32::MAX, &[]) {
            Ok(_) => panic!("hist_plot must reject histograms that exceed addressable memory"),
            Err(err) => err,
        };

        assert!(matches!(
            err,
            ViprsError::ImageTooLarge {
                width: u32::MAX,
                height: u32::MAX,
                bands: u32::MAX,
                details: "hist_plot histogram dimensions exceed addressable memory",
                ..
            }
        ));
    }

    #[test]
    fn histogram_rejects_mismatched_data_length() {
        let err = match HistPlotOp::<U16>::from_histogram(2, 1, 2, &[1u16, 2, 3]) {
            Ok(_) => panic!("hist_plot must reject mismatched histogram data lengths"),
            Err(err) => err,
        };

        assert!(matches!(
            err,
            ViprsError::Scheduler(ref message)
            if message == "hist_plot histogram data length 3 does not match 2x1x2=4"
        ));
    }

    #[test]
    fn vertical_plot_clamps_negative_output_x_to_zero_threshold() {
        let histogram = [1u16, 3];
        let op = HistPlotOp::<U16>::from_histogram(1, 2, 1, &histogram).unwrap();
        let region = Region::new(0, 0, 1, 2);
        let output_region = Region::new(-2, 0, 2, op.plot_height());
        let mut output_data = vec![0u8; output_region.pixel_count()];
        let mut state = op.dyn_start();

        op.dyn_process_region(
            state.as_mut(),
            cast_slice(&histogram),
            &mut output_data,
            region,
            output_region,
        );

        assert_eq!(output_data, vec![255, 255, 255, 255]);
    }

    #[test]
    fn horizontal_threshold_treats_equal_values_as_filled_pixels() {
        let histogram = [1u32, 2];
        let op = HistPlotOp::<U32>::from_histogram(2, 1, 1, &histogram).unwrap();
        let region = Region::new(0, 0, 2, 1);
        let output_region = Region::new(0, 1, op.plot_width(), 1);
        let mut output_data = vec![0u8; output_region.pixel_count()];
        let mut state = op.dyn_start();

        op.dyn_process_region(
            state.as_mut(),
            cast_slice(&histogram),
            &mut output_data,
            region,
            output_region,
        );

        assert_eq!(output_data, vec![255, 255]);
    }

    #[test]
    fn horizontal_plot_clamps_negative_output_y_to_top_threshold() {
        let histogram = [1u32, 2, 3];
        let op = HistPlotOp::<U32>::from_histogram(3, 1, 1, &histogram).unwrap();
        let region = Region::new(0, 0, 3, 1);
        let output_region = Region::new(0, -2, 3, 2);
        let mut output_data = vec![0u8; output_region.pixel_count()];
        let mut state = op.dyn_start();

        op.dyn_process_region(
            state.as_mut(),
            cast_slice(&histogram),
            &mut output_data,
            region,
            output_region,
        );

        assert_eq!(output_data, vec![0, 0, 255, 0, 0, 255]);
    }

    #[test]
    fn multi_band_vertical_plot_preserves_band_separation() {
        let histogram = [1u16, 3, 2, 1];
        let op = HistPlotOp::<U16>::from_histogram(1, 2, 2, &histogram).unwrap();
        let region = Region::new(0, 0, 1, 2);
        let output_region = Region::new(0, 0, op.plot_width(), op.plot_height());
        let mut output_data = vec![0u8; output_region.pixel_count() * 2];
        let mut state = op.dyn_start();

        op.dyn_process_region(
            state.as_mut(),
            cast_slice(&histogram),
            &mut output_data,
            region,
            output_region,
        );

        assert_eq!(op.node_spec(3, 2).input_tile_w, 1);
        assert_eq!(op.input_format(), BandFormatId::U16);
        assert_eq!(op.bands(), 2);
        assert_eq!(&output_data[0..2], &[255, 255]);
        assert!(output_data.iter().any(|&sample| sample == 0));
    }

    #[test]
    fn u8_histograms_use_fixed_256_bin_extent_and_horizontal_strip_metadata() {
        let histogram = [1u8, 2, 3];
        let op = HistPlotOp::<U8>::from_histogram(3, 1, 1, &histogram).unwrap();

        assert_eq!(op.plot_width(), 3);
        assert_eq!(op.plot_height(), 256);
        assert_eq!(
            op.required_input_region(&Region::new(2, 4, 3, 5)),
            Region::new(2, 0, 3, 1)
        );
        assert_eq!(op.node_spec(3, 5).input_tile_w, 3);
        assert_eq!(op.node_spec(3, 5).input_tile_h, 1);
        assert_eq!(1u8.hist_plot_value(), 1.0);
        assert_eq!(3.5f64.hist_plot_value(), 3.5);
    }

    #[test]
    fn vertical_i32_histograms_offset_negative_values_before_thresholding() {
        let histogram = [-2i32, 3];
        let op =
            HistPlotOp::<crate::domain::format::I32>::from_histogram(1, 2, 1, &histogram).unwrap();

        assert_eq!(
            op.required_input_region(&Region::new(4, 5, 3, 2)),
            Region::new(0, 5, 1, 2)
        );
        assert_eq!(op.node_spec(3, 2).input_tile_w, 1);
        assert_eq!(op.value(histogram[0]), 0.0);
        assert_eq!(op.value(histogram[1]), 5.0);
        assert_eq!((-7i32).hist_plot_value(), -7.0);
    }

    #[test]
    fn hist_plot_rejects_zero_band_histograms() {
        let err = match HistPlotOp::<U8>::from_histogram(1, 1, 0, &[]) {
            Ok(_) => panic!("hist_plot must reject zero-band histograms"),
            Err(err) => err,
        };

        assert!(
            matches!(
                err,
                crate::domain::error::ViprsError::Scheduler(ref message)
                if message == "hist_plot requires at least one band"
            ),
            "unexpected error: {err:?}"
        );
    }

    proptest! {
        #[test]
        fn zero_histogram_renders_black(
            width in 2u32..=16,
            bins in proptest::collection::vec(Just(0u32), 2..=16)
        ) {
            let len = width.min(bins.len() as u32) as usize;
            let histogram = bins[..len].to_vec();
            let (_op, output) = run_horizontal_u32(&histogram);
            prop_assert!(output.iter().all(|&value| value == 0));
        }

        #[test]
        fn columns_are_monotonic_top_to_bottom(histogram in proptest::collection::vec(0u32..=8u32, 1..=16)) {
            let (op, output) = run_horizontal_u32(&histogram);
            let width = op.plot_width() as usize;
            let height = op.plot_height() as usize;

            for col in 0..width {
                let mut seen_white = false;
                for row in 0..height {
                    let value = output[row * width + col];
                    if value == 255 {
                        seen_white = true;
                    } else {
                        prop_assert!(!seen_white);
                    }
                }
            }
        }
    }
}

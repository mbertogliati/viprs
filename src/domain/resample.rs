//! Port traits for resampling operations.
//!
//! Resampling ops (`ReduceH`, `ReduceV`, `Resize`, `Affine`, `Thumbnail`) differ
//! from pixel-local ops in two ways:
//!
//! 1. They change the output image dimensions (`output_width`/`output_height`).
//! 2. Their input tile is larger than the output tile by a kernel-dependent tap span.
//!
//! Both properties can be expressed through `DynOperation::node_spec`,
//! `output_width`, and `output_height` — no new object-safe trait is needed.
//!
//! `ResampleOp` is a refinement of `Op` that adds dimension-change declarations.
//! It is a static (monomorphized) trait: concrete implementations use it,
//! `OperationBridge` wraps them for the dynamic pipeline.

use crate::domain::image::Region;
use crate::domain::op::{NodeSpec, Op};

/// Refinement of [`Op`] for operations that change image dimensions.
pub trait ResampleOp: Op {
    /// Output image dimensions given the input image dimensions.
    fn output_size(&self, input_w: u32, input_h: u32) -> (u32, u32) {
        (self.output_width(input_w), self.output_height(input_h))
    }

    /// Width of the output image given the input image width.
    fn output_width(&self, input_w: u32) -> u32;

    /// Height of the output image given the input image height.
    fn output_height(&self, input_h: u32) -> u32;
}

/// Orientation of a separable resampling pass.
///
/// Resamplers use this to distinguish horizontal kernels from vertical kernels while reusing the
/// same core math.
///
/// # Examples
/// ```rust
/// # use viprs::domain::resample::FilterOrientation;
/// assert!(matches!(FilterOrientation::Horizontal, FilterOrientation::Horizontal));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterOrientation {
    /// Horizontal pass: iterates over columns, stride is 1 sample.
    Horizontal,
    /// Vertical pass: iterates over rows, stride is `width` samples.
    Vertical,
}

/// Shared geometry and tap configuration for reduce-family operations.
///
/// This keeps resample planning deterministic by packaging factor- and kernel-derived sizing in a
/// reusable value.
///
/// # Examples
/// ```rust
/// # use viprs::domain::resample::ReduceConfig;
/// let cfg = ReduceConfig { factor: 2.0, taps: 4, pad_before: 1 };
/// assert_eq!(cfg.output_width(8), 4);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ReduceConfig {
    /// Reduction factor >= 1.0. `factor = 2.0` produces a half-size output.
    pub factor: f64,
    /// Number of source taps contributing to each output sample.
    pub taps: u32,
    /// Number of taps read before the integer source position.
    pub pad_before: i64,
}

impl ReduceConfig {
    #[inline]
    fn saturating_axis_bounds(start: i64, end: i64) -> (i32, u32) {
        if end < 0 {
            return (0, 0);
        }

        let clamped_start = start.max(0).min(i64::from(i32::MAX));
        let clamped_end = end.max(clamped_start).min(i64::from(i32::MAX));
        let width = clamped_end.saturating_sub(clamped_start).saturating_add(1);

        (clamped_start as i32, width as u32)
    }

    #[inline]
    fn source_position(&self, index: f64) -> f64 {
        (index + 0.5).mul_add(self.factor, -0.5)
    }

    #[inline]
    fn output_len(&self, input_len: u32) -> u32 {
        if input_len == 0 {
            0
        } else {
            (f64::from(input_len) / self.factor).round() as u32
        }
    }

    #[inline]
    /// Compute the horizontal-pass buffer geometry for a scheduler tile.
    ///
    /// This tells the pipeline how much extra source width a horizontal reduce pass needs.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs::domain::resample::ReduceConfig;
    /// let cfg = ReduceConfig { factor: 2.0, taps: 4, pad_before: 1 };
    /// assert!(cfg.node_spec_h(8, 4).input_tile_w >= 8);
    /// ```
    #[must_use]
    pub fn node_spec_h(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec {
            input_tile_w: (f64::from(tile_w) * self.factor).ceil() as u32 + self.taps,
            input_tile_h: tile_h,
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    #[inline]
    /// Compute the vertical-pass buffer geometry for a scheduler tile.
    ///
    /// This tells the pipeline how much extra source height a vertical reduce pass needs.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs::domain::resample::ReduceConfig;
    /// let cfg = ReduceConfig { factor: 2.0, taps: 4, pad_before: 1 };
    /// assert!(cfg.node_spec_v(8, 4).input_tile_h >= 4);
    /// ```
    #[must_use]
    pub fn node_spec_v(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec {
            input_tile_w: tile_w,
            input_tile_h: (f64::from(tile_h) * self.factor).ceil() as u32 + self.taps,
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    #[inline]
    /// Map a horizontal output region back to the source region it needs.
    ///
    /// This captures the tap halo required around each output column.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs::domain::{image::Region, resample::ReduceConfig};
    /// let cfg = ReduceConfig { factor: 2.0, taps: 4, pad_before: 1 };
    /// let input = cfg.required_input_region_h(&Region::new(0, 0, 2, 1));
    /// assert!(input.width >= 2);
    /// ```
    #[must_use]
    pub fn required_input_region_h(&self, output: &Region) -> Region {
        if output.width == 0 {
            return Region::new(output.x, output.y, 0, output.height);
        }

        let first_src = self.source_position(f64::from(output.x)).floor() as i64;
        let last_x = output
            .x
            .saturating_add(output.width.saturating_sub(1).min(i32::MAX as u32) as i32);
        let last_src = self.source_position(f64::from(last_x)).floor() as i64;
        let start = first_src - self.pad_before;
        let end = last_src - self.pad_before + i64::from(self.taps) - 1;
        let (x, width) = Self::saturating_axis_bounds(start, end);
        Region::new(x, output.y, width, output.height)
    }

    #[inline]
    /// Map a vertical output region back to the source region it needs.
    ///
    /// This captures the tap halo required around each output row.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs::domain::{image::Region, resample::ReduceConfig};
    /// let cfg = ReduceConfig { factor: 2.0, taps: 4, pad_before: 1 };
    /// let input = cfg.required_input_region_v(&Region::new(0, 0, 1, 2));
    /// assert!(input.height >= 2);
    /// ```
    #[must_use]
    pub fn required_input_region_v(&self, output: &Region) -> Region {
        if output.height == 0 {
            return Region::new(output.x, output.y, output.width, 0);
        }

        let first_src = self.source_position(f64::from(output.y)).floor() as i64;
        let last_y = output
            .y
            .saturating_add(output.height.saturating_sub(1).min(i32::MAX as u32) as i32);
        let last_src = self.source_position(f64::from(last_y)).floor() as i64;
        let start = first_src - self.pad_before;
        let end = last_src - self.pad_before + i64::from(self.taps) - 1;
        let (y, height) = Self::saturating_axis_bounds(start, end);
        Region::new(output.x, y, output.width, height)
    }

    #[inline]
    #[must_use]
    /// Returns or performs output width.
    pub fn output_width(&self, input_w: u32) -> u32 {
        self.output_len(input_w)
    }

    #[inline]
    #[must_use]
    /// Returns or performs output height.
    pub fn output_height(&self, input_h: u32) -> u32 {
        self.output_len(input_h)
    }
}

/// Marker type for affine and resize output-format specialization.
///
/// This keeps generic resample machinery type-safe without storing any runtime data.
///
/// # Examples
/// ```rust
/// # use viprs::domain::resample::ResampleFormatMarker;
/// let _marker = ResampleFormatMarker;
/// ```
#[allow(dead_code)]
pub struct ResampleFormatMarker;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::image::Region;

    fn make_config(factor: f64, taps: u32, pad_before: i64) -> ReduceConfig {
        ReduceConfig {
            factor,
            taps,
            pad_before,
        }
    }

    #[test]
    fn node_spec_h_factor1_uses_tap_span() {
        let cfg = make_config(1.0, 7, 3);
        let spec = cfg.node_spec_h(64, 64);
        assert_eq!(spec.output_tile_w, 64);
        assert_eq!(spec.output_tile_h, 64);
        assert_eq!(spec.input_tile_w, 71);
        assert_eq!(spec.input_tile_h, 64);
    }

    #[test]
    fn node_spec_v_factor2_uses_full_vertical_span() {
        let cfg = make_config(2.0, 9, 4);
        let spec = cfg.node_spec_v(64, 64);
        assert_eq!(spec.output_tile_w, 64);
        assert_eq!(spec.output_tile_h, 64);
        assert_eq!(spec.input_tile_w, 64);
        assert_eq!(spec.input_tile_h, 137);
    }

    #[test]
    fn required_input_region_h_tracks_fractional_span() {
        let cfg = make_config(1.5, 5, 2);
        let region = cfg.required_input_region_h(&Region::new(2, 4, 3, 1));
        assert_eq!(region, Region::new(1, 4, 8, 1));
    }

    #[test]
    fn required_input_region_v_clamps_at_origin() {
        let cfg = make_config(2.0, 13, 6);
        let region = cfg.required_input_region_v(&Region::new(7, 0, 1, 2));
        assert_eq!(region, Region::new(7, 0, 1, 9));
    }

    #[test]
    fn required_input_region_h_saturates_max_coordinates() {
        let cfg = make_config(1.5, 11, 5);
        let region = cfg.required_input_region_h(&Region::new(i32::MAX, 3, 1, 1));
        assert_eq!(region, Region::new(i32::MAX, 3, 1, 1));
    }

    #[test]
    fn required_input_region_v_saturates_max_coordinates() {
        let cfg = make_config(1.5, 11, 5);
        let region = cfg.required_input_region_v(&Region::new(7, i32::MAX, 1, 1));
        assert_eq!(region, Region::new(7, i32::MAX, 1, 1));
    }

    #[test]
    fn output_len_rounds_to_nearest() {
        let cfg = make_config(1.6, 5, 2);
        assert_eq!(cfg.output_width(5), 3);
        assert_eq!(cfg.output_height(5), 3);
    }
}

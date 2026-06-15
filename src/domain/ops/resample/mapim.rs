#![allow(clippy::no_effect_underscore_binding)]
// REASON: underscore-prefixed temporaries mark intentionally ignored interpolation intermediates.

use std::{any::Any, marker::PhantomData};

use bytemuck::{Pod, cast_slice, cast_slice_mut};

use crate::domain::{
    error::BuildError,
    format::{BandFormat, BandFormatId},
    image::{DemandHint, Region, Tile},
    kernel::InterpolationKernel,
    op::{DynOperation, NodeSpec, SourceReadPlan},
};

use super::{
    affine::Affine,
    sample_conv::{FromF64, ToF64},
};

/// How out-of-bounds source coordinates are handled by `MapImOp`.
///
/// Mirrors the `VipsExtend` variants that `vips_mapim` exposes:
/// - `Background` (`VIPS_EXTEND_BACKGROUND`): fill with a fixed background value.
/// - `Copy` (`VIPS_EXTEND_COPY`): clamp to the nearest edge pixel of the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapImExtend {
    /// Fill out-of-bounds pixels with the configured background value (default: 0.0).
    Background,
    /// Clamp out-of-bounds coordinates to the nearest edge pixel (copy-extend).
    Copy,
}

/// Coordinate-map resampling over two root sources.
///
/// Slot 0 is the source image that gets sampled. Slot 1 is the coordinate image
/// whose two bands store `(x, y)` coordinates per pixel. Output size matches the
/// index image, and output samples use the source image format.
///
/// ### Extend modes
/// - `MapImExtend::Background` (default): coordinates outside the source fill with `background`.
/// - `MapImExtend::Copy`: coordinates are clamped to `[0, width-1] × [0, height-1]`.
///
/// ### Premultiplied alpha
/// When `premultiplied` is `false` and the source uses a conventional alpha-bearing
/// layout (`2` or `4` bands), the last band is treated as alpha. The source tile is premultiplied into
/// per-thread float scratch before interpolation, then unpremultiplied after
/// sampling, matching libvips `premultiplied=false` semantics.
///
/// Defaults to `premultiplied = false`, matching libvips `vips_mapim`.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::resample::mapim::MapImOp;
///
/// let op = MapImOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct MapImOp<F: BandFormat> {
    source_width: u32,
    source_height: u32,
    index_width: u32,
    index_height: u32,
    source_bands: u32,
    index_format: BandFormatId,
    kernel: InterpolationKernel,
    background: f64,
    extend: MapImExtend,
    /// When `false` and the source has a conventional alpha channel (`2` or `4` bands),
    /// `process_typed` applies
    /// premultiply → interpolate → unpremultiply, treating the last band as alpha.
    /// Default is `false`, matching libvips `vips_mapim`.
    premultiplied: bool,
    _format: PhantomData<F>,
}

struct MapImState;

impl<F: BandFormat> MapImOp<F> {
    #[must_use]
    /// Creates a new `MapImOp`.
    pub const fn new(
        source_width: u32,
        source_height: u32,
        source_bands: u32,
        index_width: u32,
        index_height: u32,
        index_format: BandFormatId,
    ) -> Self {
        Self {
            source_width,
            source_height,
            index_width,
            index_height,
            source_bands,
            index_format,
            kernel: InterpolationKernel::Bilinear,
            background: 0.0,
            extend: MapImExtend::Background,
            premultiplied: false,
            _format: PhantomData,
        }
    }

    #[must_use]
    /// Returns this value configured with kernel.
    pub const fn with_kernel(mut self, kernel: InterpolationKernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[must_use]
    /// Returns this value configured with background.
    pub const fn with_background(mut self, background: f64) -> Self {
        self.background = background;
        self
    }

    /// Set the extend mode for out-of-bounds coordinates (default: `Background`).
    #[must_use]
    pub const fn with_extend(mut self, extend: MapImExtend) -> Self {
        self.extend = extend;
        self
    }

    /// Declare that the source image is already premultiplied.
    ///
    /// When `true`, the premultiply/unpremultiply pass in `process_typed` is
    /// skipped. When `false` (default), it is applied automatically for
    /// multi-band images to match libvips semantics.
    #[must_use]
    pub const fn with_premultiplied(mut self, premultiplied: bool) -> Self {
        self.premultiplied = premultiplied;
        self
    }

    #[inline]
    #[must_use]
    /// Returns or performs source bands.
    pub const fn source_bands(&self) -> u32 {
        self.source_bands
    }

    #[inline]
    #[must_use]
    /// Returns or performs index format.
    pub const fn index_format(&self) -> BandFormatId {
        self.index_format
    }

    #[inline]
    const fn source_region(&self) -> Region {
        Region::new(0, 0, self.source_width, self.source_height)
    }

    #[inline]
    const fn has_alpha(&self) -> bool {
        matches!(self.source_bands, 2 | 4)
    }

    #[inline]
    fn source_region_for_output(&self, output: &Region) -> Region {
        if output.is_empty() {
            return Region::new(0, 0, 0, 0);
        }

        let (left_pad, right_pad) = self.kernel.affine_padding();
        let x0 = output.x - left_pad;
        let y0 = output.y - left_pad;
        let x1 = output.x + output.width as i32 - 1 + right_pad;
        let y1 = output.y + output.height as i32 - 1 + right_pad;

        Region::new(x0, y0, (x1 - x0 + 1) as u32, (y1 - y0 + 1) as u32)
            .clip_to(self.source_width, self.source_height)
    }

    #[inline]
    fn bounded_source_region_from_coord_bounds(
        &self,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
    ) -> Region {
        if !min_x.is_finite() || !max_x.is_finite() || !min_y.is_finite() || !max_y.is_finite() {
            return Region::new(0, 0, 0, 0);
        }

        let (min_x, max_x, min_y, max_y) = match self.extend {
            MapImExtend::Background => (min_x, max_x, min_y, max_y),
            MapImExtend::Copy => (
                Self::clamp_x(min_x, self.source_width),
                Self::clamp_x(max_x, self.source_width),
                Self::clamp_y(min_y, self.source_height),
                Self::clamp_y(max_y, self.source_height),
            ),
        };

        let (left_pad, right_pad) = self.kernel.affine_padding();
        let x0 = min_x.floor() as i32 - left_pad;
        let x1 = max_x.floor() as i32 + right_pad;
        let y0 = min_y.floor() as i32 - left_pad;
        let y1 = max_y.floor() as i32 + right_pad;

        Region::new(x0, y0, (x1 - x0 + 1) as u32, (y1 - y0 + 1) as u32)
            .clip_to(self.source_width, self.source_height)
    }

    #[inline]
    fn bounded_source_region_from_index_tile<I>(&self, index: &[I]) -> Region
    where
        I: Pod + Copy + ToF64,
    {
        if index.is_empty() {
            return Region::new(0, 0, 0, 0);
        }

        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        let mut saw_finite = false;

        for pair in index.chunks_exact(2) {
            let x = pair[0].to_f64();
            let y = pair[1].to_f64();
            if !x.is_finite() || !y.is_finite() {
                continue;
            }

            saw_finite = true;
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }

        if !saw_finite {
            return Region::new(0, 0, 0, 0);
        }

        self.bounded_source_region_from_coord_bounds(min_x, max_x, min_y, max_y)
    }

    fn bounded_source_region_from_dependency_bytes(
        &self,
        dependency_slot: usize,
        dependency: &[u8],
    ) -> Option<Region> {
        if dependency_slot != 1 {
            return None;
        }

        match self.index_format {
            BandFormatId::U8 => {
                Some(self.bounded_source_region_from_index_tile(cast_slice::<u8, u8>(dependency)))
            }
            BandFormatId::U16 => {
                Some(self.bounded_source_region_from_index_tile(cast_slice::<u8, u16>(dependency)))
            }
            BandFormatId::I16 => {
                Some(self.bounded_source_region_from_index_tile(cast_slice::<u8, i16>(dependency)))
            }
            BandFormatId::U32 => {
                Some(self.bounded_source_region_from_index_tile(cast_slice::<u8, u32>(dependency)))
            }
            BandFormatId::I32 => {
                Some(self.bounded_source_region_from_index_tile(cast_slice::<u8, i32>(dependency)))
            }
            BandFormatId::F32 => {
                Some(self.bounded_source_region_from_index_tile(cast_slice::<u8, f32>(dependency)))
            }
            BandFormatId::F64 => {
                Some(self.bounded_source_region_from_index_tile(cast_slice::<u8, f64>(dependency)))
            }
        }
    }

    /// Maximum representable value for the sample type used as the alpha divisor.
    ///
    /// For integer formats this is the type max cast to f64. For float formats
    /// the premultiplied range is `[0, 1]`, so `max_alpha = 1.0`.
    #[inline]
    fn alpha_max() -> f64
    where
        F::Sample: ToF64 + FromF64,
    {
        match F::ID {
            BandFormatId::F32 | BandFormatId::F64 => 1.0,
            _ => F::Sample::from_f64(f64::MAX).to_f64(),
        }
    }

    /// Clamp a source coordinate for `Copy` extend mode.
    #[inline]
    fn clamp_x(x: f64, w: u32) -> f64 {
        x.clamp(0.0, f64::from(w) - 1.0)
    }

    #[inline]
    fn clamp_y(y: f64, h: u32) -> f64 {
        y.clamp(0.0, f64::from(h) - 1.0)
    }

    #[inline]
    fn resolve_source_coord(&self, coord_x: i64, coord_y: i64) -> Option<(i32, i32)> {
        match self.extend {
            MapImExtend::Background => {
                if coord_x < 0
                    || coord_x >= i64::from(self.source_width)
                    || coord_y < 0
                    || coord_y >= i64::from(self.source_height)
                {
                    None
                } else {
                    Some((coord_x as i32, coord_y as i32))
                }
            }
            MapImExtend::Copy => {
                if self.source_width == 0 || self.source_height == 0 {
                    None
                } else {
                    let (left_pad, right_pad) = self.kernel.affine_padding();
                    let left_limit = -(i64::from(left_pad) + 1);
                    let right_limit = i64::from(self.source_width) + i64::from(right_pad);
                    let top_limit = -(i64::from(left_pad) + 1);
                    let bottom_limit = i64::from(self.source_height) + i64::from(right_pad);
                    if coord_x < left_limit
                        || coord_x > right_limit
                        || coord_y < top_limit
                        || coord_y > bottom_limit
                    {
                        return None;
                    }

                    Some((
                        coord_x.clamp(0, i64::from(self.source_width) - 1) as i32,
                        coord_y.clamp(0, i64::from(self.source_height) - 1) as i32,
                    ))
                }
            }
        }
    }

    #[inline]
    fn sample_source_band<S>(
        &self,
        source_region: Region,
        source: &[S],
        ix: i64,
        iy: i64,
        band: usize,
    ) -> f64
    where
        S: Copy + ToF64,
    {
        let Some((sample_x, sample_y)) = self.resolve_source_coord(ix, iy) else {
            return self.background;
        };

        let source_right = source_region.x + source_region.width as i32;
        let source_bottom = source_region.y + source_region.height as i32;
        debug_assert!(
            sample_x >= source_region.x
                && sample_x < source_right
                && sample_y >= source_region.y
                && sample_y < source_bottom,
            "MapIm source read plan missed required source samples"
        );
        if sample_x < source_region.x
            || sample_x >= source_right
            || sample_y < source_region.y
            || sample_y >= source_bottom
        {
            return self.background;
        }

        let bands = self.source_bands as usize;
        let source_width = source_region.width as usize;
        let local_x = (sample_x - source_region.x) as usize;
        let local_y = (sample_y - source_region.y) as usize;
        let idx = (local_y * source_width + local_x) * bands + band;
        source[idx].to_f64()
    }

    #[inline]
    fn bilinear_mix(p00: f64, p01: f64, p10: f64, p11: f64, fx: f64, fy: f64) -> f64 {
        let top = p01.mul_add(fx, p00 * (1.0 - fx));
        let bottom = p11.mul_add(fx, p10 * (1.0 - fx));
        top * (1.0 - fy) + bottom * fy
    }

    #[inline]
    fn lbb_min(x: f64, y: f64) -> f64 {
        if x <= y { x } else { y }
    }

    #[inline]
    fn lbb_max(x: f64, y: f64) -> f64 {
        if x >= y { x } else { y }
    }

    #[inline]
    fn lbb_sign(x: f64) -> f64 {
        if x >= 0.0 { 1.0 } else { -1.0 }
    }

    #[inline]
    fn lbbicubic(stencil: &[f64; 16], relative_x: f64, relative_y: f64) -> f64 {
        let [
            uno_one,
            uno_two,
            uno_thr,
            uno_fou,
            dos_one,
            dos_two,
            dos_thr,
            dos_fou,
            tre_one,
            tre_two,
            tre_thr,
            tre_fou,
            qua_one,
            qua_two,
            qua_thr,
            qua_fou,
        ] = *stencil;

        let m1 = if dos_two <= dos_thr { dos_two } else { dos_thr };
        let max1 = if dos_two <= dos_thr { dos_thr } else { dos_two };
        let m2 = if tre_two <= tre_thr { tre_two } else { tre_thr };
        let max2 = if tre_two <= tre_thr { tre_thr } else { tre_two };
        let m3 = if uno_two <= dos_one { uno_two } else { dos_one };
        let max3 = if uno_two <= dos_one { dos_one } else { uno_two };
        let m4 = if uno_thr <= dos_fou { uno_thr } else { dos_fou };
        let max4 = if uno_thr <= dos_fou { dos_fou } else { uno_thr };
        let m5 = if tre_one <= qua_two { tre_one } else { qua_two };
        let max5 = if tre_one <= qua_two { qua_two } else { tre_one };
        let m6 = if tre_fou <= qua_thr { tre_fou } else { qua_thr };
        let max6 = if tre_fou <= qua_thr { qua_thr } else { tre_fou };
        let m7 = Self::lbb_min(m1, tre_two);
        let max7 = Self::lbb_max(max1, tre_two);
        let m8 = Self::lbb_min(m1, tre_thr);
        let max8 = Self::lbb_max(max1, tre_thr);
        let m9 = Self::lbb_min(m2, dos_two);
        let max9 = Self::lbb_max(max2, dos_two);
        let m10 = Self::lbb_min(m2, dos_thr);
        let max10 = Self::lbb_max(max2, dos_thr);
        let min00 = Self::lbb_min(m7, m3);
        let max00 = Self::lbb_max(max7, max3);
        let min10 = Self::lbb_min(m8, m4);
        let max10_corner = Self::lbb_max(max8, max4);
        let min01 = Self::lbb_min(m9, m5);
        let max01 = Self::lbb_max(max9, max5);
        let min11 = Self::lbb_min(m10, m6);
        let max11 = Self::lbb_max(max10, max6);

        let u00 = dos_two - min00;
        let v00 = max00 - dos_two;
        let u10 = dos_thr - min10;
        let v10 = max10_corner - dos_thr;
        let u01 = tre_two - min01;
        let v01 = max01 - tre_two;
        let u11 = tre_thr - min11;
        let v11 = max11 - tre_thr;

        let dble_dzdx00i = dos_thr - dos_one;
        let dble_dzdy11i = qua_thr - dos_thr;
        let dble_dzdx10i = dos_fou - dos_two;
        let dble_dzdy01i = qua_two - dos_two;
        let dble_dzdx01i = tre_thr - tre_one;
        let dble_dzdy10i = tre_thr - uno_thr;
        let dble_dzdx11i = tre_fou - tre_two;
        let dble_dzdy00i = tre_two - uno_two;

        let sign_dzdx00 = Self::lbb_sign(dble_dzdx00i);
        let sign_dzdx10 = Self::lbb_sign(dble_dzdx10i);
        let sign_dzdx01 = Self::lbb_sign(dble_dzdx01i);
        let sign_dzdx11 = Self::lbb_sign(dble_dzdx11i);
        let sign_dzdy00 = Self::lbb_sign(dble_dzdy00i);
        let sign_dzdy10 = Self::lbb_sign(dble_dzdy10i);
        let sign_dzdy01 = Self::lbb_sign(dble_dzdy01i);
        let sign_dzdy11 = Self::lbb_sign(dble_dzdy11i);

        let quad_d2zdxdy00i = uno_one - uno_thr + dble_dzdx01i;
        let quad_d2zdxdy10i = uno_two - uno_fou + dble_dzdx11i;
        let quad_d2zdxdy01i = qua_thr - qua_one - dble_dzdx00i;
        let quad_d2zdxdy11i = qua_fou - qua_two - dble_dzdx10i;

        let dble_slopelimit_00 = 6.0 * Self::lbb_min(u00, v00);
        let dble_slopelimit_10 = 6.0 * Self::lbb_min(u10, v10);
        let dble_slopelimit_01 = 6.0 * Self::lbb_min(u01, v01);
        let dble_slopelimit_11 = 6.0 * Self::lbb_min(u11, v11);

        let dble_dzdx00 = if sign_dzdx00 * dble_dzdx00i <= dble_slopelimit_00 {
            dble_dzdx00i
        } else {
            sign_dzdx00 * dble_slopelimit_00
        };
        let dble_dzdy00 = if sign_dzdy00 * dble_dzdy00i <= dble_slopelimit_00 {
            dble_dzdy00i
        } else {
            sign_dzdy00 * dble_slopelimit_00
        };
        let dble_dzdx10 = if sign_dzdx10 * dble_dzdx10i <= dble_slopelimit_10 {
            dble_dzdx10i
        } else {
            sign_dzdx10 * dble_slopelimit_10
        };
        let dble_dzdy10 = if sign_dzdy10 * dble_dzdy10i <= dble_slopelimit_10 {
            dble_dzdy10i
        } else {
            sign_dzdy10 * dble_slopelimit_10
        };
        let dble_dzdx01 = if sign_dzdx01 * dble_dzdx01i <= dble_slopelimit_01 {
            dble_dzdx01i
        } else {
            sign_dzdx01 * dble_slopelimit_01
        };
        let dble_dzdy01 = if sign_dzdy01 * dble_dzdy01i <= dble_slopelimit_01 {
            dble_dzdy01i
        } else {
            sign_dzdy01 * dble_slopelimit_01
        };
        let dble_dzdx11 = if sign_dzdx11 * dble_dzdx11i <= dble_slopelimit_11 {
            dble_dzdx11i
        } else {
            sign_dzdx11 * dble_slopelimit_11
        };
        let dble_dzdy11 = if sign_dzdy11 * dble_dzdy11i <= dble_slopelimit_11 {
            dble_dzdy11i
        } else {
            sign_dzdy11 * dble_slopelimit_11
        };

        let twelve_sum00 = 6.0 * (dble_dzdx00 + dble_dzdy00);
        let _twelve_dif00 = 6.0 * (dble_dzdx00 - dble_dzdy00);
        let _twelve_sum10 = 6.0 * (dble_dzdx10 + dble_dzdy10);
        let twelve_dif10 = 6.0 * (dble_dzdx10 - dble_dzdy10);
        let _twelve_sum01 = 6.0 * (dble_dzdx01 + dble_dzdy01);
        let twelve_dif01 = 6.0 * (dble_dzdx01 - dble_dzdy01);
        let twelve_sum11 = 6.0 * (dble_dzdx11 + dble_dzdy11);
        let _twelve_dif11 = 6.0 * (dble_dzdx11 - dble_dzdy11);

        let quad_d2zdxdy00 = if twelve_sum00 * quad_d2zdxdy00i <= dble_slopelimit_00 {
            quad_d2zdxdy00i
        } else {
            sign_dzdx00 * sign_dzdy00 * dble_slopelimit_00
        };
        let quad_d2zdxdy10 = if twelve_dif10 * quad_d2zdxdy10i <= dble_slopelimit_10 {
            quad_d2zdxdy10i
        } else {
            sign_dzdx10 * sign_dzdy10 * dble_slopelimit_10
        };
        let quad_d2zdxdy01 = if twelve_dif01 * quad_d2zdxdy01i <= dble_slopelimit_01 {
            quad_d2zdxdy01i
        } else {
            sign_dzdx01 * sign_dzdy01 * dble_slopelimit_01
        };
        let quad_d2zdxdy11 = if twelve_sum11 * quad_d2zdxdy11i <= dble_slopelimit_11 {
            quad_d2zdxdy11i
        } else {
            sign_dzdx11 * sign_dzdy11 * dble_slopelimit_11
        };

        let half_d2zdx2 = 6.0f64.mul_add(-dos_two, 3.0 * (dos_thr + dos_one));
        let half_d2zdy2 = 6.0f64.mul_add(-dos_two, 3.0 * (tre_two + uno_two));
        let twelveth_d2zdxdxdy = 3.0f64.mul_add(dble_dzdy10 - dble_dzdy00, -half_d2zdx2);
        let twelveth_d2zdxdy2 = 3.0f64.mul_add(dble_dzdx01 - dble_dzdx00, -half_d2zdy2);
        let dble_d4zdx2dy2 = 2.0f64.mul_add(
            -(quad_d2zdxdy10 + quad_d2zdxdy01),
            4.0 * (quad_d2zdxdy11 + quad_d2zdxdy00),
        ) + twelveth_d2zdxdxdy
            + twelveth_d2zdxdy2;

        let c0 = dos_two;
        let c1 = dble_dzdy00;
        let c2 = half_d2zdy2;
        let c3 = tre_two - (c0 + c1 + c2);

        let c4 = dble_dzdx00;
        let c5 = quad_d2zdxdy00;
        let c6 = twelveth_d2zdxdy2;
        let c7 = dble_dzdx01 - (c4 + c5 + c6);

        let c8 = half_d2zdx2;
        let c9 = twelveth_d2zdxdxdy;
        let c10 = dble_d4zdx2dy2;
        let c11 = half_d2zdx2 + dble_dzdx11 - dble_dzdx10 - (c8 + c9 + c10);

        let c12 = dos_thr - (c0 + c4 + c8);
        let c13 = dble_dzdy10 - (c1 + c5 + c9);
        let c14 = half_d2zdy2 + dble_dzdy11 - dble_dzdy10 - (c2 + c6 + c10);
        let c15 = tre_thr
            - (c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10 + c11 + c12 + c13 + c14);

        relative_x.mul_add(
            relative_x.mul_add(relative_x.mul_add(c12, c8), c4),
            relative_y.mul_add(
                relative_x.mul_add(
                    relative_x.mul_add(
                        relative_x.mul_add(
                            relative_y.mul_add(relative_y.mul_add(c15, c14), c13),
                            relative_y.mul_add(relative_y.mul_add(c11, c10), c9),
                        ),
                        relative_y.mul_add(relative_y.mul_add(c7, c6), c5),
                    ),
                    relative_y.mul_add(relative_y.mul_add(c3, c2), c1),
                ),
                c0,
            ),
        )
    }

    #[inline]
    fn sample_source_band_premultiplied(
        &self,
        source_region: Region,
        source: &[F::Sample],
        ix: i64,
        iy: i64,
        band: usize,
        alpha_band: usize,
        alpha_max: f64,
    ) -> f64
    where
        F::Sample: Copy + ToF64,
    {
        let value = self.sample_source_band(source_region, source, ix, iy, band);
        if band == alpha_band {
            value
        } else {
            value * (self.sample_source_band(source_region, source, ix, iy, alpha_band) / alpha_max)
        }
    }

    #[inline]
    fn sample_at_premultiplied(
        &self,
        source_region: Region,
        source: &[F::Sample],
        x_in: f64,
        y_in: f64,
        band: usize,
        alpha_band: usize,
        alpha_max: f64,
    ) -> f64
    where
        F::Sample: Copy + ToF64,
    {
        if !x_in.is_finite() || !y_in.is_finite() {
            return self.background;
        }

        match self.kernel {
            InterpolationKernel::Nearest => self.sample_source_band_premultiplied(
                source_region,
                source,
                x_in.floor() as i64,
                y_in.floor() as i64,
                band,
                alpha_band,
                alpha_max,
            ),
            InterpolationKernel::Bilinear => {
                let x0 = x_in.floor() as i64;
                let y0 = y_in.floor() as i64;
                let fx = x_in - x0 as f64;
                let fy = y_in - y0 as f64;

                Self::bilinear_mix(
                    self.sample_source_band_premultiplied(
                        source_region,
                        source,
                        x0,
                        y0,
                        band,
                        alpha_band,
                        alpha_max,
                    ),
                    self.sample_source_band_premultiplied(
                        source_region,
                        source,
                        x0 + 1,
                        y0,
                        band,
                        alpha_band,
                        alpha_max,
                    ),
                    self.sample_source_band_premultiplied(
                        source_region,
                        source,
                        x0,
                        y0 + 1,
                        band,
                        alpha_band,
                        alpha_max,
                    ),
                    self.sample_source_band_premultiplied(
                        source_region,
                        source,
                        x0 + 1,
                        y0 + 1,
                        band,
                        alpha_band,
                        alpha_max,
                    ),
                    fx,
                    fy,
                )
            }
            InterpolationKernel::Vsqbs => {
                let x_floor = x_in.floor() as i64;
                let y_floor = y_in.floor() as i64;
                let x_phase = x_in - x_floor as f64;
                let y_phase = y_in - y_floor as f64;
                let mut neighborhood = [[0.0_f64; 4]; 4];

                for (row, samples) in neighborhood.iter_mut().enumerate() {
                    for (col, sample) in samples.iter_mut().enumerate() {
                        *sample = self.sample_source_band_premultiplied(
                            source_region,
                            source,
                            x_floor + col as i64 - 1,
                            y_floor + row as i64 - 1,
                            band,
                            alpha_band,
                            alpha_max,
                        );
                    }
                }

                self.kernel.interpolate_2d(x_phase, y_phase, &neighborhood)
            }
            InterpolationKernel::Nohalo => {
                let x_floor = x_in.floor() as i64;
                let y_floor = y_in.floor() as i64;
                let x_phase = x_in - x_floor as f64;
                let y_phase = y_in - y_floor as f64;
                let mut neighborhood = [[0.0_f64; 6]; 6];

                for (row, samples) in neighborhood.iter_mut().enumerate() {
                    for (col, sample) in samples.iter_mut().enumerate() {
                        *sample = self.sample_source_band_premultiplied(
                            source_region,
                            source,
                            x_floor + col as i64 - 2,
                            y_floor + row as i64 - 2,
                            band,
                            alpha_band,
                            alpha_max,
                        );
                    }
                }

                self.kernel.interpolate_2d(x_phase, y_phase, &neighborhood)
            }
            InterpolationKernel::Bicubic
            | InterpolationKernel::Quadratic
            | InterpolationKernel::CatmullRom
            | InterpolationKernel::Lanczos2 => self.sample_at_premultiplied_separable::<2>(
                source_region,
                source,
                x_in,
                y_in,
                band,
                alpha_band,
                alpha_max,
            ),
            InterpolationKernel::Lbb => {
                let ix = x_in.floor() as i64;
                let iy = y_in.floor() as i64;
                let stencil = [
                    self.sample_source_band_premultiplied(
                        source_region,
                        source,
                        ix - 1,
                        iy - 1,
                        band,
                        alpha_band,
                        alpha_max,
                    ),
                    self.sample_source_band_premultiplied(
                        source_region,
                        source,
                        ix,
                        iy - 1,
                        band,
                        alpha_band,
                        alpha_max,
                    ),
                    self.sample_source_band_premultiplied(
                        source_region,
                        source,
                        ix + 1,
                        iy - 1,
                        band,
                        alpha_band,
                        alpha_max,
                    ),
                    self.sample_source_band_premultiplied(
                        source_region,
                        source,
                        ix + 2,
                        iy - 1,
                        band,
                        alpha_band,
                        alpha_max,
                    ),
                    self.sample_source_band_premultiplied(
                        source_region,
                        source,
                        ix - 1,
                        iy,
                        band,
                        alpha_band,
                        alpha_max,
                    ),
                    self.sample_source_band_premultiplied(
                        source_region,
                        source,
                        ix,
                        iy,
                        band,
                        alpha_band,
                        alpha_max,
                    ),
                    self.sample_source_band_premultiplied(
                        source_region,
                        source,
                        ix + 1,
                        iy,
                        band,
                        alpha_band,
                        alpha_max,
                    ),
                    self.sample_source_band_premultiplied(
                        source_region,
                        source,
                        ix + 2,
                        iy,
                        band,
                        alpha_band,
                        alpha_max,
                    ),
                    self.sample_source_band_premultiplied(
                        source_region,
                        source,
                        ix - 1,
                        iy + 1,
                        band,
                        alpha_band,
                        alpha_max,
                    ),
                    self.sample_source_band_premultiplied(
                        source_region,
                        source,
                        ix,
                        iy + 1,
                        band,
                        alpha_band,
                        alpha_max,
                    ),
                    self.sample_source_band_premultiplied(
                        source_region,
                        source,
                        ix + 1,
                        iy + 1,
                        band,
                        alpha_band,
                        alpha_max,
                    ),
                    self.sample_source_band_premultiplied(
                        source_region,
                        source,
                        ix + 2,
                        iy + 1,
                        band,
                        alpha_band,
                        alpha_max,
                    ),
                    self.sample_source_band_premultiplied(
                        source_region,
                        source,
                        ix - 1,
                        iy + 2,
                        band,
                        alpha_band,
                        alpha_max,
                    ),
                    self.sample_source_band_premultiplied(
                        source_region,
                        source,
                        ix,
                        iy + 2,
                        band,
                        alpha_band,
                        alpha_max,
                    ),
                    self.sample_source_band_premultiplied(
                        source_region,
                        source,
                        ix + 1,
                        iy + 2,
                        band,
                        alpha_band,
                        alpha_max,
                    ),
                    self.sample_source_band_premultiplied(
                        source_region,
                        source,
                        ix + 2,
                        iy + 2,
                        band,
                        alpha_band,
                        alpha_max,
                    ),
                ];

                Self::lbbicubic(&stencil, x_in - ix as f64, y_in - iy as f64)
            }
            InterpolationKernel::Lanczos3 => self.sample_at_premultiplied_separable::<3>(
                source_region,
                source,
                x_in,
                y_in,
                band,
                alpha_band,
                alpha_max,
            ),
        }
    }

    #[inline]
    fn sample_at_premultiplied_separable<const SUPPORT: i64>(
        &self,
        source_region: Region,
        source: &[F::Sample],
        x_in: f64,
        y_in: f64,
        band: usize,
        alpha_band: usize,
        alpha_max: f64,
    ) -> f64
    where
        F::Sample: Copy + ToF64,
    {
        let cx = x_in.floor() as i64;
        let cy = y_in.floor() as i64;
        let mut acc = 0.0_f64;
        let mut weight_sum = 0.0_f64;
        let lo = -(SUPPORT - 1);
        let hi = SUPPORT;

        let mut ky = lo;
        while ky <= hi {
            let iy = cy + ky;
            let wy = self.kernel.interpolate((iy as f64 - y_in).abs());
            if wy != 0.0 {
                let mut kx = lo;
                while kx <= hi {
                    let ix = cx + kx;
                    let wx = self.kernel.interpolate((ix as f64 - x_in).abs());
                    if wx != 0.0 {
                        let w = wx * wy;
                        weight_sum += w;
                        acc = self
                            .sample_source_band_premultiplied(
                                source_region,
                                source,
                                ix,
                                iy,
                                band,
                                alpha_band,
                                alpha_max,
                            )
                            .mul_add(w, acc);
                    }
                    kx += 1;
                }
            }
            ky += 1;
        }

        if weight_sum > 0.0 {
            acc / weight_sum
        } else {
            self.background
        }
    }

    #[inline]
    fn sample_source_with_premultiplied_alpha_into(
        &self,
        source_region: Region,
        source: &[F::Sample],
        x_in: f64,
        y_in: f64,
        output: &mut [F::Sample],
    ) where
        F::Sample: Pod + ToF64 + FromF64 + Copy,
    {
        if !x_in.is_finite() || !y_in.is_finite() {
            output.fill(F::Sample::from_f64(self.background));
            return;
        }

        let alpha_band = output.len() - 1;
        let alpha_max = Self::alpha_max();
        let alpha = self
            .sample_at_premultiplied(
                source_region,
                source,
                x_in,
                y_in,
                alpha_band,
                alpha_band,
                alpha_max,
            )
            .clamp(0.0, alpha_max);
        let normalized_alpha = alpha / alpha_max;

        for (band, sample) in output[..alpha_band].iter_mut().enumerate() {
            let premultiplied = self.sample_at_premultiplied(
                source_region,
                source,
                x_in,
                y_in,
                band,
                alpha_band,
                alpha_max,
            );
            *sample = F::Sample::from_f64(if normalized_alpha > 0.0 {
                premultiplied / normalized_alpha
            } else {
                0.0
            });
        }
        output[alpha_band] = F::Sample::from_f64(alpha);
    }

    #[inline]
    fn sample_source_bilinear_into(
        &self,
        source_region: Region,
        source: &[F::Sample],
        x_in: f64,
        y_in: f64,
        output: &mut [F::Sample],
    ) where
        F::Sample: Pod + ToF64 + FromF64 + Copy,
    {
        let x0 = x_in.floor() as i64;
        let y0 = y_in.floor() as i64;
        let fx = x_in - x0 as f64;
        let fy = y_in - y0 as f64;
        let bands = self.source_bands as usize;
        let source_width = source_region.width as usize;
        let row_stride = source_width * bands;
        let source_right = i64::from(source_region.x) + i64::from(source_region.width);
        let source_bottom = i64::from(source_region.y) + i64::from(source_region.height);

        if x0 >= i64::from(source_region.x)
            && x0 + 1 < source_right
            && y0 >= i64::from(source_region.y)
            && y0 + 1 < source_bottom
        {
            let local_x = (x0 - i64::from(source_region.x)) as usize;
            let local_y = (y0 - i64::from(source_region.y)) as usize;
            let base00 = local_y * row_stride + local_x * bands;
            let base10 = base00 + row_stride;
            let top_row = &source[base00..base00 + (bands * 2)];
            let bottom_row = &source[base10..base10 + (bands * 2)];

            for band in 0..bands {
                let p00 = top_row[band].to_f64();
                let p01 = top_row[band + bands].to_f64();
                let p10 = bottom_row[band].to_f64();
                let p11 = bottom_row[band + bands].to_f64();
                let top = p01.mul_add(fx, p00 * (1.0 - fx));
                let bottom = p11.mul_add(fx, p10 * (1.0 - fx));
                output[band] = F::Sample::from_f64(top * (1.0 - fy) + bottom * fy);
            }
            return;
        }

        for (band, sample) in output.iter_mut().enumerate() {
            let p00 = self.sample_source_band(source_region, source, x0, y0, band);
            let p01 = self.sample_source_band(source_region, source, x0 + 1, y0, band);
            let p10 = self.sample_source_band(source_region, source, x0, y0 + 1, band);
            let p11 = self.sample_source_band(source_region, source, x0 + 1, y0 + 1, band);
            let top = p00.mul_add(1.0 - fx, p01 * fx);
            let bottom = p10 * (1.0 - fx) + p11 * fx;
            *sample = F::Sample::from_f64(top * (1.0 - fy) + bottom * fy);
        }
    }

    #[inline]
    fn sample_source_nearest_into(
        &self,
        source_region: Region,
        source: &[F::Sample],
        x_in: f64,
        y_in: f64,
        output: &mut [F::Sample],
    ) where
        F::Sample: Pod + ToF64 + FromF64 + Copy,
    {
        let ix = x_in.floor() as i64;
        let iy = y_in.floor() as i64;
        let source_right = i64::from(source_region.x) + i64::from(source_region.width);
        let source_bottom = i64::from(source_region.y) + i64::from(source_region.height);

        if ix >= i64::from(source_region.x)
            && ix < source_right
            && iy >= i64::from(source_region.y)
            && iy < source_bottom
        {
            let bands = self.source_bands as usize;
            let source_width = source_region.width as usize;
            let local_x = (ix - i64::from(source_region.x)) as usize;
            let local_y = (iy - i64::from(source_region.y)) as usize;
            let base = (local_y * source_width + local_x) * bands;
            output.copy_from_slice(&source[base..base + bands]);
            return;
        }

        for (band, sample) in output.iter_mut().enumerate() {
            *sample =
                F::Sample::from_f64(self.sample_source_band(source_region, source, ix, iy, band));
        }
    }

    #[inline]
    fn process_typed<I>(
        &self,
        _state: &mut MapImState,
        source_region: Region,
        source: &[F::Sample],
        index_region: Region,
        index: &[I],
        output: &mut [F::Sample],
        output_region: Region,
    ) where
        I: Pod + Copy + ToF64,
        F::Sample: Pod + ToF64 + FromF64 + Copy,
    {
        let source_bands = self.source_bands as usize;
        let index_bands = 2usize;
        let out_bands = self.source_bands as usize;
        let source_pixels = source_region.pixel_count();
        let index_pixels = index_region.pixel_count();
        let output_pixels = output_region.pixel_count();
        let sampler = Affine::<F>::new(
            [1.0, 0.0, 0.0, 1.0],
            0.0,
            0.0,
            self.kernel,
            self.index_width,
            self.index_height,
        )
        .with_background(self.background);

        debug_assert_eq!(
            source.len(),
            source_pixels * source_bands,
            "MapIm source buffer size mismatch"
        );
        debug_assert_eq!(
            index.len(),
            index_pixels * index_bands,
            "MapIm index buffer size mismatch"
        );
        debug_assert_eq!(
            output.len(),
            output_pixels * out_bands,
            "MapIm output buffer size mismatch"
        );
        debug_assert_eq!(
            index_region, output_region,
            "MapIm index region must match the output region"
        );

        let do_premultiply = !self.premultiplied && self.has_alpha();
        if do_premultiply {
            for row in 0..output_region.height as usize {
                for col in 0..output_region.width as usize {
                    let px = (row * output_region.width as usize + col) * out_bands;
                    let ix = (row * index_region.width as usize + col) * index_bands;

                    self.sample_source_with_premultiplied_alpha_into(
                        source_region,
                        source,
                        index[ix].to_f64(),
                        index[ix + 1].to_f64(),
                        &mut output[px..px + out_bands],
                    );
                }
            }
        } else {
            match self.kernel {
                InterpolationKernel::Nearest => {
                    for row in 0..output_region.height as usize {
                        for col in 0..output_region.width as usize {
                            let px = (row * output_region.width as usize + col) * out_bands;
                            let ix = (row * index_region.width as usize + col) * index_bands;

                            let raw_x = index[ix].to_f64();
                            let raw_y = index[ix + 1].to_f64();
                            if !raw_x.is_finite() || !raw_y.is_finite() {
                                output[px..px + out_bands]
                                    .fill(F::Sample::from_f64(self.background));
                                continue;
                            }

                            self.sample_source_nearest_into(
                                source_region,
                                source,
                                raw_x,
                                raw_y,
                                &mut output[px..px + out_bands],
                            );
                        }
                    }
                }
                InterpolationKernel::Bilinear => {
                    for row in 0..output_region.height as usize {
                        for col in 0..output_region.width as usize {
                            let px = (row * output_region.width as usize + col) * out_bands;
                            let ix = (row * index_region.width as usize + col) * index_bands;

                            let raw_x = index[ix].to_f64();
                            let raw_y = index[ix + 1].to_f64();
                            if !raw_x.is_finite() || !raw_y.is_finite() {
                                output[px..px + out_bands]
                                    .fill(F::Sample::from_f64(self.background));
                                continue;
                            }

                            self.sample_source_bilinear_into(
                                source_region,
                                source,
                                raw_x,
                                raw_y,
                                &mut output[px..px + out_bands],
                            );
                        }
                    }
                }
                _ => {
                    let source_tile = Tile::<F>::new(source_region, self.source_bands, source);
                    for row in 0..output_region.height as usize {
                        for col in 0..output_region.width as usize {
                            let px = (row * output_region.width as usize + col) * out_bands;
                            let ix = (row * index_region.width as usize + col) * index_bands;

                            let raw_x = index[ix].to_f64();
                            let raw_y = index[ix + 1].to_f64();
                            if !raw_x.is_finite() || !raw_y.is_finite() {
                                output[px..px + out_bands]
                                    .fill(F::Sample::from_f64(self.background));
                                continue;
                            }

                            for band in 0..out_bands {
                                let value = sampler.sample_at(&source_tile, raw_x, raw_y, band);
                                output[px + band] = F::Sample::from_f64(value);
                            }
                        }
                    }
                }
            }
        }
    }
}

impl<F: BandFormat> DynOperation for MapImOp<F>
where
    F::Sample: Pod + ToF64 + FromF64 + Send,
{
    fn input_format(&self) -> BandFormatId {
        F::ID
    }

    fn output_format(&self) -> BandFormatId {
        F::ID
    }

    fn bands(&self) -> u32 {
        self.source_bands
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::SmallTile
    }

    fn input_slot_count(&self) -> usize {
        2
    }

    fn input_format_slot(&self, slot: usize) -> BandFormatId {
        match slot {
            0 => F::ID,
            1 => self.index_format,
            _ => {
                debug_assert!(false, "MapImOp input slot out of range");
                F::ID
            }
        }
    }

    fn input_bands_slot(&self, slot: usize) -> u32 {
        match slot {
            0 => self.source_bands,
            1 => 2,
            _ => {
                debug_assert!(false, "MapImOp input slot out of range");
                self.source_bands
            }
        }
    }

    fn required_input_region_slot(&self, output: &Region, slot: usize) -> Region {
        match slot {
            0 => self.source_region_for_output(output),
            1 => *output,
            _ => {
                debug_assert!(false, "MapImOp input slot out of range");
                *output
            }
        }
    }

    fn source_read_plan_slot(&self, output: &Region, slot: usize) -> SourceReadPlan {
        match slot {
            0 => SourceReadPlan::rect(self.source_region_for_output(output)),
            1 => SourceReadPlan::rect(*output),
            _ => {
                debug_assert!(false, "MapImOp input slot out of range");
                SourceReadPlan::rect(*output)
            }
        }
    }

    fn required_input_region(&self, _output: &Region) -> Region {
        self.source_region()
    }

    fn coordinate_driven_source_spec(
        &self,
    ) -> Option<crate::domain::op::CoordinateDrivenSourceSpec> {
        Some(crate::domain::op::CoordinateDrivenSourceSpec {
            source_slot: 0,
            dependency_slot: 1,
        })
    }

    fn source_read_plan_slot_with_materialized_dependency(
        &self,
        _output: &Region,
        slot: usize,
        dependency_slot: usize,
        _dependency_region: Region,
        dependency: &[u8],
    ) -> Option<SourceReadPlan> {
        if slot != 0 {
            return None;
        }

        self.bounded_source_region_from_dependency_bytes(dependency_slot, dependency)
            .map(SourceReadPlan::rect)
    }

    fn output_width(&self, _input_w: u32) -> u32 {
        self.index_width
    }

    fn output_height(&self, _input_h: u32) -> u32 {
        self.index_height
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        let (left_pad, right_pad) = self.kernel.affine_padding();
        let halo = (left_pad + right_pad) as u32;
        NodeSpec {
            input_tile_w: tile_w.saturating_add(halo),
            input_tile_h: tile_h.saturating_add(halo),
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
        .with_coordinate_driven_source(0, 1)
    }

    fn validate_build_contract(
        &self,
        input_bands: u32,
        output_bands: u32,
    ) -> Result<(), BuildError> {
        let _ = (input_bands, output_bands);
        Ok(())
    }

    fn dyn_start(&self) -> Box<dyn Any + Send> {
        Box::new(MapImState)
    }

    fn dyn_start_with_tile_and_bands(
        &self,
        _tile_w: u32,
        _tile_h: u32,
        _bands: u32,
    ) -> Box<dyn Any + Send> {
        Box::new(MapImState)
    }

    #[inline]
    fn dyn_process_region(
        &self,
        _state: &mut dyn Any,
        _input: &[u8],
        _output: &mut [u8],
        _input_region: Region,
        _output_region: Region,
    ) {
        debug_assert!(false, "MapImOp is multi-input only");
    }

    #[inline]
    fn dyn_process_region_multi(
        &self,
        state: &mut dyn Any,
        inputs: &[&[u8]],
        output: &mut [u8],
        input_regions: &[Region],
        output_region: Region,
    ) {
        debug_assert_eq!(inputs.len(), 2, "MapImOp expects 2 input slices");
        debug_assert_eq!(input_regions.len(), 2, "MapImOp expects 2 input regions");

        let Some(state) = state.downcast_mut::<MapImState>() else {
            debug_assert!(false, "MapImOp state type mismatch");
            return;
        };
        let (Some(&source_bytes), Some(&index_bytes)) = (inputs.first(), inputs.get(1)) else {
            debug_assert!(false, "MapImOp missing input slices");
            return;
        };
        let (Some(&source_region), Some(&index_region)) =
            (input_regions.first(), input_regions.get(1))
        else {
            debug_assert!(false, "MapImOp missing input regions");
            return;
        };

        let source_samples = cast_slice::<u8, F::Sample>(source_bytes);
        let output_samples = cast_slice_mut::<u8, F::Sample>(output);

        match self.index_format {
            BandFormatId::U8 => self.process_typed::<u8>(
                state,
                source_region,
                source_samples,
                index_region,
                cast_slice::<u8, u8>(index_bytes),
                output_samples,
                output_region,
            ),
            BandFormatId::U16 => self.process_typed::<u16>(
                state,
                source_region,
                source_samples,
                index_region,
                cast_slice::<u8, u16>(index_bytes),
                output_samples,
                output_region,
            ),
            BandFormatId::I16 => self.process_typed::<i16>(
                state,
                source_region,
                source_samples,
                index_region,
                cast_slice::<u8, i16>(index_bytes),
                output_samples,
                output_region,
            ),
            BandFormatId::U32 => self.process_typed::<u32>(
                state,
                source_region,
                source_samples,
                index_region,
                cast_slice::<u8, u32>(index_bytes),
                output_samples,
                output_region,
            ),
            BandFormatId::I32 => self.process_typed::<i32>(
                state,
                source_region,
                source_samples,
                index_region,
                cast_slice::<u8, i32>(index_bytes),
                output_samples,
                output_region,
            ),
            BandFormatId::F32 => self.process_typed::<f32>(
                state,
                source_region,
                source_samples,
                index_region,
                cast_slice::<u8, f32>(index_bytes),
                output_samples,
                output_region,
            ),
            BandFormatId::F64 => self.process_typed::<f64>(
                state,
                source_region,
                source_samples,
                index_region,
                cast_slice::<u8, f64>(index_bytes),
                output_samples,
                output_region,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        adapters::{
            pipeline::PipelineArena, scheduler::rayon_scheduler::RayonScheduler,
            sinks::memory::MemorySink, sources::memory::MemorySource,
        },
        domain::{
            format::{F32, U8},
            op::OperationBridge,
            ops::conversion::CopyOp,
        },
        ports::scheduler::TileScheduler,
    };
    use proptest::prelude::*;

    fn run_mapim(
        op: &MapImOp<U8>,
        source: &[u8],
        index: &[f32],
        output: &mut [u8],
        source_region: Region,
        index_region: Region,
        output_region: Region,
    ) {
        let inputs: &[&[u8]] = &[
            bytemuck::cast_slice::<u8, u8>(source),
            bytemuck::cast_slice::<f32, u8>(index),
        ];
        let input_regions = [source_region, index_region];
        let mut state = op.dyn_start();
        op.dyn_process_region_multi(
            state.as_mut(),
            inputs,
            bytemuck::cast_slice_mut::<u8, u8>(output),
            &input_regions,
            output_region,
        );
    }

    #[test]
    fn slot_contract_matches_separate_roots() {
        let op = MapImOp::<U8>::new(4, 4, 1, 2, 2, BandFormatId::F32);
        assert_eq!(op.input_slot_count(), 2);
        assert_eq!(op.input_format_slot(0), BandFormatId::U8);
        assert_eq!(op.input_format_slot(1), BandFormatId::F32);
        assert_eq!(op.input_bands_slot(0), 1);
        assert_eq!(op.input_bands_slot(1), 2);
        assert_eq!(op.output_format(), BandFormatId::U8);
        assert_eq!(op.output_width(1), 2);
        assert_eq!(op.output_height(1), 2);
    }

    #[test]
    fn node_spec_scales_source_scratch_with_tile_halo() {
        let op = MapImOp::<U8>::new(4096, 4096, 1, 4096, 4096, BandFormatId::F32)
            .with_kernel(InterpolationKernel::Bilinear);

        let spec = op.node_spec(128, 128);

        assert_eq!(spec.input_tile_w, 129);
        assert_eq!(spec.input_tile_h, 129);
        assert_eq!(spec.output_tile_w, 128);
        assert_eq!(spec.output_tile_h, 128);
        assert_eq!(
            spec.coordinate_driven_source,
            Some(crate::domain::op::CoordinateDrivenSourceSpec {
                source_slot: 0,
                dependency_slot: 1,
            })
        );
    }

    #[test]
    #[should_panic(expected = "MapImOp input slot out of range")]
    fn out_of_range_slots_panic_in_debug_builds() {
        let op = MapImOp::<U8>::new(4, 4, 3, 2, 2, BandFormatId::F32);
        let output = Region::new(1, 2, 3, 4);

        let _ = output;
        let _ = op.input_format_slot(9);
    }

    #[test]
    fn dyn_start_sizes_premultiply_scratch_only_when_needed() {
        let rgba = MapImOp::<U8>::new(4096, 4096, 4, 1, 1, BandFormatId::F32);
        let opaque =
            MapImOp::<U8>::new(4096, 4096, 4, 1, 1, BandFormatId::F32).with_premultiplied(true);
        let grey = MapImOp::<U8>::new(4096, 4096, 1, 1, 1, BandFormatId::F32);

        assert!(rgba.dyn_start().downcast_ref::<MapImState>().is_some());
        assert!(opaque.dyn_start().downcast_ref::<MapImState>().is_some());
        assert!(grey.dyn_start().downcast_ref::<MapImState>().is_some());
    }

    #[test]
    fn materialized_dependency_bounds_support_u16_and_wrong_slots() {
        let op = MapImOp::<U8>::new(4, 4, 1, 2, 1, BandFormatId::U16);
        let dependency = bytemuck::cast_slice::<u16, u8>(&[0, 0, 1, 0]);
        let plan = op.source_read_plan_slot_with_materialized_dependency(
            &Region::new(0, 0, 2, 1),
            0,
            1,
            Region::new(0, 0, 2, 1),
            dependency,
        );

        assert_eq!(plan, Some(SourceReadPlan::rect(Region::new(0, 0, 3, 2))));
        assert!(
            op.source_read_plan_slot_with_materialized_dependency(
                &Region::new(0, 0, 2, 1),
                1,
                1,
                Region::new(0, 0, 2, 1),
                dependency,
            )
            .is_none()
        );
        assert!(
            op.source_read_plan_slot_with_materialized_dependency(
                &Region::new(0, 0, 2, 1),
                0,
                0,
                Region::new(0, 0, 2, 1),
                dependency,
            )
            .is_none()
        );
    }

    #[test]
    #[should_panic(expected = "MapImOp is multi-input only")]
    fn dyn_process_region_single_input_path_panics_in_debug_builds() {
        let op = MapImOp::<U8>::new(2, 2, 1, 1, 1, BandFormatId::F32);
        let mut state = op.dyn_start();
        let mut output = vec![7u8; 4];
        op.dyn_process_region(
            state.as_mut(),
            &[1, 2, 3, 4],
            &mut output,
            Region::new(0, 0, 2, 2),
            Region::new(0, 0, 1, 1),
        );
    }

    #[test]
    fn u16_index_identity_map_preserves_pixels() {
        let op = MapImOp::<U8>::new(2, 2, 1, 2, 2, BandFormatId::U16);
        let source = vec![10u8, 20, 30, 40];
        let index = [0u16, 0, 1, 0, 0, 1, 1, 1];
        let inputs: &[&[u8]] = &[&source, bytemuck::cast_slice::<u16, u8>(&index)];
        let regions = [Region::new(0, 0, 2, 2), Region::new(0, 0, 2, 2)];
        let mut output = vec![0u8; 4];
        let mut state = op.dyn_start();

        op.dyn_process_region_multi(
            state.as_mut(),
            inputs,
            &mut output,
            &regions,
            Region::new(0, 0, 2, 2),
        );

        assert_eq!(output, source);
    }

    #[test]
    fn helper_regions_cover_public_contract_and_copy_clamp() {
        let background = MapImOp::<U8>::new(5, 4, 1, 2, 2, BandFormatId::F32);
        let copy =
            MapImOp::<U8>::new(5, 4, 1, 2, 2, BandFormatId::F32).with_extend(MapImExtend::Copy);
        let output = Region::new(1, 1, 2, 2);

        assert_eq!(
            background.required_input_region(&output),
            Region::new(0, 0, 5, 4)
        );
        assert_eq!(
            background.coordinate_driven_source_spec(),
            Some(crate::domain::op::CoordinateDrivenSourceSpec {
                source_slot: 0,
                dependency_slot: 1,
            })
        );
        assert_eq!(
            background.source_read_plan_slot(&output, 0),
            SourceReadPlan::rect(Region::new(1, 1, 3, 3))
        );
        assert_eq!(
            background.source_read_plan_slot(&output, 1),
            SourceReadPlan::rect(output)
        );
        assert_eq!(
            background.bounded_source_region_from_coord_bounds(f64::NAN, 1.0, 0.0, 1.0),
            Region::new(0, 0, 0, 0)
        );
        assert_eq!(
            copy.bounded_source_region_from_coord_bounds(-10.0, 99.0, -10.0, 99.0),
            Region::new(0, 0, 5, 4)
        );
    }

    #[test]
    fn dependency_bytes_cover_remaining_index_formats() {
        let expected = Some(Region::new(0, 0, 3, 3));

        assert_eq!(
            MapImOp::<U8>::new(4, 4, 1, 1, 1, BandFormatId::U8)
                .bounded_source_region_from_dependency_bytes(1, &[0u8, 0, 1, 1])
                .map(SourceReadPlan::rect)
                .map(SourceReadPlan::produced_region),
            expected
        );
        assert_eq!(
            MapImOp::<U8>::new(4, 4, 1, 1, 1, BandFormatId::I16)
                .bounded_source_region_from_dependency_bytes(
                    1,
                    bytemuck::cast_slice::<i16, u8>(&[0, 0, 1, 1])
                )
                .map(SourceReadPlan::rect)
                .map(SourceReadPlan::produced_region),
            expected
        );
        assert_eq!(
            MapImOp::<U8>::new(4, 4, 1, 1, 1, BandFormatId::U32)
                .bounded_source_region_from_dependency_bytes(
                    1,
                    bytemuck::cast_slice::<u32, u8>(&[0, 0, 1, 1])
                )
                .map(SourceReadPlan::rect)
                .map(SourceReadPlan::produced_region),
            expected
        );
        assert_eq!(
            MapImOp::<U8>::new(4, 4, 1, 1, 1, BandFormatId::I32)
                .bounded_source_region_from_dependency_bytes(
                    1,
                    bytemuck::cast_slice::<i32, u8>(&[0, 0, 1, 1])
                )
                .map(SourceReadPlan::rect)
                .map(SourceReadPlan::produced_region),
            expected
        );
        assert_eq!(
            MapImOp::<U8>::new(4, 4, 1, 1, 1, BandFormatId::F64)
                .bounded_source_region_from_dependency_bytes(
                    1,
                    bytemuck::cast_slice::<f64, u8>(&[0.0, 0.0, 1.0, 1.0])
                )
                .map(SourceReadPlan::rect)
                .map(SourceReadPlan::produced_region),
            expected
        );
    }

    #[test]
    fn resolve_source_coord_covers_background_copy_and_empty_copy_source() {
        let background = MapImOp::<U8>::new(3, 2, 1, 1, 1, BandFormatId::F32);
        let copy =
            MapImOp::<U8>::new(3, 2, 1, 1, 1, BandFormatId::F32).with_extend(MapImExtend::Copy);
        let empty =
            MapImOp::<U8>::new(0, 0, 1, 1, 1, BandFormatId::F32).with_extend(MapImExtend::Copy);

        assert_eq!(background.resolve_source_coord(-1, 0), None);
        assert_eq!(copy.resolve_source_coord(-1, -1), Some((0, 0)));
        assert_eq!(copy.resolve_source_coord(99, 99), None);
        assert_eq!(empty.resolve_source_coord(0, 0), None);
    }

    #[test]
    fn non_finite_index_tiles_and_dyn_start_with_tile_cover_remaining_helpers() {
        let op = MapImOp::<U8>::new(4, 4, 4, 2, 2, BandFormatId::F32);
        let empty_region =
            op.bounded_source_region_from_index_tile(&[f32::NAN, 0.0, f32::INFINITY, 1.0]);
        let output = Region::new(0, 0, 0, 0);

        assert_eq!(empty_region, Region::new(0, 0, 0, 0));
        assert_eq!(
            op.required_input_region_slot(&output, 0),
            Region::new(0, 0, 0, 0)
        );

        let from_start = op.dyn_start();
        let from_tiled = op.dyn_start_with_tile_and_bands(64, 64, 4);
        assert!(from_start.downcast_ref::<MapImState>().is_some());
        assert!(from_tiled.downcast_ref::<MapImState>().is_some());
    }

    proptest! {
        #[test]
        fn identity_map_preserves_pixels(pixels in prop::collection::vec(0u8..=255, 16)) {
            let op = MapImOp::<U8>::new(4, 4, 1, 4, 4, BandFormatId::F32);
            let source_region = Region::new(0, 0, 4, 4);
            let index_region = Region::new(0, 0, 4, 4);
            let output_region = index_region;
            let index: Vec<f32> = (0..4)
                .flat_map(|y| (0..4).flat_map(move |x| [x as f32, y as f32]))
                .collect();
            let mut output = vec![0u8; 16];
            run_mapim(&op, &pixels, &index, &mut output, source_region, index_region, output_region);
            prop_assert_eq!(output, pixels);
        }

        #[test]
        fn coordinates_outside_source_fill_background(pixels in prop::collection::vec(0u8..=255, 4)) {
            let op = MapImOp::<U8>::new(2, 2, 1, 2, 1, BandFormatId::F32);
            let source_region = Region::new(0, 0, 2, 2);
            let index_region = Region::new(0, 0, 2, 1);
            let output_region = index_region;
            let index = vec![
                -1.0_f32, -1.0,
                 0.0, 0.0,
            ];
            let mut output = vec![255u8; 2];
            run_mapim(&op, &pixels, &index, &mut output, source_region, index_region, output_region);
            prop_assert_eq!(output, vec![0u8, pixels[0]]);
        }

        #[test]
        fn extend_copy_single_pixel_preserves_value_inside_antialias_edge(
            pixel in 0u8..=255,
            x_tenths in -9i32..10,
            y_tenths in -9i32..10,
        ) {
            let op = MapImOp::<U8>::new(1, 1, 1, 1, 1, BandFormatId::F32)
                .with_extend(MapImExtend::Copy);
            let source_region = Region::new(0, 0, 1, 1);
            let index_region = Region::new(0, 0, 1, 1);
            let output_region = index_region;
            let index = vec![x_tenths as f32 / 10.0, y_tenths as f32 / 10.0];
            let mut output = vec![0u8; 1];

            run_mapim(
                &op,
                &[pixel],
                &index,
                &mut output,
                source_region,
                index_region,
                output_region,
            );

            prop_assert_eq!(output, vec![pixel]);
        }

        #[test]
        fn opaque_rgba_sampling_matches_explicit_premultiplied(
            mut rgba in prop::collection::vec(0u8..=255, 8)
        ) {
            rgba[3] = 255;
            rgba[7] = 255;
            let source_region = Region::new(0, 0, 2, 1);
            let index_region = Region::new(0, 0, 1, 1);
            let output_region = index_region;
            let index = vec![0.5_f32, 0.0];

            let default_op = MapImOp::<U8>::new(2, 1, 4, 1, 1, BandFormatId::F32);
            let premultiplied_op =
                MapImOp::<U8>::new(2, 1, 4, 1, 1, BandFormatId::F32).with_premultiplied(true);
            let mut default_output = vec![0u8; 4];
            let mut premultiplied_output = vec![0u8; 4];

            run_mapim(
                &default_op,
                &rgba,
                &index,
                &mut default_output,
                source_region,
                index_region,
                output_region,
            );
            run_mapim(
                &premultiplied_op,
                &rgba,
                &index,
                &mut premultiplied_output,
                source_region,
                index_region,
                output_region,
            );

            prop_assert_eq!(default_output, premultiplied_output);
        }
    }

    // ── ExtendMode::Copy tests ────────────────────────────────────────────────

    #[test]
    fn extend_modes_match_libvips_near_edge_goldens() {
        // Verified with libvips 8.18.2:
        //   vips mapim src.v out_bg.v idx.v --background 0 --extend background
        //   vips mapim src.v out_copy.v idx.v --background 0 --extend copy
        // where src=[10,20] and idx=[(-0.5,0),(0,0)].
        let source = vec![10u8, 20];
        let source_region = Region::new(0, 0, 2, 1);
        let index_region = Region::new(0, 0, 2, 1);
        let output_region = index_region;
        let index = vec![-0.5_f32, 0.0, 0.0, 0.0];
        let background_op =
            MapImOp::<U8>::new(2, 1, 1, 2, 1, BandFormatId::F32).with_background(0.0);
        let copy_op =
            MapImOp::<U8>::new(2, 1, 1, 2, 1, BandFormatId::F32).with_extend(MapImExtend::Copy);
        let mut background_output = vec![0u8; 2];
        let mut copy_output = vec![0u8; 2];

        run_mapim(
            &background_op,
            &source,
            &index,
            &mut background_output,
            source_region,
            index_region,
            output_region,
        );
        run_mapim(
            &copy_op,
            &source,
            &index,
            &mut copy_output,
            source_region,
            index_region,
            output_region,
        );

        assert_eq!(background_output, vec![5, 10]);
        assert_eq!(copy_output, vec![10, 10]);
    }

    #[test]
    fn extend_copy_far_outside_matches_libvips_background_golden() {
        // Verified with libvips 8.18.2:
        //   vips mapim src.v out_copy.v idx.v --extend copy
        // where src=[50,100] and idx=[(99,0)] produces [0].
        let source = vec![50u8, 100];
        let source_region = Region::new(0, 0, 2, 1);
        let index_region = Region::new(0, 0, 1, 1);
        let output_region = index_region;
        let index = vec![99.0_f32, 0.0];
        let op =
            MapImOp::<U8>::new(2, 1, 1, 1, 1, BandFormatId::F32).with_extend(MapImExtend::Copy);
        let mut output = vec![255u8; 1];

        run_mapim(
            &op,
            &source,
            &index,
            &mut output,
            source_region,
            index_region,
            output_region,
        );

        assert_eq!(
            output,
            vec![0u8],
            "Copy extend only affects the antialias edge; far OOB still falls back to background"
        );
    }

    #[test]
    fn copy_extend_nan_coordinates_still_return_background() {
        let source = vec![77u8];
        let source_region = Region::new(0, 0, 1, 1);
        let index_region = Region::new(0, 0, 1, 1);
        let output_region = index_region;
        let index = vec![f32::NAN, 0.0];
        let op = MapImOp::<U8>::new(1, 1, 1, 1, 1, BandFormatId::F32)
            .with_extend(MapImExtend::Copy)
            .with_background(13.0);
        let mut output = vec![0u8; 1];

        run_mapim(
            &op,
            &source,
            &index,
            &mut output,
            source_region,
            index_region,
            output_region,
        );

        assert_eq!(output, vec![13]);
    }

    #[test]
    fn extend_background_fills_with_custom_value() {
        // 2×1 source: pixels [50, 80]
        let source = vec![50u8, 80];
        let source_region = Region::new(0, 0, 2, 1);
        let index_region = Region::new(0, 0, 2, 1);
        let output_region = index_region;

        // First pixel: OOB coordinate → should fill with background (42).
        // Second pixel: valid coordinate (0, 0) → should return 50.
        let index = vec![-5.0_f32, 0.0, 0.0_f32, 0.0];
        let op = MapImOp::<U8>::new(2, 1, 1, 2, 1, BandFormatId::F32).with_background(42.0);
        let mut output = vec![0u8; 2];
        run_mapim(
            &op,
            &source,
            &index,
            &mut output,
            source_region,
            index_region,
            output_region,
        );
        assert_eq!(
            output[0], 42,
            "Background extend: OOB should return background value"
        );
        assert_eq!(
            output[1], 50,
            "Background extend: in-bounds should return source pixel"
        );
    }

    #[test]
    fn extend_copy_identity_map_preserves_pixels() {
        // 2×2 source, identity index → output == source regardless of extend mode.
        let source = vec![10u8, 20, 30, 40];
        let source_region = Region::new(0, 0, 2, 2);
        let index_region = Region::new(0, 0, 2, 2);
        let output_region = index_region;
        let index = vec![0.0_f32, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let op =
            MapImOp::<U8>::new(2, 2, 1, 2, 2, BandFormatId::F32).with_extend(MapImExtend::Copy);
        let mut output = vec![0u8; 4];
        run_mapim(
            &op,
            &source,
            &index,
            &mut output,
            source_region,
            index_region,
            output_region,
        );
        assert_eq!(
            output, source,
            "Copy extend: identity map must preserve pixels"
        );
    }

    // ── Premultiplied alpha tests ──────────────────────────────────────────────

    /// When `premultiplied = true`, a multi-band image is passed through unchanged
    /// (no premultiply/unpremultiply wrapping). The identity map must preserve all bands.
    #[test]
    fn premultiplied_true_identity_preserves_all_bands() {
        // 2-band image: band 0 = colour, band 1 = alpha
        // With premultiplied=true, bands are passed through as-is.
        let source = vec![100u8, 200, 50, 150];
        let source_region = Region::new(0, 0, 2, 1);
        let index_region = Region::new(0, 0, 2, 1);
        let output_region = index_region;
        // Identity index: pixel 0 → (0,0), pixel 1 → (1,0)
        let index = vec![0.0_f32, 0.0, 1.0, 0.0];

        let op = MapImOp::<U8>::new(2, 1, 2, 2, 1, BandFormatId::F32).with_premultiplied(true);
        let mut output = vec![0u8; 4];

        let inputs: &[&[u8]] = &[
            bytemuck::cast_slice::<u8, u8>(&source),
            bytemuck::cast_slice::<f32, u8>(&index),
        ];
        let input_regions = [source_region, index_region];
        let mut state = op.dyn_start();
        op.dyn_process_region_multi(
            state.as_mut(),
            inputs,
            bytemuck::cast_slice_mut::<u8, u8>(&mut output),
            &input_regions,
            output_region,
        );

        assert_eq!(
            output, source,
            "premultiplied=true identity: all bands must be preserved"
        );
    }

    /// When `premultiplied = false` and alpha = 255 (fully opaque), the colour
    /// channel round-trips through premultiply/unpremultiply unchanged.
    #[test]
    fn premultiplied_false_full_alpha_is_opaque() {
        // 2-band, 1×1 source: colour=128, alpha=255 (fully opaque).
        let source = vec![128u8, 255];
        let source_region = Region::new(0, 0, 1, 1);
        let index_region = Region::new(0, 0, 1, 1);
        let output_region = index_region;
        let index = vec![0.0_f32, 0.0];

        let op = MapImOp::<U8>::new(1, 1, 2, 1, 1, BandFormatId::F32).with_premultiplied(false);
        let mut output = vec![0u8; 2];

        let inputs: &[&[u8]] = &[
            bytemuck::cast_slice::<u8, u8>(&source),
            bytemuck::cast_slice::<f32, u8>(&index),
        ];
        let input_regions = [source_region, index_region];
        let mut state = op.dyn_start();
        op.dyn_process_region_multi(
            state.as_mut(),
            inputs,
            bytemuck::cast_slice_mut::<u8, u8>(&mut output),
            &input_regions,
            output_region,
        );

        // With full alpha (255/255 = 1.0): premul=128*1.0=128; unpremul=128/1.0=128.
        assert_eq!(
            output[0], 128,
            "colour band must survive premultiply round-trip with alpha=255"
        );
        assert_eq!(output[1], 255, "alpha band must be preserved");
    }

    #[test]
    fn premultiplied_false_half_pixel_avoids_colour_bleed() {
        let source = vec![255u8, 0, 0, 0, 0, 0, 255, 255];
        let source_region = Region::new(0, 0, 2, 1);
        let index_region = Region::new(0, 0, 1, 1);
        let output_region = index_region;
        let index = vec![0.5_f32, 0.0];

        let op = MapImOp::<U8>::new(2, 1, 4, 1, 1, BandFormatId::F32);
        let mut output = vec![0u8; 4];

        let inputs: &[&[u8]] = &[
            bytemuck::cast_slice::<u8, u8>(&source),
            bytemuck::cast_slice::<f32, u8>(&index),
        ];
        let input_regions = [source_region, index_region];
        let mut state = op.dyn_start();
        op.dyn_process_region_multi(
            state.as_mut(),
            inputs,
            bytemuck::cast_slice_mut::<u8, u8>(&mut output),
            &input_regions,
            output_region,
        );

        assert_eq!(output, vec![0, 0, 255, 128]);
    }

    #[test]
    fn premultiplied_false_blends_non_constant_alpha_stencil() {
        // Left pixel: semi-transparent red. Right pixel: more transparent green.
        // The bilinear sample at x=0.5 must blend in premultiplied space:
        //   premult(red)   = (128, 0, 0, 128)
        //   premult(green) = (0, 64, 0, 64)
        //   average        = (64, 32, 0, 96)
        //   unpremultiply  = (170, 85, 0, 96)
        let source = vec![255u8, 0, 0, 128, 0, 255, 0, 64];
        let source_region = Region::new(0, 0, 2, 1);
        let index_region = Region::new(0, 0, 1, 1);
        let output_region = index_region;
        let index = vec![0.5_f32, 0.0];

        let op = MapImOp::<U8>::new(2, 1, 4, 1, 1, BandFormatId::F32);
        let mut output = vec![0u8; 4];

        let inputs: &[&[u8]] = &[
            bytemuck::cast_slice::<u8, u8>(&source),
            bytemuck::cast_slice::<f32, u8>(&index),
        ];
        let input_regions = [source_region, index_region];
        let mut state = op.dyn_start();
        op.dyn_process_region_multi(
            state.as_mut(),
            inputs,
            bytemuck::cast_slice_mut::<u8, u8>(&mut output),
            &input_regions,
            output_region,
        );

        assert_eq!(output, vec![170, 85, 0, 96]);
    }

    #[test]
    fn premultiplied_false_preserves_low_alpha_colour_detail() {
        // Interpolation must happen on the premultiplied stencil in float space.
        // If we quantize each premultiplied sample to u8 before interpolation,
        // the left pixel's 127 * (1 / 255) contribution rounds down to 0 and the
        // edge washes to black. libvips keeps the fractional premultiplied value
        // through the kernel and reconstructs the original colour on unpremultiply.
        let source = vec![127u8, 0, 0, 1, 0, 0, 0, 0];
        let source_region = Region::new(0, 0, 2, 1);
        let index_region = Region::new(0, 0, 1, 1);
        let output_region = index_region;
        let index = vec![0.5_f32, 0.0];

        let op = MapImOp::<U8>::new(2, 1, 4, 1, 1, BandFormatId::F32);
        let mut output = vec![0u8; 4];

        let inputs: &[&[u8]] = &[
            bytemuck::cast_slice::<u8, u8>(&source),
            bytemuck::cast_slice::<f32, u8>(&index),
        ];
        let input_regions = [source_region, index_region];
        let mut state = op.dyn_start();
        op.dyn_process_region_multi(
            state.as_mut(),
            inputs,
            bytemuck::cast_slice_mut::<u8, u8>(&mut output),
            &input_regions,
            output_region,
        );

        assert_eq!(output, vec![127, 0, 0, 1]);
    }

    #[test]
    fn premultiplied_false_float_alpha_uses_unit_range() {
        let source = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        let source_region = Region::new(0, 0, 2, 1);
        let index_region = Region::new(0, 0, 1, 1);
        let output_region = index_region;
        let index = vec![0.5_f32, 0.0];

        let op = MapImOp::<F32>::new(2, 1, 4, 1, 1, BandFormatId::F32);
        let mut output = vec![0.0_f32; 4];

        let inputs: &[&[u8]] = &[
            bytemuck::cast_slice::<f32, u8>(&source),
            bytemuck::cast_slice::<f32, u8>(&index),
        ];
        let input_regions = [source_region, index_region];
        let mut state = op.dyn_start();
        op.dyn_process_region_multi(
            state.as_mut(),
            inputs,
            bytemuck::cast_slice_mut::<f32, u8>(&mut output),
            &input_regions,
            output_region,
        );

        assert_eq!(output, vec![0.0, 0.0, 1.0, 0.5]);
    }

    /// When `premultiplied = false` and alpha = 0 (fully transparent), the colour
    /// bands are written as 0 (undefined colour for transparent pixels, matching
    /// libvips behaviour where division by zero yields 0).
    #[test]
    fn premultiplied_false_zero_alpha_yields_zero_colour() {
        // 2-band, 1×1 source: colour=200, alpha=0.
        let source = vec![200u8, 0];
        let source_region = Region::new(0, 0, 1, 1);
        let index_region = Region::new(0, 0, 1, 1);
        let output_region = index_region;
        let index = vec![0.0_f32, 0.0];

        let op = MapImOp::<U8>::new(1, 1, 2, 1, 1, BandFormatId::F32).with_premultiplied(false);
        let mut output = vec![99u8; 2];

        let inputs: &[&[u8]] = &[
            bytemuck::cast_slice::<u8, u8>(&source),
            bytemuck::cast_slice::<f32, u8>(&index),
        ];
        let input_regions = [source_region, index_region];
        let mut state = op.dyn_start();
        op.dyn_process_region_multi(
            state.as_mut(),
            inputs,
            bytemuck::cast_slice_mut::<u8, u8>(&mut output),
            &input_regions,
            output_region,
        );

        // alpha=0 → norm=0.0 → colour set to 0 (guard branch).
        assert_eq!(output[0], 0, "colour band must be 0 when alpha=0");
        assert_eq!(output[1], 0, "alpha band must be 0");
    }

    #[test]
    fn pipeline_connect_to_slot_runs_mapim() {
        let source_pixels = vec![0.0_f32, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let source = MemorySource::<F32>::new(2, 2, 2, source_pixels.clone()).unwrap();

        let mut arena = PipelineArena::with_source(Box::new(source));
        let upstream = arena.add_node(Box::new(OperationBridge::new_pixel_local(
            CopyOp::<F32>::default(),
            2,
        )));
        let node = arena.add_node(Box::new(
            MapImOp::<F32>::new(2, 2, 2, 2, 2, BandFormatId::F32).with_premultiplied(true),
        ));
        arena.connect(upstream, node).unwrap();
        arena.connect_to_slot(upstream, node, 1).unwrap();
        let pipeline = arena.compile().unwrap();

        let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
        RayonScheduler::new(RayonScheduler::default_threads())
            .unwrap()
            .run(&pipeline, &mut sink)
            .unwrap();

        assert_eq!(
            bytemuck::cast_slice::<u8, f32>(&sink.into_buffer()),
            source_pixels.as_slice()
        );
    }

    #[test]
    fn node_spec_sizes_full_image_sources() {
        let op = MapImOp::<U8>::new(8, 4, 1, 2, 2, BandFormatId::F32);
        // node_spec uses halo-aware scratch sizing (tile + kernel halo),
        // so input_tile is output_tile + (kernel - 1) = 32+1=33, 16+1=17.
        assert_eq!(
            op.node_spec(32, 16),
            NodeSpec {
                input_tile_w: 33,
                input_tile_h: 17,
                output_tile_w: 32,
                output_tile_h: 16,
                coordinate_driven_source: Some(crate::domain::op::CoordinateDrivenSourceSpec {
                    source_slot: 0,
                    dependency_slot: 1,
                }),
            }
        );
        assert_eq!(
            op.required_input_region(&Region::new(0, 0, 2, 2)),
            Region::new(0, 0, 8, 4)
        );
    }

    #[test]
    fn demand_hint_is_small_tile() {
        let op = MapImOp::<U8>::new(8, 4, 1, 2, 2, BandFormatId::F32);
        assert_eq!(op.demand_hint(), DemandHint::SmallTile);
    }

    #[test]
    fn source_slot_plan_is_bounded_to_requested_tile() {
        let op = MapImOp::<U8>::new(64, 64, 1, 64, 64, BandFormatId::F32);
        let output = Region::new(20, 10, 8, 6);
        let expected = Region::new(20, 10, 9, 7);

        assert_eq!(op.required_input_region_slot(&output, 0), expected);
        assert_eq!(
            op.source_read_plan_slot(&output, 0),
            SourceReadPlan::rect(expected)
        );
    }

    #[test]
    fn source_slot_plan_clips_negative_output_bounds() {
        let op = MapImOp::<U8>::new(8, 8, 1, 8, 8, BandFormatId::F32);
        let output = Region::new(-3, -2, 4, 3);
        let expected = Region::new(0, 0, 2, 2);

        assert_eq!(op.required_input_region_slot(&output, 0), expected);
        assert_eq!(
            op.source_read_plan_slot(&output, 0),
            SourceReadPlan::rect(expected)
        );
    }

    #[test]
    fn source_slot_plan_clips_out_of_bounds_with_wide_kernel() {
        let op = MapImOp::<U8>::new(8, 8, 1, 8, 8, BandFormatId::F32)
            .with_kernel(InterpolationKernel::Lanczos3);
        let output = Region::new(7, 7, 2, 2);
        let expected = Region::new(5, 5, 3, 3);

        assert_eq!(op.required_input_region_slot(&output, 0), expected);
        assert_eq!(
            op.source_read_plan_slot(&output, 0),
            SourceReadPlan::rect(expected)
        );
    }

    #[test]
    fn runtime_source_plan_uses_realized_coordinate_tile_bounds() {
        let op = MapImOp::<U8>::new(16, 16, 1, 4, 2, BandFormatId::F32);
        let dependency_region = Region::new(0, 0, 4, 2);
        let dependency = bytemuck::cast_slice(&[
            5.2_f32, 4.8, 7.9, 6.1, 6.0, 5.0, 8.1, 6.9, 5.0, 5.5, 7.2, 4.2, 6.8, 6.6, 8.4, 5.1,
        ]);

        assert_eq!(
            op.source_read_plan_slot_with_materialized_dependency(
                &dependency_region,
                0,
                1,
                dependency_region,
                dependency
            ),
            Some(SourceReadPlan::rect(Region::new(5, 4, 5, 4)))
        );
    }

    #[test]
    fn runtime_source_plan_clamps_copy_extend_to_edge_pixels() {
        let op =
            MapImOp::<U8>::new(16, 16, 1, 1, 1, BandFormatId::F32).with_extend(MapImExtend::Copy);
        let dependency_region = Region::new(0, 0, 1, 1);
        let dependency = bytemuck::cast_slice(&[-5.0_f32, -5.0]);

        assert_eq!(
            op.source_read_plan_slot_with_materialized_dependency(
                &dependency_region,
                0,
                1,
                dependency_region,
                dependency
            ),
            Some(SourceReadPlan::rect(Region::new(0, 0, 2, 2)))
        );
    }

    #[test]
    fn half_pixel_coordinates_use_selected_interpolator() {
        let source_region = Region::new(0, 0, 2, 1);
        let index_region = Region::new(0, 0, 1, 1);
        let output_region = index_region;
        let pixels = vec![0u8, 100u8];
        let index = vec![0.5_f32, 0.0_f32];

        let bilinear = MapImOp::<U8>::new(2, 1, 1, 1, 1, BandFormatId::F32);
        let mut bilinear_output = vec![0u8; 1];
        run_mapim(
            &bilinear,
            &pixels,
            &index,
            &mut bilinear_output,
            source_region,
            index_region,
            output_region,
        );
        assert_eq!(bilinear_output, vec![50]);

        let nearest = MapImOp::<U8>::new(2, 1, 1, 1, 1, BandFormatId::F32)
            .with_kernel(InterpolationKernel::Nearest);
        let mut nearest_output = vec![0u8; 1];
        run_mapim(
            &nearest,
            &pixels,
            &index,
            &mut nearest_output,
            source_region,
            index_region,
            output_region,
        );
        assert_eq!(nearest_output, vec![0]);
    }

    #[test]
    fn rgb_inputs_do_not_trigger_alpha_premultiplication() {
        let source_region = Region::new(0, 0, 2, 1);
        let index_region = Region::new(0, 0, 1, 1);
        let output_region = index_region;
        let source = vec![100u8, 0, 10, 0, 100, 250];
        let index = vec![0.5_f32, 0.0_f32];
        let op = MapImOp::<U8>::new(2, 1, 3, 1, 1, BandFormatId::F32);
        let mut output = vec![0u8; 3];

        run_mapim(
            &op,
            &source,
            &index,
            &mut output,
            source_region,
            index_region,
            output_region,
        );

        assert_eq!(output, vec![50, 50, 130]);
    }

    #[test]
    fn dyn_start_does_not_preallocate_full_image_premultiply_scratch() {
        let op = MapImOp::<U8>::new(16_384, 16_384, 4, 64, 64, BandFormatId::F32);

        assert!(op.dyn_start().downcast_ref::<MapImState>().is_some());
        assert!(
            op.dyn_start_with_tile_and_bands(64, 64, 4)
                .downcast_ref::<MapImState>()
                .is_some()
        );
    }

    /// MapImExtend default is Background; premultiplied defaults to false (libvips parity).
    #[test]
    fn builder_defaults_are_background_and_premultiplied_false() {
        let op = MapImOp::<U8>::new(4, 4, 3, 4, 4, BandFormatId::F32);
        assert_eq!(op.extend, MapImExtend::Background);
        assert!(!op.premultiplied);
    }

    #[test]
    fn with_extend_copy_sets_extend_mode() {
        let op =
            MapImOp::<U8>::new(4, 4, 1, 4, 4, BandFormatId::F32).with_extend(MapImExtend::Copy);
        assert_eq!(op.extend, MapImExtend::Copy);
    }

    #[test]
    fn with_premultiplied_true_sets_flag() {
        let op = MapImOp::<U8>::new(4, 4, 3, 4, 4, BandFormatId::F32).with_premultiplied(true);
        assert!(op.premultiplied);
    }

    #[test]
    fn validate_build_contract_accepts_large_sources_without_eager_scratch() {
        let op = MapImOp::<U8>::new(u32::MAX, u32::MAX, 4, 1, 1, BandFormatId::F32);

        assert!(op.validate_build_contract(4, 4).is_ok());
    }

    #[test]
    fn default_premultiplied_matches_explicit_false() {
        // 2-band, 1×1 source: colour=200, alpha=0.
        // With libvips default (premultiplied=false), transparent pixels force colour to 0.
        let source = vec![200u8, 0];
        let source_region = Region::new(0, 0, 1, 1);
        let index_region = Region::new(0, 0, 1, 1);
        let output_region = index_region;
        let index = vec![0.0_f32, 0.0];

        let default_op = MapImOp::<U8>::new(1, 1, 2, 1, 1, BandFormatId::F32);
        let explicit_false_op =
            MapImOp::<U8>::new(1, 1, 2, 1, 1, BandFormatId::F32).with_premultiplied(false);
        let mut default_output = vec![99u8; 2];
        let mut explicit_false_output = vec![99u8; 2];

        run_mapim(
            &default_op,
            &source,
            &index,
            &mut default_output,
            source_region,
            index_region,
            output_region,
        );
        run_mapim(
            &explicit_false_op,
            &source,
            &index,
            &mut explicit_false_output,
            source_region,
            index_region,
            output_region,
        );

        assert_eq!(default_output, explicit_false_output);
        assert_eq!(default_output, vec![0, 0]);
    }

    #[test]
    fn default_premultiplied_rgba_fixture_matches_libvips_golden() {
        // RGBA edge-case fixture:
        // - pixel 0: transparent with non-zero colour
        // - pixel 1: fully opaque
        //
        // Golden bytes were generated with libvips 8.18.2:
        //   vips mapim src_rgba.v out_default.v idx.v
        // on identity coordinates [(0,0), (1,0)].
        let source = vec![200u8, 100, 50, 0, 10, 20, 30, 255];
        let index = vec![0.0_f32, 0.0, 1.0, 0.0];
        let source_region = Region::new(0, 0, 2, 1);
        let index_region = Region::new(0, 0, 2, 1);
        let output_region = index_region;

        let default_op = MapImOp::<U8>::new(2, 1, 4, 2, 1, BandFormatId::F32);
        let explicit_false_op =
            MapImOp::<U8>::new(2, 1, 4, 2, 1, BandFormatId::F32).with_premultiplied(false);
        let explicit_true_op =
            MapImOp::<U8>::new(2, 1, 4, 2, 1, BandFormatId::F32).with_premultiplied(true);

        let mut default_output = vec![99u8; 8];
        let mut explicit_false_output = vec![99u8; 8];
        let mut explicit_true_output = vec![99u8; 8];

        run_mapim(
            &default_op,
            &source,
            &index,
            &mut default_output,
            source_region,
            index_region,
            output_region,
        );
        run_mapim(
            &explicit_false_op,
            &source,
            &index,
            &mut explicit_false_output,
            source_region,
            index_region,
            output_region,
        );
        run_mapim(
            &explicit_true_op,
            &source,
            &index,
            &mut explicit_true_output,
            source_region,
            index_region,
            output_region,
        );

        let libvips_default_golden = vec![0u8, 0, 0, 0, 10, 20, 30, 255];
        assert_eq!(default_output, libvips_default_golden);
        assert_eq!(explicit_false_output, libvips_default_golden);
        assert_eq!(explicit_true_output, source);
    }
}

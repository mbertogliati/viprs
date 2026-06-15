#![allow(dead_code)]
// REASON: affine planning helpers are kept for pending builder-level decomposition work.

use std::marker::PhantomData;

use crate::domain::ops::resample::sample_conv::{FromF64, ToF64};
use crate::domain::{
    format::{BandFormat, BandFormatId},
    image::{DemandHint, Region, Tile, TileMut},
    kernel::InterpolationKernel,
    op::{NodeSpec, Op},
};

use super::{
    Affine, AffineFastPath, AxisCubicSample, AxisLinearSample, AxisNearestSample, ExtendMode,
};

impl<F: BandFormat> Affine<F> {
    /// Construct an `Affine` transform.
    ///
    /// `matrix` is `[a, b, c, d]` (row-major 2×2). `(tx, ty)` is the translation
    /// applied after the matrix multiply, in input-pixel units.
    /// `(output_w, output_h)` are the output image dimensions.
    #[must_use]
    pub fn new(
        matrix: [f64; 4],
        tx: f64,
        ty: f64,
        kernel: InterpolationKernel,
        output_w: u32,
        output_h: u32,
    ) -> Self {
        Self {
            matrix,
            tx,
            ty,
            kernel,
            output_w,
            output_h,
            extend: ExtendMode::Background(vec![0.0]),
            premultiplied: false,
            fast_path: Self::build_fast_path(matrix, tx, ty, kernel, output_w, output_h),
            _format: PhantomData,
        }
    }

    /// Set the fill value for out-of-bounds input samples (default `0.0`).
    #[must_use]
    pub fn with_background(mut self, bg: f64) -> Self {
        self.extend = ExtendMode::Background(vec![bg]);
        self
    }

    /// Set the libvips-style extend mode for out-of-bounds samples.
    #[must_use]
    pub fn with_extend(mut self, extend: ExtendMode) -> Self {
        self.extend = extend;
        self
    }

    /// Declare that the source samples are already premultiplied by alpha.
    #[must_use]
    pub const fn with_premultiplied(mut self, premultiplied: bool) -> Self {
        self.premultiplied = premultiplied;
        self
    }

    /// Map an output coordinate pair to input floating-point coordinates.
    #[inline]
    fn map_to_input(&self, x_out: f64, y_out: f64) -> (f64, f64) {
        let [a, b, c, d] = self.matrix;
        (
            b.mul_add(y_out, a * x_out) + self.tx,
            d.mul_add(y_out, c * x_out) + self.ty,
        )
    }

    #[inline]
    const fn kernel_padding(&self) -> (i32, i32) {
        self.kernel.affine_padding()
    }

    #[inline]
    const fn has_alpha(bands: u32) -> bool {
        matches!(bands, 2 | 4)
    }

    #[inline]
    const fn should_premultiply_alpha(&self, bands: u32) -> bool {
        !self.premultiplied && Self::has_alpha(bands)
    }

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

    fn build_fast_path(
        matrix: [f64; 4],
        tx: f64,
        ty: f64,
        kernel: InterpolationKernel,
        output_w: u32,
        output_h: u32,
    ) -> Option<AffineFastPath> {
        if matrix[1].abs() > f64::EPSILON || matrix[2].abs() > f64::EPSILON {
            return None;
        }

        match kernel {
            InterpolationKernel::Nearest => Some(AffineFastPath::Nearest {
                xs: (0..output_w)
                    .map(|x| AxisNearestSample {
                        coord: matrix[0].mul_add(f64::from(x), tx) as i64,
                    })
                    .collect(),
                ys: (0..output_h)
                    .map(|y| AxisNearestSample {
                        coord: matrix[3].mul_add(f64::from(y), ty) as i64,
                    })
                    .collect(),
            }),
            InterpolationKernel::Bilinear => Some(AffineFastPath::Bilinear {
                xs: (0..output_w)
                    .map(|x| {
                        let coord = matrix[0].mul_add(f64::from(x), tx);
                        let start = coord as i64;
                        let frac = coord - start as f64;
                        AxisLinearSample {
                            start,
                            weights: [1.0 - frac, frac],
                        }
                    })
                    .collect(),
                ys: (0..output_h)
                    .map(|y| {
                        let coord = matrix[3].mul_add(f64::from(y), ty);
                        let start = coord as i64;
                        let frac = coord - start as f64;
                        AxisLinearSample {
                            start,
                            weights: [1.0 - frac, frac],
                        }
                    })
                    .collect(),
            }),
            InterpolationKernel::Bicubic => Some(AffineFastPath::Bicubic {
                xs: Self::build_cubic_axis(matrix[0], tx, output_w),
                ys: Self::build_cubic_axis(matrix[3], ty, output_h),
            }),
            _ => None,
        }
    }

    fn build_cubic_axis(scale: f64, offset: f64, len: u32) -> Box<[AxisCubicSample]> {
        (0..len)
            .map(|index| {
                let coord = scale.mul_add(f64::from(index), offset);
                let center = coord.floor() as i64;
                let start = center - 1;
                let mut weights = [0.0; 4];
                for (tap, weight) in weights.iter_mut().enumerate() {
                    *weight = InterpolationKernel::Bicubic
                        .interpolate(((start + tap as i64) as f64 - coord).abs());
                }
                let sum = weights.iter().sum::<f64>();
                if sum > 0.0 {
                    for weight in &mut weights {
                        *weight /= sum;
                    }
                }
                AxisCubicSample { start, weights }
            })
            .collect()
    }

    #[inline]
    pub(super) fn is_axis_aligned(&self) -> bool {
        self.matrix[1] == 0.0 && self.matrix[2] == 0.0
    }

    #[inline(always)]
    fn bilinear_mix(p00: f64, p01: f64, p10: f64, p11: f64, fx: f64, fy: f64) -> f64 {
        let top = p01.mul_add(fx, p00 * (1.0 - fx));
        let bottom = p11.mul_add(fx, p10 * (1.0 - fx));
        top * (1.0 - fy) + bottom * fy
    }

    const BILINEAR_FIXED_SHIFT: i64 = 15;
    pub(super) const BILINEAR_FIXED_SCALE: i64 = 1_i64 << Self::BILINEAR_FIXED_SHIFT;
    const BILINEAR_FIXED_ROUND: i64 = 1_i64 << ((Self::BILINEAR_FIXED_SHIFT * 2) - 1);

    #[inline(always)]
    pub(super) fn bilinear_fixed_weight(phase: f64) -> i64 {
        (phase * Self::BILINEAR_FIXED_SCALE as f64).round() as i64
    }

    #[inline(always)]
    pub(super) const fn bilinear_u8_coefficients(
        fx_fixed: i64,
        fy_fixed: i64,
    ) -> (i64, i64, i64, i64) {
        let wx0 = Self::BILINEAR_FIXED_SCALE - fx_fixed;
        let wy0 = Self::BILINEAR_FIXED_SCALE - fy_fixed;
        (
            wx0 * wy0,
            fx_fixed * wy0,
            wx0 * fy_fixed,
            fx_fixed * fy_fixed,
        )
    }

    #[inline(always)]
    pub(super) fn bilinear_u8_channel(
        p00: u8,
        p01: u8,
        p10: u8,
        p11: u8,
        c00: i64,
        c01: i64,
        c10: i64,
        c11: i64,
    ) -> u8 {
        let acc = i64::from(p00) * c00
            + i64::from(p01) * c01
            + i64::from(p10) * c10
            + i64::from(p11) * c11;
        ((acc + Self::BILINEAR_FIXED_ROUND) >> (Self::BILINEAR_FIXED_SHIFT * 2))
            .clamp(i64::from(u8::MIN), i64::from(u8::MAX)) as u8
    }

    #[inline(always)]
    fn interpolate_bilinear_into_samples(
        top_row: &[F::Sample],
        bottom_row: &[F::Sample],
        base: usize,
        output: &mut [F::Sample],
        fx: f64,
        fy: f64,
    ) where
        F::Sample: ToF64 + FromF64,
    {
        match output.len() {
            1 => {
                output[0] = F::Sample::from_f64(Self::bilinear_mix(
                    top_row[base].to_f64(),
                    top_row[base + 1].to_f64(),
                    bottom_row[base].to_f64(),
                    bottom_row[base + 1].to_f64(),
                    fx,
                    fy,
                ));
            }
            3 => {
                output[0] = F::Sample::from_f64(Self::bilinear_mix(
                    top_row[base].to_f64(),
                    top_row[base + 3].to_f64(),
                    bottom_row[base].to_f64(),
                    bottom_row[base + 3].to_f64(),
                    fx,
                    fy,
                ));
                output[1] = F::Sample::from_f64(Self::bilinear_mix(
                    top_row[base + 1].to_f64(),
                    top_row[base + 4].to_f64(),
                    bottom_row[base + 1].to_f64(),
                    bottom_row[base + 4].to_f64(),
                    fx,
                    fy,
                ));
                output[2] = F::Sample::from_f64(Self::bilinear_mix(
                    top_row[base + 2].to_f64(),
                    top_row[base + 5].to_f64(),
                    bottom_row[base + 2].to_f64(),
                    bottom_row[base + 5].to_f64(),
                    fx,
                    fy,
                ));
            }
            4 => {
                output[0] = F::Sample::from_f64(Self::bilinear_mix(
                    top_row[base].to_f64(),
                    top_row[base + 4].to_f64(),
                    bottom_row[base].to_f64(),
                    bottom_row[base + 4].to_f64(),
                    fx,
                    fy,
                ));
                output[1] = F::Sample::from_f64(Self::bilinear_mix(
                    top_row[base + 1].to_f64(),
                    top_row[base + 5].to_f64(),
                    bottom_row[base + 1].to_f64(),
                    bottom_row[base + 5].to_f64(),
                    fx,
                    fy,
                ));
                output[2] = F::Sample::from_f64(Self::bilinear_mix(
                    top_row[base + 2].to_f64(),
                    top_row[base + 6].to_f64(),
                    bottom_row[base + 2].to_f64(),
                    bottom_row[base + 6].to_f64(),
                    fx,
                    fy,
                ));
                output[3] = F::Sample::from_f64(Self::bilinear_mix(
                    top_row[base + 3].to_f64(),
                    top_row[base + 7].to_f64(),
                    bottom_row[base + 3].to_f64(),
                    bottom_row[base + 7].to_f64(),
                    fx,
                    fy,
                ));
            }
            bands => {
                for band in 0..bands {
                    output[band] = F::Sample::from_f64(Self::bilinear_mix(
                        top_row[base + band].to_f64(),
                        top_row[base + band + bands].to_f64(),
                        bottom_row[base + band].to_f64(),
                        bottom_row[base + band + bands].to_f64(),
                        fx,
                        fy,
                    ));
                }
            }
        }
    }

    #[inline]
    fn mapped_input_bounds(&self, output: &Region) -> ((f64, f64), (f64, f64)) {
        let ox0 = f64::from(output.x);
        let oy0 = f64::from(output.y);
        let ox1 = f64::from(output.x) + f64::from(output.width.saturating_sub(1));
        let oy1 = f64::from(output.y) + f64::from(output.height.saturating_sub(1));

        let (c0x, c0y) = self.map_to_input(ox0, oy0);
        let (c1x, c1y) = self.map_to_input(ox1, oy0);
        let (c2x, c2y) = self.map_to_input(ox0, oy1);
        let (c3x, c3y) = self.map_to_input(ox1, oy1);

        (
            (
                c0x.min(c1x).min(c2x).min(c3x),
                c0x.max(c1x).max(c2x).max(c3x),
            ),
            (
                c0y.min(c1y).min(c2y).min(c3y),
                c0y.max(c1y).max(c2y).max(c3y),
            ),
        )
    }

    #[inline]
    pub(super) fn output_region_is_background_only(
        &self,
        input: &Tile<F>,
        output: &Region,
    ) -> bool {
        if output.is_empty() {
            return true;
        }
        if !self.uses_constant_fill_extend() {
            return false;
        }

        let ((min_x, max_x), (min_y, max_y)) = self.mapped_input_bounds(output);
        let (left_pad, right_pad) = self.kernel_padding();
        let input_left = i64::from(input.region.x);
        let input_top = i64::from(input.region.y);
        let input_right = input_left + i64::from(input.region.width) - 1;
        let input_bottom = input_top + i64::from(input.region.height) - 1;

        let required_left = min_x.floor() as i64 - i64::from(left_pad);
        let required_right = max_x.floor() as i64 + i64::from(right_pad);
        let required_top = min_y.floor() as i64 - i64::from(left_pad);
        let required_bottom = max_y.floor() as i64 + i64::from(right_pad);

        required_right < input_left
            || required_left > input_right
            || required_bottom < input_top
            || required_top > input_bottom
    }

    /// Sample a single band from the input tile at integer image coordinates.
    ///
    /// Returns the extend-mode fill value when `(ix, iy)` is outside the tile region.
    #[inline]
    fn sample_band(&self, input: &Tile<F>, ix: i64, iy: i64, band: usize) -> f64
    where
        F::Sample: ToF64,
    {
        let Some((resolved_x, resolved_y)) = self.resolve_sample_coords(input, ix, iy) else {
            return self.extend_fill_value(input.bands as usize, band);
        };
        let tile_x = (resolved_x - i64::from(input.region.x)) as usize;
        let tile_y = (resolved_y - i64::from(input.region.y)) as usize;
        let idx = (tile_y * input.region.width as usize + tile_x) * input.bands as usize + band;
        input.data[idx].to_f64()
    }

    #[inline]
    fn sample_band_premultiplied(
        &self,
        input: &Tile<F>,
        ix: i64,
        iy: i64,
        band: usize,
        alpha_band: usize,
        alpha_max: f64,
    ) -> f64
    where
        F::Sample: ToF64,
    {
        let value = self.sample_band(input, ix, iy, band);
        if band == alpha_band {
            value
        } else {
            value * (self.sample_band(input, ix, iy, alpha_band) / alpha_max)
        }
    }

    /// Nearest-neighbour interpolation for a single output pixel and band.
    #[inline]
    fn interp_nearest(&self, input: &Tile<F>, x_in: f64, y_in: f64, band: usize) -> f64
    where
        F::Sample: ToF64,
    {
        self.sample_band(input, x_in as i64, y_in as i64, band)
    }

    /// Bilinear interpolation for a single output pixel and band.
    ///
    /// Uses 2×2 neighbourhood; no call to `kernel.interpolate()` — lerp is exact.
    #[inline]
    fn interp_bilinear(&self, input: &Tile<F>, x_in: f64, y_in: f64, band: usize) -> f64
    where
        F::Sample: ToF64,
    {
        let x0 = x_in as i64;
        let y0 = y_in as i64;
        let fx = x_in - x0 as f64;
        let fy = y_in - y0 as f64;
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        let p00 = self.sample_band(input, x0, y0, band);
        let p01 = self.sample_band(input, x1, y0, band);
        let p10 = self.sample_band(input, x0, y1, band);
        let p11 = self.sample_band(input, x1, y1, band);

        Self::bilinear_mix(p00, p01, p10, p11, fx, fy)
    }

    #[inline]
    pub(crate) fn sample_pixel_nearest_into(
        &self,
        input: &Tile<F>,
        x_in: f64,
        y_in: f64,
        output: &mut [F::Sample],
    ) where
        F::Sample: ToF64 + FromF64,
    {
        if !x_in.is_finite() || !y_in.is_finite() {
            self.fill_extend_samples(output);
            return;
        }

        let ix = x_in as i64;
        let iy = y_in as i64;
        let Some(base) = self.resolved_pixel_base(input, ix, iy) else {
            self.fill_extend_samples(output);
            return;
        };
        let bands = input.bands as usize;
        output.copy_from_slice(&input.data[base..base + bands]);
    }

    #[inline]
    pub(crate) fn sample_pixel_bilinear_into(
        &self,
        input: &Tile<F>,
        x_in: f64,
        y_in: f64,
        output: &mut [F::Sample],
    ) where
        F::Sample: ToF64 + FromF64,
    {
        if !x_in.is_finite() || !y_in.is_finite() {
            self.fill_extend_samples(output);
            return;
        }

        let x0 = x_in as i64;
        let y0 = y_in as i64;
        let fx = x_in - x0 as f64;
        let fy = y_in - y0 as f64;
        let tile_x = x0 - i64::from(input.region.x);
        let tile_y = y0 - i64::from(input.region.y);
        let in_w = i64::from(input.region.width);
        let in_h = i64::from(input.region.height);

        let bands = input.bands as usize;
        let row_stride = input.region.width as usize * bands;

        if tile_x >= 0 && tile_y >= 0 && tile_x + 1 < in_w && tile_y + 1 < in_h {
            let base00 = tile_y as usize * row_stride + tile_x as usize * bands;
            let base10 = base00 + row_stride;
            let top_row = &input.data[base00..base00 + (bands * 2)];
            let bottom_row = &input.data[base10..base10 + (bands * 2)];

            Self::interpolate_bilinear_into_samples(top_row, bottom_row, 0, output, fx, fy);
            return;
        }

        for (band, sample) in output.iter_mut().enumerate() {
            *sample = F::Sample::from_f64(self.interp_bilinear(input, x_in, y_in, band));
        }
    }

    /// libvips VSQBS interpolation over the kernel's 4×4 affine window.
    #[inline]
    fn interp_vsqbs(&self, input: &Tile<F>, x_in: f64, y_in: f64, band: usize) -> f64
    where
        F::Sample: ToF64,
    {
        let x_floor = x_in.floor() as i64;
        let y_floor = y_in.floor() as i64;
        let x_phase = x_in - x_floor as f64;
        let y_phase = y_in - y_floor as f64;
        let mut neighborhood = [[0.0_f64; 4]; 4];

        for (row, samples) in neighborhood.iter_mut().enumerate() {
            for (col, sample) in samples.iter_mut().enumerate() {
                *sample = self.sample_band(
                    input,
                    x_floor + col as i64 - 1,
                    y_floor + row as i64 - 1,
                    band,
                );
            }
        }

        self.kernel.interpolate_2d(x_phase, y_phase, &neighborhood)
    }

    /// libvips nohalo interpolation over the kernel's 6×6 affine window.
    #[inline]
    pub(super) fn interp_nohalo(&self, input: &Tile<F>, x_in: f64, y_in: f64, band: usize) -> f64
    where
        F::Sample: ToF64,
    {
        let x_floor = x_in.floor() as i64;
        let y_floor = y_in.floor() as i64;
        let x_phase = x_in - x_floor as f64;
        let y_phase = y_in - y_floor as f64;
        let mut neighborhood = [[0.0_f64; 6]; 6];

        for (row, samples) in neighborhood.iter_mut().enumerate() {
            for (col, sample) in samples.iter_mut().enumerate() {
                *sample = self.sample_band(
                    input,
                    x_floor + col as i64 - 2,
                    y_floor + row as i64 - 2,
                    band,
                );
            }
        }

        self.kernel.interpolate_2d(x_phase, y_phase, &neighborhood)
    }

    /// Separable 2-D kernel interpolation (`CatmullRom`, Lanczos2, or Lanczos3).
    ///
    /// `SUPPORT` is the kernel half-width: `2` for CatmullRom/Lanczos2, `3` for Lanczos3.
    /// Each axis uses `self.kernel.interpolate(dist)` independently; the 2-D weight is
    /// their product (separability assumption). Weight sum is renormalized to handle
    /// out-of-bounds taps that contributed zero sample value.
    ///
    /// Kernel tap range: `[floor(center) - SUPPORT + 1, floor(center) + SUPPORT]`
    /// — 2*SUPPORT taps per axis, (2*SUPPORT)² total. All stack-allocated.
    #[inline]
    fn interp_separable<const SUPPORT: i64>(
        &self,
        input: &Tile<F>,
        x_in: f64,
        y_in: f64,
        band: usize,
    ) -> f64
    where
        F::Sample: ToF64,
    {
        let cx = x_in.floor() as i64;
        let cy = y_in.floor() as i64;
        let kernel = self.kernel;

        let mut acc = 0.0_f64;
        let mut weight_sum = 0.0_f64;

        // lo = -(SUPPORT-1), hi = SUPPORT  →  2*SUPPORT taps per axis.
        // CatmullRom (SUPPORT=2): -1..=2 (4 taps)
        // Lanczos3   (SUPPORT=3): -2..=3 (6 taps)
        let lo = -(SUPPORT - 1);
        let hi = SUPPORT;

        let mut ky = lo;
        while ky <= hi {
            let iy = cy + ky;
            let wy = kernel.interpolate((iy as f64 - y_in).abs());
            if wy != 0.0 {
                let mut kx = lo;
                while kx <= hi {
                    let ix = cx + kx;
                    let wx = kernel.interpolate((ix as f64 - x_in).abs());
                    if wx != 0.0 {
                        let w = wx * wy;
                        weight_sum += w;
                        acc = self.sample_band(input, ix, iy, band).mul_add(w, acc);
                    }
                    kx += 1;
                }
            }
            ky += 1;
        }

        if weight_sum > 0.0 {
            acc / weight_sum
        } else {
            self.extend_fill_value(input.bands as usize, band)
        }
    }

    #[inline]
    pub(super) fn interp_lbb(&self, input: &Tile<F>, x_in: f64, y_in: f64, band: usize) -> f64
    where
        F::Sample: ToF64,
    {
        let ix = x_in.floor() as i64;
        let iy = y_in.floor() as i64;
        let stencil = [
            self.sample_band(input, ix - 1, iy - 1, band),
            self.sample_band(input, ix, iy - 1, band),
            self.sample_band(input, ix + 1, iy - 1, band),
            self.sample_band(input, ix + 2, iy - 1, band),
            self.sample_band(input, ix - 1, iy, band),
            self.sample_band(input, ix, iy, band),
            self.sample_band(input, ix + 1, iy, band),
            self.sample_band(input, ix + 2, iy, band),
            self.sample_band(input, ix - 1, iy + 1, band),
            self.sample_band(input, ix, iy + 1, band),
            self.sample_band(input, ix + 1, iy + 1, band),
            self.sample_band(input, ix + 2, iy + 1, band),
            self.sample_band(input, ix - 1, iy + 2, band),
            self.sample_band(input, ix, iy + 2, band),
            self.sample_band(input, ix + 1, iy + 2, band),
            self.sample_band(input, ix + 2, iy + 2, band),
        ];

        Self::lbbicubic(&stencil, x_in - ix as f64, y_in - iy as f64)
    }

    #[inline]
    pub(crate) fn sample_at(&self, input: &Tile<F>, x_in: f64, y_in: f64, band: usize) -> f64
    where
        F::Sample: ToF64,
    {
        if !x_in.is_finite() || !y_in.is_finite() {
            return self.extend_fill_value(input.bands as usize, band);
        }

        match self.kernel {
            InterpolationKernel::Nearest => self.interp_nearest(input, x_in, y_in, band),
            InterpolationKernel::Bilinear => self.interp_bilinear(input, x_in, y_in, band),
            InterpolationKernel::Vsqbs => self.interp_vsqbs(input, x_in, y_in, band),
            InterpolationKernel::Nohalo => self.interp_nohalo(input, x_in, y_in, band),
            InterpolationKernel::Bicubic
            | InterpolationKernel::Quadratic
            | InterpolationKernel::CatmullRom
            | InterpolationKernel::Lanczos2 => self.interp_separable::<2>(input, x_in, y_in, band),
            InterpolationKernel::Lbb => self.interp_lbb(input, x_in, y_in, band),
            InterpolationKernel::Lanczos3 => self.interp_separable::<3>(input, x_in, y_in, band),
        }
    }

    #[inline]
    fn sample_at_premultiplied(
        &self,
        input: &Tile<F>,
        x_in: f64,
        y_in: f64,
        band: usize,
        alpha_band: usize,
        alpha_max: f64,
    ) -> f64
    where
        F::Sample: ToF64,
    {
        if !x_in.is_finite() || !y_in.is_finite() {
            return self.extend_fill_value(input.bands as usize, band);
        }

        match self.kernel {
            InterpolationKernel::Nearest => self.sample_band_premultiplied(
                input,
                x_in as i64,
                y_in as i64,
                band,
                alpha_band,
                alpha_max,
            ),
            InterpolationKernel::Bilinear => {
                let x0 = x_in as i64;
                let y0 = y_in as i64;
                let fx = x_in - x0 as f64;
                let fy = y_in - y0 as f64;
                let x1 = x0 + 1;
                let y1 = y0 + 1;

                Self::bilinear_mix(
                    self.sample_band_premultiplied(input, x0, y0, band, alpha_band, alpha_max),
                    self.sample_band_premultiplied(input, x1, y0, band, alpha_band, alpha_max),
                    self.sample_band_premultiplied(input, x0, y1, band, alpha_band, alpha_max),
                    self.sample_band_premultiplied(input, x1, y1, band, alpha_band, alpha_max),
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
                        *sample = self.sample_band_premultiplied(
                            input,
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
                        *sample = self.sample_band_premultiplied(
                            input,
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
                input, x_in, y_in, band, alpha_band, alpha_max,
            ),
            InterpolationKernel::Lbb => {
                let ix = x_in.floor() as i64;
                let iy = y_in.floor() as i64;
                let stencil = [
                    self.sample_band_premultiplied(
                        input,
                        ix - 1,
                        iy - 1,
                        band,
                        alpha_band,
                        alpha_max,
                    ),
                    self.sample_band_premultiplied(input, ix, iy - 1, band, alpha_band, alpha_max),
                    self.sample_band_premultiplied(
                        input,
                        ix + 1,
                        iy - 1,
                        band,
                        alpha_band,
                        alpha_max,
                    ),
                    self.sample_band_premultiplied(
                        input,
                        ix + 2,
                        iy - 1,
                        band,
                        alpha_band,
                        alpha_max,
                    ),
                    self.sample_band_premultiplied(input, ix - 1, iy, band, alpha_band, alpha_max),
                    self.sample_band_premultiplied(input, ix, iy, band, alpha_band, alpha_max),
                    self.sample_band_premultiplied(input, ix + 1, iy, band, alpha_band, alpha_max),
                    self.sample_band_premultiplied(input, ix + 2, iy, band, alpha_band, alpha_max),
                    self.sample_band_premultiplied(
                        input,
                        ix - 1,
                        iy + 1,
                        band,
                        alpha_band,
                        alpha_max,
                    ),
                    self.sample_band_premultiplied(input, ix, iy + 1, band, alpha_band, alpha_max),
                    self.sample_band_premultiplied(
                        input,
                        ix + 1,
                        iy + 1,
                        band,
                        alpha_band,
                        alpha_max,
                    ),
                    self.sample_band_premultiplied(
                        input,
                        ix + 2,
                        iy + 1,
                        band,
                        alpha_band,
                        alpha_max,
                    ),
                    self.sample_band_premultiplied(
                        input,
                        ix - 1,
                        iy + 2,
                        band,
                        alpha_band,
                        alpha_max,
                    ),
                    self.sample_band_premultiplied(input, ix, iy + 2, band, alpha_band, alpha_max),
                    self.sample_band_premultiplied(
                        input,
                        ix + 1,
                        iy + 2,
                        band,
                        alpha_band,
                        alpha_max,
                    ),
                    self.sample_band_premultiplied(
                        input,
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
                input, x_in, y_in, band, alpha_band, alpha_max,
            ),
        }
    }

    #[inline]
    fn sample_at_premultiplied_separable<const SUPPORT: i64>(
        &self,
        input: &Tile<F>,
        x_in: f64,
        y_in: f64,
        band: usize,
        alpha_band: usize,
        alpha_max: f64,
    ) -> f64
    where
        F::Sample: ToF64,
    {
        let cx = x_in.floor() as i64;
        let cy = y_in.floor() as i64;
        let kernel = self.kernel;

        let mut acc = 0.0_f64;
        let mut weight_sum = 0.0_f64;
        let lo = -(SUPPORT - 1);
        let hi = SUPPORT;

        let mut ky = lo;
        while ky <= hi {
            let iy = cy + ky;
            let wy = kernel.interpolate((iy as f64 - y_in).abs());
            if wy != 0.0 {
                let mut kx = lo;
                while kx <= hi {
                    let ix = cx + kx;
                    let wx = kernel.interpolate((ix as f64 - x_in).abs());
                    if wx != 0.0 {
                        let w = wx * wy;
                        weight_sum += w;
                        acc = self
                            .sample_band_premultiplied(input, ix, iy, band, alpha_band, alpha_max)
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
            self.extend_fill_value(input.bands as usize, band)
        }
    }

    #[inline]
    fn sample_pixel_at_with_alpha(
        &self,
        input: &Tile<F>,
        x_in: f64,
        y_in: f64,
        output: &mut [F::Sample],
    ) where
        F::Sample: ToF64 + FromF64,
    {
        if !x_in.is_finite() || !y_in.is_finite() {
            self.fill_extend_samples(output);
            return;
        }

        let alpha_band = output.len() - 1;
        let alpha_max = Self::alpha_max();
        let alpha = self
            .sample_at_premultiplied(input, x_in, y_in, alpha_band, alpha_band, alpha_max)
            .clamp(0.0, alpha_max);
        let normalized_alpha = alpha / alpha_max;

        for (band, sample) in output[..alpha_band].iter_mut().enumerate() {
            let premultiplied =
                self.sample_at_premultiplied(input, x_in, y_in, band, alpha_band, alpha_max);
            *sample = F::Sample::from_f64(if normalized_alpha > 0.0 {
                premultiplied / normalized_alpha
            } else {
                0.0
            });
        }
        output[alpha_band] = F::Sample::from_f64(alpha);
    }

    #[inline]
    pub(crate) fn sample_pixel_at(
        &self,
        input: &Tile<F>,
        x_in: f64,
        y_in: f64,
        output: &mut [F::Sample],
    ) where
        F::Sample: ToF64 + FromF64,
    {
        if self.should_premultiply_alpha(input.bands) {
            self.sample_pixel_at_with_alpha(input, x_in, y_in, output);
            return;
        }

        match self.kernel {
            InterpolationKernel::Nearest => {
                self.sample_pixel_nearest_into(input, x_in, y_in, output);
            }
            InterpolationKernel::Bilinear => {
                self.sample_pixel_bilinear_into(input, x_in, y_in, output);
            }
            _ => {
                if !x_in.is_finite() || !y_in.is_finite() {
                    self.fill_extend_samples(output);
                    return;
                }

                for (band, sample) in output.iter_mut().enumerate() {
                    *sample = F::Sample::from_f64(self.sample_at(input, x_in, y_in, band));
                }
            }
        }
    }

    #[inline]
    pub(super) fn process_region_fast_path(&self, input: &Tile<F>, output: &mut TileMut<F>) -> bool
    where
        F::Sample: ToF64 + FromF64,
    {
        if !self.uses_constant_fill_extend() {
            return false;
        }
        if self.should_premultiply_alpha(input.bands) {
            return false;
        }

        let Ok(output_x) = usize::try_from(output.region.x) else {
            return false;
        };
        let Ok(output_y) = usize::try_from(output.region.y) else {
            return false;
        };
        if output_x + output.region.width as usize > self.output_w as usize
            || output_y + output.region.height as usize > self.output_h as usize
        {
            return false;
        }

        match self.fast_path.as_ref() {
            Some(AffineFastPath::Nearest { xs, ys }) => {
                self.process_region_nearest_fast(input, output, xs, ys);
                true
            }
            Some(AffineFastPath::Bilinear { xs, ys }) => {
                self.process_region_bilinear_fast(input, output, xs, ys);
                true
            }
            Some(AffineFastPath::Bicubic { xs, ys }) => {
                self.process_region_bicubic_fast(input, output, xs, ys);
                true
            }
            None => false,
        }
    }

    fn process_region_nearest_fast(
        &self,
        input: &Tile<F>,
        output: &mut TileMut<F>,
        xs: &[AxisNearestSample],
        ys: &[AxisNearestSample],
    ) where
        F::Sample: ToF64 + FromF64,
    {
        let out_h = output.region.height as usize;
        let out_w = output.region.width as usize;
        let bands = input.bands as usize;
        let in_left = i64::from(input.region.x);
        let in_top = i64::from(input.region.y);
        let in_right = in_left + i64::from(input.region.width);
        let in_bottom = in_top + i64::from(input.region.height);
        let row_stride = input.region.width as usize * bands;

        for y_local in 0..out_h {
            let y_sample = &ys[output.region.y as usize + y_local];
            let row_base = y_local * out_w * bands;

            if y_sample.coord < in_top || y_sample.coord >= in_bottom {
                for x_local in 0..out_w {
                    let out_base = row_base + x_local * bands;
                    self.fill_extend_samples(&mut output.data[out_base..out_base + bands]);
                }
                continue;
            }

            let input_row = (y_sample.coord - in_top) as usize * row_stride;
            for x_local in 0..out_w {
                let x_sample = &xs[output.region.x as usize + x_local];
                let out_base = row_base + x_local * bands;
                if x_sample.coord < in_left || x_sample.coord >= in_right {
                    self.fill_extend_samples(&mut output.data[out_base..out_base + bands]);
                    continue;
                }

                let input_base = input_row + (x_sample.coord - in_left) as usize * bands;
                output.data[out_base..out_base + bands]
                    .copy_from_slice(&input.data[input_base..input_base + bands]);
            }
        }
    }

    fn process_region_bilinear_fast(
        &self,
        input: &Tile<F>,
        output: &mut TileMut<F>,
        xs: &[AxisLinearSample],
        ys: &[AxisLinearSample],
    ) where
        F::Sample: ToF64 + FromF64,
    {
        let out_h = output.region.height as usize;
        let out_w = output.region.width as usize;
        let bands = input.bands as usize;
        let in_left = i64::from(input.region.x);
        let in_top = i64::from(input.region.y);
        let in_right = in_left + i64::from(input.region.width);
        let in_bottom = in_top + i64::from(input.region.height);
        let row_stride = input.region.width as usize * bands;

        for y_local in 0..out_h {
            let y_sample = &ys[output.region.y as usize + y_local];
            let row_base = y_local * out_w * bands;
            let y_start = y_sample.start;

            if y_start < in_top || y_start + 1 >= in_bottom {
                for x_local in 0..out_w {
                    let x_sample = &xs[output.region.x as usize + x_local];
                    let out_base = row_base + x_local * bands;
                    self.sample_pixel_bilinear_into(
                        input,
                        x_sample.start as f64 + x_sample.weights[1],
                        y_start as f64 + y_sample.weights[1],
                        &mut output.data[out_base..out_base + bands],
                    );
                }
                continue;
            }

            let top_row = (y_start - in_top) as usize * row_stride;
            let bottom_row = top_row + row_stride;
            let wy0 = y_sample.weights[0];
            let wy1 = y_sample.weights[1];

            for x_local in 0..out_w {
                let x_sample = &xs[output.region.x as usize + x_local];
                let out_base = row_base + x_local * bands;
                let x_start = x_sample.start;

                if x_start < in_left || x_start + 1 >= in_right {
                    self.sample_pixel_bilinear_into(
                        input,
                        x_start as f64 + x_sample.weights[1],
                        y_start as f64 + y_sample.weights[1],
                        &mut output.data[out_base..out_base + bands],
                    );
                    continue;
                }

                let input_base = (x_start - in_left) as usize * bands;
                let wx0 = x_sample.weights[0];
                let wx1 = x_sample.weights[1];

                for band in 0..bands {
                    let p00 = input.data[top_row + input_base + band].to_f64();
                    let p01 = input.data[top_row + input_base + bands + band].to_f64();
                    let p10 = input.data[bottom_row + input_base + band].to_f64();
                    let p11 = input.data[bottom_row + input_base + bands + band].to_f64();
                    let top = p01.mul_add(wx1, p00 * wx0);
                    let bottom = p11.mul_add(wx1, p10 * wx0);
                    output.data[out_base + band] =
                        F::Sample::from_f64(bottom.mul_add(wy1, top * wy0));
                }
            }
        }
    }

    fn process_region_bicubic_fast(
        &self,
        input: &Tile<F>,
        output: &mut TileMut<F>,
        xs: &[AxisCubicSample],
        ys: &[AxisCubicSample],
    ) where
        F::Sample: ToF64 + FromF64,
    {
        if F::ID == BandFormatId::U8 {
            self.process_region_bicubic_fast_u8(
                input,
                output.region,
                xs,
                ys,
                bytemuck::cast_slice(input.data),
                bytemuck::cast_slice_mut(output.data),
            );
            return;
        }

        let out_h = output.region.height as usize;
        let out_w = output.region.width as usize;
        let bands = input.bands as usize;
        let in_left = i64::from(input.region.x);
        let in_top = i64::from(input.region.y);
        let in_right = in_left + i64::from(input.region.width);
        let in_bottom = in_top + i64::from(input.region.height);
        let row_stride = input.region.width as usize * bands;

        for y_local in 0..out_h {
            let y_sample = &ys[output.region.y as usize + y_local];
            let row_base = y_local * out_w * bands;
            let y_start = y_sample.start;

            if y_start < in_top || y_start + 3 >= in_bottom {
                for x_local in 0..out_w {
                    let out_base = row_base + x_local * bands;
                    let x_in = self.matrix[0]
                        .mul_add((output.region.x as usize + x_local) as f64, self.tx);
                    let y_in = self.matrix[3]
                        .mul_add((output.region.y as usize + y_local) as f64, self.ty);
                    self.sample_pixel_at(
                        input,
                        x_in,
                        y_in,
                        &mut output.data[out_base..out_base + bands],
                    );
                }
                continue;
            }

            let row_offsets = [
                (y_start - in_top) as usize * row_stride,
                (y_start - in_top + 1) as usize * row_stride,
                (y_start - in_top + 2) as usize * row_stride,
                (y_start - in_top + 3) as usize * row_stride,
            ];

            for x_local in 0..out_w {
                let x_sample = &xs[output.region.x as usize + x_local];
                let out_base = row_base + x_local * bands;
                let x_start = x_sample.start;

                if x_start < in_left || x_start + 3 >= in_right {
                    let x_in = self.matrix[0]
                        .mul_add((output.region.x as usize + x_local) as f64, self.tx);
                    let y_in = self.matrix[3]
                        .mul_add((output.region.y as usize + y_local) as f64, self.ty);
                    self.sample_pixel_at(
                        input,
                        x_in,
                        y_in,
                        &mut output.data[out_base..out_base + bands],
                    );
                    continue;
                }

                let input_base = (x_start - in_left) as usize * bands;
                for band in 0..bands {
                    let mut acc = 0.0;
                    for (row_idx, row_offset) in row_offsets.iter().enumerate() {
                        let row_base_in = row_offset + input_base;
                        let row_acc = input.data[row_base_in + bands * 3 + band].to_f64().mul_add(
                            x_sample.weights[3],
                            input.data[row_base_in + bands * 2 + band].to_f64().mul_add(
                                x_sample.weights[2],
                                input.data[row_base_in + bands + band].to_f64().mul_add(
                                    x_sample.weights[1],
                                    input.data[row_base_in + band].to_f64() * x_sample.weights[0],
                                ),
                            ),
                        );
                        acc = row_acc.mul_add(y_sample.weights[row_idx], acc);
                    }
                    output.data[out_base + band] = F::Sample::from_f64(acc);
                }
            }
        }
    }

    fn process_region_bicubic_fast_u8(
        &self,
        input: &Tile<F>,
        output_region: Region,
        xs: &[AxisCubicSample],
        ys: &[AxisCubicSample],
        input_data: &[u8],
        output_data: &mut [u8],
    ) where
        F::Sample: ToF64 + FromF64,
    {
        #[inline]
        const fn to_u8(value: f64) -> u8 {
            value.round().clamp(u8::MIN as f64, u8::MAX as f64) as u8
        }

        let out_h = output_region.height as usize;
        let out_w = output_region.width as usize;
        let bands = input.bands as usize;
        let in_left = i64::from(input.region.x);
        let in_top = i64::from(input.region.y);
        let in_right = in_left + i64::from(input.region.width);
        let in_bottom = in_top + i64::from(input.region.height);
        let row_stride = input.region.width as usize * bands;

        for y_local in 0..out_h {
            let y_sample = &ys[output_region.y as usize + y_local];
            let row_base = y_local * out_w * bands;
            let y_start = y_sample.start;

            if y_start < in_top || y_start + 3 >= in_bottom {
                for x_local in 0..out_w {
                    let out_base = row_base + x_local * bands;
                    let x_in = self.matrix[0]
                        .mul_add((output_region.x as usize + x_local) as f64, self.tx);
                    let y_in = self.matrix[3]
                        .mul_add((output_region.y as usize + y_local) as f64, self.ty);
                    for band in 0..bands {
                        output_data[out_base + band] =
                            to_u8(self.sample_at(input, x_in, y_in, band));
                    }
                }
                continue;
            }

            let row_offsets = [
                (y_start - in_top) as usize * row_stride,
                (y_start - in_top + 1) as usize * row_stride,
                (y_start - in_top + 2) as usize * row_stride,
                (y_start - in_top + 3) as usize * row_stride,
            ];

            for x_local in 0..out_w {
                let x_sample = &xs[output_region.x as usize + x_local];
                let out_base = row_base + x_local * bands;
                let x_start = x_sample.start;

                if x_start < in_left || x_start + 3 >= in_right {
                    let x_in = self.matrix[0]
                        .mul_add((output_region.x as usize + x_local) as f64, self.tx);
                    let y_in = self.matrix[3]
                        .mul_add((output_region.y as usize + y_local) as f64, self.ty);
                    for band in 0..bands {
                        output_data[out_base + band] =
                            to_u8(self.sample_at(input, x_in, y_in, band));
                    }
                    continue;
                }

                let input_base = (x_start - in_left) as usize * bands;
                match bands {
                    3 => {
                        let mut acc0 = 0.0;
                        let mut acc1 = 0.0;
                        let mut acc2 = 0.0;
                        for (row_idx, row_offset) in row_offsets.iter().enumerate() {
                            let base = row_offset + input_base;
                            let wx = &x_sample.weights;
                            let row0 = f64::from(input_data[base + 9]).mul_add(
                                wx[3],
                                f64::from(input_data[base + 6]).mul_add(
                                    wx[2],
                                    f64::from(input_data[base + 3])
                                        .mul_add(wx[1], f64::from(input_data[base]) * wx[0]),
                                ),
                            );
                            let row1 = f64::from(input_data[base + 10]).mul_add(
                                wx[3],
                                f64::from(input_data[base + 7]).mul_add(
                                    wx[2],
                                    f64::from(input_data[base + 4])
                                        .mul_add(wx[1], f64::from(input_data[base + 1]) * wx[0]),
                                ),
                            );
                            let row2 = f64::from(input_data[base + 11]).mul_add(
                                wx[3],
                                f64::from(input_data[base + 8]).mul_add(
                                    wx[2],
                                    f64::from(input_data[base + 5])
                                        .mul_add(wx[1], f64::from(input_data[base + 2]) * wx[0]),
                                ),
                            );
                            let wy = y_sample.weights[row_idx];
                            acc0 = row0.mul_add(wy, acc0);
                            acc1 = row1.mul_add(wy, acc1);
                            acc2 = row2.mul_add(wy, acc2);
                        }
                        output_data[out_base] = to_u8(acc0);
                        output_data[out_base + 1] = to_u8(acc1);
                        output_data[out_base + 2] = to_u8(acc2);
                    }
                    4 => {
                        let mut acc0 = 0.0;
                        let mut acc1 = 0.0;
                        let mut acc2 = 0.0;
                        let mut acc3 = 0.0;
                        for (row_idx, row_offset) in row_offsets.iter().enumerate() {
                            let base = row_offset + input_base;
                            let wx = &x_sample.weights;
                            let row0 = f64::from(input_data[base + 12]).mul_add(
                                wx[3],
                                f64::from(input_data[base + 8]).mul_add(
                                    wx[2],
                                    f64::from(input_data[base + 4])
                                        .mul_add(wx[1], f64::from(input_data[base]) * wx[0]),
                                ),
                            );
                            let row1 = f64::from(input_data[base + 13]).mul_add(
                                wx[3],
                                f64::from(input_data[base + 9]).mul_add(
                                    wx[2],
                                    f64::from(input_data[base + 5])
                                        .mul_add(wx[1], f64::from(input_data[base + 1]) * wx[0]),
                                ),
                            );
                            let row2 = f64::from(input_data[base + 14]).mul_add(
                                wx[3],
                                f64::from(input_data[base + 10]).mul_add(
                                    wx[2],
                                    f64::from(input_data[base + 6])
                                        .mul_add(wx[1], f64::from(input_data[base + 2]) * wx[0]),
                                ),
                            );
                            let row3 = f64::from(input_data[base + 15]).mul_add(
                                wx[3],
                                f64::from(input_data[base + 11]).mul_add(
                                    wx[2],
                                    f64::from(input_data[base + 7])
                                        .mul_add(wx[1], f64::from(input_data[base + 3]) * wx[0]),
                                ),
                            );
                            let wy = y_sample.weights[row_idx];
                            acc0 = row0.mul_add(wy, acc0);
                            acc1 = row1.mul_add(wy, acc1);
                            acc2 = row2.mul_add(wy, acc2);
                            acc3 = row3.mul_add(wy, acc3);
                        }
                        output_data[out_base] = to_u8(acc0);
                        output_data[out_base + 1] = to_u8(acc1);
                        output_data[out_base + 2] = to_u8(acc2);
                        output_data[out_base + 3] = to_u8(acc3);
                    }
                    _ => {
                        for band in 0..bands {
                            let mut acc = 0.0;
                            for (row_idx, row_offset) in row_offsets.iter().enumerate() {
                                let row_base_in = row_offset + input_base;
                                let row_acc = f64::from(input_data[row_base_in + bands * 3 + band])
                                    .mul_add(
                                        x_sample.weights[3],
                                        f64::from(input_data[row_base_in + bands * 2 + band])
                                            .mul_add(
                                                x_sample.weights[2],
                                                f64::from(input_data[row_base_in + bands + band])
                                                    .mul_add(
                                                        x_sample.weights[1],
                                                        f64::from(input_data[row_base_in + band])
                                                            * x_sample.weights[0],
                                                    ),
                                            ),
                                    );
                                acc = row_acc.mul_add(y_sample.weights[row_idx], acc);
                            }
                            output_data[out_base + band] = to_u8(acc);
                        }
                    }
                }
            }
        }
    }
}

impl<F: BandFormat> Op for Affine<F>
where
    F::Sample: ToF64 + FromF64 + bytemuck::Pod,
{
    type Input = F;
    type Output = F;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        // Affine reads tiles in arbitrary order. SmallTile keeps the requested
        // tile size square and bounded, reducing cache thrash when the transform
        // is far from axis-aligned. TileCache must be inserted upstream.
        DemandHint::SmallTile
    }

    fn required_input_region(&self, output: &Region) -> Region {
        // libvips computes affine demand from the mapped output pixel centres plus
        // the interpolator window, not by padding the continuous output rectangle
        // symmetrically. That keeps identity nearest exact and limits bilinear/
        // cubic/lanczos demand to the taps this implementation actually reads.
        let ((min_x, max_x), (min_y, max_y)) = self.mapped_input_bounds(output);
        let (left_pad, right_pad) = self.kernel_padding();

        let x0 = min_x.floor() as i32 - left_pad;
        let x1 = max_x.floor() as i32 + right_pad;
        let y0 = min_y.floor() as i32 - left_pad;
        let y1 = max_y.floor() as i32 + right_pad;

        Region::new(x0, y0, (x1 - x0 + 1) as u32, (y1 - y0 + 1) as u32)
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        let out_span_x = self.matrix[1].abs().mul_add(
            f64::from(tile_h.saturating_sub(1)),
            self.matrix[0].abs() * f64::from(tile_w.saturating_sub(1)),
        );
        let out_span_y = self.matrix[3].abs().mul_add(
            f64::from(tile_h.saturating_sub(1)),
            self.matrix[2].abs() * f64::from(tile_w.saturating_sub(1)),
        );
        let (left_pad, right_pad) = self.kernel_padding();
        let halo = (left_pad + right_pad) as u32;

        NodeSpec {
            input_tile_w: out_span_x.ceil() as u32 + 1 + halo,
            input_tile_h: out_span_y.ceil() as u32 + 1 + halo,
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        if self.process_region_fast_path(input, output) {
            return;
        }

        let out_h = output.region.height as usize;
        let out_w = output.region.width as usize;
        let bands = input.bands as usize;
        let mut row_x = self.matrix[1].mul_add(
            f64::from(output.region.y),
            self.matrix[0] * f64::from(output.region.x),
        ) + self.tx;
        let mut row_y = self.matrix[3].mul_add(
            f64::from(output.region.y),
            self.matrix[2] * f64::from(output.region.x),
        ) + self.ty;
        let step_x = self.matrix[0];
        let step_y = self.matrix[2];

        for y_local in 0..out_h {
            let mut x_in = row_x;
            let mut y_in = row_y;
            let row_base = y_local * out_w * bands;

            for x_local in 0..out_w {
                let out_base = row_base + x_local * bands;
                self.sample_pixel_at(
                    input,
                    x_in,
                    y_in,
                    &mut output.data[out_base..out_base + bands],
                );
                x_in += step_x;
                y_in += step_y;
            }

            row_x += self.matrix[1];
            row_y += self.matrix[3];
        }
    }
}

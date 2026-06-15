use crate::domain::ops::resample::sample_conv::FromF64;
use crate::domain::{
    error::{BuildError, ViprsError},
    format::{BandFormat, BandFormatId},
    image::Tile,
    kernel::InterpolationKernel,
};

use super::{Affine, ExtendMode};

impl<F: BandFormat> Affine<F> {
    /// Returns or performs try new.
    pub fn try_new(
        matrix: [f64; 4],
        tx: f64,
        ty: f64,
        kernel: InterpolationKernel,
        output_w: u32,
        output_h: u32,
    ) -> Result<Self, ViprsError> {
        if !matrix.iter().all(|value| value.is_finite()) || !tx.is_finite() || !ty.is_finite() {
            return Err(BuildError::InvalidAffineMatrix {
                matrix,
                reason: "matrix coefficients and translation must be finite",
            }
            .into());
        }

        let determinant = matrix[1].mul_add(-matrix[2], matrix[0] * matrix[3]);
        if determinant.abs() <= f64::EPSILON {
            return Err(ViprsError::DegenerateAffineTransform {
                matrix,
                output_width: output_w,
                output_height: output_h,
                reason: "matrix determinant is singular",
            });
        }

        Ok(Self::new(matrix, tx, ty, kernel, output_w, output_h))
    }

    #[inline]
    pub(super) fn fill_extend_samples(&self, output: &mut [F::Sample])
    where
        F::Sample: FromF64,
    {
        let bands = output.len();
        for (band, sample) in output.iter_mut().enumerate() {
            *sample = F::Sample::from_f64(self.extend_fill_value(bands, band));
        }
    }

    #[inline]
    pub(super) fn extend_fill_value(&self, bands: usize, band: usize) -> f64 {
        match &self.extend {
            ExtendMode::White => Self::white_value(),
            ExtendMode::Background(values) => match values.as_slice() {
                [] => 0.0,
                [value] => *value,
                values if band < values.len() => values[band],
                values if values.len() == bands => values[band],
                values => values[values.len() - 1],
            },
            ExtendMode::Black
            | ExtendMode::Copy
            | ExtendMode::Edge
            | ExtendMode::Repeat
            | ExtendMode::Mirror => 0.0,
        }
    }

    #[inline]
    const fn white_value() -> f64 {
        match F::ID {
            BandFormatId::F32 | BandFormatId::F64 => 1.0,
            BandFormatId::U8 => u8::MAX as f64,
            BandFormatId::U16 => u16::MAX as f64,
            BandFormatId::I16 => i16::MAX as f64,
            BandFormatId::U32 => u32::MAX as f64,
            BandFormatId::I32 => i32::MAX as f64,
        }
    }

    #[inline]
    pub(super) const fn uses_constant_fill_extend(&self) -> bool {
        matches!(
            self.extend,
            ExtendMode::Black | ExtendMode::White | ExtendMode::Background(_)
        )
    }

    #[inline]
    fn resolve_source_coord(
        coord: i64,
        origin: i64,
        size: u32,
        extend: &ExtendMode,
    ) -> Option<i64> {
        let size = i64::from(size);
        if size <= 0 {
            return None;
        }

        let relative = coord - origin;
        if (0..size).contains(&relative) {
            return Some(coord);
        }

        match extend {
            ExtendMode::Black | ExtendMode::White | ExtendMode::Background(_) => None,
            ExtendMode::Copy | ExtendMode::Edge => Some(relative.clamp(0, size - 1) + origin),
            ExtendMode::Repeat => Some(relative.rem_euclid(size) + origin),
            ExtendMode::Mirror => {
                let period = size * 2;
                let wrapped = relative.rem_euclid(period);
                let mirrored = if wrapped < size {
                    wrapped
                } else {
                    period - 1 - wrapped
                };
                Some(mirrored + origin)
            }
        }
    }

    #[inline]
    pub(super) fn resolve_sample_coords(
        &self,
        input: &Tile<F>,
        ix: i64,
        iy: i64,
    ) -> Option<(i64, i64)> {
        let x = Self::resolve_source_coord(
            ix,
            i64::from(input.region.x),
            input.region.width,
            &self.extend,
        )?;
        let y = Self::resolve_source_coord(
            iy,
            i64::from(input.region.y),
            input.region.height,
            &self.extend,
        )?;
        Some((x, y))
    }

    #[inline]
    pub(super) fn resolved_pixel_base(&self, input: &Tile<F>, ix: i64, iy: i64) -> Option<usize> {
        let (resolved_x, resolved_y) = self.resolve_sample_coords(input, ix, iy)?;
        let tile_x = resolved_x - i64::from(input.region.x);
        let tile_y = resolved_y - i64::from(input.region.y);
        let width = input.region.width as usize;
        let bands = input.bands as usize;
        Some((tile_y as usize * width + tile_x as usize) * bands)
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
    pub(super) fn lbbicubic(stencil: &[f64; 16], relative_x: f64, relative_y: f64) -> f64 {
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
        let twelve_dif00 = 6.0 * (dble_dzdx00 - dble_dzdy00);
        let twelve_sum10 = 6.0 * (dble_dzdx10 + dble_dzdy10);
        let twelve_dif10 = 6.0 * (dble_dzdx10 - dble_dzdy10);
        let twelve_sum01 = 6.0 * (dble_dzdx01 + dble_dzdy01);
        let twelve_dif01 = 6.0 * (dble_dzdx01 - dble_dzdy01);
        let twelve_sum11 = 6.0 * (dble_dzdx11 + dble_dzdy11);
        let twelve_dif11 = 6.0 * (dble_dzdx11 - dble_dzdy11);

        let twelve_abs_sum00 = twelve_sum00.abs();
        let twelve_abs_sum10 = twelve_sum10.abs();
        let twelve_abs_sum01 = twelve_sum01.abs();
        let twelve_abs_sum11 = twelve_sum11.abs();

        let u00_times_36 = 36.0 * u00;
        let u10_times_36 = 36.0 * u10;
        let u01_times_36 = 36.0 * u01;
        let u11_times_36 = 36.0 * u11;

        let first_limit00 = twelve_abs_sum00 - u00_times_36;
        let first_limit10 = twelve_abs_sum10 - u10_times_36;
        let first_limit01 = twelve_abs_sum01 - u01_times_36;
        let first_limit11 = twelve_abs_sum11 - u11_times_36;

        let quad_d2zdxdy00ii = Self::lbb_max(quad_d2zdxdy00i, first_limit00);
        let quad_d2zdxdy10ii = Self::lbb_max(quad_d2zdxdy10i, first_limit10);
        let quad_d2zdxdy01ii = Self::lbb_max(quad_d2zdxdy01i, first_limit01);
        let quad_d2zdxdy11ii = Self::lbb_max(quad_d2zdxdy11i, first_limit11);

        let v00_times_36 = 36.0 * v00;
        let v10_times_36 = 36.0 * v10;
        let v01_times_36 = 36.0 * v01;
        let v11_times_36 = 36.0 * v11;

        let second_limit00 = v00_times_36 - twelve_abs_sum00;
        let second_limit10 = v10_times_36 - twelve_abs_sum10;
        let second_limit01 = v01_times_36 - twelve_abs_sum01;
        let second_limit11 = v11_times_36 - twelve_abs_sum11;

        let quad_d2zdxdy00iii = Self::lbb_min(quad_d2zdxdy00ii, second_limit00);
        let quad_d2zdxdy10iii = Self::lbb_min(quad_d2zdxdy10ii, second_limit10);
        let quad_d2zdxdy01iii = Self::lbb_min(quad_d2zdxdy01ii, second_limit01);
        let quad_d2zdxdy11iii = Self::lbb_min(quad_d2zdxdy11ii, second_limit11);

        let twelve_abs_dif00 = twelve_dif00.abs();
        let twelve_abs_dif10 = twelve_dif10.abs();
        let twelve_abs_dif01 = twelve_dif01.abs();
        let twelve_abs_dif11 = twelve_dif11.abs();

        let third_limit00 = twelve_abs_dif00 - v00_times_36;
        let third_limit10 = twelve_abs_dif10 - v10_times_36;
        let third_limit01 = twelve_abs_dif01 - v01_times_36;
        let third_limit11 = twelve_abs_dif11 - v11_times_36;

        let quad_d2zdxdy00iiii = Self::lbb_max(quad_d2zdxdy00iii, third_limit00);
        let quad_d2zdxdy10iiii = Self::lbb_max(quad_d2zdxdy10iii, third_limit10);
        let quad_d2zdxdy01iiii = Self::lbb_max(quad_d2zdxdy01iii, third_limit01);
        let quad_d2zdxdy11iiii = Self::lbb_max(quad_d2zdxdy11iii, third_limit11);

        let fourth_limit00 = u00_times_36 - twelve_abs_dif00;
        let fourth_limit10 = u10_times_36 - twelve_abs_dif10;
        let fourth_limit01 = u01_times_36 - twelve_abs_dif01;
        let fourth_limit11 = u11_times_36 - twelve_abs_dif11;

        let quad_d2zdxdy00 = Self::lbb_min(quad_d2zdxdy00iiii, fourth_limit00);
        let quad_d2zdxdy10 = Self::lbb_min(quad_d2zdxdy10iiii, fourth_limit10);
        let quad_d2zdxdy01 = Self::lbb_min(quad_d2zdxdy01iiii, fourth_limit01);
        let quad_d2zdxdy11 = Self::lbb_min(quad_d2zdxdy11iiii, fourth_limit11);

        let xp1over2 = relative_x;
        let xm1over2 = xp1over2 - 1.0;
        let onepx = 0.5 + xp1over2;
        let onemx = 1.5 - xp1over2;
        let xp1over2sq = xp1over2 * xp1over2;

        let yp1over2 = relative_y;
        let ym1over2 = yp1over2 - 1.0;
        let onepy = 0.5 + yp1over2;
        let onemy = 1.5 - yp1over2;
        let yp1over2sq = yp1over2 * yp1over2;

        let xm1over2sq = xm1over2 * xm1over2;
        let ym1over2sq = ym1over2 * ym1over2;

        let twice1px = onepx + onepx;
        let twice1py = onepy + onepy;
        let twice1mx = onemx + onemx;
        let twice1my = onemy + onemy;

        let xm1over2sq_times_ym1over2sq = xm1over2sq * ym1over2sq;
        let xp1over2sq_times_ym1over2sq = xp1over2sq * ym1over2sq;
        let xp1over2sq_times_yp1over2sq = xp1over2sq * yp1over2sq;
        let xm1over2sq_times_yp1over2sq = xm1over2sq * yp1over2sq;

        let four_times_1px_times_1py = twice1px * twice1py;
        let four_times_1mx_times_1py = twice1mx * twice1py;
        let twice_xp1over2_times_1py = xp1over2 * twice1py;
        let twice_xm1over2_times_1py = xm1over2 * twice1py;
        let twice_xm1over2_times_1my = xm1over2 * twice1my;
        let twice_xp1over2_times_1my = xp1over2 * twice1my;
        let four_times_1mx_times_1my = twice1mx * twice1my;
        let four_times_1px_times_1my = twice1px * twice1my;
        let twice_1px_times_ym1over2 = twice1px * ym1over2;
        let twice_1mx_times_ym1over2 = twice1mx * ym1over2;
        let xp1over2_times_ym1over2 = xp1over2 * ym1over2;
        let xm1over2_times_ym1over2 = xm1over2 * ym1over2;
        let xm1over2_times_yp1over2 = xm1over2 * yp1over2;
        let xp1over2_times_yp1over2 = xp1over2 * yp1over2;
        let twice_1mx_times_yp1over2 = twice1mx * yp1over2;
        let twice_1px_times_yp1over2 = twice1px * yp1over2;

        let c00 = four_times_1px_times_1py * xm1over2sq_times_ym1over2sq;
        let c00dx = twice_xp1over2_times_1py * xm1over2sq_times_ym1over2sq;
        let c00dy = twice_1px_times_yp1over2 * xm1over2sq_times_ym1over2sq;
        let c00dxdy = xp1over2_times_yp1over2 * xm1over2sq_times_ym1over2sq;

        let c10 = four_times_1mx_times_1py * xp1over2sq_times_ym1over2sq;
        let c10dx = twice_xm1over2_times_1py * xp1over2sq_times_ym1over2sq;
        let c10dy = twice_1mx_times_yp1over2 * xp1over2sq_times_ym1over2sq;
        let c10dxdy = xm1over2_times_yp1over2 * xp1over2sq_times_ym1over2sq;

        let c01 = four_times_1px_times_1my * xm1over2sq_times_yp1over2sq;
        let c01dx = twice_xp1over2_times_1my * xm1over2sq_times_yp1over2sq;
        let c01dy = twice_1px_times_ym1over2 * xm1over2sq_times_yp1over2sq;
        let c01dxdy = xp1over2_times_ym1over2 * xm1over2sq_times_yp1over2sq;

        let c11 = four_times_1mx_times_1my * xp1over2sq_times_yp1over2sq;
        let c11dx = twice_xm1over2_times_1my * xp1over2sq_times_yp1over2sq;
        let c11dy = twice_1mx_times_ym1over2 * xp1over2sq_times_yp1over2sq;
        let c11dxdy = xm1over2_times_ym1over2 * xp1over2sq_times_yp1over2sq;

        let newval1 = c00 * dos_two + c10 * dos_thr + c01 * tre_two + c11 * tre_thr;
        let newval2 = c00dx * dble_dzdx00
            + c10dx * dble_dzdx10
            + c01dx * dble_dzdx01
            + c11dx * dble_dzdx11
            + c00dy * dble_dzdy00
            + c10dy * dble_dzdy10
            + c01dy * dble_dzdy01
            + c11dy * dble_dzdy11;
        let newval3 = c00dxdy * quad_d2zdxdy00
            + c10dxdy * quad_d2zdxdy10
            + c01dxdy * quad_d2zdxdy01
            + c11dxdy * quad_d2zdxdy11;

        0.25f64.mul_add(newval3, 0.5f64.mul_add(newval2, newval1))
    }
}

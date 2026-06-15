#![allow(clippy::no_effect_underscore_binding)]
// REASON: underscore-prefixed temporaries document unused branch products in the tone-curve derivation.

use std::marker::PhantomData;

use crate::domain::{
    error::ViprsError,
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

/// Build a libvips-compatible tone LUT.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::create::tonelut::TonelutOp;
///
/// let op = TonelutOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TonelutOp<F: BandFormat> {
    in_max: u32,
    out_max: u32,
    lb: f64,
    lw: f64,
    s: f64,
    m: f64,
    h: f64,
    ls: f64,
    lm: f64,
    lh: f64,
    _format: PhantomData<F>,
}

impl<F: BandFormat> TonelutOp<F> {
    #[allow(clippy::too_many_arguments)]
    /// Creates a new `TonelutOp`.
    pub fn new(
        in_max: u32,
        out_max: u32,
        lb: f64,
        lw: f64,
        ps: f64,
        pm: f64,
        ph: f64,
        s: f64,
        m: f64,
        h: f64,
    ) -> Result<Self, ViprsError> {
        if !(1..65_536).contains(&in_max) {
            return Err(ViprsError::Scheduler(format!(
                "TonelutOp in_max must be in [1, 65535], got {in_max}"
            )));
        }
        if !(1..65_536).contains(&out_max) {
            return Err(ViprsError::Scheduler(format!(
                "TonelutOp out_max must be in [1, 65535], got {out_max}"
            )));
        }
        if !lb.is_finite()
            || !lw.is_finite()
            || !ps.is_finite()
            || !pm.is_finite()
            || !ph.is_finite()
            || !s.is_finite()
            || !m.is_finite()
            || !h.is_finite()
        {
            return Err(ViprsError::Scheduler(
                "TonelutOp parameters must be finite".to_string(),
            ));
        }

        let ls = ps.mul_add(lw - lb, lb);
        let lm = pm.mul_add(lw - lb, lb);
        let lh = ph.mul_add(lw - lb, lb);

        Ok(Self {
            in_max,
            out_max,
            lb,
            lw,
            s,
            m,
            h,
            ls,
            lm,
            lh,
            _format: PhantomData,
        })
    }

    #[must_use]
    /// Returns or performs default params.
    pub const fn default_params() -> Self {
        Self {
            in_max: 32_767,
            out_max: 32_767,
            lb: 0.0,
            lw: 100.0,
            s: 0.0,
            m: 0.0,
            h: 0.0,
            ls: 20.0,
            lm: 50.0,
            lh: 80.0,
            _format: PhantomData,
        }
    }

    #[must_use]
    /// Returns or performs width.
    pub const fn width(&self) -> u32 {
        self.in_max + 1
    }

    #[must_use]
    /// Returns or performs height.
    pub const fn height(&self) -> u32 {
        1
    }

    #[inline(always)]
    fn shad(&self, x: f64) -> f64 {
        smooth_bump(x, self.lb, self.ls, self.lm)
    }

    #[inline(always)]
    fn mid(&self, x: f64) -> f64 {
        smooth_bump(x, self.ls, self.lm, self.lh)
    }

    #[inline(always)]
    fn high(&self, x: f64) -> f64 {
        smooth_bump(x, self.lm, self.lh, self.lw)
    }

    #[inline(always)]
    fn tone_curve(&self, x: f64) -> f64 {
        self.h.mul_add(
            self.high(x),
            self.m.mul_add(self.mid(x), self.s.mul_add(self.shad(x), x)),
        )
    }

    #[inline(always)]
    fn sample_value(&self, index: u32) -> f64 {
        let tone = self.tone_curve(100.0 * f64::from(index) / f64::from(self.in_max));
        ((f64::from(self.out_max) / 100.0) * tone).clamp(0.0, f64::from(self.out_max))
    }
}

#[inline(always)]
fn smoothstep01(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    (2.0 * x * x).mul_add(-x, 3.0 * x * x)
}

#[inline(always)]
fn normalized(x: f64, start: f64, end: f64) -> f64 {
    let span = end - start;
    if span.abs() <= f64::EPSILON {
        0.0
    } else {
        (x - start) / span
    }
}

// libvips tonelut materializes integer LUT entries via C integer casts, so
// parity requires truncation for integral outputs instead of round-to-nearest.
trait TonelutSample: Copy {
    fn from_tonelut_f64(value: f64) -> Self;
}

impl TonelutSample for u8 {
    #[inline(always)]
    fn from_tonelut_f64(value: f64) -> Self {
        value
            .clamp(f64::from(Self::MIN), f64::from(Self::MAX))
            .trunc() as Self
    }
}

impl TonelutSample for u16 {
    #[inline(always)]
    fn from_tonelut_f64(value: f64) -> Self {
        value
            .clamp(f64::from(Self::MIN), f64::from(Self::MAX))
            .trunc() as Self
    }
}

impl TonelutSample for i16 {
    #[inline(always)]
    fn from_tonelut_f64(value: f64) -> Self {
        value
            .clamp(f64::from(Self::MIN), f64::from(Self::MAX))
            .trunc() as Self
    }
}

impl TonelutSample for u32 {
    #[inline(always)]
    fn from_tonelut_f64(value: f64) -> Self {
        value
            .clamp(f64::from(Self::MIN), f64::from(Self::MAX))
            .trunc() as Self
    }
}

impl TonelutSample for i32 {
    #[inline(always)]
    fn from_tonelut_f64(value: f64) -> Self {
        value
            .clamp(f64::from(Self::MIN), f64::from(Self::MAX))
            .trunc() as Self
    }
}

impl TonelutSample for f32 {
    #[inline(always)]
    fn from_tonelut_f64(value: f64) -> Self {
        value as Self
    }
}

impl TonelutSample for f64 {
    #[inline(always)]
    fn from_tonelut_f64(value: f64) -> Self {
        value
    }
}

fn smooth_bump(x: f64, start: f64, peak: f64, end: f64) -> f64 {
    if x < start {
        0.0
    } else if x < peak {
        smoothstep01(normalized(x, start, peak))
    } else if x < end {
        1.0 - smoothstep01(normalized(x, peak, end))
    } else {
        0.0
    }
}

impl<F> Op for TonelutOp<F>
where
    F: BandFormat,
    F::Sample: TonelutSample,
{
    type Input = F;
    type Output = F;
    type State = ();

    const OUTPUT_BANDS: Option<usize> = Some(1);

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), _input: &Tile<F>, output: &mut TileMut<F>) {
        debug_assert_eq!(output.bands, 1, "TonelutOp output must be single-band");
        debug_assert!(output.region.x >= 0 && output.region.y >= 0);
        debug_assert!(output.region.x as u32 + output.region.width <= self.width());
        debug_assert!(output.region.y as u32 + output.region.height <= self.height());

        let region_width = output.region.width as usize;
        for row in 0..output.region.height as usize {
            let _y = output.region.y as usize + row;
            for col in 0..region_width {
                let x = output.region.x as u32 + col as u32;
                output.data[row * region_width + col] =
                    F::Sample::from_tonelut_f64(self.sample_value(x));
            }
        }
    }
}

impl<F> PixelLocalOp for TonelutOp<F>
where
    F: BandFormat,
    F::Sample: TonelutSample,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        format::{F32, U16},
        image::{Region, Tile, TileMut},
    };
    use proptest::prelude::*;

    fn render_f32(op: TonelutOp<F32>) -> Vec<f32> {
        let region = Region::new(0, 0, op.width(), op.height());
        let input_data = vec![0.0f32; region.pixel_count()];
        let mut output_data = vec![0.0f32; region.pixel_count()];
        let input = Tile::<F32>::new(region, 1, &input_data);
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);
        output_data
    }

    fn render_u16(op: TonelutOp<U16>) -> Vec<u16> {
        let region = Region::new(0, 0, op.width(), op.height());
        let input_data = vec![0u16; region.pixel_count()];
        let mut output_data = vec![0u16; region.pixel_count()];
        let input = Tile::<U16>::new(region, 1, &input_data);
        let mut output = TileMut::<U16>::new(region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);
        output_data
    }

    fn render_f32_region(op: TonelutOp<F32>, output_region: Region) -> Vec<f32> {
        let input_region = Region::new(0, 0, op.width(), op.height());
        let input_data = vec![0.0f32; input_region.pixel_count()];
        let mut output_data = vec![0.0f32; output_region.pixel_count()];
        let input = Tile::<F32>::new(input_region, 1, &input_data);
        let mut output = TileMut::<F32>::new(output_region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);
        output_data
    }

    #[test]
    fn default_curve_is_identity() {
        let op = TonelutOp::<F32>::default_params();
        let rendered = render_f32(op);
        assert!((rendered[0] - 0.0).abs() < 1e-6);
        assert!((rendered[32_767] - 32_767.0).abs() < 1e-3);
    }

    #[test]
    fn dimensions_follow_in_max() {
        let op = TonelutOp::<F32>::new(255, 255, 0.0, 100.0, 0.2, 0.5, 0.8, 0.0, 0.0, 0.0).unwrap();
        assert_eq!(op.width(), 256);
        assert_eq!(op.height(), 1);
    }

    #[test]
    fn u16_output_stays_in_range() {
        let op =
            TonelutOp::<U16>::new(255, 511, 0.0, 100.0, 0.2, 0.5, 0.8, 10.0, 0.0, -10.0).unwrap();
        let rendered = render_u16(op);
        assert!(rendered.iter().all(|&value| value <= 511));
    }

    #[test]
    fn rejects_non_finite_parameters() {
        assert!(
            TonelutOp::<F32>::new(255, 255, f64::NAN, 100.0, 0.2, 0.5, 0.8, 0.0, 0.0, 0.0).is_err()
        );
        assert!(
            TonelutOp::<F32>::new(255, 255, 0.0, 100.0, 0.2, 0.5, 0.8, f64::INFINITY, 0.0, 0.0)
                .is_err()
        );
    }

    #[test]
    fn rejects_out_of_range_limits() {
        assert!(TonelutOp::<F32>::new(0, 255, 0.0, 100.0, 0.2, 0.5, 0.8, 0.0, 0.0, 0.0).is_err());
        assert!(
            TonelutOp::<F32>::new(255, 65_536, 0.0, 100.0, 0.2, 0.5, 0.8, 0.0, 0.0, 0.0).is_err()
        );
    }

    #[test]
    fn smooth_bump_handles_all_curve_segments() {
        assert_eq!(smoothstep01(-1.0), 0.0);
        assert_eq!(smoothstep01(2.0), 1.0);
        assert_eq!(smooth_bump(-1.0, 0.0, 5.0, 10.0), 0.0);
        assert!((smooth_bump(2.5, 0.0, 5.0, 10.0) - 0.5).abs() < 1e-6);
        assert!((smooth_bump(7.5, 0.0, 5.0, 10.0) - 0.5).abs() < 1e-6);
        assert_eq!(smooth_bump(11.0, 0.0, 5.0, 10.0), 0.0);
        assert_eq!(normalized(3.0, 1.0, 1.0), 0.0);
    }

    #[test]
    fn partial_region_uses_requested_x_offset() {
        let op = TonelutOp::<F32>::new(7, 70, 0.0, 100.0, 0.2, 0.5, 0.8, 5.0, -5.0, 2.0).unwrap();
        let full = render_f32(op);
        let partial_op =
            TonelutOp::<F32>::new(7, 70, 0.0, 100.0, 0.2, 0.5, 0.8, 5.0, -5.0, 2.0).unwrap();
        let partial = render_f32_region(partial_op, Region::new(2, 0, 3, 1));

        assert_eq!(partial, full[2..5].to_vec());
    }

    #[test]
    fn metadata_reports_single_band_identity_geometry() {
        let op = TonelutOp::<F32>::default_params();
        let region = Region::new(5, 0, 7, 1);

        assert_eq!(op.demand_hint(), DemandHint::Any);
        assert_eq!(op.required_input_region(&region), region);
        op.start();
    }

    #[test]
    fn integer_outputs_truncate_like_libvips_casts() {
        let float_op =
            TonelutOp::<F32>::new(7, 15, 0.0, 100.0, 0.2, 0.5, 0.8, 5.0, -5.0, 2.0).unwrap();
        let int_op =
            TonelutOp::<U16>::new(7, 15, 0.0, 100.0, 0.2, 0.5, 0.8, 5.0, -5.0, 2.0).unwrap();

        let float_values = render_f32(float_op);
        let int_values = render_u16(int_op);

        for (float_value, int_value) in float_values.iter().zip(int_values.iter()) {
            assert_eq!(*int_value, float_value.trunc() as u16);
        }
    }

    proptest! {
        #[test]
        fn prop_curve_has_expected_length_and_range(
            in_max in 32u32..=1024,
            out_max in 32u32..=1024,
            s in -30.0f64..=30.0,
            m in -30.0f64..=30.0,
            h in -30.0f64..=30.0,
        ) {
            let op = TonelutOp::<F32>::new(in_max, out_max, 0.0, 100.0, 0.2, 0.5, 0.8, s, m, h)
                .unwrap();
            let rendered = render_f32(op);

            prop_assert_eq!(rendered.len(), in_max as usize + 1);
            prop_assert!(rendered.iter().all(|sample| *sample >= 0.0 && *sample <= out_max as f32));
        }
    }
}

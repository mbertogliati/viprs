//! Premultiply alpha channel into colour channels.
//!
//! Multiplies each colour band by the alpha value, normalised to [0, 1].
//! Required before resampling to avoid colour fringing ("halos") at
//! semi-transparent edges. Pair with [`Unpremultiply`] to reverse.
//!
//! [`Unpremultiply`]: super::unpremultiply::Unpremultiply

use std::marker::PhantomData;

use crate::domain::{
    format::{BandFormat, BandFormatId},
    image::{DemandHint, Interpretation, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

/// Premultiply the alpha channel into the colour channels.
///
/// Assumes the last band is alpha:
/// - **U8**: alpha ∈ [0, 255] by default; colour = round(colour * alpha / `max_alpha`).
/// - **F32/F64**: alpha ∈ [0.0, 1.0] by default; colour = colour * (alpha / `max_alpha`).
/// - **Integer formats beyond `U8`**: `libvips`-style normalization is available via
///   [`Premultiply::new_for_interpretation`] or [`Premultiply::new_with_max_alpha`].
pub struct Premultiply<F: BandFormat> {
    bands: u32,
    max_alpha: f64,
    _fmt: PhantomData<F>,
}

impl<F: BandFormat> Premultiply<F> {
    /// Create a new `Premultiply` for an image with `bands` bands.
    ///
    /// `bands` must be ≥ 2 (at least one colour band and one alpha band).
    /// The last band is always treated as alpha.
    #[must_use]
    pub const fn new(bands: u32) -> Self {
        Self::new_with_max_alpha(bands, default_max_alpha(F::ID))
    }

    /// Create a new `Premultiply` using an explicit maximum alpha value.
    #[must_use]
    pub const fn new_with_max_alpha(bands: u32, max_alpha: f64) -> Self {
        Self {
            bands,
            max_alpha,
            _fmt: PhantomData,
        }
    }

    /// Create a new `Premultiply` using libvips interpretation defaults.
    pub fn new_for_interpretation(bands: u32, interpretation: Option<Interpretation>) -> Self {
        let max_alpha =
            interpretation.map_or_else(|| default_max_alpha(F::ID), Interpretation::max_alpha);
        Self::new_with_max_alpha(bands, max_alpha)
    }
}

#[inline]
const fn default_max_alpha(format: BandFormatId) -> f64 {
    match format {
        BandFormatId::F32 | BandFormatId::F64 => 1.0,
        _ => 255.0,
    }
}

macro_rules! impl_integer_premultiply {
    ($format:ty, $sample:ty) => {
        impl Op for Premultiply<$format> {
            type Input = $format;
            type Output = $format;
            type State = ();

            fn demand_hint(&self) -> DemandHint {
                DemandHint::ThinStrip
            }

            fn required_input_region(&self, output: &Region) -> Region {
                *output
            }

            fn start(&self) {}

            #[inline]
            fn process_region(
                &self,
                _state: &mut (),
                input: &Tile<$format>,
                output: &mut TileMut<$format>,
            ) {
                let bands = self.bands as usize;
                debug_assert!(bands >= 2, "premultiply requires at least 2 bands");
                debug_assert!(
                    self.max_alpha > 0.0,
                    "premultiply max_alpha must be positive"
                );
                let alpha_band = bands - 1;
                let pixel_count = input.region.pixel_count();
                debug_assert_eq!(input.data.len(), pixel_count * bands);
                debug_assert_eq!(output.data.len(), pixel_count * bands);

                for (input_pixel, output_pixel) in input
                    .data
                    .chunks_exact(bands)
                    .zip(output.data.chunks_exact_mut(bands))
                    .take(pixel_count)
                {
                    let (input_colour, input_alpha) = input_pixel.split_at(alpha_band);
                    let (output_colour, output_alpha) = output_pixel.split_at_mut(alpha_band);
                    let alpha = (input_alpha[0] as f64).clamp(0.0, self.max_alpha);
                    let normalized_alpha = alpha / self.max_alpha;

                    for (sample, out_sample) in input_colour.iter().zip(output_colour.iter_mut()) {
                        *out_sample = ((*sample as f64) * normalized_alpha)
                            .round()
                            .clamp(<$sample>::MIN as f64, <$sample>::MAX as f64)
                            as $sample;
                    }

                    output_alpha[0] = input_alpha[0];
                }
            }
        }

        impl PixelLocalOp for Premultiply<$format> {}
    };
}

impl_integer_premultiply!(crate::domain::format::U8, u8);
impl_integer_premultiply!(crate::domain::format::U16, u16);
impl_integer_premultiply!(crate::domain::format::I16, i16);
impl_integer_premultiply!(crate::domain::format::U32, u32);
impl_integer_premultiply!(crate::domain::format::I32, i32);

impl Op for Premultiply<crate::domain::format::F32> {
    type Input = crate::domain::format::F32;
    type Output = crate::domain::format::F32;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(
        &self,
        _state: &mut (),
        input: &Tile<crate::domain::format::F32>,
        output: &mut TileMut<crate::domain::format::F32>,
    ) {
        let max_alpha = self.max_alpha as f32;
        let bands = self.bands as usize;
        debug_assert!(bands >= 2, "premultiply requires at least 2 bands");
        debug_assert!(max_alpha > 0.0, "premultiply max_alpha must be positive");
        let alpha_band = bands - 1;
        let pixel_count = input.region.pixel_count();
        debug_assert_eq!(input.data.len(), pixel_count * bands);
        debug_assert_eq!(output.data.len(), pixel_count * bands);

        for (input_pixel, output_pixel) in input
            .data
            .chunks_exact(bands)
            .zip(output.data.chunks_exact_mut(bands))
            .take(pixel_count)
        {
            let (input_colour, input_alpha) = input_pixel.split_at(alpha_band);
            let (output_colour, output_alpha) = output_pixel.split_at_mut(alpha_band);
            let alpha = input_alpha[0].clamp(0.0, max_alpha) / max_alpha;

            for (sample, out_sample) in input_colour.iter().zip(output_colour.iter_mut()) {
                *out_sample = *sample * alpha;
            }

            output_alpha[0] = input_alpha[0];
        }
    }
}

impl Op for Premultiply<crate::domain::format::F64> {
    type Input = crate::domain::format::F64;
    type Output = crate::domain::format::F64;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(
        &self,
        _state: &mut (),
        input: &Tile<crate::domain::format::F64>,
        output: &mut TileMut<crate::domain::format::F64>,
    ) {
        let bands = self.bands as usize;
        debug_assert!(bands >= 2, "premultiply requires at least 2 bands");
        debug_assert!(
            self.max_alpha > 0.0,
            "premultiply max_alpha must be positive"
        );
        let alpha_band = bands - 1;
        let pixel_count = input.region.pixel_count();

        for p in 0..pixel_count {
            let base = p * bands;
            let alpha = input.data[base + alpha_band].clamp(0.0, self.max_alpha) / self.max_alpha;
            for b in 0..alpha_band {
                output.data[base + b] = input.data[base + b] * alpha;
            }
            output.data[base + alpha_band] = input.data[base + alpha_band];
        }
    }
}

/// Returns the `BandFormatId` this `Premultiply` instance is specialised for.
///
/// Used by `PremultiplyBridge` to implement `DynOperation::input_format` /
/// `output_format` without needing a separate trait bound machinery.
impl<F: BandFormat> Premultiply<F> {
    #[must_use]
    /// Returns or performs format id.
    pub const fn format_id() -> BandFormatId {
        F::ID
    }
}

impl PixelLocalOp for Premultiply<crate::domain::format::F32> {}
impl PixelLocalOp for Premultiply<crate::domain::format::F64> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        format::{BandFormatId, F32, F64, U8, U16},
        image::{DemandHint, Interpretation, Region},
    };
    use proptest::prelude::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn run_premultiply_u8(input_data: &[u8], bands: u32) -> Vec<u8> {
        let pixel_count = input_data.len() / bands as usize;
        let width = pixel_count as u32;
        let region = Region::new(0, 0, width, 1);
        let op = Premultiply::<U8>::new(bands);
        let mut out = vec![0u8; input_data.len()];
        let input = Tile::<U8>::new(region, bands, input_data);
        let mut output = TileMut::<U8>::new(region, bands, &mut out);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        out
    }

    fn run_premultiply_f32(input_data: &[f32], bands: u32) -> Vec<f32> {
        let pixel_count = input_data.len() / bands as usize;
        let width = pixel_count as u32;
        let region = Region::new(0, 0, width, 1);
        let op = Premultiply::<F32>::new(bands);
        let mut out = vec![0.0f32; input_data.len()];
        let input = Tile::<F32>::new(region, bands, input_data);
        let mut output = TileMut::<F32>::new(region, bands, &mut out);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        out
    }

    fn run_premultiply_f64(input_data: &[f64], bands: u32) -> Vec<f64> {
        let pixel_count = input_data.len() / bands as usize;
        let width = pixel_count as u32;
        let region = Region::new(0, 0, width, 1);
        let op = Premultiply::<F64>::new(bands);
        let mut out = vec![0.0f64; input_data.len()];
        let input = Tile::<F64>::new(region, bands, input_data);
        let mut output = TileMut::<F64>::new(region, bands, &mut out);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        out
    }

    fn run_premultiply_u16_for_interpretation(
        input_data: &[u16],
        bands: u32,
        interpretation: Option<Interpretation>,
    ) -> Vec<u16> {
        let pixel_count = input_data.len() / bands as usize;
        let width = pixel_count as u32;
        let region = Region::new(0, 0, width, 1);
        let op = Premultiply::<U16>::new_for_interpretation(bands, interpretation);
        let mut out = vec![0u16; input_data.len()];
        let input = Tile::<U16>::new(region, bands, input_data);
        let mut output = TileMut::<U16>::new(region, bands, &mut out);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        out
    }

    // ── U8: boundary values ───────────────────────────────────────────────────

    #[test]
    fn u8_fully_opaque_alpha_is_identity() {
        // alpha=255 → colour * (255/255) = colour * 1.0 → unchanged
        let input = vec![100u8, 200u8, 255u8]; // R=100, G=200, A=255
        let result = run_premultiply_u8(&input, 3);
        assert_eq!(result[0], 100);
        assert_eq!(result[1], 200);
        assert_eq!(result[2], 255); // alpha unchanged
    }

    #[test]
    fn u8_zero_alpha_produces_zero_colour() {
        // alpha=0 → colour * 0 = 0
        let input = vec![100u8, 200u8, 0u8]; // R=100, G=200, A=0
        let result = run_premultiply_u8(&input, 3);
        assert_eq!(result[0], 0);
        assert_eq!(result[1], 0);
        assert_eq!(result[2], 0); // alpha unchanged
    }

    #[test]
    fn u8_half_alpha_halves_colour() {
        // alpha=128 ≈ 0.502; 255 * 0.502 ≈ 128
        let input = vec![255u8, 0u8, 128u8]; // R=255, G=0, A=128
        let result = run_premultiply_u8(&input, 3);
        // 255 * (128/255) = 128.0 → round → 128
        assert_eq!(result[0], 128);
        assert_eq!(result[1], 0);
        assert_eq!(result[2], 128); // alpha unchanged
    }

    #[test]
    fn u16_rgb16_half_alpha_halves_colour() {
        let input = vec![65535u16, 32768u16, 16384u16, 32768u16];
        let result = run_premultiply_u16_for_interpretation(&input, 4, Some(Interpretation::Rgb16));
        assert_eq!(result, vec![32768u16, 16384u16, 8192u16, 32768u16]);
    }

    #[test]
    fn u16_multiband_defaults_to_255_max_alpha() {
        let input = vec![1000u16, 500u16, 128u16];
        let result =
            run_premultiply_u16_for_interpretation(&input, 3, Some(Interpretation::Multiband));
        assert_eq!(result, vec![502u16, 251u16, 128u16]);
    }

    // ── F32: boundary values ──────────────────────────────────────────────────

    #[test]
    fn f32_fully_opaque_alpha_is_identity() {
        let input = vec![0.5f32, 0.8f32, 1.0f32]; // R=0.5, G=0.8, A=1.0
        let result = run_premultiply_f32(&input, 3);
        assert!((result[0] - 0.5).abs() < f32::EPSILON);
        assert!((result[1] - 0.8).abs() < f32::EPSILON);
        assert!((result[2] - 1.0).abs() < f32::EPSILON); // alpha unchanged
    }

    #[test]
    fn f32_zero_alpha_produces_zero_colour() {
        let input = vec![0.5f32, 0.8f32, 0.0f32];
        let result = run_premultiply_f32(&input, 3);
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 0.0);
        assert_eq!(result[2], 0.0); // alpha unchanged
    }

    #[test]
    fn f32_half_alpha_halves_colour() {
        let input = vec![1.0f32, 0.0f32, 0.5f32];
        let result = run_premultiply_f32(&input, 3);
        assert!((result[0] - 0.5).abs() < f32::EPSILON);
        assert_eq!(result[1], 0.0);
        assert!((result[2] - 0.5).abs() < f32::EPSILON); // alpha unchanged
    }

    #[test]
    fn f64_half_alpha_halves_colour() {
        let input = vec![1.0f64, 0.0f64, 0.5f64];
        let result = run_premultiply_f64(&input, 3);
        assert!((result[0] - 0.5).abs() < f64::EPSILON);
        assert_eq!(result[1], 0.0);
        assert!((result[2] - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn metadata_helpers_report_expected_defaults() {
        let region = Region::new(1, 2, 3, 4);

        let f32_op = Premultiply::<F32>::new(4);
        assert_eq!(f32_op.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(f32_op.required_input_region(&region), region);
        f32_op.start();
        assert_eq!(Premultiply::<F32>::format_id(), BandFormatId::F32);

        let f64_op = Premultiply::<F64>::new(4);
        assert_eq!(f64_op.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(f64_op.required_input_region(&region), region);
        f64_op.start();
        assert_eq!(Premultiply::<F64>::format_id(), BandFormatId::F64);

        assert_eq!(Premultiply::<U16>::format_id(), BandFormatId::U16);
    }

    // ── Proptest: F32 identity at alpha=1 ─────────────────────────────────────

    proptest! {
        #[test]
        fn f32_fully_opaque_is_identity_proptest(
            r in 0.0f32..=1.0f32,
            g in 0.0f32..=1.0f32,
            b in 0.0f32..=1.0f32,
        ) {
            let input = vec![r, g, b, 1.0f32];
            let result = run_premultiply_f32(&input, 4);
            prop_assert!((result[0] - r).abs() < f32::EPSILON);
            prop_assert!((result[1] - g).abs() < f32::EPSILON);
            prop_assert!((result[2] - b).abs() < f32::EPSILON);
            prop_assert!((result[3] - 1.0f32).abs() < f32::EPSILON);
        }

        #[test]
        fn f32_zero_alpha_zeroes_colour_proptest(
            r in 0.0f32..=1.0f32,
            g in 0.0f32..=1.0f32,
            b in 0.0f32..=1.0f32,
        ) {
            let input = vec![r, g, b, 0.0f32];
            let result = run_premultiply_f32(&input, 4);
            prop_assert_eq!(result[0], 0.0f32);
            prop_assert_eq!(result[1], 0.0f32);
            prop_assert_eq!(result[2], 0.0f32);
        }

        #[test]
        fn u8_fully_opaque_is_identity_proptest(
            r in 0u8..=255u8,
            g in 0u8..=255u8,
            b in 0u8..=255u8,
        ) {
            let input = vec![r, g, b, 255u8];
            let result = run_premultiply_u8(&input, 4);
            prop_assert_eq!(result[0], r);
            prop_assert_eq!(result[1], g);
            prop_assert_eq!(result[2], b);
            prop_assert_eq!(result[3], 255u8);
        }

        #[test]
        fn u8_zero_alpha_zeroes_colour_proptest(
            r in 0u8..=255u8,
            g in 0u8..=255u8,
            b in 0u8..=255u8,
        ) {
            let input = vec![r, g, b, 0u8];
            let result = run_premultiply_u8(&input, 4);
            prop_assert_eq!(result[0], 0u8);
            prop_assert_eq!(result[1], 0u8);
            prop_assert_eq!(result[2], 0u8);
        }

        #[test]
        fn u16_rgb16_fully_opaque_is_identity_proptest(
            r in 0u16..=u16::MAX,
            g in 0u16..=u16::MAX,
            b in 0u16..=u16::MAX,
        ) {
            let input = vec![r, g, b, u16::MAX];
            let result = run_premultiply_u16_for_interpretation(
                &input,
                4,
                Some(Interpretation::Rgb16),
            );
            prop_assert_eq!(result[0], r);
            prop_assert_eq!(result[1], g);
            prop_assert_eq!(result[2], b);
            prop_assert_eq!(result[3], u16::MAX);
        }
    }
}

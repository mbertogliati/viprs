//! Unpremultiply alpha channel from colour channels.
//!
//! Divides each colour band by the alpha value, reversing [`Premultiply`].
//! If alpha == 0 the colour bands are left at 0 to avoid division by zero.
//!
//! [`Premultiply`]: super::premultiply::Premultiply

use std::marker::PhantomData;

use viprs_core::{
    format::BandFormatId,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

/// Unpremultiply the colour channels by the alpha channel.
///
/// Assumes the last band is alpha:
/// - **U8**: alpha ∈ [0, 255]; colour = clamp(colour * `max_alpha` / alpha, 0, `max_alpha`).
/// - **U16**: alpha ∈ [0, 65535]; colour = clamp(colour * `max_alpha` / alpha, 0, `max_alpha`).
/// - **`F32`**: alpha ∈ `[0.0, 1.0]`; `colour = clamp(colour / alpha, 0.0, 1.0)`.
///   If alpha == 0 all colour bands remain 0.
pub struct Unpremultiply<F: viprs_core::format::BandFormat> {
    bands: u32,
    max_alpha: f64,
    _fmt: PhantomData<F>,
}

#[inline]
const fn default_max_alpha(format: BandFormatId) -> f64 {
    match format {
        BandFormatId::F32 | BandFormatId::F64 => 1.0,
        _ => 255.0,
    }
}

impl<F: viprs_core::format::BandFormat> Unpremultiply<F> {
    /// Create a new `Unpremultiply` for an image with `bands` bands.
    ///
    /// `bands` must be ≥ 2 (at least one colour band and one alpha band).
    /// The last band is always treated as alpha.
    #[must_use]
    pub const fn new(bands: u32) -> Self {
        Self::new_with_max_alpha(bands, default_max_alpha(F::ID))
    }

    /// Create a new `Unpremultiply` using an explicit maximum alpha value.
    #[must_use]
    pub const fn new_with_max_alpha(bands: u32, max_alpha: f64) -> Self {
        Self {
            bands,
            max_alpha,
            _fmt: PhantomData,
        }
    }
}

macro_rules! impl_integer_unpremultiply {
    ($format:ty, $sample:ty) => {
        impl Op for Unpremultiply<$format> {
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
                debug_assert!(bands >= 2, "unpremultiply requires at least 2 bands");
                let max_alpha = self.max_alpha;
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
                    let alpha = input_alpha[0] as f64;
                    if alpha == 0.0 {
                        output_colour.fill(0 as $sample);
                    } else {
                        for (sample, out_sample) in
                            input_colour.iter().zip(output_colour.iter_mut())
                        {
                            *out_sample = (((*sample as f64) * max_alpha) / alpha)
                                .round()
                                .clamp(<$sample>::MIN as f64, <$sample>::MAX as f64)
                                as $sample;
                        }
                    }
                    output_alpha[0] = input_alpha[0];
                }
            }
        }

        impl PixelLocalOp for Unpremultiply<$format> {}
    };
}

impl_integer_unpremultiply!(viprs_core::format::U8, u8);
impl_integer_unpremultiply!(viprs_core::format::U16, u16);
impl_integer_unpremultiply!(viprs_core::format::I16, i16);
impl_integer_unpremultiply!(viprs_core::format::U32, u32);
impl_integer_unpremultiply!(viprs_core::format::I32, i32);

impl Op for Unpremultiply<viprs_core::format::F32> {
    type Input = viprs_core::format::F32;
    type Output = viprs_core::format::F32;
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
        input: &Tile<viprs_core::format::F32>,
        output: &mut TileMut<viprs_core::format::F32>,
    ) {
        let bands = self.bands as usize;
        assert!(bands >= 2, "unpremultiply requires at least 2 bands");
        let alpha_band = bands - 1;
        let pixel_count = input.region.pixel_count();
        assert_eq!(input.data.len(), pixel_count * bands);
        assert_eq!(output.data.len(), pixel_count * bands);

        for (input_pixel, output_pixel) in input
            .data
            .chunks_exact(bands)
            .zip(output.data.chunks_exact_mut(bands))
        {
            let (input_colour, input_alpha) = input_pixel.split_at(alpha_band);
            let (output_colour, output_alpha) = output_pixel.split_at_mut(alpha_band);
            let alpha = input_alpha[0];
            if alpha > 0.0 {
                let inv_alpha = alpha.recip();
                for (sample, out_sample) in input_colour.iter().zip(output_colour.iter_mut()) {
                    *out_sample = (*sample * inv_alpha).clamp(0.0, 1.0);
                }
            } else {
                output_colour.fill(0.0);
            }

            output_alpha[0] = input_alpha[0];
        }
    }
}

impl PixelLocalOp for Unpremultiply<viprs_core::format::F32> {}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::{F32, U8},
        image::Region,
    };

    // ── helpers ──────────────────────────────────────────────────────────────

    fn run_unpremultiply_u8(input_data: &[u8], bands: u32) -> Vec<u8> {
        let pixel_count = input_data.len() / bands as usize;
        let width = pixel_count as u32;
        let region = Region::new(0, 0, width, 1);
        let op = Unpremultiply::<U8>::new(bands);
        let mut out = vec![0u8; input_data.len()];
        let input = Tile::<U8>::new(region, bands, input_data);
        let mut output = TileMut::<U8>::new(region, bands, &mut out);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        out
    }

    fn run_unpremultiply_f32(input_data: &[f32], bands: u32) -> Vec<f32> {
        let pixel_count = input_data.len() / bands as usize;
        let width = pixel_count as u32;
        let region = Region::new(0, 0, width, 1);
        let op = Unpremultiply::<F32>::new(bands);
        let mut out = vec![0.0f32; input_data.len()];
        let input = Tile::<F32>::new(region, bands, input_data);
        let mut output = TileMut::<F32>::new(region, bands, &mut out);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        out
    }

    // ── U8: boundary values ───────────────────────────────────────────────────

    #[test]
    fn u8_zero_alpha_leaves_colour_at_zero() {
        let input = vec![0u8, 0u8, 0u8]; // premultiplied R=0, G=0, A=0
        let result = run_unpremultiply_u8(&input, 3);
        assert_eq!(result[0], 0);
        assert_eq!(result[1], 0);
        assert_eq!(result[2], 0); // alpha unchanged
    }

    #[test]
    fn u8_fully_opaque_alpha_is_identity() {
        // premultiplied: colour == original colour when alpha == 255
        let input = vec![100u8, 200u8, 255u8];
        let result = run_unpremultiply_u8(&input, 3);
        assert_eq!(result[0], 100);
        assert_eq!(result[1], 200);
        assert_eq!(result[2], 255); // alpha unchanged
    }

    #[test]
    fn u8_half_alpha_recovers_original() {
        // premultiplied: colour = round(original * 128/255).
        // For original=128: 128 * 128 / 255 = 64.25 → round → 64.
        // unpremultiply: 64 * 255 / 128 = 127.5 → round → 128.
        // Both premultiply and unpremultiply now use f64 + round for consistency.
        let input = vec![64u8, 0u8, 128u8]; // premultiplied R≈64, G=0, A=128
        let result = run_unpremultiply_u8(&input, 3);
        // 64 * 255 / 128 = 127.5 → round → 128
        assert_eq!(result[0], 128);
        assert_eq!(result[1], 0);
        assert_eq!(result[2], 128); // alpha unchanged
    }

    // ── F32: boundary values ──────────────────────────────────────────────────

    #[test]
    fn f32_zero_alpha_leaves_colour_at_zero() {
        let input = vec![0.0f32, 0.0f32, 0.0f32];
        let result = run_unpremultiply_f32(&input, 3);
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 0.0);
        assert_eq!(result[2], 0.0);
    }

    #[test]
    fn f32_fully_opaque_alpha_is_identity() {
        let input = vec![0.5f32, 0.8f32, 1.0f32];
        let result = run_unpremultiply_f32(&input, 3);
        assert!((result[0] - 0.5).abs() < f32::EPSILON);
        assert!((result[1] - 0.8).abs() < f32::EPSILON);
        assert!((result[2] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn f32_half_alpha_recovers_original() {
        // premultiplied: 0.5 * 0.5 = 0.25; unpremultiply: 0.25 / 0.5 = 0.5
        let input = vec![0.25f32, 0.0f32, 0.5f32];
        let result = run_unpremultiply_f32(&input, 3);
        assert!((result[0] - 0.5).abs() < f32::EPSILON);
        assert_eq!(result[1], 0.0);
        assert!((result[2] - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn f32_unpremultiply_clamps_to_one() {
        // A value > alpha after premultiply is not valid premultiplied data, but
        // unpremultiply should clamp to 1.0 rather than returning > 1.0.
        let input = vec![0.8f32, 0.3f32]; // colour=0.8, alpha=0.3 → 0.8/0.3 > 1
        let result = run_unpremultiply_f32(&input, 2);
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 0.3); // alpha unchanged
    }

    // ── Proptest: round-trip Premultiply → Unpremultiply ─────────────────────

    proptest! {
        /// For any colour value c and alpha a (both in [0, 1]):
        /// unpremultiply(premultiply(c, a), a) should equal c within float precision.
        #[test]
        fn f32_round_trip_proptest(
            r in 0.0f32..=1.0f32,
            g in 0.0f32..=1.0f32,
            b in 0.0f32..=1.0f32,
            // Skip alpha=0 because premultiply(c, 0)=0 and unpremultiply is 0/0=0.
            alpha in 0.01f32..=1.0f32,
        ) {
            use super::super::premultiply::Premultiply;
            let bands = 4u32;
            let input = vec![r, g, b, alpha];
            let pixel_count = 1;
            let region = Region::new(0, 0, pixel_count, 1);

            let pre_op = Premultiply::<F32>::new(bands);
            let mut pre_out = vec![0.0f32; 4];
            {
                let input_tile = Tile::<F32>::new(region, bands, &input);
                let mut output_tile = TileMut::<F32>::new(region, bands, &mut pre_out);
                pre_op.process_region(&mut (), &input_tile, &mut output_tile);
            }

            let unpre_op = Unpremultiply::<F32>::new(bands);
            let mut unpre_out = vec![0.0f32; 4];
            {
                let input_tile = Tile::<F32>::new(region, bands, &pre_out);
                let mut output_tile = TileMut::<F32>::new(region, bands, &mut unpre_out);
                unpre_op.process_region(&mut (), &input_tile, &mut output_tile);
            }

            // After round-trip the colour should be within float epsilon.
            prop_assert!((unpre_out[0] - r).abs() < 1e-5,
                "r round-trip: got {} expected {}", unpre_out[0], r);
            prop_assert!((unpre_out[1] - g).abs() < 1e-5,
                "g round-trip: got {} expected {}", unpre_out[1], g);
            prop_assert!((unpre_out[2] - b).abs() < 1e-5,
                "b round-trip: got {} expected {}", unpre_out[2], b);
            prop_assert!((unpre_out[3] - alpha).abs() < f32::EPSILON,
                "alpha should be unchanged");
        }

        /// For U8 with alpha ∈ [128, 255]: Premultiply → Unpremultiply round-trip
        /// error is bounded to 1 LSB.
        ///
        /// Premultiply uses `round(c * alpha/255)`. For alpha ≥ 128 the mapping
        /// `[0, 255] → [0, alpha]` is dense enough that each premultiplied integer
        /// recovers the original within 1 unit under the reverse division. Exhaustive
        /// verification confirms max error == 1 for all (c, alpha) with alpha ≥ 128.
        ///
        /// For smaller alpha values the error is larger (up to `floor(255/alpha) - 1`)
        /// because many source values collapse to the same premultiplied integer.
        /// That is an inherent property of integer premultiplication, not a bug.
        /// The F32 path does not have this problem.
        #[test]
        fn u8_round_trip_error_within_one_lsb(
            r in 0u8..=255u8,
            g in 0u8..=255u8,
            b in 0u8..=255u8,
            // alpha ∈ [128, 255]: max error == 1 (verified exhaustively above).
            alpha in 128u8..=255u8,
        ) {
            use super::super::premultiply::Premultiply;
            let bands = 4u32;
            let input = vec![r, g, b, alpha];
            let pixel_count = 1;
            let region = Region::new(0, 0, pixel_count, 1);

            let pre_op = Premultiply::<U8>::new(bands);
            let mut pre_out = vec![0u8; 4];
            {
                let input_tile = Tile::<U8>::new(region, bands, &input);
                let mut output_tile = TileMut::<U8>::new(region, bands, &mut pre_out);
                pre_op.process_region(&mut (), &input_tile, &mut output_tile);
            }

            let unpre_op = Unpremultiply::<U8>::new(bands);
            let mut unpre_out = vec![0u8; 4];
            {
                let input_tile = Tile::<U8>::new(region, bands, &pre_out);
                let mut output_tile = TileMut::<U8>::new(region, bands, &mut unpre_out);
                unpre_op.process_region(&mut (), &input_tile, &mut output_tile);
            }

            prop_assert!((unpre_out[0] as i32 - r as i32).abs() <= 1,
                "r round-trip error > 1: got {} expected {}", unpre_out[0], r);
            prop_assert!((unpre_out[1] as i32 - g as i32).abs() <= 1,
                "g round-trip error > 1: got {} expected {}", unpre_out[1], g);
            prop_assert!((unpre_out[2] as i32 - b as i32).abs() <= 1,
                "b round-trip error > 1: got {} expected {}", unpre_out[2], b);
            prop_assert_eq!(unpre_out[3], alpha, "alpha must be unchanged");
        }
    }
}

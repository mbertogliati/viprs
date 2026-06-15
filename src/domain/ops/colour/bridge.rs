//! `ColourConvertBridge` — erases `From`/`To` type parameters from `ColourConvert`
//! so that colour conversions can be stored in `PipelineArena` as `Box<dyn DynOperation>`.
//!
//! # Design
//!
//! `ColourConvert<From, To>` carries two colorspace type parameters that cannot appear
//! in an object-safe trait. `ColourConvertBridge<C, From, To>` erases them at the
//! boundary — the same role `OperationBridge<T: Op>` plays for `Op`.
//!
//! `DynOperation` is reused directly (no separate `DynColourOperation` trait).
//! Colorspace propagation is handled by the new `output_colorspace()` method on
//! `DynOperation`, which has a default of `None`. The bridge overrides it to return
//! `Some(To::ID)`, allowing `PipelineBuilder` to track the current colorspace
//! without a separate trait object hierarchy.

use crate::domain::{
    colorspace::{Colorspace, ColorspaceId},
    colour::ColourConvert,
    error::{BuildError, ViprsError},
    format::{BandFormat, BandFormatId},
    image::{DemandHint, ImageMetadata, Interpretation, Region, Tile, TileMut},
    op::{DynOperation, NodeSpec},
    ops::conversion::cast::CastSample,
    ops::resample::sample_conv::{FromF64, ToF64},
};
use std::any::Any;
use std::marker::PhantomData;

const MAX_COLOUR_BANDS: usize = 4;

const fn interpretation_for_colorspace(colorspace: ColorspaceId) -> Option<Interpretation> {
    match colorspace {
        ColorspaceId::SRgb => Some(Interpretation::Srgb),
        ColorspaceId::Lab => Some(Interpretation::Lab),
        ColorspaceId::Xyz => Some(Interpretation::Xyz),
        ColorspaceId::Yxy => Some(Interpretation::Yxy),
        ColorspaceId::Hsv => Some(Interpretation::Hsv),
        ColorspaceId::Lch => Some(Interpretation::Lch),
        ColorspaceId::Ucs => Some(Interpretation::Cmc),
        ColorspaceId::Cmyk => Some(Interpretation::Cmyk),
        ColorspaceId::Greyscale => Some(Interpretation::BW),
        ColorspaceId::ScRgb => Some(Interpretation::Scrgb),
        ColorspaceId::Rgb16 => Some(Interpretation::Rgb16),
        ColorspaceId::Oklab | ColorspaceId::Oklch | ColorspaceId::Cicp | ColorspaceId::Unknown => {
            None
        }
    }
}

const fn colour_input_band_requirement(colorspace: ColorspaceId) -> Option<&'static str> {
    match colorspace.band_count() {
        Some(1) => Some("at least 1 band"),
        Some(3) => Some("at least 3 bands"),
        Some(4) => Some("at least 4 bands"),
        _ => None,
    }
}

/// Bridges a `ColourConvert<From, To>` implementation to `DynOperation`.
///
/// Type parameters:
/// - `C` — the concrete converter type (e.g., `SRgbToLab`)
/// - `From` — source colorspace marker (e.g., `SRgb`)
/// - `To` — destination colorspace marker (e.g., `Lab`)
///
/// `input_bands` is the current channel count of the image before the conversion.
/// Output bands default to `To::ID.band_count()` when the destination colorspace
/// has a fixed arity (for example sRGB → 3, CMYK → 4).
pub struct ColourConvertBridge<C, From, To>
where
    C: ColourConvert<From, To>,
    From: Colorspace,
    To: Colorspace,
{
    /// Stores the `converter` value for this item.
    pub converter: C,
    /// Input band count associated with this condition.
    pub input_bands: u32,
    /// Stores the `output_bands` value for this item.
    pub output_bands: u32,
    _from: PhantomData<From>,
    _to: PhantomData<To>,
}

impl<C, From, To> ColourConvertBridge<C, From, To>
where
    C: ColourConvert<From, To>,
    From: Colorspace,
    To: Colorspace,
{
    /// Creates a new `ColourConvertBridge`.
    pub fn new(converter: C, input_bands: u32) -> Self {
        let input_core_bands = From::ID.band_count().unwrap_or(input_bands);
        let extra_bands = input_bands.saturating_sub(input_core_bands);
        Self {
            converter,
            input_bands,
            output_bands: To::ID
                .band_count()
                .unwrap_or(input_bands)
                .saturating_add(extra_bands),
            _from: PhantomData,
            _to: PhantomData,
        }
    }
}

impl<C, From, To> DynOperation for ColourConvertBridge<C, From, To>
where
    C: ColourConvert<From, To> + Send + Sync,
    From: Colorspace,
    To: Colorspace,
    C::InputFormat: BandFormat,
    C::OutputFormat: BandFormat,
    <C::InputFormat as BandFormat>::Sample: bytemuck::Pod + CastSample<f32> + Default + ToF64,
    <C::OutputFormat as BandFormat>::Sample: bytemuck::Pod + Default + FromF64,
    f32: CastSample<<C::OutputFormat as BandFormat>::Sample>,
{
    fn input_format(&self) -> BandFormatId {
        C::InputFormat::ID
    }

    fn output_format(&self) -> BandFormatId {
        C::OutputFormat::ID
    }

    fn bands(&self) -> u32 {
        self.output_bands
    }

    fn demand_hint(&self) -> DemandHint {
        self.converter.demand_hint()
    }

    fn required_input_region(&self, output: &Region) -> Region {
        self.converter.required_input_region(output)
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        // Colour conversions are pixel-local: every output pixel depends on exactly
        // the corresponding input pixel. Identity spec is always correct.
        NodeSpec::identity(tile_w, tile_h)
    }

    /// Returns the destination colorspace so that `PipelineBuilder` can update its
    /// tracked `current_colorspace` after inserting this node.
    fn output_colorspace(&self) -> Option<ColorspaceId> {
        Some(To::ID)
    }

    fn transform_metadata(&self, source: &ImageMetadata) -> ImageMetadata {
        let mut metadata = source.clone();
        metadata.interpretation = interpretation_for_colorspace(To::ID);
        metadata.icc_profile = None;
        metadata
    }

    fn validate_build_contract(
        &self,
        input_bands: u32,
        _output_bands: u32,
    ) -> Result<(), BuildError> {
        let expected_input_bands = From::ID.band_count().unwrap_or(input_bands);
        if input_bands >= expected_input_bands {
            Ok(())
        } else {
            Err(BuildError::InvalidColourConversionInput {
                from: From::ID,
                to: To::ID,
                bands: input_bands,
                expected: colour_input_band_requirement(From::ID)
                    .unwrap_or("at least the core colorspace band count"),
            })
        }
    }

    fn validate_region_contract(
        &self,
        _input_region: Region,
        input_bands: u32,
        _output_region: Region,
        output_bands: u32,
    ) -> Result<(), ViprsError> {
        self.validate_build_contract(input_bands, output_bands)
            .map_err(ViprsError::from)
    }

    fn dyn_start(&self) -> Box<dyn Any + Send> {
        Box::new(self.converter.start())
    }

    fn dyn_process_region(
        &self,
        state: &mut dyn Any,
        input: &[u8],
        output: &mut [u8],
        input_region: Region,
        output_region: Region,
    ) {
        // State downcast — a failure means the pipeline was constructed with
        // mismatched operation types. Same invariant as OperationBridge.
        let Some(state) = state.downcast_mut::<C::State>() else {
            return;
        };

        let Ok(input_samples) =
            bytemuck::try_cast_slice::<u8, <C::InputFormat as BandFormat>::Sample>(input)
        else {
            return;
        };
        let Ok(output_samples) =
            bytemuck::try_cast_slice_mut::<u8, <C::OutputFormat as BandFormat>::Sample>(output)
        else {
            return;
        };

        let input_core_bands = From::ID.band_count().unwrap_or(self.input_bands) as usize;
        let output_core_bands = To::ID.band_count().unwrap_or(self.output_bands) as usize;
        let extra_bands = self.input_bands.saturating_sub(input_core_bands as u32) as usize;
        let extra_band_scale = To::ID.max_alpha() / From::ID.max_alpha();

        if extra_bands == 0 {
            let input_tile = Tile::new(input_region, self.input_bands, input_samples);
            let mut output_tile = TileMut::new(output_region, self.output_bands, output_samples);
            self.converter
                .convert_region(state, &input_tile, &mut output_tile);
            return;
        }

        let one_pixel = Region::new(0, 0, 1, 1);
        let input_stride = self.input_bands as usize;
        let output_stride = self.output_bands as usize;

        for pixel_index in 0..input_region.pixel_count() {
            let input_offset = pixel_index * input_stride;
            let output_offset = pixel_index * output_stride;

            let mut input_pixel =
                [<C::InputFormat as BandFormat>::Sample::default(); MAX_COLOUR_BANDS];
            input_pixel[..input_core_bands]
                .copy_from_slice(&input_samples[input_offset..input_offset + input_core_bands]);

            let mut output_pixel =
                [<C::OutputFormat as BandFormat>::Sample::default(); MAX_COLOUR_BANDS];
            let input_tile = Tile::new(
                one_pixel,
                input_core_bands as u32,
                &input_pixel[..input_core_bands],
            );
            let mut output_tile = TileMut::new(
                one_pixel,
                output_core_bands as u32,
                &mut output_pixel[..output_core_bands],
            );
            self.converter
                .convert_region(state, &input_tile, &mut output_tile);

            output_samples[output_offset..output_offset + output_core_bands]
                .copy_from_slice(&output_pixel[..output_core_bands]);

            for extra_index in 0..extra_bands {
                let extra = input_samples[input_offset + input_core_bands + extra_index].to_f64()
                    * extra_band_scale;
                output_samples[output_offset + output_core_bands + extra_index] =
                    <<C::OutputFormat as BandFormat>::Sample as FromF64>::from_f64(extra);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        colorspace::{Cmyk, ColorspaceId, SRgb},
        format::{BandFormatId, U8},
        image::{DemandHint, ImageMetadata, Interpretation, Region},
        ops::colour::{cmyk::RgbToCmykOp, lab_to_srgb::LabToSRgb, srgb_to_lab::SRgbToLab},
        ops::colour::{scrgb_to_srgb::ScRgbToSRgb, srgb_to_scrgb::SRgbToScRgb},
    };
    use proptest::prelude::*;

    #[test]
    fn bridge_reports_correct_formats() {
        let bridge = ColourConvertBridge::new(SRgbToLab, 3u32);
        assert_eq!(bridge.input_format(), BandFormatId::U8);
        assert_eq!(bridge.output_format(), BandFormatId::F32);
        assert_eq!(bridge.bands(), 3);
    }

    #[test]
    fn bridge_reports_output_colorspace() {
        let bridge = ColourConvertBridge::new(SRgbToLab, 3u32);
        assert_eq!(bridge.output_colorspace(), Some(ColorspaceId::Lab));
    }

    #[test]
    fn bridge_updates_metadata_for_destination_colorspace() {
        let bridge = ColourConvertBridge::new(SRgbToLab, 3u32);
        let metadata = ImageMetadata {
            interpretation: Some(Interpretation::Srgb),
            icc_profile: Some(vec![1, 2, 3]),
            ..ImageMetadata::default()
        };

        let output = bridge.transform_metadata(&metadata);

        assert_eq!(output.interpretation, Some(Interpretation::Lab));
        assert_eq!(output.icc_profile, None);
    }

    #[test]
    fn bridge_node_spec_is_identity() {
        let bridge = ColourConvertBridge::new(SRgbToLab, 3u32);
        let spec = bridge.node_spec(64, 64);
        assert_eq!(spec, NodeSpec::identity(64, 64));
    }

    #[test]
    fn bridge_required_input_region_delegates() {
        let bridge = ColourConvertBridge::new(SRgbToLab, 3u32);
        let r = Region::new(0, 0, 10, 10);
        assert_eq!(bridge.required_input_region(&r), r);
    }

    #[test]
    fn bridge_demand_hint_delegates() {
        let bridge = ColourConvertBridge::new(SRgbToLab, 3u32);
        assert_eq!(bridge.demand_hint(), DemandHint::Any);
    }

    #[test]
    fn bridge_uses_destination_band_count_for_cmyk() {
        let bridge =
            ColourConvertBridge::<RgbToCmykOp<U8>, SRgb, Cmyk>::new(RgbToCmykOp::<U8>::new(), 3);
        assert_eq!(bridge.input_bands, 3);
        assert_eq!(bridge.bands(), 4);
    }

    /// Exercises dyn_start and dyn_process_region through the bridge using
    /// a real SRgbToLab converter on a known red pixel.
    #[test]
    fn bridge_dyn_start_and_process_region() {
        let bridge = ColourConvertBridge::new(SRgbToLab, 3u32);
        let region = Region::new(0, 0, 1, 1);

        // Input: sRGB red [255, 0, 0] as bytes
        let input: [u8; 3] = [255, 0, 0];
        // Output: 3 × f32 = 12 bytes
        let mut output = [0u8; 12];

        let mut state = bridge.dyn_start();
        bridge.dyn_process_region(state.as_mut(), &input, &mut output, region, region);

        // Interpret output as f32 Lab values; red → L*≈53, a*≈80, b*≈67
        let lab: [f32; 3] = bytemuck::cast(output);
        let l = lab[0];
        assert!(l > 40.0 && l < 70.0, "L* out of range for red: {l}");
        assert!(lab[1] > 50.0, "a* should be strongly positive: {}", lab[1]);
        assert!(lab[2] > 40.0, "b* should be positive: {}", lab[2]);
    }

    /// Exercises dyn_process_region with a white pixel to cover the gamma encode
    /// branch for values near 1.0.
    #[test]
    fn bridge_dyn_process_region_white() {
        let bridge = ColourConvertBridge::new(SRgbToLab, 3u32);
        let region = Region::new(0, 0, 1, 1);

        let input: [u8; 3] = [255, 255, 255];
        let mut output = [0u8; 12];

        let mut state = bridge.dyn_start();
        bridge.dyn_process_region(state.as_mut(), &input, &mut output, region, region);

        let lab: [f32; 3] = bytemuck::cast(output);
        // White → Lab ≈ (100, 0, 0)
        assert!(lab[0] > 95.0, "L* for white should be ~100: {}", lab[0]);
        assert!(lab[1].abs() < 2.0, "a* for white near zero: {}", lab[1]);
        assert!(lab[2].abs() < 2.0, "b* for white near zero: {}", lab[2]);
    }

    proptest! {
        #[test]
        fn bridge_preserves_extra_alpha_band_through_u8_to_f32(
            red in any::<u8>(),
            green in any::<u8>(),
            blue in any::<u8>(),
            alpha in any::<u8>(),
        ) {
            let bridge = ColourConvertBridge::new(SRgbToLab, 4u32);
            let region = Region::new(0, 0, 1, 1);
            let input = [red, green, blue, alpha];
            let mut output = [0u8; 16];

            let mut state = bridge.dyn_start();
            bridge.dyn_process_region(state.as_mut(), &input, &mut output, region, region);

            let lab_alpha: [f32; 4] = bytemuck::cast(output);
            prop_assert!((lab_alpha[3] - f32::from(alpha)).abs() < 1e-6);
        }
    }

    #[test]
    fn bridge_round_trips_extra_alpha_band_back_to_u8() {
        let region = Region::new(0, 0, 1, 1);
        let input = [32u8, 128, 224, 99];

        let to_lab = ColourConvertBridge::new(SRgbToLab, 4u32);
        let mut lab_bytes = [0u8; 16];
        let mut to_lab_state = to_lab.dyn_start();
        to_lab.dyn_process_region(
            to_lab_state.as_mut(),
            &input,
            &mut lab_bytes,
            region,
            region,
        );

        let to_srgb = ColourConvertBridge::new(LabToSRgb, 4u32);
        let mut output = [0u8; 4];
        let mut to_srgb_state = to_srgb.dyn_start();
        to_srgb.dyn_process_region(
            to_srgb_state.as_mut(),
            &lab_bytes,
            &mut output,
            region,
            region,
        );

        assert_eq!(output[3], input[3]);
    }

    proptest! {
        #[test]
        fn bridge_rescales_extra_alpha_when_max_alpha_changes(
            red in any::<u8>(),
            green in any::<u8>(),
            blue in any::<u8>(),
            alpha in any::<u8>(),
        ) {
            let bridge = ColourConvertBridge::new(SRgbToScRgb, 4u32);
            let region = Region::new(0, 0, 1, 1);
            let input = [red, green, blue, alpha];
            let mut output = [0u8; 16];

            let mut state = bridge.dyn_start();
            bridge.dyn_process_region(state.as_mut(), &input, &mut output, region, region);

            let scrgb_alpha: [f32; 4] = bytemuck::cast(output);
            prop_assert!((scrgb_alpha[3] - f32::from(alpha) / 255.0).abs() < 1e-6);
        }
    }

    #[test]
    fn bridge_scales_scrgb_extra_alpha_back_to_u8() {
        let bridge = ColourConvertBridge::new(ScRgbToSRgb, 4u32);
        let region = Region::new(0, 0, 1, 1);
        let input = [0.1f32, 0.2, 0.3, 0.5];
        let mut output = [0u8; 4];

        let mut state = bridge.dyn_start();
        bridge.dyn_process_region(
            state.as_mut(),
            bytemuck::cast_slice(&input),
            &mut output,
            region,
            region,
        );

        assert_eq!(output[3], 128);
    }
}

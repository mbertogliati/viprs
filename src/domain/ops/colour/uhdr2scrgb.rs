use std::{any::Any, marker::PhantomData};

use bytemuck::{Pod, cast_slice, cast_slice_mut};

use crate::domain::{
    colorspace::ColorspaceId,
    error::ViprsError,
    format::{BandFormat, BandFormatId},
    image::{DemandHint, ImageMetadata, Interpretation, Region, UhdrGainMapMetadata},
    op::{DynOperation, NodeSpec},
    ops::{colour::math::srgb_gamma_decode, conversion::cast::CastSample},
};

/// Applies the `uhdr2scrgb` colour transform to image pixels. Use it when a pipeline needs to
/// move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::colour::uhdr2scrgb::UhdrToScRgb;
///
/// let op = UhdrToScRgb::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug)]
pub struct UhdrToScRgb<F: BandFormat> {
    metadata: UhdrGainMapMetadata,
    gainmap_bands: u32,
    _format: PhantomData<F>,
}

impl<F> UhdrToScRgb<F>
where
    F: BandFormat,
{
    /// Creates a new `UhdrToScRgb`.
    pub fn new(metadata: UhdrGainMapMetadata, gainmap_bands: u32) -> Result<Self, ViprsError> {
        if !matches!(gainmap_bands, 1 | 3) {
            return Err(ViprsError::Codec(format!(
                "uhdr2scrgb: gainmap must have 1 or 3 bands, got {gainmap_bands}"
            )));
        }

        Ok(Self {
            metadata,
            gainmap_bands,
            _format: PhantomData,
        })
    }

    #[inline(always)]
    #[must_use]
    /// Returns or performs metadata.
    pub const fn metadata(&self) -> UhdrGainMapMetadata {
        self.metadata
    }

    #[inline(always)]
    fn apply_gain(
        base: f32,
        gain_signal: f32,
        gamma: f32,
        min_boost: f32,
        max_boost: f32,
        offset_sdr: f32,
        offset_hdr: f32,
    ) -> f32 {
        let gain_signal = if gamma == 1.0 {
            gain_signal
        } else {
            gain_signal.powf(1.0 / gamma)
        };
        let boost = max_boost
            .log2()
            .mul_add(gain_signal, min_boost.log2() * (1.0 - gain_signal));
        (base + offset_sdr).mul_add(boost.exp2(), -offset_hdr)
    }
}

impl<F> DynOperation for UhdrToScRgb<F>
where
    F: BandFormat,
    F::Sample: CastSample<f32> + Pod,
{
    fn input_format(&self) -> BandFormatId {
        F::ID
    }

    fn output_format(&self) -> BandFormatId {
        BandFormatId::F32
    }

    fn bands(&self) -> u32 {
        3
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn is_pixel_local(&self) -> bool {
        true
    }

    fn input_slot_count(&self) -> usize {
        2
    }

    fn input_format_slot(&self, _slot: usize) -> BandFormatId {
        F::ID
    }

    fn input_bands_slot(&self, slot: usize) -> u32 {
        match slot {
            0 => 3,
            1 => self.gainmap_bands,
            _ => 0,
        }
    }

    fn required_input_region_slot(&self, output: &Region, _slot: usize) -> Region {
        *output
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn output_colorspace(&self) -> Option<ColorspaceId> {
        Some(ColorspaceId::ScRgb)
    }

    fn transform_metadata(&self, source: &ImageMetadata) -> ImageMetadata {
        let mut metadata = source.clone();
        metadata.interpretation = Some(Interpretation::Scrgb);
        metadata.icc_profile = None;
        metadata
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec::identity(tile_w, tile_h)
    }

    fn dyn_start(&self) -> Box<dyn Any + Send> {
        Box::new(())
    }

    fn dyn_process_region(
        &self,
        _state: &mut dyn Any,
        _input: &[u8],
        _output: &mut [u8],
        _input_region: Region,
        _output_region: Region,
    ) {
        debug_assert!(
            false,
            "UhdrToScRgb: dyn_process_region called on a multi-input node — use dyn_process_region_multi"
        );
    }

    #[inline]
    fn dyn_process_region_multi(
        &self,
        _state: &mut dyn Any,
        inputs: &[&[u8]],
        output: &mut [u8],
        input_regions: &[Region],
        output_region: Region,
    ) {
        debug_assert_eq!(inputs.len(), 2, "UhdrToScRgb expects exactly 2 inputs");
        debug_assert_eq!(
            input_regions.len(),
            2,
            "UhdrToScRgb expects exactly 2 input regions"
        );
        debug_assert_eq!(
            input_regions[0], output_region,
            "base tile must match output tile"
        );
        debug_assert_eq!(
            input_regions[1], output_region,
            "gainmap tile must match output tile"
        );

        let Some(&base_bytes) = inputs.first() else {
            return;
        };
        let Some(&gainmap_bytes) = inputs.get(1) else {
            return;
        };

        let base = cast_slice::<u8, F::Sample>(base_bytes);
        let gainmap = cast_slice::<u8, F::Sample>(gainmap_bytes);
        let out = cast_slice_mut::<u8, f32>(output);

        for ((base_pixel, output_pixel), gainmap_pixel) in base
            .chunks_exact(3)
            .zip(out.chunks_exact_mut(3))
            .zip(gainmap.chunks_exact(self.gainmap_bands as usize))
        {
            let red = srgb_gamma_decode(base_pixel[0].cast_to());
            let green = srgb_gamma_decode(base_pixel[1].cast_to());
            let blue = srgb_gamma_decode(base_pixel[2].cast_to());

            if self.gainmap_bands == 1 {
                let gain = gainmap_pixel[0].cast_to();
                let boosted = Self::apply_gain(
                    green,
                    gain,
                    self.metadata.gamma[1],
                    self.metadata.min_content_boost[1],
                    self.metadata.max_content_boost[1],
                    self.metadata.offset_sdr[1],
                    self.metadata.offset_hdr[1],
                );
                output_pixel[0] = Self::apply_gain(
                    red,
                    gain,
                    self.metadata.gamma[1],
                    self.metadata.min_content_boost[1],
                    self.metadata.max_content_boost[1],
                    self.metadata.offset_sdr[1],
                    self.metadata.offset_hdr[1],
                );
                output_pixel[1] = boosted;
                output_pixel[2] = Self::apply_gain(
                    blue,
                    gain,
                    self.metadata.gamma[1],
                    self.metadata.min_content_boost[1],
                    self.metadata.max_content_boost[1],
                    self.metadata.offset_sdr[1],
                    self.metadata.offset_hdr[1],
                );
            } else {
                let gain_red = srgb_gamma_decode(gainmap_pixel[0].cast_to());
                let gain_green = srgb_gamma_decode(gainmap_pixel[1].cast_to());
                let gain_blue = srgb_gamma_decode(gainmap_pixel[2].cast_to());
                output_pixel[0] = Self::apply_gain(
                    red,
                    gain_red,
                    self.metadata.gamma[0],
                    self.metadata.min_content_boost[0],
                    self.metadata.max_content_boost[0],
                    self.metadata.offset_sdr[0],
                    self.metadata.offset_hdr[0],
                );
                output_pixel[1] = Self::apply_gain(
                    green,
                    gain_green,
                    self.metadata.gamma[1],
                    self.metadata.min_content_boost[1],
                    self.metadata.max_content_boost[1],
                    self.metadata.offset_sdr[1],
                    self.metadata.offset_hdr[1],
                );
                output_pixel[2] = Self::apply_gain(
                    blue,
                    gain_blue,
                    self.metadata.gamma[2],
                    self.metadata.min_content_boost[2],
                    self.metadata.max_content_boost[2],
                    self.metadata.offset_sdr[2],
                    self.metadata.offset_hdr[2],
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        format::U8,
        image::{ImageMetadata, Interpretation},
    };

    fn one_pixel_region() -> Region {
        Region::new(0, 0, 1, 1)
    }

    #[test]
    fn mono_gainmap_matches_libvips_gain_equation() {
        let op = UhdrToScRgb::<U8>::new(
            UhdrGainMapMetadata {
                gamma: [1.0, 1.0, 1.0],
                min_content_boost: [1.0, 1.0, 1.0],
                max_content_boost: [4.0, 4.0, 4.0],
                offset_hdr: [0.0, 0.0, 0.0],
                offset_sdr: [0.0, 0.0, 0.0],
            },
            1,
        )
        .unwrap();

        let base = [128_u8, 128, 128];
        let gainmap = [255_u8];
        let mut output = [0.0_f32; 3];

        op.dyn_process_region_multi(
            &mut (),
            &[bytemuck::cast_slice(&base), bytemuck::cast_slice(&gainmap)],
            bytemuck::cast_slice_mut(&mut output),
            &[one_pixel_region(), one_pixel_region()],
            one_pixel_region(),
        );

        let expected_base = srgb_gamma_decode(128.0 / 255.0);
        assert!((output[0] - expected_base * 4.0).abs() < 1e-6);
        assert!((output[1] - expected_base * 4.0).abs() < 1e-6);
        assert!((output[2] - expected_base * 4.0).abs() < 1e-6);
    }

    #[test]
    fn rgb_gainmap_uses_per_channel_metadata() {
        let op = UhdrToScRgb::<U8>::new(
            UhdrGainMapMetadata {
                gamma: [1.0, 2.0, 1.0],
                min_content_boost: [1.0, 1.0, 1.0],
                max_content_boost: [2.0, 4.0, 8.0],
                offset_hdr: [0.0, 0.0, 0.0],
                offset_sdr: [0.0, 0.0, 0.0],
            },
            3,
        )
        .unwrap();

        let base = [64_u8, 128, 192];
        let gainmap = [255_u8, 64, 255];
        let mut output = [0.0_f32; 3];

        op.dyn_process_region_multi(
            &mut (),
            &[bytemuck::cast_slice(&base), bytemuck::cast_slice(&gainmap)],
            bytemuck::cast_slice_mut(&mut output),
            &[one_pixel_region(), one_pixel_region()],
            one_pixel_region(),
        );

        let base_red = srgb_gamma_decode(64.0 / 255.0);
        let base_green = srgb_gamma_decode(128.0 / 255.0);
        let base_blue = srgb_gamma_decode(192.0 / 255.0);
        let gain_green = srgb_gamma_decode(64.0 / 255.0).powf(0.5);

        assert!((output[0] - base_red * 2.0).abs() < 1e-6);
        assert!((output[1] - base_green * 4.0_f32.powf(gain_green)).abs() < 1e-6);
        assert!((output[2] - base_blue * 8.0).abs() < 1e-6);
    }

    #[test]
    fn invalid_gainmap_band_count_is_rejected() {
        let err = UhdrToScRgb::<U8>::new(UhdrGainMapMetadata::default(), 2)
            .err()
            .expect("gainmap band count must be validated");
        assert!(err.to_string().contains("uhdr2scrgb"));
    }

    #[test]
    fn reports_scrgb_output_colorspace() {
        let op = UhdrToScRgb::<U8>::new(UhdrGainMapMetadata::default(), 1).unwrap();
        assert_eq!(op.output_colorspace(), Some(ColorspaceId::ScRgb));
    }

    #[test]
    fn metadata_and_shape_accessors_follow_gainmap_configuration() {
        let metadata = UhdrGainMapMetadata {
            gamma: [1.0, 2.0, 3.0],
            min_content_boost: [1.0, 1.5, 2.0],
            max_content_boost: [2.0, 3.0, 4.0],
            offset_hdr: [0.1, 0.2, 0.3],
            offset_sdr: [0.4, 0.5, 0.6],
        };
        let op = UhdrToScRgb::<U8>::new(metadata, 3).unwrap();
        let output = Region::new(2, 3, 4, 5);
        let mut source = ImageMetadata::default();
        source.icc_profile = Some(vec![1, 2, 3]);
        source.interpretation = Some(Interpretation::Srgb);

        assert_eq!(op.metadata(), metadata);
        assert_eq!(op.input_slot_count(), 2);
        assert_eq!(op.input_bands_slot(0), 3);
        assert_eq!(op.input_bands_slot(1), 3);
        assert_eq!(op.input_bands_slot(2), 0);
        assert_eq!(op.required_input_region(&output), output);
        assert_eq!(op.required_input_region_slot(&output, 1), output);
        assert_eq!(op.node_spec(4, 5), NodeSpec::identity(4, 5));
        assert_eq!(
            op.transform_metadata(&source).interpretation,
            Some(Interpretation::Scrgb)
        );
        assert_eq!(op.transform_metadata(&source).icc_profile, None);
    }

    #[test]
    fn mono_gainmap_uses_green_channel_metadata_for_all_outputs() {
        let metadata = UhdrGainMapMetadata {
            gamma: [3.0, 1.0, 5.0],
            min_content_boost: [1.0, 1.0, 1.0],
            max_content_boost: [2.0, 4.0, 8.0],
            offset_hdr: [0.0, 0.0, 0.0],
            offset_sdr: [0.0, 0.0, 0.0],
        };
        let op = UhdrToScRgb::<U8>::new(metadata, 1).unwrap();
        let base = [64_u8, 128, 192];
        let gainmap = [255_u8];
        let mut output = [0.0_f32; 3];

        op.dyn_process_region_multi(
            &mut (),
            &[bytemuck::cast_slice(&base), bytemuck::cast_slice(&gainmap)],
            bytemuck::cast_slice_mut(&mut output),
            &[one_pixel_region(), one_pixel_region()],
            one_pixel_region(),
        );

        let gain = 1.0f32;
        let max_boost = metadata.max_content_boost[1];
        assert!((output[0] - srgb_gamma_decode(64.0 / 255.0) * max_boost.powf(gain)).abs() < 1e-6);
        assert!((output[1] - srgb_gamma_decode(128.0 / 255.0) * max_boost.powf(gain)).abs() < 1e-6);
        assert!((output[2] - srgb_gamma_decode(192.0 / 255.0) * max_boost.powf(gain)).abs() < 1e-6);
    }

    #[test]
    fn dyn_process_region_single_input_path_panics_in_debug_builds() {
        let op = UhdrToScRgb::<U8>::new(UhdrGainMapMetadata::default(), 1).unwrap();
        let input = [0_u8; 3];
        let mut output = [0_u8; 12];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            op.dyn_process_region(
                &mut (),
                &input,
                &mut output,
                one_pixel_region(),
                one_pixel_region(),
            );
        }));

        assert!(result.is_err());
    }

    #[test]
    fn reports_pixel_local_multi_input_contract() {
        let op = UhdrToScRgb::<U8>::new(UhdrGainMapMetadata::default(), 3).unwrap();

        assert_eq!(op.input_format(), BandFormatId::U8);
        assert_eq!(op.input_format_slot(0), BandFormatId::U8);
        assert_eq!(op.input_format_slot(1), BandFormatId::U8);
        assert_eq!(op.output_format(), BandFormatId::F32);
        assert_eq!(op.bands(), 3);
        assert!(op.is_pixel_local());
        assert_eq!(op.demand_hint(), DemandHint::Any);
    }

    #[test]
    fn apply_gain_handles_gamma_and_offsets() {
        let gained = UhdrToScRgb::<U8>::apply_gain(0.25, 0.25, 2.0, 1.0, 4.0, 0.5, 0.25);
        let gain_signal = 0.25f32.powf(0.5);
        let boost = 1.0f32.log2() * (1.0 - gain_signal) + 4.0f32.log2() * gain_signal;
        let expected = ((0.25 + 0.5) * boost.exp2()) - 0.25;
        assert!((gained - expected).abs() < 1e-6);
    }

    #[test]
    fn dyn_process_region_multi_returns_early_for_missing_inputs() {
        let op = UhdrToScRgb::<U8>::new(UhdrGainMapMetadata::default(), 1).unwrap();
        let mut output = [7.0_f32; 3];
        op.dyn_process_region_multi(
            &mut (),
            &[&[], &[]],
            bytemuck::cast_slice_mut(&mut output),
            &[one_pixel_region(), one_pixel_region()],
            one_pixel_region(),
        );
        assert_eq!(output, [7.0, 7.0, 7.0]);
    }
}

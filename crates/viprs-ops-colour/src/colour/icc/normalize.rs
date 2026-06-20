#![allow(missing_docs)]
// REASON: ICC normalization helpers are re-exported for internal crate wiring, not stable end-user API.

use super::{BuildError, DynOperation, IccIntent, Interpretation, ViprsError, lcms_error};
use lcms2::{DisallowCache, Flags, GlobalContext, PixelFormat, Profile, Transform};
use std::{any::Any, sync::Arc};
use viprs_core::{
    format::BandFormatId,
    image::{ImageMetadata, Region},
    op::{DemandHint, NodeSpec},
};

use super::{
    profiles::{open_profile, profile_load},
    transform::selected_intent,
};

fn build_normalize_error(err: &ViprsError) -> BuildError {
    BuildError::SourceHint {
        context: "normalize_to_srgb",
        message: err.to_string(),
    }
}

pub fn srgb_profile_bytes() -> Result<Vec<u8>, ViprsError> {
    profile_load("srgb")
}

fn profile_matches_builtin_srgb(profile_bytes: &[u8]) -> bool {
    let Ok(mut profile) = Profile::new_icc(profile_bytes) else {
        return false;
    };
    let mut srgb = Profile::new_srgb();
    profile.set_default_profile_id();
    srgb.set_default_profile_id();
    profile.profile_id() == srgb.profile_id()
}

#[must_use]
pub fn needs_srgb_normalization(profile: Option<&[u8]>) -> bool {
    profile.is_some_and(|profile| {
        Profile::new_icc(profile).is_ok() && !profile_matches_builtin_srgb(profile)
    })
}

type SharedTransform = Arc<Transform<u8, u8, GlobalContext, DisallowCache>>;

#[derive(Clone, Copy)]
enum NormalizePlan {
    Direct,
    SplitAlpha { input_colour_bands: u32 },
}

struct NormalizeToSrgbOp {
    input_format: BandFormatId,
    input_bands: u32,
    output_bands: u32,
    transform: SharedTransform,
    plan: NormalizePlan,
    srgb_profile: Vec<u8>,
}

#[derive(Default)]
struct NormalizeToSrgbState {
    colour_input: Vec<u8>,
    colour_output: Vec<u8>,
}

impl NormalizeToSrgbState {
    fn ensure_capacity(&mut self, input_len: usize, output_len: usize) {
        if self.colour_input.len() != input_len {
            self.colour_input.resize(input_len, 0);
        }
        if self.colour_output.len() != output_len {
            self.colour_output.resize(output_len, 0);
        }
    }
}

impl NormalizeToSrgbOp {
    const fn sample_bytes(&self) -> usize {
        match self.input_format {
            BandFormatId::U8 => 1,
            BandFormatId::U16 => 2,
            _ => 0,
        }
    }
}

impl DynOperation for NormalizeToSrgbOp {
    fn input_format(&self) -> BandFormatId {
        self.input_format
    }

    fn output_format(&self) -> BandFormatId {
        self.input_format
    }

    fn bands(&self) -> u32 {
        self.output_bands
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec::identity(tile_w, tile_h)
    }

    fn transform_metadata(&self, source: &ImageMetadata) -> ImageMetadata {
        let mut metadata = source.clone();
        metadata.interpretation = Some(Interpretation::Srgb);
        metadata.icc_profile = Some(self.srgb_profile.clone());
        metadata
    }

    fn dyn_start(&self) -> Box<dyn Any + Send> {
        Box::new(NormalizeToSrgbState::default())
    }

    fn validate_build_contract(
        &self,
        input_bands: u32,
        output_bands: u32,
    ) -> Result<(), BuildError> {
        if input_bands == self.input_bands && output_bands == self.output_bands {
            Ok(())
        } else {
            Err(BuildError::FormatMismatch {
                produced: self.input_format,
                expected: self.input_format,
                hint: "normalize_to_srgb band contract mismatch",
            })
        }
    }

    fn validate_region_contract(
        &self,
        input_region: Region,
        input_bands: u32,
        output_region: Region,
        output_bands: u32,
    ) -> Result<(), ViprsError> {
        if input_region != output_region
            || input_bands != self.input_bands
            || output_bands != self.output_bands
        {
            return Err(ViprsError::Scheduler(
                "normalize_to_srgb received an unexpected tile contract".into(),
            ));
        }
        Ok(())
    }

    fn dyn_process_region(
        &self,
        state: &mut dyn Any,
        input: &[u8],
        output: &mut [u8],
        input_region: Region,
        output_region: Region,
    ) {
        if self
            .validate_region_contract(
                input_region,
                self.input_bands,
                output_region,
                self.output_bands,
            )
            .is_err()
        {
            return;
        }

        let Some(state) = state.downcast_mut::<NormalizeToSrgbState>() else {
            return;
        };

        match self.plan {
            NormalizePlan::Direct => {
                self.transform.transform_pixels(input, output);
            }
            NormalizePlan::SplitAlpha { input_colour_bands } => {
                let sample_bytes = self.sample_bytes();
                let pixel_count = input_region.pixel_count();
                let input_pixel_bytes = self.input_bands as usize * sample_bytes;
                let output_colour_bands = self.output_bands.saturating_sub(1);
                let output_pixel_bytes = self.output_bands as usize * sample_bytes;
                let input_colour_bytes = pixel_count * input_colour_bands as usize * sample_bytes;
                let output_colour_bytes = pixel_count * output_colour_bands as usize * sample_bytes;
                let input_colour_bytes_per_pixel = input_colour_bands as usize * sample_bytes;
                let output_colour_bytes_per_pixel = output_colour_bands as usize * sample_bytes;

                state.ensure_capacity(input_colour_bytes, output_colour_bytes);

                for (src_pixel, dst_pixel) in input.chunks_exact(input_pixel_bytes).zip(
                    state
                        .colour_input
                        .chunks_exact_mut(input_colour_bytes_per_pixel),
                ) {
                    dst_pixel.copy_from_slice(&src_pixel[..input_colour_bytes_per_pixel]);
                }

                self.transform
                    .transform_pixels(&state.colour_input, &mut state.colour_output);

                for ((src_pixel, colour_pixel), dst_pixel) in input
                    .chunks_exact(input_pixel_bytes)
                    .zip(
                        state
                            .colour_output
                            .chunks_exact(output_colour_bytes_per_pixel),
                    )
                    .zip(output.chunks_exact_mut(output_pixel_bytes))
                {
                    dst_pixel[..output_colour_bytes_per_pixel].copy_from_slice(colour_pixel);
                    dst_pixel[output_colour_bytes_per_pixel..output_pixel_bytes].copy_from_slice(
                        &src_pixel[input_colour_bytes_per_pixel..input_pixel_bytes],
                    );
                }
            }
        }
    }
}

fn normalize_plan(
    input_bands: u32,
    interpretation: Option<Interpretation>,
) -> Option<NormalizePlan> {
    match input_bands {
        1 | 3 => Some(NormalizePlan::Direct),
        2 => Some(NormalizePlan::SplitAlpha {
            input_colour_bands: 1,
        }),
        4 if interpretation == Some(Interpretation::Cmyk) => Some(NormalizePlan::Direct),
        4 => Some(NormalizePlan::SplitAlpha {
            input_colour_bands: 3,
        }),
        _ => None,
    }
}

// NOTE: ICC normalization picks the concrete operation graph only after reading
// runtime profile metadata, so this boundary must return erased operations.
pub fn build_normalize_to_srgb_op(
    input_format: BandFormatId,
    input_bands: u32,
    interpretation: Option<Interpretation>,
    input_profile: Option<&[u8]>,
) -> Result<Option<Box<dyn DynOperation>>, BuildError> {
    let Some(input_profile) = input_profile else {
        return Ok(None);
    };
    if !needs_srgb_normalization(Some(input_profile)) {
        return Ok(None);
    }
    let Some(plan) = normalize_plan(input_bands, interpretation) else {
        return Ok(None);
    };
    let srgb_profile = srgb_profile_bytes().map_err(|err| build_normalize_error(&err))?;
    let input_profile =
        open_profile(input_profile, "input").map_err(|err| build_normalize_error(&err))?;
    let output_profile =
        open_profile(&srgb_profile, "output").map_err(|err| build_normalize_error(&err))?;
    let intent = selected_intent(IccIntent::Auto, &input_profile, &output_profile)
        .map_err(|err| build_normalize_error(&err))?;
    let flags: Flags<DisallowCache> = Flags::NO_CACHE | Flags::BLACKPOINT_COMPENSATION;
    let input_colour_bands = match plan {
        NormalizePlan::Direct => input_bands,
        NormalizePlan::SplitAlpha { input_colour_bands } => input_colour_bands,
    };
    let output_colour_bands = match plan {
        NormalizePlan::Direct
            if input_bands == 4 && interpretation == Some(Interpretation::Cmyk) =>
        {
            3
        }
        NormalizePlan::Direct if input_bands == 1 => 3,
        NormalizePlan::Direct => input_bands,
        NormalizePlan::SplitAlpha { .. } => 3,
    };
    let output_bands = match plan {
        NormalizePlan::Direct => output_colour_bands,
        NormalizePlan::SplitAlpha { .. } => output_colour_bands + 1,
    };
    let input_pixel_format = super::transform::input_pixel_format_for_layout(
        input_format,
        input_profile.color_space(),
        input_colour_bands,
    )
    .map_err(|err| build_normalize_error(&err))?;
    let output_pixel_format = match (input_format, output_colour_bands) {
        (BandFormatId::U8, 3) => PixelFormat::RGB_8,
        (BandFormatId::U16, 3) => PixelFormat::RGB_16,
        (BandFormatId::U8, 1) => PixelFormat::GRAY_8,
        (BandFormatId::U16, 1) => PixelFormat::GRAY_16,
        (format, _) => {
            return Err(BuildError::UnsupportedFormat {
                op: "normalize_to_srgb",
                format,
            });
        }
    };
    let transform: SharedTransform = Arc::new(
        Transform::<u8, u8, GlobalContext, DisallowCache>::new_flags_context(
            GlobalContext::new(),
            &input_profile,
            input_pixel_format,
            &output_profile,
            output_pixel_format,
            intent,
            flags,
        )
        .map_err(lcms_error)
        .map_err(|err| build_normalize_error(&err))?,
    );

    Ok(Some(Box::new(NormalizeToSrgbOp {
        input_format,
        input_bands,
        output_bands,
        transform,
        plan,
        srgb_profile,
    })))
}

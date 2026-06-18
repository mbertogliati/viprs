//! Shrink On Load adapter module.
//!
//! This module provides concrete codec-related infrastructure used by the
//! adapter layer for loading, saving, or normalizing external image formats.

#[cfg(test)]
use crate::domain::error::ViprsError;

/// Backend API used for shrink-on-load.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
/// The `ShrinkOnLoadBackend` enum models adapter-specific runtime choices in the `codecs` module.
/// It is used to represent configuration or branching decisions in concrete adapter code.
///
/// # Examples
///
/// ```ignore
/// let _ = core::mem::size_of::<viprs::adapters::codecs::shrink_on_load::ShrinkOnLoadBackend>();
/// ```
pub(crate) enum ShrinkOnLoadBackend {
    /// libjpeg-turbo exposes DCT-domain scaling via `scale_num/scale_denom`.
    JpegTurboScaledIdct,
    /// `libwebp-sys 0.9.6` exposes `WebPDecoderConfig.options.use_scaling`.
    WebpDecoderConfigScaling,
    /// Animated WebP uses `WebPDemuxGetFrame` plus per-fragment native scaling
    /// and canvas composition to match libvips semantics.
    WebpDemuxFragmentScaling,
}

impl ShrinkOnLoadBackend {
    #[inline]
    pub(crate) const fn codec_name(self) -> &'static str {
        match self {
            Self::JpegTurboScaledIdct => "jpeg",
            Self::WebpDecoderConfigScaling | Self::WebpDemuxFragmentScaling => "webp",
        }
    }
}

/// Effective shrink-on-load plan for the codec backend APIs used by viprs.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
/// The `ShrinkOnLoadPlan` type provides concrete adapter functionality in the `codecs` module.
/// Use this type when you need the runtime behavior implemented by this adapter.
///
/// # Examples
///
/// ```ignore
/// let _ = core::mem::size_of::<viprs::adapters::codecs::shrink_on_load::ShrinkOnLoadPlan>();
/// ```
pub(crate) struct ShrinkOnLoadPlan {
    factor: u8,
    backend: ShrinkOnLoadBackend,
}

impl ShrinkOnLoadPlan {
    #[inline]
    pub(crate) const fn new(requested_factor: u8, backend: ShrinkOnLoadBackend) -> Self {
        Self {
            factor: normalize_shrink_factor(requested_factor),
            backend,
        }
    }

    #[inline]
    pub(crate) const fn factor(self) -> u8 {
        self.factor
    }

    #[inline]
    pub(crate) const fn backend(self) -> ShrinkOnLoadBackend {
        self.backend
    }
}

/// Return the effective integer shrink factor used by codec fallbacks.
///
/// Unsupported values are treated as "no shrink" per the decoder contract.
#[inline]
pub(crate) const fn normalize_shrink_factor(factor: u8) -> u8 {
    match factor {
        2 | 4 | 8 => factor,
        _ => 1,
    }
}

/// Apply an integer box-average shrink to an interleaved U8 image.
///
/// This is a codec-side fallback used only when the backend API cannot honour
/// `LoadOptions::shrink_factor` natively.
#[cfg(test)]
/// `shrink_u8_pixels_into` exposes adapter behavior needed by the surrounding module.
/// Call it when you need the concrete operation implemented here.
///
/// # Examples
///
/// ```rust
/// let _ = viprs::adapters::codecs::shrink_on_load::shrink_u8_pixels_into;
/// ```
pub(crate) fn shrink_u8_pixels_into(
    pixels: &[u8],
    width: u32,
    height: u32,
    bands: u32,
    shrink_factor: u8,
    codec_name: &str,
    output: &mut [u8],
) -> Result<(u32, u32), ViprsError> {
    let factor = normalize_shrink_factor(shrink_factor);
    if factor == 1 || width == 0 || height == 0 || bands == 0 {
        if output.len() != pixels.len() {
            return Err(ViprsError::Codec(format!(
                "{codec_name}: output buffer length mismatch (got {}, expected {})",
                output.len(),
                pixels.len()
            )));
        }
        output.copy_from_slice(pixels);
        return Ok((width, height));
    }

    let width_usize = width as usize;
    let height_usize = height as usize;
    let bands_usize = bands as usize;
    let expected_len = width_usize
        .checked_mul(height_usize)
        .and_then(|px| px.checked_mul(bands_usize))
        .ok_or_else(|| ViprsError::Codec(format!("{codec_name}: decoded dimensions overflow")))?;

    if pixels.len() != expected_len {
        return Err(ViprsError::Codec(format!(
            "{codec_name}: decoded buffer length mismatch (got {}, expected {expected_len})",
            pixels.len()
        )));
    }

    let factor_usize = usize::from(factor);
    let out_width = (width / u32::from(factor)).max(1);
    let out_height = (height / u32::from(factor)).max(1);
    let out_width_usize = out_width as usize;
    let out_height_usize = out_height as usize;
    let expected_output_len = out_width_usize
        .checked_mul(out_height_usize)
        .and_then(|px| px.checked_mul(bands_usize))
        .ok_or_else(|| ViprsError::Codec(format!("{codec_name}: shrunk dimensions overflow")))?;
    if output.len() != expected_output_len {
        return Err(ViprsError::Codec(format!(
            "{codec_name}: output buffer length mismatch (got {}, expected {expected_output_len})",
            output.len()
        )));
    }

    for out_y in 0..out_height_usize {
        let src_y0 = out_y * factor_usize;
        let src_y1 = ((out_y + 1) * factor_usize).min(height_usize);

        for out_x in 0..out_width_usize {
            let src_x0 = out_x * factor_usize;
            let src_x1 = ((out_x + 1) * factor_usize).min(width_usize);
            let sample_count = ((src_y1 - src_y0) * (src_x1 - src_x0)) as u32;
            let dst_base = (out_y * out_width_usize + out_x) * bands_usize;

            for band in 0..bands_usize {
                let mut sum = 0u32;

                for src_y in src_y0..src_y1 {
                    let row_base = src_y * width_usize * bands_usize;
                    for src_x in src_x0..src_x1 {
                        let src_idx = row_base + src_x * bands_usize + band;
                        sum += u32::from(pixels[src_idx]);
                    }
                }

                output[dst_base + band] = (sum / sample_count) as u8;
            }
        }
    }

    Ok((out_width, out_height))
}

#[cfg(test)]
/// `shrink_u8_pixels` exposes adapter behavior needed by the surrounding module.
/// Call it when you need the concrete operation implemented here.
///
/// # Examples
///
/// ```rust
/// let _ = viprs::adapters::codecs::shrink_on_load::shrink_u8_pixels;
/// ```
pub(crate) fn shrink_u8_pixels(
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    bands: u32,
    shrink_factor: u8,
    codec_name: &str,
) -> Result<(u32, u32, Vec<u8>), ViprsError> {
    let factor = normalize_shrink_factor(shrink_factor);
    if factor == 1 || width == 0 || height == 0 || bands == 0 {
        return Ok((width, height, pixels));
    }

    let out_width = (width / u32::from(factor)).max(1);
    let out_height = (height / u32::from(factor)).max(1);
    let mut output = vec![0u8; out_width as usize * out_height as usize * bands as usize];
    let (width, height) = shrink_u8_pixels_into(
        &pixels,
        width,
        height,
        bands,
        shrink_factor,
        codec_name,
        &mut output,
    )?;
    Ok((width, height, output))
}

#[cfg(test)]
mod tests {
    use super::{ShrinkOnLoadBackend, ShrinkOnLoadPlan, normalize_shrink_factor, shrink_u8_pixels};

    #[test]
    fn shrink_factor_two_box_averages_u8_pixels() {
        let pixels = vec![
            1, 2, 3, 4, //
            5, 6, 7, 8, //
            9, 10, 11, 12, //
            13, 14, 15, 16,
        ];

        let (width, height, shrunk) = shrink_u8_pixels(pixels, 4, 4, 1, 2, "test").unwrap();

        assert_eq!((width, height), (2, 2));
        assert_eq!(shrunk, vec![3, 5, 11, 13]);
    }

    #[test]
    fn shrink_factor_two_truncates_fractional_box_average() {
        let pixels = vec![0, 100, 200, 255];

        let (width, height, shrunk) = shrink_u8_pixels(pixels, 4, 1, 1, 2, "test").unwrap();

        assert_eq!((width, height), (2, 1));
        assert_eq!(shrunk, vec![50, 227]);
    }

    #[test]
    fn unsupported_shrink_factor_is_ignored() {
        let pixels = vec![10, 20, 30, 40];
        let (width, height, shrunk) = shrink_u8_pixels(pixels.clone(), 2, 2, 1, 3, "test").unwrap();

        assert_eq!((width, height), (2, 2));
        assert_eq!(shrunk, pixels);
    }

    #[test]
    fn shrink_plan_preserves_backend_limitation_and_normalized_factor() {
        let plan = ShrinkOnLoadPlan::new(4, ShrinkOnLoadBackend::WebpDemuxFragmentScaling);

        assert_eq!(plan.factor(), 4);
        assert_eq!(
            plan.backend(),
            ShrinkOnLoadBackend::WebpDemuxFragmentScaling
        );
        assert_eq!(plan.backend().codec_name(), "webp");
    }

    #[test]
    fn shrink_plan_ignores_unsupported_factor_without_changing_backend_reason() {
        let plan = ShrinkOnLoadPlan::new(3, ShrinkOnLoadBackend::JpegTurboScaledIdct);

        assert_eq!(normalize_shrink_factor(3), 1);
        assert_eq!(plan.factor(), 1);
        assert_eq!(plan.backend(), ShrinkOnLoadBackend::JpegTurboScaledIdct);
        assert_eq!(plan.backend().codec_name(), "jpeg");
    }
}

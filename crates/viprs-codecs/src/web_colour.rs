//! Web Colour adapter module.
//!
//! This module provides concrete codec-related infrastructure used by the
//! adapter layer for loading, saving, or normalizing external image formats.

use std::borrow::Cow;

use viprs_core::{
    error::ViprsError,
    format::{U8, U16},
    image::{Image, Interpretation},
};

#[cfg(feature = "icc")]
use viprs_ops_colour::colour::{
    IccImage, IccIntent, IccTransformOptions, icc::needs_srgb_normalization,
    icc::srgb_profile_bytes, icc_transform,
};

/// `normalize_web_output_u8` exposes adapter behavior needed by the surrounding module.
/// Call it when you need the concrete operation implemented here.
///
/// # Examples
///
/// ```ignore
/// let _ = viprs::adapters::codecs::web_colour::normalize_web_output_u8;
/// ```
pub fn normalize_web_output_u8(image: &Image<U8>) -> Result<Cow<'_, Image<U8>>, ViprsError> {
    #[cfg(not(feature = "icc"))]
    {
        Ok(Cow::Borrowed(image))
    }

    #[cfg(feature = "icc")]
    {
        enabled::normalize_web_output_u8(image)
    }
}

/// `normalize_web_output_u16` exposes adapter behavior needed by the surrounding module.
/// Call it when you need the concrete operation implemented here.
///
/// # Examples
///
/// ```ignore
/// let _ = viprs::adapters::codecs::web_colour::normalize_web_output_u16;
/// ```
pub fn normalize_web_output_u16(image: &Image<U16>) -> Result<Cow<'_, Image<U16>>, ViprsError> {
    #[cfg(not(feature = "icc"))]
    {
        Ok(Cow::Borrowed(image))
    }

    #[cfg(feature = "icc")]
    {
        enabled::normalize_web_output_u16(image)
    }
}

#[cfg(feature = "icc")]
mod enabled {
    use super::{
        Cow, IccImage, IccIntent, IccTransformOptions, Image, Interpretation, U8, U16, ViprsError,
        icc_transform, needs_srgb_normalization, srgb_profile_bytes,
    };

    fn should_normalize<F>(image: &Image<F>) -> bool
    where
        F: viprs_core::format::BandFormat,
    {
        needs_srgb_normalization(image.metadata().icc_profile.as_deref())
    }

    const fn srgb_options(depth: u8) -> IccTransformOptions<'static> {
        IccTransformOptions {
            input_profile: None,
            intent: IccIntent::Auto,
            black_point_compensation: true,
            depth: Some(depth),
        }
    }

    fn srgb_profile() -> Result<Vec<u8>, ViprsError> {
        srgb_profile_bytes()
    }

    fn transformed_u8(image: IccImage) -> Result<Image<U8>, ViprsError> {
        match image {
            IccImage::U8(image) => Ok(image),
            IccImage::U16(_) | IccImage::F32(_) => Err(ViprsError::Codec(
                "web-output ICC normalization expected U8 output".into(),
            )),
        }
    }

    fn transformed_u16(image: IccImage) -> Result<Image<U16>, ViprsError> {
        match image {
            IccImage::U16(image) => Ok(image),
            IccImage::U8(_) | IccImage::F32(_) => Err(ViprsError::Codec(
                "web-output ICC normalization expected U16 output".into(),
            )),
        }
    }

    fn split_alpha_u8(image: &Image<U8>) -> Result<(Image<U8>, Vec<u8>), ViprsError> {
        let colour_bands = image.bands().checked_sub(1).ok_or_else(|| {
            ViprsError::Codec(
                "web-output ICC normalization requires at least one colour band".into(),
            )
        })?;
        let mut colour = Vec::with_capacity(
            image.pixels().len() / image.bands() as usize * colour_bands as usize,
        );
        let mut alpha = Vec::with_capacity(image.pixels().len() / image.bands() as usize);
        for pixel in image.pixels().chunks_exact(image.bands() as usize) {
            colour.extend_from_slice(&pixel[..colour_bands as usize]);
            alpha.push(pixel[colour_bands as usize]);
        }
        let colour_image = Image::from_buffer(image.width(), image.height(), colour_bands, colour)?
            .with_metadata(image.metadata().clone());
        Ok((colour_image, alpha))
    }

    fn split_alpha_u16(image: &Image<U16>) -> Result<(Image<U16>, Vec<u16>), ViprsError> {
        let colour_bands = image.bands().checked_sub(1).ok_or_else(|| {
            ViprsError::Codec(
                "web-output ICC normalization requires at least one colour band".into(),
            )
        })?;
        let mut colour = Vec::with_capacity(
            image.pixels().len() / image.bands() as usize * colour_bands as usize,
        );
        let mut alpha = Vec::with_capacity(image.pixels().len() / image.bands() as usize);
        for pixel in image.pixels().chunks_exact(image.bands() as usize) {
            colour.extend_from_slice(&pixel[..colour_bands as usize]);
            alpha.push(pixel[colour_bands as usize]);
        }
        let colour_image = Image::from_buffer(image.width(), image.height(), colour_bands, colour)?
            .with_metadata(image.metadata().clone());
        Ok((colour_image, alpha))
    }

    fn normalize_alpha_u8(image: &Image<U8>, srgb: &[u8]) -> Result<Image<U8>, ViprsError> {
        let (colour, alpha) = split_alpha_u8(image)?;
        let colour = transformed_u8(icc_transform(&colour, srgb, &srgb_options(8))?)?;
        join_alpha_u8(&colour, &alpha)
    }

    fn normalize_alpha_u16(image: &Image<U16>, srgb: &[u8]) -> Result<Image<U16>, ViprsError> {
        let (colour, alpha) = split_alpha_u16(image)?;
        let colour = transformed_u16(icc_transform(&colour, srgb, &srgb_options(16))?)?;
        join_alpha_u16(&colour, &alpha)
    }

    fn join_alpha_u8(colour: &Image<U8>, alpha: &[u8]) -> Result<Image<U8>, ViprsError> {
        let pixel_count = colour.width() as usize * colour.height() as usize;
        if alpha.len() != pixel_count {
            return Err(ViprsError::Codec(
                "web-output ICC normalization alpha channel length mismatch".into(),
            ));
        }
        let mut pixels = Vec::with_capacity(pixel_count * (colour.bands() as usize + 1));
        for (pixel, alpha_sample) in colour
            .pixels()
            .chunks_exact(colour.bands() as usize)
            .zip(alpha.iter().copied())
        {
            pixels.extend_from_slice(pixel);
            pixels.push(alpha_sample);
        }
        Ok(
            Image::from_buffer(colour.width(), colour.height(), colour.bands() + 1, pixels)?
                .with_metadata(colour.metadata().clone()),
        )
    }

    fn join_alpha_u16(colour: &Image<U16>, alpha: &[u16]) -> Result<Image<U16>, ViprsError> {
        let pixel_count = colour.width() as usize * colour.height() as usize;
        if alpha.len() != pixel_count {
            return Err(ViprsError::Codec(
                "web-output ICC normalization alpha channel length mismatch".into(),
            ));
        }
        let mut pixels = Vec::with_capacity(pixel_count * (colour.bands() as usize + 1));
        for (pixel, alpha_sample) in colour
            .pixels()
            .chunks_exact(colour.bands() as usize)
            .zip(alpha.iter().copied())
        {
            pixels.extend_from_slice(pixel);
            pixels.push(alpha_sample);
        }
        Ok(
            Image::from_buffer(colour.width(), colour.height(), colour.bands() + 1, pixels)?
                .with_metadata(colour.metadata().clone()),
        )
    }

    /// `normalize_web_output_u8` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::codecs::web_colour::normalize_web_output_u8;
    /// ```
    pub(super) fn normalize_web_output_u8(
        image: &Image<U8>,
    ) -> Result<Cow<'_, Image<U8>>, ViprsError> {
        if !should_normalize(image) {
            return Ok(Cow::Borrowed(image));
        }

        let srgb = srgb_profile()?;
        let normalized = match image.bands() {
            1 | 3 => transformed_u8(icc_transform(image, &srgb, &srgb_options(8))?)?,
            4 if image.metadata().interpretation == Some(Interpretation::Cmyk) => {
                transformed_u8(icc_transform(image, &srgb, &srgb_options(8))?)?
            }
            2 | 4 => normalize_alpha_u8(image, &srgb)?,
            _ => return Ok(Cow::Borrowed(image)),
        };
        Ok(Cow::Owned(normalized))
    }

    /// `normalize_web_output_u16` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::codecs::web_colour::normalize_web_output_u16;
    /// ```
    pub(super) fn normalize_web_output_u16(
        image: &Image<U16>,
    ) -> Result<Cow<'_, Image<U16>>, ViprsError> {
        if !should_normalize(image) {
            return Ok(Cow::Borrowed(image));
        }

        let srgb = srgb_profile()?;
        let normalized = match image.bands() {
            1 | 3 => transformed_u16(icc_transform(image, &srgb, &srgb_options(16))?)?,
            4 if image.metadata().interpretation == Some(Interpretation::Cmyk) => {
                transformed_u16(icc_transform(image, &srgb, &srgb_options(16))?)?
            }
            2 | 4 => normalize_alpha_u16(image, &srgb)?,
            _ => return Ok(Cow::Borrowed(image)),
        };
        Ok(Cow::Owned(normalized))
    }
}

#[cfg(all(test, feature = "icc"))]
#[cfg(all(test, feature = "_integration"))]
mod tests {
    use super::*;
    use viprs_core::image::ImageMetadata;
    use viprs_ops_colour::colour::profile_load;

    #[test]
    fn rgba_gray_profile_normalization_preserves_alpha_and_promotes_to_srgb() {
        let image = Image::<U8>::from_buffer(2, 1, 2, vec![32, 7, 160, 9])
            .unwrap()
            .with_metadata(ImageMetadata {
                icc_profile: Some(profile_load("gray").expect("load gray profile")),
                ..ImageMetadata::default()
            });

        let normalized = normalize_web_output_u8(&image)
            .expect("normalize gray+alpha")
            .into_owned();

        assert_eq!(normalized.bands(), 4);
        assert_eq!(
            normalized.metadata().interpretation,
            Some(Interpretation::Srgb)
        );
        assert_eq!(normalized.pixels()[3], 7);
        assert_eq!(normalized.pixels()[7], 9);
    }
}

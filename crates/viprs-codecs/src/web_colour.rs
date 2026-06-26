//! Web Colour adapter module.
//!
//! This module provides concrete codec-related infrastructure used by the
//! adapter layer for loading, saving, or normalizing external image formats.

use std::borrow::Cow;

use viprs_core::{
  error::ViprsError,
  format::{U8, U16},
  image::{InMemoryImage, Interpretation},
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
/// let _ = viprs_codecs::web_colour::normalize_web_output_u8;
/// ```
pub fn normalize_web_output_u8(image: &InMemoryImage<U8>) -> Result<Cow<'_, InMemoryImage<U8>>, ViprsError> {
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
/// let _ = viprs_codecs::web_colour::normalize_web_output_u16;
/// ```
pub fn normalize_web_output_u16(image: &InMemoryImage<U16>) -> Result<Cow<'_, InMemoryImage<U16>>, ViprsError> {
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
      Cow, IccImage, IccIntent, IccTransformOptions, InMemoryImage, Interpretation, U8, U16, ViprsError,
      icc_transform, needs_srgb_normalization, srgb_profile_bytes,
    };

    fn should_normalize<F>(image: &InMemoryImage<F>) -> bool
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

    fn transformed_u8(image: IccImage) -> Result<InMemoryImage<U8>, ViprsError> {
        match image {
            IccImage::U8(image) => Ok(image),
            IccImage::U16(_) | IccImage::F32(_) => Err(ViprsError::Codec(
                "web-output ICC normalization expected U8 output".into(),
            )),
        }
    }

    fn transformed_u16(image: IccImage) -> Result<InMemoryImage<U16>, ViprsError> {
        match image {
            IccImage::U16(image) => Ok(image),
            IccImage::U8(_) | IccImage::F32(_) => Err(ViprsError::Codec(
                "web-output ICC normalization expected U16 output".into(),
            )),
        }
    }

    fn split_alpha_u8(image: &InMemoryImage<U8>) -> Result<(InMemoryImage<U8>, Vec<u8>), ViprsError> {
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
        let colour_image = InMemoryImage::from_buffer(image.width(), image.height(), colour_bands, colour)?
            .with_metadata(image.metadata().clone());
        Ok((colour_image, alpha))
    }

    fn split_alpha_u16(image: &InMemoryImage<U16>) -> Result<(InMemoryImage<U16>, Vec<u16>), ViprsError> {
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
        let colour_image = InMemoryImage::from_buffer(image.width(), image.height(), colour_bands, colour)?
            .with_metadata(image.metadata().clone());
        Ok((colour_image, alpha))
    }

    fn normalize_alpha_u8(image: &InMemoryImage<U8>, srgb: &[u8]) -> Result<InMemoryImage<U8>, ViprsError> {
        let (colour, alpha) = split_alpha_u8(image)?;
        let colour = transformed_u8(icc_transform(&colour, srgb, &srgb_options(8))?)?;
        join_alpha_u8(&colour, &alpha)
    }

    fn normalize_alpha_u16(image: &InMemoryImage<U16>, srgb: &[u8]) -> Result<InMemoryImage<U16>, ViprsError> {
        let (colour, alpha) = split_alpha_u16(image)?;
        let colour = transformed_u16(icc_transform(&colour, srgb, &srgb_options(16))?)?;
        join_alpha_u16(&colour, &alpha)
    }

    fn join_alpha_u8(colour: &InMemoryImage<U8>, alpha: &[u8]) -> Result<InMemoryImage<U8>, ViprsError> {
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
          InMemoryImage::from_buffer(colour.width(), colour.height(), colour.bands() + 1, pixels)?
                .with_metadata(colour.metadata().clone()),
        )
    }

    fn join_alpha_u16(colour: &InMemoryImage<U16>, alpha: &[u16]) -> Result<InMemoryImage<U16>, ViprsError> {
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
          InMemoryImage::from_buffer(colour.width(), colour.height(), colour.bands() + 1, pixels)?
                .with_metadata(colour.metadata().clone()),
        )
    }

    /// `normalize_web_output_u8` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs_codecs::web_colour::normalize_web_output_u8;
    /// ```
    pub(super) fn normalize_web_output_u8(
      image: &InMemoryImage<U8>,
    ) -> Result<Cow<'_, InMemoryImage<U8>>, ViprsError> {
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
    /// let _ = viprs_codecs::web_colour::normalize_web_output_u16;
    /// ```
    pub(super) fn normalize_web_output_u16(
      image: &InMemoryImage<U16>,
    ) -> Result<Cow<'_, InMemoryImage<U16>>, ViprsError> {
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
        let image = InMemoryImage::<U8>::from_buffer(2, 1, 2, vec![32, 7, 160, 9])
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

    #[test]
    fn normalization_borrows_when_profile_is_absent_or_already_srgb() {
        let plain = InMemoryImage::<U8>::from_buffer(1, 1, 3, vec![1, 2, 3]).unwrap();
        assert!(matches!(
            normalize_web_output_u8(&plain).expect("plain image"),
            Cow::Borrowed(_)
        ));

        let srgb = plain.with_metadata(ImageMetadata {
            icc_profile: Some(profile_load("srgb").expect("load srgb profile")),
            ..ImageMetadata::default()
        });
        assert!(matches!(
            normalize_web_output_u8(&srgb).expect("srgb image"),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn unsupported_band_counts_are_borrowed_even_with_non_srgb_profile() {
        let image = InMemoryImage::<U8>::from_buffer(1, 1, 5, vec![1, 2, 3, 4, 5])
            .unwrap()
            .with_metadata(ImageMetadata {
                icc_profile: Some(profile_load("gray").expect("load gray profile")),
                ..ImageMetadata::default()
            });

        assert!(matches!(
            normalize_web_output_u8(&image).expect("unsupported band count"),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn gray_u16_profile_normalization_preserves_alpha_and_promotes_to_srgb() {
        let image = InMemoryImage::<U16>::from_buffer(2, 1, 2, vec![1024, 17, 49152, 23])
            .unwrap()
            .with_metadata(ImageMetadata {
                icc_profile: Some(profile_load("gray").expect("load gray profile")),
                ..ImageMetadata::default()
            });

        let normalized = normalize_web_output_u16(&image)
            .expect("normalize gray16+alpha")
            .into_owned();

        assert_eq!(normalized.bands(), 4);
        assert_eq!(
            normalized.metadata().interpretation,
            Some(Interpretation::Srgb)
        );
        assert_eq!(normalized.pixels()[3], 17);
        assert_eq!(normalized.pixels()[7], 23);
    }

    #[test]
    fn cmyk_profiles_normalize_without_splitting_alpha() {
        let profile = match profile_load("cmyk") {
            Ok(profile) => profile,
            Err(_) => return,
        };
        let image = InMemoryImage::<U8>::from_buffer(1, 1, 4, vec![0, 0, 0, 0])
            .unwrap()
            .with_metadata(ImageMetadata {
                interpretation: Some(Interpretation::Cmyk),
                icc_profile: Some(profile),
                ..ImageMetadata::default()
            });

        let normalized = normalize_web_output_u8(&image)
            .expect("normalize cmyk")
            .into_owned();

        assert_eq!(normalized.bands(), 3);
        assert_eq!(
            normalized.metadata().interpretation,
            Some(Interpretation::Srgb)
        );
    }
}

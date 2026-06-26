use super::{IccImage, IccTransformOptions, cms_unimplemented};
use viprs_core::{error::ViprsError, format::BandFormat, image::InMemoryImage};

/// Returns or performs icc transform.
pub const fn icc_transform<F: BandFormat>(
  _image: &InMemoryImage<F>,
  _output_profile: &[u8],
  _options: &IccTransformOptions<'_>,
) -> Result<IccImage, ViprsError> {
    Err(cms_unimplemented("icc_transform"))
}

/// Returns or performs icc import.
pub const fn icc_import<F: BandFormat>(
  _image: &InMemoryImage<F>,
  _profile: &[u8],
) -> Result<IccImage, ViprsError> {
    Err(cms_unimplemented("icc_import"))
}

/// Returns or performs icc export.
pub const fn icc_export<F: BandFormat>(
  _image: &InMemoryImage<F>,
  _profile: Option<&[u8]>,
) -> Result<IccImage, ViprsError> {
    Err(cms_unimplemented("icc_export"))
}

/// Returns or performs profile load.
pub const fn profile_load(_name: &str) -> Result<Vec<u8>, ViprsError> {
    Err(cms_unimplemented("profile_load"))
}

#[cfg(test)]
mod tests {
    use super::super::{F32, ICC_DETAILS, U8, U16};
    use super::*;

    fn sample_u8_image() -> Image<U8> {
        Image::<U8>::from_buffer(1, 1, 3, vec![1, 2, 3]).unwrap()
    }

    fn assert_unimplemented(err: ViprsError, feature: &'static str) {
        match err {
            ViprsError::Unimplemented {
                feature: actual,
                details,
            } => {
                assert_eq!(actual, feature);
                assert!(
                    details.contains("littlecms2"),
                    "expected littlecms2 detail, got {details}"
                );
            }
            other => panic!("expected unimplemented error, got {other:?}"),
        }
    }

    #[test]
    fn icc_image_accessors_only_expose_matching_formats() {
        let u8_image = sample_u8_image();
        let u16_image = Image::<U16>::from_buffer(1, 1, 1, vec![9]).unwrap();
        let f32_image = Image::<F32>::from_buffer(1, 1, 1, vec![0.5]).unwrap();

        let u8_variant = IccImage::U8(u8_image);
        assert!(u8_variant.as_u8().is_some());
        assert!(u8_variant.as_u16().is_none());
        assert!(u8_variant.as_f32().is_none());

        let u16_variant = IccImage::U16(u16_image);
        assert!(u16_variant.as_u8().is_none());
        assert!(u16_variant.as_u16().is_some());
        assert!(u16_variant.as_f32().is_none());

        let f32_variant = IccImage::F32(f32_image);
        assert!(f32_variant.as_u8().is_none());
        assert!(f32_variant.as_u16().is_none());
        assert!(f32_variant.as_f32().is_some());
    }

    #[test]
    fn no_feature_entry_points_return_typed_unimplemented_errors() {
        let image = sample_u8_image();
        let options = IccTransformOptions::default();
        for result in [
            icc_transform(&image, &[1, 2, 3], &options),
            icc_import(&image, &[1, 2, 3]),
            icc_export(&image, Some(&[1, 2, 3])),
        ] {
            match result {
                Err(ViprsError::Unimplemented { details, .. }) => assert_eq!(details, ICC_DETAILS),
                other => panic!("expected unimplemented error, got {other:?}"),
            }
        }

        match profile_load("sRGB") {
            Err(ViprsError::Unimplemented {
                feature: "profile_load",
                details,
            }) => assert_eq!(details, ICC_DETAILS),
            other => panic!("expected profile_load to be unimplemented, got {other:?}"),
        }
    }

    #[test]
    fn icc_transform_stub_returns_unimplemented() {
        let image = Image::<U8>::from_buffer(1, 1, 3, vec![0, 0, 0]).expect("valid test image");
        let err = icc_transform(&image, b"output-profile", &IccTransformOptions::default())
            .expect_err("stub must fail");
        assert_unimplemented(err, "icc_transform");
    }

    #[test]
    fn icc_import_stub_returns_unimplemented() {
        let image = Image::<U8>::from_buffer(1, 1, 3, vec![0, 0, 0]).expect("valid test image");
        let err = icc_import(&image, b"input-profile").expect_err("stub must fail");
        assert_unimplemented(err, "icc_import");
    }

    #[test]
    fn icc_export_stub_returns_unimplemented() {
        let image = Image::<U8>::from_buffer(1, 1, 3, vec![0, 0, 0]).expect("valid test image");
        let err = icc_export(&image, Some(b"output-profile")).expect_err("stub must fail");
        assert_unimplemented(err, "icc_export");
    }

    #[test]
    fn profile_load_stub_returns_unimplemented() {
        let err = profile_load("srgb").expect_err("stub must fail");
        assert_unimplemented(err, "profile_load");
    }
}

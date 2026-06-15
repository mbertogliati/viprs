use super::*;
use crate::domain::format::BandFormat;
use crate::domain::image::{ImageMetadata, Interpretation};

fn sample_u8_rgb() -> Image<U8> {
    Image::from_buffer(
        2,
        2,
        3,
        vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 120, 120, 120],
    )
    .expect("valid ICC sample image")
    .with_metadata(ImageMetadata {
        interpretation: Some(Interpretation::Srgb),
        ..ImageMetadata::default()
    })
}

fn sample_u16_rgb() -> Image<U16> {
    Image::from_buffer(
        2,
        2,
        3,
        vec![
            65535u16, 0, 0, // red
            0, 65535, 0, // green
            0, 0, 65535, // blue
            30000, 30000, 30000, // mid-grey
        ],
    )
    .expect("valid U16 sample image")
    .with_metadata(ImageMetadata {
        interpretation: Some(Interpretation::Rgb16),
        ..ImageMetadata::default()
    })
}

fn sample_u8_gray() -> Image<U8> {
    Image::from_buffer(1, 4, 1, vec![0u8, 64, 128, 255])
        .expect("valid gray sample")
        .with_metadata(ImageMetadata {
            interpretation: Some(Interpretation::BW),
            ..ImageMetadata::default()
        })
}

fn sample_u16_gray() -> Image<U16> {
    Image::from_buffer(1, 4, 1, vec![0u16, 16384, 32768, 65535])
        .expect("valid u16 gray sample")
        .with_metadata(ImageMetadata {
            interpretation: Some(Interpretation::Grey16),
            ..ImageMetadata::default()
        })
}

fn sample_u8_cmyk() -> Image<U8> {
    Image::from_buffer(
        1,
        2,
        4,
        vec![
            255, 0, 0, 0, // 100% cyan
            0, 255, 0, 0, // 100% magenta
        ],
    )
    .expect("valid CMYK U8 sample")
}

fn sample_u16_cmyk() -> Image<U16> {
    Image::from_buffer(
        1,
        2,
        4,
        vec![
            65535u16, 0, 0, 0, // 100% cyan
            0, 65535, 0, 0, // 100% magenta
        ],
    )
    .expect("valid CMYK U16 sample")
}

fn cmyk_profile_bytes() -> Vec<u8> {
    use lcms2::{ColorSpaceSignature as Csc, Profile as LcmsProfile};
    LcmsProfile::ink_limiting(Csc::CmykData, 300.0)
        .expect("create ink-limiting CMYK profile")
        .icc()
        .expect("serialize CMYK profile")
}

fn with_icc_profile<F: BandFormat>(image: &Image<F>, profile: &[u8]) -> Image<F>
where
    F::Sample: Clone,
{
    let mut metadata = image.metadata().clone();
    metadata.icc_profile = Some(profile.to_vec());
    image.clone().with_metadata(metadata)
}

#[test]
fn icc_import_converts_to_lab_and_preserves_source_profile() {
    let image = sample_u8_rgb();
    let srgb = profile_load("srgb").expect("load srgb profile");

    let imported = icc_import(&image, &srgb).expect("import to lab pcs");
    let imported = imported.as_f32().expect("expected Lab PCS image");

    assert_eq!(
        imported.metadata().icc_profile.as_deref(),
        Some(srgb.as_slice())
    );
    assert_eq!(
        imported.metadata().interpretation,
        Some(Interpretation::Lab)
    );
}

#[test]
fn icc_export_uses_embedded_device_profile_when_none_given() {
    let image = sample_u8_rgb();
    let srgb = profile_load("srgb").expect("load srgb profile");
    let imported = icc_import(&image, &srgb).expect("import to pcs");
    let imported = imported.as_f32().expect("expected Lab PCS image").clone();

    let exported = icc_export(&imported, None).expect("export keeps embedded profile");
    let exported = exported.as_u8().expect("expected RGB output");

    assert_eq!(
        exported.metadata().icc_profile.as_deref(),
        Some(srgb.as_slice())
    );
}

#[test]
fn srgb_lab_srgb_roundtrip_stays_within_tolerance() {
    let image = sample_u8_rgb();
    let srgb = profile_load("srgb").expect("load srgb profile");

    let lab_image = icc_import(&image, &srgb).expect("import to lab pcs");
    let lab_image = lab_image.as_f32().expect("expected f32 Lab image").clone();
    assert_eq!(
        lab_image.metadata().interpretation,
        Some(Interpretation::Lab)
    );
    assert_eq!(
        lab_image.metadata().icc_profile.as_deref(),
        Some(srgb.as_slice())
    );

    let exported = icc_export(&lab_image, Some(&srgb)).expect("export srgb profile");
    let exported = exported.as_u8().expect("expected u8 sRGB image").clone();

    for (expected, actual) in image.pixels().iter().zip(exported.pixels()) {
        let delta = i16::from(*expected) - i16::from(*actual);
        assert!(
            delta.abs() <= 2,
            "expected {expected}, got {actual}, delta {delta}"
        );
    }
    assert_eq!(
        exported.metadata().icc_profile.as_deref(),
        Some(srgb.as_slice())
    );
    assert_eq!(
        exported.metadata().interpretation,
        Some(Interpretation::Srgb)
    );
}

#[test]
fn profile_load_srgb_returns_valid_icc_blob() {
    let bytes = profile_load("srgb").expect("profile_load(\"srgb\") must succeed");
    assert!(!bytes.is_empty(), "srgb profile must be non-empty");
    lcms2::Profile::new_icc(&bytes).expect("srgb bytes must be parseable by lcms2");
}

#[test]
fn profile_load_lab_returns_valid_icc_blob() {
    let bytes = profile_load("lab").expect("profile_load(\"lab\") must succeed");
    assert!(!bytes.is_empty());
    lcms2::Profile::new_icc(&bytes).expect("lab bytes must be parseable by lcms2");
}

#[test]
fn profile_load_xyz_returns_valid_icc_blob() {
    let bytes = profile_load("xyz").expect("profile_load(\"xyz\") must succeed");
    assert!(!bytes.is_empty());
    lcms2::Profile::new_icc(&bytes).expect("xyz bytes must be parseable by lcms2");
}

#[test]
fn profile_load_adobergb_returns_valid_icc_blob() {
    let bytes = profile_load("adobergb").expect("profile_load(\"adobergb\") must succeed");
    assert!(!bytes.is_empty(), "adobergb profile must be non-empty");
    lcms2::Profile::new_icc(&bytes).expect("adobergb bytes must be parseable by lcms2");
}

#[test]
fn profile_load_prophoto_returns_valid_icc_blob() {
    let bytes = profile_load("prophoto").expect("profile_load(\"prophoto\") must succeed");
    assert!(!bytes.is_empty(), "prophoto profile must be non-empty");
    lcms2::Profile::new_icc(&bytes).expect("prophoto bytes must be parseable by lcms2");
}

#[test]
fn profile_load_gray_aliases_all_work() {
    for alias in &["sgrey", "gray", "grey"] {
        let bytes =
            profile_load(alias).unwrap_or_else(|e| panic!("profile_load({alias:?}) failed: {e}"));
        assert!(!bytes.is_empty(), "{alias} profile must be non-empty");
        lcms2::Profile::new_icc(&bytes).unwrap_or_else(|e| panic!("{alias} bytes invalid: {e}"));
    }
}

#[test]
fn profile_load_case_insensitive() {
    let lower = profile_load("srgb").expect("srgb lower");
    let upper = profile_load("sRGB").expect("sRGB mixed");
    let all_upper = profile_load("SRGB").expect("SRGB upper");
    for (label, bytes) in [
        ("lower", &lower),
        ("upper", &upper),
        ("all_upper", &all_upper),
    ] {
        assert!(!bytes.is_empty(), "{label} must be non-empty");
        lcms2::Profile::new_icc(bytes)
            .unwrap_or_else(|e| panic!("{label} not a valid lcms2 profile: {e}"));
    }
}

#[test]
fn profile_load_none_returns_error() {
    let err = profile_load("none").expect_err("\"none\" must not return bytes");
    match &err {
        ViprsError::Codec(msg) => assert!(
            msg.contains("sentinel") || msg.contains("strip"),
            "unexpected error message: {msg}"
        ),
        other => panic!("expected Codec error, got {other:?}"),
    }
}

#[test]
fn profile_load_unknown_alias_returns_codec_error() {
    let err = profile_load("__no_such_profile__").expect_err("unknown alias must fail");
    assert!(matches!(err, ViprsError::Codec(_)));
}

#[test]
fn profile_load_explicit_path_roundtrip() {
    use std::io::Write as _;
    let srgb_bytes = profile_load("srgb").expect("srgb profile bytes");
    let tmp = std::env::current_dir()
        .expect("current working directory")
        .join("target/test-artifacts")
        .join(format!("viprs_b234_profile_{}.icc", std::process::id()));
    if let Some(parent) = tmp.parent() {
        std::fs::create_dir_all(parent).expect("create test artifacts directory");
    }
    {
        let mut f = std::fs::File::create(&tmp).expect("create temp profile file");
        f.write_all(&srgb_bytes).expect("write profile bytes");
    }
    let loaded =
        profile_load(tmp.to_str().unwrap()).expect("loading by explicit path must succeed");
    assert_eq!(loaded, srgb_bytes, "loaded bytes must match written bytes");
    std::fs::remove_file(&tmp).ok();
}

#[test]
fn profile_load_nonexistent_path_returns_error() {
    let err = profile_load("/definitely/nonexistent/profile-b124.icc")
        .expect_err("nonexistent path must fail");
    assert!(matches!(err, ViprsError::Codec(_) | ViprsError::Io(_)));
}

#[test]
fn profile_load_absolute_nonexistent_path_returns_error() {
    let err = profile_load("/nonexistent/absolute/path/profile.icc")
        .expect_err("absolute nonexistent path must fail");
    assert!(matches!(err, ViprsError::Codec(_) | ViprsError::Io(_)));
}

#[test]
fn profile_load_cmyk_returns_valid_profile_or_explains_absence() {
    let result = profile_load("cmyk");
    assert!(matches!(&result, Ok(_) | Err(ViprsError::Codec(_))));
    if let Ok(bytes) = result {
        assert!(!bytes.is_empty(), "cmyk profile must be non-empty");
        lcms2::Profile::new_icc(&bytes).expect("cmyk must be a valid ICC profile");
    }
}

#[test]
fn profile_load_p3_returns_valid_profile_or_explains_absence() {
    let result = profile_load("p3");
    assert!(matches!(&result, Ok(_) | Err(ViprsError::Codec(_))));
    if let Ok(bytes) = result {
        assert!(!bytes.is_empty(), "p3 profile must be non-empty");
        lcms2::Profile::new_icc(&bytes).expect("p3 must be a valid ICC profile");
    }
}

#[test]
fn adobergb_and_prophoto_aliases_support_transform_operations() {
    let image = sample_u8_rgb();
    let adobe = profile_load("adobergb").expect("load adobe rgb profile");
    let prophoto = profile_load("prophoto").expect("load prophoto profile");
    let imported = with_icc_profile(&image, &adobe);

    let converted = icc_transform(&imported, &prophoto, &IccTransformOptions::default())
        .expect("adobergb to prophoto transform");
    let converted = converted.as_u8().expect("expected u8 prophoto image");
    assert_eq!(
        converted.metadata().icc_profile.as_deref(),
        Some(prophoto.as_slice())
    );
    assert_eq!(
        converted.metadata().interpretation,
        Some(Interpretation::Srgb)
    );
    assert_ne!(converted.pixels(), image.pixels());
}

#[test]
fn u16_rgb_to_u8_rgb_transform_produces_correct_variant() {
    let image = sample_u16_rgb();
    let srgb = profile_load("srgb").expect("load srgb");
    let imported = with_icc_profile(&image, &srgb);
    let result = icc_transform(
        &imported,
        &srgb,
        &IccTransformOptions {
            depth: Some(8),
            ..IccTransformOptions::default()
        },
    )
    .expect("u16→u8 identity transform");
    assert!(
        result.as_u8().is_some(),
        "expected IccImage::U8, got {:?}",
        result
    );
}

#[test]
fn u16_rgb_to_u16_rgb_roundtrip_stays_within_tolerance() {
    let image = sample_u16_rgb();
    let srgb = profile_load("srgb").expect("load srgb");
    let imported = with_icc_profile(&image, &srgb);

    let result = icc_transform(
        &imported,
        &srgb,
        &IccTransformOptions {
            depth: Some(16),
            ..IccTransformOptions::default()
        },
    )
    .expect("u16→u16 identity");
    let out = result.as_u16().expect("expected IccImage::U16");
    assert_eq!(out.metadata().interpretation, Some(Interpretation::Srgb));
    for (orig, xformed) in image.pixels().iter().zip(out.pixels()) {
        let delta = i32::from(*orig) - i32::from(*xformed);
        assert!(
            delta.abs() <= 1,
            "pixel mismatch: orig {orig}, xformed {xformed}"
        );
    }
}

#[test]
fn u16_rgb_to_lab_f32_produces_f32_variant() {
    let image = sample_u16_rgb();
    let srgb = profile_load("srgb").expect("load srgb");
    let lab = profile_load("lab").expect("load lab");
    let imported = with_icc_profile(&image, &srgb);

    let result = icc_transform(&imported, &lab, &IccTransformOptions::default()).expect("u16→lab");
    assert!(
        result.as_f32().is_some(),
        "expected IccImage::F32, got {:?}",
        result
    );
}

#[test]
fn gray_u8_to_gray_u8_roundtrip() {
    let image = sample_u8_gray();
    let gray = profile_load("gray").expect("load gray profile");
    let imported = with_icc_profile(&image, &gray);

    let result = icc_transform(
        &imported,
        &gray,
        &IccTransformOptions {
            depth: Some(8),
            ..IccTransformOptions::default()
        },
    )
    .expect("gray u8→u8 identity");
    let out = result.as_u8().expect("expected IccImage::U8");
    assert_eq!(out.metadata().interpretation, Some(Interpretation::BW));
    assert_eq!(out.bands(), 1);
}

#[test]
fn gray_u8_to_gray_u16_produces_u16_variant() {
    let image = sample_u8_gray();
    let gray = profile_load("gray").expect("load gray profile");
    let imported = with_icc_profile(&image, &gray);

    let result = icc_transform(
        &imported,
        &gray,
        &IccTransformOptions {
            depth: Some(16),
            ..IccTransformOptions::default()
        },
    )
    .expect("gray u8→u16");
    let out = result.as_u16().expect("expected IccImage::U16");
    assert_eq!(out.metadata().interpretation, Some(Interpretation::Grey16));
    assert_eq!(out.bands(), 1);
}

#[test]
fn gray_u16_to_gray_u8_produces_u8_variant() {
    let image = sample_u16_gray();
    let gray = profile_load("gray").expect("load gray profile");
    let imported = with_icc_profile(&image, &gray);

    let result = icc_transform(
        &imported,
        &gray,
        &IccTransformOptions {
            depth: Some(8),
            ..IccTransformOptions::default()
        },
    )
    .expect("gray u16→u8");
    let out = result.as_u8().expect("expected IccImage::U8");
    assert_eq!(out.bands(), 1);
    assert_eq!(out.pixels()[0], 0u8);
    assert_eq!(out.pixels()[3], 255u8);
}

#[test]
fn gray_u16_to_gray_u16_roundtrip() {
    let image = sample_u16_gray();
    let gray = profile_load("gray").expect("load gray profile");
    let imported = with_icc_profile(&image, &gray);

    let result = icc_transform(
        &imported,
        &gray,
        &IccTransformOptions {
            depth: Some(16),
            ..IccTransformOptions::default()
        },
    )
    .expect("gray u16→u16 identity");
    let out = result.as_u16().expect("expected IccImage::U16");
    assert_eq!(out.metadata().interpretation, Some(Interpretation::Grey16));
    for (orig, xformed) in image.pixels().iter().zip(out.pixels()) {
        let delta = i32::from(*orig) - i32::from(*xformed);
        assert!(
            delta.abs() <= 1,
            "pixel mismatch: orig {orig}, xformed {xformed}"
        );
    }
}

#[test]
fn cmyk_u8_to_cmyk_u8_identity_produces_u8_variant() {
    let image = sample_u8_cmyk();
    let cmyk = cmyk_profile_bytes();

    let imported = with_icc_profile(&image, &cmyk);
    let result = icc_transform(
        &imported,
        &cmyk,
        &IccTransformOptions {
            depth: Some(8),
            ..IccTransformOptions::default()
        },
    )
    .expect("CMYK u8→u8");
    let out = result.as_u8().expect("expected IccImage::U8");
    assert_eq!(out.metadata().interpretation, Some(Interpretation::Cmyk));
    assert_eq!(out.bands(), 4);
}

#[test]
fn cmyk_u8_to_cmyk_u16_produces_u16_variant() {
    let image = sample_u8_cmyk();
    let cmyk = cmyk_profile_bytes();

    let imported = with_icc_profile(&image, &cmyk);
    let result = icc_transform(
        &imported,
        &cmyk,
        &IccTransformOptions {
            depth: Some(16),
            ..IccTransformOptions::default()
        },
    )
    .expect("CMYK u8→u16");
    let out = result.as_u16().expect("expected IccImage::U16");
    assert_eq!(out.metadata().interpretation, Some(Interpretation::Cmyk));
    assert_eq!(out.bands(), 4);
}

#[test]
fn cmyk_u16_to_cmyk_u8_produces_u8_variant() {
    let image = sample_u16_cmyk();
    let cmyk = cmyk_profile_bytes();

    let imported = with_icc_profile(&image, &cmyk);
    let result = icc_transform(
        &imported,
        &cmyk,
        &IccTransformOptions {
            depth: Some(8),
            ..IccTransformOptions::default()
        },
    )
    .expect("CMYK u16→u8");
    let out = result.as_u8().expect("expected IccImage::U8");
    assert_eq!(out.metadata().interpretation, Some(Interpretation::Cmyk));
    assert_eq!(out.bands(), 4);
}

#[test]
fn cmyk_u16_to_cmyk_u16_roundtrip() {
    let image = sample_u16_cmyk();
    let cmyk = cmyk_profile_bytes();

    let imported = with_icc_profile(&image, &cmyk);
    let result = icc_transform(
        &imported,
        &cmyk,
        &IccTransformOptions {
            depth: Some(16),
            ..IccTransformOptions::default()
        },
    )
    .expect("CMYK u16→u16");
    let out = result.as_u16().expect("expected IccImage::U16");
    assert_eq!(out.metadata().interpretation, Some(Interpretation::Cmyk));
    assert_eq!(out.bands(), 4);
}

#[test]
fn icc_image_accessors_match_variants() {
    let u8_img = sample_u8_rgb();
    let u16_img = sample_u16_rgb();
    let f32_img = Image::<F32>::from_buffer(1, 1, 3, vec![50.0, 0.0, 0.0]).expect("f32 image");

    let u8_variant = IccImage::U8(u8_img);
    assert!(u8_variant.as_u8().is_some());
    assert!(u8_variant.as_u16().is_none());
    assert!(u8_variant.as_f32().is_none());

    let u16_variant = IccImage::U16(u16_img);
    assert!(u16_variant.as_u8().is_none());
    assert!(u16_variant.as_u16().is_some());
    assert!(u16_variant.as_f32().is_none());

    let f32_variant = IccImage::F32(f32_img);
    assert!(f32_variant.as_u8().is_none());
    assert!(f32_variant.as_u16().is_none());
    assert!(f32_variant.as_f32().is_some());
}

#[test]
fn transform_u8_to_lab_and_xyz_returns_f32() {
    let image = sample_u8_rgb();
    let srgb = profile_load("srgb").expect("load srgb");
    let lab = profile_load("lab").expect("load lab");
    let xyz = profile_load("xyz").expect("load xyz");
    let imported = with_icc_profile(&image, &srgb);

    let to_lab = icc_transform(&imported, &lab, &IccTransformOptions::default())
        .expect("u8 to lab transform");
    assert_eq!(
        to_lab
            .as_f32()
            .expect("expected f32")
            .metadata()
            .interpretation,
        Some(Interpretation::Lab)
    );

    let to_xyz = icc_transform(&imported, &xyz, &IccTransformOptions::default())
        .expect("u8 to xyz transform");
    assert_eq!(
        to_xyz
            .as_f32()
            .expect("expected f32")
            .metadata()
            .interpretation,
        Some(Interpretation::Xyz)
    );
}

#[test]
fn transform_f32_to_rgb_lab_xyz_and_integer_depths() {
    let lab_profile = profile_load("lab").expect("load lab");
    let xyz_profile = profile_load("xyz").expect("load xyz");
    let srgb = profile_load("srgb").expect("load srgb");
    let gray = profile_load("gray").expect("load gray");
    let cmyk = cmyk_profile_bytes();

    let lab_image = Image::<F32>::from_buffer(2, 1, 3, vec![50.0, 0.0, 0.0, 80.0, -5.0, 15.0])
        .expect("valid Lab f32 image")
        .with_metadata(ImageMetadata {
            interpretation: Some(Interpretation::Lab),
            ..ImageMetadata::default()
        });

    let xyz_image = Image::<F32>::from_buffer(2, 1, 3, vec![0.20, 0.30, 0.10, 0.40, 0.50, 0.25])
        .expect("valid XYZ f32 image")
        .with_metadata(ImageMetadata {
            interpretation: Some(Interpretation::Xyz),
            ..ImageMetadata::default()
        });

    let rgb_u8 = icc_transform(
        &lab_image,
        &srgb,
        &IccTransformOptions {
            input_profile: Some(&lab_profile),
            ..IccTransformOptions::default()
        },
    )
    .expect("f32 Lab to u8 RGB");
    assert_eq!(
        rgb_u8
            .as_u8()
            .expect("expected u8 output")
            .metadata()
            .interpretation,
        Some(Interpretation::Srgb)
    );

    let rgb_u16 = icc_transform(
        &lab_image,
        &srgb,
        &IccTransformOptions {
            input_profile: Some(&lab_profile),
            depth: Some(16),
            black_point_compensation: true,
            ..IccTransformOptions::default()
        },
    )
    .expect("f32 Lab to u16 RGB");
    assert_eq!(
        rgb_u16
            .as_u16()
            .expect("expected u16 output")
            .metadata()
            .interpretation,
        Some(Interpretation::Srgb)
    );

    let gray_u8 = icc_transform(
        &lab_image,
        &gray,
        &IccTransformOptions {
            input_profile: Some(&lab_profile),
            ..IccTransformOptions::default()
        },
    )
    .expect("f32 Lab to u8 gray");
    assert_eq!(gray_u8.as_u8().expect("expected u8 output").bands(), 1);

    let gray_u16 = icc_transform(
        &lab_image,
        &gray,
        &IccTransformOptions {
            input_profile: Some(&lab_profile),
            depth: Some(16),
            ..IccTransformOptions::default()
        },
    )
    .expect("f32 Lab to u16 gray");
    assert_eq!(gray_u16.as_u16().expect("expected u16 output").bands(), 1);

    let cmyk_u8_err = icc_transform(
        &lab_image,
        &cmyk,
        &IccTransformOptions {
            input_profile: Some(&lab_profile),
            ..IccTransformOptions::default()
        },
    )
    .expect_err("f32 Lab to u8 cmyk should fail with generated CMYK profile");
    assert!(matches!(cmyk_u8_err, ViprsError::Codec(_)));

    let cmyk_u16_err = icc_transform(
        &lab_image,
        &cmyk,
        &IccTransformOptions {
            input_profile: Some(&lab_profile),
            depth: Some(16),
            ..IccTransformOptions::default()
        },
    )
    .expect_err("f32 Lab to u16 cmyk should fail with generated CMYK profile");
    assert!(matches!(cmyk_u16_err, ViprsError::Codec(_)));

    let to_lab = icc_transform(
        &xyz_image,
        &lab_profile,
        &IccTransformOptions {
            input_profile: Some(&xyz_profile),
            ..IccTransformOptions::default()
        },
    )
    .expect("f32 XYZ to f32 Lab");
    assert_eq!(
        to_lab
            .as_f32()
            .expect("expected f32 output")
            .metadata()
            .interpretation,
        Some(Interpretation::Lab)
    );

    let to_xyz = icc_transform(
        &lab_image,
        &xyz_profile,
        &IccTransformOptions {
            input_profile: Some(&lab_profile),
            ..IccTransformOptions::default()
        },
    )
    .expect("f32 Lab to f32 XYZ");
    assert_eq!(
        to_xyz
            .as_f32()
            .expect("expected f32 output")
            .metadata()
            .interpretation,
        Some(Interpretation::Xyz)
    );
}

#[test]
fn fallback_input_profile_supports_u8_u16_and_f32_interpretations() {
    let srgb = profile_load("srgb").expect("load srgb");
    let lab = profile_load("lab").expect("load lab");
    let xyz = profile_load("xyz").expect("load xyz");
    let gray = profile_load("gray").expect("load gray");

    let rgb_u8 = sample_u8_rgb().with_metadata(ImageMetadata::default());
    let rgb_u16 = sample_u16_rgb().with_metadata(ImageMetadata::default());
    let gray_u8 = sample_u8_gray().with_metadata(ImageMetadata::default());
    let lab_f32 = Image::<F32>::from_buffer(1, 1, 3, vec![55.0, 2.0, -1.5])
        .expect("f32 lab image")
        .with_metadata(ImageMetadata {
            interpretation: Some(Interpretation::Lab),
            ..ImageMetadata::default()
        });
    let xyz_f32 = Image::<F32>::from_buffer(1, 1, 3, vec![0.3, 0.4, 0.2])
        .expect("f32 xyz image")
        .with_metadata(ImageMetadata {
            interpretation: Some(Interpretation::Xyz),
            ..ImageMetadata::default()
        });

    assert!(icc_transform(&rgb_u8, &srgb, &IccTransformOptions::default()).is_ok());
    assert!(icc_transform(&rgb_u16, &srgb, &IccTransformOptions::default()).is_ok());
    assert!(icc_transform(&gray_u8, &gray, &IccTransformOptions::default()).is_ok());
    assert!(icc_transform(&lab_f32, &lab, &IccTransformOptions::default()).is_ok());
    assert!(icc_transform(&xyz_f32, &xyz, &IccTransformOptions::default()).is_ok());
}

#[test]
fn fallback_input_profile_rejects_unknown_format_without_profile() {
    use crate::domain::format::I16;

    let image = Image::<I16>::from_buffer(1, 1, 2, vec![1i16, 2]).expect("valid image");
    let srgb = profile_load("srgb").expect("load srgb");
    let err = icc_transform(&image, &srgb, &IccTransformOptions::default())
        .expect_err("missing profile should fail");
    assert!(err.to_string().contains("no embedded ICC profile"));
}

#[test]
fn open_profile_rejects_empty_input_and_invalid_icc_blob() {
    let image = sample_u8_rgb();
    let err_empty = icc_import(&image, b"").expect_err("empty profile must fail");
    assert!(err_empty.to_string().contains("input profile is empty"));

    let err_invalid =
        icc_import(&image, b"not-an-icc-profile").expect_err("invalid profile must fail");
    assert!(matches!(err_invalid, ViprsError::Codec(_)));
}

#[test]
fn transform_rejects_unsupported_depth() {
    let image = sample_u8_rgb();
    let srgb = profile_load("srgb").expect("load srgb");
    let imported = with_icc_profile(&image, &srgb);

    let err = icc_transform(
        &imported,
        &srgb,
        &IccTransformOptions {
            depth: Some(32),
            ..IccTransformOptions::default()
        },
    )
    .expect_err("depth 32 must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("32"),
        "error should mention the rejected depth: {msg}"
    );
}

#[test]
fn transform_rejects_rgb_input_with_wrong_band_count() {
    let image = Image::<U8>::from_buffer(1, 1, 2, vec![128u8, 64]).expect("valid 2-band image");
    let srgb = profile_load("srgb").expect("load srgb");
    let imported = with_icc_profile(&image, &srgb);

    let err = icc_transform(&imported, &srgb, &IccTransformOptions::default())
        .expect_err("mismatched band count must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("unsupported input format"),
        "error should mention format mismatch: {msg}"
    );
}

#[test]
fn transform_rejects_empty_output_profile() {
    let image = sample_u8_rgb();
    let err = icc_transform(&image, b"", &IccTransformOptions::default())
        .expect_err("empty profile must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("output profile is empty"),
        "error should mention empty profile: {msg}"
    );
}

#[test]
fn transform_rejects_garbage_output_profile() {
    let image = sample_u8_rgb();
    let srgb = profile_load("srgb").expect("load srgb");
    let imported = with_icc_profile(&image, &srgb);

    let err = icc_transform(
        &imported,
        b"not-an-icc-profile",
        &IccTransformOptions::default(),
    )
    .expect_err("garbage profile must fail");
    assert!(
        matches!(err, ViprsError::Codec(_)),
        "expected Codec error, got {err:?}"
    );
}

#[cfg(target_pointer_width = "32")]
#[test]
fn integer_icc_transform_rejects_output_pixel_count_overflow() {
    use lcms2::{Flags, Intent, PixelFormat};

    let srgb = profile_load("srgb").expect("load srgb profile");
    let input_profile = super::profiles::open_profile(&srgb, "input").expect("open input profile");
    let output_profile =
        super::profiles::open_profile(&srgb, "output").expect("open output profile");
    let metadata = crate::domain::image::ImageMetadata {
        interpretation: Some(crate::domain::image::Interpretation::Srgb),
        ..crate::domain::image::ImageMetadata::default()
    };

    let err = super::transform::transform_int_input(
        &[],
        65_536,
        65_536,
        PixelFormat::RGB_8,
        &metadata,
        &input_profile,
        &output_profile,
        &srgb,
        Intent::RelativeColorimetric,
        Flags::NO_CACHE,
        super::transform::OutputSpec::U8Rgb,
    )
    .expect_err("oversized output dimensions must fail");

    match err {
        ViprsError::ImageTooLarge {
            width,
            height,
            bands,
            details,
            ..
        } => {
            assert_eq!(width, 65_536);
            assert_eq!(height, 65_536);
            assert_eq!(bands, 3);
            assert_eq!(
                details,
                "ICC transform output dimensions exceed addressable memory"
            );
        }
        other => panic!("expected ImageTooLarge, got {other:?}"),
    }
}

#[cfg(target_pointer_width = "64")]
#[test]
fn integer_icc_output_sizes_use_full_64_bit_counts() {
    let (pixels, sample_count) = super::transform::checked_icc_output_sizes(65_536, 65_536, 3)
        .expect("64-bit hosts must preserve large ICC output counts");
    assert_eq!(pixels, 4_294_967_296usize);
    assert_eq!(sample_count, 12_884_901_888usize);
}

#[cfg(target_pointer_width = "32")]
#[test]
fn f32_icc_transform_rejects_output_pixel_count_overflow() {
    let err = super::transform::checked_icc_output_pixels(65_536, 65_536, 3)
        .expect_err("oversized f32 output dimensions must fail");

    match err {
        ViprsError::ImageTooLarge {
            width,
            height,
            bands,
            details,
            ..
        } => {
            assert_eq!(width, 65_536);
            assert_eq!(height, 65_536);
            assert_eq!(bands, 3);
            assert_eq!(details, super::transform::ICC_OUTPUT_TOO_LARGE_DETAILS);
        }
        other => panic!("expected ImageTooLarge, got {other:?}"),
    }
}

#[cfg(target_pointer_width = "64")]
#[test]
fn f32_icc_output_pixels_use_full_64_bit_counts() {
    let pixels = super::transform::checked_icc_output_pixels(65_536, 65_536, 3)
        .expect("64-bit hosts must preserve large ICC pixel counts");
    assert_eq!(pixels, 4_294_967_296usize);
}

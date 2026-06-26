use super::*;
use crate::{error::ViprsError, format::U8};

#[test]
fn region_expand_can_produce_negative_coordinates() {
    let r = Region::new(5, 5, 10, 10);
    let expanded = r.expand(10, 0, 0, 10);
    assert_eq!(expanded.x, -5);
    assert_eq!(expanded.y, -5);
    assert_eq!(expanded.width, 20);
    assert_eq!(expanded.height, 20);
}

#[test]
fn region_expand_clamps_coordinates_at_i32_min() {
    let expanded = Region::new(i32::MIN, i32::MIN, 1, 1).expand(1, 0, 0, 1);

    assert_eq!(expanded.x, i32::MIN);
    assert_eq!(expanded.y, i32::MIN);
    assert_eq!(expanded.width, 2);
    assert_eq!(expanded.height, 2);
}

#[test]
fn region_expand_saturates_dimensions_at_u32_max() {
    let expanded = Region::new(0, 0, u32::MAX, u32::MAX).expand(0, 1, 1, 0);

    assert_eq!(expanded.x, 0);
    assert_eq!(expanded.y, 0);
    assert_eq!(expanded.width, u32::MAX);
    assert_eq!(expanded.height, u32::MAX);
}

#[test]
fn region_clip_to_clamps_at_zero() {
    let r = Region::new(-10, -10, 30, 30);
    let clipped = r.clip_to(100, 100);
    assert_eq!(clipped.x, 0);
    assert_eq!(clipped.y, 0);
    assert_eq!(clipped.width, 20);
    assert_eq!(clipped.height, 20);
}

#[test]
fn region_clip_to_clamps_u32_max_width_at_image_bounds() {
    let clipped = Region::new(0, 0, u32::MAX, 1).clip_to(10, 1);

    assert_eq!(clipped, Region::new(0, 0, 10, 1));
}

#[test]
fn region_clip_to_clamps_u32_max_width_from_positive_offset() {
    let clipped = Region::new(5, 0, u32::MAX, 1).clip_to(10, 1);

    assert_eq!(clipped, Region::new(5, 0, 5, 1));
}

#[test]
fn tile_new_with_correct_size() {
    let region = Region::new(0, 0, 4, 4);
    let data = vec![0u8; 16];
    let tile: Tile<U8> = Tile::new(region, 1, &data);
    assert_eq!(tile.data.len(), 16);
}

#[test]
#[should_panic(expected = "tile shape overflow")]
fn tile_new_panics_with_clear_message_on_overflowing_shape() {
    let region = Region::new(0, 0, u32::MAX, u32::MAX);
    let data = [];
    let _: Tile<U8> = Tile::new(region, u32::MAX, &data);
}

#[test]
#[should_panic(expected = "tile shape overflow")]
fn tile_mut_new_panics_with_clear_message_on_overflowing_shape() {
    let region = Region::new(0, 0, u32::MAX, u32::MAX);
    let mut data = [];
    let _: TileMut<U8> = TileMut::new(region, u32::MAX, &mut data);
}

#[test]
fn image_metadata_defaults_to_empty_fields() {
    assert_eq!(ImageMetadata::default().interpretation, None);
    assert_eq!(ImageMetadata::default().orientation, None);
    assert_eq!(ImageMetadata::default().icc_profile, None);
    assert_eq!(ImageMetadata::default().exif, None);
    assert_eq!(ImageMetadata::default().xmp, None);
    assert_eq!(ImageMetadata::default().xres, None);
    assert_eq!(ImageMetadata::default().yres, None);
    assert_eq!(ImageMetadata::default().page_height, None);
    assert_eq!(ImageMetadata::default().n_pages, None);
}

#[test]
fn image_metadata_remove_orientation_scrubs_raw_ifd0_exif() {
    let mut metadata = ImageMetadata {
        orientation: Some(6),
        exif: Some(decode_hex_fixture(include_str!(
            "../../../../tests/fixtures/autorot/exif_ifd0_orientation_6.hex"
        ))),
        ..ImageMetadata::default()
    };

    metadata.remove_orientation();

    assert_eq!(metadata.orientation, None);
    assert_eq!(
        metadata.exif,
        Some(decode_hex_fixture(include_str!(
            "../../../../tests/fixtures/autorot/exif_ifd0_without_orientation.hex"
        )))
    );
}

#[test]
fn image_metadata_remove_orientation_preserves_malformed_exif() {
    let exif = vec![b'E', b'x', b'i', b'f', 0, 0, b'n', b'o', b'p', b'e'];
    let mut metadata = ImageMetadata {
        orientation: Some(8),
        exif: Some(exif.clone()),
        ..ImageMetadata::default()
    };

    metadata.remove_orientation();

    assert_eq!(metadata.orientation, None);
    assert_eq!(metadata.exif, Some(exif));
}

#[test]
fn from_buffer_starts_with_default_metadata() {
    let image = InMemoryImage::<U8>::from_buffer(2, 1, 1, vec![1, 2]).unwrap();
    assert_eq!(image.metadata(), &ImageMetadata::default());
}

#[test]
fn from_buffer_rejects_dimension_overflow_before_length_check() {
    let err = InMemoryImage::<U8>::from_buffer(u32::MAX, u32::MAX, 4, Vec::new())
        .expect_err("oversized dimensions must be rejected before buffer length checks");

    assert!(matches!(
        err,
        ViprsError::ImageTooLarge {
            width: u32::MAX,
            height: u32::MAX,
            bands: 4,
            ..
        }
    ));
}

#[test]
fn with_metadata_replaces_default_metadata() {
    let metadata = ImageMetadata {
        interpretation: Some(Interpretation::Srgb),
        orientation: Some(6),
        icc_profile: Some(vec![1, 2, 3]),
        exif: Some(vec![4, 5, 6]),
        xmp: Some(vec![7, 8, 9]),
        xres: Some(11.0),
        yres: Some(12.0),
        page_height: Some(32),
        n_pages: Some(4),
        animation_loop_count: Some(AnimationLoopCount::Finite(3)),
        ..ImageMetadata::default()
    };

    let image = InMemoryImage::<U8>::from_buffer(1, 1, 1, vec![7])
        .unwrap()
        .with_metadata(metadata.clone());

    assert_eq!(image.metadata(), &metadata);
}

#[test]
fn with_frames_exposes_animation_sequence() {
    let frame0 = InMemoryImage::<U8>::from_buffer(1, 1, 1, vec![1]).unwrap();
    let frame1 = InMemoryImage::<U8>::from_buffer(1, 1, 1, vec![2]).unwrap();
    let animated = frame0
        .clone()
        .with_frames(vec![frame0.clone(), frame1.clone()]);

    let frames = animated
        .frames()
        .expect("animated image must expose frames");
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].pixels(), frame0.pixels());
    assert_eq!(frames[1].pixels(), frame1.pixels());
}

#[test]
fn from_frames_preserves_animation_metadata() {
    let frame0 = AnimationFrame::new(
        InMemoryImage::<U8>::from_buffer(1, 1, 1, vec![1]).unwrap(),
        40,
        FrameDisposal::Keep,
    );
    let frame1 = AnimationFrame::new(
        InMemoryImage::<U8>::from_buffer(1, 1, 1, vec![2]).unwrap(),
        70,
        FrameDisposal::Background,
    );

    let animated = InMemoryImage::<U8>::from_frames(vec![frame0, frame1])
        .unwrap()
        .with_animation_loop_count(AnimationLoopCount::Infinite);

    assert_eq!(animated.metadata().n_pages, Some(2));
    assert_eq!(animated.metadata().page_height, Some(1));
    assert_eq!(
        animated.metadata().animation_loop_count,
        Some(AnimationLoopCount::Infinite)
    );

    let frames = animated
        .animation_frames()
        .expect("animated image must expose typed animation frames");
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].delay_ms(), 40);
    assert_eq!(frames[0].disposal(), FrameDisposal::Keep);
    assert_eq!(frames[1].delay_ms(), 70);
    assert_eq!(frames[1].disposal(), FrameDisposal::Background);
}

mod metadata_strip_tests {
    use super::*;

    fn sample_metadata() -> ImageMetadata {
        ImageMetadata {
            interpretation: Some(Interpretation::Srgb),
            orientation: Some(6),
            icc_profile: Some(vec![1, 2, 3]),
            exif: Some(vec![4, 5, 6]),
            xmp: Some(vec![7, 8, 9]),
            xres: Some(300.0),
            yres: Some(300.0),
            page_height: None,
            n_pages: None,
            animation_loop_count: None,
            uhdr_gainmap: None,
            extra: std::iter::once(("key".to_string(), "val".to_string())).collect(),
        }
    }

    #[test]
    fn strip_all_removes_private_fields() {
        let meta = sample_metadata().strip_all();
        assert!(meta.icc_profile.is_none());
        assert!(meta.exif.is_none());
        assert!(meta.xmp.is_none());
        assert!(meta.orientation.is_none());
        assert!(meta.extra.is_empty());
        assert_eq!(meta.interpretation, Some(Interpretation::Srgb));
        assert_eq!(meta.xres, Some(300.0));
    }

    #[test]
    fn strip_preserving_icc_keeps_profile() {
        let meta = sample_metadata().strip_preserving_icc();
        assert_eq!(meta.icc_profile, Some(vec![1, 2, 3]));
        assert!(meta.exif.is_none());
        assert!(meta.xmp.is_none());
        assert!(meta.orientation.is_none());
    }

    #[test]
    fn has_methods() {
        let meta = sample_metadata();
        assert!(meta.has_exif());
        assert!(meta.has_icc_profile());
        assert!(meta.has_xmp());

        let stripped = meta.strip_all();
        assert!(!stripped.has_exif());
        assert!(!stripped.has_icc_profile());
        assert!(!stripped.has_xmp());
    }
}

fn decode_hex_fixture(source: &str) -> Vec<u8> {
    source
        .split_ascii_whitespace()
        .map(|byte| u8::from_str_radix(byte, 16).unwrap())
        .collect()
}

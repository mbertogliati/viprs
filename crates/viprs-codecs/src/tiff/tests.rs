use super::decode::count_pages;
use super::pyramid::{downsample_half, ifd_entry_value_pos, patch_first_ifd_offset, tiff_read_u32};
use super::*;
#[cfg(all(test, feature = "_integration"))]
use std::fs;
use std::num::NonZeroU8;
#[cfg(all(test, feature = "_integration"))]
use std::path::Path;
use viprs_core::codec_options::{LoadOptions, SaveOptions, TiffCompression, TiffPredictor};
use viprs_core::format::{F32, U8, U16};
use viprs_core::image::{Image, Region};
#[cfg(feature = "icc")]
use viprs_ops_colour::colour::{IccTransformOptions, icc_transform, profile_load};
#[cfg(all(test, feature = "_integration"))]
use viprs_ports::source::ImageSource;
#[cfg(all(test, feature = "_integration"))]
use viprs_runtime::sources::decoder_source::DecoderSource;

fn encode_two_page_rgb_tiff() -> Vec<u8> {
    let mut output = Vec::new();
    let mut encoder = RawTiffEncoder::new(Cursor::new(&mut output)).unwrap();
    encoder
        .write_image::<tiff_ct::RGB8>(2, 1, &[255u8, 0, 0, 0, 255, 0])
        .unwrap();
    encoder
        .write_image::<tiff_ct::RGB8>(2, 1, &[0u8, 0, 255, 255, 255, 0])
        .unwrap();
    output
}

fn clamped_region_pixels_u8(image: &Image<U8>, region: Region) -> Vec<u8> {
    let bands = image.bands() as usize;
    let mut output = vec![0u8; region.pixel_count() * bands];
    for out_y in 0..region.height {
        let src_y = (region.y + out_y as i32).clamp(0, image.height() as i32 - 1) as usize;
        for out_x in 0..region.width {
            let src_x = (region.x + out_x as i32).clamp(0, image.width() as i32 - 1) as usize;
            let src = (src_y * image.width() as usize + src_x) * bands;
            let dst = (out_y as usize * region.width as usize + out_x as usize) * bands;
            output[dst..dst + bands].copy_from_slice(&image.pixels()[src..src + bands]);
        }
    }
    output
}

#[test]
fn shared_write_buffer_into_inner_returns_written_bytes() {
    let writer = SharedWriteBuffer::default();
    let mut clone = writer.clone();
    clone.write_all(&[1u8, 2, 3, 4]).unwrap();
    drop(clone);

    assert_eq!(writer.into_inner(), vec![1u8, 2, 3, 4]);
}

#[test]
fn tile_decoder_matches_eager_decode_region_for_tiled_input() {
    let pixels: Vec<u8> = (0..8 * 6 * 3).map(|value| (value % 251) as u8).collect();
    let image = Image::<U8>::from_buffer(8, 6, 3, pixels).unwrap();
    let encoded = TiffEncoder::default()
        .encode_with_options(
            &image,
            &SaveOptions::default()
                .with_tile_width(4)
                .with_tile_height(3),
        )
        .unwrap();
    let eager = TiffDecoder
        .decode_with_options::<U8>(&encoded, &LoadOptions::default())
        .unwrap();
    let region = Region::new(-1, 2, 5, 3);
    let mut actual = vec![0u8; region.pixel_count() * eager.bands() as usize];

    TiffDecoder
        .decode_region_into::<U8>(&encoded, &LoadOptions::default(), region, &mut actual)
        .unwrap();

    assert_eq!(actual, clamped_region_pixels_u8(&eager, region));
}

#[cfg(all(test, feature = "_integration"))]
#[test]
fn streaming_path_source_reads_tiff_regions_without_resident_frame() {
    let pixels: Vec<u8> = (0..8 * 6 * 3).map(|value| (value % 251) as u8).collect();
    let image = Image::<U8>::from_buffer(8, 6, 3, pixels).unwrap();
    let encoded = TiffEncoder::default()
        .encode_with_options(
            &image,
            &SaveOptions::default()
                .with_tile_width(4)
                .with_tile_height(3),
        )
        .unwrap();
    let eager = TiffDecoder
        .decode_with_options::<U8>(&encoded, &LoadOptions::default())
        .unwrap();
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("tiff-streaming-region.tiff");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(&path, &encoded).unwrap();

    let source =
        DecoderSource::<_, U8>::streaming_path(TiffDecoder, &path, LoadOptions::default()).unwrap();

    assert!(source.is_streaming());
    assert_eq!(source.resident_decoded_bytes(), 0);

    let lower = Region::new(4, 2, 3, 3);
    let mut lower_output = vec![0u8; lower.pixel_count() * eager.bands() as usize];
    source.read_region(lower, &mut lower_output).unwrap();
    assert_eq!(lower_output, clamped_region_pixels_u8(&eager, lower));

    let edge = Region::new(-1, 4, 4, 3);
    let mut edge_output = vec![0u8; edge.pixel_count() * eager.bands() as usize];
    source.read_region(edge, &mut edge_output).unwrap();
    assert_eq!(edge_output, clamped_region_pixels_u8(&eager, edge));
}

#[test]
fn decode_region_into_returns_image_too_large_for_overflowing_region() {
    let image = Image::<U8>::from_buffer(1, 1, 3, vec![1, 2, 3]).unwrap();
    let encoded = TiffEncoder::default().encode(&image).unwrap();
    let region = Region::new(0, 0, u32::MAX, u32::MAX);
    let mut output = Vec::new();

    let result = TiffDecoder.decode_region_into::<U8>(
        &encoded,
        &LoadOptions::default(),
        region,
        &mut output,
    );

    assert!(matches!(
        result,
        Err(ViprsError::ImageTooLarge {
            width,
            height,
            bands,
            ..
        }) if width == u32::MAX && height == u32::MAX && bands == 3
    ));
}

fn encode_tagged_rgb_tiff() -> Vec<u8> {
    let mut output = Vec::new();
    let mut encoder = RawTiffEncoder::new(Cursor::new(&mut output)).unwrap();
    let mut image = encoder.new_image::<tiff_ct::RGB8>(2, 1).unwrap();
    image.encoder().write_tag(Tag::Orientation, 6u16).unwrap();
    image.resolution_unit(ResolutionUnit::Centimeter);
    image.x_resolution(Rational { n: 120, d: 1 });
    image.y_resolution(Rational { n: 80, d: 1 });
    image.write_data(&[1u8, 2, 3, 4, 5, 6]).unwrap();
    output
}

fn encode_rgb_f32_tiff() -> Vec<u8> {
    let mut output = Vec::new();
    let mut encoder = RawTiffEncoder::new(Cursor::new(&mut output)).unwrap();
    encoder
        .write_image::<tiff_ct::RGB32Float>(1, 2, &[0.0f32, 0.5, 1.0, 0.25, 0.75, 0.125])
        .unwrap();
    output
}

fn sample_rgb_image() -> Image<U8> {
    Image::<U8>::from_buffer(6, 4, 3, (0u8..72).collect()).unwrap()
}

fn solid_gray_image(value: u8) -> Image<U8> {
    Image::<U8>::from_buffer(8, 8, 1, vec![value; 64]).unwrap()
}

fn multi_strip_rgb_image() -> Image<U8> {
    let width = 7u32;
    let height = DEFAULT_TIFF_ROWS_PER_STRIP * 2 + 5;
    let bands = 3u32;
    let pixel_count = (width * height * bands) as usize;
    let pixels = (0..pixel_count)
        .map(|index| u8::try_from(index % 251).unwrap())
        .collect();
    Image::<U8>::from_buffer(width, height, bands, pixels).unwrap()
}

#[test]
fn sniff_recognises_little_endian_tiff() {
    assert!(TiffDecoder.sniff(&[0x49, 0x49, 0x2A, 0x00, 0x08, 0x00]));
}

#[test]
fn sniff_rejects_png() {
    assert!(!TiffDecoder.sniff(&[137, 80, 78, 71, 13, 10, 26, 10]));
}

#[test]
fn round_trip_u8_rgb_is_pixel_exact() {
    let codec = TiffCodec::default();
    let original = Image::<U8>::from_buffer(4, 4, 3, (0u8..48).collect()).unwrap();

    let encoded = codec.encode(&original).unwrap();
    let decoded = codec.decode::<U8>(&encoded).unwrap();

    assert_eq!(decoded.width(), 4);
    assert_eq!(decoded.height(), 4);
    assert_eq!(decoded.bands(), 3);
    assert_eq!(decoded.pixels(), original.pixels());
    assert_eq!(
        decoded.metadata().interpretation,
        Some(Interpretation::Srgb)
    );
    assert_eq!(decoded.metadata().n_pages, Some(1));
}

#[test]
fn round_trip_u16_grayscale_is_pixel_exact() {
    let codec = TiffCodec::default();
    let pixels: Vec<u16> = (0u16..16).map(|value| value * 257).collect();
    let original = Image::<U16>::from_buffer(4, 4, 1, pixels).unwrap();

    let encoded = codec.encode(&original).unwrap();
    let decoded = codec.decode::<U16>(&encoded).unwrap();

    assert_eq!(decoded.width(), 4);
    assert_eq!(decoded.height(), 4);
    assert_eq!(decoded.bands(), 1);
    assert_eq!(decoded.pixels(), original.pixels());
    assert_eq!(
        decoded.metadata().interpretation,
        Some(Interpretation::Grey16)
    );
}

#[cfg(feature = "icc")]
#[test]
fn round_trip_preserves_embedded_icc_profile() {
    let codec = TiffCodec::default();
    let icc_profile = profile_load("srgb").expect("load srgb profile");
    let original = sample_rgb_image().with_metadata(ImageMetadata {
        interpretation: Some(Interpretation::Srgb),
        icc_profile: Some(icc_profile.clone()),
        ..ImageMetadata::default()
    });

    let encoded = codec.encode(&original).unwrap();
    let decoded = codec.decode::<U8>(&encoded).unwrap();

    assert_eq!(
        decoded.metadata().icc_profile.as_deref(),
        Some(icc_profile.as_slice())
    );
}

#[cfg(feature = "icc")]
#[test]
fn decode_embedded_gray_profile_transforms_to_srgb_correctly() {
    let codec = TiffCodec::default();
    let gray_profile = profile_load("gray").expect("load gray profile");
    let srgb_profile = profile_load("srgb").expect("load srgb profile");
    let original = solid_gray_image(128).with_metadata(ImageMetadata {
        interpretation: Some(Interpretation::BW),
        icc_profile: Some(gray_profile.clone()),
        ..ImageMetadata::default()
    });

    let encoded = codec.encode(&original).unwrap();
    let decoded = codec.decode::<U8>(&encoded).unwrap();
    assert_eq!(
        decoded.metadata().icc_profile.as_deref(),
        Some(gray_profile.as_slice())
    );

    let transformed = icc_transform(&decoded, &srgb_profile, &IccTransformOptions::default())
        .unwrap()
        .as_u8()
        .unwrap()
        .clone();
    let expected = icc_transform(
        &solid_gray_image(128),
        &srgb_profile,
        &IccTransformOptions {
            input_profile: Some(&gray_profile),
            ..IccTransformOptions::default()
        },
    )
    .unwrap()
    .as_u8()
    .unwrap()
    .clone();

    assert_eq!(transformed.bands(), 3);
    assert_eq!(transformed.pixels(), expected.pixels());
}

#[test]
fn pyramid_tiff_round_trips_with_subifd_level_selection() {
    let codec = TiffCodec::default();
    let original = Image::<U8>::from_buffer(8, 8, 1, (0u8..64).collect()).unwrap();

    let mut encoded = codec
        .encode_with_options(
            &original,
            &SaveOptions::default()
                .with_pyramid(true)
                .with_tile_width(4)
                .with_tile_height(4),
        )
        .unwrap();

    assert_eq!(count_pages(&encoded).unwrap(), 1);

    let mut decoder = Decoder::new(Cursor::new(encoded.as_slice())).unwrap();
    let subifd_offsets = decoder
        .find_tag_unsigned_vec::<u32>(TIFF_SUB_IFD_TAG)
        .unwrap()
        .unwrap();
    assert_eq!(subifd_offsets.len(), 1);
    let selected_offset = subifd_offsets[0];

    let selected_pixels = [
        251u8, 17, 199, 77, 13, 243, 89, 131, 201, 29, 167, 53, 109, 229, 41, 173,
    ];
    let selected_level_data_value_pos =
        ifd_entry_value_pos(&encoded, selected_offset, Tag::StripOffsets)
            .or_else(|_| ifd_entry_value_pos(&encoded, selected_offset, Tag::TileOffsets))
            .unwrap();
    let selected_level_data_offset =
        usize::try_from(tiff_read_u32(&encoded, selected_level_data_value_pos).unwrap()).unwrap();
    encoded[selected_level_data_offset..selected_level_data_offset + selected_pixels.len()]
        .copy_from_slice(&selected_pixels);

    let patched_subifd = patch_first_ifd_offset(&encoded, selected_offset).unwrap();
    let expected_selected = codec.decode::<U8>(&patched_subifd).unwrap();
    assert_eq!(expected_selected.width(), 4);
    assert_eq!(expected_selected.height(), 4);
    assert_eq!(expected_selected.pixels(), &selected_pixels);

    let resized_root = downsample_half(&original).unwrap().unwrap();
    assert_ne!(resized_root.pixels(), &selected_pixels);

    let reduced = codec
        .decode_with_options::<U8>(
            &encoded,
            &LoadOptions::default().with_shrink(NonZeroU8::new(2).unwrap()),
        )
        .unwrap();

    assert_eq!(reduced.width(), 4);
    assert_eq!(reduced.height(), 4);
    assert_eq!(reduced.pixels(), expected_selected.pixels());
}

#[test]
fn decode_f32_rgb_tiff_preserves_samples() {
    let decoded = TiffDecoder.decode::<F32>(&encode_rgb_f32_tiff()).unwrap();

    assert_eq!(decoded.width(), 1);
    assert_eq!(decoded.height(), 2);
    assert_eq!(decoded.bands(), 3);
    assert_eq!(decoded.pixels(), &[0.0, 0.5, 1.0, 0.25, 0.75, 0.125]);
    assert_eq!(
        decoded.metadata().interpretation,
        Some(Interpretation::Scrgb)
    );
}

#[test]
fn decode_first_page_only_but_records_total_page_count() {
    let decoded = TiffDecoder
        .decode::<U8>(&encode_two_page_rgb_tiff())
        .unwrap();

    assert_eq!(decoded.width(), 2);
    assert_eq!(decoded.height(), 1);
    assert_eq!(decoded.pixels(), &[255u8, 0, 0, 0, 255, 0]);
    assert_eq!(decoded.metadata().n_pages, Some(2));
    assert_eq!(decoded.metadata().page_height, Some(1));
    assert!(decoded.frames().is_none());
}

#[test]
fn decode_page_option_selects_requested_page() {
    let decoded = TiffDecoder
        .decode_with_options::<U8>(
            &encode_two_page_rgb_tiff(),
            &LoadOptions::default().with_page(1),
        )
        .unwrap();

    assert_eq!(decoded.pixels(), &[0u8, 0, 255, 255, 255, 0]);
    assert_eq!(decoded.metadata().n_pages, Some(2));
}

#[test]
fn decode_n_option_stacks_requested_pages() {
    let decoded = TiffDecoder
        .decode_with_options::<U8>(
            &encode_two_page_rgb_tiff(),
            &LoadOptions::default().with_page(0).with_n(2),
        )
        .unwrap();

    assert_eq!(decoded.width(), 2);
    assert_eq!(decoded.height(), 2);
    assert_eq!(decoded.metadata().page_height, Some(1));
    assert_eq!(decoded.metadata().n_pages, Some(2));
    let frames = decoded.frames().unwrap();
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[1].pixels(), &[0u8, 0, 255, 255, 255, 0]);
}

#[test]
fn decode_populates_orientation_and_resolution_metadata() {
    let decoded = TiffDecoder.decode::<U8>(&encode_tagged_rgb_tiff()).unwrap();
    let metadata = decoded.metadata();

    assert_eq!(metadata.orientation, Some(6));
    assert_eq!(metadata.xres, Some(12.0));
    assert_eq!(metadata.yres, Some(8.0));
}

#[test]
fn lzw_round_trip_is_pixel_exact() {
    let codec = TiffEncoder::with_compression(TiffCompression::Lzw);
    let original = Image::<U8>::from_buffer(3, 2, 3, (0u8..18).collect()).unwrap();

    let encoded = codec.encode(&original).unwrap();
    let decoded = TiffDecoder.decode::<U8>(&encoded).unwrap();

    assert_eq!(decoded.pixels(), original.pixels());
}

#[test]
fn lzw_writes_horizontal_predictor_by_default() {
    let original = sample_rgb_image();
    let encoded = TiffEncoder::with_compression(TiffCompression::Lzw)
        .encode(&original)
        .unwrap();
    let mut decoder = Decoder::new(Cursor::new(encoded.as_slice())).unwrap();

    assert_eq!(
        decoder.find_tag_unsigned::<u16>(Tag::Predictor).unwrap(),
        Some(2)
    );
}

#[test]
fn deflate_round_trip_is_pixel_exact() {
    let codec = TiffEncoder::with_compression(TiffCompression::Deflate)
        .with_predictor(TiffPredictor::Horizontal);
    let original =
        Image::<U16>::from_buffer(3, 2, 1, vec![0, 1024, 2048, 4096, 8192, 16384]).unwrap();

    let encoded = codec
        .encode_with_options(&original, &SaveOptions::default().with_compression_level(9))
        .unwrap();
    let decoded = TiffDecoder.decode::<U16>(&encoded).unwrap();

    assert_eq!(decoded.pixels(), original.pixels());
}

#[test]
fn packbits_round_trip_is_pixel_exact() {
    let original = sample_rgb_image();
    let encoded = TiffEncoder::default()
        .encode_with_options(
            &original,
            &SaveOptions::default().with_tiff_compression(TiffCompression::PackBits),
        )
        .unwrap();
    let decoded = TiffDecoder.decode::<U8>(&encoded).unwrap();

    assert_eq!(decoded.pixels(), original.pixels());
}

#[test]
fn none_round_trip_is_pixel_exact() {
    let original = sample_rgb_image();
    let encoded = TiffEncoder::default()
        .encode_with_options(
            &original,
            &SaveOptions::default().with_tiff_compression(TiffCompression::None),
        )
        .unwrap();
    let decoded = TiffDecoder.decode::<U8>(&encoded).unwrap();

    assert_eq!(decoded.pixels(), original.pixels());
}

#[test]
fn decode_multi_strip_tiff_is_pixel_exact() {
    let original = multi_strip_rgb_image();
    let encoded = TiffEncoder::default()
        .encode_with_options(
            &original,
            &SaveOptions::default().with_tiff_compression(TiffCompression::Deflate),
        )
        .unwrap();
    let mut decoder = Decoder::new(Cursor::new(encoded.as_slice())).unwrap();

    assert_eq!(decoder.get_chunk_type(), ChunkType::Strip);
    assert_eq!(
        decoder.find_tag_unsigned::<u32>(Tag::RowsPerStrip).unwrap(),
        Some(DEFAULT_TIFF_ROWS_PER_STRIP)
    );

    let decoded = TiffDecoder.decode::<U8>(&encoded).unwrap();
    assert_eq!(decoded.width(), original.width());
    assert_eq!(decoded.height(), original.height());
    assert_eq!(decoded.bands(), original.bands());
    assert_eq!(decoded.pixels(), original.pixels());
}

#[test]
fn uncompressed_strip_output_uses_single_full_height_strip() {
    let original = multi_strip_rgb_image();
    let encoded = TiffEncoder::default().encode(&original).unwrap();
    let mut decoder = Decoder::new(Cursor::new(encoded.as_slice())).unwrap();

    assert_eq!(decoder.get_chunk_type(), ChunkType::Strip);
    assert_eq!(
        decoder.find_tag_unsigned::<u32>(Tag::RowsPerStrip).unwrap(),
        Some(original.height())
    );
}

#[test]
fn jpeg_round_trip_stays_within_small_error() {
    let original = solid_gray_image(96);
    let encoded = TiffEncoder::default()
        .encode_with_options(
            &original,
            &SaveOptions::default()
                .with_tiff_compression(TiffCompression::Jpeg)
                .with_quality(100),
        )
        .unwrap();
    let decoded = TiffDecoder.decode::<U8>(&encoded).unwrap();

    assert_eq!(decoded.width(), original.width());
    assert_eq!(decoded.height(), original.height());
    assert_eq!(decoded.bands(), original.bands());
    for (expected, actual) in original.pixels().iter().zip(decoded.pixels()) {
        assert!((i16::from(*expected) - i16::from(*actual)).abs() <= 1);
    }
}

#[test]
fn tiled_output_sets_requested_tile_dimensions() {
    let original = sample_rgb_image();
    let encoded = TiffEncoder::default()
        .encode_with_options(
            &original,
            &SaveOptions::default()
                .with_tile_width(4)
                .with_tile_height(3)
                .with_tiff_compression(TiffCompression::Deflate)
                .with_tiff_predictor(TiffPredictor::Horizontal),
        )
        .unwrap();
    let mut decoder = Decoder::new(Cursor::new(encoded.as_slice())).unwrap();

    assert_eq!(decoder.get_chunk_type(), ChunkType::Tile);
    assert_eq!(
        decoder.find_tag_unsigned::<u32>(Tag::TileWidth).unwrap(),
        Some(4)
    );
    assert_eq!(
        decoder.find_tag_unsigned::<u32>(Tag::TileLength).unwrap(),
        Some(3)
    );
}

#[test]
fn round_trip_f32_rgb_is_pixel_exact() {
    let codec = TiffCodec::default();
    let original =
        Image::<F32>::from_buffer(2, 1, 3, vec![0.0, 0.5, 1.0, 0.25, 0.75, 0.125]).unwrap();

    let encoded = codec.encode(&original).unwrap();
    let mut decoder = Decoder::new(Cursor::new(encoded.as_slice())).unwrap();
    let sample_format = decoder
        .find_tag_unsigned_vec::<u16>(Tag::SampleFormat)
        .unwrap();
    let bits_per_sample = decoder
        .find_tag_unsigned_vec::<u16>(Tag::BitsPerSample)
        .unwrap();
    let decoded = codec.decode::<F32>(&encoded).unwrap();

    assert_eq!(
        sample_format,
        Some(vec![
            SampleFormat::IEEEFP.to_u16();
            original.bands() as usize
        ])
    );
    assert_eq!(
        bits_per_sample,
        Some(vec![32u16; original.bands() as usize])
    );
    assert_eq!(decoded.width(), original.width());
    assert_eq!(decoded.height(), original.height());
    assert_eq!(decoded.bands(), original.bands());
    assert_eq!(decoded.pixels(), original.pixels());
}

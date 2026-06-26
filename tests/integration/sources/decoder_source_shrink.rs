#[cfg(feature = "jpeg")]
use std::num::NonZeroU8;

#[cfg(feature = "jpeg")]
use viprs::adapters::codecs::JpegCodec;
#[cfg(feature = "jpeg")]
use viprs::adapters::sources::decoder_source::DecoderSource;
#[cfg(feature = "jpeg")]
use viprs::domain::codec_options::{LoadOptions, SaveOptions};
#[cfg(feature = "jpeg")]
use viprs::domain::format::U8;
#[cfg(feature = "jpeg")]
use viprs::domain::image::InMemoryImage;
#[cfg(feature = "jpeg")]
use viprs::ports::codec::{ImageDecoder, ImageEncoder};
#[cfg(feature = "jpeg")]
use viprs::ports::source::ImageSource;

#[cfg(feature = "jpeg")]
fn patterned_rgb_8x8() -> InMemoryImage<U8> {
    let mut data = Vec::with_capacity(8 * 8 * 3);
    for y in 0u8..8 {
        for x in 0u8..8 {
            data.push(x.saturating_mul(17));
            data.push(y.saturating_mul(19));
            data.push(x.saturating_add(y).saturating_mul(11));
        }
    }
    InMemoryImage::from_buffer(8, 8, 3, data).unwrap()
}

#[test]
#[cfg(feature = "jpeg")]
fn decoder_source_applies_shrink_factor_to_decoded_size() {
    let codec = JpegCodec;
    let encoded = codec
        .encode_with_options(
            &patterned_rgb_8x8(),
            &SaveOptions::default().with_quality(100),
        )
        .unwrap();

    let source = DecoderSource::<_, U8>::with_options(
        codec,
        &encoded,
        LoadOptions::default().with_shrink(NonZeroU8::new(2).unwrap()),
    )
    .unwrap();

    assert_eq!(source.shrink_factor(), 2);
    assert_eq!(source.load_options().shrink_factor, NonZeroU8::new(2));
    assert_eq!(source.width(), 4);
    assert_eq!(source.height(), 4);
    assert_eq!(source.bands(), 3);

    let decoded = <JpegCodec as ImageDecoder>::decode_with_options::<U8>(
        &JpegCodec,
        &encoded,
        &LoadOptions::default().with_shrink(NonZeroU8::new(2).unwrap()),
    )
    .unwrap();
    assert_eq!(source.image().unwrap().pixels(), decoded.pixels());
}

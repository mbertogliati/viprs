use std::{any::Any, path::Path};

use viprs_core::{
    codec_options::{LoadOptions, SaveOptions},
    error::ViprsError,
    format::{BandFormat, BandFormatId, F32, F64, I16, I32, U8, U16, U32},
    image::Image,
};
use viprs_ports::codec::{ImageCodec, ImageDecoder, ImageEncoder};

struct CodecBridge<C> {
    codec: C,
    file_extensions: &'static [&'static str],
}

impl<C> CodecBridge<C> {
    const fn new(codec: C, file_extensions: &'static [&'static str]) -> Self {
        Self {
            codec,
            file_extensions,
        }
    }

    fn decode_typed<F: BandFormat>(
        &self,
        src: &[u8],
        opts: &LoadOptions,
    ) -> Result<Box<dyn Any + Send>, ViprsError>
    where
        C: ImageDecoder,
    {
        Ok(Box::new(self.codec.decode_with_options::<F>(src, opts)?))
    }

    fn decode_path_typed<F: BandFormat>(
        &self,
        path: &Path,
        opts: &LoadOptions,
    ) -> Result<Box<dyn Any + Send>, ViprsError>
    where
        C: ImageDecoder,
    {
        Ok(Box::new(
            self.codec.decode_path_with_options::<F>(path, opts)?,
        ))
    }

    fn encode_typed<F: BandFormat>(
        &self,
        image: &(dyn Any + Send + Sync),
        opts: &SaveOptions,
    ) -> Result<Vec<u8>, ViprsError>
    where
        C: ImageEncoder,
    {
        let image = image.downcast_ref::<Image<F>>().ok_or_else(|| {
            ViprsError::Codec(format!(
                "foreign: codec '{}' received mismatched image type for {:?}",
                ImageEncoder::format_name(&self.codec),
                F::ID
            ))
        })?;
        self.codec.encode_with_options::<F>(image, opts)
    }
}

impl<C> ImageCodec for CodecBridge<C>
where
    C: ImageDecoder + ImageEncoder + Send + Sync + 'static,
{
    fn format_name(&self) -> &'static str {
        ImageDecoder::format_name(&self.codec)
    }

    fn file_extensions(&self) -> &'static [&'static str] {
        self.file_extensions
    }

    fn sniff(&self, header: &[u8]) -> bool {
        self.codec.sniff(header)
    }

    fn can_decode_path(&self, path: &Path) -> bool {
        self.codec.can_decode_path(path)
    }

    fn decode_boxed(
        &self,
        src: &[u8],
        band_format: BandFormatId,
        opts: &LoadOptions,
    ) -> Result<Box<dyn Any + Send>, ViprsError> {
        match band_format {
            BandFormatId::U8 => self.decode_typed::<U8>(src, opts),
            BandFormatId::U16 => self.decode_typed::<U16>(src, opts),
            BandFormatId::I16 => self.decode_typed::<I16>(src, opts),
            BandFormatId::U32 => self.decode_typed::<U32>(src, opts),
            BandFormatId::I32 => self.decode_typed::<I32>(src, opts),
            BandFormatId::F32 => self.decode_typed::<F32>(src, opts),
            BandFormatId::F64 => self.decode_typed::<F64>(src, opts),
        }
    }

    fn decode_boxed_path(
        &self,
        path: &Path,
        band_format: BandFormatId,
        opts: &LoadOptions,
    ) -> Result<Box<dyn Any + Send>, ViprsError> {
        match band_format {
            BandFormatId::U8 => self.decode_path_typed::<U8>(path, opts),
            BandFormatId::U16 => self.decode_path_typed::<U16>(path, opts),
            BandFormatId::I16 => self.decode_path_typed::<I16>(path, opts),
            BandFormatId::U32 => self.decode_path_typed::<U32>(path, opts),
            BandFormatId::I32 => self.decode_path_typed::<I32>(path, opts),
            BandFormatId::F32 => self.decode_path_typed::<F32>(path, opts),
            BandFormatId::F64 => self.decode_path_typed::<F64>(path, opts),
        }
    }

    fn encode_boxed(
        &self,
        image: &(dyn Any + Send + Sync),
        band_format: BandFormatId,
        opts: &SaveOptions,
    ) -> Result<Vec<u8>, ViprsError> {
        match band_format {
            BandFormatId::U8 => self.encode_typed::<U8>(image, opts),
            BandFormatId::U16 => self.encode_typed::<U16>(image, opts),
            BandFormatId::I16 => self.encode_typed::<I16>(image, opts),
            BandFormatId::U32 => self.encode_typed::<U32>(image, opts),
            BandFormatId::I32 => self.encode_typed::<I32>(image, opts),
            BandFormatId::F32 => self.encode_typed::<F32>(image, opts),
            BandFormatId::F64 => self.encode_typed::<F64>(image, opts),
        }
    }
}

pub fn boxed_codec<C>(codec: C, file_extensions: &'static [&'static str]) -> Box<dyn ImageCodec>
where
    C: ImageDecoder + ImageEncoder + Send + Sync + 'static,
{
    Box::new(CodecBridge::new(codec, file_extensions))
}

struct DecoderBridge<C> {
    codec: C,
    file_extensions: &'static [&'static str],
    extension_decode_fallback: bool,
}

impl<C> DecoderBridge<C> {
    const fn new(
        codec: C,
        file_extensions: &'static [&'static str],
        extension_decode_fallback: bool,
    ) -> Self {
        Self {
            codec,
            file_extensions,
            extension_decode_fallback,
        }
    }

    fn decode_typed<F: BandFormat>(
        &self,
        src: &[u8],
        opts: &LoadOptions,
    ) -> Result<Box<dyn Any + Send>, ViprsError>
    where
        C: ImageDecoder,
    {
        Ok(Box::new(self.codec.decode_with_options::<F>(src, opts)?))
    }

    fn decode_path_typed<F: BandFormat>(
        &self,
        path: &Path,
        opts: &LoadOptions,
    ) -> Result<Box<dyn Any + Send>, ViprsError>
    where
        C: ImageDecoder,
    {
        Ok(Box::new(
            self.codec.decode_path_with_options::<F>(path, opts)?,
        ))
    }
}

impl<C> ImageCodec for DecoderBridge<C>
where
    C: ImageDecoder + Send + Sync + 'static,
{
    fn format_name(&self) -> &'static str {
        ImageDecoder::format_name(&self.codec)
    }

    fn file_extensions(&self) -> &'static [&'static str] {
        self.file_extensions
    }

    fn supports_extension_decode_fallback(&self) -> bool {
        self.extension_decode_fallback
    }

    fn can_encode(&self) -> bool {
        false
    }

    fn sniff(&self, header: &[u8]) -> bool {
        self.codec.sniff(header)
    }

    fn can_decode_path(&self, path: &Path) -> bool {
        self.codec.can_decode_path(path)
            || (self.extension_decode_fallback
                && path
                    .extension()
                    .and_then(std::ffi::OsStr::to_str)
                    .is_some_and(|extension| self.supports_format(extension)))
    }

    fn decode_boxed(
        &self,
        src: &[u8],
        band_format: BandFormatId,
        opts: &LoadOptions,
    ) -> Result<Box<dyn Any + Send>, ViprsError> {
        match band_format {
            BandFormatId::U8 => self.decode_typed::<U8>(src, opts),
            BandFormatId::U16 => self.decode_typed::<U16>(src, opts),
            BandFormatId::I16 => self.decode_typed::<I16>(src, opts),
            BandFormatId::U32 => self.decode_typed::<U32>(src, opts),
            BandFormatId::I32 => self.decode_typed::<I32>(src, opts),
            BandFormatId::F32 => self.decode_typed::<F32>(src, opts),
            BandFormatId::F64 => self.decode_typed::<F64>(src, opts),
        }
    }

    fn decode_boxed_path(
        &self,
        path: &Path,
        band_format: BandFormatId,
        opts: &LoadOptions,
    ) -> Result<Box<dyn Any + Send>, ViprsError> {
        match band_format {
            BandFormatId::U8 => self.decode_path_typed::<U8>(path, opts),
            BandFormatId::U16 => self.decode_path_typed::<U16>(path, opts),
            BandFormatId::I16 => self.decode_path_typed::<I16>(path, opts),
            BandFormatId::U32 => self.decode_path_typed::<U32>(path, opts),
            BandFormatId::I32 => self.decode_path_typed::<I32>(path, opts),
            BandFormatId::F32 => self.decode_path_typed::<F32>(path, opts),
            BandFormatId::F64 => self.decode_path_typed::<F64>(path, opts),
        }
    }

    fn encode_boxed(
        &self,
        _image: &(dyn Any + Send + Sync),
        _band_format: BandFormatId,
        _opts: &SaveOptions,
    ) -> Result<Vec<u8>, ViprsError> {
        Err(ViprsError::Codec(format!(
            "foreign: codec '{}' is decode-only",
            self.format_name()
        )))
    }
}

pub fn boxed_decoder<C>(codec: C, file_extensions: &'static [&'static str]) -> Box<dyn ImageCodec>
where
    C: ImageDecoder + Send + Sync + 'static,
{
    Box::new(DecoderBridge::new(codec, file_extensions, false))
}

#[cfg(any(
    all(test, feature = "_integration"),
    feature = "dcraw",
    feature = "openslide"
))]
pub fn boxed_extension_decoder<C>(
    codec: C,
    file_extensions: &'static [&'static str],
) -> Box<dyn ImageCodec>
where
    C: ImageDecoder + Send + Sync + 'static,
{
    Box::new(DecoderBridge::new(codec, file_extensions, true))
}

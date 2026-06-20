#![allow(clippy::unused_self)]
// REASON: registry helpers remain instance methods for API consistency with mutable registry flows.

use std::{
    any::Any,
    fs::{self, File},
    io::Read,
    path::Path,
    sync::OnceLock,
};

use crate::{AnalyzeCodec, MatCodec, MatrixCodec, RawCodec};
use viprs_core::{
    codec_options::{LoadOptions, SaveOptions},
    error::ViprsError,
    format::{BandFormat, U8},
    image::Image,
};
use viprs_ports::codec::ImageCodec;

#[cfg(feature = "csv")]
use crate::CsvCodec;

#[cfg(feature = "avif")]
use crate::AvifCodec;
#[cfg(feature = "exr")]
use crate::ExrCodec;
#[cfg(feature = "fits")]
use crate::FitsCodec;
#[cfg(feature = "gif")]
use crate::GifCodec;
#[cfg(feature = "heif")]
use crate::HeifCodec;
#[cfg(feature = "jp2k")]
use crate::Jp2kCodec;
#[cfg(feature = "jpeg")]
use crate::JpegCodec;
#[cfg(feature = "jxl")]
use crate::JxlCodec;
#[cfg(feature = "nifti")]
use crate::NiftiCodec;
#[cfg(feature = "pdf-poppler")]
use crate::PdfPopplerDecoder;
#[cfg(feature = "pfm")]
use crate::PfmCodec;
#[cfg(feature = "png")]
use crate::PngCodec;
#[cfg(feature = "radiance")]
use crate::RadianceCodec;
#[cfg(feature = "svg")]
use crate::SvgDecoder;
#[cfg(feature = "tiff")]
use crate::TiffCodec;
#[cfg(feature = "uhdr")]
use crate::UhdrCodec;
#[cfg(feature = "vips-format")]
use crate::VipsCodec;
#[cfg(feature = "webp")]
use crate::WebpCodec;
#[cfg(feature = "dcraw")]
use crate::{DCRAW_EXTENSIONS, DcrawDecoder};
#[cfg(feature = "magick")]
use crate::{MAGICK_FALLBACK_SAVERS, MagickFallbackLoader};
#[cfg(feature = "openslide")]
use crate::{OPENSLIDE_EXTENSIONS, OpenSlideDecoder};

#[cfg(any(feature = "dcraw", feature = "openslide"))]
use super::boxed_extension_decoder;
use super::{
    boxed_codec, boxed_decoder, deferred_decode_error, deferred_encode_error,
    is_deepzoom_extension, save_deepzoom,
};

const SNIFF_HEADER_LEN: usize = 1_000;
static DEFAULT_FOREIGN_REGISTRY: OnceLock<ForeignRegistry> = OnceLock::new();

/// Runtime registry for foreign codecs.
///
/// `dyn ImageCodec` is allowed here by project rule: this is the plugin
/// registry boundary, not a pixel-path abstraction.
pub struct ForeignRegistry {
    codecs: Vec<Box<dyn ImageCodec>>,
}

impl ForeignRegistry {
    #[must_use]
    /// `new` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::codecs::registry::new;
    /// ```
    pub fn new() -> Self {
        Self { codecs: Vec::new() }
    }

    /// `register` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::codecs::registry::register;
    /// ```
    pub fn register(&mut self, codec: Box<dyn ImageCodec>) {
        self.codecs.push(codec);
    }

    /// `detect_format` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::codecs::registry::detect_format;
    /// ```
    pub fn detect_format(&self, path: &Path) -> Option<&dyn ImageCodec> {
        let extension = path.extension()?.to_str()?;
        self.codecs
            .iter()
            .find(|codec| codec.can_encode() && codec.supports_format(extension))
            .map(std::boxed::Box::as_ref)
    }

    /// `load` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::codecs::registry::load;
    /// ```
    pub fn load(&self, path: &Path) -> Result<Image<U8>, ViprsError> {
        self.load_as_with_options(path, &LoadOptions::default())
    }

    /// `load_with_options` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::codecs::registry::load_with_options;
    /// ```
    pub fn load_with_options(
        &self,
        path: &Path,
        opts: &LoadOptions,
    ) -> Result<Image<U8>, ViprsError> {
        self.load_as_with_options(path, opts)
    }

    /// `load_as` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::codecs::registry::load_as;
    /// ```
    pub fn load_as<F: BandFormat>(&self, path: &Path) -> Result<Image<F>, ViprsError> {
        self.load_as_with_options(path, &LoadOptions::default())
    }

    /// `load_as_with_options` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::codecs::registry::load_as_with_options;
    /// ```
    pub fn load_as_with_options<F: BandFormat>(
        &self,
        path: &Path,
        opts: &LoadOptions,
    ) -> Result<Image<F>, ViprsError> {
        if path.is_dir() {
            if let Some(codec) = self.find_path_decoder(path) {
                return self.decode_from_codec_path::<F>(codec, path, opts);
            }
            return Err(deferred_decode_error(path, &[]).unwrap_or_else(|| {
                ViprsError::Codec(format!("foreign: no decoder matched '{}'", path.display()))
            }));
        }

        let header = read_header(path)?;
        let codec = self
            .find_decoder(&header)
            .or_else(|| self.find_path_decoder(path))
            .or_else(|| self.find_decoder_by_extension(path))
            .ok_or_else(|| {
                deferred_decode_error(path, &header).unwrap_or_else(|| {
                    ViprsError::Codec(format!("foreign: no decoder matched '{}'", path.display()))
                })
            })?;
        self.decode_from_codec_path::<F>(codec, path, opts)
    }

    /// `load_from_memory` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::codecs::registry::load_from_memory;
    /// ```
    pub fn load_from_memory(&self, src: &[u8]) -> Result<(Image<U8>, &'static str), ViprsError> {
        self.load_from_memory_as_with_options(src, &LoadOptions::default())
    }

    /// `load_from_memory_with_options` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::codecs::registry::load_from_memory_with_options;
    /// ```
    pub fn load_from_memory_with_options(
        &self,
        src: &[u8],
        opts: &LoadOptions,
    ) -> Result<(Image<U8>, &'static str), ViprsError> {
        self.load_from_memory_as_with_options(src, opts)
    }

    /// `load_from_memory_as` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::codecs::registry::load_from_memory_as;
    /// ```
    pub fn load_from_memory_as<F: BandFormat>(
        &self,
        src: &[u8],
    ) -> Result<(Image<F>, &'static str), ViprsError> {
        self.load_from_memory_as_with_options(src, &LoadOptions::default())
    }

    /// `load_from_memory_as_with_options` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::codecs::registry::load_from_memory_as_with_options;
    /// ```
    pub fn load_from_memory_as_with_options<F: BandFormat>(
        &self,
        src: &[u8],
        opts: &LoadOptions,
    ) -> Result<(Image<F>, &'static str), ViprsError> {
        let codec = self.find_decoder(src).ok_or_else(|| {
            ViprsError::Codec("foreign: no decoder matched in-memory input".into())
        })?;
        let image = self.decode_from_codec_memory::<F>(codec, src, opts)?;
        Ok((image, codec.format_name()))
    }

    /// `save` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::codecs::registry::save;
    /// ```
    pub fn save(&self, image: &Image<U8>, path: &Path) -> Result<(), ViprsError> {
        self.save_as_with_options(image, path, &SaveOptions::default())
    }

    /// `save_with_options` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::codecs::registry::save_with_options;
    /// ```
    pub fn save_with_options(
        &self,
        image: &Image<U8>,
        path: &Path,
        opts: &SaveOptions,
    ) -> Result<(), ViprsError> {
        self.save_as_with_options(image, path, opts)
    }

    /// `save_as` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::codecs::registry::save_as;
    /// ```
    pub fn save_as<F: BandFormat>(&self, image: &Image<F>, path: &Path) -> Result<(), ViprsError> {
        self.save_as_with_options(image, path, &SaveOptions::default())
    }

    /// `save_as_with_options` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::codecs::registry::save_as_with_options;
    /// ```
    pub fn save_as_with_options<F: BandFormat>(
        &self,
        image: &Image<F>,
        path: &Path,
        opts: &SaveOptions,
    ) -> Result<(), ViprsError> {
        if is_deepzoom_extension(path) {
            return save_deepzoom(image, path, opts);
        }

        let codec = self.detect_format(path).ok_or_else(|| {
            if let Some(err) = deferred_encode_error(path) {
                return err;
            }
            let extension = path
                .extension()
                .and_then(std::ffi::OsStr::to_str)
                .unwrap_or("<none>");
            ViprsError::Codec(format!("foreign: no encoder registered for '.{extension}'"))
        })?;
        let image: &(dyn Any + Send + Sync) = image;
        let encoded = codec.encode_boxed(image, F::ID, opts)?;
        fs::write(path, encoded)?;
        Ok(())
    }

    fn find_decoder(&self, header: &[u8]) -> Option<&dyn ImageCodec> {
        self.codecs
            .iter()
            .find(|codec| codec.sniff(header))
            .map(std::boxed::Box::as_ref)
    }

    pub(crate) fn find_decoder_by_extension(&self, path: &Path) -> Option<&dyn ImageCodec> {
        let extension = path.extension()?.to_str()?;
        self.codecs
            .iter()
            .find(|codec| {
                codec.supports_extension_decode_fallback() && codec.supports_format(extension)
            })
            .map(std::boxed::Box::as_ref)
    }

    fn find_path_decoder(&self, path: &Path) -> Option<&dyn ImageCodec> {
        self.codecs
            .iter()
            .find(|codec| codec.can_decode_path(path))
            .map(std::boxed::Box::as_ref)
    }

    fn decode_from_codec_path<F: BandFormat>(
        &self,
        codec: &dyn ImageCodec,
        path: &Path,
        opts: &LoadOptions,
    ) -> Result<Image<F>, ViprsError> {
        let decoded = codec.decode_boxed_path(path, F::ID, opts)?;
        decoded
            .downcast::<Image<F>>()
            .map(|image| *image)
            .map_err(|_| {
                ViprsError::Codec(format!(
                    "foreign: codec '{}' returned an unexpected image type for {:?}",
                    codec.format_name(),
                    F::ID
                ))
            })
    }

    fn decode_from_codec_memory<F: BandFormat>(
        &self,
        codec: &dyn ImageCodec,
        src: &[u8],
        opts: &LoadOptions,
    ) -> Result<Image<F>, ViprsError> {
        let decoded = codec.decode_boxed(src, F::ID, opts)?;
        decoded
            .downcast::<Image<F>>()
            .map(|image| *image)
            .map_err(|_| {
                ViprsError::Codec(format!(
                    "foreign: codec '{}' returned an unexpected image type for {:?}",
                    codec.format_name(),
                    F::ID
                ))
            })
    }
}

pub fn read_header(path: &Path) -> Result<Vec<u8>, ViprsError> {
    let mut file = File::open(path)?;
    let mut header = vec![0u8; SNIFF_HEADER_LEN];
    let read = file.read(&mut header)?;
    header.truncate(read);
    Ok(header)
}

impl Default for ForeignRegistry {
    fn default() -> Self {
        let mut registry = Self::new();

        #[cfg(feature = "uhdr")]
        registry.register(boxed_decoder(UhdrCodec, &["jpg", "jpeg", "jpe"]));
        #[cfg(feature = "jpeg")]
        registry.register(boxed_codec(JpegCodec, &["jpg", "jpeg", "jpe"]));
        #[cfg(feature = "jp2k")]
        registry.register(boxed_codec(
            Jp2kCodec,
            &["jp2", "j2k", "jpf", "jpx", "j2c", "jpc"],
        ));
        #[cfg(feature = "openslide")]
        registry.register(boxed_extension_decoder(
            OpenSlideDecoder,
            OPENSLIDE_EXTENSIONS,
        ));
        #[cfg(feature = "png")]
        registry.register(boxed_codec(PngCodec::default(), &["png"]));
        #[cfg(feature = "webp")]
        registry.register(boxed_codec(WebpCodec, &["webp"]));
        #[cfg(feature = "tiff")]
        registry.register(boxed_codec(TiffCodec::default(), &["tif", "tiff"]));
        #[cfg(feature = "gif")]
        registry.register(boxed_codec(GifCodec::default(), &["gif"]));
        #[cfg(feature = "exr")]
        registry.register(boxed_codec(ExrCodec, &["exr"]));
        #[cfg(feature = "radiance")]
        registry.register(boxed_codec(RadianceCodec, &["hdr"]));
        #[cfg(feature = "pfm")]
        registry.register(boxed_codec(PfmCodec, &["pfm"]));
        #[cfg(feature = "fits")]
        registry.register(boxed_codec(FitsCodec, &["fits", "fit", "fts"]));
        #[cfg(feature = "avif")]
        registry.register(boxed_codec(AvifCodec, &["avif"]));
        #[cfg(feature = "heif")]
        registry.register(boxed_codec(HeifCodec, &["heif", "heic", "hif"]));
        #[cfg(feature = "svg")]
        registry.register(boxed_decoder(SvgDecoder, &["svg", "svgz"]));
        #[cfg(feature = "jxl")]
        registry.register(boxed_decoder(JxlCodec, &["jxl"]));
        #[cfg(feature = "magick")]
        registry.register(Box::new(MagickFallbackLoader));
        #[cfg(feature = "pdf-poppler")]
        registry.register(boxed_decoder(PdfPopplerDecoder, &["pdf"]));
        #[cfg(feature = "dcraw")]
        registry.register(boxed_extension_decoder(DcrawDecoder, DCRAW_EXTENSIONS));

        // Text/binary formats that need no external dependencies.
        #[cfg(feature = "csv")]
        registry.register(boxed_codec(CsvCodec, &["csv"]));
        registry.register(boxed_codec(MatrixCodec, &["mat"]));
        registry.register(boxed_decoder(MatCodec, &["mat"]));
        registry.register(boxed_codec(AnalyzeCodec, &["img", "hdr"]));
        registry.register(boxed_codec(RawCodec, &["raw"]));
        #[cfg(feature = "nifti")]
        registry.register(boxed_decoder(NiftiCodec, &["nii", "hdr"]));
        #[cfg(feature = "vips-format")]
        registry.register(boxed_codec(VipsCodec, &["v", "vips"]));
        #[cfg(feature = "magick")]
        for saver in MAGICK_FALLBACK_SAVERS {
            registry.register(Box::new(*saver));
        }

        registry
    }
}

impl ForeignRegistry {
    /// `shared` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::codecs::registry::shared;
    /// ```
    pub fn shared() -> &'static Self {
        DEFAULT_FOREIGN_REGISTRY.get_or_init(Self::default)
    }
}

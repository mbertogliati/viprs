//! Codec adapters: format-specific implementations of
//! [`viprs_ports::codec::ImageDecoder`] and
//! [`viprs_ports::codec::ImageEncoder`].
//!
//! Each codec is gated behind a Cargo feature flag:
//!
//! | Feature | Crate(s)                     | Formats                          |
//! |---------|------------------------------|----------------------------------|
//! | `jpeg`  | `turbojpeg`                  | JPEG decode + encode             |
//! | `bmp`   | `image`                      | BMP decode + encode              |
//! | `png`   | `png`                        | PNG decode + encode              |
//! | `pnm`   | none                         | PBM/PGM/PPM/PNM decode + encode  |
//! | `webp`  | `webp`                       | WebP decode + encode             |
//! | `tiff`  | `tiff`                       | TIFF decode + encode (pure Rust) |
//! | `gif`   | `gif`                        | GIF decode + encode (pure Rust)  |
//! | `exr`   | `exr`                        | `OpenEXR` half/float decode + `F32` encode |
//! | `radiance` | none                      | Radiance HDR decode + encode      |
//! | `pfm`   | none                         | Portable Float Map decode + encode |
//! | `avif`  | `ravif`, `image`             | AVIF encode (pure Rust) + decode (libdav1d) |
//! | `heif`  | `libheif-rs`                 | HEIF/HEIC decode (requires libheif C lib)   |
//! | `svg`   | `resvg`                      | SVG decode/rasterize only        |
//! | `jxl`   | `jpegxl-rs`, `jxl-oxide`     | JPEG XL decode only              |
//! | `pdf-poppler` | external `pdfinfo`/`pdftoppm` | PDF decode via Poppler tools |
//! | `jp2k`  | `jpeg2k`, `openjpeg-sys`     | JPEG 2000 decode + encode        |
//! | `openslide` | `openslide-rs`            | Whole-slide microscopy decode    |
//! | `uhdr`  | none                         | Ultra HDR decode (JPEGR base)    |
//! | `dcraw` | external `dcraw`/`dcraw_emu` | Camera RAW decode via libraw toolchain |
//! | `csv`   | none                         | CSV float matrix decode + encode |
//! | `raw`   | none                         | RAW headerless decode + encode   |
//! | `nifti` | `nifti-rs`                   | NIfTI-1 decode + encode (pure Rust) |
//! | `analyze` | none                       | Analyze 7.5 (`.hdr` + `.img`) decode + encode |
//! | `matrix` | none                        | libvips text matrix decode + encode |
//! | `fits`  | `fitsio-sys`                 | FITS decode + encode (cfitsio)         |
//! | `mat`   | none (`mat-hdf5`: `hdf5-reader`) | MATLAB MAT v5 decode + v7.3 HDF5 decode |
//! | `magick`| external `magick` CLI        | `ImageMagick` fallback load + save |
//! | `vips-format` | none                   | Native `VIPS` `.v` / `.vips` decode + encode |
//!
//! All implementations are `Send + Sync` and do not hold locks across tile reads.
//! Codec API design covers `shrink-on-load`, `SaveOptions`, and related behavior.

#[cfg(feature = "csv")]
pub mod csv;

#[cfg(feature = "bmp")]
pub mod bmp;

#[cfg(feature = "jpeg")]
pub mod jpeg;

#[cfg(any(feature = "jpeg", feature = "webp"))]
pub(crate) mod shrink_on_load;

#[cfg(any(feature = "jpeg", feature = "png", feature = "webp"))]
pub(crate) mod web_colour;

#[cfg(feature = "png")]
pub mod png;

#[cfg(feature = "pnm")]
pub mod pnm;

#[cfg(feature = "webp")]
pub mod webp;

#[cfg(feature = "tiff")]
pub mod tiff;

#[cfg(feature = "gif")]
pub mod gif;

#[cfg(feature = "exr")]
pub mod exr;

#[cfg(feature = "radiance")]
pub mod radiance;

#[cfg(feature = "pfm")]
pub mod pfm;

#[cfg(feature = "avif")]
pub mod avif;

#[cfg(feature = "heif")]
pub mod heif;

#[cfg(any(feature = "avif", feature = "heif"))]
pub(crate) mod heif_support;

#[cfg(feature = "svg")]
pub mod svg;

#[cfg(feature = "jxl")]
pub mod jxl;

#[cfg(feature = "dcraw")]
pub mod dcraw;
#[cfg(feature = "fits")]
pub mod fits;
#[cfg(feature = "jp2k")]
pub mod jp2k;
#[cfg(feature = "magick")]
pub mod magick;
#[cfg(feature = "openslide")]
pub mod openslide;
#[cfg(feature = "pdf-poppler")]
pub mod pdf_poppler;

#[cfg(feature = "uhdr")]
pub mod uhdr;

pub mod analyze;
#[cfg(feature = "deepzoom")]
pub mod deepzoom;
pub mod mat;
pub mod matrix;
#[cfg(feature = "nifti")]
pub mod nifti;
pub mod raw;
pub mod registry;
#[cfg(feature = "vips-format")]
pub mod vips_format;

#[cfg(all(test, feature = "_root_test_support"))]
#[path = "../../../src/test_support.rs"]
mod test_support;

#[cfg(all(test, feature = "_root_test_support"))]
#[global_allocator]
static TEST_ALLOCATOR: test_support::CountingAllocator = test_support::CountingAllocator;

#[cfg(feature = "bmp")]
pub use bmp::BmpCodec;

#[cfg(feature = "jpeg")]
pub use jpeg::JpegCodec;

#[cfg(feature = "png")]
pub use png::{PngCodec, PngEncoder};

#[cfg(feature = "pnm")]
pub use pnm::PnmCodec;

#[cfg(feature = "webp")]
pub use webp::{WebpCodec, WebpEncodeOptions};

#[cfg(feature = "tiff")]
pub use tiff::{TiffCodec, TiffCompression, TiffDecoder, TiffEncoder};

#[cfg(feature = "gif")]
pub use gif::GifCodec;

#[cfg(feature = "exr")]
pub use exr::ExrCodec;

#[cfg(feature = "radiance")]
pub use radiance::RadianceCodec;

#[cfg(feature = "pfm")]
pub use pfm::PfmCodec;

#[cfg(feature = "avif")]
pub use avif::AvifCodec;

#[cfg(feature = "heif")]
pub use heif::HeifCodec;

#[cfg(feature = "svg")]
pub use svg::SvgDecoder;

#[cfg(feature = "jxl")]
pub use jxl::JxlCodec;

#[cfg(feature = "dcraw")]
pub use dcraw::{DCRAW_EXTENSIONS, DcrawDecoder};
#[cfg(feature = "fits")]
pub use fits::FitsCodec;
#[cfg(feature = "jp2k")]
pub use jp2k::Jp2kCodec;
#[cfg(feature = "magick")]
pub use magick::{
    MAGICK_FALLBACK_DECODE_EXTENSIONS, MAGICK_FALLBACK_SAVERS, MagickFallbackLoader,
    MagickFallbackSaver,
};
#[cfg(feature = "openslide")]
pub use openslide::{OPENSLIDE_EXTENSIONS, OpenSlideDecoder};
#[cfg(feature = "pdf-poppler")]
pub use pdf_poppler::PdfPopplerDecoder;

#[cfg(feature = "uhdr")]
pub use uhdr::UhdrCodec;

pub use analyze::AnalyzeCodec;
#[cfg(feature = "csv")]
pub use csv::CsvCodec;
pub use mat::MatCodec;
pub use matrix::MatrixCodec;
#[cfg(feature = "nifti")]
pub use nifti::NiftiCodec;
pub use raw::{RawCodec, RawDecoder, RawEncoder, RawLoadOptions, RawSaveOptions};
pub use registry::ForeignRegistry;
#[cfg(feature = "vips-format")]
pub use vips_format::VipsCodec;

#[allow(unused_macros)]
macro_rules! viprs_span {
    ($level:expr, $name:expr, $($field:tt)*) => {
        #[cfg(feature = "tracing")]
        let _span = tracing::span!($level, $name, $($field)*).entered();
        #[cfg(not(feature = "tracing"))]
        let _ = ();
    };
    ($level:expr, $name:expr) => {
        #[cfg(feature = "tracing")]
        let _span = tracing::span!($level, $name).entered();
        #[cfg(not(feature = "tracing"))]
        let _ = ();
    };
}


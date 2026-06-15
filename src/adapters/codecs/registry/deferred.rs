use std::path::Path;

use crate::domain::error::ViprsError;

struct DeferredForeignFormat {
    extensions: &'static [&'static str],
    decode_by_header: Option<fn(&[u8]) -> bool>,
    decode_feature: &'static str,
    encode_feature: &'static str,
    details: &'static str,
    supports_decode: bool,
}

const DEFERRED_FOREIGN_FORMATS: &[DeferredForeignFormat] = &[
    #[cfg(not(feature = "fits"))]
    DeferredForeignFormat {
        extensions: &["fits", "fit", "fts"],
        decode_by_header: None,
        decode_feature: "foreign decode: fits",
        encode_feature: "foreign encode: fits",
        details: "FITS codec parity is not yet implemented (cfitsio-backed fitsload/fitssave).",
        supports_decode: true,
    },
    #[cfg(not(feature = "jp2k"))]
    DeferredForeignFormat {
        extensions: &["jp2", "j2k", "jpf", "jpx"],
        decode_by_header: None,
        decode_feature: "foreign decode: jp2k",
        encode_feature: "foreign encode: jp2k",
        details: "JPEG 2000 parity is not yet implemented (jp2kload/jp2ksave + Ultra HDR family).",
        supports_decode: true,
    },
    DeferredForeignFormat {
        extensions: &["pdf"],
        decode_by_header: Some(pdf_header_sniff),
        decode_feature: "foreign decode: pdf-poppler",
        encode_feature: "foreign encode: pdfium/poppler",
        details: "PDF decode requires feature `pdf-poppler` (Poppler `pdfinfo` + `pdftoppm`) with page/n/dpi support; PDF encode is not yet implemented.",
        supports_decode: cfg!(not(feature = "pdf-poppler")),
    },
    #[cfg(not(feature = "openslide"))]
    DeferredForeignFormat {
        extensions: &["svs", "vms", "vmu", "ndpi", "scn", "mrxs", "svslide", "bif"],
        decode_by_header: None,
        decode_feature: "foreign decode: openslide",
        encode_feature: "foreign encode: openslide",
        details: "OpenSlide whole-slide decode requires feature `openslide`.",
        supports_decode: true,
    },
    DeferredForeignFormat {
        extensions: &["dz", "szi", "dzi"],
        decode_by_header: None,
        decode_feature: "foreign decode: deepzoom",
        encode_feature: "foreign encode: deepzoom",
        details: "DeepZoom export is not yet implemented (dzsave tile pyramid backend).",
        supports_decode: false,
    },
    #[cfg(not(feature = "dcraw"))]
    DeferredForeignFormat {
        extensions: &[
            "3fr", "ari", "arw", "cap", "cin", "cr2", "cr3", "crw", "dcr", "dng", "erf", "fff",
            "iiq", "k25", "kdc", "mdc", "mos", "mrw", "nef", "nrw", "orf", "ori", "pef", "pxn",
            "raf", "rw2", "rwl", "sr2", "srf", "srw", "x3f",
        ],
        decode_by_header: None,
        decode_feature: "foreign decode: dcraw/libraw",
        encode_feature: "foreign encode: dcraw/libraw",
        details: "Camera RAW support is not yet implemented (dcrawload/libraw backend).",
        supports_decode: true,
    },
    #[cfg(not(feature = "magick"))]
    DeferredForeignFormat {
        extensions: &[
            "bmp", "dib", "ico", "icns", "psd", "pcx", "tga", "eps", "ps", "xcf", "dcm",
        ],
        decode_by_header: None,
        decode_feature: "foreign decode: magick-fallback",
        encode_feature: "foreign encode: magick-fallback",
        details: "ImageMagick fallback support is not yet implemented (magickload/magicksave with low priority).",
        supports_decode: true,
    },
];

const PDF_MAGIC_MAX_OFFSET: usize = 32;

pub fn pdf_header_sniff(header: &[u8]) -> bool {
    if header.len() < 4 {
        return false;
    }

    let max_offset = PDF_MAGIC_MAX_OFFSET.min(header.len().saturating_sub(4));
    (0..=max_offset).any(|offset| &header[offset..offset + 4] == b"%PDF")
}

fn find_deferred_format(extension: &str) -> Option<&'static DeferredForeignFormat> {
    DEFERRED_FOREIGN_FORMATS.iter().find(|entry| {
        entry
            .extensions
            .iter()
            .any(|candidate| candidate.eq_ignore_ascii_case(extension))
    })
}

pub fn deferred_decode_error(path: &Path, header: &[u8]) -> Option<ViprsError> {
    let extension_match = path
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .and_then(find_deferred_format);
    let header_match = DEFERRED_FOREIGN_FORMATS
        .iter()
        .find(|entry| entry.decode_by_header.is_some_and(|sniff| sniff(header)));

    extension_match
        .or(header_match)
        .and_then(|entry| entry.supports_decode.then_some(entry))
        .map(|entry| ViprsError::Unimplemented {
            feature: entry.decode_feature,
            details: entry.details,
        })
}

pub fn deferred_encode_error(path: &Path) -> Option<ViprsError> {
    let extension = path.extension()?.to_str()?;
    find_deferred_format(extension).map(|entry| ViprsError::Unimplemented {
        feature: entry.encode_feature,
        details: entry.details,
    })
}

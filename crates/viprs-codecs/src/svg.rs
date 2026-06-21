//! Svg adapter module.
//!
//! This module provides concrete codec-related infrastructure used by the
//! adapter layer for loading, saving, or normalizing external image formats.

#![cfg(feature = "svg")]

//! SVG rasterizer codec backed by `resvg`.
//!
//! Decode support only: SVG is a read-only foreign format in libvips as well.
//! Output is always RGBA `U8`.

use resvg::{tiny_skia, usvg};

use viprs_core::codec_options::LoadOptions;
use viprs_core::error::ViprsError;
use viprs_core::format::{BandFormat, BandFormatId};
use viprs_core::image::{Image, ImageMetadata, Interpretation};
use viprs_ports::codec::ImageDecoder;

const DEFAULT_SVG_DPI: f64 = 72.0;
const DEFAULT_SVG_SCALE: f64 = 1.0;
const SVG_BANDS: u32 = 4;
const SVG_MAX_RASTER_BYTES: u128 = 256_u128 * 1024 * 1024;

/// SVG decoder implementation.
#[derive(Debug, Clone, Copy, Default)]
/// The `SvgDecoder` type provides concrete adapter functionality in the `codecs` module.
/// Use this type when you need the runtime behavior implemented by this adapter.
///
/// # Examples
///
/// ```rust
/// let _ = core::mem::size_of::<viprs_codecs::svg::SvgDecoder>();
/// ```
pub struct SvgDecoder;

#[inline]
fn require_u8<F: BandFormat>() -> Result<(), ViprsError> {
    if F::ID != BandFormatId::U8 {
        return Err(ViprsError::Codec(format!(
            "svg: unsupported format {:?}; only U8 is supported",
            F::ID
        )));
    }
    Ok(())
}

fn parse_svg_tree(src: &[u8]) -> Result<usvg::Tree, ViprsError> {
    let options = usvg::Options {
        dpi: DEFAULT_SVG_DPI as f32,
        ..usvg::Options::default()
    };
    usvg::Tree::from_data(src, &options).map_err(|err| ViprsError::Codec(err.to_string()))
}

fn normalize_page_selection(opts: &LoadOptions) -> Result<(), ViprsError> {
    let page = opts.page.unwrap_or(0);
    if page > 0 {
        return Err(ViprsError::Codec(format!(
            "svg: requested page {page}, but SVG only exposes page 0"
        )));
    }

    if let Some(value) = opts.n
        && value != -1
        && value <= 0
    {
        return Err(ViprsError::Codec(format!(
            "svg: n must be positive or -1, got {value}"
        )));
    }

    Ok(())
}

fn raster_scale(opts: &LoadOptions) -> Result<f64, ViprsError> {
    let dpi = opts.dpi.unwrap_or(DEFAULT_SVG_DPI);
    if !dpi.is_finite() || dpi <= 0.0 {
        return Err(ViprsError::Codec(format!("svg: invalid dpi {dpi}")));
    }

    let scale = opts.scale.unwrap_or(DEFAULT_SVG_SCALE);
    if !scale.is_finite() || scale <= 0.0 {
        return Err(ViprsError::Codec(format!("svg: invalid scale {scale}")));
    }

    Ok(scale * dpi / DEFAULT_SVG_DPI)
}

fn scaled_dimension(dimension: u32, scale: f64) -> Result<u32, ViprsError> {
    let scaled = f64::from(dimension) * scale;
    if !scaled.is_finite() || scaled > f64::from(u32::MAX) {
        return Err(ViprsError::Codec("svg: scaled image is too large".into()));
    }

    let rounded = scaled.round();
    if rounded <= 0.0 {
        return Err(ViprsError::Codec("svg: zero-sized image".into()));
    }

    Ok(rounded as u32)
}

fn raster_size(tree: &usvg::Tree, opts: &LoadOptions) -> Result<(u32, u32, f64), ViprsError> {
    let base_size = tree.size().to_int_size();
    let scale = raster_scale(opts)?;
    let width = scaled_dimension(base_size.width(), scale)?;
    let height = scaled_dimension(base_size.height(), scale)?;
    Ok((width, height, scale))
}

fn ensure_raster_within_limit(width: u32, height: u32) -> Result<(), ViprsError> {
    let bytes = u128::from(width) * u128::from(height) * u128::from(SVG_BANDS);
    if bytes > SVG_MAX_RASTER_BYTES {
        return Err(ViprsError::ImageTooLarge {
            width,
            height,
            bands: SVG_BANDS,
            bytes,
            limit_bytes: SVG_MAX_RASTER_BYTES,
            details: "svg decode still rasterizes eagerly; tiled rendering is not implemented yet",
        });
    }

    Ok(())
}

fn svg_metadata(dpi: f64) -> ImageMetadata {
    let pixels_per_mm = dpi / 25.4;
    ImageMetadata {
        interpretation: Some(Interpretation::Srgb),
        xres: Some(pixels_per_mm),
        yres: Some(pixels_per_mm),
        n_pages: Some(1),
        ..ImageMetadata::default()
    }
}

fn image_from_rgba_pixels<F: BandFormat>(
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    metadata: ImageMetadata,
) -> Result<Image<F>, ViprsError> {
    let samples = bytemuck::allocation::try_cast_vec::<u8, F::Sample>(pixels).map_err(
        |(_err, _pixels)| ViprsError::Codec("svg: sample cast failed (internal error)".into()),
    )?;
    Image::from_buffer(width, height, SVG_BANDS, samples)
        .map(|image| image.with_metadata(metadata))
        .map_err(|err| ViprsError::Codec(err.to_string()))
}

fn sniff_svg_markup(header: &[u8]) -> bool {
    let Ok(text) = std::str::from_utf8(header) else {
        return false;
    };
    text.to_ascii_lowercase().contains("<svg")
}

impl ImageDecoder for SvgDecoder {
    fn format_name(&self) -> &'static str {
        "svg"
    }

    fn sniff(&self, header: &[u8]) -> bool
    where
        Self: Sized,
    {
        sniff_svg_markup(header)
    }

    fn decode<F: BandFormat>(&self, src: &[u8]) -> Result<Image<F>, ViprsError> {
        self.decode_with_options(src, &LoadOptions::default())
    }

    fn decode_with_options<F: BandFormat>(
        &self,
        src: &[u8],
        opts: &LoadOptions,
    ) -> Result<Image<F>, ViprsError>
    where
        Self: Sized,
    {
        require_u8::<F>()?;
        normalize_page_selection(opts)?;

        let tree = parse_svg_tree(src)?;
        let (width, height, scale) = raster_size(&tree, opts)?;
        ensure_raster_within_limit(width, height)?;
        let mut pixmap = tiny_skia::Pixmap::new(width, height)
            .ok_or_else(|| ViprsError::Codec("svg: failed to allocate target pixmap".into()))?;
        let transform = tiny_skia::Transform::from_scale(scale as f32, scale as f32);
        resvg::render(&tree, transform, &mut pixmap.as_mut());

        let dpi = opts.dpi.unwrap_or(DEFAULT_SVG_DPI);
        image_from_rgba_pixels(width, height, pixmap.take_demultiplied(), svg_metadata(dpi))
    }

    fn probe(&self, src: &[u8]) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized,
    {
        let tree = parse_svg_tree(src)?;
        let (width, height, _scale) = raster_size(&tree, &LoadOptions::default())?;
        Ok((width, height, SVG_BANDS))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use viprs_core::format::U8;

    const SOLID_GREEN_SVG: &str = r##"
        <svg xmlns="http://www.w3.org/2000/svg" width="2" height="1">
            <rect width="2" height="1" fill="#00ff00" />
        </svg>
    "##;
    const TWO_COLOR_SVG: &str = r##"
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="10">
            <rect x="0" y="0" width="10" height="10" fill="#ff0000" />
            <rect x="10" y="0" width="10" height="10" fill="#0000ff" />
        </svg>
    "##;

    fn repeated_pixels(width: usize, height: usize, left: [u8; 4], right: [u8; 4]) -> Vec<u8> {
        let mut pixels = Vec::with_capacity(width * height * SVG_BANDS as usize);
        for _ in 0..height {
            for x in 0..width {
                let rgba = if x < width / 2 { left } else { right };
                pixels.extend_from_slice(&rgba);
            }
        }
        pixels
    }

    #[test]
    fn sniff_accepts_svg_markup() {
        assert!(
            SvgDecoder.sniff(br#"<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"/>"#)
        );
        assert!(!SvgDecoder.sniff(b"\x89PNG\r\n\x1a\n"));
    }

    #[test]
    fn decode_inline_svg_rasterizes_to_rgba() {
        let image = SvgDecoder.decode::<U8>(SOLID_GREEN_SVG.as_bytes()).unwrap();

        assert_eq!(image.width(), 2);
        assert_eq!(image.height(), 1);
        assert_eq!(image.bands(), SVG_BANDS);
        assert_eq!(image.metadata().interpretation, Some(Interpretation::Srgb));
        assert_eq!(image.metadata().n_pages, Some(1));
        assert_eq!(image.pixels(), &[0, 255, 0, 255, 0, 255, 0, 255]);
    }

    #[test]
    fn decode_with_dpi_increases_raster_size() {
        let opts = LoadOptions::default().with_dpi(144.0);

        let image = SvgDecoder
            .decode_with_options::<U8>(TWO_COLOR_SVG.as_bytes(), &opts)
            .unwrap();

        assert_eq!(image.width(), 40);
        assert_eq!(image.height(), 20);
        assert_eq!(image.metadata().xres, Some(144.0 / 25.4));
        assert_eq!(image.metadata().yres, Some(144.0 / 25.4));
        assert_eq!(
            image.pixels(),
            repeated_pixels(40, 20, [255, 0, 0, 255], [0, 0, 255, 255])
        );
    }

    #[test]
    fn decode_with_scale_preserves_rendered_split() {
        let opts = LoadOptions::default().with_scale(2.0);

        let image = SvgDecoder
            .decode_with_options::<U8>(TWO_COLOR_SVG.as_bytes(), &opts)
            .unwrap();

        assert_eq!(image.width(), 40);
        assert_eq!(image.height(), 20);
        assert_eq!(image.metadata().xres, Some(DEFAULT_SVG_DPI / 25.4));
        assert_eq!(image.metadata().yres, Some(DEFAULT_SVG_DPI / 25.4));
        assert_eq!(
            image.pixels(),
            repeated_pixels(40, 20, [255, 0, 0, 255], [0, 0, 255, 255])
        );
    }

    #[test]
    fn decode_with_dpi_and_scale_compose_without_cancelling() {
        let opts = LoadOptions::default().with_dpi(144.0).with_scale(1.5);

        let image = SvgDecoder
            .decode_with_options::<U8>(TWO_COLOR_SVG.as_bytes(), &opts)
            .unwrap();

        assert_eq!(image.width(), 60);
        assert_eq!(image.height(), 30);
        assert_eq!(image.metadata().xres, Some(144.0 / 25.4));
        assert_eq!(image.metadata().yres, Some(144.0 / 25.4));
        assert_eq!(
            image.pixels(),
            repeated_pixels(60, 30, [255, 0, 0, 255], [0, 0, 255, 255])
        );
    }

    #[test]
    fn decode_rejects_nonzero_page_selection() {
        let opts = LoadOptions::default().with_page(1);
        let err = SvgDecoder
            .decode_with_options::<U8>(SOLID_GREEN_SVG.as_bytes(), &opts)
            .unwrap_err();

        assert!(err.to_string().contains("page 1"));
    }

    #[test]
    fn probe_reports_default_size() {
        let svg = br#"<svg xmlns="http://www.w3.org/2000/svg" width="3" height="4"/>"#;
        assert_eq!(SvgDecoder.probe(svg).unwrap(), (3, 4, SVG_BANDS));
    }

    #[test]
    fn decode_rejects_huge_canvas_before_allocating_full_raster() {
        let svg = br#"<svg xmlns="http://www.w3.org/2000/svg" width="100000" height="100000"/>"#;
        let err = SvgDecoder.decode::<U8>(svg).unwrap_err();

        match err {
            ViprsError::ImageTooLarge {
                width,
                height,
                bands,
                bytes,
                limit_bytes,
                details,
            } => {
                assert_eq!(width, 100000);
                assert_eq!(height, 100000);
                assert_eq!(bands, SVG_BANDS);
                assert!(bytes > limit_bytes);
                assert_eq!(
                    details,
                    "svg decode still rasterizes eagerly; tiled rendering is not implemented yet"
                );
            }
            other => panic!("expected ImageTooLarge, got {other:?}"),
        }
    }
}

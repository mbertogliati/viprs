//! Text image source adapter.
//!
//! This module exposes concrete source implementations or helpers that feed
//! pixels into compiled pipelines.

use std::{
    fs,
    path::{Path, PathBuf},
};

use ab_glyph::{Font, FontArc, Glyph, PxScale, PxScaleFont, ScaleFont, point};

use crate::{
    domain::{
        error::{TextError, ViprsError},
        format::U8,
        image::{DemandHint, Region},
    },
    ports::source::{ImageSource, RandomAccessSource},
};

use super::common::{clamp_coord, validate_output_len};

const BANDS: u32 = 4;
const DEFAULT_FONT_CANDIDATES: &[&str] = &[
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "C:\\Windows\\Fonts\\arial.ttf",
    "C:\\Windows\\Fonts\\segoeui.ttf",
];

#[derive(Clone)]
struct PositionedGlyph {
    glyph: Glyph,
}

/// Synthetic RGBA text source for watermark and caption overlays.
pub struct TextSource {
    text: String,
    font_size: f32,
    colour: [u8; 4],
    font_path: Option<PathBuf>,
    width: u32,
    height: u32,
    pixels: Box<[u8]>,
}

impl TextSource {
    /// Rasterize a single-line RGBA text image.
    pub fn new<P>(
        text: impl Into<String>,
        font_size: f32,
        colour: [u8; 4],
        font_path: Option<P>,
    ) -> Result<Self, ViprsError>
    where
        P: Into<PathBuf>,
    {
        let text = text.into();
        if text.is_empty() {
            return Err(TextError::EmptyText.into());
        }
        if text.contains('\n') || text.contains('\r') {
            return Err(TextError::MultilineUnsupported.into());
        }
        if !font_size.is_finite() || font_size <= 0.0 {
            return Err(TextError::InvalidFontSize { font_size }.into());
        }

        let requested_font_path = font_path.map(Into::into);
        let resolved_font_path = resolve_font_path(requested_font_path.as_deref())?;
        let font = load_font(&resolved_font_path)?;
        let scaled_font = font.as_scaled(PxScale::from(font_size));
        let glyphs = layout_glyphs(&text, &scaled_font);
        let (width, height, min_x, min_y) = raster_bounds(&glyphs, &scaled_font);
        let pixels = rasterize_text(&scaled_font, &glyphs, width, height, min_x, min_y, colour);

        Ok(Self {
            text,
            font_size,
            colour,
            font_path: Some(resolved_font_path),
            width,
            height,
            pixels,
        })
    }

    /// Source text used during rasterization.
    #[must_use]
    /// `text` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs_runtime::sources::generators::text::text;
    /// ```
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Requested font size in pixels.
    #[must_use]
    pub const fn font_size(&self) -> f32 {
        self.font_size
    }

    /// Straight-alpha RGBA colour applied to every covered glyph pixel.
    #[must_use]
    pub const fn colour(&self) -> [u8; 4] {
        self.colour
    }

    /// Resolved font path used during rasterization.
    #[must_use]
    /// `font_path` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs_runtime::sources::generators::text::font_path;
    /// ```
    pub fn font_path(&self) -> Option<&Path> {
        self.font_path.as_deref()
    }
}

fn resolve_font_path(font_path: Option<&Path>) -> Result<PathBuf, ViprsError> {
    if let Some(path) = font_path {
        return Ok(path.to_path_buf());
    }

    DEFAULT_FONT_CANDIDATES
        .iter()
        .map(Path::new)
        .find(|path| path.is_file())
        .map(Path::to_path_buf)
        .ok_or_else(|| TextError::DefaultFontUnavailable.into())
}

fn load_font(path: &Path) -> Result<FontArc, ViprsError> {
    let data = fs::read(path).map_err(|reason| TextError::FontLoad {
        path: path.display().to_string(),
        reason: reason.to_string(),
    })?;

    FontArc::try_from_vec(data)
        .map_err(|reason| TextError::FontLoad {
            path: path.display().to_string(),
            reason: reason.to_string(),
        })
        .map_err(Into::into)
}

fn layout_glyphs(text: &str, font: &PxScaleFont<&FontArc>) -> Vec<PositionedGlyph> {
    let baseline = font.ascent();
    let mut caret = 0.0f32;
    let mut previous = None;
    let mut glyphs = Vec::with_capacity(text.chars().count());

    for character in text.chars() {
        let glyph_id = font.glyph_id(character);
        if let Some(previous_glyph) = previous {
            caret += font.kern(previous_glyph, glyph_id);
        }

        let mut glyph = font.scaled_glyph(character);
        glyph.position = point(caret, baseline);
        glyphs.push(PositionedGlyph { glyph });
        caret += font.h_advance(glyph_id);
        previous = Some(glyph_id);
    }

    glyphs
}

fn raster_bounds(glyphs: &[PositionedGlyph], font: &PxScaleFont<&FontArc>) -> (u32, u32, i32, i32) {
    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;
    let mut has_outline = false;

    for glyph in glyphs {
        if let Some(outlined) = font.outline_glyph(glyph.glyph.clone()) {
            let bounds = outlined.px_bounds();
            min_x = min_x.min(bounds.min.x.floor() as i32);
            min_y = min_y.min(bounds.min.y.floor() as i32);
            max_x = max_x.max(bounds.max.x.ceil() as i32);
            max_y = max_y.max(bounds.max.y.ceil() as i32);
            has_outline = true;
        }
    }

    if !has_outline {
        let advance_width = glyphs
            .last()
            .map_or(1, |glyph| {
                let glyph_id = glyph.glyph.id;
                (glyph.glyph.position.x + font.h_advance(glyph_id)).ceil() as i32
            })
            .max(1);
        let height = (font.height().ceil() as i32).max(1);
        return (advance_width as u32, height as u32, 0, 0);
    }

    let width = (max_x - min_x).max(1) as u32;
    let height = (max_y - min_y).max(1) as u32;
    (width, height, min_x, min_y)
}

fn rasterize_text(
    font: &PxScaleFont<&FontArc>,
    glyphs: &[PositionedGlyph],
    width: u32,
    height: u32,
    min_x: i32,
    min_y: i32,
    colour: [u8; 4],
) -> Box<[u8]> {
    let mut pixels = vec![0u8; width as usize * height as usize * BANDS as usize];

    for glyph in glyphs {
        let Some(outlined) = font.outline_glyph(glyph.glyph.clone()) else {
            continue;
        };
        let bounds = outlined.px_bounds();
        let origin_x = bounds.min.x.floor() as i32 - min_x;
        let origin_y = bounds.min.y.floor() as i32 - min_y;

        outlined.draw(|x, y, coverage| {
            let dst_x = origin_x + x as i32;
            let dst_y = origin_y + y as i32;
            if dst_x < 0 || dst_y < 0 || dst_x >= width as i32 || dst_y >= height as i32 {
                return;
            }

            let pixel_base = ((dst_y as usize * width as usize) + dst_x as usize) * BANDS as usize;
            pixels[pixel_base] = colour[0];
            pixels[pixel_base + 1] = colour[1];
            pixels[pixel_base + 2] = colour[2];
            pixels[pixel_base + 3] = (f32::from(colour[3]) * coverage).round() as u8;
        });
    }

    pixels.into_boxed_slice()
}

impl ImageSource for TextSource {
    type Format = U8;

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn bands(&self) -> u32 {
        BANDS
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    #[inline]
    fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        validate_output_len(
            region,
            self.bands(),
            std::mem::size_of::<u8>(),
            output,
            self.width,
            self.height,
        )?;

        let region_width = region.width as usize;
        for row in 0..region.height as usize {
            let src_y = clamp_coord(region.y + row as i32, self.height) as usize;
            for col in 0..region_width {
                let src_x = clamp_coord(region.x + col as i32, self.width) as usize;
                let src_base = (src_y * self.width as usize + src_x) * BANDS as usize;
                let dst_base = (row * region_width + col) * BANDS as usize;
                output[dst_base..dst_base + BANDS as usize]
                    .copy_from_slice(&self.pixels[src_base..src_base + BANDS as usize]);
            }
        }

        Ok(())
    }
}

impl RandomAccessSource for TextSource {}

#[cfg(test)]
mod tests {
    use crate::{
        domain::{
            error::{TextError, ViprsError},
            image::Region,
        },
        ports::source::ImageSource,
    };

    use super::TextSource;

    #[test]
    fn rejects_empty_text() {
        let result = TextSource::new("", 24.0, [255, 255, 255, 255], None::<&str>);
        assert!(matches!(
            result,
            Err(ViprsError::Text(TextError::EmptyText))
        ));
    }

    #[test]
    fn rejects_multiline_text() {
        let result = TextSource::new("viprs\ntext", 24.0, [255, 255, 255, 255], None::<&str>);
        assert!(matches!(
            result,
            Err(ViprsError::Text(TextError::MultilineUnsupported))
        ));
    }

    #[test]
    fn renders_rgba_pixels_for_single_line_text() {
        let source =
            TextSource::new("viprs", 28.0, [255, 64, 32, 255], None::<&str>).expect("text source");
        let mut output = vec![0u8; source.width() as usize * source.height() as usize * 4];
        source
            .read_region(
                Region::new(0, 0, source.width(), source.height()),
                &mut output,
            )
            .expect("read text");

        assert_eq!(source.bands(), 4);
        assert!(source.width() > 0);
        assert!(source.height() > 0);
        assert!(output.chunks_exact(4).any(|pixel| pixel[3] > 0));
    }

    #[test]
    fn region_reads_clamp_to_the_raster_edges() {
        let source =
            TextSource::new("viprs", 24.0, [64, 128, 255, 255], None::<&str>).expect("text source");
        let mut output = vec![0u8; 2 * 2 * 4];
        source
            .read_region(Region::new(-2, -2, 2, 2), &mut output)
            .expect("read text");

        let top_left = &output[0..4];
        let repeated = &output[4..8];
        assert_eq!(top_left, repeated);
    }
}

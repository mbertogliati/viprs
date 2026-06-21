//! Pdf Poppler adapter module.
//!
//! This module provides concrete codec-related infrastructure used by the
//! adapter layer for loading, saving, or normalizing external image formats.

#![cfg(feature = "pdf-poppler")]

//! PDF document loader backed by Poppler command-line tools (`pdfinfo`, `pdftoppm`).
//!
//! Decode support only:
//! - options parity: `LoadOptions::page`, `LoadOptions::n`, `LoadOptions::dpi`
//! - multi-page decode via vertical stacking (libvips `pdfload` semantics)
//! - output format: RGBA `U8`

use std::io::Write;
use std::process::{Command, Stdio};

use viprs_core::codec_options::LoadOptions;
use viprs_core::error::ViprsError;
use viprs_core::format::{BandFormat, BandFormatId};
use viprs_core::image::{Image, ImageMetadata, Interpretation};
use viprs_ports::codec::ImageDecoder;

const DEFAULT_DPI: f64 = 72.0;
const PDF_MAGIC_MAX_OFFSET: usize = 32;
const PDF_BANDS: u32 = 4;

#[derive(Debug, Clone, Copy, Default)]
/// The `PdfPopplerDecoder` type provides concrete adapter functionality in the `codecs` module.
/// Use this type when you need the runtime behavior implemented by this adapter.
///
/// # Examples
///
/// ```rust
/// let _ = core::mem::size_of::<viprs_codecs::pdf_poppler::PdfPopplerDecoder>();
/// ```
pub struct PdfPopplerDecoder;

#[derive(Debug)]
struct RenderedPage {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

#[inline]
fn require_u8<F: BandFormat>() -> Result<(), ViprsError> {
    if F::ID != BandFormatId::U8 {
        return Err(ViprsError::Codec(format!(
            "pdf-poppler: unsupported format {:?}; only U8 is supported",
            F::ID
        )));
    }

    Ok(())
}

fn poppler_tool_error(tool: &str, err: &std::io::Error) -> ViprsError {
    ViprsError::Codec(format!(
        "pdf-poppler: unable to run '{tool}': {err}. Install Poppler tools (`pdfinfo`, `pdftoppm`)"
    ))
}

fn run_poppler_tool(tool: &str, args: &[&str], src: &[u8]) -> Result<Vec<u8>, ViprsError> {
    let mut command = Command::new(tool);
    command
        .args(args)
        .env("LC_ALL", "C")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command
        .spawn()
        .map_err(|err| poppler_tool_error(tool, &err))?;

    {
        let stdin = child.stdin.as_mut().ok_or_else(|| {
            ViprsError::Codec(format!(
                "pdf-poppler: failed to open stdin for '{tool}' process"
            ))
        })?;
        stdin.write_all(src).map_err(|err| {
            ViprsError::Codec(format!("pdf-poppler: write to '{tool}' stdin: {err}"))
        })?;
    }

    let output = child.wait_with_output().map_err(|err| {
        ViprsError::Codec(format!("pdf-poppler: waiting on '{tool}' failed: {err}"))
    })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ViprsError::Codec(format!(
            "pdf-poppler: '{tool}' failed: {}",
            stderr.trim()
        )));
    }

    Ok(output.stdout)
}

fn parse_pdf_page_count(pdfinfo_stdout: &[u8]) -> Result<u32, ViprsError> {
    let info = String::from_utf8_lossy(pdfinfo_stdout);
    for line in info.lines() {
        let trimmed = line.trim();
        if let Some(value) = trimmed.strip_prefix("Pages:") {
            let pages = value.trim().parse::<u32>().map_err(|err| {
                ViprsError::Codec(format!(
                    "pdf-poppler: invalid 'Pages' value from pdfinfo ('{value}'): {err}"
                ))
            })?;
            if pages == 0 {
                return Err(ViprsError::Codec(
                    "pdf-poppler: document has zero pages".into(),
                ));
            }
            return Ok(pages);
        }
    }

    Err(ViprsError::Codec(
        "pdf-poppler: unable to parse page count from pdfinfo output".into(),
    ))
}

fn parse_pnm_token<'a>(bytes: &'a [u8], cursor: &mut usize) -> Option<&'a [u8]> {
    while *cursor < bytes.len() {
        let byte = bytes[*cursor];
        if byte.is_ascii_whitespace() {
            *cursor += 1;
            continue;
        }
        if byte == b'#' {
            while *cursor < bytes.len() && bytes[*cursor] != b'\n' {
                *cursor += 1;
            }
            continue;
        }
        break;
    }

    if *cursor >= bytes.len() {
        return None;
    }

    let start = *cursor;
    while *cursor < bytes.len() && !bytes[*cursor].is_ascii_whitespace() {
        *cursor += 1;
    }
    Some(&bytes[start..*cursor])
}

fn parse_ppm(ppm: &[u8]) -> Result<(u32, u32, &[u8]), ViprsError> {
    let mut cursor = 0usize;
    let magic = parse_pnm_token(ppm, &mut cursor)
        .ok_or_else(|| ViprsError::Codec("pdf-poppler: missing PPM magic".into()))?;
    if magic != b"P6" {
        return Err(ViprsError::Codec(format!(
            "pdf-poppler: expected P6 PPM from pdftoppm, got '{}'",
            String::from_utf8_lossy(magic)
        )));
    }

    let width_bytes = parse_pnm_token(ppm, &mut cursor)
        .ok_or_else(|| ViprsError::Codec("pdf-poppler: missing PPM width".into()))?;
    let height_bytes = parse_pnm_token(ppm, &mut cursor)
        .ok_or_else(|| ViprsError::Codec("pdf-poppler: missing PPM height".into()))?;
    let maxval_bytes = parse_pnm_token(ppm, &mut cursor)
        .ok_or_else(|| ViprsError::Codec("pdf-poppler: missing PPM maxval".into()))?;

    let width = std::str::from_utf8(width_bytes)
        .map_err(|err| ViprsError::Codec(format!("pdf-poppler: non-utf8 width token: {err}")))?
        .parse::<u32>()
        .map_err(|err| ViprsError::Codec(format!("pdf-poppler: invalid width token: {err}")))?;
    let height = std::str::from_utf8(height_bytes)
        .map_err(|err| ViprsError::Codec(format!("pdf-poppler: non-utf8 height token: {err}")))?
        .parse::<u32>()
        .map_err(|err| ViprsError::Codec(format!("pdf-poppler: invalid height token: {err}")))?;
    let maxval = std::str::from_utf8(maxval_bytes)
        .map_err(|err| ViprsError::Codec(format!("pdf-poppler: non-utf8 maxval token: {err}")))?
        .parse::<u16>()
        .map_err(|err| ViprsError::Codec(format!("pdf-poppler: invalid maxval token: {err}")))?;

    if maxval != 255 {
        return Err(ViprsError::Codec(format!(
            "pdf-poppler: unsupported maxval {maxval}; expected 255"
        )));
    }

    while cursor < ppm.len() && ppm[cursor].is_ascii_whitespace() {
        cursor += 1;
    }

    let rgb = &ppm[cursor..];
    let expected_len = usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().map(|h| w * h * 3))
        .ok_or_else(|| ViprsError::Codec("pdf-poppler: PPM dimensions overflow".into()))?;
    if rgb.len() != expected_len {
        return Err(ViprsError::Codec(format!(
            "pdf-poppler: malformed PPM payload length {}, expected {}",
            rgb.len(),
            expected_len
        )));
    }

    Ok((width, height, rgb))
}

fn render_page(
    src: &[u8],
    page_index_zero_based: u32,
    dpi: f64,
) -> Result<RenderedPage, ViprsError> {
    let page_number = page_index_zero_based + 1;
    let page_flag = page_number.to_string();
    let dpi_flag = dpi.to_string();
    let args = [
        "-f",
        page_flag.as_str(),
        "-l",
        page_flag.as_str(),
        "-r",
        dpi_flag.as_str(),
        "-singlefile",
        "-",
    ];

    let ppm = run_poppler_tool("pdftoppm", &args, src)?;
    let (width, height, rgb) = parse_ppm(&ppm)?;
    let mut rgba = Vec::with_capacity(rgb.len() / 3 * 4);
    for chunk in rgb.chunks_exact(3) {
        rgba.extend_from_slice(chunk);
        rgba.push(255);
    }

    Ok(RenderedPage {
        width,
        height,
        rgba,
    })
}

fn normalize_page_selection(
    total_pages: u32,
    opts: &LoadOptions,
) -> Result<(u32, u32), ViprsError> {
    let page = opts.page.unwrap_or(0);
    if page >= total_pages {
        return Err(ViprsError::Codec(format!(
            "pdf-poppler: requested page {page}, but file only has {total_pages} page(s)"
        )));
    }

    let remaining = total_pages - page;
    let requested = match opts.n {
        None => 1,
        Some(-1) => remaining,
        Some(value) if value > 0 => u32::try_from(value)
            .map_err(|_| ViprsError::Codec(format!("pdf-poppler: invalid page count {value}")))?,
        Some(value) => {
            return Err(ViprsError::Codec(format!(
                "pdf-poppler: n must be positive or -1, got {value}"
            )));
        }
    };

    Ok((page, requested.min(remaining)))
}

fn resolved_dpi(opts: &LoadOptions) -> Result<f64, ViprsError> {
    let dpi = opts.dpi.unwrap_or(DEFAULT_DPI);
    if !dpi.is_finite() || dpi <= 0.0 {
        return Err(ViprsError::Codec(format!("pdf-poppler: invalid dpi {dpi}")));
    }
    Ok(dpi)
}

fn pdf_metadata(total_pages: u32, dpi: f64, page_height: Option<u32>) -> ImageMetadata {
    let pixels_per_mm = dpi / 25.4;
    ImageMetadata {
        interpretation: Some(Interpretation::Srgb),
        xres: Some(pixels_per_mm),
        yres: Some(pixels_per_mm),
        n_pages: Some(total_pages),
        page_height,
        ..ImageMetadata::default()
    }
}

fn cast_u8_samples<F: BandFormat>(samples: Vec<u8>) -> Result<Vec<F::Sample>, ViprsError> {
    bytemuck::allocation::try_cast_vec::<u8, F::Sample>(samples).map_err(|(_err, _samples)| {
        ViprsError::Codec("pdf-poppler: sample cast failed (internal error)".into())
    })
}

fn decode_pdf<F: BandFormat>(src: &[u8], opts: &LoadOptions) -> Result<Image<F>, ViprsError> {
    require_u8::<F>()?;
    let dpi = resolved_dpi(opts)?;
    let total_pages = parse_pdf_page_count(&run_poppler_tool("pdfinfo", &["-"], src)?)?;
    let (start_page, page_count) = normalize_page_selection(total_pages, opts)?;

    let mut pages = Vec::with_capacity(page_count as usize);
    for offset in 0..page_count {
        pages.push(render_page(src, start_page + offset, dpi)?);
    }

    let max_width = pages.iter().map(|page| page.width).max().unwrap_or(0);
    let total_height = pages
        .iter()
        .fold(0u32, |acc, page| acc.saturating_add(page.height));
    if max_width == 0 || total_height == 0 {
        return Err(ViprsError::Codec(
            "pdf-poppler: rendered output is empty".into(),
        ));
    }

    let mut stacked = vec![0u8; max_width as usize * total_height as usize * PDF_BANDS as usize];
    let mut top = 0u32;
    let mut frames = Vec::with_capacity(pages.len());
    for page in &pages {
        let row_bytes = page.width as usize * PDF_BANDS as usize;
        for row in 0..page.height as usize {
            let dst_row = (top as usize + row) * max_width as usize * PDF_BANDS as usize;
            let src_row = row * row_bytes;
            stacked[dst_row..dst_row + row_bytes]
                .copy_from_slice(&page.rgba[src_row..src_row + row_bytes]);
        }

        let frame_samples = cast_u8_samples::<F>(page.rgba.clone())?;
        let frame = Image::from_buffer(page.width, page.height, PDF_BANDS, frame_samples)
            .map_err(|err| ViprsError::Codec(err.to_string()))?;
        frames.push(frame);
        top += page.height;
    }

    let page_height = (page_count > 1).then_some(pages[0].height);
    let metadata = pdf_metadata(total_pages, dpi, page_height);
    let stacked_samples = cast_u8_samples::<F>(stacked)?;
    Image::from_buffer(max_width, total_height, PDF_BANDS, stacked_samples)
        .map(|image| image.with_metadata(metadata).with_frames(frames))
        .map_err(|err| ViprsError::Codec(err.to_string()))
}

fn sniff_pdf(header: &[u8]) -> bool {
    if header.len() < 4 {
        return false;
    }
    let max_offset = PDF_MAGIC_MAX_OFFSET.min(header.len().saturating_sub(4));
    (0..=max_offset).any(|offset| &header[offset..offset + 4] == b"%PDF")
}

impl ImageDecoder for PdfPopplerDecoder {
    fn format_name(&self) -> &'static str {
        "pdf-poppler"
    }

    fn sniff(&self, header: &[u8]) -> bool
    where
        Self: Sized,
    {
        sniff_pdf(header)
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
        decode_pdf(src, opts)
    }

    fn probe(&self, src: &[u8]) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized,
    {
        let image = self.decode::<viprs_core::format::U8>(src)?;
        Ok((image.width(), image.height(), image.bands()))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use super::*;
    use viprs_core::format::{U8, U16};

    fn solid_rgba(width: usize, height: usize, rgba: [u8; 4]) -> Vec<u8> {
        let mut pixels = Vec::with_capacity(width * height * PDF_BANDS as usize);
        for _ in 0..(width * height) {
            pixels.extend_from_slice(&rgba);
        }
        pixels
    }

    fn stacked_red_then_green() -> Vec<u8> {
        let mut pixels = vec![0u8; 144 * 144 * PDF_BANDS as usize];
        for y in 0..72usize {
            for x in 0..72usize {
                let offset = (y * 144 + x) * PDF_BANDS as usize;
                pixels[offset..offset + PDF_BANDS as usize].copy_from_slice(&[255, 0, 0, 255]);
            }
        }
        for y in 72..144usize {
            for x in 0..144usize {
                let offset = (y * 144 + x) * PDF_BANDS as usize;
                pixels[offset..offset + PDF_BANDS as usize].copy_from_slice(&[0, 255, 0, 255]);
            }
        }
        pixels
    }

    fn poppler_available() -> bool {
        Command::new("pdfinfo")
            .arg("-v")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
            && Command::new("pdftoppm")
                .arg("-v")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .is_ok_and(|status| status.success())
    }

    fn fixture_bytes(name: &str) -> Vec<u8> {
        fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests")
                .join("fixtures")
                .join("images")
                .join(name),
        )
        .unwrap_or_else(|err| panic!("fixture '{name}' missing: {err}"))
    }

    #[test]
    fn sniff_recognises_pdf_magic_with_offset() {
        assert!(PdfPopplerDecoder.sniff(b" \n\t%PDF-1.7"));
        assert!(!PdfPopplerDecoder.sniff(b"\x89PNG\r\n\x1A\n"));
        assert!(!sniff_pdf(b"%P"));
    }

    #[test]
    fn require_u8_rejects_non_u8_formats() {
        let err = require_u8::<U16>().unwrap_err();
        assert!(err.to_string().contains("only U8 is supported"));
    }

    #[test]
    fn parse_pdf_page_count_parses_valid_input() {
        let page_count = parse_pdf_page_count(b"Title: fixture\nPages: 3\n").unwrap();
        assert_eq!(page_count, 3);
    }

    #[test]
    fn parse_pdf_page_count_rejects_zero_and_missing_pages() {
        let zero_pages = parse_pdf_page_count(b"Pages: 0\n").unwrap_err();
        assert!(zero_pages.to_string().contains("zero pages"));

        let missing_pages = parse_pdf_page_count(b"Title: fixture\n").unwrap_err();
        assert!(
            missing_pages
                .to_string()
                .contains("unable to parse page count")
        );

        let invalid_pages = parse_pdf_page_count(b"Pages: nope\n").unwrap_err();
        assert!(invalid_pages.to_string().contains("invalid 'Pages' value"));
    }

    #[test]
    fn parse_pnm_token_skips_comments_and_whitespace() {
        let mut cursor = 0usize;
        let token = parse_pnm_token(b" \n# comment\nP6 2 1 255", &mut cursor).unwrap();
        assert_eq!(token, b"P6");

        let width = parse_pnm_token(b" \n# comment\nP6 2 1 255", &mut cursor).unwrap();
        assert_eq!(width, b"2");
    }

    #[test]
    fn parse_ppm_parses_comment_prefixed_header() {
        let ppm = b"P6\n# generated by test\n2 1\n255\n\xff\x00\x00\x00\xff\x00";
        let (width, height, rgb) = parse_ppm(ppm).unwrap();

        assert_eq!(width, 2);
        assert_eq!(height, 1);
        assert_eq!(rgb, &[255, 0, 0, 0, 255, 0]);
    }

    #[test]
    fn parse_ppm_rejects_bad_magic_and_payload_length() {
        let bad_magic = parse_ppm(b"P3\n1 1\n255\n\xff\x00\x00").unwrap_err();
        assert!(bad_magic.to_string().contains("expected P6"));

        let bad_payload = parse_ppm(b"P6\n1 1\n255\n\xff\x00").unwrap_err();
        assert!(
            bad_payload
                .to_string()
                .contains("malformed PPM payload length")
        );
    }

    #[test]
    fn parse_ppm_rejects_missing_and_invalid_header_tokens() {
        let missing_magic = parse_ppm(b"").unwrap_err();
        assert!(missing_magic.to_string().contains("missing PPM magic"));

        let missing_width = parse_ppm(b"P6\n").unwrap_err();
        assert!(missing_width.to_string().contains("missing PPM width"));

        let invalid_width = parse_ppm(b"P6\nwide 1\n255\n\xff\x00\x00").unwrap_err();
        assert!(invalid_width.to_string().contains("invalid width token"));

        let invalid_height = parse_ppm(b"P6\n1 tall\n255\n\xff\x00\x00").unwrap_err();
        assert!(invalid_height.to_string().contains("invalid height token"));

        let unsupported_maxval = parse_ppm(b"P6\n1 1\n7\n\xff\x00\x00").unwrap_err();
        assert!(
            unsupported_maxval
                .to_string()
                .contains("unsupported maxval 7")
        );
    }

    #[test]
    fn normalize_page_selection_validates_page_and_n() {
        let default_selection = normalize_page_selection(4, &LoadOptions::default()).unwrap();
        assert_eq!(default_selection, (0, 1));

        let all_remaining =
            normalize_page_selection(4, &LoadOptions::default().with_page(1).with_n(-1)).unwrap();
        assert_eq!(all_remaining, (1, 3));

        let clamped =
            normalize_page_selection(4, &LoadOptions::default().with_page(2).with_n(5)).unwrap();
        assert_eq!(clamped, (2, 2));

        let invalid_page =
            normalize_page_selection(1, &LoadOptions::default().with_page(1)).unwrap_err();
        assert!(invalid_page.to_string().contains("requested page 1"));

        let invalid_n = normalize_page_selection(2, &LoadOptions::default().with_n(0)).unwrap_err();
        assert!(invalid_n.to_string().contains("n must be positive or -1"));
    }

    #[test]
    fn resolved_dpi_rejects_non_positive_values() {
        assert_eq!(resolved_dpi(&LoadOptions::default()).unwrap(), DEFAULT_DPI);

        let zero = resolved_dpi(&LoadOptions::default().with_dpi(0.0)).unwrap_err();
        assert!(zero.to_string().contains("invalid dpi 0"));

        let nan = resolved_dpi(&LoadOptions::default().with_dpi(f64::NAN)).unwrap_err();
        assert!(nan.to_string().contains("invalid dpi"));
    }

    #[test]
    fn cast_u8_samples_rejects_misaligned_target_type() {
        let err = cast_u8_samples::<U16>(vec![1, 2, 3]).unwrap_err();
        assert!(err.to_string().contains("sample cast failed"));
    }

    #[test]
    fn decode_single_page_fixture_reports_size_and_metadata() {
        if !poppler_available() {
            eprintln!("skipping pdf-poppler single-page test: poppler tools not available");
            return;
        }

        let bytes = fixture_bytes("pdf-single-page-72pt.pdf");
        let image = PdfPopplerDecoder.decode::<U8>(&bytes).unwrap();
        assert_eq!(image.width(), 72);
        assert_eq!(image.height(), 72);
        assert_eq!(image.bands(), PDF_BANDS);
        assert_eq!(image.metadata().n_pages, Some(1));
        assert_eq!(image.metadata().page_height, None);
        assert_eq!(image.metadata().xres, Some(DEFAULT_DPI / 25.4));
        assert_eq!(image.metadata().yres, Some(DEFAULT_DPI / 25.4));
        assert_eq!(image.pixels(), solid_rgba(72, 72, [0, 0, 255, 255]));
    }

    #[test]
    fn decode_respects_dpi_scaling() {
        if !poppler_available() {
            eprintln!("skipping pdf-poppler dpi test: poppler tools not available");
            return;
        }

        let bytes = fixture_bytes("pdf-single-page-72pt.pdf");
        let image = PdfPopplerDecoder
            .decode_with_options::<U8>(&bytes, &LoadOptions::default().with_dpi(144.0))
            .unwrap();
        assert_eq!(image.width(), 144);
        assert_eq!(image.height(), 144);
        assert_eq!(image.metadata().xres, Some(144.0 / 25.4));
        assert_eq!(image.pixels(), solid_rgba(144, 144, [0, 0, 255, 255]));
    }

    #[test]
    fn decode_respects_page_and_n_and_stacks_vertically() {
        if !poppler_available() {
            eprintln!("skipping pdf-poppler multipage test: poppler tools not available");
            return;
        }

        let bytes = fixture_bytes("pdf-two-pages-72pt-144pt.pdf");
        let first_page = PdfPopplerDecoder
            .decode_with_options::<U8>(&bytes, &LoadOptions::default().with_page(0).with_n(1))
            .unwrap();
        assert_eq!(first_page.width(), 72);
        assert_eq!(first_page.height(), 72);
        assert_eq!(first_page.pixels(), solid_rgba(72, 72, [255, 0, 0, 255]));

        let second_page = PdfPopplerDecoder
            .decode_with_options::<U8>(&bytes, &LoadOptions::default().with_page(1).with_n(1))
            .unwrap();
        assert_eq!(second_page.width(), 144);
        assert_eq!(second_page.height(), 72);
        assert_eq!(second_page.pixels(), solid_rgba(144, 72, [0, 255, 0, 255]));

        let stacked = PdfPopplerDecoder
            .decode_with_options::<U8>(&bytes, &LoadOptions::default().with_page(0).with_n(2))
            .unwrap();
        assert_eq!(stacked.width(), 144);
        assert_eq!(stacked.height(), 144);
        assert_eq!(stacked.metadata().n_pages, Some(2));
        assert_eq!(stacked.metadata().page_height, Some(72));
        assert_eq!(stacked.frames().map(|frames| frames.len()), Some(2));
        assert_eq!(stacked.pixels(), stacked_red_then_green());
        let frames = stacked.frames().expect("stacked PDF must expose frames");
        assert_eq!(frames[0].pixels(), solid_rgba(72, 72, [255, 0, 0, 255]));
        assert_eq!(frames[1].pixels(), solid_rgba(144, 72, [0, 255, 0, 255]));
    }

    #[test]
    fn decode_supports_n_minus_one_until_end() {
        if !poppler_available() {
            eprintln!("skipping pdf-poppler n=-1 test: poppler tools not available");
            return;
        }

        let bytes = fixture_bytes("pdf-two-pages-72pt-144pt.pdf");
        let stacked = PdfPopplerDecoder
            .decode_with_options::<U8>(&bytes, &LoadOptions::default().with_page(1).with_n(-1))
            .unwrap();
        assert_eq!(stacked.width(), 144);
        assert_eq!(stacked.height(), 72);
        assert_eq!(stacked.metadata().n_pages, Some(2));
        assert_eq!(stacked.pixels(), solid_rgba(144, 72, [0, 255, 0, 255]));
    }

    #[test]
    fn decode_rejects_out_of_range_pages() {
        if !poppler_available() {
            eprintln!("skipping pdf-poppler range test: poppler tools not available");
            return;
        }

        let bytes = fixture_bytes("pdf-single-page-72pt.pdf");
        let err = PdfPopplerDecoder
            .decode_with_options::<U8>(&bytes, &LoadOptions::default().with_page(1))
            .unwrap_err();
        assert!(err.to_string().contains("requested page 1"));
    }

    #[test]
    fn probe_reports_dimensions_and_bands() {
        if !poppler_available() {
            eprintln!("skipping pdf-poppler probe test: poppler tools not available");
            return;
        }

        let bytes = fixture_bytes("pdf-single-page-72pt.pdf");
        assert_eq!(
            PdfPopplerDecoder.probe(&bytes).unwrap(),
            (72, 72, PDF_BANDS)
        );
    }
}

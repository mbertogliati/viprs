//! Dcraw adapter module.
//!
//! This module provides concrete codec-related infrastructure used by the
//! adapter layer for loading, saving, or normalizing external image formats.

#![cfg(feature = "dcraw")]

//! Camera RAW loader backed by external `dcraw` / `dcraw_emu` (libraw toolchain).
//!
//! This is a decode-only adapter that mirrors libvips `dcrawload` semantics:
//! camera white balance on, no autorotate during decode, and output bit depth
//! selected by requested band format (`U8` or `U16`).

use std::{
    ffi::{OsStr, OsString},
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

use viprs_core::{
    codec_options::LoadOptions,
    error::ViprsError,
    format::{BandFormat, BandFormatId},
    image::{Image, ImageMetadata, Interpretation},
};
use viprs_ports::codec::ImageDecoder;

/// File extensions recognized as raw camera formats (decoded via dcraw).
pub const DCRAW_EXTENSIONS: &[&str] = &[
    "3fr", "ari", "arw", "cap", "cin", "cr2", "cr3", "crw", "dcr", "dng", "erf", "fff", "iiq",
    "k25", "kdc", "mdc", "mos", "mrw", "nef", "nrw", "orf", "ori", "pef", "pxn", "raf", "raw",
    "rw2", "rwl", "sr2", "srf", "srw", "x3f",
];

const DCRAW_RUNTIME_DIR: &str = "target/dcraw-runtime";
const PNM_ASCII_MAXVAL_U8: u32 = 255;

static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Default)]
/// The `DcrawDecoder` type provides concrete adapter functionality in the `codecs` module.
/// Use this type when you need the runtime behavior implemented by this adapter.
///
/// # Examples
///
/// ```rust
/// let _ = core::mem::size_of::<viprs_codecs::dcraw::DcrawDecoder>();
/// ```
pub struct DcrawDecoder;

#[derive(Clone, Copy)]
enum DcrawBitDepth {
    U8,
    U16,
}

struct ParsedPnmHeader {
    magic: [u8; 2],
    width: u32,
    height: u32,
    maxval: u32,
    data_offset: usize,
}

impl DcrawBitDepth {
    const fn bytes_per_sample(maxval: u32) -> usize {
        if maxval <= PNM_ASCII_MAXVAL_U8 { 1 } else { 2 }
    }
}

fn dcraw_program() -> OsString {
    std::env::var_os("VIPRS_DCRAW_BIN")
        .or_else(|| std::env::var_os("VIPRS_DCRAW_PROGRAM"))
        .unwrap_or_else(|| OsString::from("dcraw"))
}

fn parse_decimal(token: &[u8], field: &str) -> Result<u32, ViprsError> {
    let text = std::str::from_utf8(token)
        .map_err(|_| ViprsError::Codec(format!("dcraw: non-UTF8 token for {field}")))?;
    text.parse::<u32>()
        .map_err(|_| ViprsError::Codec(format!("dcraw: invalid {field} '{text}'")))
}

fn next_pnm_token<'a>(src: &'a [u8], cursor: &mut usize) -> Option<&'a [u8]> {
    while *cursor < src.len() {
        match src[*cursor] {
            byte if byte.is_ascii_whitespace() => *cursor += 1,
            b'#' => {
                while *cursor < src.len() && src[*cursor] != b'\n' {
                    *cursor += 1;
                }
            }
            _ => break,
        }
    }

    if *cursor >= src.len() {
        return None;
    }

    let start = *cursor;
    while *cursor < src.len() {
        let byte = src[*cursor];
        if byte.is_ascii_whitespace() || byte == b'#' {
            break;
        }
        *cursor += 1;
    }

    (start != *cursor).then_some(&src[start..*cursor])
}

fn parse_pnm_header(src: &[u8]) -> Result<ParsedPnmHeader, ViprsError> {
    let mut cursor = 0usize;
    let magic = next_pnm_token(src, &mut cursor)
        .ok_or_else(|| ViprsError::Codec("dcraw: missing PNM magic".into()))?;
    if magic != b"P6" && magic != b"P5" {
        return Err(ViprsError::Codec(format!(
            "dcraw: expected P5/P6 output, got '{}'",
            String::from_utf8_lossy(magic)
        )));
    }

    let width = parse_decimal(
        next_pnm_token(src, &mut cursor)
            .ok_or_else(|| ViprsError::Codec("dcraw: missing PNM width".into()))?,
        "width",
    )?;
    let height = parse_decimal(
        next_pnm_token(src, &mut cursor)
            .ok_or_else(|| ViprsError::Codec("dcraw: missing PNM height".into()))?,
        "height",
    )?;
    let maxval = parse_decimal(
        next_pnm_token(src, &mut cursor)
            .ok_or_else(|| ViprsError::Codec("dcraw: missing PNM maxval".into()))?,
        "maxval",
    )?;

    if width == 0 || height == 0 {
        return Err(ViprsError::Codec(
            "dcraw: decoded output dimensions must be greater than zero".into(),
        ));
    }
    if maxval == 0 || maxval > u16::MAX as u32 {
        return Err(ViprsError::Codec(format!(
            "dcraw: unsupported PNM maxval {maxval}"
        )));
    }

    while cursor < src.len() && src[cursor].is_ascii_whitespace() {
        cursor += 1;
    }
    if cursor >= src.len() {
        return Err(ViprsError::Codec(
            "dcraw: decoded PNM payload missing".into(),
        ));
    }

    Ok(ParsedPnmHeader {
        magic: [magic[0], magic[1]],
        width,
        height,
        maxval,
        data_offset: cursor,
    })
}

fn sample_count(width: u32, height: u32, bands: u32) -> Result<usize, ViprsError> {
    usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
        .and_then(|px| usize::try_from(bands).ok().and_then(|b| px.checked_mul(b)))
        .ok_or_else(|| ViprsError::Codec("dcraw: decoded dimensions overflow".into()))
}

fn decode_pnm_as_u8(stdout: &[u8]) -> Result<(u32, u32, u32, Vec<u8>), ViprsError> {
    let header = parse_pnm_header(stdout)?;
    let bands = if header.magic == *b"P6" { 3 } else { 1 };
    let sample_count = sample_count(header.width, header.height, bands)?;
    let bytes_per_sample = if header.maxval <= PNM_ASCII_MAXVAL_U8 {
        1
    } else {
        2
    };
    let expected_len = sample_count
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| ViprsError::Codec("dcraw: decoded payload length overflow".into()))?;
    let payload = stdout
        .get(header.data_offset..)
        .ok_or_else(|| ViprsError::Codec("dcraw: decoded payload missing".into()))?;
    if payload.len() != expected_len {
        return Err(ViprsError::Codec(format!(
            "dcraw: decoded payload length mismatch, expected {expected_len}, got {}",
            payload.len()
        )));
    }

    if bytes_per_sample == 1 {
        return Ok((header.width, header.height, bands, payload.to_vec()));
    }

    let mut u8_samples = Vec::with_capacity(sample_count);
    for chunk in payload.chunks_exact(2) {
        let value = u16::from_be_bytes([chunk[0], chunk[1]]);
        u8_samples.push((value >> 8) as u8);
    }
    Ok((header.width, header.height, bands, u8_samples))
}

fn decode_pnm_as_u16(stdout: &[u8]) -> Result<(u32, u32, u32, Vec<u16>), ViprsError> {
    let header = parse_pnm_header(stdout)?;
    let bands = if header.magic == *b"P6" { 3 } else { 1 };
    let sample_count = sample_count(header.width, header.height, bands)?;
    let bytes_per_sample = DcrawBitDepth::bytes_per_sample(header.maxval);
    let expected_len = sample_count
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| ViprsError::Codec("dcraw: decoded payload length overflow".into()))?;
    let payload = stdout
        .get(header.data_offset..)
        .ok_or_else(|| ViprsError::Codec("dcraw: decoded payload missing".into()))?;
    if payload.len() != expected_len {
        return Err(ViprsError::Codec(format!(
            "dcraw: decoded payload length mismatch, expected {expected_len}, got {}",
            payload.len()
        )));
    }

    if bytes_per_sample == 1 {
        let mut samples = Vec::with_capacity(sample_count);
        samples.extend(payload.iter().map(|sample| u16::from(*sample) * 257));
        return Ok((header.width, header.height, bands, samples));
    }

    let mut samples = Vec::with_capacity(sample_count);
    for chunk in payload.chunks_exact(2) {
        samples.push(u16::from_be_bytes([chunk[0], chunk[1]]));
    }
    Ok((header.width, header.height, bands, samples))
}

const fn orientation_from_dcraw_flip(flip: i32) -> Option<u8> {
    match flip {
        0 => Some(1),
        3 => Some(3),
        5 => Some(8),
        6 => Some(6),
        _ => None,
    }
}

fn parse_dcraw_metadata(text: &str, bitdepth: DcrawBitDepth, bands: u32) -> ImageMetadata {
    let mut metadata = ImageMetadata {
        interpretation: match (bands, bitdepth) {
            (1, DcrawBitDepth::U16) => Some(Interpretation::Grey16),
            (1, DcrawBitDepth::U8) => Some(Interpretation::BW),
            (3, DcrawBitDepth::U16) => Some(Interpretation::Rgb16),
            (3, DcrawBitDepth::U8) => Some(Interpretation::Srgb),
            _ => None,
        },
        ..ImageMetadata::default()
    };

    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(value) = trimmed.strip_prefix("Camera:") {
            let camera = value.trim();
            if !camera.is_empty() {
                metadata
                    .extra
                    .insert("raw-camera".into(), camera.to_string());
                let mut camera_parts = camera.split_whitespace();
                if let Some(make) = camera_parts.next() {
                    metadata.extra.insert("raw-make".into(), make.to_string());
                    let model = camera_parts.collect::<Vec<_>>().join(" ");
                    if !model.is_empty() {
                        metadata.extra.insert("raw-model".into(), model);
                    }
                }
            }
        } else if let Some(value) = trimmed.strip_prefix("ISO speed:") {
            metadata
                .extra
                .insert("raw-iso".into(), value.trim().to_string());
        } else if let Some(value) = trimmed.strip_prefix("Shutter:") {
            metadata
                .extra
                .insert("raw-shutter".into(), value.trim().to_string());
        } else if let Some(value) = trimmed.strip_prefix("Aperture:") {
            metadata
                .extra
                .insert("raw-aperture".into(), value.trim().to_string());
        } else if let Some(value) = trimmed.strip_prefix("Focal length:") {
            metadata
                .extra
                .insert("raw-focal-length".into(), value.trim().to_string());
        } else if let Some(value) = trimmed.strip_prefix("Timestamp:") {
            metadata
                .extra
                .insert("raw-timestamp".into(), value.trim().to_string());
        } else if let Some(value) = trimmed.strip_prefix("Flip:")
            && let Ok(flip) = value.trim().parse::<i32>()
        {
            metadata.orientation = orientation_from_dcraw_flip(flip);
        }
    }

    metadata
}

fn parse_probe_dimensions(text: &str) -> Option<(u32, u32)> {
    for line in text.lines() {
        let trimmed = line.trim();
        let value = trimmed
            .strip_prefix("Output size:")
            .or_else(|| trimmed.strip_prefix("Image size:"))?;
        let parts = value.split_whitespace().collect::<Vec<_>>();
        if parts.len() < 3 {
            continue;
        }
        let width = parts[0].parse::<u32>().ok()?;
        let height = parts[2].parse::<u32>().ok()?;
        if width > 0 && height > 0 {
            return Some((width, height));
        }
    }

    None
}

fn parse_probe_bands(text: &str) -> u32 {
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(value) = trimmed.strip_prefix("Raw colors:")
            && let Ok(colors) = value.trim().parse::<u32>()
        {
            return if colors > 1 { 3 } else { 1 };
        }
    }

    3
}

fn run_dcraw(program: &OsStr, input_path: &Path, args: &[&str]) -> Result<Vec<u8>, ViprsError> {
    let output = Command::new(program)
        .args(args)
        .arg(input_path)
        .output()
        .map_err(|err| {
            ViprsError::Codec(format!(
                "dcraw: unable to run '{}': {err}. Install dcraw/dcraw_emu and set VIPRS_DCRAW_BIN if needed",
                program.to_string_lossy()
            ))
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ViprsError::Codec(format!(
            "dcraw: command failed ({status}) {args:?}: {stderr}",
            status = output.status
        )));
    }

    Ok(output.stdout)
}

fn run_dcraw_info(program: &OsStr, input_path: &Path) -> Result<String, ViprsError> {
    let output = Command::new(program)
        .args(["-i", "-v"])
        .arg(input_path)
        .output()
        .map_err(|err| {
            ViprsError::Codec(format!(
                "dcraw: unable to run info command '{}': {err}",
                program.to_string_lossy()
            ))
        })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ViprsError::Codec(format!(
            "dcraw: info command failed ({status}): {stderr}",
            status = output.status
        )));
    }

    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    if !output.stderr.is_empty() {
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str(&String::from_utf8_lossy(&output.stderr));
    }
    Ok(text)
}

fn decode_with_dcraw<F: BandFormat>(
    src: &[u8],
    bitdepth: DcrawBitDepth,
) -> Result<Image<F>, ViprsError> {
    let program = dcraw_program();
    with_dcraw_input_file(src, |path| {
        let info = run_dcraw_info(&program, path)?;
        let decode_args = match bitdepth {
            DcrawBitDepth::U8 => vec!["-c", "-w", "-o", "1", "-t", "0"],
            DcrawBitDepth::U16 => vec!["-c", "-w", "-o", "1", "-t", "0", "-6"],
        };
        let stdout = run_dcraw(&program, path, &decode_args)?;

        match bitdepth {
            DcrawBitDepth::U8 => {
                let (width, height, bands, samples) = decode_pnm_as_u8(&stdout)?;
                let metadata = parse_dcraw_metadata(&info, bitdepth, bands);
                let typed = bytemuck::allocation::try_cast_vec::<u8, F::Sample>(samples).map_err(
                    |(_err, _samples)| {
                        ViprsError::Codec(format!(
                            "dcraw: failed to cast decoded U8 samples into {:?}",
                            F::ID
                        ))
                    },
                )?;
                Image::from_buffer(width, height, bands, typed)
                    .map(|image| image.with_metadata(metadata))
            }
            DcrawBitDepth::U16 => {
                let (width, height, bands, samples) = decode_pnm_as_u16(&stdout)?;
                let metadata = parse_dcraw_metadata(&info, bitdepth, bands);
                let typed = bytemuck::allocation::try_cast_vec::<u16, F::Sample>(samples).map_err(
                    |(_err, _samples)| {
                        ViprsError::Codec(format!(
                            "dcraw: failed to cast decoded U16 samples into {:?}",
                            F::ID
                        ))
                    },
                )?;
                Image::from_buffer(width, height, bands, typed)
                    .map(|image| image.with_metadata(metadata))
            }
        }
    })
}

fn probe_with_dcraw(src: &[u8]) -> Result<(u32, u32, u32), ViprsError> {
    let program = dcraw_program();
    with_dcraw_input_file(src, |path| {
        let info = run_dcraw_info(&program, path)?;
        if let Some((width, height)) = parse_probe_dimensions(&info) {
            let bands = parse_probe_bands(&info);
            return Ok((width, height, bands));
        }

        let stdout = run_dcraw(&program, path, &["-c", "-w", "-o", "1", "-t", "0"])?;
        let header = parse_pnm_header(&stdout)?;
        let bands = if header.magic == *b"P6" { 3 } else { 1 };
        Ok((header.width, header.height, bands))
    })
}

fn with_dcraw_input_file<T, F>(src: &[u8], f: F) -> Result<T, ViprsError>
where
    F: FnOnce(&Path) -> Result<T, ViprsError>,
{
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(DCRAW_RUNTIME_DIR);
    fs::create_dir_all(&dir).map_err(|err| {
        ViprsError::Codec(format!(
            "dcraw: failed to create runtime directory '{}': {err}",
            dir.display()
        ))
    })?;

    let counter = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let filename = format!("dcraw-input-{}-{counter}.raw", std::process::id());
    let path: PathBuf = dir.join(filename);

    fs::write(&path, src).map_err(|err| {
        ViprsError::Codec(format!(
            "dcraw: failed to write runtime input '{}': {err}",
            path.display()
        ))
    })?;

    let result = f(&path);
    let _ = fs::remove_file(&path);
    result
}

impl ImageDecoder for DcrawDecoder {
    fn format_name(&self) -> &'static str {
        "dcraw"
    }

    fn sniff(&self, _header: &[u8]) -> bool {
        false
    }

    fn decode<F: BandFormat>(&self, src: &[u8]) -> Result<Image<F>, ViprsError> {
        self.decode_with_options(src, &LoadOptions::default())
    }

    fn decode_with_options<F: BandFormat>(
        &self,
        src: &[u8],
        _opts: &LoadOptions,
    ) -> Result<Image<F>, ViprsError> {
        match F::ID {
            BandFormatId::U8 => decode_with_dcraw(src, DcrawBitDepth::U8),
            BandFormatId::U16 => decode_with_dcraw(src, DcrawBitDepth::U16),
            _ => Err(ViprsError::Codec(format!(
                "dcraw: unsupported output format {:?}; only U8/U16 are supported",
                F::ID
            ))),
        }
    }

    fn probe(&self, src: &[u8]) -> Result<(u32, u32, u32), ViprsError> {
        probe_with_dcraw(src)
    }
}

#[cfg(all(test, feature = "dcraw", unix))]
mod tests {
    use super::*;
    use std::{os::unix::fs::PermissionsExt, sync::Mutex};
    use viprs_core::format::{I16, U8, U16};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn test_dir() -> PathBuf {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("dcraw-codec-tests");
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn fake_dcraw_script(name: &str) -> PathBuf {
        let path = test_dir().join(format!("{name}-{}.sh", std::process::id()));
        let script = r#"#!/bin/sh
set -eu
mode="decode"
bitdepth="8"
for arg in "$@"; do
  if [ "$arg" = "-i" ]; then
    mode="info"
  fi
  if [ "$arg" = "-6" ]; then
    bitdepth="16"
  fi
done
if [ "$mode" = "info" ]; then
  cat <<'EOF'
Filename: fixture.nef
Camera: TESTCAM MODEL42
ISO speed: 200
Shutter: 1/125 sec
Aperture: f/2.8
Focal length: 35.0 mm
Timestamp: Mon Jan  1 00:00:00 2024
Flip: 6
Output size: 2 x 1
EOF
  exit 0
fi
if [ "$bitdepth" = "16" ]; then
  printf 'P6\n2 1\n65535\n'
  printf '\000\020\000\040\000\060\000\100\000\120\000\140'
else
  printf 'P6\n2 1\n255\n'
  printf '\020\040\060\100\120\140'
fi
"#;
        fs::write(&path, script).unwrap();
        let mut perms = fs::metadata(&path).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&path, perms).unwrap();
        path
    }

    fn with_fake_dcraw<T>(script: &Path, run: impl FnOnce() -> T) -> T {
        let _guard = ENV_LOCK.lock().unwrap();
        let key = "VIPRS_DCRAW_BIN";
        let prev = std::env::var_os(key);
        // SAFETY: tests serialize access to process env with ENV_LOCK.
        unsafe {
            std::env::set_var(key, script.as_os_str());
        }
        let result = run();
        if let Some(value) = prev {
            // SAFETY: tests serialize access to process env with ENV_LOCK.
            unsafe {
                std::env::set_var(key, value);
            }
        } else {
            // SAFETY: tests serialize access to process env with ENV_LOCK.
            unsafe {
                std::env::remove_var(key);
            }
        }
        result
    }

    #[test]
    fn decode_u8_via_dcraw_script() {
        let script = fake_dcraw_script("dcraw-u8");
        let src = include_bytes!("../../../tests/fixtures/images/camera-stub.nef");
        let image = with_fake_dcraw(&script, || DcrawDecoder.decode::<U8>(src).unwrap());
        assert_eq!(image.width(), 2);
        assert_eq!(image.height(), 1);
        assert_eq!(image.bands(), 3);
        assert_eq!(image.pixels(), &[16, 32, 48, 64, 80, 96]);
        assert_eq!(
            image.metadata().extra.get("raw-camera").map(String::as_str),
            Some("TESTCAM MODEL42")
        );
        assert_eq!(image.metadata().orientation, Some(6));
    }

    #[test]
    fn decode_u16_via_dcraw_script() {
        let script = fake_dcraw_script("dcraw-u16");
        let src = include_bytes!("../../../tests/fixtures/images/camera-stub.nef");
        let image = with_fake_dcraw(&script, || DcrawDecoder.decode::<U16>(src).unwrap());
        assert_eq!(image.width(), 2);
        assert_eq!(image.height(), 1);
        assert_eq!(image.bands(), 3);
        assert_eq!(image.pixels(), &[16, 32, 48, 64, 80, 96]);
        assert_eq!(image.metadata().interpretation, Some(Interpretation::Rgb16));
    }

    #[test]
    fn probe_uses_dcraw_info_output() {
        let script = fake_dcraw_script("dcraw-probe");
        let src = include_bytes!("../../../tests/fixtures/images/camera-stub.nef");
        let (width, height, bands) = with_fake_dcraw(&script, || DcrawDecoder.probe(src).unwrap());
        assert_eq!((width, height, bands), (2, 1, 3));
    }

    #[test]
    fn reject_non_u8_u16_formats() {
        let script = fake_dcraw_script("dcraw-invalid-format");
        let src = include_bytes!("../../../tests/fixtures/images/camera-stub.nef");
        let err = with_fake_dcraw(&script, || DcrawDecoder.decode::<I16>(src).unwrap_err());
        assert!(err.to_string().contains("only U8/U16 are supported"));
    }

    #[test]
    fn parse_probe_dimensions_handles_standard_lines() {
        assert_eq!(
            parse_probe_dimensions("Output size: 4032 x 3024"),
            Some((4032, 3024))
        );
        assert_eq!(
            parse_probe_dimensions("Image size: 1024 x 768"),
            Some((1024, 768))
        );
        assert_eq!(parse_probe_dimensions("no dimensions"), None);
    }

    #[test]
    fn parse_probe_bands_defaults_to_rgb_and_supports_monochrome() {
        assert_eq!(parse_probe_bands("Raw colors: 3"), 3);
        assert_eq!(parse_probe_bands("Raw colors: 1"), 1);
        assert_eq!(parse_probe_bands("no colors field"), 3);
    }

    #[test]
    fn metadata_parser_maps_flip_to_orientation() {
        let metadata = parse_dcraw_metadata("Flip: 5\nCamera: ACME CAM\n", DcrawBitDepth::U8, 3);
        assert_eq!(metadata.orientation, Some(8));
        assert_eq!(
            metadata.extra.get("raw-camera").map(String::as_str),
            Some("ACME CAM")
        );
    }
}

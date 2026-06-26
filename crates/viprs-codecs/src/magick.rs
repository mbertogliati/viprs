//! Magick adapter module.
//!
//! This module provides concrete codec-related infrastructure used by the
//! adapter layer for loading, saving, or normalizing external image formats.

use std::{
    any::Any,
    ffi::OsString,
    io::Write,
    process::{Command, Stdio},
};

use viprs_core::{
  codec_options::{LoadOptions, SaveOptions},
  error::ViprsError,
  format::{BandFormatId, U8},
  image::InMemoryImage,
};
use viprs_ports::codec::ImageCodec;

/// File extensions decoded via `ImageMagick` fallback when no native codec is available.
pub const MAGICK_FALLBACK_DECODE_EXTENSIONS: &[&str] = &[
    "bmp", "dib", "ico", "icns", "psd", "pcx", "tga", "eps", "ps", "xcf", "dcm",
];

#[derive(Clone, Copy)]
/// The `MagickFallbackSaver` type provides concrete adapter functionality in the `codecs` module.
/// Use this type when you need the runtime behavior implemented by this adapter.
///
/// # Examples
///
/// ```rust
/// let _ = core::mem::size_of::<viprs_codecs::magick::MagickFallbackSaver>();
/// ```
pub struct MagickFallbackSaver {
    format_name: &'static str,
    output_spec: &'static str,
    extensions: &'static [&'static str],
}

impl MagickFallbackSaver {
    const fn new(
        format_name: &'static str,
        output_spec: &'static str,
        extensions: &'static [&'static str],
    ) -> Self {
        Self {
            format_name,
            output_spec,
            extensions,
        }
    }
}

/// Registry of ImageMagick-based format savers for encoding to non-native formats.
pub const MAGICK_FALLBACK_SAVERS: &[MagickFallbackSaver] = &[
    MagickFallbackSaver::new("magick-bmp", "bmp:-", &["bmp", "dib"]),
    MagickFallbackSaver::new("magick-ico", "ico:-", &["ico"]),
    MagickFallbackSaver::new("magick-icns", "icns:-", &["icns"]),
    MagickFallbackSaver::new("magick-psd", "psd:-", &["psd"]),
    MagickFallbackSaver::new("magick-pcx", "pcx:-", &["pcx"]),
    MagickFallbackSaver::new("magick-tga", "tga:-", &["tga"]),
    MagickFallbackSaver::new("magick-eps", "eps:-", &["eps"]),
    MagickFallbackSaver::new("magick-ps", "ps:-", &["ps"]),
    MagickFallbackSaver::new("magick-xcf", "xcf:-", &["xcf"]),
    MagickFallbackSaver::new("magick-dcm", "dcm:-", &["dcm"]),
];

/// The `MagickFallbackLoader` type provides concrete adapter functionality in the `codecs` module.
/// Use this type when you need the runtime behavior implemented by this adapter.
///
/// # Examples
///
/// ```rust
/// let _ = core::mem::size_of::<viprs_codecs::magick::MagickFallbackLoader>();
/// ```
pub struct MagickFallbackLoader;

impl ImageCodec for MagickFallbackLoader {
    fn format_name(&self) -> &'static str {
        "magick-fallback-load"
    }

    fn file_extensions(&self) -> &'static [&'static str] {
        MAGICK_FALLBACK_DECODE_EXTENSIONS
    }

    fn can_encode(&self) -> bool {
        false
    }

    fn supports_extension_decode_fallback(&self) -> bool {
        true
    }

    fn sniff(&self, _header: &[u8]) -> bool {
        false
    }

    fn decode_boxed(
        &self,
        src: &[u8],
        band_format: BandFormatId,
        opts: &LoadOptions,
    ) -> Result<Box<dyn Any + Send>, ViprsError> {
        if band_format != BandFormatId::U8 {
            return Err(ViprsError::Codec(format!(
                "magick fallback decode supports only U8, got {band_format:?}"
            )));
        }

        let pam_bytes = run_magick_decode(src, opts, "pam:-")?;
        let image = parse_pam_u8(&pam_bytes)?;
        Ok(Box::new(image))
    }

    fn encode_boxed(
        &self,
        _image: &(dyn Any + Send + Sync),
        _band_format: BandFormatId,
        _opts: &SaveOptions,
    ) -> Result<Vec<u8>, ViprsError> {
        Err(ViprsError::Codec(
            "magick fallback loader is decode-only".into(),
        ))
    }
}

impl ImageCodec for MagickFallbackSaver {
    fn format_name(&self) -> &'static str {
        self.format_name
    }

    fn file_extensions(&self) -> &'static [&'static str] {
        self.extensions
    }

    fn sniff(&self, _header: &[u8]) -> bool {
        false
    }

    fn decode_boxed(
        &self,
        _src: &[u8],
        _band_format: BandFormatId,
        _opts: &LoadOptions,
    ) -> Result<Box<dyn Any + Send>, ViprsError> {
        Err(ViprsError::Codec(format!(
            "magick fallback saver '{}' is encode-only",
            self.format_name
        )))
    }

    fn encode_boxed(
        &self,
        image: &(dyn Any + Send + Sync),
        band_format: BandFormatId,
        _opts: &SaveOptions,
    ) -> Result<Vec<u8>, ViprsError> {
        if band_format != BandFormatId::U8 {
            return Err(ViprsError::Codec(format!(
                "magick fallback encode supports only U8, got {band_format:?}"
            )));
        }

        let image = image
            .downcast_ref::<InMemoryImage<U8>>()
            .ok_or_else(|| ViprsError::Codec("magick fallback expected Image<U8>".into()))?;
        let pam = build_pam_u8(image);
        run_magick(&pam, self.output_spec)
    }
}

fn magick_program() -> OsString {
    std::env::var_os("VIPRS_MAGICK_BIN").unwrap_or_else(|| OsString::from("magick"))
}

const fn normalize_decode_shrink_factor(factor: u8) -> u8 {
    match factor {
        2 | 4 | 8 => factor,
        _ => 1,
    }
}

fn run_magick(input: &[u8], output_spec: &str) -> Result<Vec<u8>, ViprsError> {
    let program = magick_program();
    run_magick_program(
        &program,
        "convert",
        &[OsString::from("-"), OsString::from(output_spec)],
        input,
    )
}

fn run_magick_decode(
    input: &[u8],
    opts: &LoadOptions,
    output_spec: &str,
) -> Result<Vec<u8>, ViprsError> {
    let program = magick_program();
    let mut args = vec![OsString::from("-")];
    args.extend(magick_decode_resize_args(&program, input, opts)?);
    args.push(OsString::from(output_spec));
    run_magick_program(&program, "convert", &args, input)
}

fn magick_decode_resize_args(
    program: &std::ffi::OsStr,
    input: &[u8],
    opts: &LoadOptions,
) -> Result<Vec<OsString>, ViprsError> {
    if let Some(factor) = opts
        .shrink_factor
        .map(|factor| normalize_decode_shrink_factor(factor.get()))
        .filter(|factor| *factor > 1)
    {
        let (width, height) = probe_magick_dimensions(program, input)?;
        let factor = u32::from(factor);
        let target_width = (width / factor).max(1);
        let target_height = (height / factor).max(1);
        return Ok(vec![
            OsString::from("-resize"),
            OsString::from(format!("{target_width}x{target_height}")),
        ]);
    }

    if let Some(max_dimension) = opts.max_dimension.filter(|value| *value > 0) {
        return Ok(vec![
            OsString::from("-resize"),
            OsString::from(format!("{max_dimension}x{max_dimension}>")),
        ]);
    }

    Ok(Vec::new())
}

fn probe_magick_dimensions(
    program: &std::ffi::OsStr,
    input: &[u8],
) -> Result<(u32, u32), ViprsError> {
    let output = run_magick_program(
        program,
        "identify",
        &[
            OsString::from("-ping"),
            OsString::from("-format"),
            OsString::from("%w %h"),
            OsString::from("-"),
        ],
        input,
    )?;
    let text = std::str::from_utf8(&output)
        .map_err(|_| ViprsError::Codec("magick fallback identify output is not UTF-8".into()))?;
    let mut parts = text.split_whitespace();
    let width = parts
        .next()
        .ok_or_else(|| ViprsError::Codec("magick fallback identify missing width".into()))?
        .parse::<u32>()
        .map_err(|_| ViprsError::Codec("magick fallback identify width is invalid".into()))?;
    let height = parts
        .next()
        .ok_or_else(|| ViprsError::Codec("magick fallback identify missing height".into()))?
        .parse::<u32>()
        .map_err(|_| ViprsError::Codec("magick fallback identify height is invalid".into()))?;
    if parts.next().is_some() {
        return Err(ViprsError::Codec(
            "magick fallback identify returned extra fields".into(),
        ));
    }
    Ok((width, height))
}

fn run_magick_program(
    program: &std::ffi::OsStr,
    subcommand: &str,
    args: &[OsString],
    input: &[u8],
) -> Result<Vec<u8>, ViprsError> {
    let mut child = Command::new(program)
        .arg(subcommand)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| {
            ViprsError::Codec(format!(
                "magick fallback failed to spawn '{}': {err}",
                program.to_string_lossy()
            ))
        })?;

    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| ViprsError::Codec("magick fallback could not open stdin".into()))?;
    stdin
        .write_all(input)
        .map_err(|err| ViprsError::Codec(format!("magick fallback stdin write failed: {err}")))?;
    drop(stdin);

    let output = child
        .wait_with_output()
        .map_err(|err| ViprsError::Codec(format!("magick fallback process wait failed: {err}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ViprsError::Codec(format!(
            "magick fallback {subcommand} failed ({status}): {stderr}",
            status = output.status
        )));
    }

    Ok(output.stdout)
}

fn parse_pam_u8(bytes: &[u8]) -> Result<InMemoryImage<U8>, ViprsError> {
    if !bytes.starts_with(b"P7\n") {
        return Err(ViprsError::Codec(
            "magick fallback decode expected PAM (P7) output".into(),
        ));
    }

    let mut offset = 3usize;
    let mut width = None;
    let mut height = None;
    let mut depth = None;
    let mut maxval = None;

    while offset < bytes.len() {
        let rel_end = bytes[offset..]
            .iter()
            .position(|byte| *byte == b'\n')
            .ok_or_else(|| {
                ViprsError::Codec("magick fallback decode malformed PAM header".into())
            })?;
        let end = offset + rel_end;
        let line = &bytes[offset..end];
        offset = end + 1;

        if line == b"ENDHDR" {
            break;
        }
        if line.is_empty() || line[0] == b'#' {
            continue;
        }

        let mut parts = line.split(|byte| *byte == b' ');
        let key = parts.next().unwrap_or_default();
        let value = parts.next().ok_or_else(|| {
            ViprsError::Codec("magick fallback decode malformed PAM field".into())
        })?;
        if parts.next().is_some() {
            return Err(ViprsError::Codec(
                "magick fallback decode malformed PAM field".into(),
            ));
        }

        match key {
            b"WIDTH" => width = Some(parse_header_u32(value, "WIDTH")?),
            b"HEIGHT" => height = Some(parse_header_u32(value, "HEIGHT")?),
            b"DEPTH" => depth = Some(parse_header_u32(value, "DEPTH")?),
            b"MAXVAL" => maxval = Some(parse_header_u32(value, "MAXVAL")?),
            _ => {}
        }
    }

    let width =
        width.ok_or_else(|| ViprsError::Codec("magick fallback PAM missing WIDTH".into()))?;
    let height =
        height.ok_or_else(|| ViprsError::Codec("magick fallback PAM missing HEIGHT".into()))?;
    let depth =
        depth.ok_or_else(|| ViprsError::Codec("magick fallback PAM missing DEPTH".into()))?;
    let maxval =
        maxval.ok_or_else(|| ViprsError::Codec("magick fallback PAM missing MAXVAL".into()))?;

    if maxval != 255 {
        return Err(ViprsError::Codec(format!(
            "magick fallback PAM MAXVAL must be 255, got {maxval}"
        )));
    }

    let expected_len = usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
        .and_then(|px| usize::try_from(depth).ok().and_then(|d| px.checked_mul(d)))
        .ok_or_else(|| ViprsError::Codec("magick fallback PAM dimensions overflow".into()))?;

    let payload = bytes
        .get(offset..)
        .ok_or_else(|| ViprsError::Codec("magick fallback PAM missing payload".into()))?;
    if payload.len() != expected_len {
        return Err(ViprsError::Codec(format!(
            "magick fallback PAM payload length mismatch: expected {expected_len}, got {}",
            payload.len()
        )));
    }

    InMemoryImage::<U8>::from_buffer(width, height, depth, payload.to_vec())
}

fn parse_header_u32(value: &[u8], key: &str) -> Result<u32, ViprsError> {
    let text = std::str::from_utf8(value)
        .map_err(|_| ViprsError::Codec(format!("magick fallback PAM field {key} is not UTF-8")))?;
    text.parse::<u32>()
        .map_err(|_| ViprsError::Codec(format!("magick fallback PAM field {key} is invalid")))
}

fn build_pam_u8(image: &InMemoryImage<U8>) -> Vec<u8> {
    let tuple_type = match image.bands() {
        1 => "GRAYSCALE",
        2 => "GRAYSCALE_ALPHA",
        3 => "RGB",
        4 => "RGB_ALPHA",
        _ => "MULTIBAND",
    };

    let mut out = format!(
        "P7\nWIDTH {}\nHEIGHT {}\nDEPTH {}\nMAXVAL 255\nTUPLTYPE {tuple_type}\nENDHDR\n",
        image.width(),
        image.height(),
        image.bands()
    )
    .into_bytes();
    out.extend_from_slice(image.pixels());
    out
}

#[cfg(all(test, feature = "magick", unix))]
mod tests {
    use super::*;
    use std::{
        fs,
        path::{Path, PathBuf},
        sync::Mutex,
    };

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn test_dir() -> PathBuf {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("magick-codec-tests");
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn fake_magick_script(name: &str) -> PathBuf {
        let path = test_dir().join(format!("{name}-{}.sh", std::process::id()));
        let script = r#"#!/bin/sh
set -eu
cmd="$1"
shift
last=""
resize=""
prev=""
for arg in "$@"; do
  last="$arg"
  if [ "$prev" = "-resize" ]; then
    resize="$arg"
  fi
  prev="$arg"
done
cat >/dev/null
if [ "$cmd" = "identify" ]; then
  printf '2048 1024'
elif [ "$last" = "pam:-" ] && [ "$resize" = "256x256>" ]; then
  printf 'P7\nWIDTH 256\nHEIGHT 128\nDEPTH 1\nMAXVAL 255\nTUPLTYPE GRAYSCALE\nENDHDR\n'
  dd if=/dev/zero bs=32768 count=1 2>/dev/null
elif [ "$last" = "pam:-" ] && [ "$resize" = "512x256" ]; then
  printf 'P7\nWIDTH 512\nHEIGHT 256\nDEPTH 1\nMAXVAL 255\nTUPLTYPE GRAYSCALE\nENDHDR\n'
  dd if=/dev/zero bs=131072 count=1 2>/dev/null
elif [ "$last" = "pam:-" ]; then
  printf 'P7\nWIDTH 2\nHEIGHT 1\nDEPTH 3\nMAXVAL 255\nTUPLTYPE RGB\nENDHDR\n\001\002\003\004\005\006'
elif [ "$last" = "ico:-" ]; then
  printf 'ICO-BYTES'
elif [ "$last" = "bmp:-" ]; then
  printf 'BMP-BYTES'
else
  printf 'unknown target: %s\n' "$last" >&2
  exit 2
fi
"#;
        fs::write(&path, script).unwrap();
        let mut perms = fs::metadata(&path).unwrap().permissions();
        use std::os::unix::fs::PermissionsExt;
        perms.set_mode(0o755);
        fs::set_permissions(&path, perms).unwrap();
        path
    }

    #[test]
    fn loader_decodes_via_magick_fallback_command() {
        let _guard = ENV_LOCK.lock().unwrap();
        let script = fake_magick_script("decode");

        // SAFETY: tests in this module serialize environment mutation via ENV_LOCK.
        unsafe { std::env::set_var("VIPRS_MAGICK_BIN", &script) };
        let loader = MagickFallbackLoader;
        let decoded = loader
            .decode_boxed(b"input-image", BandFormatId::U8, &LoadOptions::default())
            .unwrap()
            .downcast::<Image<U8>>()
            .unwrap();
        // SAFETY: tests in this module serialize environment mutation via ENV_LOCK.
        unsafe { std::env::remove_var("VIPRS_MAGICK_BIN") };

        assert_eq!(decoded.width(), 2);
        assert_eq!(decoded.height(), 1);
        assert_eq!(decoded.bands(), 3);
        assert_eq!(decoded.pixels(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn loader_applies_max_dimension_resize_hint() {
        let _guard = ENV_LOCK.lock().unwrap();
        let script = fake_magick_script("decode-max-dimension");

        // SAFETY: tests in this module serialize environment mutation via ENV_LOCK.
        unsafe { std::env::set_var("VIPRS_MAGICK_BIN", &script) };
        let loader = MagickFallbackLoader;
        let decoded = loader
            .decode_boxed(
                b"input-image",
                BandFormatId::U8,
                &LoadOptions::default().with_max_dimension(256),
            )
            .unwrap()
            .downcast::<Image<U8>>()
            .unwrap();
        // SAFETY: tests in this module serialize environment mutation via ENV_LOCK.
        unsafe { std::env::remove_var("VIPRS_MAGICK_BIN") };

        assert_eq!((decoded.width(), decoded.height()), (256, 128));
        assert!(decoded.width() <= 256);
        assert!(decoded.height() <= 256);
    }

    #[test]
    fn loader_prefers_shrink_factor_over_max_dimension() {
        let _guard = ENV_LOCK.lock().unwrap();
        let script = fake_magick_script("decode-shrink");

        // SAFETY: tests in this module serialize environment mutation via ENV_LOCK.
        unsafe { std::env::set_var("VIPRS_MAGICK_BIN", &script) };
        let loader = MagickFallbackLoader;
        let decoded = loader
            .decode_boxed(
                b"input-image",
                BandFormatId::U8,
                &LoadOptions::default()
                    .with_max_dimension(256)
                    .with_shrink(std::num::NonZeroU8::new(4).unwrap()),
            )
            .unwrap()
            .downcast::<Image<U8>>()
            .unwrap();
        // SAFETY: tests in this module serialize environment mutation via ENV_LOCK.
        unsafe { std::env::remove_var("VIPRS_MAGICK_BIN") };

        assert_eq!((decoded.width(), decoded.height()), (512, 256));
    }

    #[test]
    fn saver_encodes_via_format_specific_magick_target() {
        let _guard = ENV_LOCK.lock().unwrap();
        let script = fake_magick_script("encode");

        // SAFETY: tests in this module serialize environment mutation via ENV_LOCK.
        unsafe { std::env::set_var("VIPRS_MAGICK_BIN", &script) };
        let saver = MagickFallbackSaver::new("magick-ico", "ico:-", &["ico"]);
        let image = Image::<U8>::from_buffer(1, 1, 3, vec![9, 8, 7]).unwrap();
        let encoded = saver
            .encode_boxed(&image, BandFormatId::U8, &SaveOptions::default())
            .unwrap();
        // SAFETY: tests in this module serialize environment mutation via ENV_LOCK.
        unsafe { std::env::remove_var("VIPRS_MAGICK_BIN") };

        assert_eq!(encoded, b"ICO-BYTES");
    }
}

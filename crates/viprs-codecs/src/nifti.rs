//! NIfTI-1 codec — pure-Rust implementation of the NIfTI-1 image format.
//!
//! Parity target: libvips `niftiload` / `niftisave` in
//! `libvips/foreign/niftiload.c` and `libvips/foreign/niftisave.c`.
//!
//! # Format overview
//!
//! NIfTI-1 consists of:
//! - A 348-byte binary header (compatible with the Analyze 7.5 format).
//! - A 4-byte extension indicator at bytes 348–351.
//! - Optional extension data (length encoded in the extension indicator).
//! - Raw pixel data.
//!
//! Two sub-formats exist:
//! - `.nii` — header and pixel data in one file (magic `"n+1\0"`).
//! - `.hdr` + `.img` — split header/data files (magic `"ni1\0"`).
//!
//! The in-memory codec path handles only the single-file (`.nii`) layout.
//! Path-based decode (`decode_path`) supports both `.nii` and split
//! `.hdr` + `.img` layouts.
//!
//! # Datatype mapping
//!
//! | `NIfTI DT_*` | `viprs` |
//! |------------|----------|
//! | `DT_UINT8`   | `U8`       |
//! | `DT_UINT16`  | `U16`      |
//! | `DT_INT16`   | `I16`      |
//! | `DT_INT32`   | `I32`      |
//! | `DT_UINT32`  | `U32`      |
//! | `DT_FLOAT32` | `F32`      |
//! | `DT_FLOAT64` | `F64`      |
//! | `DT_RGB`     | `U8` × 3 bands |
//! | `DT_RGBA32`  | `U8` × 4 bands |
//!
//! Note: `DT_INT8` is not supported because `viprs` has no `I8` `BandFormat`.
//!
//! # References
//!
//! - NIfTI-1 specification: <https://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h>
//! - libvips parity: `.libvips_repo/libvips/foreign/niftiload.c`

use ::nifti::{InMemNiftiVolume, NiftiObject, ReaderOptions};
use std::path::Path;
use viprs_core::codec_options::{LoadOptions, SaveOptions};
use viprs_core::error::ViprsError;
use viprs_core::format::{BandFormat, BandFormatId};
use viprs_core::image::{Image, ImageMetadata, Interpretation};
use viprs_ports::codec::{ImageDecoder, ImageEncoder};

// ── NIfTI-1 datatype constants ─────────────────────────────────────────────

/// `NIfTI-1` `datatype` field values (subset supported by libvips parity).
///
/// Note: `DT_INT8` (256) is omitted because `viprs` has no `I8` `BandFormat`.
/// `NIfTI` files with `DT_INT8` data will return a `ViprsError::Codec` on decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i16)]
enum NiftiDatatype {
    Uint8 = 2,
    Int16 = 4,
    Int32 = 8,
    Float32 = 16,
    Float64 = 64,
    Uint16 = 512,
    Uint32 = 768,
    Rgb24 = 128,   // DT_RGB — 3 × U8
    Rgba32 = 2304, // DT_RGBA32 — 4 × U8
}

impl NiftiDatatype {
    const fn from_i16(v: i16) -> Option<Self> {
        match v {
            2 => Some(Self::Uint8),
            4 => Some(Self::Int16),
            8 => Some(Self::Int32),
            16 => Some(Self::Float32),
            64 => Some(Self::Float64),
            512 => Some(Self::Uint16),
            768 => Some(Self::Uint32),
            128 => Some(Self::Rgb24),
            2304 => Some(Self::Rgba32),
            _ => None,
        }
    }

    /// Number of bytes per voxel for this datatype.
    const fn bytes_per_voxel(self) -> usize {
        match self {
            Self::Uint8 => 1,
            Self::Int16 | Self::Uint16 => 2,
            Self::Int32 | Self::Uint32 | Self::Float32 | Self::Rgb24 => 4,
            Self::Float64 | Self::Rgba32 => 8,
        }
    }

    /// Number of colour bands for this datatype.
    const fn bands(self) -> u32 {
        match self {
            Self::Rgb24 => 3,
            Self::Rgba32 => 4,
            _ => 1,
        }
    }

    /// The `BandFormatId` corresponding to this datatype.
    const fn band_format_id(self) -> BandFormatId {
        match self {
            Self::Uint8 | Self::Rgb24 | Self::Rgba32 => BandFormatId::U8,
            Self::Uint16 => BandFormatId::U16,
            Self::Int16 => BandFormatId::I16,
            Self::Uint32 => BandFormatId::U32,
            Self::Int32 => BandFormatId::I32,
            Self::Float32 => BandFormatId::F32,
            Self::Float64 => BandFormatId::F64,
        }
    }
}

// ── NIfTI-1 header (348 bytes) ─────────────────────────────────────────────

/// The parsed contents of a NIfTI-1 348-byte header.
#[derive(Debug, Clone)]
struct Nifti1Header {
    /// Image dimensions: `dim[0] = ndim`, `dim[1..7] = size along each axis`.
    dim: [i16; 8],
    /// `NIfTI` `datatype` code.
    datatype: i16,
    /// Byte offset from the start of the file to the pixel data.
    vox_offset: f32,
    /// The magic field (`"n+1\0"` = single-file, `"ni1\0"` = dual-file).
    magic: [u8; 4],
}

impl Nifti1Header {
    /// NIfTI-1 header size in bytes.
    const SIZE: usize = 348;
    const DIM_OFFSET: usize = 40;
    const DATATYPE_OFFSET: usize = 70;
    const VOX_OFFSET_OFFSET: usize = 108;
    const MAGIC_OFFSET: usize = 344;

    fn parse(src: &[u8]) -> Result<Self, ViprsError> {
        if src.len() < Self::SIZE {
            return Err(ViprsError::Codec(format!(
                "nifti: header too short: got {} bytes, need {}",
                src.len(),
                Self::SIZE
            )));
        }

        // Detect byte order from dim[0]: should be 1–7 when correct.
        let dim0_le = i16::from_le_bytes(
            src[Self::DIM_OFFSET..Self::DIM_OFFSET + 2]
                .try_into()
                .map_err(|_| ViprsError::Codec("nifti: dim[0] slice error".into()))?,
        );
        let little_endian = (1..=7).contains(&dim0_le);

        let read_i16 = |offset: usize| -> i16 {
            let bytes: [u8; 2] = src[offset..offset + 2].try_into().unwrap_or([0; 2]);
            if little_endian {
                i16::from_le_bytes(bytes)
            } else {
                i16::from_be_bytes(bytes)
            }
        };

        let read_f32 = |offset: usize| -> f32 {
            let bytes: [u8; 4] = src[offset..offset + 4].try_into().unwrap_or([0; 4]);
            if little_endian {
                f32::from_le_bytes(bytes)
            } else {
                f32::from_be_bytes(bytes)
            }
        };

        let mut dim = [0i16; 8];
        for (index, value) in dim.iter_mut().enumerate() {
            *value = read_i16(Self::DIM_OFFSET + index * 2);
        }

        let datatype = read_i16(Self::DATATYPE_OFFSET);
        let vox_offset = read_f32(Self::VOX_OFFSET_OFFSET);

        let mut magic = [0u8; 4];
        magic.copy_from_slice(&src[Self::MAGIC_OFFSET..Self::MAGIC_OFFSET + 4]);

        Ok(Self {
            dim,
            datatype,
            vox_offset,
            magic,
        })
    }

    fn validate_magic(&self) -> Result<(), ViprsError> {
        if &self.magic == b"n+1\0" || &self.magic == b"ni1\0" {
            Ok(())
        } else {
            Err(ViprsError::Codec(format!(
                "nifti: unrecognised magic {:?}",
                &self.magic
            )))
        }
    }

    const fn ndim(&self) -> i16 {
        self.dim[0]
    }

    const fn width(&self) -> u32 {
        if self.dim[1] > 1 {
            self.dim[1] as u32
        } else {
            1
        }
    }

    /// Height, folding higher spatial dims in as libvips does.
    fn height(&self) -> u32 {
        let ny = self.dim[2].max(1) as u32;
        let ndim = self.ndim();
        if ndim >= 3 {
            let mut h = ny;
            for i in 3..8usize {
                if i <= ndim as usize {
                    let d = self.dim[i].max(1) as u32;
                    h = h.saturating_mul(d);
                }
            }
            h
        } else {
            ny
        }
    }

    /// Byte offset from the beginning of the `src` buffer to pixel data.
    const fn pixel_data_offset(&self) -> usize {
        let off = self.vox_offset as usize;
        // Minimum is 352 (header + 4-byte extension block).
        if off < Self::SIZE + 4 {
            Self::SIZE + 4
        } else {
            off
        }
    }
}

// ── Sniff helper ──────────────────────────────────────────────────────────────

fn sniff_nifti(header: &[u8]) -> bool {
    if header.len() < 348 {
        return false;
    }
    let magic = &header[344..348];
    magic == b"n+1\0"
}

// ── NIfTI-1 encoder helper ────────────────────────────────────────────────────

/// Write a minimal NIfTI-1 header into a 348-byte buffer (little-endian).
fn write_nifti1_header_le(buf: &mut [u8; 348], width: u32, height: u32, dt: NiftiDatatype) {
    // sizeof_hdr = 348
    buf[0..4].copy_from_slice(&348i32.to_le_bytes());

    // dim[0]=2, dim[1]=width, dim[2]=height, rest=1
    buf[40..42].copy_from_slice(&2i16.to_le_bytes());
    buf[42..44].copy_from_slice(&(width as i16).to_le_bytes());
    buf[44..46].copy_from_slice(&(height as i16).to_le_bytes());
    for i in 2usize..7 {
        let off = 40 + (i + 1) * 2;
        buf[off..off + 2].copy_from_slice(&1i16.to_le_bytes());
    }

    // datatype
    let dt_code = dt as i16;
    buf[70..72].copy_from_slice(&dt_code.to_le_bytes());

    // bitpix — bits per voxel
    let bitpix = (dt.bytes_per_voxel() * 8) as i16;
    buf[72..74].copy_from_slice(&bitpix.to_le_bytes());

    // vox_offset = 352.0 (header + 4-byte extension block)
    buf[108..112].copy_from_slice(&352.0f32.to_le_bytes());

    // scl_slope = 1.0, scl_inter = 0.0
    buf[112..116].copy_from_slice(&1.0f32.to_le_bytes());
    buf[116..120].copy_from_slice(&0.0f32.to_le_bytes());

    // magic "n+1\0"
    buf[344..348].copy_from_slice(b"n+1\0");
}

// ── BandFormatId → NiftiDatatype ──────────────────────────────────────────────

fn band_format_to_nifti(id: BandFormatId, bands: u32) -> Result<NiftiDatatype, ViprsError> {
    if id == BandFormatId::U8 {
        return match bands {
            3 => Ok(NiftiDatatype::Rgb24),
            4 => Ok(NiftiDatatype::Rgba32),
            1 => Ok(NiftiDatatype::Uint8),
            _ => Err(ViprsError::Codec(format!(
                "nifti: U8 with {bands} bands is not supported (use 1, 3, or 4)"
            ))),
        };
    }
    if bands != 1 {
        return Err(ViprsError::Codec(format!(
            "nifti: multi-band images are only supported for U8; got {id:?} with {bands} bands"
        )));
    }
    let dt = match id {
        BandFormatId::U16 => NiftiDatatype::Uint16,
        BandFormatId::I16 => NiftiDatatype::Int16,
        BandFormatId::U32 => NiftiDatatype::Uint32,
        BandFormatId::I32 => NiftiDatatype::Int32,
        BandFormatId::F32 => NiftiDatatype::Float32,
        BandFormatId::F64 => NiftiDatatype::Float64,
        BandFormatId::U8 => {
            return Err(ViprsError::Codec(format!(
                "nifti: band format {id:?} cannot be encoded as NIfTI-1"
            )));
        }
    };
    Ok(dt)
}

const fn pick_interpretation(bands: u32, id: BandFormatId) -> Interpretation {
    match (bands, id) {
        (3, BandFormatId::U8) => Interpretation::Srgb,
        (3, BandFormatId::U16 | BandFormatId::I16) => Interpretation::Rgb16,
        (1, BandFormatId::U16 | BandFormatId::I16) => Interpretation::Grey16,
        _ => Interpretation::Multiband,
    }
}

fn ends_with_ignore_ascii_case(name: &str, suffix: &str) -> bool {
    name.len() >= suffix.len() && name[name.len() - suffix.len()..].eq_ignore_ascii_case(suffix)
}

fn nifti_path_supported(path: &Path) -> bool {
    path.file_name()
        .and_then(std::ffi::OsStr::to_str)
        .is_some_and(|name| {
            ends_with_ignore_ascii_case(name, ".nii")
                || ends_with_ignore_ascii_case(name, ".nii.gz")
                || ends_with_ignore_ascii_case(name, ".hdr")
                || ends_with_ignore_ascii_case(name, ".hdr.gz")
        })
}

fn dimensions_from_header(dim: [u16; 8]) -> Result<(u32, u32), ViprsError> {
    let ndim = usize::from(dim[0]);
    if !(1..=7).contains(&ndim) {
        return Err(ViprsError::Codec(format!(
            "nifti: unsupported ndim={ndim}; must be 1–7"
        )));
    }

    let width = u32::from(dim[1].max(1));
    let mut height = u32::from(dim[2].max(1));
    for value in dim.iter().take(ndim + 1).skip(3) {
        height = height.saturating_mul(u32::from((*value).max(1)));
    }

    Ok((width, height))
}

fn cast_samples_to_band_format<T: bytemuck::Pod, F: BandFormat>(
    samples: Vec<T>,
) -> Result<Vec<F::Sample>, ViprsError> {
    bytemuck::allocation::try_cast_vec::<T, F::Sample>(samples).map_err(|(_err, _samples)| {
        ViprsError::Codec(format!(
            "nifti: failed to convert decoded samples into {:?}",
            F::ID
        ))
    })
}

fn decode_path_object<F: BandFormat>(
    header: &::nifti::NiftiHeader,
    volume: InMemNiftiVolume,
) -> Result<Image<F>, ViprsError> {
    let dt = NiftiDatatype::from_i16(header.datatype).ok_or_else(|| {
        ViprsError::Codec(format!("nifti: unsupported datatype={}", header.datatype))
    })?;

    if dt.band_format_id() != F::ID {
        return Err(ViprsError::Codec(format!(
            "nifti: header datatype {:?} maps to {:?}, but caller requested {:?}",
            dt,
            dt.band_format_id(),
            F::ID,
        )));
    }

    let (width, height) = dimensions_from_header(header.dim)?;
    let bands = dt.bands();
    let samples = match dt {
        NiftiDatatype::Uint8 | NiftiDatatype::Rgb24 | NiftiDatatype::Rgba32 => {
            cast_samples_to_band_format::<u8, F>(volume.into_raw_data())?
        }
        NiftiDatatype::Uint16 => cast_samples_to_band_format::<u16, F>(
            volume
                .into_nifti_typed_data::<u16>()
                .map_err(|err| ViprsError::Codec(format!("nifti: {err}")))?,
        )?,
        NiftiDatatype::Int16 => cast_samples_to_band_format::<i16, F>(
            volume
                .into_nifti_typed_data::<i16>()
                .map_err(|err| ViprsError::Codec(format!("nifti: {err}")))?,
        )?,
        NiftiDatatype::Uint32 => cast_samples_to_band_format::<u32, F>(
            volume
                .into_nifti_typed_data::<u32>()
                .map_err(|err| ViprsError::Codec(format!("nifti: {err}")))?,
        )?,
        NiftiDatatype::Int32 => cast_samples_to_band_format::<i32, F>(
            volume
                .into_nifti_typed_data::<i32>()
                .map_err(|err| ViprsError::Codec(format!("nifti: {err}")))?,
        )?,
        NiftiDatatype::Float32 => cast_samples_to_band_format::<f32, F>(
            volume
                .into_nifti_typed_data::<f32>()
                .map_err(|err| ViprsError::Codec(format!("nifti: {err}")))?,
        )?,
        NiftiDatatype::Float64 => cast_samples_to_band_format::<f64, F>(
            volume
                .into_nifti_typed_data::<f64>()
                .map_err(|err| ViprsError::Codec(format!("nifti: {err}")))?,
        )?,
    };

    let metadata = ImageMetadata {
        interpretation: Some(pick_interpretation(bands, F::ID)),
        ..ImageMetadata::default()
    };

    Image::from_buffer(width, height, bands, samples)
        .map(|img| img.with_metadata(metadata))
        .map_err(|err| ViprsError::Codec(err.to_string()))
}

// ── Codec ─────────────────────────────────────────────────────────────────────

/// NIfTI-1 codec backed by pure-Rust parsing (`nifti-rs` for path decode).
///
/// In-memory decode supports the single-file (`.nii`) layout. Path-based decode
/// supports both `.nii` and split `.hdr`/`.img` layouts.
pub struct NiftiCodec;

impl NiftiCodec {
    /// Decode a NIfTI-1 byte stream.
    ///
    /// `F` must match the `datatype` field in the header.
    ///
    /// # Errors
    ///
    /// Returns [`ViprsError::Codec`] when the buffer is too short, the magic
    /// field is invalid, the datatype is unsupported, or `F::ID` does not
    /// match the header's datatype.
    pub fn decode_nifti<F: BandFormat>(&self, src: &[u8]) -> Result<Image<F>, ViprsError> {
        let hdr = Nifti1Header::parse(src)?;
        hdr.validate_magic()?;
        if &hdr.magic == b"ni1\0" {
            return Err(ViprsError::Codec(
                "nifti: split-file headers require decode_path()/probe_path()".into(),
            ));
        }

        let ndim = hdr.ndim();
        if !(1..=7).contains(&ndim) {
            return Err(ViprsError::Codec(format!(
                "nifti: unsupported ndim={ndim}; must be 1–7"
            )));
        }

        let dt = NiftiDatatype::from_i16(hdr.datatype).ok_or_else(|| {
            ViprsError::Codec(format!("nifti: unsupported datatype={}", hdr.datatype))
        })?;

        if dt.band_format_id() != F::ID {
            return Err(ViprsError::Codec(format!(
                "nifti: header datatype {:?} maps to {:?}, but caller requested {:?}",
                dt,
                dt.band_format_id(),
                F::ID,
            )));
        }

        let width = hdr.width();
        let height = hdr.height();
        let bands = dt.bands();

        let pixel_offset = hdr.pixel_data_offset();
        if pixel_offset > src.len() {
            return Err(ViprsError::Codec(format!(
                "nifti: vox_offset {} exceeds buffer length {}",
                pixel_offset,
                src.len()
            )));
        }

        let pixel_bytes = &src[pixel_offset..];
        let expected = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(bands as usize)
            .saturating_mul(std::mem::size_of::<F::Sample>());

        if pixel_bytes.len() < expected {
            return Err(ViprsError::Codec(format!(
                "nifti: pixel data too short: need {expected} bytes, got {}",
                pixel_bytes.len()
            )));
        }

        // SAFETY: F::Sample is Pod (BandFormat bound), re-allocated into Vec
        // so alignment is correct; length is a multiple of size_of::<F::Sample>.
        let samples: Vec<F::Sample> =
            bytemuck::cast_slice::<u8, F::Sample>(&pixel_bytes[..expected]).to_vec();

        let interp = pick_interpretation(bands, F::ID);
        let metadata = ImageMetadata {
            interpretation: Some(interp),
            ..ImageMetadata::default()
        };

        Image::from_buffer(width, height, bands, samples)
            .map(|img| img.with_metadata(metadata))
            .map_err(|err| ViprsError::Codec(err.to_string()))
    }
}

impl ImageDecoder for NiftiCodec {
    fn format_name(&self) -> &'static str {
        "nifti"
    }

    fn can_decode_path(&self, path: &Path) -> bool {
        nifti_path_supported(path)
    }

    fn sniff(&self, header: &[u8]) -> bool
    where
        Self: Sized,
    {
        sniff_nifti(header)
    }

    fn decode<F: BandFormat>(&self, src: &[u8]) -> Result<Image<F>, ViprsError> {
        self.decode_nifti(src)
    }

    fn decode_path<F: BandFormat>(&self, path: &Path) -> Result<Image<F>, ViprsError>
    where
        Self: Sized,
    {
        self.decode_path_with_options(path, &LoadOptions::default())
    }

    fn decode_with_options<F: BandFormat>(
        &self,
        src: &[u8],
        _opts: &LoadOptions,
    ) -> Result<Image<F>, ViprsError>
    where
        Self: Sized,
    {
        self.decode_nifti(src)
    }

    fn decode_path_with_options<F: BandFormat>(
        &self,
        path: &Path,
        _opts: &LoadOptions,
    ) -> Result<Image<F>, ViprsError>
    where
        Self: Sized,
    {
        let obj = ReaderOptions::new()
            .read_file(path)
            .map_err(|err| ViprsError::Codec(format!("nifti: {err}")))?;
        let header = obj.header().clone();
        let volume = obj.into_volume();
        decode_path_object::<F>(&header, volume)
    }

    fn probe(&self, src: &[u8]) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized,
    {
        let hdr = Nifti1Header::parse(src)?;
        hdr.validate_magic()?;
        if &hdr.magic == b"ni1\0" {
            return Err(ViprsError::Codec(
                "nifti: split-file headers require decode_path()/probe_path()".into(),
            ));
        }
        let dt = NiftiDatatype::from_i16(hdr.datatype).ok_or_else(|| {
            ViprsError::Codec(format!("nifti: unsupported datatype={}", hdr.datatype))
        })?;
        Ok((hdr.width(), hdr.height(), dt.bands()))
    }

    fn probe_path(&self, path: &Path) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized,
    {
        let obj = ReaderOptions::new()
            .read_file(path)
            .map_err(|err| ViprsError::Codec(format!("nifti: {err}")))?;
        let header = obj.header();
        let dt = NiftiDatatype::from_i16(header.datatype).ok_or_else(|| {
            ViprsError::Codec(format!("nifti: unsupported datatype={}", header.datatype))
        })?;
        let (width, height) = dimensions_from_header(header.dim)?;
        Ok((width, height, dt.bands()))
    }
}

impl ImageEncoder for NiftiCodec {
    fn format_name(&self) -> &'static str {
        "nifti"
    }

    fn encode<F: BandFormat>(&self, image: &Image<F>) -> Result<Vec<u8>, ViprsError> {
        let dt = band_format_to_nifti(F::ID, image.bands())?;

        let mut header = [0u8; 348];
        write_nifti1_header_le(&mut header, image.width(), image.height(), dt);

        let ext_block = [0u8; 4];
        let pixel_bytes = bytemuck::cast_slice::<F::Sample, u8>(image.pixels());

        let mut out = Vec::with_capacity(348 + 4 + pixel_bytes.len());
        out.extend_from_slice(&header);
        out.extend_from_slice(&ext_block);
        out.extend_from_slice(pixel_bytes);
        Ok(out)
    }

    fn encode_with_options<F: BandFormat>(
        &self,
        image: &Image<F>,
        _opts: &SaveOptions,
    ) -> Result<Vec<u8>, ViprsError>
    where
        Self: Sized,
    {
        self.encode(image)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        path::{Path, PathBuf},
        time::{SystemTime, UNIX_EPOCH},
    };
    use viprs_core::format::{F32, F64, I16, I32, U8, U16};

    // ── Header-building helper ─────────────────────────────────────────────

    fn make_nifti_header_with_layout(
        width: u16,
        height: u16,
        datatype: i16,
        bitpix: i16,
        magic: &[u8; 4],
        vox_offset: f32,
    ) -> Vec<u8> {
        let mut buf = vec![0u8; 352]; // header + 4-byte extension block

        buf[0..4].copy_from_slice(&348i32.to_le_bytes());

        buf[40..42].copy_from_slice(&2i16.to_le_bytes()); // ndim = 2
        buf[42..44].copy_from_slice(&(width as i16).to_le_bytes());
        buf[44..46].copy_from_slice(&(height as i16).to_le_bytes());
        for i in 2usize..7 {
            let off = 40 + (i + 1) * 2;
            buf[off..off + 2].copy_from_slice(&1i16.to_le_bytes());
        }

        buf[70..72].copy_from_slice(&datatype.to_le_bytes());
        buf[72..74].copy_from_slice(&bitpix.to_le_bytes());
        buf[108..112].copy_from_slice(&vox_offset.to_le_bytes());
        buf[344..348].copy_from_slice(magic);

        buf
    }

    fn make_nifti_header(width: u16, height: u16, datatype: i16, bitpix: i16) -> Vec<u8> {
        make_nifti_header_with_layout(width, height, datatype, bitpix, b"n+1\0", 352.0)
    }

    fn make_nifti_stream<S: bytemuck::Pod>(
        width: u16,
        height: u16,
        datatype: i16,
        bitpix: i16,
        pixels: &[S],
    ) -> Vec<u8> {
        let mut buf = make_nifti_header(width, height, datatype, bitpix);
        buf.extend_from_slice(bytemuck::cast_slice(pixels));
        buf
    }

    struct TestDir {
        path: PathBuf,
    }

    impl TestDir {
        fn new(prefix: &str) -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let path = std::env::current_dir()
                .unwrap_or_else(|err| panic!("current_dir failed: {err}"))
                .join(format!(".viprs-{prefix}-{unique}"));
            fs::create_dir_all(&path)
                .unwrap_or_else(|err| panic!("create_dir_all({}): {err}", path.display()));
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    // ── sniff ──────────────────────────────────────────────────────────────

    #[test]
    fn sniff_recognises_n_plus_1_magic() {
        let hdr = make_nifti_header(4, 4, 2, 8);
        assert!(NiftiCodec.sniff(&hdr));
    }

    #[test]
    fn sniff_rejects_ni1_magic() {
        let hdr = make_nifti_header_with_layout(4, 4, 2, 8, b"ni1\0", 0.0);
        assert!(!NiftiCodec.sniff(&hdr));
    }

    #[test]
    fn sniff_rejects_non_nifti_bytes() {
        let mut buf = vec![0u8; 352];
        buf[344..348].copy_from_slice(b"JUNK");
        assert!(!NiftiCodec.sniff(&buf));
    }

    #[test]
    fn sniff_rejects_short_buffer() {
        assert!(!NiftiCodec.sniff(&[0u8; 100]));
    }

    // ── probe ──────────────────────────────────────────────────────────────

    #[test]
    fn probe_returns_correct_dimensions_u8() {
        let hdr = make_nifti_header(16, 8, 2, 8);
        assert_eq!(NiftiCodec.probe(&hdr).unwrap(), (16, 8, 1));
    }

    #[test]
    fn probe_returns_3_bands_for_rgb() {
        let hdr = make_nifti_header(4, 4, 128, 24);
        assert_eq!(NiftiCodec.probe(&hdr).unwrap(), (4, 4, 3));
    }

    #[test]
    fn probe_returns_4_bands_for_rgba() {
        let hdr = make_nifti_header(2, 2, 2304, 32);
        assert_eq!(NiftiCodec.probe(&hdr).unwrap(), (2, 2, 4));
    }

    // ── decode ─────────────────────────────────────────────────────────────

    #[test]
    fn decode_u8_greyscale() {
        let pixels: Vec<u8> = (0u8..16).collect();
        let stream = make_nifti_stream::<u8>(4, 4, 2, 8, &pixels);
        let img = NiftiCodec.decode::<U8>(&stream).unwrap();
        assert_eq!((img.width(), img.height(), img.bands()), (4, 4, 1));
        assert_eq!(img.pixels(), pixels.as_slice());
    }

    #[test]
    fn decode_u8_rgb() {
        let pixels: Vec<u8> = (0u8..48).collect();
        let stream = make_nifti_stream::<u8>(4, 4, 128, 24, &pixels);
        let img = NiftiCodec.decode::<U8>(&stream).unwrap();
        assert_eq!(img.bands(), 3);
        assert_eq!(img.pixels(), pixels.as_slice());
    }

    #[test]
    fn decode_f32() {
        let pixels: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let stream = make_nifti_stream::<f32>(4, 4, 16, 32, &pixels);
        let img = NiftiCodec.decode::<F32>(&stream).unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());
    }

    #[test]
    fn decode_i16() {
        let pixels: Vec<i16> = (0..9).map(|i| i as i16 - 4).collect();
        let stream = make_nifti_stream::<i16>(3, 3, 4, 16, &pixels);
        let img = NiftiCodec.decode::<I16>(&stream).unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());
    }

    #[test]
    fn decode_u16_greyscale() {
        let pixels: Vec<u16> = (0u16..4).collect();
        let stream = make_nifti_stream::<u16>(2, 2, 512, 16, &pixels);
        let img = NiftiCodec.decode::<U16>(&stream).unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());
    }

    #[test]
    fn decode_i32() {
        let pixels: Vec<i32> = vec![-1, 0, 1, 2];
        let stream = make_nifti_stream::<i32>(2, 2, 8, 32, &pixels);
        let img = NiftiCodec.decode::<I32>(&stream).unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());
    }

    #[test]
    fn decode_f64() {
        let pixels: Vec<f64> = (0..4).map(|i| i as f64 * 0.25).collect();
        let stream = make_nifti_stream::<f64>(2, 2, 64, 64, &pixels);
        let img = NiftiCodec.decode::<F64>(&stream).unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());
    }

    // ── decode error paths ─────────────────────────────────────────────────

    #[test]
    fn error_on_wrong_format() {
        let pixels = vec![0u8; 16];
        let stream = make_nifti_stream::<u8>(4, 4, 2, 8, &pixels);
        let err = NiftiCodec.decode::<F32>(&stream).unwrap_err();
        assert!(
            err.to_string().contains("datatype") || err.to_string().contains("Uint8"),
            "expected format mismatch error, got: {err}"
        );
    }

    #[test]
    fn error_on_header_too_short() {
        let err = NiftiCodec.decode::<U8>(&[0u8; 100]).unwrap_err();
        assert!(err.to_string().contains("too short"), "{err}");
    }

    #[test]
    fn error_on_bad_magic() {
        let mut hdr = make_nifti_header(4, 4, 2, 8);
        hdr[344..348].copy_from_slice(b"JUNK");
        hdr.extend_from_slice(&[0u8; 16]);
        let err = NiftiCodec.decode::<U8>(&hdr).unwrap_err();
        assert!(err.to_string().contains("magic"), "{err}");
    }

    #[test]
    fn error_on_pixel_data_too_short() {
        let stream = make_nifti_stream::<u8>(4, 4, 2, 8, &[0u8; 8]); // 4×4 needs 16
        let err = NiftiCodec.decode::<U8>(&stream).unwrap_err();
        assert!(
            err.to_string().contains("too short") || err.to_string().contains("need"),
            "{err}"
        );
    }

    // ── round-trip encode → decode ─────────────────────────────────────────

    #[test]
    fn round_trip_u8_greyscale() {
        let pixels: Vec<u8> = (0u8..64).collect();
        let image = Image::<U8>::from_buffer(8, 8, 1, pixels.clone()).unwrap();
        let encoded = NiftiCodec.encode(&image).unwrap();
        assert!(NiftiCodec.sniff(&encoded));
        let decoded = NiftiCodec.decode::<U8>(&encoded).unwrap();
        assert_eq!(
            (decoded.width(), decoded.height(), decoded.bands()),
            (8, 8, 1)
        );
        assert_eq!(decoded.pixels(), pixels.as_slice());
    }

    #[test]
    fn round_trip_f32() {
        let pixels: Vec<f32> = (0..16).map(|i| i as f32 / 15.0).collect();
        let image = Image::<F32>::from_buffer(4, 4, 1, pixels.clone()).unwrap();
        let encoded = NiftiCodec.encode(&image).unwrap();
        let decoded = NiftiCodec.decode::<F32>(&encoded).unwrap();
        assert_eq!(decoded.pixels(), pixels.as_slice());
    }

    #[test]
    fn round_trip_u16() {
        let pixels: Vec<u16> = (0u16..16).collect();
        let image = Image::<U16>::from_buffer(4, 4, 1, pixels.clone()).unwrap();
        let encoded = NiftiCodec.encode(&image).unwrap();
        let decoded = NiftiCodec.decode::<U16>(&encoded).unwrap();
        assert_eq!(decoded.pixels(), pixels.as_slice());
    }

    #[test]
    fn round_trip_f64() {
        let pixels: Vec<f64> = (0..4).map(|i| i as f64 * 0.1).collect();
        let image = Image::<F64>::from_buffer(2, 2, 1, pixels.clone()).unwrap();
        let encoded = NiftiCodec.encode(&image).unwrap();
        let decoded = NiftiCodec.decode::<F64>(&encoded).unwrap();
        assert_eq!(decoded.pixels(), pixels.as_slice());
    }

    #[test]
    fn round_trip_i32() {
        let pixels: Vec<i32> = vec![-100, 0, 100, i32::MAX];
        let image = Image::<I32>::from_buffer(2, 2, 1, pixels.clone()).unwrap();
        let encoded = NiftiCodec.encode(&image).unwrap();
        let decoded = NiftiCodec.decode::<I32>(&encoded).unwrap();
        assert_eq!(decoded.pixels(), pixels.as_slice());
    }

    #[test]
    fn round_trip_rgb_u8() {
        let pixels: Vec<u8> = (0u8..48).collect(); // 4×4 RGB
        let image = Image::<U8>::from_buffer(4, 4, 3, pixels.clone()).unwrap();
        let encoded = NiftiCodec.encode(&image).unwrap();
        let decoded = NiftiCodec.decode::<U8>(&encoded).unwrap();
        assert_eq!(decoded.bands(), 3);
        assert_eq!(decoded.pixels(), pixels.as_slice());
    }

    #[test]
    fn round_trip_rgba_u8() {
        let pixels: Vec<u8> = (0u8..64).collect(); // 4×4 RGBA
        let image = Image::<U8>::from_buffer(4, 4, 4, pixels.clone()).unwrap();
        let encoded = NiftiCodec.encode(&image).unwrap();
        let decoded = NiftiCodec.decode::<U8>(&encoded).unwrap();
        assert_eq!(decoded.bands(), 4);
        assert_eq!(decoded.pixels(), pixels.as_slice());
    }

    #[test]
    fn decode_path_round_trip_single_file_f32() {
        let dir = TestDir::new("nifti-single");
        let path = dir.path().join("single.nii");
        let pixels: Vec<f32> = (0..100).map(|i| i as f32 * 0.25).collect();
        let image = Image::<F32>::from_buffer(10, 10, 1, pixels.clone()).unwrap();

        fs::write(&path, NiftiCodec.encode(&image).unwrap())
            .unwrap_or_else(|err| panic!("write {} failed: {err}", path.display()));

        assert!(NiftiCodec.can_decode_path(&path));
        let decoded = NiftiCodec.decode_path::<F32>(&path).unwrap();
        assert_eq!(
            (decoded.width(), decoded.height(), decoded.bands()),
            (10, 10, 1)
        );
        assert_eq!(decoded.pixels(), pixels.as_slice());
    }

    #[test]
    fn decode_path_round_trip_split_header_image_f32() {
        let dir = TestDir::new("nifti-split");
        let hdr_path = dir.path().join("split.hdr");
        let img_path = dir.path().join("split.img");
        let pixels: Vec<f32> = (0..100).map(|i| i as f32 / 10.0).collect();

        let header = make_nifti_header_with_layout(10, 10, 16, 32, b"ni1\0", 0.0);
        fs::write(&hdr_path, &header)
            .unwrap_or_else(|err| panic!("write {} failed: {err}", hdr_path.display()));
        fs::write(&img_path, bytemuck::cast_slice::<f32, u8>(&pixels))
            .unwrap_or_else(|err| panic!("write {} failed: {err}", img_path.display()));

        assert!(NiftiCodec.can_decode_path(&hdr_path));
        assert!(!NiftiCodec.can_decode_path(&img_path));
        assert_eq!(NiftiCodec.probe_path(&hdr_path).unwrap(), (10, 10, 1));

        let decoded = NiftiCodec.decode_path::<F32>(&hdr_path).unwrap();
        assert_eq!(
            (decoded.width(), decoded.height(), decoded.bands()),
            (10, 10, 1)
        );
        assert_eq!(decoded.pixels(), pixels.as_slice());
    }

    // ── encode error paths ─────────────────────────────────────────────────

    #[test]
    fn error_on_encode_u8_with_2_bands() {
        let image = Image::<U8>::from_buffer(2, 2, 2, vec![0u8; 8]).unwrap();
        let err = NiftiCodec.encode(&image).unwrap_err();
        assert!(err.to_string().contains("bands"), "{err}");
    }

    // ── proptest: identity round-trip ──────────────────────────────────────

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_round_trip_u8_greyscale(
            width in 1u16..=64,
            height in 1u16..=64,
        ) {
            let count = (width as usize) * (height as usize);
            let pixels: Vec<u8> = (0..count).map(|i| (i % 256) as u8).collect();
            let image = Image::<U8>::from_buffer(width as u32, height as u32, 1, pixels.clone()).unwrap();
            let encoded = NiftiCodec.encode(&image).unwrap();
            let decoded = NiftiCodec.decode::<U8>(&encoded).unwrap();
            prop_assert_eq!(decoded.width(), width as u32);
            prop_assert_eq!(decoded.height(), height as u32);
            prop_assert_eq!(decoded.pixels(), pixels.as_slice());
        }

        #[test]
        fn prop_round_trip_f32(width in 1u16..=32, height in 1u16..=32) {
            let count = (width as usize) * (height as usize);
            let pixels: Vec<f32> = (0..count).map(|i| i as f32).collect();
            let image = Image::<F32>::from_buffer(width as u32, height as u32, 1, pixels.clone()).unwrap();
            let encoded = NiftiCodec.encode(&image).unwrap();
            let decoded = NiftiCodec.decode::<F32>(&encoded).unwrap();
            prop_assert_eq!(decoded.pixels(), pixels.as_slice());
        }
    }
}

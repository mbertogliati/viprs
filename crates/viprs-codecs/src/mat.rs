//! MATLAB MAT codec:
//! - v5: pure-Rust decoder for MATLAB Level 5 `.mat` files.
//! - v7.3: optional HDF5 backend behind feature `mat-hdf5`.
//!
//! Parity target: libvips `matload` in `libvips/foreign/matlab.c`.
//! There is no save path in libvips; this codec is decode-only.
//!
//! # Format overview
//!
//! A MAT v5 file is structured as:
//! - A 128-byte file header:
//!   - 116 bytes: text description (typically "MATLAB 5.0 MAT-file …").
//!   - 8 bytes: subsystem data (set to zeros if not used).
//!   - 2 bytes: version (0x0100 = MAT-file version 5).
//!   - 2 bytes: endian indicator (`"MI"` = little-endian, `"IM"` = big-endian).
//! - Zero or more data elements, each composed of:
//!   - 4 bytes: data type tag.
//!   - 4 bytes: number of bytes in the data section.
//!   - `n` bytes: data (padded to 8-byte boundary).
//!
//! Each numeric array element uses the `miMATRIX` tag (14) and contains
//! nested sub-elements for: flags, dimensions, name, and real data.
//!
//! # What is loaded
//!
//! Following libvips semantics: the **first variable** in the file with rank
//! 1–3 is loaded as an image. 1-D → single-column image, 2-D → greyscale,
//! 3-D → multi-band (third dimension becomes the band count).
//!
//! Data layout in MATLAB is **column-major**: elements along the first
//! dimension (rows) are contiguous. viprs images are row-major. The loader
//! transposes the data on load, matching libvips.
//!
//! # Supported MATLAB classes
//!
//! | MATLAB class        | viprs format |
//! |---------------------|--------------|
//! | `mxUINT8_CLASS` (9) | U8           |
//! | `mxUINT16_CLASS`(11)| U16          |
//! | `mxINT16_CLASS` (10)| I16          |
//! | `mxUINT32_CLASS`(13)| U32          |
//! | `mxINT32_CLASS` (12)| I32          |
//! | `mxSINGLE_CLASS`(7) | F32          |
//! | `mxDOUBLE_CLASS`(6) | F64          |
//!
//! Note: `mxINT8_CLASS` (8) is not supported because viprs has no `I8` format.
//! Complex arrays, sparse matrices, and cell arrays are not supported.
//!
//! # References
//!
//! - `MATLAB Level 5 MAT-File Format spec (MathWorks pdf)`
//! - libvips parity: `.libvips_repo/libvips/foreign/matlab.c`

#[cfg(feature = "mat-hdf5")]
use hdf5_pure_rust::{Dataset as Hdf5Dataset, File as Hdf5File, Group as Hdf5Group};
#[cfg(feature = "mat-hdf5")]
use std::fs;
#[cfg(feature = "mat-hdf5")]
use std::path::{Path, PathBuf};
#[cfg(feature = "mat-hdf5")]
use std::sync::atomic::{AtomicU64, Ordering};
use viprs_core::codec_options::{LoadOptions, SaveOptions};
use viprs_core::error::ViprsError;
use viprs_core::format::{BandFormat, BandFormatId};
use viprs_core::image::{InMemoryImage, ImageMetadata, Interpretation};
use viprs_ports::codec::{ImageDecoder, ImageEncoder};

// ── File-level constants ───────────────────────────────────────────────────

/// MAT v5 miMATRIX tag.
const MI_MATRIX: u32 = 14;

/// MATLAB array class codes (from mxClassID).
const MX_DOUBLE_CLASS: u8 = 6;
const MX_SINGLE_CLASS: u8 = 7;
const MX_UINT8_CLASS: u8 = 9;
const MX_INT16_CLASS: u8 = 10;
const MX_UINT16_CLASS: u8 = 11;
const MX_INT32_CLASS: u8 = 12;
const MX_UINT32_CLASS: u8 = 13;

/// Byte position of the endian indicator within the file header.
const ENDIAN_OFFSET: usize = 126;
/// File header size in bytes.
const HEADER_SIZE: usize = 128;
/// HDF5 superblock signature used by MATLAB v7.3 files.
const HDF5_SIGNATURE: [u8; 8] = [0x89, b'H', b'D', b'F', 0x0D, 0x0A, 0x1A, 0x0A];
/// MATLAB v7.3 commonly stores HDF5 superblock after a 512-byte user block.
const HDF5_USERBLOCK_OFFSET: usize = 512;

// ── Helper: byte reading ───────────────────────────────────────────────────

fn read_u32_le(src: &[u8], offset: usize) -> Result<u32, ViprsError> {
    src.get(offset..offset + 4)
        .and_then(|b| b.try_into().ok())
        .map(u32::from_le_bytes)
        .ok_or_else(|| ViprsError::Codec(format!("mat: read_u32_le at offset {offset} OOB")))
}

fn read_u32_be(src: &[u8], offset: usize) -> Result<u32, ViprsError> {
    src.get(offset..offset + 4)
        .and_then(|b| b.try_into().ok())
        .map(u32::from_be_bytes)
        .ok_or_else(|| ViprsError::Codec(format!("mat: read_u32_be at offset {offset} OOB")))
}

fn read_i32_le(src: &[u8], offset: usize) -> Result<i32, ViprsError> {
    src.get(offset..offset + 4)
        .and_then(|b| b.try_into().ok())
        .map(i32::from_le_bytes)
        .ok_or_else(|| ViprsError::Codec(format!("mat: read_i32_le at offset {offset} OOB")))
}

// ── Detect endianness ──────────────────────────────────────────────────────

fn is_little_endian(src: &[u8]) -> Result<bool, ViprsError> {
    if src.len() < HEADER_SIZE {
        return Err(ViprsError::Codec("mat: buffer too short for header".into()));
    }
    match &src[ENDIAN_OFFSET..ENDIAN_OFFSET + 2] {
        b"MI" => Ok(true),
        b"IM" => Ok(false),
        other => Err(ViprsError::Codec(format!(
            "mat: unrecognised endian indicator {other:?}"
        ))),
    }
}

// ── Sniff ──────────────────────────────────────────────────────────────────

fn sniff_mat_v5(header: &[u8]) -> bool {
    header.len() >= 10 && header[..10] == *b"MATLAB 5.0"
}

fn has_hdf5_signature(src: &[u8]) -> bool {
    let at_start = src
        .get(..HDF5_SIGNATURE.len())
        .is_some_and(|head| head == HDF5_SIGNATURE);
    let at_userblock = src
        .get(HDF5_USERBLOCK_OFFSET..HDF5_USERBLOCK_OFFSET + HDF5_SIGNATURE.len())
        .is_some_and(|head| head == HDF5_SIGNATURE);
    at_start || at_userblock
}

fn sniff_mat(header: &[u8]) -> bool {
    sniff_mat_v5(header) || has_hdf5_signature(header)
}

// ── Matrix description ─────────────────────────────────────────────────────

struct MatrixInfo {
    /// MATLAB class code.
    class: u8,
    /// Whether the matrix has a complex part (unsupported).
    is_complex: bool,
    /// Dimensions [rows, cols, bands?]. Length 1–3.
    dims: Vec<usize>,
    /// Byte offset (from start of `src`) to the real-data sub-element body.
    data_offset: usize,
    /// MAT tag type of the real-data sub-element.
    data_tag: u32,
    /// Length of the real-data region in bytes.
    data_len: usize,
}

// ── Parse a miMATRIX element ───────────────────────────────────────────────

fn parse_matrix(
    src: &[u8],
    matrix_data: &[u8],
    little_endian: bool,
) -> Result<MatrixInfo, ViprsError> {
    let read_u32 = if little_endian {
        read_u32_le
    } else {
        read_u32_be
    };

    let mut pos: usize = 0;

    // Sub-element 1: Array flags (miUINT32, 8 bytes)
    if pos + 8 > matrix_data.len() {
        return Err(ViprsError::Codec("mat: flags sub-element missing".into()));
    }
    let _flags_tag = read_u32(matrix_data, pos)?;
    let flags_size = read_u32(matrix_data, pos + 4)? as usize;
    if pos + 8 + flags_size > matrix_data.len() {
        return Err(ViprsError::Codec("mat: flags sub-element OOB".into()));
    }
    let flags_data = &matrix_data[pos + 8..pos + 8 + flags_size];
    if flags_data.len() < 8 {
        return Err(ViprsError::Codec("mat: flags data too short".into()));
    }
    let class = flags_data[0];
    let flags_byte = flags_data[1];
    let is_complex = (flags_byte & 0x08) != 0;
    let padded_flags = (flags_size + 7) & !7;
    pos += 8 + padded_flags;

    // Sub-element 2: Dimensions (miINT32)
    if pos + 8 > matrix_data.len() {
        return Err(ViprsError::Codec("mat: dims sub-element missing".into()));
    }
    let _dims_tag = read_u32(matrix_data, pos)?;
    let dims_size = read_u32(matrix_data, pos + 4)? as usize;
    if pos + 8 + dims_size > matrix_data.len() {
        return Err(ViprsError::Codec("mat: dims sub-element OOB".into()));
    }
    let dims_data = &matrix_data[pos + 8..pos + 8 + dims_size];
    let ndims = dims_size / 4;
    if !(1..=8).contains(&ndims) {
        return Err(ViprsError::Codec(format!(
            "mat: unsupported rank {ndims}; must be 1–8"
        )));
    }
    let mut dims = Vec::with_capacity(ndims);
    for i in 0..ndims {
        let d = read_i32_le(dims_data, i * 4)? as usize;
        dims.push(d);
    }
    let padded_dims = (dims_size + 7) & !7;
    pos += 8 + padded_dims;

    // Sub-element 3: Array name (may be empty)
    if pos + 8 > matrix_data.len() {
        return Err(ViprsError::Codec("mat: name sub-element missing".into()));
    }
    let _name_tag = read_u32(matrix_data, pos)?;
    let name_size = read_u32(matrix_data, pos + 4)? as usize;
    let padded_name = (name_size + 7) & !7;
    pos += 8 + padded_name;

    // Sub-element 4: Real data.
    if pos + 8 > matrix_data.len() {
        return Err(ViprsError::Codec(
            "mat: real-data sub-element missing".into(),
        ));
    }
    let data_tag = read_u32(matrix_data, pos)?;
    let data_size = read_u32(matrix_data, pos + 4)? as usize;
    let data_offset = {
        let matrix_start = matrix_data.as_ptr() as usize - src.as_ptr() as usize;
        matrix_start + pos + 8
    };

    Ok(MatrixInfo {
        class,
        is_complex,
        dims,
        data_offset,
        data_tag,
        data_len: data_size,
    })
}

// ── Image dimensions from MatrixInfo ──────────────────────────────────────

/// Returns (width, height, bands) from a matrix's dims following libvips.
///
/// MATLAB dims are `(rows × cols × bands)`; libvips swaps rows and cols
/// (see `matlab.c`: `"21/8/14 — swap width/height"`), so:
/// - width = cols (`dim[1]`)
/// - height = rows (`dim[0]`)
fn matrix_image_dims(info: &MatrixInfo) -> Result<(u32, u32, u32), ViprsError> {
    let rows = *info.dims.first().unwrap_or(&1);
    let cols = if info.dims.len() >= 2 {
        info.dims[1]
    } else {
        1
    };
    let bands = if info.dims.len() >= 3 {
        info.dims[2]
    } else {
        1
    };

    let width = cols as u32;
    let height = rows as u32;
    let band_count = bands as u32;

    if width == 0 || height == 0 || band_count == 0 {
        return Err(ViprsError::Codec("mat: zero-size dimension".into()));
    }
    checked_mat_sample_count(width as usize, height as usize, band_count as usize)?;
    Ok((width, height, band_count))
}

// ── Transpose: column-major → row-major ───────────────────────────────────

fn checked_mat_sample_count(
    width: usize,
    height: usize,
    bands: usize,
) -> Result<usize, ViprsError> {
    let width_u64 = u64::try_from(width).map_err(|_| ViprsError::ImageTooLarge {
        width: u32::MAX,
        height: u32::MAX,
        bands: u32::MAX,
        bytes: (width as u128) * (height as u128) * (bands as u128),
        limit_bytes: usize::MAX as u128,
        details: "mat transpose sample count exceeds addressable memory",
    })?;
    let height_u64 = u64::try_from(height).map_err(|_| ViprsError::ImageTooLarge {
        width: u32::MAX,
        height: u32::MAX,
        bands: u32::MAX,
        bytes: (width as u128) * (height as u128) * (bands as u128),
        limit_bytes: usize::MAX as u128,
        details: "mat transpose sample count exceeds addressable memory",
    })?;
    let bands_u64 = u64::try_from(bands).map_err(|_| ViprsError::ImageTooLarge {
        width: u32::MAX,
        height: u32::MAX,
        bands: u32::MAX,
        bytes: (width as u128) * (height as u128) * (bands as u128),
        limit_bytes: usize::MAX as u128,
        details: "mat transpose sample count exceeds addressable memory",
    })?;
    let Some(samples) = width_u64
        .checked_mul(height_u64)
        .and_then(|plane| plane.checked_mul(bands_u64))
    else {
        return Err(ViprsError::ImageTooLarge {
            width: width.try_into().unwrap_or(u32::MAX),
            height: height.try_into().unwrap_or(u32::MAX),
            bands: bands.try_into().unwrap_or(u32::MAX),
            bytes: (width as u128) * (height as u128) * (bands as u128),
            limit_bytes: usize::MAX as u128,
            details: "mat transpose sample count exceeds addressable memory",
        });
    };

    usize::try_from(samples).map_err(|_| ViprsError::ImageTooLarge {
        width: width.try_into().unwrap_or(u32::MAX),
        height: height.try_into().unwrap_or(u32::MAX),
        bands: bands.try_into().unwrap_or(u32::MAX),
        bytes: u128::from(samples),
        limit_bytes: usize::MAX as u128,
        details: "mat transpose sample count exceeds addressable memory",
    })
}

fn checked_mat_byte_len(
    width: usize,
    height: usize,
    bands: usize,
    bytes_per_sample: usize,
) -> Result<usize, ViprsError> {
    let sample_count = checked_mat_sample_count(width, height, bands)?;
    sample_count
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| ViprsError::ImageTooLarge {
            width: width.try_into().unwrap_or(u32::MAX),
            height: height.try_into().unwrap_or(u32::MAX),
            bands: bands.try_into().unwrap_or(u32::MAX),
            bytes: (width as u128)
                * (height as u128)
                * (bands as u128)
                * (bytes_per_sample as u128),
            limit_bytes: usize::MAX as u128,
            details: "mat transpose byte count exceeds addressable memory",
        })
}

/// Transpose a column-major MATLAB band-plane into a row-major interleaved
/// viprs layout.
///
/// MATLAB layout: `src[row + col * height + band * height * width]`
/// viprs layout:  `out[(row * width + col) * bands + band]`
fn transpose_mat_data<T: Copy>(
    src: &[T],
    width: usize,
    height: usize,
    bands: usize,
) -> Result<Vec<T>, ViprsError> {
    let n = checked_mat_sample_count(width, height, bands)?;
    let mut out = src[..n].to_vec();
    let plane_size = checked_mat_sample_count(width, height, 1)?;

    for y in 0..height {
        for x in 0..width {
            for b in 0..bands {
                let src_idx = y + x * height + b * plane_size;
                let dst_idx = (y * width + x) * bands + b;
                out[dst_idx] = src[src_idx];
            }
        }
    }
    Ok(out)
}

// ── BandFormat mapping ─────────────────────────────────────────────────────

const fn mat_class_to_band_format(class: u8) -> Option<BandFormatId> {
    // Note: MX_INT8_CLASS (8) is omitted — viprs has no I8 BandFormat.
    match class {
        MX_UINT8_CLASS => Some(BandFormatId::U8),
        MX_UINT16_CLASS => Some(BandFormatId::U16),
        MX_INT16_CLASS => Some(BandFormatId::I16),
        MX_UINT32_CLASS => Some(BandFormatId::U32),
        MX_INT32_CLASS => Some(BandFormatId::I32),
        MX_SINGLE_CLASS => Some(BandFormatId::F32),
        MX_DOUBLE_CLASS => Some(BandFormatId::F64),
        _ => None,
    }
}

const fn pick_interpretation(bands: u32, id: BandFormatId) -> Interpretation {
    match (bands, id) {
        (3, BandFormatId::U8) => Interpretation::Srgb,
        (3, BandFormatId::U16 | BandFormatId::I16) => Interpretation::Rgb16,
        (1, BandFormatId::U16 | BandFormatId::I16) => Interpretation::Grey16,
        _ => Interpretation::Multiband,
    }
}

// ── Transpose dispatch by format ──────────────────────────────────────────

fn transpose_by_format(
    raw: &[u8],
    width: usize,
    height: usize,
    bands: usize,
    format: BandFormatId,
    little_endian: bool,
    _data_tag: u32,
) -> Result<Vec<u8>, ViprsError> {
    let n = checked_mat_sample_count(width, height, bands)?;

    macro_rules! transpose_pod {
        ($ty:ty) => {{
            let size = std::mem::size_of::<$ty>();
            let byte_len = checked_mat_byte_len(width, height, bands, size)?;
            if raw.len() < byte_len {
                return Err(ViprsError::Codec(format!(
                    "mat: real-data too short: need {} bytes, got {}",
                    byte_len,
                    raw.len()
                )));
            }
            let samples: Vec<$ty> = (0..n)
                .map(|i| {
                    let b: [u8; std::mem::size_of::<$ty>()] = raw[i * size..(i + 1) * size]
                        .try_into()
                        .unwrap_or([0; std::mem::size_of::<$ty>()]);
                    if little_endian {
                        <$ty>::from_le_bytes(b)
                    } else {
                        <$ty>::from_be_bytes(b)
                    }
                })
                .collect();
            let transposed = transpose_mat_data(&samples, width, height, bands)?;
            bytemuck::cast_slice::<$ty, u8>(&transposed).to_vec()
        }};
    }

    match format {
        BandFormatId::U8 => {
            if raw.len() < n {
                return Err(ViprsError::Codec(format!(
                    "mat: real-data too short: need {n}, got {}",
                    raw.len()
                )));
            }
            transpose_mat_data(&raw[..n], width, height, bands)
        }
        BandFormatId::U16 => Ok(transpose_pod!(u16)),
        BandFormatId::I16 => Ok(transpose_pod!(i16)),
        BandFormatId::U32 => Ok(transpose_pod!(u32)),
        BandFormatId::I32 => Ok(transpose_pod!(i32)),
        BandFormatId::F32 => Ok(transpose_pod!(f32)),
        BandFormatId::F64 => Ok(transpose_pod!(f64)),
    }
}

#[cfg(feature = "mat-hdf5")]
fn hdf5_dtype_to_band_format(dtype: &hdf5_pure_rust::Datatype) -> Option<BandFormatId> {
    if dtype.is_integer() {
        let signed = dtype.is_signed().unwrap_or(false);
        return match (dtype.size(), signed) {
            (1, false) => Some(BandFormatId::U8),
            (2, false) => Some(BandFormatId::U16),
            (2, true) => Some(BandFormatId::I16),
            (4, false) => Some(BandFormatId::U32),
            (4, true) => Some(BandFormatId::I32),
            _ => None,
        };
    }

    if dtype.is_float() {
        return match dtype.size() {
            4 => Some(BandFormatId::F32),
            8 => Some(BandFormatId::F64),
            _ => None,
        };
    }

    None
}

#[cfg(feature = "mat-hdf5")]
fn write_hdf5_bytes_to_workfile(src: &[u8]) -> Result<PathBuf, ViprsError> {
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);

    let dir = Path::new("target").join("mat_hdf5_decode");
    fs::create_dir_all(&dir)
        .map_err(|err| ViprsError::Codec(format!("mat: cannot create HDF5 work dir: {err}")))?;

    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let path = dir.join(format!("decode_{}_{}.mat", std::process::id(), id));
    fs::write(&path, src)
        .map_err(|err| ViprsError::Codec(format!("mat: cannot materialize HDF5 bytes: {err}")))?;
    Ok(path)
}

#[cfg(feature = "mat-hdf5")]
fn decode_hdf5_dataset(dataset: &Hdf5Dataset) -> Result<Option<DecodedMatrix>, ViprsError> {
    let shape = dataset
        .shape()
        .map_err(|err| ViprsError::Codec(format!("mat: HDF5 shape read failed: {err}")))?;
    if !(1..=3).contains(&shape.len()) {
        return Ok(None);
    }

    let dtype = dataset
        .dtype()
        .map_err(|err| ViprsError::Codec(format!("mat: HDF5 dtype read failed: {err}")))?;
    let Some(format) = hdf5_dtype_to_band_format(&dtype) else {
        return Ok(None);
    };

    let rows = usize::try_from(shape[0]).map_err(|_| {
        ViprsError::Codec(format!(
            "mat: rows dimension {} does not fit usize",
            shape[0]
        ))
    })?;
    let cols = if shape.len() >= 2 {
        usize::try_from(shape[1]).map_err(|_| {
            ViprsError::Codec(format!(
                "mat: cols dimension {} does not fit usize",
                shape[1]
            ))
        })?
    } else {
        1
    };
    let bands = if shape.len() >= 3 {
        usize::try_from(shape[2]).map_err(|_| {
            ViprsError::Codec(format!(
                "mat: bands dimension {} does not fit usize",
                shape[2]
            ))
        })?
    } else {
        1
    };

    if rows == 0 || cols == 0 || bands == 0 {
        return Err(ViprsError::Codec(
            "mat: zero-size dimension in HDF5 dataset".into(),
        ));
    }

    let width = u32::try_from(cols)
        .map_err(|_| ViprsError::Codec(format!("mat: width {cols} does not fit u32")))?;
    let height = u32::try_from(rows)
        .map_err(|_| ViprsError::Codec(format!("mat: height {rows} does not fit u32")))?;
    let bands_u32 = u32::try_from(bands)
        .map_err(|_| ViprsError::Codec(format!("mat: bands {bands} does not fit u32")))?;

    macro_rules! read_and_transpose {
        ($ty:ty, $fmt:expr) => {{
            let array = dataset
                .read::<$ty>()
                .map_err(|err| ViprsError::Codec(format!("mat: HDF5 read failed: {err}")))?;
            let pixel_bytes = bytemuck::cast_slice::<$ty, u8>(&array).to_vec();
            Ok(Some(DecodedMatrix {
                width,
                height,
                bands: bands_u32,
                format: $fmt,
                pixel_bytes,
            }))
        }};
    }

    match format {
        BandFormatId::U8 => read_and_transpose!(u8, BandFormatId::U8),
        BandFormatId::U16 => read_and_transpose!(u16, BandFormatId::U16),
        BandFormatId::I16 => read_and_transpose!(i16, BandFormatId::I16),
        BandFormatId::U32 => read_and_transpose!(u32, BandFormatId::U32),
        BandFormatId::I32 => read_and_transpose!(i32, BandFormatId::I32),
        BandFormatId::F32 => read_and_transpose!(f32, BandFormatId::F32),
        BandFormatId::F64 => read_and_transpose!(f64, BandFormatId::F64),
    }
}

#[cfg(feature = "mat-hdf5")]
fn decode_first_numeric_in_group(group: &Hdf5Group) -> Result<Option<DecodedMatrix>, ViprsError> {
    let mut datasets = group
        .datasets()
        .map_err(|err| ViprsError::Codec(format!("mat: HDF5 group read failed: {err}")))?;
    datasets.sort_by(|a, b| a.name().cmp(b.name()));

    for dataset in datasets {
        if let Some(decoded) = decode_hdf5_dataset(&dataset)? {
            return Ok(Some(decoded));
        }
    }

    let mut groups = group
        .groups()
        .map_err(|err| ViprsError::Codec(format!("mat: HDF5 group read failed: {err}")))?;
    groups.sort_by(|a, b| a.name().cmp(b.name()));

    for child in groups {
        if let Some(decoded) = decode_first_numeric_in_group(&child)? {
            return Ok(Some(decoded));
        }
    }

    Ok(None)
}

#[cfg(feature = "mat-hdf5")]
fn decode_hdf5_first_matrix(src: &[u8]) -> Result<DecodedMatrix, ViprsError> {
    let workfile = write_hdf5_bytes_to_workfile(src)?;
    let file = Hdf5File::open(&workfile)
        .map_err(|err| ViprsError::Codec(format!("mat: invalid HDF5 payload: {err}")))?;

    let decoded = if let Ok(group) = file.group("/matlab/variables") {
        if let Some(decoded) = decode_first_numeric_in_group(&group)? {
            Ok(decoded)
        } else {
            let root = file.root_group().map_err(|err| {
                ViprsError::Codec(format!("mat: cannot read HDF5 root group: {err}"))
            })?;
            decode_first_numeric_in_group(&root)?.ok_or_else(|| {
                ViprsError::Codec(
                    "mat: no supported numeric dataset found in HDF5 MAT v7.3 payload".into(),
                )
            })
        }
    } else {
        let root = file
            .root_group()
            .map_err(|err| ViprsError::Codec(format!("mat: cannot read HDF5 root group: {err}")))?;
        decode_first_numeric_in_group(&root)?.ok_or_else(|| {
            ViprsError::Codec(
                "mat: no supported numeric dataset found in HDF5 MAT v7.3 payload".into(),
            )
        })
    };

    let _ = fs::remove_file(workfile);
    decoded
}

// ── Core decode ───────────────────────────────────────────────────────────

struct DecodedMatrix {
    width: u32,
    height: u32,
    bands: u32,
    format: BandFormatId,
    pixel_bytes: Vec<u8>,
}

fn decode_first_matrix(src: &[u8]) -> Result<DecodedMatrix, ViprsError> {
    if has_hdf5_signature(src) {
        #[cfg(feature = "mat-hdf5")]
        {
            return decode_hdf5_first_matrix(src);
        }
        #[cfg(not(feature = "mat-hdf5"))]
        {
            return Err(ViprsError::Codec(
                "mat: detected MATLAB v7.3 (HDF5) file; rebuild with feature 'mat-hdf5'".into(),
            ));
        }
    }

    if src.len() < HEADER_SIZE {
        return Err(ViprsError::Codec("mat: buffer too short for header".into()));
    }

    if !sniff_mat_v5(src) {
        return Err(ViprsError::Codec(
            "mat: missing 'MATLAB 5.0' signature".into(),
        ));
    }

    let little_endian = is_little_endian(src)?;
    let read_u32 = if little_endian {
        read_u32_le
    } else {
        read_u32_be
    };
    let mut pos = HEADER_SIZE;

    loop {
        if pos >= src.len() {
            return Err(ViprsError::Codec(
                "mat: no numeric matrix variable found in file".into(),
            ));
        }
        if pos + 8 > src.len() {
            return Err(ViprsError::Codec("mat: truncated element header".into()));
        }

        let tag = read_u32(src, pos)?;
        let size = read_u32(src, pos + 4)? as usize;

        let data_start = pos + 8;
        let data_end = data_start + size;
        if data_end > src.len() {
            return Err(ViprsError::Codec(format!(
                "mat: element at {pos} claims {size} bytes but buffer ends at {}",
                src.len()
            )));
        }

        let padded = (size + 7) & !7;
        let next_pos = data_start + padded;

        if tag != MI_MATRIX {
            pos = next_pos;
            continue;
        }

        let matrix_data = &src[data_start..data_end];
        let Ok(info) = parse_matrix(src, matrix_data, little_endian) else {
            pos = next_pos;
            continue;
        };

        let ndim = info.dims.len();
        if !(1..=3).contains(&ndim) {
            pos = next_pos;
            continue;
        }

        if mat_class_to_band_format(info.class).is_none() {
            pos = next_pos;
            continue;
        }

        if info.is_complex {
            return Err(ViprsError::Codec(
                "mat: complex arrays are not supported".into(),
            ));
        }

        let format = mat_class_to_band_format(info.class).ok_or_else(|| {
            ViprsError::Codec(format!(
                "mat: unsupported matrix class {} after validation",
                info.class
            ))
        })?;
        let (width, height, bands) = matrix_image_dims(&info)?;

        let data_end_abs = info.data_offset + info.data_len;
        if data_end_abs > src.len() {
            return Err(ViprsError::Codec(format!(
                "mat: real-data sub-element OOB: end={data_end_abs} buffer={}",
                src.len()
            )));
        }
        let raw = &src[info.data_offset..data_end_abs];

        let pixel_bytes = transpose_by_format(
            raw,
            width as usize,
            height as usize,
            bands as usize,
            format,
            little_endian,
            info.data_tag,
        )?;

        return Ok(DecodedMatrix {
            width,
            height,
            bands,
            format,
            pixel_bytes,
        });
    }
}

// ── Codec ─────────────────────────────────────────────────────────────────────

/// MAT v5 codec (pure Rust, decode-only).
///
/// Decodes the first numeric variable in a MATLAB Level 5 `.mat` file.
/// Save is not supported (libvips has no matload save path).
pub struct MatCodec;

impl MatCodec {
    /// Decode the first numeric matrix in a MAT v5 byte stream.
    ///
    /// `F` must match the MATLAB class stored in the file.
    ///
    /// # Errors
    ///
    /// Returns [`ViprsError::Codec`] when:
    /// - The buffer is too short for the 128-byte header.
    /// - The "MATLAB 5.0" signature is absent.
    /// - No numeric matrix with rank 1–3 is found.
    /// - `F::ID` does not match the MATLAB class of the first matrix.
    pub fn decode_mat<F: BandFormat>(&self, src: &[u8]) -> Result<InMemoryImage<F>, ViprsError> {
        let dec = decode_first_matrix(src)?;

        if dec.format != F::ID {
            return Err(ViprsError::Codec(format!(
                "mat: file contains {:?} data but caller requested {:?}",
                dec.format,
                F::ID,
            )));
        }

        // SAFETY: F::Sample is Pod (BandFormat bound); dec.pixel_bytes was built
        // by casting via bytemuck so alignment is correct.
        let samples: Vec<F::Sample> =
            bytemuck::cast_slice::<u8, F::Sample>(&dec.pixel_bytes).to_vec();

        let interp = pick_interpretation(dec.bands, F::ID);
        let metadata = ImageMetadata {
            interpretation: Some(interp),
            ..ImageMetadata::default()
        };

        InMemoryImage::from_buffer(dec.width, dec.height, dec.bands, samples)
            .map(|img| img.with_metadata(metadata))
            .map_err(|err| ViprsError::Codec(err.to_string()))
    }
}

impl ImageDecoder for MatCodec {
    fn format_name(&self) -> &'static str {
        "mat"
    }

    fn sniff(&self, header: &[u8]) -> bool
    where
        Self: Sized,
    {
        sniff_mat(header)
    }

    fn decode<F: BandFormat>(&self, src: &[u8]) -> Result<InMemoryImage<F>, ViprsError> {
        self.decode_mat(src)
    }

    fn decode_with_options<F: BandFormat>(
        &self,
        src: &[u8],
        _opts: &LoadOptions,
    ) -> Result<InMemoryImage<F>, ViprsError>
    where
        Self: Sized,
    {
        self.decode_mat(src)
    }

    fn probe(&self, src: &[u8]) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized,
    {
        let dec = decode_first_matrix(src)?;
        Ok((dec.width, dec.height, dec.bands))
    }
}

/// MAT v5 has no save path in libvips; encoding is not supported.
impl ImageEncoder for MatCodec {
    fn format_name(&self) -> &'static str {
        "mat"
    }

    fn encode<F: BandFormat>(&self, _image: &InMemoryImage<F>) -> Result<Vec<u8>, ViprsError> {
        Err(ViprsError::Unimplemented {
            feature: "mat encode",
            details: "MATLAB .mat save is not supported (no libvips parity path exists)",
        })
    }

    fn encode_with_options<F: BandFormat>(
      &self,
      image: &InMemoryImage<F>,
      _opts: &SaveOptions,
    ) -> Result<Vec<u8>, ViprsError>
    where
        Self: Sized,
    {
        self.encode(image)
    }
}

// ── Test helpers ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod builder {
    // MAT v5 data type tag constants (only used in test helpers).
    pub const MI_UINT8: u32 = 2;
    pub const MI_INT16: u32 = 3;
    pub const MI_UINT16: u32 = 4;
    pub const MI_INT32: u32 = 5;
    pub const MI_SINGLE: u32 = 7;
    pub const MI_DOUBLE: u32 = 9;

    // MATLAB class codes (only used in test helpers).
    pub const MX_DOUBLE_CLASS: u8 = 6;
    pub const MX_SINGLE_CLASS: u8 = 7;
    pub const MX_UINT8_CLASS: u8 = 9;
    pub const MX_INT16_CLASS: u8 = 10;
    pub const MX_UINT16_CLASS: u8 = 11;
    pub const MX_INT32_CLASS: u8 = 12;
    pub const MX_UINT32_CLASS: u8 = 13;

    /// Build a minimal MAT v5 byte stream containing one numeric array.
    ///
    /// Data must be provided in column-major order (rows vary fastest):
    /// `data[row + col * rows + band * rows * cols]`
    pub fn make_mat_stream<T: bytemuck::Pod + Copy>(
        rows: usize,
        cols: usize,
        bands: usize,
        data: &[T],
        class_id: u8,
        data_type_tag: u32,
    ) -> Vec<u8> {
        let mut out = Vec::new();

        // 128-byte file header.
        let mut hdr = [0u8; 128];
        let desc = b"MATLAB 5.0 MAT-file, viprs test";
        let desc_len = desc.len().min(116);
        hdr[..desc_len].copy_from_slice(&desc[..desc_len]);
        hdr[124] = 0x00;
        hdr[125] = 0x01; // version 0x0100 LE
        hdr[126] = b'M';
        hdr[127] = b'I'; // "MI" = little-endian
        out.extend_from_slice(&hdr);

        // miMATRIX body sub-elements.
        let mut body: Vec<u8> = Vec::new();

        // Sub-element 1: Array flags (miUINT32 tag=6, size=8).
        body.extend_from_slice(&6u32.to_le_bytes()); // miUINT32
        body.extend_from_slice(&8u32.to_le_bytes());
        body.push(class_id);
        body.push(0x00); // real, non-sparse
        body.extend_from_slice(&[0u8; 6]);

        // Sub-element 2: Dimensions (miINT32 tag=5).
        let ndim_bytes = if bands > 1 { 12usize } else { 8 };
        body.extend_from_slice(&5u32.to_le_bytes()); // miINT32
        body.extend_from_slice(&(ndim_bytes as u32).to_le_bytes());
        body.extend_from_slice(&(rows as i32).to_le_bytes());
        body.extend_from_slice(&(cols as i32).to_le_bytes());
        if bands > 1 {
            body.extend_from_slice(&(bands as i32).to_le_bytes());
        }
        let pad = (8 - (ndim_bytes % 8)) % 8;
        body.extend_from_slice(&vec![0u8; pad]);

        // Sub-element 3: Array name (miINT8 tag=1, size=0).
        body.extend_from_slice(&1u32.to_le_bytes());
        body.extend_from_slice(&0u32.to_le_bytes());

        // Sub-element 4: Real data.
        let data_bytes = bytemuck::cast_slice::<T, u8>(data);
        body.extend_from_slice(&data_type_tag.to_le_bytes());
        body.extend_from_slice(&(data_bytes.len() as u32).to_le_bytes());
        body.extend_from_slice(data_bytes);
        let data_pad = (8 - (data_bytes.len() % 8)) % 8;
        body.extend_from_slice(&vec![0u8; data_pad]);

        // Write the outer miMATRIX element.
        out.extend_from_slice(&14u32.to_le_bytes()); // MI_MATRIX
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&body);

        out
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::builder::{
        MI_DOUBLE, MI_INT16, MI_INT32, MI_SINGLE, MI_UINT8, MI_UINT16, MX_DOUBLE_CLASS,
        MX_INT16_CLASS, MX_INT32_CLASS, MX_SINGLE_CLASS, MX_UINT8_CLASS, MX_UINT16_CLASS,
        MX_UINT32_CLASS, make_mat_stream,
    };
    use super::*;
    use viprs_core::format::{F32, F64, I16, I32, U8, U16};

    fn mat_header_bytes() -> Vec<u8> {
        let mut hdr = [0u8; HEADER_SIZE];
        let desc = b"MATLAB 5.0 MAT-file, viprs test";
        hdr[..desc.len()].copy_from_slice(desc);
        hdr[124] = 0x00;
        hdr[125] = 0x01;
        hdr[126] = b'M';
        hdr[127] = b'I';
        hdr.to_vec()
    }

    fn wrap_matrix_bodies(bodies: &[Vec<u8>]) -> Vec<u8> {
        let mut stream = mat_header_bytes();
        for body in bodies {
            stream.extend_from_slice(&MI_MATRIX.to_le_bytes());
            stream.extend_from_slice(&(body.len() as u32).to_le_bytes());
            stream.extend_from_slice(body);
        }
        stream
    }

    fn make_matrix_body_with_dims<T: bytemuck::Pod>(
        dims: &[i32],
        data: &[T],
        class: u8,
        data_type_tag: u32,
    ) -> Vec<u8> {
        let mut body = Vec::new();

        body.extend_from_slice(&6u32.to_le_bytes());
        body.extend_from_slice(&8u32.to_le_bytes());
        body.push(class);
        body.push(0);
        body.extend_from_slice(&[0u8; 6]);

        let dims_bytes_len = dims.len() * std::mem::size_of::<i32>();
        body.extend_from_slice(&5u32.to_le_bytes());
        body.extend_from_slice(&(dims_bytes_len as u32).to_le_bytes());
        for dim in dims {
            body.extend_from_slice(&dim.to_le_bytes());
        }
        let dims_padding = (8 - (dims_bytes_len % 8)) % 8;
        body.extend_from_slice(&vec![0u8; dims_padding]);

        body.extend_from_slice(&1u32.to_le_bytes());
        body.extend_from_slice(&0u32.to_le_bytes());

        let data_bytes = bytemuck::cast_slice::<T, u8>(data);
        body.extend_from_slice(&data_type_tag.to_le_bytes());
        body.extend_from_slice(&(data_bytes.len() as u32).to_le_bytes());
        body.extend_from_slice(data_bytes);
        let data_padding = (8 - (data_bytes.len() % 8)) % 8;
        body.extend_from_slice(&vec![0u8; data_padding]);

        body
    }

    // ── sniff ──────────────────────────────────────────────────────────────

    #[test]
    fn sniff_recognises_matlab_magic() {
        let stream = make_mat_stream::<u8>(1, 1, 1, &[42], MX_UINT8_CLASS, MI_UINT8);
        assert!(MatCodec.sniff(&stream));
    }

    #[test]
    fn sniff_rejects_non_mat() {
        assert!(!MatCodec.sniff(b"JUNK"));
        assert!(!MatCodec.sniff(b""));
    }

    #[test]
    fn sniff_recognises_hdf5_signature_at_userblock_offset() {
        let mut stream = vec![0u8; HDF5_USERBLOCK_OFFSET + HDF5_SIGNATURE.len()];
        stream[HDF5_USERBLOCK_OFFSET..HDF5_USERBLOCK_OFFSET + HDF5_SIGNATURE.len()]
            .copy_from_slice(&HDF5_SIGNATURE);
        assert!(MatCodec.sniff(&stream));
    }

    // ── probe ──────────────────────────────────────────────────────────────

    #[test]
    fn probe_returns_dims_for_greyscale() {
        let data: Vec<u8> = (0..32).collect();
        let stream = make_mat_stream::<u8>(4, 8, 1, &data, MX_UINT8_CLASS, MI_UINT8);
        let (w, h, b) = MatCodec.probe(&stream).unwrap();
        // rows=4 → height=4, cols=8 → width=8
        assert_eq!((w, h, b), (8, 4, 1));
    }

    #[test]
    fn probe_returns_bands_for_3d_array() {
        let data: Vec<u8> = (0..27).collect(); // 3×3×3
        let stream = make_mat_stream::<u8>(3, 3, 3, &data, MX_UINT8_CLASS, MI_UINT8);
        let (_w, _h, b) = MatCodec.probe(&stream).unwrap();
        assert_eq!(b, 3);
    }

    // ── decode: column-major → row-major transpose ─────────────────────────

    /// Verify transpose is correct for a 2×3 matrix.
    ///
    /// MATLAB column-major layout (2 rows, 3 cols):
    ///   col 0: [0, 1]  col 1: [2, 3]  col 2: [4, 5]
    /// raw bytes: [0, 1, 2, 3, 4, 5]
    ///
    /// viprs row-major: row 0 = [0, 2, 4], row 1 = [1, 3, 5]
    #[test]
    fn decode_transpose_2x3_matrix() {
        let data: Vec<u8> = vec![0, 1, 2, 3, 4, 5];
        let stream = make_mat_stream::<u8>(2, 3, 1, &data, MX_UINT8_CLASS, MI_UINT8);
        let img = MatCodec.decode::<U8>(&stream).unwrap();
        // width = cols = 3, height = rows = 2
        assert_eq!((img.width(), img.height(), img.bands()), (3, 2, 1));
        assert_eq!(img.pixels(), &[0u8, 2, 4, 1, 3, 5]);
    }

    #[test]
    fn decode_u8_identity_1x1() {
        let stream = make_mat_stream::<u8>(1, 1, 1, &[99u8], MX_UINT8_CLASS, MI_UINT8);
        let img = MatCodec.decode::<U8>(&stream).unwrap();
        assert_eq!(img.pixels(), &[99u8]);
    }

    #[test]
    fn decode_f32_values() {
        // col-major [1,2,3,4] for 2×2 → row-major [1,3,2,4]
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let stream = make_mat_stream::<f32>(2, 2, 1, &data, MX_SINGLE_CLASS, MI_SINGLE);
        let img = MatCodec.decode::<F32>(&stream).unwrap();
        assert_eq!(img.pixels(), &[1.0f32, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn decode_f64_values() {
        let data: Vec<f64> = vec![10.0, 20.0];
        let stream = make_mat_stream::<f64>(1, 2, 1, &data, MX_DOUBLE_CLASS, MI_DOUBLE);
        let img = MatCodec.decode::<F64>(&stream).unwrap();
        assert_eq!(img.pixels(), &[10.0f64, 20.0]);
    }

    #[test]
    fn decode_i16_values() {
        let data: Vec<i16> = vec![-1, 0, 1, 2];
        let stream = make_mat_stream::<i16>(2, 2, 1, &data, MX_INT16_CLASS, MI_INT16);
        let img = MatCodec.decode::<I16>(&stream).unwrap();
        assert_eq!(img.pixels().len(), 4);
    }

    #[test]
    fn decode_u16_values() {
        let data: Vec<u16> = vec![100, 200, 300, 400];
        let stream = make_mat_stream::<u16>(2, 2, 1, &data, MX_UINT16_CLASS, MI_UINT16);
        let img = MatCodec.decode::<U16>(&stream).unwrap();
        assert_eq!(img.pixels().len(), 4);
    }

    #[test]
    fn decode_i32_values() {
        let data: Vec<i32> = vec![i32::MIN, -1, 0, i32::MAX];
        let stream = make_mat_stream::<i32>(2, 2, 1, &data, MX_INT32_CLASS, MI_INT32);
        let img = MatCodec.decode::<I32>(&stream).unwrap();
        assert_eq!(img.pixels().len(), 4);
    }

    #[test]
    fn decode_3band_rgb() {
        let data: Vec<u8> = (0u8..12).collect(); // 2×2×3
        let stream = make_mat_stream::<u8>(2, 2, 3, &data, MX_UINT8_CLASS, MI_UINT8);
        let img = MatCodec.decode::<U8>(&stream).unwrap();
        assert_eq!((img.bands(), img.width(), img.height()), (3, 2, 2));
        assert_eq!(img.pixels().len(), 12);
    }

    // ── u32 decode ─────────────────────────────────────────────────────────

    #[test]
    fn decode_u32_values() {
        use viprs_core::format::U32;
        let data: Vec<u32> = vec![0, u32::MAX / 2, u32::MAX];
        let stream = make_mat_stream::<u32>(1, 3, 1, &data, MX_UINT32_CLASS, 6);
        let img = MatCodec.decode::<U32>(&stream).unwrap();
        assert_eq!(img.pixels().len(), 3);
    }

    // ── decode error paths ─────────────────────────────────────────────────

    #[test]
    fn error_on_wrong_format() {
        let data = vec![0u8; 4];
        let stream = make_mat_stream::<u8>(2, 2, 1, &data, MX_UINT8_CLASS, MI_UINT8);
        let err = MatCodec.decode::<F32>(&stream).unwrap_err();
        assert!(
            matches!(err, ViprsError::Codec(ref message) if message.contains("file contains U8 data but caller requested F32")),
            "expected typed wrong-format error, got: {err:?}"
        );
    }

    #[test]
    fn error_on_short_buffer() {
        let err = MatCodec.decode::<U8>(&[0u8; 10]).unwrap_err();
        assert!(
            matches!(err, ViprsError::Codec(ref message) if message.contains("buffer too short for header")),
            "expected typed short-buffer error, got: {err:?}"
        );
    }

    #[test]
    fn error_on_missing_matlab_magic() {
        let buf = vec![0u8; 256];
        let err = MatCodec.decode::<U8>(&buf).unwrap_err();
        assert!(
            matches!(err, ViprsError::Codec(ref message) if message.contains("missing 'MATLAB 5.0' signature")),
            "expected typed missing-signature error, got: {err:?}"
        );
    }

    #[test]
    fn error_on_oversized_u8_dimensions() {
        let oversized = make_matrix_body_with_dims::<u8>(
            &[i32::MAX, i32::MAX, i32::MAX],
            &[1],
            MX_UINT8_CLASS,
            MI_UINT8,
        );
        let stream = wrap_matrix_bodies(&[oversized]);

        let err = MatCodec.decode::<U8>(&stream).unwrap_err();

        assert!(matches!(
            err,
            ViprsError::ImageTooLarge {
                width,
                height,
                bands,
                ..
            } if width == i32::MAX as u32
                && height == i32::MAX as u32
                && bands == i32::MAX as u32
        ));
    }

    #[test]
    fn error_on_oversized_u16_dimensions() {
        let oversized = make_matrix_body_with_dims::<u16>(
            &[i32::MAX, i32::MAX, i32::MAX],
            &[1],
            MX_UINT16_CLASS,
            MI_UINT16,
        );
        let stream = wrap_matrix_bodies(&[oversized]);

        let err = MatCodec.decode::<U16>(&stream).unwrap_err();

        assert!(matches!(
            err,
            ViprsError::ImageTooLarge {
                width,
                height,
                bands,
                ..
            } if width == i32::MAX as u32
                && height == i32::MAX as u32
                && bands == i32::MAX as u32
        ));
    }

    #[cfg(not(feature = "mat-hdf5"))]
    #[test]
    fn error_on_hdf5_without_feature_flag() {
        let mut stream = vec![0u8; HDF5_USERBLOCK_OFFSET + HDF5_SIGNATURE.len()];
        stream[HDF5_USERBLOCK_OFFSET..HDF5_USERBLOCK_OFFSET + HDF5_SIGNATURE.len()]
            .copy_from_slice(&HDF5_SIGNATURE);

        let err = MatCodec.decode::<F64>(&stream).unwrap_err();
        assert!(
            err.to_string().contains("mat-hdf5") && err.to_string().contains("v7.3"),
            "{err}"
        );
    }

    #[test]
    fn encode_always_errors() {
        let image = InMemoryImage::<U8>::from_buffer(2, 2, 1, vec![0u8; 4]).unwrap();
        let err = MatCodec.encode(&image).unwrap_err();
        assert!(
            err.to_string()
                .contains("MATLAB .mat save is not supported (no libvips parity path exists)"),
            "expected unsupported-save error, got: {err:?}"
        );
    }

    #[test]
    fn endian_helpers_and_sniff_cover_additional_header_paths() {
        assert_eq!(read_u32_be(&[0, 0, 0, 7], 0).unwrap(), 7);
        assert_eq!(read_u32_le(&[7, 0, 0, 0], 0).unwrap(), 7);
        assert_eq!(read_i32_le(&[0xff, 0xff, 0xff, 0xff], 0).unwrap(), -1);
        assert!(read_u32_be(&[0, 0, 0], 0).is_err());
        assert!(read_i32_le(&[0, 0, 0], 0).is_err());

        let mut big_endian = mat_header_bytes();
        big_endian[126] = b'I';
        big_endian[127] = b'M';
        assert!(!is_little_endian(&big_endian).unwrap());

        let mut bad = mat_header_bytes();
        bad[126] = b'X';
        bad[127] = b'Y';
        assert!(is_little_endian(&bad).is_err());
        assert!(has_hdf5_signature(&HDF5_SIGNATURE));
        assert_eq!(ImageDecoder::format_name(&MatCodec), "mat");
        assert_eq!(ImageEncoder::format_name(&MatCodec), "mat");
    }

    #[test]
    fn decode_skips_non_matrix_and_invalid_matrix_elements_before_valid_data() {
        let valid = make_mat_stream::<u8>(1, 1, 1, &[42], MX_UINT8_CLASS, MI_UINT8);
        let mut stream = mat_header_bytes();
        stream.extend_from_slice(&1u32.to_le_bytes());
        stream.extend_from_slice(&0u32.to_le_bytes());
        stream.extend_from_slice(&MI_MATRIX.to_le_bytes());
        stream.extend_from_slice(&4u32.to_le_bytes());
        stream.extend_from_slice(&[0u8; 4]);
        stream.extend_from_slice(&[0u8; 4]);
        stream.extend_from_slice(&valid[HEADER_SIZE..]);

        let img = MatCodec.decode::<U8>(&stream).unwrap();

        assert_eq!(img.pixels(), &[42u8]);
    }

    #[test]
    fn decode_skips_unsupported_rank_and_class_before_valid_data() {
        let unsupported_rank =
            make_matrix_body_with_dims::<u8>(&[1, 1, 1, 1], &[7], MX_UINT8_CLASS, MI_UINT8);
        let unsupported_class = make_matrix_body_with_dims::<u8>(&[1, 1], &[9], 0xFE, MI_UINT8);
        let valid = make_matrix_body_with_dims::<u8>(&[1, 1], &[42], MX_UINT8_CLASS, MI_UINT8);
        let stream = wrap_matrix_bodies(&[unsupported_rank, unsupported_class, valid]);

        let img = MatCodec.decode::<U8>(&stream).unwrap();

        assert_eq!(img.pixels(), &[42u8]);
    }

    #[test]
    fn parse_matrix_rejects_unsupported_rank_with_typed_error() {
        let matrix_body = make_matrix_body_with_dims::<u8>(
            &[1, 1, 1, 1, 1, 1, 1, 1, 1],
            &[1],
            MX_UINT8_CLASS,
            MI_UINT8,
        );

        assert!(
            matches!(
                parse_matrix(&matrix_body, &matrix_body, true),
                Err(ViprsError::Codec(ref message)) if message.contains("unsupported rank 9")
            ),
            "expected typed unsupported-rank error"
        );
    }

    #[test]
    fn decode_returns_typed_error_for_real_data_sub_element_oob() {
        let mut matrix_body =
            make_matrix_body_with_dims::<u8>(&[1, 1], &[1], MX_UINT8_CLASS, MI_UINT8);
        matrix_body[44..48].copy_from_slice(&1024u32.to_le_bytes());
        let stream = wrap_matrix_bodies(&[matrix_body]);

        assert!(
            matches!(
                decode_first_matrix(&stream),
                Err(ViprsError::Codec(ref message)) if message.contains("real-data sub-element OOB")
            ),
            "expected typed real-data OOB error"
        );
    }

    #[test]
    fn decode_returns_typed_error_when_only_unsupported_class_matrices_exist() {
        let unsupported_class = make_matrix_body_with_dims::<u8>(&[1, 1], &[9], 0xFE, MI_UINT8);
        let stream = wrap_matrix_bodies(&[unsupported_class]);

        assert!(
            matches!(
                decode_first_matrix(&stream),
                Err(ViprsError::Codec(ref message)) if message.contains("no numeric matrix variable found")
            ),
            "expected typed no-numeric-matrix error"
        );
    }

    #[test]
    fn decode_errors_when_only_non_numeric_elements_exist() {
        let mut stream = mat_header_bytes();
        stream.extend_from_slice(&1u32.to_le_bytes());
        stream.extend_from_slice(&0u32.to_le_bytes());
        assert!(matches!(
            decode_first_matrix(&stream),
            Err(ViprsError::Codec(ref message)) if message.contains("no numeric matrix variable")
        ));
    }

    // ── proptest ───────────────────────────────────────────────────────────

    use proptest::prelude::*;

    proptest! {
        /// Verify the transpose is a lossless permutation: the set of values
        /// before and after must be identical.
        #[test]
        fn prop_transpose_preserves_values_u8(rows in 1usize..=16, cols in 1usize..=16) {
            let data: Vec<u8> = (0..(rows * cols)).map(|i| (i % 256) as u8).collect();
            let stream = make_mat_stream::<u8>(rows, cols, 1, &data, MX_UINT8_CLASS, MI_UINT8);
            let img = MatCodec.decode::<U8>(&stream).unwrap();
            let mut sorted_in = data.clone();
            let mut sorted_out = img.pixels().to_vec();
            sorted_in.sort_unstable();
            sorted_out.sort_unstable();
            prop_assert_eq!(sorted_in, sorted_out);
        }

        #[test]
        fn prop_transpose_preserves_values_f32(rows in 1usize..=8, cols in 1usize..=8) {
            let data: Vec<f32> = (0..(rows * cols)).map(|i| i as f32).collect();
            let stream = make_mat_stream::<f32>(rows, cols, 1, &data, MX_SINGLE_CLASS, MI_SINGLE);
            let img = MatCodec.decode::<F32>(&stream).unwrap();
            let mut sorted_in: Vec<u32> = data.iter().map(|x| x.to_bits()).collect();
            let mut sorted_out: Vec<u32> = img.pixels().iter().map(|x| x.to_bits()).collect();
            sorted_in.sort_unstable();
            sorted_out.sort_unstable();
            prop_assert_eq!(sorted_in, sorted_out);
        }
    }

    #[cfg(feature = "mat-hdf5")]
    mod hdf5_tests {
        use super::*;

        #[test]
        fn decode_v73_hdf5_f64_matrix() {
            let src = include_bytes!("../../../tests/fixtures/mat/v73_f64.mat");
            let img = MatCodec.decode::<F64>(src.as_slice()).unwrap();
            assert_eq!((img.width(), img.height(), img.bands()), (3, 2, 1));
            assert_eq!(img.pixels(), &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        }
    }
}

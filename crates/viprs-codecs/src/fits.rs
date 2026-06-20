//! FITS codec backed by cfitsio (`fitsload` / `fitssave` parity).
//!
//! Parity target: libvips `fitsload` / `fitssave` in
//! `.libvips_repo/libvips/foreign/fits*.c`.

use std::ffi::{CStr, CString};
use std::fs;
use std::os::raw::{c_int, c_long, c_void};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};

use bytemuck::{Pod, Zeroable, allocation::try_cast_vec};

use viprs_core::codec_options::{LoadOptions, SaveOptions};
use viprs_core::error::ViprsError;
use viprs_core::format::{BandFormat, BandFormatId};
use viprs_core::image::{Image, ImageMetadata, Interpretation};
use viprs_ports::codec::{ImageDecoder, ImageEncoder};

const MAX_NAXIS: usize = 10;
const FITS_HEADER_CARD_LEN: usize = 81;
const SIMPLE_TRUE_MAGIC: &[u8] = b"SIMPLE  =                    T";

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

const FITS_BASIC_PREFIXES: [&str; 14] = [
    "SIMPLE ",
    "BITPIX ",
    "NAXIS ",
    "NAXIS1 ",
    "NAXIS2 ",
    "NAXIS3 ",
    "EXTEND ",
    "BZERO ",
    "BSCALE ",
    "COMMENT   FITS (Flexible Image Transport System) format",
    "COMMENT   and Astrophysics', volume 376, page 359; bibcode:",
    "XTENSION",
    "PCOUNT ",
    "GCOUNT ",
];

#[derive(Clone, Copy)]
struct FitsType {
    bitpix: c_int,
    datatype: c_int,
    band_format: BandFormatId,
}

impl FitsType {
    const fn from_bitpix(bitpix: c_int) -> Option<Self> {
        match bitpix {
            x if x == fitsio_sys::BYTE_IMG as c_int => Some(Self {
                bitpix: x,
                datatype: fitsio_sys::TBYTE as c_int,
                band_format: BandFormatId::U8,
            }),
            x if x == fitsio_sys::SHORT_IMG as c_int => Some(Self {
                bitpix: x,
                datatype: fitsio_sys::TSHORT as c_int,
                band_format: BandFormatId::I16,
            }),
            x if x == fitsio_sys::LONG_IMG as c_int => Some(Self {
                bitpix: x,
                datatype: fitsio_sys::TINT as c_int,
                band_format: BandFormatId::I32,
            }),
            x if x == fitsio_sys::FLOAT_IMG as c_int => Some(Self {
                bitpix: x,
                datatype: fitsio_sys::TFLOAT as c_int,
                band_format: BandFormatId::F32,
            }),
            x if x == fitsio_sys::DOUBLE_IMG as c_int => Some(Self {
                bitpix: x,
                datatype: fitsio_sys::TDOUBLE as c_int,
                band_format: BandFormatId::F64,
            }),
            _ => None,
        }
    }

    const fn from_band_format(id: BandFormatId) -> Option<Self> {
        match id {
            BandFormatId::U8 => Some(Self {
                bitpix: fitsio_sys::BYTE_IMG as c_int,
                datatype: fitsio_sys::TBYTE as c_int,
                band_format: BandFormatId::U8,
            }),
            BandFormatId::I16 => Some(Self {
                bitpix: fitsio_sys::SHORT_IMG as c_int,
                datatype: fitsio_sys::TSHORT as c_int,
                band_format: BandFormatId::I16,
            }),
            BandFormatId::I32 => Some(Self {
                bitpix: fitsio_sys::LONG_IMG as c_int,
                datatype: fitsio_sys::TINT as c_int,
                band_format: BandFormatId::I32,
            }),
            BandFormatId::F32 => Some(Self {
                bitpix: fitsio_sys::FLOAT_IMG as c_int,
                datatype: fitsio_sys::TFLOAT as c_int,
                band_format: BandFormatId::F32,
            }),
            BandFormatId::F64 => Some(Self {
                bitpix: fitsio_sys::DOUBLE_IMG as c_int,
                datatype: fitsio_sys::TDOUBLE as c_int,
                band_format: BandFormatId::F64,
            }),
            BandFormatId::U16 | BandFormatId::U32 => None,
        }
    }
}

struct FitsHeader {
    image_type: FitsType,
    width: u32,
    height: u32,
    bands: u32,
    metadata: ImageMetadata,
}

struct FitsFile(*mut fitsio_sys::fitsfile);

impl FitsFile {
    const fn as_ptr(&self) -> *mut fitsio_sys::fitsfile {
        self.0
    }
}

impl Drop for FitsFile {
    fn drop(&mut self) {
        if self.0.is_null() {
            return;
        }
        let mut status = 0;
        // SAFETY: self.0 is a valid FITS handle created by cfitsio open/create calls.
        unsafe {
            fitsio_sys::ffclos(self.0, &raw mut status);
        }
        self.0 = ptr::null_mut();
    }
}

struct TempFitsPath {
    path: PathBuf,
}

impl TempFitsPath {
    fn new(kind: &str) -> Result<Self, ViprsError> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("fits-codec-tmp");
        fs::create_dir_all(&dir).map_err(ViprsError::from)?;

        let id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let filename = format!("{kind}-{}-{}.fits", std::process::id(), id);
        Ok(Self {
            path: dir.join(filename),
        })
    }

    fn as_path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempFitsPath {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn fits_error(prefix: &str, status: c_int) -> ViprsError {
    let mut err_buf = [0i8; FITS_HEADER_CARD_LEN];
    // SAFETY: ffgerr writes a NUL-terminated status message into `err_buf`.
    unsafe {
        fitsio_sys::ffgerr(status, err_buf.as_mut_ptr());
    }
    // SAFETY: `ffgerr` guarantees a C string in `err_buf`.
    let message = unsafe { CStr::from_ptr(err_buf.as_ptr()) }
        .to_string_lossy()
        .trim()
        .to_owned();
    ViprsError::Codec(format!("fits: {prefix}: {message} (status {status})"))
}

fn check_status(prefix: &str, status: c_int) -> Result<(), ViprsError> {
    if status == 0 {
        Ok(())
    } else {
        Err(fits_error(prefix, status))
    }
}

fn to_c_int(value: usize, what: &str) -> Result<c_int, ViprsError> {
    c_int::try_from(value).map_err(|_| {
        ViprsError::Codec(format!(
            "fits: {what} value {value} does not fit c_int on this platform"
        ))
    })
}

fn to_c_long(value: u32) -> c_long {
    value.into()
}

const fn interpretation_for(bands: u32, format: BandFormatId) -> Interpretation {
    match (bands, format) {
        (1, BandFormatId::U16) => Interpretation::Grey16,
        (1, _) => Interpretation::BW,
        (3, BandFormatId::U8) => Interpretation::Srgb,
        (3, BandFormatId::I16 | BandFormatId::U16) => Interpretation::Rgb16,
        _ => Interpretation::Multiband,
    }
}

fn open_fits_for_read(path: &Path) -> Result<FitsFile, ViprsError> {
    let path_c = CString::new(path.to_string_lossy().as_bytes())
        .map_err(|_| ViprsError::Codec("fits: path contains interior NUL byte".into()))?;
    let mut fptr = ptr::null_mut();
    let mut status = 0;

    // SAFETY: `path_c` is a valid C string and `fptr`/`status` are valid out-pointers.
    unsafe {
        fitsio_sys::ffopen(
            &raw mut fptr,
            path_c.as_ptr(),
            fitsio_sys::READONLY as c_int,
            &raw mut status,
        );
    }
    check_status("opening FITS file", status)?;
    Ok(FitsFile(fptr))
}

fn create_fits_for_write(path: &Path) -> Result<FitsFile, ViprsError> {
    let path_c = CString::new(path.to_string_lossy().as_bytes())
        .map_err(|_| ViprsError::Codec("fits: path contains interior NUL byte".into()))?;
    let mut fptr = ptr::null_mut();
    let mut status = 0;

    // SAFETY: `path_c` is a valid C string and `fptr`/`status` are valid out-pointers.
    unsafe {
        fitsio_sys::ffinit(&raw mut fptr, path_c.as_ptr(), &raw mut status);
    }
    check_status("creating FITS file", status)?;
    Ok(FitsFile(fptr))
}

fn parse_hdu_header(fptr: *mut fitsio_sys::fitsfile) -> Result<FitsHeader, ViprsError> {
    let mut status = 0;
    let mut imgtype = 0;
    let mut naxis = 0;
    let mut naxes = [0 as c_long; MAX_NAXIS];

    // SAFETY: `fptr` is a valid open FITS handle and all out-pointers are valid.
    unsafe {
        fitsio_sys::ffgipr(
            fptr,
            to_c_int(MAX_NAXIS, "maxaxis")?,
            &raw mut imgtype,
            &raw mut naxis,
            naxes.as_mut_ptr(),
            &raw mut status,
        );
    }
    check_status("reading FITS image parameters", status)?;

    let naxis_usize = usize::try_from(naxis)
        .map_err(|_| ViprsError::Codec(format!("fits: invalid NAXIS value {naxis}")))?;
    if naxis_usize != 2 && naxis_usize != 3 {
        return Err(ViprsError::Codec(format!(
            "fits: unsupported NAXIS={naxis}; expected 2 or 3"
        )));
    }

    let image_type = FitsType::from_bitpix(imgtype)
        .ok_or_else(|| ViprsError::Codec(format!("fits: unsupported BITPIX value {imgtype}")))?;

    let width = u32::try_from(naxes[0])
        .map_err(|_| ViprsError::Codec(format!("fits: invalid NAXIS1 value {}", naxes[0])))?;
    let height = u32::try_from(naxes[1])
        .map_err(|_| ViprsError::Codec(format!("fits: invalid NAXIS2 value {}", naxes[1])))?;
    let bands = if naxis_usize == 3 {
        u32::try_from(naxes[2])
            .map_err(|_| ViprsError::Codec(format!("fits: invalid NAXIS3 value {}", naxes[2])))?
    } else {
        1
    };

    if width == 0 || height == 0 || bands == 0 {
        return Err(ViprsError::Codec(format!(
            "fits: invalid dimensions {width}x{height}x{bands}"
        )));
    }

    let mut key_count = 0;
    let mut more_keys = 0;
    let mut metadata = ImageMetadata::default();

    status = 0;
    // SAFETY: `fptr` is valid and output pointers are initialized.
    unsafe {
        fitsio_sys::ffghsp(
            fptr,
            &raw mut key_count,
            &raw mut more_keys,
            &raw mut status,
        );
    }
    check_status("reading FITS header key count", status)?;

    for idx in 1..=key_count {
        let mut card = [0i8; FITS_HEADER_CARD_LEN];
        status = 0;
        // SAFETY: `card` has space for the 80-char record + NUL terminator.
        unsafe {
            fitsio_sys::ffgrec(fptr, idx, card.as_mut_ptr(), &raw mut status);
        }
        check_status("reading FITS header record", status)?;

        // SAFETY: `ffgrec` writes a NUL-terminated card string.
        let card_text = unsafe { CStr::from_ptr(card.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        metadata
            .extra
            .insert(format!("fits-{}", idx - 1), card_text);
    }

    Ok(FitsHeader {
        image_type,
        width,
        height,
        bands,
        metadata,
    })
}

fn read_planar_pixels<T: Pod + Zeroable>(
    fptr: *mut fitsio_sys::fitsfile,
    datatype: c_int,
    pixel_count: usize,
) -> Result<Vec<T>, ViprsError> {
    let mut data = vec![T::zeroed(); pixel_count];
    let mut any_null = 0;
    let mut status = 0;

    // SAFETY: `data` is a valid writable buffer of `pixel_count` typed elements.
    unsafe {
        fitsio_sys::ffgpv(
            fptr,
            datatype,
            1,
            pixel_count as i64,
            ptr::null_mut(),
            data.as_mut_ptr().cast::<c_void>(),
            &raw mut any_null,
            &raw mut status,
        );
    }
    check_status("reading FITS pixel data", status)?;
    Ok(data)
}

fn write_planar_pixels<T: Pod>(
    fptr: *mut fitsio_sys::fitsfile,
    datatype: c_int,
    pixels: &[T],
) -> Result<(), ViprsError> {
    let mut status = 0;
    // SAFETY: `pixels` points to `nelem` contiguous typed samples to write.
    unsafe {
        fitsio_sys::ffppr(
            fptr,
            datatype,
            1,
            pixels.len() as i64,
            pixels.as_ptr().cast::<c_void>().cast_mut(),
            &raw mut status,
        );
    }
    check_status("writing FITS pixel data", status)
}

fn write_metadata_records(
    fptr: *mut fitsio_sys::fitsfile,
    metadata: &ImageMetadata,
) -> Result<(), ViprsError> {
    let mut entries = metadata
        .extra
        .iter()
        .filter_map(|(key, value)| {
            key.strip_prefix("fits-")
                .and_then(|suffix| suffix.parse::<usize>().ok())
                .map(|idx| (idx, value.as_str()))
        })
        .collect::<Vec<_>>();
    entries.sort_by_key(|(idx, _)| *idx);

    for (_, card) in entries {
        if FITS_BASIC_PREFIXES
            .iter()
            .any(|prefix| card.starts_with(prefix))
        {
            continue;
        }

        let card_c = CString::new(card).map_err(|_| {
            ViprsError::Codec("fits: metadata card contains interior NUL byte".into())
        })?;
        let mut status = 0;
        // SAFETY: `card_c` is a valid NUL-terminated FITS header card string.
        unsafe {
            fitsio_sys::ffprec(fptr, card_c.as_ptr(), &raw mut status);
        }
        check_status("writing FITS header metadata", status)?;
    }

    Ok(())
}

fn decode_pixels_as<T, F>(
    fptr: *mut fitsio_sys::fitsfile,
    datatype: c_int,
    pixel_count: usize,
    width: usize,
    height: usize,
    bands: usize,
    label: &str,
) -> Result<Vec<F::Sample>, ViprsError>
where
    T: Pod + Zeroable,
    F: BandFormat,
{
    let planar = read_planar_pixels::<T>(fptr, datatype, pixel_count)?;
    let interleaved = planar_to_interleaved_flipped(&planar, width, height, bands);
    try_cast_vec(interleaved)
        .map_err(|_| ViprsError::Codec(format!("fits: failed to cast {label} buffer")))
}

fn encode_pixels_as<T, F>(
    fptr: *mut fitsio_sys::fitsfile,
    datatype: c_int,
    image: &Image<F>,
    width: usize,
    height: usize,
    bands: usize,
    label: &str,
) -> Result<(), ViprsError>
where
    T: Pod,
    F: BandFormat,
{
    let typed = bytemuck::try_cast_slice::<F::Sample, T>(image.pixels())
        .map_err(|_| ViprsError::Codec(format!("fits: failed to cast {label} pixels")))?;
    let planar = interleaved_to_planar_flipped(typed, width, height, bands);
    write_planar_pixels(fptr, datatype, &planar)
}

fn planar_to_interleaved_flipped<T: Copy>(
    src: &[T],
    width: usize,
    height: usize,
    bands: usize,
) -> Vec<T> {
    let mut out = vec![src[0]; src.len()];
    if bands == 1 {
        let row_len = width;
        for y in 0..height {
            let src_offset = y * row_len;
            let dst_offset = (height - 1 - y) * row_len;
            out[dst_offset..dst_offset + row_len]
                .copy_from_slice(&src[src_offset..src_offset + row_len]);
        }
        return out;
    }

    for band in 0..bands {
        for y in 0..height {
            let src_row = (band * height + y) * width;
            let dst_y = height - 1 - y;
            for x in 0..width {
                let dst_idx = (dst_y * width + x) * bands + band;
                out[dst_idx] = src[src_row + x];
            }
        }
    }

    out
}

fn interleaved_to_planar_flipped<T: Copy>(
    src: &[T],
    width: usize,
    height: usize,
    bands: usize,
) -> Vec<T> {
    let mut out = vec![src[0]; src.len()];
    if bands == 1 {
        let row_len = width;
        for y in 0..height {
            let dst_offset = y * row_len;
            let src_offset = (height - 1 - y) * row_len;
            out[dst_offset..dst_offset + row_len]
                .copy_from_slice(&src[src_offset..src_offset + row_len]);
        }
        return out;
    }

    for band in 0..bands {
        for y in 0..height {
            let src_y = height - 1 - y;
            let dst_row = (band * height + y) * width;
            for x in 0..width {
                let src_idx = (src_y * width + x) * bands + band;
                out[dst_row + x] = src[src_idx];
            }
        }
    }

    out
}

fn sniff_fits(header: &[u8]) -> bool {
    header.starts_with(SIMPLE_TRUE_MAGIC)
}

/// The `FitsCodec` type provides concrete adapter functionality in the `codecs` module.
/// Use this type when you need the runtime behavior implemented by this adapter.
///
/// # Examples
///
/// ```rust
/// let _ = core::mem::size_of::<viprs_codecs::fits::FitsCodec>();
/// ```
pub struct FitsCodec;

impl FitsCodec {
    /// `decode_fits` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// let _ = viprs_codecs::fits::FitsCodec::decode_fits::<viprs_core::format::U8>;
    /// ```
    pub fn decode_fits<F: BandFormat>(&self, src: &[u8]) -> Result<Image<F>, ViprsError> {
        let temp = TempFitsPath::new("decode")?;
        fs::write(temp.as_path(), src)?;

        let fits = open_fits_for_read(temp.as_path())?;
        let header = parse_hdu_header(fits.as_ptr())?;

        if header.image_type.band_format != F::ID {
            return Err(ViprsError::Codec(format!(
                "fits: BITPIX {} maps to {:?}, but caller requested {:?}",
                header.image_type.bitpix,
                header.image_type.band_format,
                F::ID
            )));
        }

        let pixel_count = (header.width as usize)
            .checked_mul(header.height as usize)
            .and_then(|v| v.checked_mul(header.bands as usize))
            .ok_or_else(|| {
                ViprsError::Codec(format!(
                    "fits: dimensions overflow {}x{}x{}",
                    header.width, header.height, header.bands
                ))
            })?;

        let width = header.width as usize;
        let height = header.height as usize;
        let bands = header.bands as usize;

        let pixels: Vec<F::Sample> = match header.image_type.band_format {
            BandFormatId::U8 => decode_pixels_as::<u8, F>(
                fits.as_ptr(),
                header.image_type.datatype,
                pixel_count,
                width,
                height,
                bands,
                "U8",
            )?,
            BandFormatId::I16 => decode_pixels_as::<i16, F>(
                fits.as_ptr(),
                header.image_type.datatype,
                pixel_count,
                width,
                height,
                bands,
                "I16",
            )?,
            BandFormatId::I32 => decode_pixels_as::<i32, F>(
                fits.as_ptr(),
                header.image_type.datatype,
                pixel_count,
                width,
                height,
                bands,
                "I32",
            )?,
            BandFormatId::F32 => decode_pixels_as::<f32, F>(
                fits.as_ptr(),
                header.image_type.datatype,
                pixel_count,
                width,
                height,
                bands,
                "F32",
            )?,
            BandFormatId::F64 => decode_pixels_as::<f64, F>(
                fits.as_ptr(),
                header.image_type.datatype,
                pixel_count,
                width,
                height,
                bands,
                "F64",
            )?,
            other => {
                return Err(ViprsError::Codec(format!(
                    "fits: unsupported decoded band format {other:?}"
                )));
            }
        };

        let mut metadata = header.metadata;
        metadata.interpretation = Some(interpretation_for(
            header.bands,
            header.image_type.band_format,
        ));

        Image::from_buffer(header.width, header.height, header.bands, pixels)
            .map(|img| img.with_metadata(metadata))
            .map_err(|err| ViprsError::Codec(err.to_string()))
    }
}

impl ImageDecoder for FitsCodec {
    fn format_name(&self) -> &'static str {
        "fits"
    }

    fn sniff(&self, header: &[u8]) -> bool
    where
        Self: Sized,
    {
        sniff_fits(header)
    }

    fn decode<F: BandFormat>(&self, src: &[u8]) -> Result<Image<F>, ViprsError> {
        self.decode_fits(src)
    }

    fn decode_with_options<F: BandFormat>(
        &self,
        src: &[u8],
        _opts: &LoadOptions,
    ) -> Result<Image<F>, ViprsError>
    where
        Self: Sized,
    {
        self.decode_fits(src)
    }

    fn probe(&self, src: &[u8]) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized,
    {
        let temp = TempFitsPath::new("probe")?;
        fs::write(temp.as_path(), src)?;

        let fits = open_fits_for_read(temp.as_path())?;
        let header = parse_hdu_header(fits.as_ptr())?;
        Ok((header.width, header.height, header.bands))
    }
}

impl ImageEncoder for FitsCodec {
    fn format_name(&self) -> &'static str {
        "fits"
    }

    fn encode<F: BandFormat>(&self, image: &Image<F>) -> Result<Vec<u8>, ViprsError> {
        let fits_type = FitsType::from_band_format(F::ID).ok_or_else(|| {
            ViprsError::Codec(format!("fits: unsupported save format {:?}", F::ID))
        })?;

        let temp = TempFitsPath::new("encode")?;
        let fits = create_fits_for_write(temp.as_path())?;

        let mut naxes = [0 as c_long; 3];
        naxes[0] = to_c_long(image.width());
        naxes[1] = to_c_long(image.height());

        let naxis = if image.bands() == 1 {
            2
        } else {
            naxes[2] = to_c_long(image.bands());
            3
        };

        let mut status = 0;
        // SAFETY: `fits` is a writable FITS handle and `naxes` carries valid dimensions.
        unsafe {
            fitsio_sys::ffphpr(
                fits.as_ptr(),
                1,
                fits_type.bitpix,
                naxis,
                naxes.as_mut_ptr(),
                0,
                1,
                1,
                &raw mut status,
            );
        }
        check_status("creating FITS image header", status)?;

        write_metadata_records(fits.as_ptr(), image.metadata())?;

        let width = image.width() as usize;
        let height = image.height() as usize;
        let bands = image.bands() as usize;

        match fits_type.band_format {
            BandFormatId::U8 => encode_pixels_as::<u8, F>(
                fits.as_ptr(),
                fits_type.datatype,
                image,
                width,
                height,
                bands,
                "U8",
            )?,
            BandFormatId::I16 => encode_pixels_as::<i16, F>(
                fits.as_ptr(),
                fits_type.datatype,
                image,
                width,
                height,
                bands,
                "I16",
            )?,
            BandFormatId::I32 => encode_pixels_as::<i32, F>(
                fits.as_ptr(),
                fits_type.datatype,
                image,
                width,
                height,
                bands,
                "I32",
            )?,
            BandFormatId::F32 => encode_pixels_as::<f32, F>(
                fits.as_ptr(),
                fits_type.datatype,
                image,
                width,
                height,
                bands,
                "F32",
            )?,
            BandFormatId::F64 => encode_pixels_as::<f64, F>(
                fits.as_ptr(),
                fits_type.datatype,
                image,
                width,
                height,
                bands,
                "F64",
            )?,
            other => {
                return Err(ViprsError::Codec(format!(
                    "fits: unsupported encoded band format {other:?}"
                )));
            }
        }

        drop(fits);
        fs::read(temp.as_path()).map_err(ViprsError::from)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;
    use viprs_core::codec_options::{LoadOptions, SaveOptions};
    use viprs_core::format::{BandFormatId, F32, F64, I16, I32, U8, U16, U32};

    fn create_header_only_file(
        bitpix: c_int,
        naxis: c_int,
        naxes: &mut [c_long; 3],
    ) -> TempFitsPath {
        let temp = TempFitsPath::new("header-only").expect("temp fits path");
        let fits = create_fits_for_write(temp.as_path()).expect("create fits file");
        let mut status = 0;
        // SAFETY: `fits` is a valid writable FITS handle and `naxes` points to initialized dimensions.
        unsafe {
            fitsio_sys::ffphpr(
                fits.as_ptr(),
                1,
                bitpix,
                naxis,
                naxes.as_mut_ptr(),
                0,
                1,
                1,
                &mut status,
            );
        }
        check_status("creating FITS image header", status).expect("write header");
        drop(fits);
        temp
    }

    #[test]
    fn sniff_recognises_simple_true_magic() {
        let mut header = [b' '; 80];
        header[..SIMPLE_TRUE_MAGIC.len()].copy_from_slice(SIMPLE_TRUE_MAGIC);
        assert!(FitsCodec.sniff(&header));
    }

    #[test]
    fn sniff_rejects_non_fits_header() {
        assert!(!FitsCodec.sniff(b"NOTFITS"));
    }

    #[test]
    fn roundtrip_u8_pixels_identity() {
        let pixels: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let image = Image::<U8>::from_buffer(2, 2, 3, pixels.clone()).expect("valid input image");

        let encoded = FitsCodec.encode(&image).expect("encode should succeed");
        assert!(FitsCodec.sniff(&encoded));

        let decoded = FitsCodec
            .decode::<U8>(&encoded)
            .expect("decode should succeed");
        assert_eq!(decoded.width(), 2);
        assert_eq!(decoded.height(), 2);
        assert_eq!(decoded.bands(), 3);
        assert_eq!(decoded.pixels(), pixels.as_slice());
        assert_eq!(
            decoded.metadata().interpretation,
            Some(Interpretation::Srgb)
        );
    }

    #[test]
    fn roundtrip_f32_pixels_identity() {
        let pixels: Vec<f32> = vec![0.0, 1.25, -2.5, 3.75, 4.5, -6.0];
        let image = Image::<F32>::from_buffer(3, 2, 1, pixels.clone()).expect("valid input image");

        let encoded = FitsCodec.encode(&image).expect("encode should succeed");
        let decoded = FitsCodec
            .decode::<F32>(&encoded)
            .expect("decode should succeed");

        assert_eq!(decoded.width(), 3);
        assert_eq!(decoded.height(), 2);
        assert_eq!(decoded.bands(), 1);
        assert_eq!(decoded.pixels(), pixels.as_slice());
        assert_eq!(decoded.metadata().interpretation, Some(Interpretation::BW));
    }

    #[test]
    fn roundtrip_i16_pixels_identity() {
        let pixels: Vec<i16> = vec![-4, -3, -2, -1, 1, 2, 3, 4];
        let image = Image::<I16>::from_buffer(2, 2, 2, pixels.clone()).expect("valid input image");

        let encoded = FitsCodec.encode(&image).expect("encode should succeed");
        let decoded = FitsCodec
            .decode::<I16>(&encoded)
            .expect("decode should succeed");

        assert_eq!(decoded.width(), 2);
        assert_eq!(decoded.height(), 2);
        assert_eq!(decoded.bands(), 2);
        assert_eq!(decoded.pixels(), pixels.as_slice());
        assert_eq!(
            decoded.metadata().interpretation,
            Some(Interpretation::Multiband)
        );
    }

    #[test]
    fn roundtrip_i32_pixels_identity() {
        let pixels: Vec<i32> = vec![-40, -30, -20, -10, 10, 20, 30, 40];
        let image = Image::<I32>::from_buffer(2, 2, 2, pixels.clone()).expect("valid input image");

        let encoded = FitsCodec.encode(&image).expect("encode should succeed");
        let decoded = FitsCodec
            .decode::<I32>(&encoded)
            .expect("decode should succeed");

        assert_eq!(decoded.width(), 2);
        assert_eq!(decoded.height(), 2);
        assert_eq!(decoded.bands(), 2);
        assert_eq!(decoded.pixels(), pixels.as_slice());
    }

    #[test]
    fn roundtrip_f64_pixels_identity() {
        let pixels: Vec<f64> = vec![-4.0, -1.5, 0.0, 1.25, 2.5, 4.75, 8.0, 16.0];
        let image = Image::<F64>::from_buffer(2, 2, 2, pixels.clone()).expect("valid input image");

        let encoded = FitsCodec.encode(&image).expect("encode should succeed");
        let decoded = FitsCodec
            .decode::<F64>(&encoded)
            .expect("decode should succeed");

        assert_eq!(decoded.width(), 2);
        assert_eq!(decoded.height(), 2);
        assert_eq!(decoded.bands(), 2);
        assert_eq!(decoded.pixels(), pixels.as_slice());
    }

    #[test]
    fn probe_reports_dimensions() {
        let image = Image::<U8>::from_buffer(4, 3, 1, vec![7u8; 12]).expect("valid input image");
        let encoded = FitsCodec.encode(&image).expect("encode should succeed");
        assert_eq!(
            FitsCodec.probe(&encoded).expect("probe should succeed"),
            (4, 3, 1)
        );
    }

    #[test]
    fn decode_rejects_requested_type_mismatch() {
        let image = Image::<U8>::from_buffer(2, 2, 1, vec![1u8, 2, 3, 4]).expect("valid image");
        let encoded = FitsCodec.encode(&image).expect("encode should succeed");
        let err = FitsCodec
            .decode::<F32>(&encoded)
            .expect_err("must reject mismatched output type");
        let message = err.to_string();
        assert!(
            message.contains("caller requested"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn encode_rejects_unsigned_formats() {
        let u16_image = Image::<U16>::from_buffer(2, 1, 1, vec![1u16, 2]).expect("valid image");
        let u16_err = FitsCodec
            .encode(&u16_image)
            .expect_err("u16 encode must fail");
        assert!(u16_err.to_string().contains("unsupported save format"));

        let u32_image = Image::<U32>::from_buffer(2, 1, 1, vec![1u32, 2]).expect("valid image");
        let u32_err = FitsCodec
            .encode(&u32_image)
            .expect_err("u32 encode must fail");
        assert!(u32_err.to_string().contains("unsupported save format"));
    }

    #[test]
    fn decode_with_options_and_probe_path_work() {
        let pixels: Vec<f32> = vec![0.5, 1.5, 2.5, 3.5];
        let image = Image::<F32>::from_buffer(2, 2, 1, pixels.clone()).expect("valid image");
        let encoded = FitsCodec
            .encode_with_options(&image, &SaveOptions::default())
            .expect("encode_with_options should succeed");

        let decoded = FitsCodec
            .decode_with_options::<F32>(&encoded, &LoadOptions::default())
            .expect("decode_with_options should succeed");
        assert_eq!(decoded.pixels(), pixels.as_slice());

        let temp = TempFitsPath::new("probe-path").expect("temp fits path");
        fs::write(temp.as_path(), &encoded).expect("write encoded fits");
        assert_eq!(
            FitsCodec
                .probe_path(temp.as_path())
                .expect("probe_path should succeed"),
            (2, 2, 1)
        );
        let decoded_path = FitsCodec
            .decode_path::<F32>(temp.as_path())
            .expect("decode_path should succeed");
        assert_eq!(decoded_path.pixels(), pixels.as_slice());
    }

    #[test]
    fn metadata_roundtrip_preserves_custom_cards() {
        let mut metadata = ImageMetadata::default();
        metadata
            .extra
            .insert("fits-40".into(), "COMMENT viprs metadata roundtrip".into());
        metadata
            .extra
            .insert("fits-41".into(), "HISTORY created by tests".into());
        let image = Image::<F32>::from_buffer(2, 1, 1, vec![1.0, 2.0])
            .expect("valid image")
            .with_metadata(metadata);

        let encoded = FitsCodec.encode(&image).expect("encode should succeed");
        let decoded = FitsCodec
            .decode::<F32>(&encoded)
            .expect("decode should succeed");

        assert!(
            decoded
                .metadata()
                .extra
                .values()
                .any(|value| value == "COMMENT viprs metadata roundtrip")
        );
        assert!(
            decoded
                .metadata()
                .extra
                .values()
                .any(|value| value == "HISTORY created by tests")
        );
    }

    #[test]
    fn fits_type_maps_supported_bitpix_values() {
        let cases = [
            (
                fitsio_sys::BYTE_IMG as c_int,
                fitsio_sys::TBYTE as c_int,
                BandFormatId::U8,
            ),
            (
                fitsio_sys::SHORT_IMG as c_int,
                fitsio_sys::TSHORT as c_int,
                BandFormatId::I16,
            ),
            (
                fitsio_sys::LONG_IMG as c_int,
                fitsio_sys::TINT as c_int,
                BandFormatId::I32,
            ),
            (
                fitsio_sys::FLOAT_IMG as c_int,
                fitsio_sys::TFLOAT as c_int,
                BandFormatId::F32,
            ),
            (
                fitsio_sys::DOUBLE_IMG as c_int,
                fitsio_sys::TDOUBLE as c_int,
                BandFormatId::F64,
            ),
        ];

        for (bitpix, datatype, band_format) in cases {
            let fits_type = FitsType::from_bitpix(bitpix).expect("supported BITPIX");
            assert_eq!(fits_type.bitpix, bitpix);
            assert_eq!(fits_type.datatype, datatype);
            assert_eq!(fits_type.band_format, band_format);
        }
    }

    #[test]
    fn fits_type_rejects_unsupported_formats() {
        assert!(FitsType::from_bitpix(123).is_none());
        assert!(FitsType::from_band_format(BandFormatId::U16).is_none());
        assert!(FitsType::from_band_format(BandFormatId::U32).is_none());
    }

    #[test]
    fn decoder_and_encoder_report_format_name() {
        assert_eq!(ImageDecoder::format_name(&FitsCodec), "fits");
        assert_eq!(ImageEncoder::format_name(&FitsCodec), "fits");
    }

    #[test]
    fn planar_and_interleaved_conversions_flip_y_axis() {
        let planar = vec![1, 2, 3, 4, 10, 11, 12, 13];
        let interleaved = planar_to_interleaved_flipped(&planar, 2, 2, 2);
        assert_eq!(interleaved, vec![3, 12, 4, 13, 1, 10, 2, 11]);
        assert_eq!(interleaved_to_planar_flipped(&interleaved, 2, 2, 2), planar);

        let grey = vec![1, 2, 3, 4, 5, 6];
        let flipped = planar_to_interleaved_flipped(&grey, 3, 2, 1);
        assert_eq!(flipped, vec![4, 5, 6, 1, 2, 3]);
        assert_eq!(interleaved_to_planar_flipped(&flipped, 3, 2, 1), grey);
    }

    #[test]
    fn interpretation_mapping_matches_band_layout() {
        assert_eq!(interpretation_for(1, BandFormatId::U8), Interpretation::BW);
        assert_eq!(
            interpretation_for(1, BandFormatId::U16),
            Interpretation::Grey16
        );
        assert_eq!(
            interpretation_for(3, BandFormatId::U8),
            Interpretation::Srgb
        );
        assert_eq!(
            interpretation_for(3, BandFormatId::I16),
            Interpretation::Rgb16
        );
        assert_eq!(
            interpretation_for(4, BandFormatId::F32),
            Interpretation::Multiband
        );
    }

    #[test]
    fn create_and_open_reject_paths_with_interior_nul() {
        let invalid = Path::new(OsStr::from_bytes(b"bad\0path.fits"));

        let open_err = match open_fits_for_read(invalid) {
            Ok(_) => panic!("open must reject interior NUL"),
            Err(err) => err,
        };
        assert!(open_err.to_string().contains("interior NUL byte"));

        let create_err = match create_fits_for_write(invalid) {
            Ok(_) => panic!("create must reject interior NUL"),
            Err(err) => err,
        };
        assert!(create_err.to_string().contains("interior NUL byte"));
    }

    #[test]
    fn fits_error_and_check_status_report_cfitsio_failures() {
        let err = fits_error("unit-test", fitsio_sys::FILE_NOT_OPENED as c_int);
        let message = err.to_string();
        assert!(message.contains("unit-test"));
        assert!(message.contains("status"));

        let checked = check_status("unit-test", fitsio_sys::FILE_NOT_OPENED as c_int)
            .expect_err("non-zero status must fail");
        assert!(checked.to_string().contains("unit-test"));
    }

    #[test]
    fn to_c_int_reports_overflow() {
        let err = to_c_int(usize::MAX, "huge").expect_err("usize::MAX exceeds c_int");
        assert!(err.to_string().contains("does not fit c_int"));
    }

    #[test]
    fn open_fits_for_read_reports_missing_file() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("fits-codec-tmp")
            .join("missing-file-does-not-exist.fits");
        let err = match open_fits_for_read(&path) {
            Ok(_) => panic!("missing file must fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("opening FITS file"));
    }

    #[test]
    fn create_fits_for_write_reports_existing_file() {
        let temp = TempFitsPath::new("existing").expect("temp fits path");
        fs::write(temp.as_path(), b"occupied").expect("seed existing file");

        let err = match create_fits_for_write(temp.as_path()) {
            Ok(_) => panic!("existing file must fail without overwrite prefix"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("creating FITS file"));
    }

    #[test]
    fn parse_hdu_header_rejects_non_2d_or_3d_images() {
        let mut naxes = [4 as c_long, 0, 0];
        let temp = create_header_only_file(fitsio_sys::FLOAT_IMG as c_int, 1, &mut naxes);
        let fits = open_fits_for_read(temp.as_path()).expect("open generated fits");

        let err = match parse_hdu_header(fits.as_ptr()) {
            Ok(_) => panic!("NAXIS=1 must fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("unsupported NAXIS=1"));
    }

    #[test]
    fn write_metadata_records_skips_reserved_cards_and_rejects_nul() {
        let temp = TempFitsPath::new("metadata-cards").expect("temp fits path");
        let fits = create_fits_for_write(temp.as_path()).expect("create fits file");
        let mut naxes = [1 as c_long, 1, 0];
        let mut status = 0;
        // SAFETY: `fits` is a valid writable FITS handle and `naxes` describes a 1x1 image.
        unsafe {
            fitsio_sys::ffphpr(
                fits.as_ptr(),
                1,
                fitsio_sys::FLOAT_IMG as c_int,
                2,
                naxes.as_mut_ptr(),
                0,
                1,
                1,
                &mut status,
            );
        }
        check_status("creating FITS image header", status).expect("write header");

        let mut metadata = ImageMetadata::default();
        metadata
            .extra
            .insert("fits-0".into(), "SIMPLE  =                    T".into());
        metadata
            .extra
            .insert("fits-1".into(), "COMMENT bad\0card".into());

        let err = write_metadata_records(fits.as_ptr(), &metadata)
            .expect_err("interior NUL in metadata card must fail");
        assert!(
            err.to_string()
                .contains("metadata card contains interior NUL byte")
        );
    }
}

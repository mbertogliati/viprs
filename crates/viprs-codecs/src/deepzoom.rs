//! Deepzoom adapter module.
//!
//! This module provides concrete codec-related infrastructure used by the
//! adapter layer for loading, saving, or normalizing external image formats.

use std::{
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

use viprs_core::{codec_options::SaveOptions, error::ViprsError, format::U8, image::Image};

const DEFAULT_TILE_SIZE: u32 = 254;
const DEFAULT_OVERLAP: u32 = 1;
const DEEPZOOM_EXTENSION_DZI: &str = "dzi";
const DEEPZOOM_EXTENSION_DZ: &str = "dz";
const DEEPZOOM_EXTENSION_SZI: &str = "szi";
const TILE_SUFFIX: &str = "ppm";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeepZoomContainer {
    Filesystem,
    SziZip,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DeepZoomTarget {
    parent: PathBuf,
    base_name: String,
    container: DeepZoomContainer,
    descriptor_path: PathBuf,
    tile_root_path: PathBuf,
    output_path: PathBuf,
}

impl DeepZoomTarget {
    fn from_path(path: &Path) -> Result<Self, ViprsError> {
        let file_name = path
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            .ok_or_else(|| {
                ViprsError::Codec(format!(
                    "deepzoom: output path '{}' must include a UTF-8 filename",
                    path.display()
                ))
            })?;

        let extension = path
            .extension()
            .and_then(std::ffi::OsStr::to_str)
            .map(str::to_ascii_lowercase)
            .ok_or_else(|| {
                ViprsError::Codec(format!(
                    "deepzoom: output path '{}' needs one of .dzi/.dz/.szi",
                    path.display()
                ))
            })?;

        let container = match extension.as_str() {
            DEEPZOOM_EXTENSION_DZI | DEEPZOOM_EXTENSION_DZ => DeepZoomContainer::Filesystem,
            DEEPZOOM_EXTENSION_SZI => DeepZoomContainer::SziZip,
            _ => {
                return Err(ViprsError::Codec(format!(
                    "deepzoom: unsupported extension '.{}' for '{}'",
                    extension,
                    path.display()
                )));
            }
        };

        let parent = path
            .parent()
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
        let stem = Path::new(file_name)
            .file_stem()
            .and_then(std::ffi::OsStr::to_str)
            .ok_or_else(|| {
                ViprsError::Codec(format!(
                    "deepzoom: could not derive base name from '{}'",
                    path.display()
                ))
            })?;

        let descriptor_path = parent.join(format!("{stem}.{DEEPZOOM_EXTENSION_DZI}"));
        let tile_root_path = parent.join(format!("{stem}_files"));

        Ok(Self {
            parent,
            base_name: stem.to_owned(),
            container,
            descriptor_path,
            tile_root_path,
            output_path: path.to_path_buf(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DeepZoomLevel<'a> {
    width: u32,
    height: u32,
    pixels: &'a [u8],
}

#[derive(Debug, PartialEq, Eq)]
struct OwnedDeepZoomLevel {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

enum LevelPixels<'a> {
    Borrowed(&'a [u8]),
    Owned(Vec<u8>),
}

impl LevelPixels<'_> {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Borrowed(pixels) => pixels,
            Self::Owned(pixels) => pixels.as_slice(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
/// The `DeepZoomExporter` type provides concrete adapter functionality in the `codecs` module.
/// Use this type when you need the runtime behavior implemented by this adapter.
///
/// # Examples
///
/// ```rust
/// let _ = core::mem::size_of::<viprs::adapters::codecs::deepzoom::DeepZoomExporter>();
/// ```
pub struct DeepZoomExporter {
    tile_size: u32,
    overlap: u32,
}

impl DeepZoomExporter {
    /// `from_options` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let _ = viprs::adapters::codecs::deepzoom::from_options;
    /// ```
    pub fn from_options(opts: &SaveOptions) -> Result<Self, ViprsError> {
        if let (Some(tile_width), Some(tile_height)) = (opts.tile_width, opts.tile_height)
            && tile_width != tile_height
        {
            return Err(ViprsError::Codec(format!(
                "deepzoom: tile_width ({tile_width}) and tile_height ({tile_height}) must match"
            )));
        }

        let tile_size = opts
            .tile_width
            .or(opts.tile_height)
            .unwrap_or(DEFAULT_TILE_SIZE);
        if tile_size == 0 {
            return Err(ViprsError::Codec(
                "deepzoom: tile size must be greater than zero".to_owned(),
            ));
        }

        Ok(Self {
            tile_size,
            overlap: DEFAULT_OVERLAP,
        })
    }

    /// `export` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let _ = viprs::adapters::codecs::deepzoom::export;
    /// ```
    pub fn export(&self, image: &Image<U8>, path: &Path) -> Result<(), ViprsError> {
        let target = DeepZoomTarget::from_path(path)?;
        let descriptor = descriptor_xml(
            &target.base_name,
            image.width(),
            image.height(),
            self.tile_size,
            self.overlap,
        );

        match target.container {
            DeepZoomContainer::Filesystem => write_filesystem_output(
                &target,
                image.width(),
                image.height(),
                image.bands(),
                image.pixels(),
                &descriptor,
                self.tile_size,
                self.overlap,
            ),
            DeepZoomContainer::SziZip => write_szi_output(
                &target,
                image.width(),
                image.height(),
                image.bands(),
                image.pixels(),
                &descriptor,
                self.tile_size,
                self.overlap,
            ),
        }
    }
}

fn validate_level_input(
    width: u32,
    height: u32,
    bands: u32,
    pixels: &[u8],
) -> Result<usize, ViprsError> {
    let bands_usize = usize::try_from(bands).map_err(|_| {
        ViprsError::Codec(format!(
            "deepzoom: unsupported band count {bands} (does not fit usize)"
        ))
    })?;
    if bands_usize == 0 {
        return Err(ViprsError::Codec(
            "deepzoom: band count must be greater than zero".to_owned(),
        ));
    }

    let expected_len = usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
        .and_then(|px| px.checked_mul(bands_usize))
        .ok_or_else(|| {
            ViprsError::Codec(format!(
                "deepzoom: dimensions overflow usize: {width}x{height}x{bands}"
            ))
        })?;
    if pixels.len() != expected_len {
        return Err(ViprsError::Codec(format!(
            "deepzoom: pixel buffer length {} does not match {width}x{height}x{bands}={expected_len}",
            pixels.len()
        )));
    }

    Ok(bands_usize)
}

fn pyramid_level_count(width: u32, height: u32) -> usize {
    let mut levels = 1usize;
    let mut current_width = width;
    let mut current_height = height;
    while current_width > 1 || current_height > 1 {
        current_width = current_width.div_ceil(2);
        current_height = current_height.div_ceil(2);
        levels += 1;
    }
    levels
}

fn stream_levels<F>(
    width: u32,
    height: u32,
    bands: u32,
    pixels: &[u8],
    mut on_level: F,
) -> Result<(), ViprsError>
where
    F: FnMut(u32, DeepZoomLevel<'_>, usize) -> Result<(), ViprsError>,
{
    let bands_usize = validate_level_input(width, height, bands, pixels)?;
    let mut current_width = width;
    let mut current_height = height;
    let mut current_pixels = LevelPixels::Borrowed(pixels);
    let mut level_number = u32::try_from(pyramid_level_count(width, height) - 1)
        .map_err(|_| ViprsError::Codec("deepzoom: too many pyramid levels".to_owned()))?;

    loop {
        let level = DeepZoomLevel {
            width: current_width,
            height: current_height,
            pixels: current_pixels.as_slice(),
        };
        on_level(level_number, level, level.pixels.len())?;

        if current_width == 1 && current_height == 1 {
            break;
        }

        let next = downsample_half(level, bands_usize)?;
        level_number = level_number
            .checked_sub(1)
            .ok_or_else(|| ViprsError::Codec("deepzoom: level index underflow".to_owned()))?;
        current_width = next.width;
        current_height = next.height;
        current_pixels = LevelPixels::Owned(next.pixels);
    }

    Ok(())
}

fn downsample_half(
    level: DeepZoomLevel<'_>,
    bands: usize,
) -> Result<OwnedDeepZoomLevel, ViprsError> {
    let dst_width = level.width.div_ceil(2);
    let dst_height = level.height.div_ceil(2);
    let dst_len = usize::try_from(dst_width)
        .ok()
        .and_then(|w| {
            usize::try_from(dst_height)
                .ok()
                .and_then(|h| w.checked_mul(h))
        })
        .and_then(|px| px.checked_mul(bands))
        .ok_or_else(|| {
            ViprsError::Codec(format!(
                "deepzoom: dimensions overflow while downsampling {}x{}",
                level.width, level.height
            ))
        })?;
    let mut dst = vec![0u8; dst_len];

    let src_width = usize::try_from(level.width)
        .map_err(|_| ViprsError::Codec("deepzoom: source width overflow".to_owned()))?;
    let src_height = usize::try_from(level.height)
        .map_err(|_| ViprsError::Codec("deepzoom: source height overflow".to_owned()))?;
    let dst_width_usize = usize::try_from(dst_width)
        .map_err(|_| ViprsError::Codec("deepzoom: target width overflow".to_owned()))?;
    let dst_height_usize = usize::try_from(dst_height)
        .map_err(|_| ViprsError::Codec("deepzoom: target height overflow".to_owned()))?;

    for dy in 0..dst_height_usize {
        for dx in 0..dst_width_usize {
            let sx = dx * 2;
            let sy = dy * 2;
            for band in 0..bands {
                let mut sum: u16 = 0;
                let mut count: u16 = 0;
                for oy in 0..2usize {
                    for ox in 0..2usize {
                        let px = sx + ox;
                        let py = sy + oy;
                        if px < src_width && py < src_height {
                            let src_index = ((py * src_width) + px) * bands + band;
                            sum = sum.saturating_add(u16::from(level.pixels[src_index]));
                            count = count.saturating_add(1);
                        }
                    }
                }
                let dst_index = ((dy * dst_width_usize) + dx) * bands + band;
                let rounded = (sum + (count / 2)) / count;
                dst[dst_index] = u8::try_from(rounded).map_err(|_| {
                    ViprsError::Codec("deepzoom: averaged sample overflowed u8".to_owned())
                })?;
            }
        }
    }

    Ok(OwnedDeepZoomLevel {
        width: dst_width,
        height: dst_height,
        pixels: dst,
    })
}

fn descriptor_xml(
    _base_name: &str,
    width: u32,
    height: u32,
    tile_size: u32,
    overlap: u32,
) -> String {
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<Image xmlns=\"http://schemas.microsoft.com/deepzoom/2008\"\n  Format=\"{TILE_SUFFIX}\"\n  Overlap=\"{overlap}\"\n  TileSize=\"{tile_size}\"\n  >\n  <Size \n    Height=\"{height}\"\n    Width=\"{width}\"\n  />\n</Image>\n"
    )
}

fn write_filesystem_output(
    target: &DeepZoomTarget,
    width: u32,
    height: u32,
    bands: u32,
    pixels: &[u8],
    descriptor: &str,
    tile_size: u32,
    overlap: u32,
) -> Result<(), ViprsError> {
    fs::create_dir_all(&target.parent)?;
    fs::create_dir_all(&target.tile_root_path)?;
    fs::write(&target.descriptor_path, descriptor.as_bytes())?;

    stream_levels(width, height, bands, pixels, |level_number, level, _| {
        let level_dir = target.tile_root_path.join(level_number.to_string());
        fs::create_dir_all(&level_dir)?;
        write_level_tiles(
            level,
            tile_size,
            overlap,
            |tile_x, tile_y, bytes| -> Result<(), ViprsError> {
                let tile_path = level_dir.join(format!("{tile_x}_{tile_y}.{TILE_SUFFIX}"));
                fs::write(tile_path, bytes)?;
                Ok(())
            },
        )
    })
}

fn write_szi_output(
    target: &DeepZoomTarget,
    width: u32,
    height: u32,
    bands: u32,
    pixels: &[u8],
    descriptor: &str,
    tile_size: u32,
    overlap: u32,
) -> Result<(), ViprsError> {
    fs::create_dir_all(&target.parent)?;
    let prefix = format!("{}/", target.base_name);
    let mut writer = ZipStreamWriter::create(&target.output_path)?;
    writer.write_entry(
        &format!("{prefix}{}.{}", target.base_name, DEEPZOOM_EXTENSION_DZI),
        descriptor.as_bytes(),
    )?;

    stream_levels(width, height, bands, pixels, |level_number, level, _| {
        write_level_tiles(
            level,
            tile_size,
            overlap,
            |tile_x, tile_y, bytes| -> Result<(), ViprsError> {
                writer.write_entry(
                    &format!(
                        "{prefix}{}_files/{level_number}/{tile_x}_{tile_y}.{TILE_SUFFIX}",
                        target.base_name
                    ),
                    bytes,
                )
            },
        )
    })?;

    writer.finish()
}

fn write_level_tiles<F>(
    level: DeepZoomLevel<'_>,
    tile_size: u32,
    overlap: u32,
    mut writer: F,
) -> Result<(), ViprsError>
where
    F: FnMut(u32, u32, &[u8]) -> Result<(), ViprsError>,
{
    let tiles_across = level.width.div_ceil(tile_size);
    let tiles_down = level.height.div_ceil(tile_size);
    for tile_y in 0..tiles_down {
        for tile_x in 0..tiles_across {
            let tile = extract_tile(level, tile_x, tile_y, tile_size, overlap)?;
            writer(tile_x, tile_y, &tile)?;
        }
    }

    Ok(())
}

fn extract_tile(
    level: DeepZoomLevel<'_>,
    tile_x: u32,
    tile_y: u32,
    tile_size: u32,
    overlap: u32,
) -> Result<Vec<u8>, ViprsError> {
    let base_x = tile_x.saturating_mul(tile_size);
    let base_y = tile_y.saturating_mul(tile_size);
    let left = base_x.saturating_sub(overlap);
    let top = base_y.saturating_sub(overlap);
    let right = base_x
        .saturating_add(tile_size)
        .saturating_add(overlap)
        .min(level.width);
    let bottom = base_y
        .saturating_add(tile_size)
        .saturating_add(overlap)
        .min(level.height);
    let width = right.saturating_sub(left);
    let height = bottom.saturating_sub(top);

    let bands = level
        .pixels
        .len()
        .checked_div(
            usize::try_from(level.width)
                .ok()
                .and_then(|w| {
                    usize::try_from(level.height)
                        .ok()
                        .and_then(|h| w.checked_mul(h))
                })
                .ok_or_else(|| {
                    ViprsError::Codec(format!(
                        "deepzoom: level dimensions overflow {}x{}",
                        level.width, level.height
                    ))
                })?,
        )
        .ok_or_else(|| ViprsError::Codec("deepzoom: invalid level pixel layout".to_owned()))?;

    let mut payload = Vec::new();
    let header = format!("P6\n{width} {height}\n255\n");
    payload.extend_from_slice(header.as_bytes());
    let rgb_len = usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
        .and_then(|px| px.checked_mul(3))
        .ok_or_else(|| {
            ViprsError::Codec(format!(
                "deepzoom: tile dimensions overflow {}x{}",
                width, height
            ))
        })?;
    let mut rgb = vec![0u8; rgb_len];
    let level_width = usize::try_from(level.width)
        .map_err(|_| ViprsError::Codec("deepzoom: level width overflow".to_owned()))?;
    let tile_width = usize::try_from(width)
        .map_err(|_| ViprsError::Codec("deepzoom: tile width overflow".to_owned()))?;
    let tile_height = usize::try_from(height)
        .map_err(|_| ViprsError::Codec("deepzoom: tile height overflow".to_owned()))?;
    let top_usize =
        usize::try_from(top).map_err(|_| ViprsError::Codec("deepzoom: top overflow".to_owned()))?;
    let left_usize = usize::try_from(left)
        .map_err(|_| ViprsError::Codec("deepzoom: left overflow".to_owned()))?;

    for y in 0..tile_height {
        for x in 0..tile_width {
            let src_x = left_usize + x;
            let src_y = top_usize + y;
            let src_pixel = (src_y * level_width + src_x) * bands;
            let dst = (y * tile_width + x) * 3;
            match bands {
                0 => {
                    return Err(ViprsError::Codec(
                        "deepzoom: level has zero bands".to_owned(),
                    ));
                }
                1 => {
                    let gray = level.pixels[src_pixel];
                    rgb[dst] = gray;
                    rgb[dst + 1] = gray;
                    rgb[dst + 2] = gray;
                }
                2 => {
                    rgb[dst] = level.pixels[src_pixel];
                    rgb[dst + 1] = level.pixels[src_pixel + 1];
                    rgb[dst + 2] = level.pixels[src_pixel];
                }
                _ => {
                    rgb[dst] = level.pixels[src_pixel];
                    rgb[dst + 1] = level.pixels[src_pixel + 1];
                    rgb[dst + 2] = level.pixels[src_pixel + 2];
                }
            }
        }
    }
    payload.extend_from_slice(&rgb);

    Ok(payload)
}

#[derive(Debug, Clone)]
struct CentralDirectoryRecord {
    name: Vec<u8>,
    crc32: u32,
    compressed_size: u32,
    uncompressed_size: u32,
    local_header_offset: u32,
}

struct ZipStreamWriter {
    writer: File,
    central_directory: Vec<CentralDirectoryRecord>,
    offset: u32,
}

impl ZipStreamWriter {
    fn create(path: &Path) -> Result<Self, ViprsError> {
        Ok(Self {
            writer: File::create(path)?,
            central_directory: Vec::new(),
            offset: 0,
        })
    }

    fn write_entry(&mut self, name: &str, data: &[u8]) -> Result<(), ViprsError> {
        let name_bytes = name.as_bytes();
        let name_len = u16::try_from(name_bytes.len()).map_err(|_| {
            ViprsError::Codec(format!("deepzoom: zip entry name too long '{}'", name))
        })?;
        let data_len = u32::try_from(data.len()).map_err(|_| {
            ViprsError::Codec(format!(
                "deepzoom: zip entry '{}' exceeds zip32 size limit",
                name
            ))
        })?;
        let crc32 = crc32_ieee(data);

        self.writer.write_all(&0x0403_4b50u32.to_le_bytes())?;
        self.writer.write_all(&20u16.to_le_bytes())?;
        self.writer.write_all(&0u16.to_le_bytes())?;
        self.writer.write_all(&0u16.to_le_bytes())?;
        self.writer.write_all(&0u16.to_le_bytes())?;
        self.writer.write_all(&0u16.to_le_bytes())?;
        self.writer.write_all(&crc32.to_le_bytes())?;
        self.writer.write_all(&data_len.to_le_bytes())?;
        self.writer.write_all(&data_len.to_le_bytes())?;
        self.writer.write_all(&name_len.to_le_bytes())?;
        self.writer.write_all(&0u16.to_le_bytes())?;
        self.writer.write_all(name_bytes)?;
        self.writer.write_all(data)?;

        self.central_directory.push(CentralDirectoryRecord {
            name: name_bytes.to_vec(),
            crc32,
            compressed_size: data_len,
            uncompressed_size: data_len,
            local_header_offset: self.offset,
        });

        let header_len = 30u32 + u32::from(name_len);
        self.offset = self
            .offset
            .checked_add(header_len)
            .and_then(|value| value.checked_add(data_len))
            .ok_or_else(|| ViprsError::Codec("deepzoom: zip offset overflow".to_owned()))?;

        Ok(())
    }

    fn finish(mut self) -> Result<(), ViprsError> {
        let central_offset = self.offset;
        for record in &self.central_directory {
            let name_len = u16::try_from(record.name.len()).map_err(|_| {
                ViprsError::Codec("deepzoom: central record name too long".to_owned())
            })?;
            self.writer.write_all(&0x0201_4b50u32.to_le_bytes())?;
            self.writer.write_all(&20u16.to_le_bytes())?;
            self.writer.write_all(&20u16.to_le_bytes())?;
            self.writer.write_all(&0u16.to_le_bytes())?;
            self.writer.write_all(&0u16.to_le_bytes())?;
            self.writer.write_all(&0u16.to_le_bytes())?;
            self.writer.write_all(&0u16.to_le_bytes())?;
            self.writer.write_all(&record.crc32.to_le_bytes())?;
            self.writer
                .write_all(&record.compressed_size.to_le_bytes())?;
            self.writer
                .write_all(&record.uncompressed_size.to_le_bytes())?;
            self.writer.write_all(&name_len.to_le_bytes())?;
            self.writer.write_all(&0u16.to_le_bytes())?;
            self.writer.write_all(&0u16.to_le_bytes())?;
            self.writer.write_all(&0u16.to_le_bytes())?;
            self.writer.write_all(&0u16.to_le_bytes())?;
            self.writer.write_all(&0u32.to_le_bytes())?;
            self.writer
                .write_all(&record.local_header_offset.to_le_bytes())?;
            self.writer.write_all(&record.name)?;
            self.offset = self
                .offset
                .checked_add(46u32 + u32::from(name_len))
                .ok_or_else(|| ViprsError::Codec("deepzoom: central size overflow".to_owned()))?;
        }

        let central_size = self
            .offset
            .checked_sub(central_offset)
            .ok_or_else(|| ViprsError::Codec("deepzoom: invalid central size".to_owned()))?;
        let entry_count = u16::try_from(self.central_directory.len())
            .map_err(|_| ViprsError::Codec("deepzoom: too many zip entries".to_owned()))?;

        self.writer.write_all(&0x0605_4b50u32.to_le_bytes())?;
        self.writer.write_all(&0u16.to_le_bytes())?;
        self.writer.write_all(&0u16.to_le_bytes())?;
        self.writer.write_all(&entry_count.to_le_bytes())?;
        self.writer.write_all(&entry_count.to_le_bytes())?;
        self.writer.write_all(&central_size.to_le_bytes())?;
        self.writer.write_all(&central_offset.to_le_bytes())?;
        self.writer.write_all(&0u16.to_le_bytes())?;
        self.writer.flush()?;
        Ok(())
    }
}

fn crc32_ieee(bytes: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg() & 0xedb8_8320;
            crc = (crc >> 1) ^ mask;
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;
    use viprs_core::{codec_options::SaveOptions, format::U8};

    #[test]
    fn from_options_rejects_mismatched_tile_dimensions() {
        let err = DeepZoomExporter::from_options(
            &SaveOptions::default()
                .with_tile_width(128)
                .with_tile_height(64),
        )
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("tile_width (128) and tile_height (64) must match")
        );
    }

    #[test]
    fn target_rejects_unsupported_extension() {
        let err = DeepZoomTarget::from_path(Path::new("invalid.txt")).unwrap_err();
        assert!(err.to_string().contains("unsupported extension '.txt'"));
    }

    #[test]
    fn target_requires_filename_and_extension() {
        let missing_filename = DeepZoomTarget::from_path(Path::new("")).unwrap_err();
        assert!(
            missing_filename
                .to_string()
                .contains("must include a UTF-8 filename")
        );

        let missing_extension = DeepZoomTarget::from_path(Path::new("deepzoom")).unwrap_err();
        assert!(
            missing_extension
                .to_string()
                .contains("needs one of .dzi/.dz/.szi")
        );
    }

    #[test]
    fn validate_level_input_rejects_invalid_buffer_length() {
        let err = validate_level_input(4, 4, 3, &[0; 12]).unwrap_err();
        assert!(
            err.to_string()
                .contains("pixel buffer length 12 does not match 4x4x3=48")
        );
    }

    #[test]
    fn validate_level_input_rejects_zero_bands_and_overflowing_dimensions() {
        let zero_bands = validate_level_input(1, 1, 0, &[]).unwrap_err();
        assert!(
            zero_bands
                .to_string()
                .contains("band count must be greater than zero")
        );

        let overflow = validate_level_input(u32::MAX, u32::MAX, 4, &[]).unwrap_err();
        assert!(overflow.to_string().contains("dimensions overflow usize"));
    }

    #[test]
    fn from_options_rejects_zero_tile_size() {
        let err =
            DeepZoomExporter::from_options(&SaveOptions::default().with_tile_width(0)).unwrap_err();
        assert!(
            err.to_string()
                .contains("tile size must be greater than zero")
        );
    }

    #[test]
    fn downsample_half_rounds_odd_edges() {
        let level = DeepZoomLevel {
            width: 3,
            height: 3,
            pixels: &[0, 10, 20, 30, 40, 50, 60, 70, 80],
        };

        let downsampled = downsample_half(level, 1).unwrap();

        assert_eq!(downsampled.width, 2);
        assert_eq!(downsampled.height, 2);
        assert_eq!(downsampled.pixels, vec![20, 35, 65, 80]);
    }

    #[test]
    fn downsample_half_rejects_overflowing_dimensions() {
        let err = downsample_half(
            DeepZoomLevel {
                width: u32::MAX,
                height: u32::MAX,
                pixels: &[0],
            },
            4,
        )
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("dimensions overflow while downsampling")
        );
    }

    #[test]
    fn extract_tile_expands_grayscale_samples() {
        let tile = extract_tile(
            DeepZoomLevel {
                width: 2,
                height: 1,
                pixels: &[10, 200],
            },
            0,
            0,
            2,
            0,
        )
        .unwrap();

        assert_eq!(tile, b"P6\n2 1\n255\n\n\n\n\xc8\xc8\xc8".to_vec());
    }

    #[test]
    fn extract_tile_maps_two_band_samples_to_rgb() {
        let tile = extract_tile(
            DeepZoomLevel {
                width: 1,
                height: 1,
                pixels: &[5, 10],
            },
            0,
            0,
            1,
            0,
        )
        .unwrap();

        assert_eq!(tile, b"P6\n1 1\n255\n\x05\x0a\x05".to_vec());
    }

    #[test]
    fn extract_tile_rejects_zero_band_level() {
        let err = extract_tile(
            DeepZoomLevel {
                width: 1,
                height: 1,
                pixels: &[],
            },
            0,
            0,
            1,
            0,
        )
        .unwrap_err();

        assert!(err.to_string().contains("level has zero bands"));
    }

    #[test]
    fn stream_levels_handles_single_pixel_image() {
        let image = Image::<U8>::from_buffer(1, 1, 1, vec![42]).unwrap();
        let mut seen = Vec::new();

        stream_levels(
            image.width(),
            image.height(),
            image.bands(),
            image.pixels(),
            |level_number, level, active_pixel_bytes| {
                seen.push((level_number, level.width, level.height, active_pixel_bytes));
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(seen, vec![(0, 1, 1, 1)]);
    }

    #[test]
    fn zip_stream_writer_writes_entries_without_buffering_all_tiles() {
        let output_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("deepzoom-unit-tests");
        fs::create_dir_all(&output_dir).unwrap();
        let zip_path = output_dir.join(format!("deepzoom-zip-writer-{}.szi", std::process::id()));

        let mut writer = ZipStreamWriter::create(&zip_path).unwrap();
        writer.write_entry("sample/a.txt", b"alpha").unwrap();
        writer.write_entry("sample/b.txt", b"beta").unwrap();
        writer.finish().unwrap();

        let bytes = fs::read(&zip_path).unwrap();
        assert!(bytes.starts_with(&0x0403_4b50u32.to_le_bytes()));
        assert!(
            bytes
                .windows("sample/a.txt".len())
                .any(|window| window == b"sample/a.txt")
        );
        assert!(
            bytes
                .windows("sample/b.txt".len())
                .any(|window| window == b"sample/b.txt")
        );
        assert!(bytes.windows(5).any(|window| window == b"alpha"));
        assert!(bytes.windows(4).any(|window| window == b"beta"));

        let _ = fs::remove_file(zip_path);
    }

    #[test]
    fn zip_stream_writer_rejects_oversized_entry_names() {
        let output_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("deepzoom-unit-tests");
        fs::create_dir_all(&output_dir).unwrap();
        let zip_path = output_dir.join(format!(
            "deepzoom-zip-writer-error-{}.szi",
            std::process::id()
        ));

        let mut writer = ZipStreamWriter::create(&zip_path).unwrap();
        let long_name = "a".repeat(70_000);
        let err = writer.write_entry(&long_name, b"payload").unwrap_err();
        assert!(err.to_string().contains("zip entry name too long"));

        let _ = fs::remove_file(zip_path);
    }

    #[test]
    fn write_szi_output_rejects_oversized_descriptor_entry_name() {
        let output_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("deepzoom-unit-tests");
        fs::create_dir_all(&output_dir).unwrap();
        let zip_path = output_dir.join(format!(
            "deepzoom-oversized-descriptor-{}.szi",
            std::process::id()
        ));
        let target = DeepZoomTarget {
            parent: output_dir,
            base_name: "a".repeat(70_000),
            container: DeepZoomContainer::SziZip,
            descriptor_path: PathBuf::new(),
            tile_root_path: PathBuf::new(),
            output_path: zip_path.clone(),
        };

        let err = write_szi_output(&target, 1, 1, 1, &[42], "descriptor", 1, 0).unwrap_err();
        assert!(err.to_string().contains("zip entry name too long"));

        let _ = fs::remove_file(zip_path);
    }

    #[test]
    fn stream_levels_only_reports_current_level_pixels() {
        let image = Image::<U8>::from_buffer(5, 3, 1, (0u8..15).collect()).unwrap();
        let mut seen = Vec::new();

        stream_levels(
            image.width(),
            image.height(),
            image.bands(),
            image.pixels(),
            |level_number, level, active_pixel_bytes| {
                seen.push((level_number, level.width, level.height, active_pixel_bytes));
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(
            seen,
            vec![(3, 5, 3, 15), (2, 3, 2, 6), (1, 2, 1, 2), (0, 1, 1, 1)]
        );
        assert!(
            seen.iter()
                .all(|(_, _, _, active_pixel_bytes)| *active_pixel_bytes < 24),
            "streaming levels should not retain the full 24-byte pyramid"
        );
    }

    #[test]
    fn export_writes_descriptor_and_tiles_for_dzi() {
        let output_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("deepzoom-unit-tests");
        fs::create_dir_all(&output_dir).unwrap();
        let descriptor_path =
            output_dir.join(format!("deepzoom-structure-{}.dzi", std::process::id()));
        let tile_root = output_dir.join(format!("deepzoom-structure-{}_files", std::process::id()));

        let image = Image::<U8>::from_buffer(4, 4, 3, (0u8..48).collect()).unwrap();
        let exporter =
            DeepZoomExporter::from_options(&SaveOptions::default().with_tile_width(2)).unwrap();
        exporter.export(&image, &descriptor_path).unwrap();

        let descriptor = fs::read_to_string(&descriptor_path).unwrap();
        assert!(descriptor.contains("TileSize=\"2\""));
        assert!(descriptor.contains("Overlap=\"1\""));
        assert!(descriptor.contains("Width=\"4\""));
        assert!(descriptor.contains("Height=\"4\""));

        let top_level_tile = tile_root.join(format!("2/0_0.{TILE_SUFFIX}"));
        let tile_bytes = fs::read(&top_level_tile).unwrap();
        assert!(tile_bytes.starts_with(b"P6\n"));

        let _ = fs::remove_file(&descriptor_path);
        let _ = fs::remove_dir_all(&tile_root);
    }

    #[test]
    fn export_writes_szi_zip_container() {
        let output_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("deepzoom-unit-tests");
        fs::create_dir_all(&output_dir).unwrap();
        let szi_path = output_dir.join(format!("deepzoom-archive-{}.szi", std::process::id()));

        let image = Image::<U8>::from_buffer(3, 3, 3, (0u8..27).collect()).unwrap();
        let exporter = DeepZoomExporter::from_options(&SaveOptions::default()).unwrap();
        exporter.export(&image, &szi_path).unwrap();

        let bytes = fs::read(&szi_path).unwrap();
        assert!(bytes.starts_with(&0x0403_4b50u32.to_le_bytes()));
        assert!(bytes.windows(4).any(|window| window == b".dzi"));
        assert!(bytes.windows(6).any(|window| window == b"_files"));

        let _ = fs::remove_file(szi_path);
    }

    #[test]
    fn export_writes_expected_tile_layout_for_64x64_image() {
        let output_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("deepzoom-unit-tests");
        fs::create_dir_all(&output_dir).unwrap();
        let descriptor_path = output_dir.join(format!("deepzoom-large-{}.dzi", std::process::id()));
        let tile_root = output_dir.join(format!("deepzoom-large-{}_files", std::process::id()));

        let pixels: Vec<u8> = (0..(64usize * 64usize * 3usize))
            .map(|index| u8::try_from(index % 251).unwrap())
            .collect();
        let image = Image::<U8>::from_buffer(64, 64, 3, pixels).unwrap();
        let exporter =
            DeepZoomExporter::from_options(&SaveOptions::default().with_tile_width(16)).unwrap();

        exporter.export(&image, &descriptor_path).unwrap();

        let descriptor = fs::read_to_string(&descriptor_path).unwrap();
        assert!(descriptor.contains("TileSize=\"16\""));
        assert!(descriptor.contains("Width=\"64\""));
        assert!(descriptor.contains("Height=\"64\""));

        let mut tile_paths = Vec::new();
        for level in 0..=6 {
            let level_dir = tile_root.join(level.to_string());
            if level_dir.exists() {
                for entry in fs::read_dir(level_dir).unwrap() {
                    tile_paths.push(entry.unwrap().path());
                }
            }
        }

        assert_eq!(tile_paths.len(), 25);
        assert_eq!(
            crc32_ieee(&fs::read(tile_root.join(format!("6/0_0.{TILE_SUFFIX}"))).unwrap()),
            0x9752_d9ec
        );
        assert_eq!(
            crc32_ieee(&fs::read(tile_root.join(format!("6/3_3.{TILE_SUFFIX}"))).unwrap()),
            0xc011_c998
        );
        assert_eq!(
            crc32_ieee(&fs::read(tile_root.join(format!("5/1_1.{TILE_SUFFIX}"))).unwrap()),
            0x4a6d_2df1
        );
        assert_eq!(
            crc32_ieee(&fs::read(tile_root.join(format!("0/0_0.{TILE_SUFFIX}"))).unwrap()),
            0x8071_8f72
        );

        let _ = fs::remove_file(&descriptor_path);
        let _ = fs::remove_dir_all(&tile_root);
    }
}

#[allow(clippy::wildcard_imports)]
// REASON: TIFF pyramid helpers rely on many sibling codec constants/types.
use super::*;

pub(super) fn tiff_is_big_endian(bytes: &[u8]) -> Result<bool, ViprsError> {
    if bytes.len() < 4 {
        return Err(ViprsError::Codec("tiff: header is too short".into()));
    }

    match &bytes[..4] {
        magic if magic == TIFF_LE_MAGIC => Ok(false),
        magic if magic == TIFF_BE_MAGIC => Ok(true),
        _ => Err(ViprsError::Codec("tiff: invalid header".into())),
    }
}

pub(super) fn tiff_read_u16(bytes: &[u8], offset: usize) -> Result<u16, ViprsError> {
    let end = offset
        .checked_add(2)
        .ok_or_else(|| ViprsError::Codec("tiff: offset overflow".into()))?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| ViprsError::Codec("tiff: truncated directory entry".into()))?;
    let array = [slice[0], slice[1]];
    Ok(if tiff_is_big_endian(bytes)? {
        u16::from_be_bytes(array)
    } else {
        u16::from_le_bytes(array)
    })
}

pub(super) fn tiff_read_u32(bytes: &[u8], offset: usize) -> Result<u32, ViprsError> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| ViprsError::Codec("tiff: offset overflow".into()))?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| ViprsError::Codec("tiff: truncated directory entry".into()))?;
    let array = [slice[0], slice[1], slice[2], slice[3]];
    Ok(if tiff_is_big_endian(bytes)? {
        u32::from_be_bytes(array)
    } else {
        u32::from_le_bytes(array)
    })
}

pub(super) fn tiff_write_u32(
    bytes: &mut [u8],
    offset: usize,
    value: u32,
) -> Result<(), ViprsError> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| ViprsError::Codec("tiff: offset overflow".into()))?;
    let is_big_endian = tiff_is_big_endian(bytes)?;
    let target = bytes
        .get_mut(offset..end)
        .ok_or_else(|| ViprsError::Codec("tiff: truncated directory entry".into()))?;
    let encoded = if is_big_endian {
        value.to_be_bytes()
    } else {
        value.to_le_bytes()
    };
    target.copy_from_slice(&encoded);
    Ok(())
}

pub(super) fn first_ifd_offset(bytes: &[u8]) -> Result<u32, ViprsError> {
    tiff_read_u32(bytes, 4)
}

pub(super) fn next_ifd_pointer_pos(bytes: &[u8], ifd_offset: u32) -> Result<usize, ViprsError> {
    let ifd_offset = usize::try_from(ifd_offset).map_err(|e| ViprsError::Codec(e.to_string()))?;
    let entry_count = usize::from(tiff_read_u16(bytes, ifd_offset)?);
    let entries_end = ifd_offset
        .checked_add(2)
        .and_then(|value| value.checked_add(entry_count * 12))
        .ok_or_else(|| ViprsError::Codec("tiff: IFD size overflow".into()))?;
    if entries_end + 4 > bytes.len() {
        return Err(ViprsError::Codec("tiff: truncated IFD".into()));
    }
    Ok(entries_end)
}

pub(super) fn ifd_entry_value_pos(
    bytes: &[u8],
    ifd_offset: u32,
    tag: Tag,
) -> Result<usize, ViprsError> {
    let ifd_offset = usize::try_from(ifd_offset).map_err(|e| ViprsError::Codec(e.to_string()))?;
    let entry_count = usize::from(tiff_read_u16(bytes, ifd_offset)?);
    let target_tag = tag.to_u16();

    for entry_index in 0..entry_count {
        let entry_offset = ifd_offset
            .checked_add(2)
            .and_then(|value| value.checked_add(entry_index * 12))
            .ok_or_else(|| ViprsError::Codec("tiff: IFD entry overflow".into()))?;
        if tiff_read_u16(bytes, entry_offset)? == target_tag {
            return Ok(entry_offset + 8);
        }
    }

    Err(ViprsError::Codec(format!(
        "tiff: missing tag {target_tag} in IFD"
    )))
}

pub(super) fn patch_first_ifd_offset(src: &[u8], ifd_offset: u32) -> Result<Vec<u8>, ViprsError> {
    let mut patched = src.to_vec();
    tiff_write_u32(&mut patched, 4, ifd_offset)?;
    Ok(patched)
}

pub(super) fn patch_subifd_offsets(
    output: &mut [u8],
    table_offset: u32,
    offsets: &[u32],
) -> Result<(), ViprsError> {
    let mut position =
        usize::try_from(table_offset).map_err(|e| ViprsError::Codec(e.to_string()))?;
    for &offset in offsets {
        tiff_write_u32(output, position, offset)?;
        position += 4;
    }
    Ok(())
}

pub(super) fn requested_pyramid_max_dimension(
    width: u32,
    height: u32,
    opts: &LoadOptions,
) -> Option<u32> {
    if let Some(factor) = opts.shrink_factor {
        let factor = u32::from(factor.get());
        if factor > 1 {
            return Some(width.max(height).div_ceil(factor).max(1));
        }
    }

    opts.max_dimension.filter(|&value| value > 0)
}

pub(super) fn choose_pyramid_level(
    root_dimensions: (u32, u32),
    subifd_dimensions: &[(u32, (u32, u32))],
    opts: &LoadOptions,
) -> Option<u32> {
    let target = requested_pyramid_max_dimension(root_dimensions.0, root_dimensions.1, opts)?;
    let mut best_fit = None;
    let mut smallest_level = (root_dimensions.0.max(root_dimensions.1), None);

    for &(offset, (width, height)) in subifd_dimensions {
        let max_dimension = width.max(height);
        if max_dimension < smallest_level.0 {
            smallest_level = (max_dimension, Some(offset));
        }
        if max_dimension <= target {
            match best_fit {
                Some((best_dimension, _)) if best_dimension >= max_dimension => {}
                _ => best_fit = Some((max_dimension, Some(offset))),
            }
        }
    }

    best_fit.and_then(|(_, offset)| offset).or(smallest_level.1)
}

pub(super) fn current_page_subifd_offsets(
    decoder: &mut Decoder<Cursor<&[u8]>>,
) -> Result<Vec<u32>, ViprsError> {
    let Some(value) = decoder
        .find_tag(TIFF_SUB_IFD_TAG)
        .map_err(|e| ViprsError::Codec(e.to_string()))?
    else {
        return Ok(Vec::new());
    };

    let offsets = match value {
        TiffValueRef::Ifd(offset) | TiffValueRef::Unsigned(offset) => vec![offset],
        TiffValueRef::IfdBig(offset) | TiffValueRef::UnsignedBig(offset) => {
            vec![u32::try_from(offset).map_err(|e| ViprsError::Codec(e.to_string()))?]
        }
        TiffValueRef::List(values) => values
            .into_iter()
            .map(|value| {
                value
                    .into_u32()
                    .map_err(|e| ViprsError::Codec(e.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => Vec::new(),
    };

    Ok(offsets.into_iter().filter(|offset| *offset != 0).collect())
}

pub(super) fn probe_ifd_dimensions(src: &[u8], ifd_offset: u32) -> Result<(u32, u32), ViprsError> {
    let patched = patch_first_ifd_offset(src, ifd_offset)?;
    let mut decoder = Decoder::new(Cursor::new(patched.as_slice()))
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    decoder
        .dimensions()
        .map_err(|e| ViprsError::Codec(e.to_string()))
}

pub(super) fn downsample_half<F: BandFormat>(
    image: &InMemoryImage<F>,
) -> Result<Option<InMemoryImage<F>>, ViprsError>
where
    F::Sample: PyramidSample,
{
    if image.width() <= 1 && image.height() <= 1 {
        return Ok(None);
    }

    let new_width = (image.width() / 2).max(1);
    let new_height = (image.height() / 2).max(1);
    if new_width == image.width() && new_height == image.height() {
        return Ok(None);
    }

    let bands = image.bands() as usize;
    let src_row_stride = image.width() as usize * bands;
    let src_width = image.width() as usize;
    let src_height = image.height() as usize;
    let mut data = Vec::with_capacity(new_width as usize * new_height as usize * bands);

    for y in 0..new_height as usize {
        let src_y = y * 2;
        for x in 0..new_width as usize {
            let src_x = x * 2;
            for band in 0..bands {
                let mut samples =
                    [image.pixels()[src_y * src_row_stride + src_x * bands + band]; 4];
                let mut sample_count = 1usize;

                if src_x + 1 < src_width {
                    samples[sample_count] =
                        image.pixels()[src_y * src_row_stride + (src_x + 1) * bands + band];
                    sample_count += 1;
                }
                if src_y + 1 < src_height {
                    samples[sample_count] =
                        image.pixels()[(src_y + 1) * src_row_stride + src_x * bands + band];
                    sample_count += 1;

                    if src_x + 1 < src_width {
                        samples[sample_count] = image.pixels()
                            [(src_y + 1) * src_row_stride + (src_x + 1) * bands + band];
                        sample_count += 1;
                    }
                }

                data.push(F::Sample::average_box(&samples[..sample_count]));
            }
        }
    }

    InMemoryImage::from_buffer(new_width, new_height, image.bands(), data)
        .map(Some)
        .map_err(|e| ViprsError::Codec(e.to_string()))
}

pub(super) fn pyramid_levels<F: BandFormat>(
    image: &InMemoryImage<F>,
    tile: Option<(u32, u32)>,
) -> Result<Vec<InMemoryImage<F>>, ViprsError>
where
    F::Sample: PyramidSample,
{
    let stop_at = tile.map_or(DEFAULT_TIFF_TILE_SIZE, |(tile_width, tile_height)| {
        tile_width.max(tile_height).max(1)
    });
    let mut current = image.clone();
    let mut levels = Vec::new();

    while let Some(next) = downsample_half(&current)? {
        let should_stop = next.width() <= stop_at || next.height() <= stop_at;
        levels.push(next.clone());
        if should_stop || (next.width() == 1 && next.height() == 1) {
            break;
        }
        current = next;
    }

    Ok(levels)
}
pub(super) fn write_subifd_tag<W, K>(
    directory: &mut DirectoryEncoder<'_, W, K>,
    subifd_count: usize,
) -> Result<Option<SubIfdPatchTarget>, ViprsError>
where
    W: Write + std::io::Seek,
    K: TiffKind,
{
    if subifd_count == 0 {
        return Ok(None);
    }

    if subifd_count == 1 {
        directory
            .write_tag(
                TIFF_SUB_IFD_TAG,
                SubIfdTagValue {
                    offset: 0,
                    count: 1,
                },
            )
            .map_err(|e| ViprsError::Codec(e.to_string()))?;
        return Ok(Some(SubIfdPatchTarget::Inline));
    }

    let placeholder_offsets = vec![0u32; subifd_count];
    let table_offset = directory
        .write_data(&placeholder_offsets[..])
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    let table_offset = u32::try_from(table_offset).map_err(|e| ViprsError::Codec(e.to_string()))?;
    directory
        .write_tag(
            TIFF_SUB_IFD_TAG,
            SubIfdTagValue {
                offset: table_offset,
                count: subifd_count,
            },
        )
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    Ok(Some(SubIfdPatchTarget::Table(table_offset)))
}

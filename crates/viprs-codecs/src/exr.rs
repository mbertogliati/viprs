//! Exr adapter module.
//!
//! This module provides concrete codec-related infrastructure used by the
//! adapter layer for loading, saving, or normalizing external image formats.

#![cfg(feature = "exr")]

//! `OpenEXR` codec — decode HDR half/float images and encode HDR float images via the pure-Rust
//! `exr` crate.
//!
//! Decode supports arbitrary flat channel sets and multipart files. Standard
//! RGBA and luminance channel layouts are reordered into the expected Viprs
//! band order; additional layers are exposed through `Image::frames()`.

use std::io::Cursor;

use exr::{
    block::{
        BlockIndex as ExrBlockIndex, UncompressedBlock as ExrUncompressedBlock,
        chunk::TileCoordinates as ExrTileCoordinates,
    },
    error::Error as ExrError,
    image::read::{
        image::LayersReader as ExrLayersReader, layers::ChannelsReader as ExrChannelsReader,
    },
    image::{
        AnyChannel as ExrAnyChannel, AnyChannels as ExrAnyChannels, Blocks as ExrBlocks,
        Encoding as ExrEncoding, FlatSamples as ExrFlatSamples, Image as ExrImage,
        Layer as ExrLayer, SpecificChannels as ExrSpecificChannels,
    },
    math::Vec2 as ExrVec2,
    meta::{
        BlockDescription as ExrBlockDescription, MetaData as ExrMetaData,
        attribute::{IntegerBounds as ExrIntegerBounds, SampleType as ExrSampleType},
        header::{
            Header as ExrHeader, ImageAttributes as ExrImageAttributes,
            LayerAttributes as ExrLayerAttributes,
        },
    },
    prelude::{ReadChannels, ReadLayers, SmallVec as ExrSmallVec, WritableImage},
};

use viprs_core::{
  codec_options::{LoadOptions, SaveOptions},
  error::{ExrCodecError, ViprsError},
  format::{BandFormat, BandFormatId, F32},
  image::{InMemoryImage, ImageMetadata, Interpretation},
};
use viprs_ports::codec::{ImageDecoder, ImageEncoder};

const EXR_MAGIC: [u8; 4] = [0x76, 0x2f, 0x31, 0x01];
const EXR_LAYER_NAME_KEY: &str = "exr.layer_name";
const EXR_CHROMATICITIES_KEY: &str = "exr:chromaticities";
const EXR_DISPLAY_WINDOW_KEY: &str = "exr:display_window";
const EXR_DATA_WINDOW_KEY: &str = "exr:data_window";
const EXR_LAYER_NAMES_KEY: &str = "exr:layer_names";
const EXR_OWNER_KEY: &str = "exr:owner";
const EXR_COMMENTS_KEY: &str = "exr:comments";
const EXR_CHANNEL_NAMES_KEY: &str = "exr:channel_names";
const EXR_UNSUPPORTED_SUBSAMPLING_PREFIX: &str = "viprs:unsupported_exr_subsampling:";
const EXR_FAST_PATH_MIN_DIMENSION: u32 = 4096;
const EXR_INVALID_MULTIPART_BLOCK_REFERENCE_PREFIX: &str =
    "viprs:invalid_multipart_block_reference:";

#[derive(Debug, Clone, Copy, Default)]
/// The `ExrCodec` type provides concrete adapter functionality in the `codecs` module.
/// Use this type when you need the runtime behavior implemented by this adapter.
///
/// # Examples
///
/// ```rust
/// let _ = core::mem::size_of::<viprs_codecs::exr::ExrCodec>();
/// ```
pub struct ExrCodec;

fn invalid_multipart_block_reference_message(
    requested_layer: usize,
    selected_start: usize,
    selected_count: usize,
    header_count: usize,
) -> String {
    format!(
        "{EXR_INVALID_MULTIPART_BLOCK_REFERENCE_PREFIX}{requested_layer}:{selected_start}:{selected_count}:{header_count}"
    )
}

fn parse_invalid_multipart_block_reference(message: &str) -> Option<ExrCodecError> {
    let values = message
        .strip_prefix(EXR_INVALID_MULTIPART_BLOCK_REFERENCE_PREFIX)?
        .split(':')
        .map(str::parse::<usize>)
        .collect::<Result<Vec<_>, _>>()
        .ok()?;

    let [
        requested_layer,
        selected_start,
        selected_count,
        header_count,
    ] = values.as_slice()
    else {
        return None;
    };

    Some(ExrCodecError::InvalidMultipartBlockReference {
        requested_layer: *requested_layer,
        selected_start: *selected_start,
        selected_count: *selected_count,
        header_count: *header_count,
    })
}

fn unsupported_subsampling_message(channel: &str, x: usize, y: usize) -> String {
    format!("{EXR_UNSUPPORTED_SUBSAMPLING_PREFIX}{channel}:{x}:{y}")
}

fn parse_unsupported_subsampling(message: &str) -> Option<ExrCodecError> {
    let (channel, rest) = message
        .strip_prefix(EXR_UNSUPPORTED_SUBSAMPLING_PREFIX)?
        .split_once(':')?;
    let (x, y) = rest.split_once(':')?;

    Some(ExrCodecError::UnsupportedSubsampling {
        channel: channel.to_string(),
        x: x.parse().ok()?,
        y: y.parse().ok()?,
    })
}

fn unsupported_subsampling_error(channel: &str, x: usize, y: usize) -> ExrError {
    ExrError::Invalid(unsupported_subsampling_message(channel, x, y).into())
}

fn invalid_multipart_block_reference_error(
    requested_layer: usize,
    selected_start: usize,
    selected_count: usize,
    header_count: usize,
) -> ExrError {
    ExrError::Invalid(
        invalid_multipart_block_reference_message(
            requested_layer,
            selected_start,
            selected_count,
            header_count,
        )
        .into(),
    )
}

fn exr_error(error: ExrError) -> ViprsError {
    match error {
        ExrError::Invalid(message) => parse_invalid_multipart_block_reference(message.as_ref())
            .or_else(|| parse_unsupported_subsampling(message.as_ref()))
            .unwrap_or_else(|| ExrCodecError::Backend(ExrError::Invalid(message).to_string()))
            .into(),
        other => ExrCodecError::Backend(other.to_string()).into(),
    }
}

fn selection_from_options(
    total_layers: usize,
    opts: &LoadOptions,
) -> Result<(usize, usize), ViprsError> {
    if total_layers == 0 {
        return Err(ExrCodecError::NoLayers.into());
    }

    if opts.page.is_none() && opts.n.is_none() {
        return Ok((0, total_layers));
    }

    let page = opts.page.unwrap_or(0) as usize;
    if page >= total_layers {
        return Err(ExrCodecError::RequestedLayerOutOfRange {
            requested: page,
            total_layers,
        }
        .into());
    }

    let remaining = total_layers - page;
    let requested = match opts.n {
        None => 1,
        Some(-1) => remaining,
        Some(value) if value > 0 => usize::try_from(value)
            .map_err(|_| ViprsError::from(ExrCodecError::InvalidLayerCount { value }))?,
        Some(value) => {
            return Err(ExrCodecError::InvalidLayerCount { value }.into());
        }
    };

    Ok((page, requested.min(remaining)))
}

#[cfg(test)]
use std::{cell::RefCell, thread_local};

#[cfg(test)]
thread_local! {
    static MATERIALIZED_LAYERS: RefCell<Option<Vec<usize>>> = const { RefCell::new(None) };
}

#[cfg(test)]
fn record_materialized_layers(indices: impl IntoIterator<Item = usize>) {
    MATERIALIZED_LAYERS.with(|layers| {
        if let Some(recorded_layers) = layers.borrow_mut().as_mut() {
            recorded_layers.extend(indices);
        }
    });
}

#[cfg(test)]
fn capture_materialized_layers<T>(action: impl FnOnce() -> T) -> (T, Vec<usize>) {
    MATERIALIZED_LAYERS.with(|layers| *layers.borrow_mut() = Some(Vec::new()));
    let value = action();
    let materialized_layers =
        MATERIALIZED_LAYERS.with(|layers| layers.borrow_mut().take().unwrap_or_default());

    (value, materialized_layers)
}

fn approx_pixels_per_mm(horizontal_density_ppi: f32, pixel_aspect: f32) -> (f64, f64) {
    let xres = f64::from(horizontal_density_ppi) / 25.4;
    let safe_aspect = if pixel_aspect.is_finite() && pixel_aspect > 0.0 {
        f64::from(pixel_aspect)
    } else {
        1.0
    };
    (xres, xres / safe_aspect)
}

fn interpretation_for_channel_names(names: &[String]) -> Interpretation {
    let has = |name: &str| names.iter().any(|candidate| candidate == name);
    let luminance_name = if has("Y") {
        Some("Y")
    } else if has("L") {
        Some("L")
    } else {
        None
    };

    if has("R") && has("G") && has("B") {
        Interpretation::Scrgb
    } else if luminance_name.is_some() && names.len() <= 2 {
        Interpretation::BW
    } else {
        Interpretation::Multiband
    }
}

fn preferred_channel_order(names: &[String]) -> Vec<usize> {
    let mut order = Vec::with_capacity(names.len());

    let push_named = |name: &str, names: &[String], order: &mut Vec<usize>| {
        if let Some(index) = names.iter().position(|candidate| candidate == name) {
            order.push(index);
        }
    };

    if names.iter().any(|name| name == "R")
        && names.iter().any(|name| name == "G")
        && names.iter().any(|name| name == "B")
    {
        push_named("R", names, &mut order);
        push_named("G", names, &mut order);
        push_named("B", names, &mut order);
        push_named("A", names, &mut order);
    } else if names.iter().any(|name| name == "Y") || names.iter().any(|name| name == "L") {
        if names.iter().any(|name| name == "Y") {
            push_named("Y", names, &mut order);
        } else {
            push_named("L", names, &mut order);
        }
        push_named("A", names, &mut order);
    }

    for index in 0..names.len() {
        if !order.contains(&index) {
            order.push(index);
        }
    }

    order
}

#[derive(Debug, Clone)]
struct InterleavedChannels {
    pixels: Vec<f32>,
    channel_names: Vec<String>,
}

#[derive(Debug, Clone, Copy, Default)]
struct ReadInterleavedChannels;

#[derive(Debug, Clone)]
struct InterleavedChannelsReader {
    width: usize,
    height: usize,
    bands: usize,
    channel_band_map: Vec<Option<usize>>,
    channel_names: Vec<String>,
    pixels: Vec<f32>,
}

impl InterleavedChannelsReader {
    fn row_offset(&self, x: usize, y: usize) -> usize {
        debug_assert!(x <= self.width, "x offset outside layer width");
        debug_assert!(y < self.height, "y offset outside layer height");
        (y * self.width + x) * self.bands
    }

    fn write_line_f16(
        &mut self,
        line: exr::block::lines::LineRef<'_>,
        band_index: usize,
    ) -> std::result::Result<(), ExrError> {
        let mut dst_index =
            self.row_offset(line.location.position.x(), line.location.position.y()) + band_index;

        for sample in line.read_samples::<exr::prelude::f16>() {
            self.pixels[dst_index] = sample?.to_f32();
            dst_index += self.bands;
        }

        Ok(())
    }

    fn write_line_f32(
        &mut self,
        line: exr::block::lines::LineRef<'_>,
        band_index: usize,
    ) -> std::result::Result<(), ExrError> {
        let mut dst_index =
            self.row_offset(line.location.position.x(), line.location.position.y()) + band_index;

        for sample in line.read_samples::<f32>() {
            self.pixels[dst_index] = sample?;
            dst_index += self.bands;
        }

        Ok(())
    }

    fn write_line_u32(
        &mut self,
        line: exr::block::lines::LineRef<'_>,
        band_index: usize,
    ) -> std::result::Result<(), ExrError> {
        let mut dst_index =
            self.row_offset(line.location.position.x(), line.location.position.y()) + band_index;

        for sample in line.read_samples::<u32>() {
            self.pixels[dst_index] = sample? as f32;
            dst_index += self.bands;
        }

        Ok(())
    }
}

impl ReadChannels<'_> for ReadInterleavedChannels {
    type Reader = InterleavedChannelsReader;

    fn create_channels_reader(
        &self,
        header: &ExrHeader,
    ) -> std::result::Result<Self::Reader, ExrError> {
        let channel_names: Vec<String> = header
            .channels
            .list
            .iter()
            .map(|channel| channel.name.to_string())
            .collect();
        let order = preferred_channel_order(&channel_names);
        let mut channel_band_map = vec![None; header.channels.list.len()];
        let mut ordered_channel_names = Vec::with_capacity(order.len());

        for (band_index, channel_index) in order.iter().copied().enumerate() {
            let channel = &header.channels.list[channel_index];
            if channel.sampling.width() != 1 || channel.sampling.height() != 1 {
                return Err(unsupported_subsampling_error(
                    &channel.name.to_string(),
                    channel.sampling.width(),
                    channel.sampling.height(),
                ));
            }

            channel_band_map[channel_index] = Some(band_index);
            ordered_channel_names.push(channel.name.to_string());
        }

        let width = header.layer_size.width();
        let height = header.layer_size.height();
        let bands = order.len();
        let pixel_count = width
            .checked_mul(height)
            .and_then(|count| count.checked_mul(bands))
            .ok_or_else(|| ExrError::Invalid("viprs:exr_interleaved_buffer_overflow".into()))?;

        Ok(InterleavedChannelsReader {
            width,
            height,
            bands,
            channel_band_map,
            channel_names: ordered_channel_names,
            pixels: vec![0.0; pixel_count],
        })
    }
}

impl ExrChannelsReader for InterleavedChannelsReader {
    type Channels = InterleavedChannels;

    fn filter_block(&self, tile: ExrTileCoordinates) -> bool {
        tile.is_largest_resolution_level()
    }

    fn read_block(
        &mut self,
        header: &ExrHeader,
        block: ExrUncompressedBlock,
    ) -> std::result::Result<(), ExrError> {
        for line in block.lines(&header.channels) {
            let channel_index = line.location.channel;
            let Some(Some(band_index)) = self.channel_band_map.get(channel_index).copied() else {
                continue;
            };

            debug_assert_eq!(
                line.location.level,
                ExrVec2(0, 0),
                "filtered blocks must stay on the largest level"
            );
            debug_assert!(
                line.location.position.x() + line.location.sample_count <= self.width,
                "line must stay within layer width"
            );

            match header.channels.list[channel_index].sample_type {
                ExrSampleType::F16 => self.write_line_f16(line, band_index)?,
                ExrSampleType::F32 => self.write_line_f32(line, band_index)?,
                ExrSampleType::U32 => self.write_line_u32(line, band_index)?,
            }
        }

        Ok(())
    }

    fn into_channels(self) -> Self::Channels {
        InterleavedChannels {
            pixels: self.pixels,
            channel_names: self.channel_names,
        }
    }
}

fn format_exr_bounds(bounds: ExrIntegerBounds) -> String {
    let max = bounds.max();
    format!(
        "{},{},{},{}",
        bounds.position.x(),
        bounds.position.y(),
        max.x(),
        max.y()
    )
}

fn format_exr_chromaticities(chromaticities: exr::meta::attribute::Chromaticities) -> String {
    format!(
        "red=({},{});green=({},{});blue=({},{});white=({},{})",
        chromaticities.red.x(),
        chromaticities.red.y(),
        chromaticities.green.x(),
        chromaticities.green.y(),
        chromaticities.blue.x(),
        chromaticities.blue.y(),
        chromaticities.white.x(),
        chromaticities.white.y()
    )
}

fn layer_data_window<Channels>(layer: &ExrLayer<Channels>) -> ExrIntegerBounds {
    ExrIntegerBounds::new(layer.attributes.layer_position, layer.size)
}

fn insert_exr_text(
    metadata: &mut ImageMetadata,
    key: &str,
    value: Option<&exr::meta::attribute::Text>,
) {
    if let Some(value) = value
        .map(ToString::to_string)
        .filter(|value| !value.is_empty())
    {
        metadata.extra.insert(key.into(), value);
    }
}

fn decode_layer(
    layer: ExrLayer<InterleavedChannels>,
    image_attributes: &ExrImageAttributes,
    layer_names: &[String],
    pixel_aspect: f32,
) -> Result<InMemoryImage<F32>, ViprsError> {
    let data_window = layer_data_window(&layer);
    let width = u32::try_from(layer.size.width())
        .map_err(|_| ViprsError::from(ExrCodecError::LayerWidthExceedsU32))?;
    let height = u32::try_from(layer.size.height())
        .map_err(|_| ViprsError::from(ExrCodecError::LayerHeightExceedsU32))?;

    let InterleavedChannels {
        pixels,
        channel_names,
    } = layer.channel_data;
    let bands = u32::try_from(channel_names.len())
        .map_err(|_| ViprsError::from(ExrCodecError::BandCountExceedsU32))?;

    let mut metadata = ImageMetadata {
        interpretation: Some(interpretation_for_channel_names(&channel_names)),
        ..ImageMetadata::default()
    };
    metadata
        .extra
        .insert(EXR_CHANNEL_NAMES_KEY.into(), channel_names.join(","));
    metadata.extra.insert(
        EXR_DISPLAY_WINDOW_KEY.into(),
        format_exr_bounds(image_attributes.display_window),
    );
    metadata
        .extra
        .insert(EXR_DATA_WINDOW_KEY.into(), format_exr_bounds(data_window));
    if !layer_names.is_empty() {
        metadata
            .extra
            .insert(EXR_LAYER_NAMES_KEY.into(), layer_names.join(","));
    }
    if let Some(chromaticities) = image_attributes.chromaticities {
        metadata.extra.insert(
            EXR_CHROMATICITIES_KEY.into(),
            format_exr_chromaticities(chromaticities),
        );
    }
    if let Some(horizontal_density) = layer.attributes.horizontal_density {
        let (xres, yres) = approx_pixels_per_mm(horizontal_density, pixel_aspect);
        metadata.xres = Some(xres);
        metadata.yres = Some(yres);
    }
    if let Some(layer_name) = layer
        .attributes
        .layer_name
        .as_ref()
        .map(ToString::to_string)
        .filter(|name| !name.is_empty())
    {
        metadata.extra.insert(EXR_LAYER_NAME_KEY.into(), layer_name);
    }
    insert_exr_text(
        &mut metadata,
        EXR_OWNER_KEY,
        layer.attributes.owner.as_ref(),
    );
    insert_exr_text(
        &mut metadata,
        EXR_COMMENTS_KEY,
        layer.attributes.comments.as_ref(),
    );

    InMemoryImage::from_buffer(width, height, bands, pixels)
        .map(|image| image.with_metadata(metadata))
        .map_err(|error| ExrCodecError::Backend(error.to_string()).into())
}

fn load_exr_metadata(src: &[u8]) -> Result<ExrMetaData, ViprsError> {
    ExrMetaData::read_from_buffered(Cursor::new(src), false).map_err(exr_error)
}

const fn exr_encoding_from_header(header: &ExrHeader) -> ExrEncoding {
    ExrEncoding {
        compression: header.compression,
        line_order: header.line_order,
        blocks: match header.blocks {
            ExrBlockDescription::ScanLines => ExrBlocks::ScanLines,
            ExrBlockDescription::Tiles(tile_description) => {
                ExrBlocks::Tiles(tile_description.tile_size)
            }
        },
    }
}

struct ReadSelectedLayers<ReadChannels> {
    read_channels: ReadChannels,
    start: usize,
    count: usize,
}

struct SelectedLayersReader<ChannelsReader> {
    start: usize,
    layer_readers: ExrSmallVec<[SelectedLayerReader<ChannelsReader>; 2]>,
}

struct SelectedLayerReader<ChannelsReader> {
    channels_reader: ChannelsReader,
    attributes: ExrLayerAttributes,
    size: ExrVec2<usize>,
    encoding: ExrEncoding,
}

impl<ChannelsReader> SelectedLayersReader<ChannelsReader> {
    fn selected_index(&self, source_index: usize) -> Option<usize> {
        source_index
            .checked_sub(self.start)
            .filter(|&index| index < self.layer_readers.len())
    }

    fn reader_for_source_index(
        &self,
        source_index: usize,
    ) -> Option<&SelectedLayerReader<ChannelsReader>> {
        self.selected_index(source_index)
            .and_then(|index| self.layer_readers.get(index))
    }

    fn reader_for_source_index_mut(
        &mut self,
        source_index: usize,
    ) -> Option<&mut SelectedLayerReader<ChannelsReader>> {
        let index = self.selected_index(source_index)?;
        self.layer_readers.get_mut(index)
    }
}

impl<'s, C> ReadLayers<'s> for ReadSelectedLayers<C>
where
    C: ReadChannels<'s>,
{
    type Layers = exr::image::Layers<<C::Reader as ExrChannelsReader>::Channels>;
    type Reader = SelectedLayersReader<C::Reader>;

    fn create_layers_reader(
        &'s self,
        headers: &[ExrHeader],
    ) -> std::result::Result<Self::Reader, ExrError> {
        let mut layer_readers = ExrSmallVec::new();

        for header in headers.iter().skip(self.start).take(self.count) {
            #[cfg(test)]
            record_materialized_layers([self.start + layer_readers.len()]);

            layer_readers.push(SelectedLayerReader {
                channels_reader: self.read_channels.create_channels_reader(header)?,
                attributes: header.own_attributes.clone(),
                size: header.layer_size,
                encoding: exr_encoding_from_header(header),
            });
        }

        Ok(SelectedLayersReader {
            start: self.start,
            layer_readers,
        })
    }
}

impl<C> ExrLayersReader for SelectedLayersReader<C>
where
    C: ExrChannelsReader,
{
    type Layers = exr::image::Layers<C::Channels>;

    fn filter_block(
        &self,
        _: &ExrMetaData,
        tile: ExrTileCoordinates,
        block: ExrBlockIndex,
    ) -> bool {
        self.reader_for_source_index(block.layer)
            .is_some_and(|layer| layer.channels_reader.filter_block(tile))
    }

    fn read_block(
        &mut self,
        headers: &[ExrHeader],
        block: ExrUncompressedBlock,
    ) -> std::result::Result<(), ExrError> {
        let header_index = block.index.layer;
        let selected_start = self.start;
        let selected_count = self.layer_readers.len();
        let header_count = headers.len();
        let layer = self
            .reader_for_source_index_mut(header_index)
            .ok_or_else(|| {
                invalid_multipart_block_reference_error(
                    header_index,
                    selected_start,
                    selected_count,
                    header_count,
                )
            })?;
        let header = headers.get(header_index).ok_or_else(|| {
            invalid_multipart_block_reference_error(
                header_index,
                selected_start,
                selected_count,
                header_count,
            )
        })?;
        layer.channels_reader.read_block(header, block)
    }

    fn into_layers(self) -> Self::Layers {
        self.layer_readers
            .into_iter()
            .map(|layer| ExrLayer {
                channel_data: layer.channels_reader.into_channels(),
                attributes: layer.attributes,
                size: layer.size,
                encoding: layer.encoding,
            })
            .collect()
    }
}

fn load_selected_exr_layers(
    src: &[u8],
    start: usize,
    count: usize,
) -> Result<ExrImage<exr::image::Layers<InterleavedChannels>>, ViprsError> {
    ReadSelectedLayers {
        read_channels: ReadInterleavedChannels,
        start,
        count,
    }
    .all_attributes()
    .from_buffered(Cursor::new(src))
    .map_err(exr_error)
}

#[cfg(test)]
fn load_exr_layers(
    src: &[u8],
) -> Result<ExrImage<exr::image::Layers<InterleavedChannels>>, ViprsError> {
    let total_layers = load_exr_metadata(src)?.headers.len();
    load_selected_exr_layers(src, 0, total_layers)
}

fn recast_image_from_f32<F: BandFormat>(image: InMemoryImage<F32>) -> Result<InMemoryImage<F>, ViprsError> {
    let width = image.width();
    let height = image.height();
    let bands = image.bands();
    let metadata = image.metadata().clone();
    let frames = image.frames().map(ToOwned::to_owned);
    let samples = bytemuck::allocation::try_cast_vec::<f32, F::Sample>(image.into_buffer())
        .map_err(|(error, _)| {
            ViprsError::from(ExrCodecError::CastError {
                details: format!("{error:?}"),
            })
        })?;

    let mut recast = InMemoryImage::from_buffer(width, height, bands, samples)
        .map_err(|error| ViprsError::from(ExrCodecError::Backend(error.to_string())))?
        .with_metadata(metadata);

    if let Some(frames) = frames {
        recast = recast.with_frames(
            frames
                .into_iter()
                .map(recast_image_from_f32::<F>)
                .collect::<Result<Vec<_>, _>>()?,
        );
    }

    Ok(recast)
}

fn decode_exr<F: BandFormat>(src: &[u8], opts: &LoadOptions) -> Result<InMemoryImage<F>, ViprsError> {
    if F::ID != BandFormatId::F32 {
        return Err(ExrCodecError::UnsupportedFormat { format: F::ID }.into());
    }

    let metadata = load_exr_metadata(src)?;
    let total_layers = metadata.headers.len();
    let (start, count) = selection_from_options(total_layers, opts)?;
    let all_layer_names = metadata
        .headers
        .iter()
        .filter_map(|header| {
            header
                .own_attributes
                .layer_name
                .as_ref()
                .map(ToString::to_string)
                .filter(|name| !name.is_empty())
        })
        .collect::<Vec<_>>();
    let exr_image = load_selected_exr_layers(src, start, count)?;

    let mut decoded_layers = exr_image.layer_data.into_iter().map(|layer| {
        decode_layer(
            layer,
            &exr_image.attributes,
            &all_layer_names,
            exr_image.attributes.pixel_aspect,
        )
    });

    let Some(first_layer) = decoded_layers.next().transpose()? else {
        return Err(ExrCodecError::NoDecodableLayersSelected.into());
    };
    let remaining_frames = decoded_layers.collect::<Result<Vec<_>, _>>()?;

    let mut metadata = first_layer.metadata().clone();
    metadata.n_pages = Some(total_layers as u32);
    if total_layers > 1 || !remaining_frames.is_empty() {
        metadata.page_height = Some(first_layer.height());
    }

    let image = first_layer.with_metadata(metadata);
    let image = if remaining_frames.is_empty() {
        image
    } else {
        let mut frames = Vec::with_capacity(remaining_frames.len() + 1);
        frames.push(image.clone());
        frames.extend(remaining_frames);
        image.with_frames(frames)
    };

    recast_image_from_f32(image)
}

fn default_layer_name(index: usize, total_layers: usize) -> Option<String> {
    if total_layers == 1 {
        None
    } else {
        Some(format!("layer{index}"))
    }
}

fn channel_names(bands: u32) -> Vec<String> {
    match bands {
        1 => vec!["Y".into()],
        2 => vec!["Y".into(), "A".into()],
        3 => vec!["R".into(), "G".into(), "B".into()],
        4 => vec!["R".into(), "G".into(), "B".into(), "A".into()],
        count => (0..count).map(|index| format!("C{index}")).collect(),
    }
}

fn pixels_per_mm_to_ppi(value: f64) -> Option<f32> {
    let ppi = value * 25.4;
    (ppi.is_finite() && ppi >= 0.0 && ppi <= f64::from(f32::MAX)).then_some(ppi as f32)
}

fn exr_layer_from_image<F: BandFormat>(
  image: &InMemoryImage<F>,
  index: usize,
  total_layers: usize,
  strip_metadata: bool,
) -> ExrLayer<ExrAnyChannels<ExrFlatSamples>>
where
    F::Sample: Clone,
{
    let band_names = channel_names(image.bands());
    let pixel_count = image.width() as usize * image.height() as usize;
    let band_count = image.bands() as usize;
    let pixels = bytemuck::cast_slice::<F::Sample, f32>(image.pixels());

    let mut channels: Vec<ExrAnyChannel<ExrFlatSamples>> = Vec::with_capacity(band_count);
    for band_index in 0..band_count {
        let mut band_pixels = Vec::with_capacity(pixel_count);
        for pixel in pixels.chunks_exact(band_count) {
            band_pixels.push(pixel[band_index]);
        }
        channels.push(ExrAnyChannel::new(
            band_names[band_index].as_str(),
            ExrFlatSamples::F32(band_pixels),
        ));
    }

    let layer_attributes = exr_layer_attributes(image, index, total_layers, strip_metadata);

    ExrLayer::new(
        (image.width() as usize, image.height() as usize),
        layer_attributes,
        ExrEncoding::default(),
        ExrAnyChannels::sort(channels.into_iter().collect::<ExrSmallVec<[_; 4]>>()),
    )
}

fn layers_for_encode<F: BandFormat>(image: &InMemoryImage<F>) -> Vec<&InMemoryImage<F>> {
    image
        .frames()
        .map_or_else(|| vec![image], |frames| frames.iter().collect())
}

fn exr_layer_attributes<F: BandFormat>(
  image: &InMemoryImage<F>,
  index: usize,
  total_layers: usize,
  strip_metadata: bool,
) -> ExrLayerAttributes {
    let layer_name = image
        .metadata()
        .extra
        .get(EXR_LAYER_NAME_KEY)
        .map(String::as_str)
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| default_layer_name(index, total_layers));
    let mut layer_attributes = layer_name.map_or_else(ExrLayerAttributes::default, |name| {
        ExrLayerAttributes::named(name.as_str())
    });

    if !strip_metadata && let Some(xres) = image.metadata().xres.and_then(pixels_per_mm_to_ppi) {
        layer_attributes.horizontal_density = Some(xres);
    }

    layer_attributes
}

fn set_exr_pixel_aspect(
    attributes: &mut ExrImageAttributes,
    metadata: &ImageMetadata,
    strip_metadata: bool,
) {
    if !strip_metadata
        && let (Some(xres), Some(yres)) = (metadata.xres, metadata.yres)
        && xres.is_finite()
        && yres.is_finite()
        && xres > 0.0
        && yres > 0.0
    {
        attributes.pixel_aspect = (xres / yres) as f32;
    }
}

fn encode_exr_single_layer_fast(
    width: u32,
    height: u32,
    bands: u32,
    pixels: &[f32],
    metadata: &ImageMetadata,
    opts: &SaveOptions,
) -> Result<Vec<u8>, ViprsError> {
    let strip_metadata = opts.strip_metadata.unwrap_or(false);
    let width = width as usize;
    let height = height as usize;
    let bands = bands as usize;
    let pixel_base = |position: ExrVec2<usize>| (position.y() * width + position.x()) * bands;
    let layer_attributes = {
        let layer_name = metadata
            .extra
            .get(EXR_LAYER_NAME_KEY)
            .map(String::as_str)
            .filter(|name| !name.is_empty())
            .map(ToOwned::to_owned);
        let mut layer_attributes = layer_name.map_or_else(ExrLayerAttributes::default, |name| {
            ExrLayerAttributes::named(name.as_str())
        });

        if !strip_metadata && let Some(xres) = metadata.xres.and_then(pixels_per_mm_to_ppi) {
            layer_attributes.horizontal_density = Some(xres);
        }

        layer_attributes
    };
    let mut output = Vec::new();

    match bands {
        1 => {
            let channels = ExrSpecificChannels::build()
                .with_channel::<f32>("Y")
                .with_pixels(|position: ExrVec2<usize>| {
                    let base = pixel_base(position);
                    (pixels[base],)
                });
            let layer = ExrLayer::new(
                (width, height),
                layer_attributes,
                ExrEncoding::default(),
                channels,
            );
            let mut exr_image = ExrImage::from_layer(layer);
            set_exr_pixel_aspect(&mut exr_image.attributes, metadata, strip_metadata);
            exr_image
                .write()
                .to_buffered(Cursor::new(&mut output))
                .map_err(exr_error)?;
        }
        2 => {
            let channels = ExrSpecificChannels::build()
                .with_channel::<f32>("Y")
                .with_channel::<f32>("A")
                .with_pixels(|position: ExrVec2<usize>| {
                    let base = pixel_base(position);
                    (pixels[base], pixels[base + 1])
                });
            let layer = ExrLayer::new(
                (width, height),
                layer_attributes,
                ExrEncoding::default(),
                channels,
            );
            let mut exr_image = ExrImage::from_layer(layer);
            set_exr_pixel_aspect(&mut exr_image.attributes, metadata, strip_metadata);
            exr_image
                .write()
                .to_buffered(Cursor::new(&mut output))
                .map_err(exr_error)?;
        }
        3 => {
            let channels = ExrSpecificChannels::rgb(|position: ExrVec2<usize>| {
                let base = pixel_base(position);
                (pixels[base], pixels[base + 1], pixels[base + 2])
            });
            let layer = ExrLayer::new(
                (width, height),
                layer_attributes,
                ExrEncoding::default(),
                channels,
            );
            let mut exr_image = ExrImage::from_layer(layer);
            set_exr_pixel_aspect(&mut exr_image.attributes, metadata, strip_metadata);
            exr_image
                .write()
                .to_buffered(Cursor::new(&mut output))
                .map_err(exr_error)?;
        }
        4 => {
            let channels = ExrSpecificChannels::rgba(|position: ExrVec2<usize>| {
                let base = pixel_base(position);
                (
                    pixels[base],
                    pixels[base + 1],
                    pixels[base + 2],
                    pixels[base + 3],
                )
            });
            let layer = ExrLayer::new(
                (width, height),
                layer_attributes,
                ExrEncoding::default(),
                channels,
            );
            let mut exr_image = ExrImage::from_layer(layer);
            set_exr_pixel_aspect(&mut exr_image.attributes, metadata, strip_metadata);
            exr_image
                .write()
                .to_buffered(Cursor::new(&mut output))
                .map_err(exr_error)?;
        }
        _ => {
            return Err(
                ExrCodecError::Backend(format!("exr: unsupported band count {bands}")).into(),
            );
        }
    }

    Ok(output)
}

fn encode_exr<F: BandFormat>(image: &InMemoryImage<F>, opts: &SaveOptions) -> Result<Vec<u8>, ViprsError>
where
    F::Sample: Clone,
{
    if F::ID != BandFormatId::F32 {
        return Err(ExrCodecError::UnsupportedFormat { format: F::ID }.into());
    }

    if image.frames().is_none()
        && image.bands() <= 4
        && image.width().max(image.height()) >= EXR_FAST_PATH_MIN_DIMENSION
    {
        return encode_exr_single_layer_fast(
            image.width(),
            image.height(),
            image.bands(),
            bytemuck::cast_slice::<F::Sample, f32>(image.pixels()),
            image.metadata(),
            opts,
        );
    }

    let layers = layers_for_encode(image);
    let total_layers = layers.len();
    let strip_metadata = opts.strip_metadata.unwrap_or(false);

    let max_width = layers.iter().map(|frame| frame.width()).max().unwrap_or(0) as usize;
    let max_height = layers.iter().map(|frame| frame.height()).max().unwrap_or(0) as usize;

    let mut image_attributes =
        ExrImageAttributes::new(ExrIntegerBounds::from_dimensions((max_width, max_height)));
    set_exr_pixel_aspect(&mut image_attributes, image.metadata(), strip_metadata);

    let exr_layers = layers
        .iter()
        .enumerate()
        .map(|(index, frame)| exr_layer_from_image(frame, index, total_layers, strip_metadata))
        .collect::<ExrSmallVec<[_; 2]>>();

    let mut output = Vec::new();
    let exr_image = ExrImage::from_layers(image_attributes, exr_layers);
    exr_image
        .write()
        .to_buffered(Cursor::new(&mut output))
        .map_err(exr_error)?;

    Ok(output)
}

impl ImageDecoder for ExrCodec {
    fn format_name(&self) -> &'static str {
        "exr"
    }

    fn sniff(&self, header: &[u8]) -> bool
    where
        Self: Sized,
    {
        header.len() >= EXR_MAGIC.len() && header[..EXR_MAGIC.len()] == EXR_MAGIC
    }

    fn decode<F: BandFormat>(&self, src: &[u8]) -> Result<InMemoryImage<F>, ViprsError>
    where
        F::Sample: Clone,
    {
        self.decode_with_options(src, &LoadOptions::default())
    }

    fn decode_with_options<F: BandFormat>(
        &self,
        src: &[u8],
        opts: &LoadOptions,
    ) -> Result<InMemoryImage<F>, ViprsError>
    where
        Self: Sized,
        F::Sample: Clone,
    {
        decode_exr(src, opts)
    }

    fn probe(&self, src: &[u8]) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized,
    {
        let metadata = load_exr_metadata(src)?;
        let first = metadata
            .headers
            .first()
            .ok_or_else(|| ViprsError::from(ExrCodecError::NoLayers))?;
        Ok((
            u32::try_from(first.layer_size.width())
                .map_err(|_| ViprsError::from(ExrCodecError::LayerWidthExceedsU32))?,
            u32::try_from(first.layer_size.height())
                .map_err(|_| ViprsError::from(ExrCodecError::LayerHeightExceedsU32))?,
            u32::try_from(first.channels.list.len())
                .map_err(|_| ViprsError::from(ExrCodecError::BandCountExceedsU32))?,
        ))
    }
}

impl ImageEncoder for ExrCodec {
    fn format_name(&self) -> &'static str {
        "exr"
    }

    fn encode<F: BandFormat>(&self, image: &InMemoryImage<F>) -> Result<Vec<u8>, ViprsError>
    where
        F::Sample: Clone,
    {
        self.encode_with_options(image, &SaveOptions::default())
    }

    fn encode_with_options<F: BandFormat>(
      &self,
      image: &InMemoryImage<F>,
      opts: &SaveOptions,
    ) -> Result<Vec<u8>, ViprsError>
    where
        Self: Sized,
        F::Sample: Clone,
    {
        encode_exr(image, opts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use exr::{math::Vec2, meta::attribute::Chromaticities, prelude::f16};
    use viprs_core::format::U8;

    const METADATA_FIXTURE: &[u8] =
        include_bytes!("../../../tests/fixtures/images/exr_metadata_layers.exr");
    const RGB_HALF_FIXTURE: &[u8] =
        include_bytes!("../../../tests/fixtures/images/exr_rgb_half.exr");
    const RGBA_HALF_FIXTURE: &[u8] =
        include_bytes!("../../../tests/fixtures/images/exr_rgba_half.exr");
    const RGB_FLOAT_FIXTURE: &[u8] =
        include_bytes!("../../../tests/fixtures/images/exr_rgb_float.exr");
    const RGBA_FLOAT_FIXTURE: &[u8] =
        include_bytes!("../../../tests/fixtures/images/exr_rgba_float.exr");

    fn rgba_image() -> InMemoryImage<F32> {
        InMemoryImage::<F32>::from_buffer(
            2,
            2,
            4,
            vec![
                0.0, 0.5, 1.0, 1.0, //
                1.5, 2.0, 2.5, 0.75, //
                3.0, 3.5, 4.0, 0.5, //
                4.5, 5.0, 5.5, 0.25,
            ],
        )
        .unwrap()
    }

    fn corrupt_second_layer_offset_table_entry(encoded: &[u8]) -> Vec<u8> {
        let metadata = load_exr_metadata(encoded).unwrap();
        assert_eq!(
            metadata.headers.len(),
            2,
            "test fixture expects two headers"
        );
        assert_eq!(
            metadata.headers[0].chunk_count, 1,
            "test fixture expects one chunk in first header"
        );
        assert_eq!(
            metadata.headers[1].chunk_count, 1,
            "test fixture expects one chunk in second header"
        );

        let mut metadata_prefix = Vec::new();
        metadata_prefix.extend_from_slice(&EXR_MAGIC);
        metadata.requirements.write(&mut metadata_prefix).unwrap();
        ExrHeader::write_all(
            metadata.headers.as_slice(),
            &mut metadata_prefix,
            metadata.requirements.is_multilayer(),
        )
        .unwrap();

        let offset_table_start = metadata_prefix.len();
        let first_entry_range = offset_table_start..offset_table_start + std::mem::size_of::<u64>();
        let second_entry_range =
            first_entry_range.end..first_entry_range.end + std::mem::size_of::<u64>();

        let mut corrupted = encoded.to_vec();
        let first_offset = corrupted[first_entry_range].to_vec();
        corrupted[second_entry_range].copy_from_slice(&first_offset);
        corrupted
    }

    fn rgb_image() -> InMemoryImage<F32> {
        InMemoryImage::<F32>::from_buffer(
            2,
            2,
            3,
            vec![
                0.1, 0.2, 0.3, //
                0.4, 0.5, 0.6, //
                0.7, 0.8, 0.9, //
                1.0, 1.1, 1.2,
            ],
        )
        .unwrap()
    }

    fn luminance_alpha_image() -> InMemoryImage<F32> {
        InMemoryImage::<F32>::from_buffer(2, 2, 2, vec![0.0, 1.0, 0.25, 0.75, 0.5, 0.5, 1.0, 0.0]).unwrap()
    }

    fn luminance_image() -> InMemoryImage<F32> {
        InMemoryImage::<F32>::from_buffer(2, 2, 1, vec![0.1, 0.25, 1.0, 8.0]).unwrap()
    }

    fn multiband_image() -> InMemoryImage<F32> {
        InMemoryImage::<F32>::from_buffer(
            2,
            1,
            5,
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )
        .unwrap()
    }

    fn rgb_half_fixture_pixels() -> Vec<f32> {
        [
            0.0, 0.5, 1.0, //
            1.5, 2.0, 2.5, //
            3.0, 3.5, 4.0, //
            4.5, 5.0, 5.5,
        ]
        .into_iter()
        .map(|sample| f16::from_f32(sample).to_f32())
        .collect()
    }

    fn rgba_half_fixture_pixels() -> Vec<f32> {
        [
            0.0, 0.5, 1.0, 1.0, //
            1.5, 2.0, 2.5, 0.75, //
            3.0, 3.5, 4.0, 0.5, //
            4.5, 5.0, 5.5, 0.25,
        ]
        .into_iter()
        .map(|sample| f16::from_f32(sample).to_f32())
        .collect()
    }

    fn assert_f32_pixels_eq(lhs: &[f32], rhs: &[f32]) {
        assert_eq!(lhs.len(), rhs.len());
        for (left, right) in lhs.iter().zip(rhs.iter()) {
            assert!((left - right).abs() <= f32::EPSILON, "{left} != {right}");
        }
    }

    fn with_exr_layer_name(image: InMemoryImage<F32>, name: &str) -> InMemoryImage<F32> {
        let mut metadata = image.metadata().clone();
        metadata
            .extra
            .insert(EXR_LAYER_NAME_KEY.into(), name.to_string());
        image.with_metadata(metadata)
    }

    fn encoded_layer_names(encoded: &[u8]) -> Vec<Option<String>> {
        load_exr_layers(encoded)
            .unwrap()
            .layer_data
            .iter()
            .map(|layer| {
                layer
                    .attributes
                    .layer_name
                    .as_ref()
                    .map(ToString::to_string)
            })
            .collect()
    }

    fn metadata_value<'a>(metadata: &'a ImageMetadata, key: &str) -> &'a str {
        metadata
            .extra
            .get(key)
            .map(String::as_str)
            .unwrap_or_else(|| panic!("missing metadata key {key}"))
    }

    // ── sniff ────────────────────────────────────────────────────────────────

    #[test]
    fn sniff_recognises_exr_magic() {
        assert!(ExrCodec.sniff(&[0x76, 0x2f, 0x31, 0x01, 0x02]));
        assert!(!ExrCodec.sniff(b"\x89PNG\r\n\x1a\n"));
    }

    #[test]
    fn sniff_rejects_buffer_shorter_than_magic() {
        // Boundary: buffer with fewer than 4 bytes must not panic and return false.
        assert!(!ExrCodec.sniff(&[]));
        assert!(!ExrCodec.sniff(&[0x76, 0x2f, 0x31]));
    }

    // ── round-trip: pixel correctness ────────────────────────────────────────

    #[test]
    fn round_trip_rgba_f32_preserves_pixels() {
        let original = rgba_image();

        let encoded = ExrCodec.encode(&original).unwrap();
        let decoded = ExrCodec.decode::<F32>(&encoded).unwrap();

        assert_eq!(decoded.width(), original.width());
        assert_eq!(decoded.height(), original.height());
        assert_eq!(decoded.bands(), 4);
        assert_eq!(
            decoded.metadata().interpretation,
            Some(Interpretation::Scrgb)
        );
        assert_f32_pixels_eq(decoded.pixels(), original.pixels());
    }

    #[test]
    fn round_trip_rgb_f32_preserves_pixels() {
        let original = rgb_image();

        let encoded = ExrCodec.encode(&original).unwrap();
        let decoded = ExrCodec.decode::<F32>(&encoded).unwrap();

        assert_eq!(decoded.width(), original.width());
        assert_eq!(decoded.height(), original.height());
        assert_eq!(decoded.bands(), 3);
        assert_eq!(
            decoded.metadata().interpretation,
            Some(Interpretation::Scrgb)
        );
        assert_f32_pixels_eq(decoded.pixels(), original.pixels());
    }

    #[test]
    fn round_trip_luminance_f32_preserves_pixels() {
        let original = luminance_image();

        let encoded = ExrCodec.encode(&original).unwrap();
        let decoded = ExrCodec.decode::<F32>(&encoded).unwrap();

        assert_eq!(decoded.width(), original.width());
        assert_eq!(decoded.height(), original.height());
        assert_eq!(decoded.bands(), 1);
        assert_eq!(decoded.metadata().interpretation, Some(Interpretation::BW));
        assert_f32_pixels_eq(decoded.pixels(), original.pixels());
    }

    #[test]
    fn round_trip_luminance_alpha_f32_preserves_pixels() {
        let original = luminance_alpha_image();

        let encoded = ExrCodec.encode(&original).unwrap();
        let decoded = ExrCodec.decode::<F32>(&encoded).unwrap();

        assert_eq!(decoded.bands(), 2);
        // Y + A: 2 bands — interpretation should be BW (luminance-family)
        assert_eq!(decoded.metadata().interpretation, Some(Interpretation::BW));
        assert_f32_pixels_eq(decoded.pixels(), original.pixels());
    }

    #[test]
    fn round_trip_multiband_f32_preserves_pixels() {
        let original = multiband_image();

        let encoded = ExrCodec.encode(&original).unwrap();
        let decoded = ExrCodec.decode::<F32>(&encoded).unwrap();

        assert_eq!(decoded.bands(), 5);
        assert_eq!(
            decoded.metadata().interpretation,
            Some(Interpretation::Multiband)
        );
        assert_f32_pixels_eq(decoded.pixels(), original.pixels());
    }

    // ── round-trip: resolution metadata ──────────────────────────────────────

    #[test]
    fn resolution_metadata_survives_round_trip() {
        let xres_mm = 3.937_007_9_f64; // ~100 DPI
        let yres_mm = 3.937_007_9_f64;
        let original = rgba_image().with_metadata(ImageMetadata {
            xres: Some(xres_mm),
            yres: Some(yres_mm),
            ..ImageMetadata::default()
        });

        let encoded = ExrCodec.encode(&original).unwrap();
        let decoded = ExrCodec.decode::<F32>(&encoded).unwrap();

        // Resolution may lose a tiny bit of precision through the PPI↔mm conversion.
        let decoded_xres = decoded.metadata().xres.expect("xres must be present");
        assert!(
            (decoded_xres - xres_mm).abs() < 1e-3,
            "xres mismatch: {decoded_xres} vs {xres_mm}"
        );
    }

    #[test]
    fn strip_metadata_omits_resolution() {
        let original = rgba_image().with_metadata(ImageMetadata {
            xres: Some(3.937),
            yres: Some(3.937),
            ..ImageMetadata::default()
        });

        let opts = SaveOptions::default();
        let encoded_with = ExrCodec.encode_with_options(&original, &opts).unwrap();
        let decoded_with = ExrCodec.decode::<F32>(&encoded_with).unwrap();
        // Default (strip=false) should preserve resolution.
        assert!(decoded_with.metadata().xres.is_some());

        let strip_opts = SaveOptions {
            strip_metadata: Some(true),
            ..SaveOptions::default()
        };
        let encoded_stripped = ExrCodec
            .encode_with_options(&original, &strip_opts)
            .unwrap();
        let decoded_stripped = ExrCodec.decode::<F32>(&encoded_stripped).unwrap();
        // strip=true must drop horizontal_density from the layer.
        assert!(decoded_stripped.metadata().xres.is_none());
    }

    // ── multipart decode ──────────────────────────────────────────────────────

    #[test]
    fn multipart_exr_preserves_explicit_layer_names() {
        let rgba = with_exr_layer_name(rgba_image(), "beauty");
        let luminance = with_exr_layer_name(luminance_image(), "matte");
        let multipart = rgba
            .clone()
            .with_frames(vec![rgba.clone(), luminance.clone()]);

        let encoded = ExrCodec.encode(&multipart).unwrap();
        assert_eq!(
            encoded_layer_names(&encoded),
            vec![Some("beauty".into()), Some("matte".into())]
        );

        let decoded = ExrCodec.decode::<F32>(&encoded).unwrap();

        assert_eq!(decoded.metadata().n_pages, Some(2));
        assert_eq!(decoded.metadata().page_height, Some(2));
        let frames = decoded.frames().expect("multipart EXR must expose frames");
        assert_eq!(frames.len(), 2);
        assert_eq!(
            frames[0]
                .metadata()
                .extra
                .get(EXR_LAYER_NAME_KEY)
                .map(String::as_str),
            Some("beauty")
        );
        assert_eq!(
            frames[1]
                .metadata()
                .extra
                .get(EXR_LAYER_NAME_KEY)
                .map(String::as_str),
            Some("matte")
        );
        assert_eq!(frames[0].bands(), 4);
        assert_eq!(frames[1].bands(), 1);
        assert_f32_pixels_eq(frames[0].pixels(), rgba.pixels());
        assert_f32_pixels_eq(frames[1].pixels(), luminance.pixels());
    }

    #[test]
    fn multipart_exr_without_explicit_layer_names_uses_stub_fallback() {
        let rgba = rgba_image();
        let luminance = luminance_image();
        let multipart = rgba
            .clone()
            .with_frames(vec![rgba.clone(), luminance.clone()]);

        let encoded = ExrCodec.encode(&multipart).unwrap();
        // Honest fallback: unnamed multipart layers still need a writable EXR name.
        assert_eq!(
            encoded_layer_names(&encoded),
            vec![Some("layer0".into()), Some("layer1".into())]
        );
    }

    #[test]
    fn decode_fixture_preserves_exr_metadata_fields() {
        let decoded = ExrCodec.decode::<F32>(METADATA_FIXTURE).unwrap();
        let frames = decoded
            .frames()
            .expect("fixture must decode as multipart EXR");

        assert_eq!(
            metadata_value(decoded.metadata(), EXR_LAYER_NAMES_KEY),
            "beauty,matte"
        );
        assert_eq!(
            metadata_value(decoded.metadata(), EXR_DISPLAY_WINDOW_KEY),
            format_exr_bounds(ExrIntegerBounds::new((10, 20), (8, 8)))
        );
        assert_eq!(
            metadata_value(decoded.metadata(), EXR_DATA_WINDOW_KEY),
            format_exr_bounds(ExrIntegerBounds::new((1, 2), (2, 2)))
        );
        assert_eq!(metadata_value(decoded.metadata(), EXR_OWNER_KEY), "alice");
        assert_eq!(
            metadata_value(decoded.metadata(), EXR_COMMENTS_KEY),
            "beauty comment"
        );
        assert_eq!(
            metadata_value(decoded.metadata(), EXR_CHANNEL_NAMES_KEY),
            "R,G,B"
        );
        assert_eq!(
            metadata_value(decoded.metadata(), EXR_CHROMATICITIES_KEY),
            format_exr_chromaticities(Chromaticities {
                red: Vec2(0.64, 0.33),
                green: Vec2(0.30, 0.60),
                blue: Vec2(0.15, 0.06),
                white: Vec2(0.3127, 0.3290),
            })
        );

        assert_eq!(
            metadata_value(frames[1].metadata(), EXR_LAYER_NAME_KEY),
            "matte"
        );
        assert_eq!(metadata_value(frames[1].metadata(), EXR_OWNER_KEY), "bob");
        assert_eq!(
            metadata_value(frames[1].metadata(), EXR_COMMENTS_KEY),
            "matte comment"
        );
        assert_eq!(
            metadata_value(frames[1].metadata(), EXR_DATA_WINDOW_KEY),
            format_exr_bounds(ExrIntegerBounds::new((3, 4), (2, 2)))
        );
        assert_eq!(
            metadata_value(frames[1].metadata(), EXR_CHANNEL_NAMES_KEY),
            "Y"
        );
    }

    #[test]
    fn decode_rgb_half_fixture_preserves_pixels() {
        let decoded = ExrCodec.decode::<F32>(RGB_HALF_FIXTURE).unwrap();

        assert_eq!(
            (decoded.width(), decoded.height(), decoded.bands()),
            (2, 2, 3)
        );
        assert_eq!(
            decoded.metadata().interpretation,
            Some(Interpretation::Scrgb)
        );
        assert_f32_pixels_eq(decoded.pixels(), &rgb_half_fixture_pixels());
    }

    #[test]
    fn decode_rgba_half_fixture_preserves_pixels() {
        let decoded = ExrCodec.decode::<F32>(RGBA_HALF_FIXTURE).unwrap();

        assert_eq!(
            (decoded.width(), decoded.height(), decoded.bands()),
            (2, 2, 4)
        );
        assert_eq!(
            decoded.metadata().interpretation,
            Some(Interpretation::Scrgb)
        );
        assert_f32_pixels_eq(decoded.pixels(), &rgba_half_fixture_pixels());
    }

    #[test]
    fn decode_rgb_float_fixture_preserves_pixels() {
        let decoded = ExrCodec.decode::<F32>(RGB_FLOAT_FIXTURE).unwrap();

        assert_eq!(
            (decoded.width(), decoded.height(), decoded.bands()),
            (2, 2, 3)
        );
        assert_eq!(
            decoded.metadata().interpretation,
            Some(Interpretation::Scrgb)
        );
        assert_f32_pixels_eq(decoded.pixels(), rgb_image().pixels());
    }

    #[test]
    fn decode_rgba_float_fixture_preserves_pixels() {
        let decoded = ExrCodec.decode::<F32>(RGBA_FLOAT_FIXTURE).unwrap();

        assert_eq!(
            (decoded.width(), decoded.height(), decoded.bands()),
            (2, 2, 4)
        );
        assert_eq!(
            decoded.metadata().interpretation,
            Some(Interpretation::Scrgb)
        );
        assert_f32_pixels_eq(decoded.pixels(), rgba_image().pixels());
    }

    // ── decode_with_options: page selection ───────────────────────────────────

    #[test]
    fn decode_with_options_selects_single_page() {
        let rgba = rgba_image();
        let luminance = luminance_image();
        let multipart = rgba
            .clone()
            .with_frames(vec![rgba.clone(), luminance.clone()]);

        let encoded = ExrCodec.encode(&multipart).unwrap();

        // page=0, n=1 → only first layer (RGBA)
        let opts = LoadOptions::default().with_page(0).with_n(1);
        let decoded = ExrCodec
            .decode_with_options::<F32>(&encoded, &opts)
            .unwrap();
        assert_eq!(decoded.bands(), 4);
        assert!(decoded.frames().is_none());
    }

    #[test]
    fn decode_with_options_selects_second_page() {
        let rgba = rgba_image();
        let luminance = luminance_image();
        let multipart = rgba
            .clone()
            .with_frames(vec![rgba.clone(), luminance.clone()]);

        let encoded = ExrCodec.encode(&multipart).unwrap();

        // page=1, n=1 → only second layer (luminance)
        let opts = LoadOptions::default().with_page(1).with_n(1);
        let decoded = ExrCodec
            .decode_with_options::<F32>(&encoded, &opts)
            .unwrap();
        assert_eq!(decoded.bands(), 1);
        assert!(decoded.frames().is_none());
    }

    #[test]
    fn decode_with_options_materializes_only_requested_page() {
        let rgba = rgba_image();
        let luminance = luminance_image();
        let multiband = multiband_image();
        let multipart =
            rgba.clone()
                .with_frames(vec![rgba.clone(), luminance.clone(), multiband.clone()]);

        let encoded = ExrCodec.encode(&multipart).unwrap();
        let opts = LoadOptions::default().with_page(1).with_n(1);

        let (decoded, materialized_layers) = capture_materialized_layers(|| {
            ExrCodec
                .decode_with_options::<F32>(&encoded, &opts)
                .unwrap()
        });

        assert_eq!(decoded.bands(), 1);
        assert!(decoded.frames().is_none());
        assert_eq!(materialized_layers, vec![1]);
    }

    #[test]
    fn decode_with_options_materializes_only_requested_page_range() {
        let rgba = rgba_image();
        let luminance = luminance_image();
        let multiband = multiband_image();
        let multipart =
            rgba.clone()
                .with_frames(vec![rgba.clone(), luminance.clone(), multiband.clone()]);

        let encoded = ExrCodec.encode(&multipart).unwrap();
        let opts = LoadOptions::default().with_page(1).with_n(-1);

        let (decoded, materialized_layers) = capture_materialized_layers(|| {
            ExrCodec
                .decode_with_options::<F32>(&encoded, &opts)
                .unwrap()
        });

        let frames = decoded
            .frames()
            .expect("page range selection must keep requested frames");
        assert_eq!(frames.len(), 2);
        assert_eq!(materialized_layers, vec![1, 2]);
    }

    #[test]
    fn decode_with_options_n_minus_one_selects_all_remaining() {
        let rgba = rgba_image();
        let luminance = luminance_image();
        let multipart = rgba
            .clone()
            .with_frames(vec![rgba.clone(), luminance.clone()]);

        let encoded = ExrCodec.encode(&multipart).unwrap();

        // page=0, n=-1 → all layers
        let opts = LoadOptions::default().with_page(0).with_n(-1);
        let decoded = ExrCodec
            .decode_with_options::<F32>(&encoded, &opts)
            .unwrap();
        let frames = decoded.frames().expect("n=-1 must produce frames");
        assert_eq!(frames.len(), 2);
    }

    #[test]
    fn decode_with_options_on_corrupt_multipart_offset_table_returns_typed_error() {
        let rgba = rgba_image();
        let luminance = luminance_image();
        let multipart = rgba
            .clone()
            .with_frames(vec![rgba.clone(), luminance.clone()]);
        let encoded = ExrCodec.encode(&multipart).unwrap();
        let corrupted = corrupt_second_layer_offset_table_entry(&encoded);

        let opts = LoadOptions::default().with_page(1).with_n(1);
        let result = ExrCodec.decode_with_options::<F32>(&corrupted, &opts);

        assert!(matches!(
            result,
            Err(ViprsError::Exr(
                ExrCodecError::InvalidMultipartBlockReference {
                    requested_layer: 0,
                    selected_start: 1,
                    selected_count: 1,
                    header_count: 2,
                }
            ))
        ));
    }

    // ── probe ────────────────────────────────────────────────────────────────

    #[test]
    fn probe_returns_dimensions_of_first_layer() {
        let original = rgba_image();
        let encoded = ExrCodec.encode(&original).unwrap();

        let (w, h, bands) = ExrCodec.probe(&encoded).unwrap();
        assert_eq!(w, 2);
        assert_eq!(h, 2);
        assert_eq!(bands, 4);
    }

    #[test]
    fn probe_reads_only_metadata_without_materializing_layers() {
        let rgba = rgba_image();
        let luminance = luminance_image();
        let multipart = rgba.clone().with_frames(vec![rgba, luminance]);
        let encoded = ExrCodec.encode(&multipart).unwrap();

        let ((w, h, bands), materialized_layers) =
            capture_materialized_layers(|| ExrCodec.probe(&encoded).unwrap());

        assert_eq!((w, h, bands), (2, 2, 4));
        assert!(materialized_layers.is_empty());
    }

    #[test]
    fn probe_on_corrupted_data_returns_error() {
        let result = ExrCodec.probe(b"not an exr file at all");
        assert!(result.is_err());
    }

    // ── error paths ──────────────────────────────────────────────────────────

    #[test]
    fn decode_non_f32_format_returns_typed_error() {
        let original = rgba_image();
        let encoded = ExrCodec.encode(&original).unwrap();
        let result = ExrCodec.decode::<U8>(&encoded);
        assert!(matches!(
            result,
            Err(ViprsError::Exr(ExrCodecError::UnsupportedFormat {
                format: BandFormatId::U8
            }))
        ));
    }

    #[test]
    fn encode_non_f32_format_returns_typed_error() {
        let u8_image = InMemoryImage::<U8>::from_buffer(2, 2, 3, vec![0u8; 12]).unwrap();
        let result = ExrCodec.encode(&u8_image);
        assert!(matches!(
            result,
            Err(ViprsError::Exr(ExrCodecError::UnsupportedFormat {
                format: BandFormatId::U8
            }))
        ));
    }

    #[test]
    fn decode_empty_buffer_returns_error() {
        let result = ExrCodec.decode::<F32>(b"");
        assert!(result.is_err());
    }

    #[test]
    fn decode_page_out_of_range_returns_typed_error() {
        let original = luminance_image();
        let encoded = ExrCodec.encode(&original).unwrap();

        let opts = LoadOptions::default().with_page(5).with_n(1);
        let result = ExrCodec.decode_with_options::<F32>(&encoded, &opts);
        assert!(matches!(
            result,
            Err(ViprsError::Exr(ExrCodecError::RequestedLayerOutOfRange {
                requested: 5,
                total_layers: 1
            }))
        ));
    }

    #[test]
    fn decode_invalid_negative_n_returns_typed_error() {
        let original = luminance_image();
        let encoded = ExrCodec.encode(&original).unwrap();

        let opts = LoadOptions::default().with_n(-2);
        let result = ExrCodec.decode_with_options::<F32>(&encoded, &opts);
        assert!(matches!(
            result,
            Err(ViprsError::Exr(ExrCodecError::InvalidLayerCount {
                value: -2
            }))
        ));
    }

    // ── interpretation mapping ────────────────────────────────────────────────

    #[test]
    fn interpretation_for_rgb_channels_is_scrgb() {
        let names = vec!["R".to_string(), "G".to_string(), "B".to_string()];
        assert_eq!(
            interpretation_for_channel_names(&names),
            Interpretation::Scrgb
        );
    }

    #[test]
    fn interpretation_for_single_luminance_y_is_bw() {
        let names = vec!["Y".to_string()];
        assert_eq!(interpretation_for_channel_names(&names), Interpretation::BW);
    }

    #[test]
    fn interpretation_for_single_luminance_l_is_bw() {
        let names = vec!["L".to_string()];
        assert_eq!(interpretation_for_channel_names(&names), Interpretation::BW);
    }

    #[test]
    fn interpretation_for_ya_channels_is_bw() {
        let names = vec!["Y".to_string(), "A".to_string()];
        assert_eq!(interpretation_for_channel_names(&names), Interpretation::BW);
    }

    #[test]
    fn interpretation_for_many_non_rgb_channels_is_multiband() {
        let names = vec![
            "C0".to_string(),
            "C1".to_string(),
            "C2".to_string(),
            "C3".to_string(),
            "C4".to_string(),
        ];
        assert_eq!(
            interpretation_for_channel_names(&names),
            Interpretation::Multiband
        );
    }

    // ── channel_names helper ──────────────────────────────────────────────────

    #[test]
    fn channel_names_for_standard_band_counts() {
        assert_eq!(channel_names(1), vec!["Y"]);
        assert_eq!(channel_names(2), vec!["Y", "A"]);
        assert_eq!(channel_names(3), vec!["R", "G", "B"]);
        assert_eq!(channel_names(4), vec!["R", "G", "B", "A"]);
    }

    #[test]
    fn channel_names_for_arbitrary_count_uses_c_prefix() {
        let names = channel_names(6);
        assert_eq!(names.len(), 6);
        for (i, name) in names.iter().enumerate() {
            assert_eq!(name, &format!("C{i}"));
        }
    }

    // ── pixels_per_mm_to_ppi helper ───────────────────────────────────────────

    #[test]
    fn pixels_per_mm_to_ppi_converts_correctly() {
        // 25.4 px/mm ≈ 645 DPI
        let result = pixels_per_mm_to_ppi(25.4);
        assert!(result.is_some());
        let ppi = result.unwrap();
        assert!((ppi - 645.16_f32).abs() < 0.1, "unexpected ppi: {ppi}");
    }

    #[test]
    fn pixels_per_mm_to_ppi_rejects_non_finite() {
        assert!(pixels_per_mm_to_ppi(f64::INFINITY).is_none());
        assert!(pixels_per_mm_to_ppi(f64::NAN).is_none());
        assert!(pixels_per_mm_to_ppi(-1.0).is_none());
    }

    // ── approx_pixels_per_mm helper ───────────────────────────────────────────

    #[test]
    fn approx_pixels_per_mm_with_unit_aspect_is_symmetric() {
        let (xres, yres) = approx_pixels_per_mm(96.0, 1.0);
        assert!(
            (xres - yres).abs() < 1e-9,
            "x and y should match when aspect=1"
        );
    }

    #[test]
    fn approx_pixels_per_mm_with_non_finite_aspect_falls_back_to_one() {
        let (xres_normal, _) = approx_pixels_per_mm(96.0, 1.0);
        let (xres_nan, yres_nan) = approx_pixels_per_mm(96.0, f32::NAN);
        // With NaN aspect, y should fall back to x (aspect treated as 1.0)
        assert!((xres_nan - xres_normal).abs() < 1e-9);
        assert!((yres_nan - xres_normal).abs() < 1e-9);
    }
}

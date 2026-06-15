//! GIF codec — decode and encode via the `gif` 0.13 crate (pure Rust).
//!
//! # Limitations
//!
//! - **Decode format**: only `U8` is supported. Decoded output is **RGB** for
//!   opaque GIFs and **RGBA** (4 bands) when any frame uses transparency, so
//!   transparent pixels preserve their alpha channel on decode.
//! - **Encode**: produces a static (single-frame) GIF. Animated GIF output is
//!   not supported.
//! - **Encode memory model**: encoding is inherently two-pass. The codec must
//!   inspect the full frame to build a palette before it can write indexed
//!   pixels, so tile-by-tile streaming is not supported.
//! - **Encode format**: supports `U8` RGB (3 bands) and RGBA (4 bands). RGBA is
//!   thresholded to GIF's single transparent index using the same 128 alpha cut
//!   that libvips `cgifsave` applies before quantization.
//! - **Quantisation parity**: libvips uses libimagequant/quantizr plus explicit
//!   Floyd-Steinberg dithering. Viprs approximates that behaviour with an
//!   in-tree median-cut palette builder and Floyd-Steinberg remapper.
//!
//! # Gating
//!
//! This file is compiled only when the `gif` Cargo feature is enabled.

use std::{
    borrow::Cow,
    collections::HashSet,
    io::Cursor,
    sync::{Mutex, MutexGuard},
};

use gif::{
    ColorOutput, DecodeOptions, DisposalMethod as GifDisposalMethod, Encoder, Frame, Repeat,
};

use crate::domain::codec_options::{LoadOptions, SaveOptions};
use crate::domain::error::ViprsError;
use crate::domain::format::{BandFormat, BandFormatId};
use crate::domain::image::{
    AnimationFrame, AnimationLoopCount, FrameDisposal, Image, ImageMetadata,
};
use crate::ports::codec::{ImageDecoder, ImageEncoder};

/// GIF codec: implements both [`ImageDecoder`] and [`ImageEncoder`].
pub struct GifCodec {
    encode_buffer: Mutex<GifEncodeBuffer>,
}

const DEFAULT_GIF_COLORS: u16 = 256;
const DEFAULT_DITHER: bool = true;
const TRANSPARENT_ALPHA_THRESHOLD: u8 = 128;
const QUANTIZED_RGB_BITS: usize = 5;
const QUANTIZED_RGB_SHIFT: usize = 8 - QUANTIZED_RGB_BITS;
const QUANTIZED_RGB_LEVELS: usize = 1 << QUANTIZED_RGB_BITS;
const QUANTIZED_RGB_BINS: usize =
    QUANTIZED_RGB_LEVELS * QUANTIZED_RGB_LEVELS * QUANTIZED_RGB_LEVELS;

#[derive(Clone, Copy)]
struct WeightedColor {
    rgb: [u8; 3],
    count: u32,
}

struct ColorBox {
    colors: Vec<WeightedColor>,
    total_weight: u32,
    min: [u8; 3],
    max: [u8; 3],
}

struct GifEncodeBuffer {
    quantized_histogram: Vec<u32>,
    quantized_sum_r: Vec<u64>,
    quantized_sum_g: Vec<u64>,
    quantized_sum_b: Vec<u64>,
    touched_quantized_bins: Vec<usize>,
    palette_lookup: Vec<u8>,
    dither_errors: Vec<[i32; 3]>,
    index_map: Vec<u8>,
}

impl GifEncodeBuffer {
    fn new() -> Self {
        Self {
            quantized_histogram: Vec::new(),
            quantized_sum_r: Vec::new(),
            quantized_sum_g: Vec::new(),
            quantized_sum_b: Vec::new(),
            touched_quantized_bins: Vec::new(),
            palette_lookup: Vec::new(),
            dither_errors: Vec::new(),
            index_map: Vec::new(),
        }
    }

    fn prepare_quantized_histogram(&mut self) {
        if self.quantized_histogram.len() < QUANTIZED_RGB_BINS {
            self.quantized_histogram.resize(QUANTIZED_RGB_BINS, 0);
            self.quantized_sum_r.resize(QUANTIZED_RGB_BINS, 0);
            self.quantized_sum_g.resize(QUANTIZED_RGB_BINS, 0);
            self.quantized_sum_b.resize(QUANTIZED_RGB_BINS, 0);
        }
        for &index in &self.touched_quantized_bins {
            self.quantized_histogram[index] = 0;
            self.quantized_sum_r[index] = 0;
            self.quantized_sum_g[index] = 0;
            self.quantized_sum_b[index] = 0;
        }
        self.touched_quantized_bins.clear();
    }

    fn prepare_palette_lookup(&mut self, opaque_palette: &[[u8; 3]]) -> &[u8] {
        if self.palette_lookup.len() < QUANTIZED_RGB_BINS {
            self.palette_lookup.resize(QUANTIZED_RGB_BINS, 0);
        }
        for (bin, palette_offset) in self.palette_lookup.iter_mut().enumerate() {
            *palette_offset = nearest_palette_offset(quantized_rgb(bin), opaque_palette);
        }
        &self.palette_lookup[..QUANTIZED_RGB_BINS]
    }

    fn prepare_index_map(&mut self, pixel_count: usize) -> &mut [u8] {
        if self.index_map.len() < pixel_count {
            self.index_map.resize(pixel_count, 0);
        }
        let indices = &mut self.index_map[..pixel_count];
        indices.fill(0);
        indices
    }

    fn prepare_frame_buffers(
        &mut self,
        pixel_count: usize,
        width: usize,
    ) -> (&mut [u8], &mut [[i32; 3]]) {
        let GifEncodeBuffer {
            index_map,
            dither_errors,
            ..
        } = self;
        if index_map.len() < pixel_count {
            index_map.resize(pixel_count, 0);
        }
        let indices = &mut index_map[..pixel_count];
        indices.fill(0);
        let error_len = (width + 2) * 2;
        if dither_errors.len() < error_len {
            dither_errors.resize(error_len, [0; 3]);
        }
        let errors = &mut dither_errors[..error_len];
        errors.fill([0; 3]);
        (indices, errors)
    }
}

impl GifCodec {
    #[must_use]
    /// `new` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let _ = viprs::adapters::codecs::gif::new;
    /// ```
    pub fn new() -> Self {
        Self {
            encode_buffer: Mutex::new(GifEncodeBuffer::new()),
        }
    }
}

impl Default for GifCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl ColorBox {
    fn new(colors: Vec<WeightedColor>) -> Option<Self> {
        if colors.is_empty() {
            return None;
        }

        let mut min = [u8::MAX; 3];
        let mut max = [u8::MIN; 3];
        let mut total_weight = 0u32;

        for color in &colors {
            total_weight = total_weight.saturating_add(color.count);
            for channel in 0..3 {
                min[channel] = min[channel].min(color.rgb[channel]);
                max[channel] = max[channel].max(color.rgb[channel]);
            }
        }

        Some(Self {
            colors,
            total_weight,
            min,
            max,
        })
    }

    fn split_score(&self) -> (u8, u32) {
        let ranges = self.channel_ranges();
        let max_range = *ranges.iter().max().unwrap_or(&0);
        (max_range, self.total_weight)
    }

    fn channel_ranges(&self) -> [u8; 3] {
        [
            self.max[0].saturating_sub(self.min[0]),
            self.max[1].saturating_sub(self.min[1]),
            self.max[2].saturating_sub(self.min[2]),
        ]
    }

    fn split(self) -> Option<(Self, Self)> {
        if self.colors.len() < 2 {
            return None;
        }

        let ranges = self.channel_ranges();
        let split_channel = (0..3).max_by_key(|&channel| (ranges[channel], self.max[channel]))?;
        let mut colors = self.colors;
        colors.sort_unstable_by_key(|color| color.rgb[split_channel]);

        let mut cumulative_weight = 0u32;
        let midpoint = self.total_weight / 2;
        let mut split_at = 0usize;
        for (index, color) in colors.iter().enumerate() {
            cumulative_weight = cumulative_weight.saturating_add(color.count);
            if cumulative_weight >= midpoint {
                split_at = index + 1;
                break;
            }
        }

        split_at = split_at.clamp(1, colors.len().saturating_sub(1));
        let right_colors = colors.split_off(split_at);
        let left = Self::new(colors)?;
        let right = Self::new(right_colors)?;
        Some((left, right))
    }

    fn representative_color(&self) -> [u8; 3] {
        let mut weighted_sum = [0u64; 3];
        let total_weight = u64::from(self.total_weight.max(1));

        for color in &self.colors {
            let count = u64::from(color.count);
            for (channel, sum) in weighted_sum.iter_mut().enumerate() {
                *sum = sum.saturating_add(u64::from(color.rgb[channel]) * count);
            }
        }

        [
            ((weighted_sum[0] + total_weight / 2) / total_weight) as u8,
            ((weighted_sum[1] + total_weight / 2) / total_weight) as u8,
            ((weighted_sum[2] + total_weight / 2) / total_weight) as u8,
        ]
    }
}

fn has_transparency(pixel_bytes: &[u8], bands: usize) -> bool {
    bands == 4
        && pixel_bytes
            .chunks_exact(bands)
            .any(|pixel| pixel[3] < TRANSPARENT_ALPHA_THRESHOLD)
}

fn validate_color_limit(opts: &SaveOptions) -> Result<u16, ViprsError> {
    let colors = opts.colors.unwrap_or(DEFAULT_GIF_COLORS);
    if !(2..=DEFAULT_GIF_COLORS).contains(&colors) {
        return Err(ViprsError::Codec(format!(
            "gif: colors must be in 2..=256, got {colors}"
        )));
    }
    Ok(colors)
}

fn quantized_index(rgb: [u8; 3]) -> usize {
    ((usize::from(rgb[0]) >> QUANTIZED_RGB_SHIFT) << (QUANTIZED_RGB_BITS * 2))
        | ((usize::from(rgb[1]) >> QUANTIZED_RGB_SHIFT) << QUANTIZED_RGB_BITS)
        | (usize::from(rgb[2]) >> QUANTIZED_RGB_SHIFT)
}

fn quantized_rgb(index: usize) -> [u8; 3] {
    let midpoint = 1usize << (QUANTIZED_RGB_SHIFT.saturating_sub(1));
    let red =
        ((index >> (QUANTIZED_RGB_BITS * 2)) & (QUANTIZED_RGB_LEVELS - 1)) << QUANTIZED_RGB_SHIFT;
    let green = ((index >> QUANTIZED_RGB_BITS) & (QUANTIZED_RGB_LEVELS - 1)) << QUANTIZED_RGB_SHIFT;
    let blue = (index & (QUANTIZED_RGB_LEVELS - 1)) << QUANTIZED_RGB_SHIFT;
    [
        (red + midpoint).min(255) as u8,
        (green + midpoint).min(255) as u8,
        (blue + midpoint).min(255) as u8,
    ]
}

fn exact_palette(pixel_bytes: &[u8], bands: usize, max_colors: usize) -> Option<Vec<[u8; 3]>> {
    let mut palette = HashSet::with_capacity(max_colors.min(usize::from(DEFAULT_GIF_COLORS)));
    for pixel in pixel_bytes.chunks_exact(bands) {
        if bands == 4 && pixel[3] < TRANSPARENT_ALPHA_THRESHOLD {
            continue;
        }

        let rgb = [pixel[0], pixel[1], pixel[2]];
        if palette.insert(rgb) && palette.len() > max_colors {
            return None;
        }
    }

    let mut palette: Vec<[u8; 3]> = palette.into_iter().collect();
    palette.sort_unstable();
    Some(palette)
}

fn build_quantized_histogram(pixel_bytes: &[u8], bands: usize, buffer: &mut GifEncodeBuffer) {
    buffer.prepare_quantized_histogram();
    for pixel in pixel_bytes.chunks_exact(bands) {
        if bands == 4 && pixel[3] < TRANSPARENT_ALPHA_THRESHOLD {
            continue;
        }

        let rgb = [pixel[0], pixel[1], pixel[2]];
        let index = quantized_index(rgb);
        if buffer.quantized_histogram[index] == 0 {
            buffer.touched_quantized_bins.push(index);
        }
        buffer.quantized_histogram[index] = buffer.quantized_histogram[index].saturating_add(1);
        buffer.quantized_sum_r[index] =
            buffer.quantized_sum_r[index].saturating_add(u64::from(rgb[0]));
        buffer.quantized_sum_g[index] =
            buffer.quantized_sum_g[index].saturating_add(u64::from(rgb[1]));
        buffer.quantized_sum_b[index] =
            buffer.quantized_sum_b[index].saturating_add(u64::from(rgb[2]));
    }
}

fn rgba_to_rgb(width: u32, height: u32, rgba: &[u8]) -> Result<Vec<u8>, ViprsError> {
    let expected_len = width as usize * height as usize * 4;
    if rgba.len() < expected_len {
        return Err(ViprsError::Codec(format!(
            "gif: frame buffer too small: expected {expected_len}, got {}",
            rgba.len()
        )));
    }

    let mut rgb: Vec<u8> = Vec::with_capacity(width as usize * height as usize * 3);
    for chunk in rgba[..expected_len].chunks_exact(4) {
        rgb.push(chunk[0]);
        rgb.push(chunk[1]);
        rgb.push(chunk[2]);
    }

    Ok(rgb)
}

fn checked_rgba_len(width: u32, height: u32) -> Result<usize, ViprsError> {
    let pixel_count = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or_else(|| {
            ViprsError::Codec(format!("gif: canvas dimensions overflow: {width}x{height}"))
        })?;
    let byte_count = pixel_count.checked_mul(4).ok_or_else(|| {
        ViprsError::Codec(format!(
            "gif: canvas byte length overflow: {width}x{height}"
        ))
    })?;
    usize::try_from(byte_count).map_err(|_| {
        ViprsError::Codec(format!(
            "gif: canvas byte length overflows usize: {width}x{height}"
        ))
    })
}

fn rgba_has_transparency(rgba: &[u8]) -> bool {
    rgba.chunks_exact(4).any(|pixel| pixel[3] < u8::MAX)
}

fn validate_frame_bounds(
    canvas_width: u32,
    canvas_height: u32,
    frame_left: u16,
    frame_top: u16,
    frame_width: u16,
    frame_height: u16,
) -> Result<(), ViprsError> {
    let frame_right = u32::from(frame_left) + u32::from(frame_width);
    let frame_bottom = u32::from(frame_top) + u32::from(frame_height);
    if frame_right > canvas_width || frame_bottom > canvas_height {
        return Err(ViprsError::Codec(format!(
            "gif: frame rect {}x{} at ({}, {}) exceeds logical screen {}x{}",
            frame_width, frame_height, frame_left, frame_top, canvas_width, canvas_height
        )));
    }
    Ok(())
}

fn composite_frame_rgba(
    canvas: &mut [u8],
    canvas_width: u32,
    canvas_height: u32,
    frame: &Frame<'_>,
) -> Result<(), ViprsError> {
    validate_frame_bounds(
        canvas_width,
        canvas_height,
        frame.left,
        frame.top,
        frame.width,
        frame.height,
    )?;

    let frame_len = usize::from(frame.width) * usize::from(frame.height) * 4;
    if frame.buffer.len() != frame_len {
        return Err(ViprsError::Codec(format!(
            "gif: frame buffer length {} does not match {}x{} RGBA payload {}",
            frame.buffer.len(),
            frame.width,
            frame.height,
            frame_len
        )));
    }

    let canvas_width = usize::try_from(canvas_width)
        .map_err(|_| ViprsError::Codec("gif: canvas width overflows usize".into()))?;
    let frame_width = usize::from(frame.width);
    let frame_left = usize::from(frame.left);
    let frame_top = usize::from(frame.top);
    let frame_height = usize::from(frame.height);

    for row in 0..frame_height {
        let src_row_start = row * frame_width * 4;
        let dest_row_start = ((frame_top + row) * canvas_width + frame_left) * 4;
        let src_row = &frame.buffer[src_row_start..src_row_start + frame_width * 4];
        let dest_row = &mut canvas[dest_row_start..dest_row_start + frame_width * 4];

        for (src_pixel, dest_pixel) in src_row.chunks_exact(4).zip(dest_row.chunks_exact_mut(4)) {
            if src_pixel[3] == 0 {
                continue;
            }
            dest_pixel.copy_from_slice(src_pixel);
        }
    }

    Ok(())
}

fn clear_frame_rect_rgba(
    canvas: &mut [u8],
    canvas_width: u32,
    frame_left: u16,
    frame_top: u16,
    frame_width: u16,
    frame_height: u16,
    canvas_height: u32,
) -> Result<(), ViprsError> {
    validate_frame_bounds(
        canvas_width,
        canvas_height,
        frame_left,
        frame_top,
        frame_width,
        frame_height,
    )?;

    let canvas_width = usize::try_from(canvas_width)
        .map_err(|_| ViprsError::Codec("gif: canvas width overflows usize".into()))?;
    let frame_width = usize::from(frame_width);
    let frame_left = usize::from(frame_left);
    let frame_top = usize::from(frame_top);
    let frame_height = usize::from(frame_height);

    for row in 0..frame_height {
        let dest_row_start = ((frame_top + row) * canvas_width + frame_left) * 4;
        canvas[dest_row_start..dest_row_start + frame_width * 4].fill(0);
    }

    Ok(())
}

fn frame_has_transparency(frame: &Frame<'_>) -> bool {
    frame.transparent.is_some()
}

#[inline]
fn delay_ms_to_gif_units(delay_ms: u32) -> u16 {
    let centiseconds = delay_ms.saturating_add(5) / 10;
    centiseconds.min(u32::from(u16::MAX)) as u16
}

#[inline]
fn delay_ms_from_gif_units(delay_units: u16) -> u32 {
    u32::from(delay_units) * 10
}

#[inline]
const fn frame_disposal_to_gif(disposal: FrameDisposal) -> GifDisposalMethod {
    match disposal {
        FrameDisposal::Any => GifDisposalMethod::Any,
        FrameDisposal::Keep => GifDisposalMethod::Keep,
        FrameDisposal::Background => GifDisposalMethod::Background,
        FrameDisposal::Previous => GifDisposalMethod::Previous,
    }
}

#[inline]
const fn frame_disposal_from_gif(disposal: GifDisposalMethod) -> FrameDisposal {
    match disposal {
        GifDisposalMethod::Any => FrameDisposal::Any,
        GifDisposalMethod::Keep => FrameDisposal::Keep,
        GifDisposalMethod::Background => FrameDisposal::Background,
        GifDisposalMethod::Previous => FrameDisposal::Previous,
    }
}

#[inline]
const fn animation_loop_to_gif(loop_count: Option<AnimationLoopCount>) -> Repeat {
    match loop_count {
        Some(AnimationLoopCount::Infinite) => Repeat::Infinite,
        Some(AnimationLoopCount::Finite(loop_count)) => Repeat::Finite(loop_count),
        None => Repeat::Finite(0),
    }
}

#[inline]
const fn animation_loop_from_gif(loop_count: Repeat) -> AnimationLoopCount {
    match loop_count {
        Repeat::Infinite => AnimationLoopCount::Infinite,
        Repeat::Finite(loop_count) => AnimationLoopCount::Finite(loop_count),
    }
}

fn median_cut_palette(buffer: &GifEncodeBuffer, max_colors: usize) -> Vec<[u8; 3]> {
    // libvips `cgifsave` delegates high-colour palette search to libimagequant.
    // Viprs intentionally keeps an in-tree median-cut fallback, so high-colour
    // parity is validated with bounded MAE against libvips rather than exact
    // palette identity.
    if buffer.touched_quantized_bins.is_empty() {
        return Vec::new();
    }

    let weighted_colors: Vec<WeightedColor> = buffer
        .touched_quantized_bins
        .iter()
        .copied()
        .map(|index| {
            let count = u64::from(buffer.quantized_histogram[index].max(1));
            WeightedColor {
                rgb: [
                    ((buffer.quantized_sum_r[index] + count / 2) / count) as u8,
                    ((buffer.quantized_sum_g[index] + count / 2) / count) as u8,
                    ((buffer.quantized_sum_b[index] + count / 2) / count) as u8,
                ],
                count: buffer.quantized_histogram[index],
            }
        })
        .collect();

    let Some(root_box) = ColorBox::new(weighted_colors) else {
        return Vec::new();
    };

    let mut boxes = vec![root_box];
    while boxes.len() < max_colors {
        let Some((split_index, _)) = boxes
            .iter()
            .enumerate()
            .filter(|(_, palette_box)| palette_box.colors.len() > 1)
            .max_by_key(|(_, palette_box)| palette_box.split_score())
        else {
            break;
        };

        let palette_box = boxes.swap_remove(split_index);
        if let Some((left, right)) = palette_box.split() {
            boxes.push(left);
            boxes.push(right);
        } else {
            break;
        }
    }

    let mut palette: Vec<[u8; 3]> = boxes.iter().map(ColorBox::representative_color).collect();
    palette.sort_unstable();
    palette
}

fn build_palette_bytes(opaque_palette: &[[u8; 3]], has_transparent_index: bool) -> Vec<u8> {
    let transparent_entries = usize::from(has_transparent_index);
    let mut palette_bytes = Vec::with_capacity((opaque_palette.len() + transparent_entries) * 3);

    if has_transparent_index {
        palette_bytes.extend_from_slice(&[0, 0, 0]);
    }

    for color in opaque_palette {
        palette_bytes.extend_from_slice(color);
    }

    palette_bytes
}

fn nearest_palette_offset(rgb: [u8; 3], opaque_palette: &[[u8; 3]]) -> u8 {
    let mut best_index = 0usize;
    let mut best_distance = u32::MAX;

    for (index, palette_color) in opaque_palette.iter().enumerate() {
        let dr = i32::from(rgb[0]) - i32::from(palette_color[0]);
        let dg = i32::from(rgb[1]) - i32::from(palette_color[1]);
        let db = i32::from(rgb[2]) - i32::from(palette_color[2]);
        let distance = (dr * dr + dg * dg + db * db) as u32;
        if distance < best_distance {
            best_distance = distance;
            best_index = index;
            if distance == 0 {
                break;
            }
        }
    }

    best_index as u8
}

fn encode_index(offset: u8, transparent: bool) -> u8 {
    if transparent {
        offset.saturating_add(1)
    } else {
        offset
    }
}

fn apply_dither_error(sample: u8, error: i32) -> u8 {
    let rounded_error = if error >= 0 {
        (error + 8) >> 4
    } else {
        -(((-error) + 8) >> 4)
    };
    (i32::from(sample) + rounded_error).clamp(0, 255) as u8
}

fn remap_without_dither(
    pixel_bytes: &[u8],
    bands: usize,
    palette_lookup: &[u8],
    has_transparent_index: bool,
    indices: &mut [u8],
) {
    for (pixel, index) in pixel_bytes.chunks_exact(bands).zip(indices.iter_mut()) {
        if bands == 4 && pixel[3] < TRANSPARENT_ALPHA_THRESHOLD {
            *index = 0;
            continue;
        }

        let palette_offset = palette_lookup[quantized_index([pixel[0], pixel[1], pixel[2]])];
        *index = encode_index(palette_offset, has_transparent_index);
    }
}

fn remap_with_dither(
    pixel_bytes: &[u8],
    width: usize,
    height: usize,
    bands: usize,
    palette_lookup: &[u8],
    opaque_palette: &[[u8; 3]],
    has_transparent_index: bool,
    indices: &mut [u8],
    dither_errors: &mut [[i32; 3]],
) {
    let row_len = width + 2;
    let (mut current_errors, mut next_errors) = dither_errors.split_at_mut(row_len);
    for y in 0..height {
        next_errors.fill([0; 3]);

        for x in 0..width {
            let pixel_index = y * width + x;
            let source = &pixel_bytes[pixel_index * bands..(pixel_index + 1) * bands];
            if bands == 4 && source[3] < TRANSPARENT_ALPHA_THRESHOLD {
                indices[pixel_index] = 0;
                continue;
            }

            let adjusted = [
                apply_dither_error(source[0], current_errors[x + 1][0]),
                apply_dither_error(source[1], current_errors[x + 1][1]),
                apply_dither_error(source[2], current_errors[x + 1][2]),
            ];
            let palette_offset = palette_lookup[quantized_index(adjusted)];
            let palette_color = opaque_palette[usize::from(palette_offset)];
            indices[pixel_index] = encode_index(palette_offset, has_transparent_index);

            let error = [
                i32::from(adjusted[0]) - i32::from(palette_color[0]),
                i32::from(adjusted[1]) - i32::from(palette_color[1]),
                i32::from(adjusted[2]) - i32::from(palette_color[2]),
            ];

            for channel in 0..3 {
                current_errors[x + 2][channel] += error[channel] * 7;
                next_errors[x][channel] += error[channel] * 3;
                next_errors[x + 1][channel] += error[channel] * 5;
                next_errors[x + 2][channel] += error[channel];
            }
        }

        std::mem::swap(&mut current_errors, &mut next_errors);
    }
}

fn remap_exact_palette(
    pixel_bytes: &[u8],
    bands: usize,
    opaque_palette: &[[u8; 3]],
    has_transparent_index: bool,
    indices: &mut [u8],
) {
    for (pixel, index) in pixel_bytes.chunks_exact(bands).zip(indices.iter_mut()) {
        if bands == 4 && pixel[3] < TRANSPARENT_ALPHA_THRESHOLD {
            *index = 0;
            continue;
        }

        let rgb = [pixel[0], pixel[1], pixel[2]];
        let palette_offset = match opaque_palette.binary_search(&rgb) {
            Ok(found) => found as u8,
            Err(_) => {
                debug_assert!(
                    false,
                    "exact-palette remap requires every opaque pixel to exist in the palette"
                );
                0
            }
        };
        *index = encode_index(palette_offset, has_transparent_index);
    }
}

fn quantize_frame<'a>(
    pixel_bytes: &[u8],
    width: u16,
    height: u16,
    bands: usize,
    opts: &SaveOptions,
    buffer: &'a mut GifEncodeBuffer,
) -> Result<Frame<'a>, ViprsError> {
    // Build a fixed palette first, then remap with optional dithering.
    let total_colors = validate_color_limit(opts)?;
    let uses_transparency = has_transparency(pixel_bytes, bands);
    let opaque_limit = total_colors
        .saturating_sub(u16::from(uses_transparency))
        .max(1);
    let pixel_count = usize::from(width) * usize::from(height);
    let exact_palette = exact_palette(pixel_bytes, bands, usize::from(opaque_limit));
    let uses_exact_palette = exact_palette.is_some();
    let opaque_palette = if let Some(exact_palette) = exact_palette {
        exact_palette
    } else {
        build_quantized_histogram(pixel_bytes, bands, buffer);
        median_cut_palette(buffer, usize::from(opaque_limit))
    };

    let palette_bytes = build_palette_bytes(&opaque_palette, uses_transparency);
    if uses_exact_palette && !opaque_palette.is_empty() {
        let indices = buffer.prepare_index_map(pixel_count);
        remap_exact_palette(
            pixel_bytes,
            bands,
            &opaque_palette,
            uses_transparency,
            indices,
        );
        return Ok(Frame {
            width,
            height,
            palette: Some(palette_bytes),
            buffer: Cow::Borrowed(indices),
            transparent: uses_transparency.then_some(0),
            ..Frame::default()
        });
    }
    let palette_lookup = if opaque_palette.is_empty() {
        Vec::new()
    } else {
        buffer.prepare_palette_lookup(&opaque_palette).to_vec()
    };
    let (indices, dither_errors) = buffer.prepare_frame_buffers(pixel_count, usize::from(width));
    if opts.dither.unwrap_or(DEFAULT_DITHER) && !opaque_palette.is_empty() {
        remap_with_dither(
            pixel_bytes,
            usize::from(width),
            usize::from(height),
            bands,
            &palette_lookup,
            &opaque_palette,
            uses_transparency,
            indices,
            dither_errors,
        );
    } else if opaque_palette.is_empty() {
        indices.fill(0);
    } else {
        remap_without_dither(
            pixel_bytes,
            bands,
            &palette_lookup,
            uses_transparency,
            indices,
        );
    }

    Ok(Frame {
        width,
        height,
        palette: Some(palette_bytes),
        buffer: Cow::Borrowed(indices),
        transparent: uses_transparency.then_some(0),
        ..Frame::default()
    })
}

// ── ImageDecoder ──────────────────────────────────────────────────────────────

impl ImageDecoder for GifCodec {
    fn format_name(&self) -> &'static str {
        "gif"
    }

    /// Recognise a GIF byte stream by its magic bytes.
    ///
    /// GIF files begin with either `GIF87a` or `GIF89a`.
    fn sniff(&self, header: &[u8]) -> bool
    where
        Self: Sized,
    {
        header.len() >= 6 && (&header[..6] == b"GIF87a" || &header[..6] == b"GIF89a")
    }

    fn decode<F: BandFormat>(&self, src: &[u8]) -> Result<Image<F>, ViprsError> {
        self.decode_with_options(src, &LoadOptions::default())
    }

    /// Decode all frames of a GIF file.
    ///
    /// Only `U8` is supported — GIF is an 8-bit format. The decoded output is
    /// RGB (3 bands) for opaque images and RGBA (4 bands) when any frame uses
    /// transparency. Animated GIFs are returned as an `Image` whose primary
    /// pixel buffer is the first frame and whose `frames()` sequence contains
    /// every decoded frame.
    ///
    /// GIF LoadOptions are silently ignored per the codec contract.
    fn decode_with_options<F: BandFormat>(
        &self,
        src: &[u8],
        _opts: &LoadOptions,
    ) -> Result<Image<F>, ViprsError>
    where
        Self: Sized,
    {
        if F::ID != BandFormatId::U8 {
            return Err(ViprsError::Codec(format!(
                "gif: unsupported format {:?} — only U8 is supported",
                F::ID
            )));
        }

        let mut options = DecodeOptions::new();
        options.set_color_output(ColorOutput::RGBA);

        let mut dec = options
            .read_info(Cursor::new(src))
            .map_err(|e| ViprsError::Codec(e.to_string()))?;

        let width = dec.width() as u32;
        let height = dec.height() as u32;
        let canvas_len = checked_rgba_len(width, height)?;
        let mut canvas = vec![0u8; canvas_len];
        let mut restore_canvas = vec![0u8; canvas_len];
        let mut rgba_frames = Vec::new();
        let mut has_transparency = false;

        while let Some(frame) = dec
            .read_next_frame()
            .map_err(|e| ViprsError::Codec(e.to_string()))?
        {
            let disposal = frame_disposal_from_gif(frame.dispose);
            if matches!(disposal, FrameDisposal::Previous) {
                restore_canvas.copy_from_slice(&canvas);
            }

            composite_frame_rgba(&mut canvas, width, height, frame)?;
            has_transparency |= frame_has_transparency(frame) || rgba_has_transparency(&canvas);
            rgba_frames.push((
                canvas.clone(),
                delay_ms_from_gif_units(frame.delay),
                disposal,
            ));

            match disposal {
                FrameDisposal::Any | FrameDisposal::Keep => {}
                FrameDisposal::Background => clear_frame_rect_rgba(
                    &mut canvas,
                    width,
                    frame.left,
                    frame.top,
                    frame.width,
                    frame.height,
                    height,
                )?,
                FrameDisposal::Previous => canvas.copy_from_slice(&restore_canvas),
            }
        }

        if rgba_frames.is_empty() {
            return Err(ViprsError::Codec("gif: no frames found".into()));
        }

        let bands: u32 = if has_transparency { 4 } else { 3 };
        let frames = rgba_frames
            .into_iter()
            .map(|(rgba, delay_ms, disposal)| {
                let pixels = if has_transparency {
                    rgba
                } else {
                    rgba_to_rgb(width, height, &rgba)?
                };
                // SAFETY: F::ID == U8 (checked above) implies F::Sample == u8.
                let samples = bytemuck::allocation::try_cast_vec::<u8, F::Sample>(pixels)
                    .map_err(|(e, _)| ViprsError::Codec(format!("gif: cast error: {e:?}")))?;
                Image::from_buffer(width, height, bands, samples)
                    .map(|image| AnimationFrame::new(image, delay_ms, disposal))
                    .map_err(|e| ViprsError::Codec(e.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let metadata = ImageMetadata {
            n_pages: Some(frames.len() as u32),
            page_height: (frames.len() > 1).then_some(height),
            animation_loop_count: Some(animation_loop_from_gif(dec.repeat())),
            ..ImageMetadata::default()
        };

        let image = frames
            .first()
            .map(|frame| frame.image().clone())
            .ok_or_else(|| ViprsError::Codec("gif: no frames found".into()))?
            .with_metadata(metadata)
            .with_animation_frames(frames);
        Ok(image)
    }

    fn probe(&self, src: &[u8]) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized,
    {
        let mut options = DecodeOptions::new();
        options.set_color_output(ColorOutput::RGBA);

        let mut dec = options
            .read_info(Cursor::new(src))
            .map_err(|e| ViprsError::Codec(e.to_string()))?;

        let mut has_transparency = false;
        while let Some(frame) = dec
            .next_frame_info()
            .map_err(|e| ViprsError::Codec(e.to_string()))?
        {
            if frame_has_transparency(frame) {
                has_transparency = true;
                break;
            }
        }

        Ok((
            dec.width() as u32,
            dec.height() as u32,
            if has_transparency { 4 } else { 3 },
        ))
    }
}

// ── ImageEncoder ──────────────────────────────────────────────────────────────

impl ImageEncoder for GifCodec {
    fn format_name(&self) -> &'static str {
        "gif"
    }

    fn encode<F: BandFormat>(&self, image: &Image<F>) -> Result<Vec<u8>, ViprsError> {
        self.encode_with_options(image, &SaveOptions::default())
    }

    /// Encode a static or animated GIF.
    ///
    /// Only `U8` with 3 bands (RGB) or 4 bands (RGBA) is supported. `SaveOptions`
    /// honours `colors` (2–256 palette entries) and `dither` (Floyd-Steinberg
    /// remap on/off). Other fields are ignored per the codec contract.
    fn encode_with_options<F: BandFormat>(
        &self,
        image: &Image<F>,
        opts: &SaveOptions,
    ) -> Result<Vec<u8>, ViprsError>
    where
        Self: Sized,
    {
        if F::ID != BandFormatId::U8 {
            return Err(ViprsError::Codec(format!(
                "gif: unsupported format {:?} — only U8 is supported",
                F::ID
            )));
        }
        let animated_frames = image.animation_frames();
        let plain_frames = image.frames();
        let first_frame = if let Some(frames) = animated_frames {
            frames.first().map(AnimationFrame::image).ok_or_else(|| {
                ViprsError::Codec("gif: animation sequence must contain at least one frame".into())
            })?
        } else if let Some(frames) = plain_frames {
            frames.first().ok_or_else(|| {
                ViprsError::Codec("gif: frame sequence must contain at least one frame".into())
            })?
        } else {
            image
        };

        if first_frame.bands() != 3 && first_frame.bands() != 4 {
            return Err(ViprsError::Codec(format!(
                "gif: unsupported band count {} — only 3-band RGB and 4-band RGBA are supported",
                first_frame.bands()
            )));
        }

        let width = u16::try_from(first_frame.width()).map_err(|_| {
            ViprsError::Codec(format!("gif: width {} exceeds u16", first_frame.width()))
        })?;
        let height = u16::try_from(first_frame.height()).map_err(|_| {
            ViprsError::Codec(format!("gif: height {} exceeds u16", first_frame.height()))
        })?;
        let mut encode_buffer = lock_encode_buffer(&self.encode_buffer);
        let mut output = Vec::new();
        {
            let mut enc = Encoder::new(&mut output, width, height, &[])
                .map_err(|e| ViprsError::Codec(e.to_string()))?;
            enc.set_repeat(animation_loop_to_gif(image.metadata().animation_loop_count))
                .map_err(|e| ViprsError::Codec(e.to_string()))?;
            if let Some(frames) = animated_frames {
                for (frame_index, animation_frame) in frames.iter().enumerate() {
                    let frame_image = animation_frame.image();
                    validate_animation_frame_shape(frame_image, first_frame, frame_index)?;
                    let pixel_bytes: &[u8] = bytemuck::cast_slice(frame_image.pixels());
                    let mut frame = quantize_frame(
                        pixel_bytes,
                        width,
                        height,
                        frame_image.bands() as usize,
                        opts,
                        &mut encode_buffer,
                    )?;
                    frame.delay = delay_ms_to_gif_units(animation_frame.delay_ms());
                    frame.dispose = frame_disposal_to_gif(animation_frame.disposal());
                    enc.write_frame(&frame)
                        .map_err(|e| ViprsError::Codec(e.to_string()))?;
                }
            } else if let Some(frames) = plain_frames {
                for (frame_index, frame_image) in frames.iter().enumerate() {
                    validate_animation_frame_shape(frame_image, first_frame, frame_index)?;
                    let pixel_bytes: &[u8] = bytemuck::cast_slice(frame_image.pixels());
                    let mut frame = quantize_frame(
                        pixel_bytes,
                        width,
                        height,
                        frame_image.bands() as usize,
                        opts,
                        &mut encode_buffer,
                    )?;
                    frame.delay = 0;
                    frame.dispose = GifDisposalMethod::Keep;
                    enc.write_frame(&frame)
                        .map_err(|e| ViprsError::Codec(e.to_string()))?;
                }
            } else {
                let pixel_bytes: &[u8] = bytemuck::cast_slice(first_frame.pixels());
                let mut frame = quantize_frame(
                    pixel_bytes,
                    width,
                    height,
                    first_frame.bands() as usize,
                    opts,
                    &mut encode_buffer,
                )?;
                frame.delay = 0;
                frame.dispose = GifDisposalMethod::Keep;
                enc.write_frame(&frame)
                    .map_err(|e| ViprsError::Codec(e.to_string()))?;
            }
        }

        Ok(output)
    }
}

fn validate_animation_frame_shape<F: BandFormat>(
    frame: &Image<F>,
    reference: &Image<F>,
    frame_index: usize,
) -> Result<(), ViprsError> {
    if frame.width() != reference.width()
        || frame.height() != reference.height()
        || frame.bands() != reference.bands()
    {
        return Err(ViprsError::Codec(format!(
            "gif: frame {frame_index} has shape {}x{}x{}, expected {}x{}x{}",
            frame.width(),
            frame.height(),
            frame.bands(),
            reference.width(),
            reference.height(),
            reference.bands(),
        )));
    }

    if frame.bands() != 3 && frame.bands() != 4 {
        return Err(ViprsError::Codec(format!(
            "gif: frame {frame_index} has unsupported band count {} — only 3-band RGB and 4-band RGBA are supported",
            frame.bands()
        )));
    }

    Ok(())
}

fn lock_encode_buffer(mutex: &Mutex<GifEncodeBuffer>) -> MutexGuard<'_, GifEncodeBuffer> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use gif::DecodingError;
    use proptest::prelude::*;
    use std::{
        collections::BTreeSet,
        fs,
        path::{Path, PathBuf},
        process::Command,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use super::*;
    use crate::domain::format::U8;

    static GIF_PARITY_CASE_COUNTER: AtomicUsize = AtomicUsize::new(0);

    prop_compose! {
        fn rgb_u8_image()
            (width in 1u32..=64u32, height in 1u32..=64u32)
            (
                width in Just(width),
                height in Just(height),
                pixels in proptest::collection::vec(any::<u8>(), (width * height * 3) as usize),
            ) -> (u32, u32, Vec<u8>) {
                (width, height, pixels)
            }
    }

    fn exact_palette_rgb_u8_image() -> impl Strategy<Value = (u32, u32, Vec<u8>)> {
        (
            1u32..=64u32,
            1u32..=64u32,
            proptest::collection::btree_set(any::<[u8; 3]>(), 1..=256),
        )
            .prop_flat_map(|(width, height, palette)| {
                let palette = palette.into_iter().collect::<Vec<_>>();
                let palette_len = palette.len();
                proptest::collection::vec(0usize..palette_len, (width * height) as usize).prop_map(
                    move |indices| {
                        let mut pixels = Vec::with_capacity(indices.len() * 3);
                        for palette_index in indices {
                            pixels.extend_from_slice(&palette[palette_index]);
                        }
                        (width, height, pixels)
                    },
                )
            })
    }

    fn encoded_palette_colour_count(encoded: &[u8]) -> usize {
        let options = DecodeOptions::new();
        let mut decoder = options.read_info(Cursor::new(encoded)).unwrap();
        let _frame = decoder.read_next_frame().unwrap().unwrap();

        decoder
            .palette()
            .unwrap()
            .chunks_exact(3)
            .map(|rgb| [rgb[0], rgb[1], rgb[2]])
            .collect::<BTreeSet<_>>()
            .len()
    }

    fn indexed_frame(encoded: &[u8]) -> Result<Vec<u8>, DecodingError> {
        let options = DecodeOptions::new();
        let mut decoder = options.read_info(Cursor::new(encoded))?;
        let frame = decoder.read_next_frame()?.unwrap();
        Ok(frame.buffer.to_vec())
    }

    fn indexed_transition_count(indices: &[u8]) -> usize {
        indices.windows(2).filter(|pair| pair[0] != pair[1]).count()
    }

    fn vips_available() -> bool {
        Command::new("vips")
            .arg("--version")
            .output()
            .is_ok_and(|output| output.status.success())
    }

    fn gif_parity_runtime_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("gif-parity")
    }

    fn create_gif_parity_case_dir(case: &str) -> PathBuf {
        let sequence = GIF_PARITY_CASE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let case_dir = gif_parity_runtime_dir().join(format!("{case}-{sequence}"));
        fs::create_dir_all(&case_dir).unwrap();
        case_dir
    }

    fn write_rgb_ppm(path: &Path, width: u32, height: u32, pixels: &[u8]) {
        assert_eq!(
            pixels.len(),
            width as usize * height as usize * 3,
            "PPM input must be packed RGB pixels"
        );

        let mut ppm = Vec::with_capacity(pixels.len() + 32);
        ppm.extend_from_slice(format!("P6\n{width} {height}\n255\n").as_bytes());
        ppm.extend_from_slice(pixels);
        fs::write(path, ppm).unwrap();
    }

    fn encode_with_libvips_gif(width: u32, height: u32, pixels: &[u8]) -> Vec<u8> {
        let case_dir = create_gif_parity_case_dir("high-colour-libvips");
        let input_path = case_dir.join("source.ppm");
        let output_path = case_dir.join("expected.gif");
        write_rgb_ppm(&input_path, width, height, pixels);

        let output_spec = format!("{}[dither=1.0,effort=7,bitdepth=8]", output_path.display());
        let output = Command::new("vips")
            .args([
                "copy",
                input_path.to_str().expect("ppm path utf8"),
                output_spec.as_str(),
            ])
            .output()
            .unwrap();

        assert!(
            output.status.success(),
            "vips copy failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        let encoded = fs::read(&output_path).unwrap();
        fs::remove_dir_all(case_dir).unwrap();
        encoded
    }

    fn mean_absolute_error(lhs: &[u8], rhs: &[u8]) -> f32 {
        assert_eq!(
            lhs.len(),
            rhs.len(),
            "MAE requires buffers with the same length"
        );

        let total_error: u64 = lhs
            .iter()
            .zip(rhs.iter())
            .map(|(&left, &right)| (i16::from(left) - i16::from(right)).unsigned_abs() as u64)
            .sum();

        total_error as f32 / lhs.len() as f32
    }

    fn distinct_rgb_count(pixels: &[u8]) -> usize {
        pixels
            .chunks_exact(3)
            .map(|rgb| [rgb[0], rgb[1], rgb[2]])
            .collect::<BTreeSet<_>>()
            .len()
    }

    fn high_colour_gradient_rgb(width: u32, height: u32) -> Vec<u8> {
        let mut pixels = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                pixels.push(((x * 17 + y * 13) % 256) as u8);
                pixels.push(((x * 29 + y * 7 + (x * y) % 31) % 256) as u8);
                pixels.push(((x * 11 + y * 19 + (x * 3 + y * 5)) % 256) as u8);
            }
        }
        pixels
    }

    fn rgba_frame(encoded: &[u8]) -> Vec<u8> {
        let mut options = DecodeOptions::new();
        options.set_color_output(ColorOutput::RGBA);
        let mut decoder = options.read_info(Cursor::new(encoded)).unwrap();
        let frame = decoder.read_next_frame().unwrap().unwrap();
        frame.buffer.to_vec()
    }

    fn animated_gif_bytes() -> Vec<u8> {
        let mut output = Vec::new();
        let mut encoder = Encoder::new(&mut output, 2, 1, &[]).unwrap();
        encoder.set_repeat(Repeat::Infinite).unwrap();

        let mut frame0_rgba = vec![255u8, 0, 0, 255, 255, 0, 0, 255];
        let mut frame0 = Frame::from_rgba_speed(2, 1, &mut frame0_rgba, 1);
        frame0.delay = 4;
        encoder.write_frame(&frame0).unwrap();

        let mut frame1_rgba = vec![0u8, 255, 0, 255, 0, 255, 0, 255];
        let mut frame1 = Frame::from_rgba_speed(2, 1, &mut frame1_rgba, 1);
        frame1.delay = 6;
        encoder.write_frame(&frame1).unwrap();

        drop(encoder);
        output
    }

    fn animated_gif_with_offsets_and_disposal_bytes() -> Vec<u8> {
        let mut output = Vec::new();
        let mut encoder = Encoder::new(&mut output, 3, 2, &[]).unwrap();
        encoder.set_repeat(Repeat::Infinite).unwrap();

        let mut frame0_rgba = vec![
            255u8, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255,
            0, 0, 255,
        ];
        let mut frame0 = Frame::from_rgba_speed(3, 2, &mut frame0_rgba, 1);
        frame0.delay = 3;
        frame0.dispose = gif::DisposalMethod::Keep;
        encoder.write_frame(&frame0).unwrap();

        let mut frame1_rgba = vec![0u8, 255, 0, 255, 0, 0, 0, 0];
        let mut frame1 = Frame::from_rgba_speed(2, 1, &mut frame1_rgba, 1);
        frame1.left = 1;
        frame1.top = 0;
        frame1.delay = 4;
        frame1.dispose = gif::DisposalMethod::Background;
        encoder.write_frame(&frame1).unwrap();

        let mut frame2_rgba = vec![0u8, 0, 255, 255];
        let mut frame2 = Frame::from_rgba_speed(1, 1, &mut frame2_rgba, 1);
        frame2.left = 2;
        frame2.top = 1;
        frame2.delay = 5;
        frame2.dispose = gif::DisposalMethod::Previous;
        encoder.write_frame(&frame2).unwrap();

        let mut frame3_rgba = vec![255u8, 255, 0, 255];
        let mut frame3 = Frame::from_rgba_speed(1, 1, &mut frame3_rgba, 1);
        frame3.left = 0;
        frame3.top = 1;
        frame3.delay = 6;
        frame3.dispose = gif::DisposalMethod::Keep;
        encoder.write_frame(&frame3).unwrap();

        drop(encoder);
        output
    }

    // ── sniff ─────────────────────────────────────────────────────────────────

    #[test]
    fn sniff_recognises_gif89a() {
        let codec = GifCodec::default();
        assert!(
            codec.sniff(b"GIF89a\x00\x00"),
            "must recognise GIF89a magic"
        );
    }

    #[test]
    fn sniff_recognises_gif87a() {
        let codec = GifCodec::default();
        assert!(
            codec.sniff(b"GIF87a\x00\x00"),
            "must recognise GIF87a magic"
        );
    }

    #[test]
    fn sniff_rejects_png() {
        let codec = GifCodec::default();
        let header = [137_u8, 80, 78, 71, 13, 10, 26, 10];
        assert!(!codec.sniff(&header));
    }

    #[test]
    fn sniff_rejects_short_header() {
        let codec = GifCodec::default();
        assert!(!codec.sniff(b"GIF8"));
    }

    #[test]
    fn sniff_empty_returns_false() {
        let codec = GifCodec::default();
        assert!(!codec.sniff(&[]));
    }

    // ── round-trip ────────────────────────────────────────────────────────────

    #[test]
    fn gif_encode_decode_roundtrip() {
        let codec = GifCodec::default();
        let width = 8u32;
        let height = 8u32;
        let mut pixels = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                pixels.push((x * 17) as u8);
                pixels.push((y * 19) as u8);
                pixels.push(((x * 11 + y * 7) % 256) as u8);
            }
        }

        let image = Image::<U8>::from_buffer(width, height, 3, pixels).unwrap();
        let encoded = codec.encode::<U8>(&image).unwrap();
        let decoded = codec.decode::<U8>(&encoded).unwrap();

        assert_eq!(decoded.width(), width);
        assert_eq!(decoded.height(), height);
        assert_eq!(decoded.bands(), 3);
        assert_eq!(decoded.metadata().n_pages, Some(1));
    }

    #[test]
    fn decode_multiframe_gif_exposes_all_frames() {
        let codec = GifCodec::default();
        let decoded = codec.decode::<U8>(&animated_gif_bytes()).unwrap();

        assert_eq!(decoded.width(), 2);
        assert_eq!(decoded.height(), 1);
        assert_eq!(decoded.bands(), 3);
        assert_eq!(decoded.metadata().n_pages, Some(2));
        assert_eq!(decoded.metadata().page_height, Some(1));

        let frames = decoded.frames().expect("animated GIF must expose frames");
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].pixels(), &[255u8, 0, 0, 255, 0, 0]);
        assert_eq!(frames[1].pixels(), &[0u8, 255, 0, 0, 255, 0]);
    }

    #[test]
    fn decode_multiframe_gif_composites_offsets_transparency_and_disposal() {
        let codec = GifCodec::default();
        let decoded = codec
            .decode::<U8>(&animated_gif_with_offsets_and_disposal_bytes())
            .unwrap();

        assert_eq!(decoded.width(), 3);
        assert_eq!(decoded.height(), 2);
        assert_eq!(decoded.bands(), 4);
        assert_eq!(decoded.metadata().n_pages, Some(4));
        assert_eq!(decoded.metadata().page_height, Some(2));

        let frames = decoded.frames().expect("animated GIF must expose frames");
        assert_eq!(frames.len(), 4);

        assert_eq!(
            frames[0].pixels(),
            &[
                255u8, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255,
                255, 0, 0, 255,
            ]
        );
        assert_eq!(
            frames[1].pixels(),
            &[
                255u8, 0, 0, 255, 0, 255, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255,
                255, 0, 0, 255,
            ]
        );
        assert_eq!(
            frames[2].pixels(),
            &[
                255u8, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 255, 0, 0, 255, 0, 0,
                255, 255,
            ]
        );
        assert_eq!(
            frames[3].pixels(),
            &[
                255u8, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 255, 255, 0, 0, 255, 255, 0,
                0, 255,
            ]
        );
    }

    #[test]
    fn encode_multiframe_gif_preserves_delay_disposal_and_loop_count() {
        let codec = GifCodec::default();
        let animated = Image::<U8>::from_frames(vec![
            AnimationFrame::new(
                Image::<U8>::from_buffer(2, 1, 3, vec![255, 0, 0, 255, 0, 0]).unwrap(),
                40,
                FrameDisposal::Keep,
            ),
            AnimationFrame::new(
                Image::<U8>::from_buffer(2, 1, 3, vec![0, 255, 0, 0, 255, 0]).unwrap(),
                70,
                FrameDisposal::Background,
            ),
        ])
        .unwrap()
        .with_animation_loop_count(AnimationLoopCount::Infinite);

        let encoded = codec.encode::<U8>(&animated).unwrap();

        let options = DecodeOptions::new();
        let mut decoder = options.read_info(Cursor::new(&encoded)).unwrap();
        assert_eq!(decoder.repeat(), Repeat::Infinite);

        let frame0 = decoder.read_next_frame().unwrap().unwrap().clone();
        assert_eq!(frame0.delay, 4);
        assert_eq!(frame0.dispose, gif::DisposalMethod::Keep);

        let frame1 = decoder.read_next_frame().unwrap().unwrap().clone();
        assert_eq!(frame1.delay, 7);
        assert_eq!(frame1.dispose, gif::DisposalMethod::Background);

        assert!(
            decoder.read_next_frame().unwrap().is_none(),
            "animation must contain exactly two frames"
        );
    }

    #[test]
    fn encode_rejects_empty_animation_sequence() {
        let image = Image::<U8>::from_buffer(1, 1, 3, vec![12, 34, 56])
            .unwrap()
            .with_animation_frames(vec![]);

        let err = GifCodec::default().encode(&image).unwrap_err();

        assert!(matches!(
            err,
            ViprsError::Codec(message)
                if message == "gif: animation sequence must contain at least one frame"
        ));
    }

    #[test]
    fn gif_chaos_one_pixel_animation_with_many_frames_round_trips_frame_count() {
        let codec = GifCodec::default();
        let frames = (0..512)
            .map(|index| {
                let rgb = if index % 2 == 0 {
                    vec![255, 0, 0]
                } else {
                    vec![0, 0, 255]
                };
                AnimationFrame::new(
                    Image::<U8>::from_buffer(1, 1, 3, rgb).unwrap(),
                    10,
                    FrameDisposal::Keep,
                )
            })
            .collect::<Vec<_>>();
        let image = frames[0]
            .image()
            .clone()
            .with_animation_frames(frames.clone());

        let encoded = codec.encode(&image).unwrap();
        let decoded = codec.decode::<U8>(&encoded).unwrap();

        assert_eq!(decoded.width(), 1);
        assert_eq!(decoded.height(), 1);
        assert_eq!(decoded.animation_frames().unwrap().len(), 512);
    }

    /// GIF uses a 256-colour palette; encode→decode is not pixel-perfect for
    /// arbitrary images. A solid-colour image is used to avoid quantisation
    /// artefacts: all pixels map to the same palette entry, so the round-trip
    /// must preserve the colour exactly (within the 8-bit GIF colour space).
    #[test]
    fn round_trip_solid_colour_rgb_4x4() {
        let codec = GifCodec::default();
        let pixels: Vec<u8> = [200u8, 50, 50].repeat(4 * 4);
        let original = Image::<U8>::from_buffer(4, 4, 3, pixels).unwrap();

        let encoded = codec.encode::<U8>(&original).unwrap();
        assert!(
            codec.sniff(&encoded),
            "encoded output must have GIF magic: {:?}",
            &encoded[..6.min(encoded.len())]
        );

        let decoded = codec.decode::<U8>(&encoded).unwrap();

        assert_eq!(decoded.width(), 4, "width must be preserved");
        assert_eq!(decoded.height(), 4, "height must be preserved");
        assert_eq!(decoded.bands(), 3, "band count must be 3 (RGB output)");

        let orig = original.pixels();
        let dec = decoded.pixels();
        for i in (0..orig.len()).step_by(3) {
            assert_eq!(
                &orig[i..i + 3],
                &dec[i..i + 3],
                "pixel {}: solid colour must survive GIF round-trip",
                i / 3
            );
        }
    }

    // ── dimensions preserved ──────────────────────────────────────────────────

    #[test]
    fn probe_returns_correct_dimensions() {
        let codec = GifCodec::default();
        let pixels: Vec<u8> = [128u8, 64, 32].repeat(6 * 5);
        let image = Image::<U8>::from_buffer(6, 5, 3, pixels).unwrap();
        let encoded = codec.encode::<U8>(&image).unwrap();

        let (w, h, bands) = codec.probe(&encoded).unwrap();
        assert_eq!(w, 6);
        assert_eq!(h, 5);
        assert_eq!(bands, 3);
    }

    #[test]
    fn encode_high_colour_gradient_retains_large_palette() {
        let codec = GifCodec::default();
        let width = 32u32;
        let height = 16u32;
        let mut pixels = Vec::with_capacity((width * height * 3) as usize);

        for y in 0..height {
            for x in 0..width {
                pixels.push((x * 8) as u8);
                pixels.push((y * 16) as u8);
                pixels.push(((x * 13 + y * 7) % 256) as u8);
            }
        }

        let image = Image::<U8>::from_buffer(width, height, 3, pixels).unwrap();
        let encoded = codec.encode::<U8>(&image).unwrap();

        assert!(
            encoded_palette_colour_count(&encoded) >= 128,
            "quantized palette should retain many colours for a high-colour gradient"
        );
    }

    #[test]
    fn encode_high_colour_gradient_stays_close_to_libvips_quantization() {
        if !vips_available() {
            eprintln!("skipping GIF/libvips parity test: `vips` CLI not available");
            return;
        }

        let codec = GifCodec::default();
        let width = 64u32;
        let height = 48u32;
        let pixels = high_colour_gradient_rgb(width, height);
        assert!(
            distinct_rgb_count(&pixels) > 256,
            "parity input must exceed GIF palette capacity"
        );

        let image = Image::<U8>::from_buffer(width, height, 3, pixels.clone()).unwrap();
        let viprs_encoded = codec.encode::<U8>(&image).unwrap();
        let viprs_decoded = codec.decode::<U8>(&viprs_encoded).unwrap();
        let libvips_encoded = encode_with_libvips_gif(width, height, &pixels);
        let libvips_decoded = codec.decode::<U8>(&libvips_encoded).unwrap();

        assert_eq!(
            (viprs_decoded.width(), viprs_decoded.height()),
            (width, height)
        );
        assert_eq!(
            (libvips_decoded.width(), libvips_decoded.height()),
            (width, height)
        );
        assert_eq!(viprs_decoded.bands(), 3);
        assert_eq!(libvips_decoded.bands(), 3);

        let viprs_mae = mean_absolute_error(viprs_decoded.pixels(), &pixels);
        let libvips_mae = mean_absolute_error(libvips_decoded.pixels(), &pixels);
        let parity_mae = mean_absolute_error(viprs_decoded.pixels(), libvips_decoded.pixels());

        assert!(
            viprs_mae <= 20.0,
            "viprs high-colour GIF round-trip MAE exceeded budget: {viprs_mae:.3}"
        );
        assert!(
            parity_mae <= 14.0,
            "viprs/libvips high-colour GIF parity MAE exceeded budget: {parity_mae:.3}"
        );
        assert!(
            viprs_mae <= libvips_mae + 8.0,
            "viprs quantization drifted too far from libvips: viprs_mae={viprs_mae:.3}, libvips_mae={libvips_mae:.3}"
        );
    }

    #[test]
    fn encode_with_color_limit_caps_palette_size() {
        let codec = GifCodec::default();
        let width = 32u32;
        let height = 16u32;
        let mut pixels = Vec::with_capacity((width * height * 3) as usize);

        for y in 0..height {
            for x in 0..width {
                pixels.push((x * 8) as u8);
                pixels.push((y * 16) as u8);
                pixels.push(((x * 13 + y * 7) % 256) as u8);
            }
        }

        let image = Image::<U8>::from_buffer(width, height, 3, pixels).unwrap();
        let encoded = codec
            .encode_with_options::<U8>(&image, &SaveOptions::default().with_colors(16))
            .unwrap();

        assert!(
            encoded_palette_colour_count(&encoded) <= 16,
            "palette must honor SaveOptions::colors"
        );
    }

    #[test]
    fn encode_with_dither_changes_remap_pattern() {
        let codec = GifCodec::default();
        let width = 16u32;
        let mut pixels = Vec::with_capacity((width * 3) as usize);
        for x in 0..width {
            let sample = (x * 16) as u8;
            pixels.extend_from_slice(&[sample, sample, sample]);
        }

        let image = Image::<U8>::from_buffer(width, 1, 3, pixels).unwrap();
        let without_dither = codec
            .encode_with_options::<U8>(
                &image,
                &SaveOptions::default().with_colors(2).with_dither(false),
            )
            .unwrap();
        let with_dither = codec
            .encode_with_options::<U8>(
                &image,
                &SaveOptions::default().with_colors(2).with_dither(true),
            )
            .unwrap();

        let no_dither_indices = indexed_frame(&without_dither).unwrap();
        let dithered_indices = indexed_frame(&with_dither).unwrap();

        assert!(
            indexed_transition_count(&dithered_indices)
                > indexed_transition_count(&no_dither_indices),
            "Floyd-Steinberg remap should create a denser index pattern than direct nearest-colour mapping"
        );
    }

    #[test]
    fn encode_exact_palette_with_dither_matches_no_dither_indices() {
        let codec = GifCodec::default();
        let width = 8u32;
        let height = 8u32;
        let mut pixels = Vec::with_capacity((width * height * 3) as usize);

        for y in 0..height {
            for x in 0..width {
                let rgb = if (x + y) % 2 == 0 {
                    [255u8, 0, 0]
                } else {
                    [0u8, 0, 255]
                };
                pixels.extend_from_slice(&rgb);
            }
        }

        let image = Image::<U8>::from_buffer(width, height, 3, pixels).unwrap();
        let without_dither = codec
            .encode_with_options::<U8>(&image, &SaveOptions::default().with_dither(false))
            .unwrap();
        let with_dither = codec
            .encode_with_options::<U8>(&image, &SaveOptions::default().with_dither(true))
            .unwrap();

        assert_eq!(
            indexed_frame(&with_dither).unwrap(),
            indexed_frame(&without_dither).unwrap(),
            "exact-palette inputs should skip the dithering remap path"
        );
    }

    #[test]
    fn encode_rgba_preserves_transparency_index() {
        let codec = GifCodec::default();
        let pixels: Vec<u8> = vec![255, 0, 0, 255, 0, 255, 0, 0];
        let image = Image::<U8>::from_buffer(2, 1, 4, pixels).unwrap();

        let encoded = codec.encode::<U8>(&image).unwrap();
        let decoded_rgba = rgba_frame(&encoded);

        assert_eq!(&decoded_rgba[..4], &[255, 0, 0, 255]);
        assert_eq!(
            decoded_rgba[7], 0,
            "transparent pixel must decode with alpha 0"
        );
    }

    #[test]
    fn decode_transparent_gif_preserves_alpha_and_probe_reports_rgba() {
        let codec = GifCodec::default();
        let pixels: Vec<u8> = vec![255, 0, 0, 255, 0, 255, 0, 0];
        let image = Image::<U8>::from_buffer(2, 1, 4, pixels.clone()).unwrap();

        let encoded = codec.encode::<U8>(&image).unwrap();
        assert_eq!(codec.probe(&encoded).unwrap(), (2, 1, 4));

        let decoded = codec.decode::<U8>(&encoded).unwrap();
        assert_eq!(decoded.width(), 2);
        assert_eq!(decoded.height(), 1);
        assert_eq!(decoded.bands(), 4);
        assert_eq!(&decoded.pixels()[..4], &pixels[..4]);
        // GIF transparency is a palette index, so the hidden RGB behind a fully
        // transparent pixel is not part of the round-trip contract. Only alpha
        // parity is stable across encoders/decoders.
        assert_eq!(decoded.pixels()[7], 0);
    }

    proptest! {
        #[test]
        fn prop_round_trip_preserves_dimensions_and_band_count(
            (width, height, pixels) in rgb_u8_image(),
        ) {
            let codec = GifCodec::default();
            let original = Image::<U8>::from_buffer(width, height, 3, pixels).unwrap();

            let encoded = codec.encode::<U8>(&original).unwrap();
            let decoded = codec.decode::<U8>(&encoded).unwrap();

            prop_assert_eq!(decoded.width(), width);
            prop_assert_eq!(decoded.height(), height);
            prop_assert_eq!(decoded.bands(), 3);
            prop_assert_eq!(decoded.pixels().len(), original.pixels().len());
        }

        #[test]
        fn prop_round_trip_preserves_exact_palette_pixels(
            (width, height, pixels) in exact_palette_rgb_u8_image(),
        ) {
            let codec = GifCodec::default();
            let original = Image::<U8>::from_buffer(width, height, 3, pixels).unwrap();

            let palette_size = u16::try_from(
                original
                .pixels()
                .chunks_exact(3)
                .collect::<BTreeSet<_>>()
                .len(),
            )
            .unwrap();
            let encoded = codec
                .encode_with_options::<U8>(
                    &original,
                    &SaveOptions::default()
                        .with_colors(palette_size.max(2))
                        .with_dither(false),
                )
                .unwrap();
            let decoded = codec.decode::<U8>(&encoded).unwrap();

            prop_assert_eq!(decoded.width(), width);
            prop_assert_eq!(decoded.height(), height);
            prop_assert_eq!(decoded.bands(), 3);
            prop_assert_eq!(decoded.pixels(), original.pixels());
        }
    }

    // ── error cases ───────────────────────────────────────────────────────────

    #[test]
    fn decode_empty_slice_returns_codec_error() {
        let codec = GifCodec::default();
        let result = codec.decode::<U8>(&[]);
        assert!(
            matches!(result, Err(ViprsError::Codec(_))),
            "empty input must return ViprsError::Codec, got: {result:?}"
        );
    }

    #[test]
    fn encode_unsupported_format_returns_error() {
        use crate::domain::format::U16;
        let codec = GifCodec::default();
        let pixels: Vec<u16> = vec![0u16; 4 * 4 * 3];
        let image = Image::<U16>::from_buffer(4, 4, 3, pixels).unwrap();
        let result = codec.encode::<U16>(&image);
        assert!(
            matches!(result, Err(ViprsError::Codec(_))),
            "U16 must return ViprsError::Codec"
        );
    }

    #[test]
    fn encode_wrong_band_count_returns_error() {
        let codec = GifCodec::default();
        let pixels: Vec<u8> = vec![0u8; 4 * 4 * 2];
        let image = Image::<U8>::from_buffer(4, 4, 2, pixels).unwrap();
        let result = codec.encode::<U8>(&image);
        assert!(
            matches!(result, Err(ViprsError::Codec(_))),
            "2-band image must return ViprsError::Codec"
        );
    }

    #[test]
    fn encode_invalid_color_limit_returns_error() {
        let codec = GifCodec::default();
        let pixels: Vec<u8> = [64u8, 64, 64].repeat(4 * 4);
        let image = Image::<U8>::from_buffer(4, 4, 3, pixels).unwrap();
        let result =
            codec.encode_with_options::<U8>(&image, &SaveOptions::default().with_colors(1));
        assert!(matches!(result, Err(ViprsError::Codec(_))));
    }

    #[test]
    fn decode_unsupported_format_returns_error() {
        use crate::domain::format::U16;
        let codec = GifCodec::default();
        let result = codec.decode::<U16>(b"GIF89a");
        assert!(
            matches!(result, Err(ViprsError::Codec(_))),
            "U16 decode must return ViprsError::Codec"
        );
    }

    // ── format_name ───────────────────────────────────────────────────────────

    #[test]
    fn format_name_is_gif() {
        let codec = GifCodec::default();
        assert_eq!(<GifCodec as ImageDecoder>::format_name(&codec), "gif");
        assert_eq!(<GifCodec as ImageEncoder>::format_name(&codec), "gif");
    }
}

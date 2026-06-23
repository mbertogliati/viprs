use std::fs::File;
use std::io::BufReader;
use std::num::{NonZeroU8, NonZeroUsize};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use bytemuck::Pod;
use mozjpeg::decompress::DecompressStarted;
use mozjpeg::{ALL_MARKERS, ColorSpace, ColorSpaceExt, Decompress};
use viprs::adapters::codecs::{PngCodec, TiffDecoder, WebpCodec};
use viprs::adapters::pipeline::{CompiledPipeline, PipelineArena, PipelineBuilder};
use viprs::adapters::sources::create::GreySource;
use viprs::adapters::sources::decoder_source::DecoderSource;
use viprs::adapters::sources::memory::MemorySource;
use viprs::domain::codec_options::LoadOptions;
use viprs::domain::colorspace::{
    Cmyk, ColorspaceId, Greyscale, Hsv, Lab, Lch, Oklab, Oklch, Rgb16, SRgb, ScRgb, Ucs, Xyz, Yxy,
};
use viprs::domain::draw::DrawOp;
use viprs::domain::error::{BuildError, ViprsError};
use viprs::domain::format::{
    BandFormat, BandFormatId, F32, F64, I16, I32, NumericBand, U8, U16, U32,
};
use viprs::domain::image::{
    DemandHint, Image, ImageMetadata, Interpretation, Region, Tile, TileMut,
};
use viprs::domain::kernel::InterpolationKernel;
use viprs::domain::op::{Op, OperationBridge, PixelLocalOp};
use viprs::domain::ops::arithmetic::{Abs, Add, Ceil, Floor, Multiply, Round, Sign, Subtract};
use viprs::domain::ops::arithmetic::{Matrix, RecombOp};
use viprs::domain::ops::boolean::And;
use viprs::domain::ops::colour::{SRgbLabAdjust, SRgbLabRoundtrip};
use viprs::domain::ops::conversion::bandmean::BandMeanSample;
use viprs::domain::ops::conversion::{
    BandMean, BlendMode, Cast, CompositeOp, CopyOp, ExtendMode, Flip, FlipDirection, GammaOp,
};
use viprs::domain::ops::convolution::gauss_blur::ToF32;
use viprs::domain::ops::convolution::{
    ConvOp, GaussBlurH, GaussBlurV, Prewitt, Sharpen, Sobel, ToF64,
};
use viprs::domain::ops::draw::{DrawCircleOp, DrawLineOp, DrawRectOp};
use viprs::domain::ops::freqfilt::{FwFftOp, InvFftOp};
use viprs::domain::ops::morphology::{Close, Dilate, Erode, Median, Open};
use viprs::domain::ops::relational::Equal;
use viprs::domain::ops::resample::Resize;
use viprs::domain::ops::resample::thumbnail::{Thumbnail, ThumbnailTarget};
use viprs::ports::codec::TileImageDecoder;
use viprs::ports::source::{ImageSource, RandomAccessSource};

use super::helpers::load_bench_image;
use super::types::BenchImage;

const DEFAULT_AFFINE_FORWARD: [f64; 4] = [1.0, 0.2, -0.1, 0.95];
const DEFAULT_SIMILARITY_SCALE: f64 = 0.9;
const DEFAULT_SIMILARITY_ANGLE: f64 = 15.0;
const EMBED_OFFSET_X: u32 = 64;
const EMBED_OFFSET_Y: u32 = 48;
const EMBED_PAD_WIDTH: u32 = 256;
const EMBED_PAD_HEIGHT: u32 = 192;
const EXTRACT_OFFSET_X: u32 = 32;
const EXTRACT_OFFSET_Y: u32 = 24;
const EXTRACT_TRIM_WIDTH: u32 = EXTRACT_OFFSET_X * 2;
const EXTRACT_TRIM_HEIGHT: u32 = EXTRACT_OFFSET_Y * 2;
const DEFAULT_SRGB_COLOURSPACE_ROUTE: &[ColorspaceId] = &[ColorspaceId::Lab, ColorspaceId::SRgb];
const DEFAULT_GREYSCALE_COLOURSPACE_ROUTE: &[ColorspaceId] =
    &[ColorspaceId::SRgb, ColorspaceId::Greyscale];
const DEFAULT_SCRGB_COLOURSPACE_ROUTE: &[ColorspaceId] = &[ColorspaceId::Xyz, ColorspaceId::ScRgb];
const DEFAULT_RGB16_COLOURSPACE_ROUTE: &[ColorspaceId] = &[ColorspaceId::SRgb, ColorspaceId::Rgb16];
const DEFAULT_LAB_COLOURSPACE_ROUTE: &[ColorspaceId] = &[ColorspaceId::SRgb, ColorspaceId::Lab];

fn cached_perceptual_lab_adjust() -> Result<SRgbLabAdjust, BuildError> {
    static OP: OnceLock<SRgbLabAdjust> = OnceLock::new();
    if let Some(op) = OP.get() {
        return Ok(op.clone());
    }

    let op = SRgbLabAdjust::new(1.05, -3.5)?;
    let _ = OP.set(op.clone());
    Ok(op)
}

type MozJpegReader = BufReader<File>;

fn with_mozjpeg<T>(f: impl FnOnce() -> std::io::Result<T>) -> Result<T, ViprsError> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(result) => result.map_err(|err| ViprsError::Codec(format!("jpeg scanline: {err}"))),
        Err(_) => Err(ViprsError::Codec(
            "jpeg scanline: decoder panicked while processing libjpeg error".into(),
        )),
    }
}

fn normalize_jpeg_native_shrink_factor(requested: u8) -> u8 {
    match requested {
        2 => 2,
        4 => 4,
        8 | 16 => 8,
        _ => 1,
    }
}

fn split_jpeg_thumbnail_shrink_factor(requested: u8) -> (u8, u8) {
    let native = normalize_jpeg_native_shrink_factor(requested);
    if native <= 1 || requested <= native || requested % native != 0 {
        (native, 1)
    } else {
        (native, requested / native)
    }
}

fn strict_shrunk_dimension(dimension: u32, shrink_factor: u8) -> u32 {
    if shrink_factor <= 1 {
        dimension
    } else {
        (dimension / u32::from(shrink_factor)).max(1)
    }
}

fn mozjpeg_scale_numerator(shrink_factor: u8) -> u8 {
    match shrink_factor {
        2 => 4,
        4 => 2,
        8 => 1,
        _ => 8,
    }
}

enum ActiveJpegDecoder {
    Gray(DecompressStarted<MozJpegReader>),
    Rgb(DecompressStarted<MozJpegReader>),
    Cmyk(DecompressStarted<MozJpegReader>),
}

// SAFETY: the decoder never leaves `JpegSequentialScanlineSource`'s mutex-guarded state.
// The benchmark source opts into sequential access, so scanline reads are serialized.
unsafe impl Send for ActiveJpegDecoder {}

impl ActiveJpegDecoder {
    fn width(&self) -> usize {
        match self {
            Self::Gray(decoder) => decoder.width(),
            Self::Rgb(decoder) => decoder.width(),
            Self::Cmyk(decoder) => decoder.width(),
        }
    }

    fn read_scanlines_into(&mut self, output: &mut [u8]) -> Result<(), ViprsError> {
        with_mozjpeg(|| match self {
            Self::Gray(decoder) => decoder.read_scanlines_into(output).map(|_| ()),
            Self::Rgb(decoder) => decoder.read_scanlines_into(output).map(|_| ()),
            Self::Cmyk(decoder) => decoder.read_scanlines_into(output).map(|_| ()),
        })
    }
}

fn box_shrink_row_pair_factor2(top: &[u8], bottom: &[u8], bands: usize, output: &mut [u8]) {
    let width = output.len() / bands;
    debug_assert_eq!(top.len(), width * bands * 2);
    debug_assert_eq!(bottom.len(), width * bands * 2);
    debug_assert_eq!(output.len(), width * bands);

    for x in 0..width {
        let src = x * bands * 2;
        let dst = x * bands;
        for band in 0..bands {
            let sum = u16::from(top[src + band])
                + u16::from(top[src + bands + band])
                + u16::from(bottom[src + band])
                + u16::from(bottom[src + bands + band]);
            output[dst + band] = ((sum + 2) >> 2) as u8;
        }
    }
}

struct JpegSequentialState {
    width: u32,
    height: u32,
    native_shrink_factor: u8,
    residual_shrink_factor: u8,
    cached_rows_count: u32,
    decoder: Option<ActiveJpegDecoder>,
    decode_row: Vec<u8>,
    pending_row: Vec<u8>,
    cached_rows: Vec<u8>,
}

struct JpegSequentialScanlineSource {
    path: Arc<std::path::PathBuf>,
    source_width: u32,
    source_height: u32,
    bands: u32,
    metadata: ImageMetadata,
    state: Mutex<JpegSequentialState>,
}

impl JpegSequentialScanlineSource {
    fn open(path: &Path) -> Result<Self, ViprsError> {
        let decoder = Decompress::with_markers(ALL_MARKERS)
            .from_path(path)
            .map_err(|err| ViprsError::Codec(format!("jpeg scanline probe: {err}")))?;
        let source_width = decoder.width() as u32;
        let source_height = decoder.height() as u32;
        let bands = decoder.color_space().num_components() as u32;
        let interpretation = match bands {
            1 => Some(Interpretation::BW),
            4 => Some(Interpretation::Cmyk),
            _ => Some(Interpretation::Srgb),
        };

        Ok(Self {
            path: Arc::new(path.to_path_buf()),
            source_width,
            source_height,
            bands,
            metadata: ImageMetadata {
                interpretation,
                ..ImageMetadata::default()
            },
            state: Mutex::new(JpegSequentialState {
                width: source_width,
                height: source_height,
                native_shrink_factor: 1,
                residual_shrink_factor: 1,
                cached_rows_count: 0,
                decoder: None,
                decode_row: Vec::new(),
                pending_row: Vec::new(),
                cached_rows: Vec::new(),
            }),
        })
    }

    fn start_decoder(&self, shrink_factor: u8) -> Result<ActiveJpegDecoder, ViprsError> {
        with_mozjpeg(|| {
            let mut decoder =
                Decompress::with_markers(ALL_MARKERS).from_path(self.path.as_ref())?;
            decoder.scale(mozjpeg_scale_numerator(shrink_factor));
            decoder.do_fancy_upsampling(false);
            match self.bands {
                1 => decoder.grayscale().map(ActiveJpegDecoder::Gray),
                4 => decoder
                    .to_colorspace(ColorSpace::JCS_CMYK)
                    .map(ActiveJpegDecoder::Cmyk),
                _ => decoder.rgb().map(ActiveJpegDecoder::Rgb),
            }
        })
    }
}

impl ImageSource for JpegSequentialScanlineSource {
    type Format = U8;

    fn width(&self) -> u32 {
        self.state
            .lock()
            .map_or(self.source_width, |state| state.width)
    }

    fn height(&self) -> u32 {
        self.state
            .lock()
            .map_or(self.source_height, |state| state.height)
    }

    fn bands(&self) -> u32 {
        self.bands
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::FatStrip
    }

    fn metadata(&self) -> ImageMetadata {
        self.metadata.clone()
    }

    fn set_thumbnail_shrink_on_load(&mut self, factor: NonZeroU8) -> Result<bool, ViprsError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| ViprsError::Codec("jpeg scanline: state mutex poisoned".into()))?;
        if state.decoder.is_some() {
            return Ok(false);
        }

        let requested_factor = factor.get();
        let (native_factor, residual_factor) = split_jpeg_thumbnail_shrink_factor(requested_factor);
        if native_factor <= 1 {
            return Ok(false);
        }

        state.native_shrink_factor = native_factor;
        state.residual_shrink_factor = residual_factor;
        state.width = strict_shrunk_dimension(self.source_width, requested_factor);
        state.height = strict_shrunk_dimension(self.source_height, requested_factor);
        state.cached_rows_count = 0;
        state.decode_row.clear();
        state.pending_row.clear();
        state.cached_rows.clear();
        Ok(true)
    }

    fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| ViprsError::Codec("jpeg scanline: state mutex poisoned".into()))?;
        let expected_len = region.pixel_count() * self.bands as usize;
        if output.len() != expected_len {
            return Err(ViprsError::Codec(format!(
                "jpeg scanline: output buffer size mismatch (got {}, expected {expected_len})",
                output.len()
            )));
        }
        if state.decoder.is_none() {
            state.decoder = Some(self.start_decoder(state.native_shrink_factor)?);
        }

        let decoder_width = state
            .decoder
            .as_ref()
            .map_or(state.width as usize, ActiveJpegDecoder::width);
        let target_row_bytes = state.width as usize * self.bands as usize;
        let decoder_row_bytes = decoder_width * self.bands as usize;
        if state.decode_row.len() != decoder_row_bytes {
            state.decode_row.resize(decoder_row_bytes, 0);
        }
        if state.residual_shrink_factor > 1 && state.pending_row.len() != decoder_row_bytes {
            state.pending_row.resize(decoder_row_bytes, 0);
        }
        let cached_row_bytes = target_row_bytes;
        let mut decode_row = std::mem::take(&mut state.decode_row);
        let mut pending_row = std::mem::take(&mut state.pending_row);
        for row_index in 0..region.height as usize {
            let source_y = (region.y + row_index as i32).clamp(0, state.height as i32 - 1) as u32;
            while state.cached_rows_count <= source_y {
                let JpegSequentialState {
                    residual_shrink_factor,
                    cached_rows,
                    decoder,
                    cached_rows_count,
                    ..
                } = &mut *state;
                let decoder = decoder
                    .as_mut()
                    .expect("decoder should be initialized before reading rows");
                let row_start = cached_rows.len();
                cached_rows.resize(row_start + cached_row_bytes, 0);
                let cached_row = &mut cached_rows[row_start..row_start + cached_row_bytes];

                if *residual_shrink_factor > 1 {
                    decoder.read_scanlines_into(&mut pending_row)?;
                    decoder.read_scanlines_into(&mut decode_row)?;
                    box_shrink_row_pair_factor2(
                        &pending_row,
                        &decode_row,
                        self.bands as usize,
                        cached_row,
                    );
                } else {
                    decoder.read_scanlines_into(cached_row)?;
                }
                *cached_rows_count += 1;
            }

            let src_offset = source_y as usize * cached_row_bytes;
            let src_row = &state.cached_rows[src_offset..src_offset + cached_row_bytes];
            let dst_row = &mut output[row_index * region.width as usize * self.bands as usize
                ..(row_index + 1) * region.width as usize * self.bands as usize];

            let pixel_bytes = self.bands as usize;
            let width_pixels = region.width as usize;
            let left_pad = 0i32.saturating_sub(region.x) as usize;
            let right_excess =
                (region.x + region.width as i32 - state.width as i32).max(0) as usize;
            let center_pixels = width_pixels
                .saturating_sub(left_pad.min(width_pixels))
                .saturating_sub(right_excess.min(width_pixels));
            let src_x0 = region.x.max(0) as usize;

            if left_pad > 0 {
                let left_pixel = &src_row[..pixel_bytes];
                for pixel in 0..left_pad.min(width_pixels) {
                    let dst = pixel * pixel_bytes;
                    dst_row[dst..dst + pixel_bytes].copy_from_slice(left_pixel);
                }
            }

            if center_pixels > 0 {
                let src = src_x0 * pixel_bytes;
                let dst = left_pad * pixel_bytes;
                let len = center_pixels * pixel_bytes;
                dst_row[dst..dst + len].copy_from_slice(&src_row[src..src + len]);
            }

            if right_excess > 0 {
                let right_pixel_start = (state.width as usize - 1) * pixel_bytes;
                let right_pixel = &src_row[right_pixel_start..right_pixel_start + pixel_bytes];
                let start = left_pad + center_pixels;
                for pixel in 0..right_excess.min(width_pixels.saturating_sub(start)) {
                    let dst = (start + pixel) * pixel_bytes;
                    dst_row[dst..dst + pixel_bytes].copy_from_slice(right_pixel);
                }
            }
        }
        state.decode_row = decode_row;
        state.pending_row = pending_row;

        Ok(())
    }
}
const DEFAULT_XYZ_COLOURSPACE_ROUTE: &[ColorspaceId] = &[ColorspaceId::Lab, ColorspaceId::Xyz];
const DEFAULT_CMYK_COLOURSPACE_ROUTE: &[ColorspaceId] = &[ColorspaceId::SRgb, ColorspaceId::Cmyk];
const DEFAULT_HSV_COLOURSPACE_ROUTE: &[ColorspaceId] = &[ColorspaceId::SRgb, ColorspaceId::Hsv];
const DEFAULT_LCH_COLOURSPACE_ROUTE: &[ColorspaceId] = &[ColorspaceId::Ucs, ColorspaceId::Lch];
const DEFAULT_UCS_COLOURSPACE_ROUTE: &[ColorspaceId] = &[ColorspaceId::Lch, ColorspaceId::Ucs];
const DEFAULT_OKLAB_COLOURSPACE_ROUTE: &[ColorspaceId] =
    &[ColorspaceId::Oklch, ColorspaceId::Oklab];
const DEFAULT_OKLCH_COLOURSPACE_ROUTE: &[ColorspaceId] =
    &[ColorspaceId::Oklab, ColorspaceId::Oklch];

fn default_viprs_cache_bytes() -> NonZeroUsize {
    NonZeroUsize::new(super::helpers::DEFAULT_VIPRS_CACHE_BYTES)
        .expect("default cache byte budget is non-zero")
}

fn interpretation_to_bench_colorspace(interpretation: Interpretation) -> Option<ColorspaceId> {
    match interpretation {
        Interpretation::BW | Interpretation::Grey16 => Some(ColorspaceId::Greyscale),
        Interpretation::Xyz => Some(ColorspaceId::Xyz),
        Interpretation::Lab => Some(ColorspaceId::Lab),
        Interpretation::Cmyk => Some(ColorspaceId::Cmyk),
        Interpretation::Cmc => Some(ColorspaceId::Ucs),
        Interpretation::Lch => Some(ColorspaceId::Lch),
        Interpretation::Srgb | Interpretation::Rgb => Some(ColorspaceId::SRgb),
        Interpretation::Yxy => Some(ColorspaceId::Yxy),
        Interpretation::Rgb16 => Some(ColorspaceId::Rgb16),
        Interpretation::Scrgb => Some(ColorspaceId::ScRgb),
        Interpretation::Hsv => Some(ColorspaceId::Hsv),
        _ => None,
    }
}

fn parse_colourspace_destination(arg: &str) -> Option<ColorspaceId> {
    match arg.to_ascii_lowercase().as_str() {
        "srgb" | "rgb" => Some(ColorspaceId::SRgb),
        "rgb16" => Some(ColorspaceId::Rgb16),
        "lab" => Some(ColorspaceId::Lab),
        "xyz" => Some(ColorspaceId::Xyz),
        "yxy" => Some(ColorspaceId::Yxy),
        "hsv" => Some(ColorspaceId::Hsv),
        "cmyk" => Some(ColorspaceId::Cmyk),
        "scrgb" => Some(ColorspaceId::ScRgb),
        "greyscale" | "grayscale" | "grey" | "gray" | "bw" | "b-w" => Some(ColorspaceId::Greyscale),
        "lch" => Some(ColorspaceId::Lch),
        "ucs" | "cmc" => Some(ColorspaceId::Ucs),
        "oklab" => Some(ColorspaceId::Oklab),
        "oklch" => Some(ColorspaceId::Oklch),
        _ => None,
    }
}

fn default_colourspace_route(source_colorspace: ColorspaceId) -> &'static [ColorspaceId] {
    match source_colorspace {
        ColorspaceId::Greyscale => DEFAULT_GREYSCALE_COLOURSPACE_ROUTE,
        ColorspaceId::ScRgb => DEFAULT_SCRGB_COLOURSPACE_ROUTE,
        ColorspaceId::Rgb16 => DEFAULT_RGB16_COLOURSPACE_ROUTE,
        ColorspaceId::Lab => DEFAULT_LAB_COLOURSPACE_ROUTE,
        ColorspaceId::Xyz => DEFAULT_XYZ_COLOURSPACE_ROUTE,
        ColorspaceId::Cmyk => DEFAULT_CMYK_COLOURSPACE_ROUTE,
        ColorspaceId::Hsv => DEFAULT_HSV_COLOURSPACE_ROUTE,
        ColorspaceId::Lch => DEFAULT_LCH_COLOURSPACE_ROUTE,
        ColorspaceId::Ucs => DEFAULT_UCS_COLOURSPACE_ROUTE,
        ColorspaceId::Oklab => DEFAULT_OKLAB_COLOURSPACE_ROUTE,
        ColorspaceId::Oklch => DEFAULT_OKLCH_COLOURSPACE_ROUTE,
        _ => DEFAULT_SRGB_COLOURSPACE_ROUTE,
    }
}

fn colourspace_route_from_args(
    source_colorspace: ColorspaceId,
    op_args: &[String],
) -> Vec<ColorspaceId> {
    if op_args.is_empty() {
        return default_colourspace_route(source_colorspace).to_vec();
    }

    op_args
        .iter()
        .map(|arg| {
            parse_colourspace_destination(arg).unwrap_or_else(|| {
                eprintln!(
                    "colourspace only accepts destinations from {{srgb, lab, xyz, yxy, hsv, cmyk, scrgb, greyscale, lch, ucs, oklab, oklch}}, got '{arg}'"
                );
                std::process::exit(1);
            })
        })
        .collect()
}

fn apply_colourspace_destination(
    builder: PipelineBuilder,
    destination: ColorspaceId,
) -> Result<PipelineBuilder, BuildError> {
    match destination {
        ColorspaceId::SRgb => builder.colourspace::<SRgb>(),
        ColorspaceId::Lab => builder.colourspace::<Lab>(),
        ColorspaceId::Xyz => builder.colourspace::<Xyz>(),
        ColorspaceId::Yxy => builder.colourspace::<Yxy>(),
        ColorspaceId::Hsv => builder.colourspace::<Hsv>(),
        ColorspaceId::Lch => builder.colourspace::<Lch>(),
        ColorspaceId::Ucs => builder.colourspace::<Ucs>(),
        ColorspaceId::Oklab => builder.colourspace::<Oklab>(),
        ColorspaceId::Oklch => builder.colourspace::<Oklch>(),
        ColorspaceId::Cmyk => builder.colourspace::<Cmyk>(),
        ColorspaceId::Greyscale => builder.colourspace::<Greyscale>(),
        ColorspaceId::ScRgb => builder.colourspace::<ScRgb>(),
        ColorspaceId::Rgb16 => builder.colourspace::<Rgb16>(),
        unsupported => {
            eprintln!("Unsupported benchmark colourspace destination: {unsupported:?}");
            std::process::exit(1);
        }
    }
}

fn parse_morphology_kernel_size(op_args: &[String]) -> u32 {
    op_args.first().and_then(|s| s.parse().ok()).unwrap_or(3)
}

fn parse_median_kernel_size(op_args: &[String]) -> u32 {
    op_args.first().and_then(|s| s.parse().ok()).unwrap_or(3)
}

fn parse_cast_target(current_format: BandFormatId, op_args: &[String]) -> BandFormatId {
    match op_args.first().map(|arg| arg.to_ascii_lowercase()) {
        Some(target) => match target.as_str() {
            "u8" => BandFormatId::U8,
            "u16" => BandFormatId::U16,
            "f32" => BandFormatId::F32,
            other => {
                eprintln!(
                    "cast only accepts optional target arg 'u8', 'u16', or 'f32', got '{other}'"
                );
                std::process::exit(1);
            }
        },
        None if current_format == BandFormatId::U8 => BandFormatId::F32,
        None => BandFormatId::U8,
    }
}

fn parse_flip_direction(op_args: &[String]) -> FlipDirection {
    match op_args.first().map(String::as_str).unwrap_or("horizontal") {
        "horizontal" | "h" => FlipDirection::Horizontal,
        "vertical" | "v" => FlipDirection::Vertical,
        other => {
            eprintln!(
                "flip only accepts optional direction arg 'horizontal' or 'vertical', got '{other}'"
            );
            std::process::exit(1);
        }
    }
}

fn parse_gamma_exponent(op_args: &[String]) -> f64 {
    op_args
        .first()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2.4)
}

fn box_3x3_kernel() -> Vec<Vec<f64>> {
    let weight = 1.0 / 9.0;
    vec![
        vec![weight, weight, weight],
        vec![weight, weight, weight],
        vec![weight, weight, weight],
    ]
}

fn sharpen_3x3_kernel() -> Vec<Vec<f64>> {
    vec![
        vec![0.0, -1.0, 0.0],
        vec![-1.0, 5.0, -1.0],
        vec![0.0, -1.0, 0.0],
    ]
}

fn sobel_x_3x3_kernel() -> Vec<Vec<f64>> {
    vec![
        vec![-1.0, 0.0, 1.0],
        vec![-2.0, 0.0, 2.0],
        vec![-1.0, 0.0, 1.0],
    ]
}

fn laplacian_3x3_kernel() -> Vec<Vec<f64>> {
    vec![
        vec![0.0, -1.0, 0.0],
        vec![-1.0, 4.0, -1.0],
        vec![0.0, -1.0, 0.0],
    ]
}

fn apply_thumbnail(builder: PipelineBuilder, width: u32) -> Result<PipelineBuilder, BuildError> {
    let target = ThumbnailTarget::Width(width);
    let thumbnail = Thumbnail::new(target, InterpolationKernel::Lanczos3);
    builder.thumbnail(thumbnail)
}

fn apply_resize(builder: PipelineBuilder, scale: f64) -> Result<PipelineBuilder, BuildError> {
    let resize = Resize::new(scale, scale, InterpolationKernel::Lanczos3);
    builder.resize(resize)
}

fn apply_sharpen(
    builder: PipelineBuilder,
    sigma: f32,
    strength: f32,
) -> Result<PipelineBuilder, BuildError> {
    builder.sharpen(sigma, 2.0, 10.0, 20.0, 0.0, strength)
}

fn recomb_matrix() -> Matrix {
    #[rustfmt::skip]
    let values = vec![
        0.299, 0.587, 0.114,
        1.000, 0.000, -1.000,
    ];
    Matrix::new(2, 3, values)
}

fn build_viprs_grey_pipeline(width: u32, height: u32) -> CompiledPipeline {
    PipelineBuilder::from_source(GreySource::<F32>::new(width, height))
        .build()
        .expect("grey pipeline build failed")
}

fn apply_gauss_blur<F>(
    builder: PipelineBuilder,
    bands: u32,
    sigma: f32,
) -> Result<PipelineBuilder, BuildError>
where
    F: BandFormat + 'static,
    F::Sample: Pod + ToF32,
    GaussBlurH<F>: Op,
    GaussBlurV<F32>: Op,
{
    let builder = builder.then(Box::new(OperationBridge::new(
        GaussBlurH::<F>::new(sigma),
        bands,
    )))?;

    match F::ID {
        BandFormatId::U8 => builder.then(Box::new(OperationBridge::new(
            GaussBlurV::<U8>::new(sigma),
            bands,
        ))),
        BandFormatId::U16
        | BandFormatId::I16
        | BandFormatId::U32
        | BandFormatId::I32
        | BandFormatId::F32
        | BandFormatId::F64 => builder.then(Box::new(OperationBridge::new(
            GaussBlurV::<F32>::new(sigma),
            bands,
        ))),
    }
}

fn build_viprs_u8_morphology_pipeline_from_source<S>(
    source: S,
    op: &str,
    op_args: &[String],
) -> CompiledPipeline
where
    S: ImageSource<Format = U8> + 'static,
{
    let bands = source.bands();
    let kernel_size = parse_morphology_kernel_size(op_args);
    let builder = PipelineBuilder::from_source(source);

    let builder = match op {
        "dilate" => builder
            .then(Box::new(OperationBridge::new(
                Dilate::rect(kernel_size).expect("dilate kernel"),
                bands,
            )))
            .expect("dilate failed"),
        "erode" => builder
            .then(Box::new(OperationBridge::new(
                Erode::rect(kernel_size).expect("erode kernel"),
                bands,
            )))
            .expect("erode failed"),
        "open" => builder
            .then(Box::new(OperationBridge::new(
                Open::rect(kernel_size).expect("open kernel"),
                bands,
            )))
            .expect("open failed"),
        "close" => builder
            .then(Box::new(OperationBridge::new(
                Close::rect(kernel_size).expect("close kernel"),
                bands,
            )))
            .expect("close failed"),
        other => {
            eprintln!("Unsupported U8 morphology operation: {other}");
            std::process::exit(1);
        }
    };

    let builder = builder
        .cache_last_op(default_viprs_cache_bytes())
        .expect("cache_last_op failed");

    builder.build().expect("pipeline build failed")
}

struct DrawLinePipelineOp<F: BandFormat> {
    inner: DrawLineOp<F>,
}

impl<F: BandFormat> DrawLinePipelineOp<F> {
    fn new(width: u32, height: u32, ink: Vec<F::Sample>) -> Self {
        Self {
            inner: DrawLineOp::new(
                0,
                (height / 2) as i32,
                width.saturating_sub(1) as i32,
                (height / 2) as i32,
                ink,
            )
            .expect("draw_line op"),
        }
    }
}

impl<F: BandFormat> Op for DrawLinePipelineOp<F> {
    type Input = F;
    type Output = F;
    type State = ();

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        output.data.copy_from_slice(input.data);
        self.inner.draw(output);
    }
}

impl<F: BandFormat> PixelLocalOp for DrawLinePipelineOp<F> {}

struct DrawRectPipelineOp<F: BandFormat> {
    inner: DrawRectOp<F>,
}

impl<F: BandFormat> DrawRectPipelineOp<F> {
    fn new(width: u32, height: u32, ink: Vec<F::Sample>) -> Self {
        let rect_width = (width / 2).max(1);
        let rect_height = (height / 2).max(1);
        let left = (width.saturating_sub(rect_width) / 2) as i32;
        let top = (height.saturating_sub(rect_height) / 2) as i32;

        Self {
            inner: DrawRectOp::new(left, top, rect_width, rect_height, ink, true)
                .expect("draw_rect op"),
        }
    }
}

impl<F: BandFormat> Op for DrawRectPipelineOp<F> {
    type Input = F;
    type Output = F;
    type State = ();

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        output.data.copy_from_slice(input.data);
        self.inner.draw(output);
    }
}

impl<F: BandFormat> PixelLocalOp for DrawRectPipelineOp<F> {}

struct DrawCirclePipelineOp<F: BandFormat> {
    inner: DrawCircleOp<F>,
}

impl<F: BandFormat> DrawCirclePipelineOp<F> {
    fn new(width: u32, height: u32, ink: Vec<F::Sample>) -> Self {
        Self {
            inner: DrawCircleOp::new(
                (width / 2) as i32,
                (height / 2) as i32,
                (width.min(height) / 4).max(1),
                ink,
                false,
            )
            .expect("draw_circle op"),
        }
    }
}

impl<F: BandFormat> Op for DrawCirclePipelineOp<F> {
    type Input = F;
    type Output = F;
    type State = ();

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        output.data.copy_from_slice(input.data);
        self.inner.draw(output);
    }
}

impl<F: BandFormat> PixelLocalOp for DrawCirclePipelineOp<F> {}

fn parse_affine_forward_matrix(op_args: &[String]) -> [f64; 4] {
    let mut matrix = DEFAULT_AFFINE_FORWARD;
    for (idx, slot) in matrix.iter_mut().enumerate() {
        if let Some(value) = op_args.get(idx).and_then(|value| value.parse::<f64>().ok()) {
            *slot = value;
        }
    }
    matrix
}

fn invert_affine_forward_matrix(matrix: [f64; 4]) -> [f64; 4] {
    let det = matrix[0] * matrix[3] - matrix[1] * matrix[2];
    if !det.is_finite() || det.abs() < f64::EPSILON {
        eprintln!(
            "affine requires an invertible forward matrix, got [{}, {}, {}, {}]",
            matrix[0], matrix[1], matrix[2], matrix[3]
        );
        std::process::exit(1);
    }

    [
        matrix[3] / det,
        -matrix[1] / det,
        -matrix[2] / det,
        matrix[0] / det,
    ]
}

pub fn image_into_memory_source<F: BandFormat>(image: Image<F>) -> MemorySource<F> {
    let width = image.width();
    let height = image.height();
    let bands = image.bands();
    let metadata = image.metadata().clone();
    MemorySource::<F>::new(width, height, bands, image.into_buffer())
        .expect("source")
        .with_metadata(metadata)
}

pub struct SharedMemorySource<F: BandFormat> {
    width: u32,
    height: u32,
    bands: u32,
    // Arc<Vec<T>> wraps zero-copy: Arc::new(vec) does not copy the pixel data.
    // Arc<[T]>::from(vec) would allocate a new backing store and copy all pixels.
    data: Arc<Vec<F::Sample>>,
    metadata: ImageMetadata,
}

impl<F: BandFormat> SharedMemorySource<F> {
    fn from_image(image: Image<F>) -> Self {
        let width = image.width();
        let height = image.height();
        let bands = image.bands();
        let metadata = image.metadata().clone();
        Self {
            width,
            height,
            bands,
            // Zero-copy: Arc::new wraps the Vec without copying pixel data.
            // Arc<[T]>::from(Vec<T>) allocates a fresh backing store and copies.
            data: Arc::new(image.into_buffer()),
            metadata,
        }
    }
}

impl<F: BandFormat> Clone for SharedMemorySource<F> {
    fn clone(&self) -> Self {
        Self {
            width: self.width,
            height: self.height,
            bands: self.bands,
            data: Arc::clone(&self.data),
            metadata: self.metadata.clone(),
        }
    }
}

impl<F> ImageSource for SharedMemorySource<F>
where
    F: BandFormat,
    F::Sample: Pod,
{
    type Format = F;

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn bands(&self) -> u32 {
        self.bands
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn metadata(&self) -> ImageMetadata {
        self.metadata.clone()
    }

    fn borrow_region(&self, region: Region) -> Option<&[u8]> {
        let in_bounds = region.x >= 0
            && region.y >= 0
            && region.x + region.width as i32 <= self.width as i32
            && region.y + region.height as i32 <= self.height as i32;
        let tightly_packed = region.x == 0;

        if !in_bounds || !tightly_packed {
            return None;
        }

        let row_width = self.width as usize * self.bands as usize;
        let start = region.y as usize * row_width;
        let len = region.height as usize * row_width;
        Some(bytemuck::cast_slice(&self.data[start..start + len]))
    }

    fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        let bytes_per_sample = std::mem::size_of::<F::Sample>();
        let bands = self.bands as usize;
        let row_samples = region.width as usize * bands;
        let src_stride = self.width as usize * bands;

        if self.width == 0 || self.height == 0 {
            output.fill(0);
            return Ok(());
        }

        let in_bounds = region.x >= 0
            && region.y >= 0
            && region.x + region.width as i32 <= self.width as i32
            && region.y + region.height as i32 <= self.height as i32;

        if in_bounds {
            if region.x == 0 && region.width == self.width {
                let src_start = region.y as usize * src_stride;
                let src_end = src_start + region.height as usize * src_stride;
                let src_bytes: &[u8] = bytemuck::cast_slice(&self.data[src_start..src_end]);
                output.copy_from_slice(src_bytes);
                return Ok(());
            }

            for row in 0..region.height as usize {
                let src_row_start =
                    (region.y as usize + row) * src_stride + region.x as usize * bands;
                let src_row = &self.data[src_row_start..src_row_start + row_samples];
                let src_bytes: &[u8] = bytemuck::cast_slice(src_row);
                let dst_byte_start = row * row_samples * bytes_per_sample;
                output[dst_byte_start..dst_byte_start + row_samples * bytes_per_sample]
                    .copy_from_slice(src_bytes);
            }
            return Ok(());
        }

        let pixel_bytes = bands * bytes_per_sample;
        let row_bytes = self.width as usize * pixel_bytes;
        let image_bytes: &[u8] = bytemuck::cast_slice(&self.data);
        let dst_row_bytes = region.width as usize * pixel_bytes;

        for row in 0..region.height as i32 {
            let dst_row_start = row as usize * dst_row_bytes;
            let dst_row = &mut output[dst_row_start..dst_row_start + dst_row_bytes];

            let src_y = (region.y + row).clamp(0, self.height as i32 - 1) as usize;
            let src_row_start = src_y * row_bytes;
            let src_row = &image_bytes[src_row_start..src_row_start + row_bytes];

            let src_x0 = region.x.clamp(0, self.width as i32) as usize;
            let src_x1 = (region.x + region.width as i32).clamp(0, self.width as i32) as usize;
            let center_pixels = src_x1.saturating_sub(src_x0);
            let left_pad = if region.x < 0 {
                (-region.x) as usize
            } else {
                0
            }
            .min(region.width as usize);
            let right_pad = region.width as usize - left_pad - center_pixels;

            let left_pixel = &src_row[..pixel_bytes];
            for pixel in 0..left_pad {
                let dst = pixel * pixel_bytes;
                dst_row[dst..dst + pixel_bytes].copy_from_slice(left_pixel);
            }

            if center_pixels > 0 {
                let src = src_x0 * pixel_bytes;
                let dst = left_pad * pixel_bytes;
                let len = center_pixels * pixel_bytes;
                dst_row[dst..dst + len].copy_from_slice(&src_row[src..src + len]);
            }

            let right_pixel_start = (self.width as usize - 1) * pixel_bytes;
            let right_pixel = &src_row[right_pixel_start..right_pixel_start + pixel_bytes];
            for pixel in 0..right_pad {
                let dst = (left_pad + center_pixels + pixel) * pixel_bytes;
                dst_row[dst..dst + pixel_bytes].copy_from_slice(right_pixel);
            }
        }

        Ok(())
    }
}

impl<F> RandomAccessSource for SharedMemorySource<F>
where
    F: BandFormat,
    F::Sample: Pod,
{
}

pub enum PreloadedBenchSource {
    U8(SharedMemorySource<U8>),
    U16(SharedMemorySource<U16>),
    F32(SharedMemorySource<F32>),
}

pub fn preload_bench_source(input: &Path) -> PreloadedBenchSource {
    match load_bench_image(input) {
        BenchImage::U8(image) => PreloadedBenchSource::U8(SharedMemorySource::from_image(image)),
        BenchImage::U16(image) => PreloadedBenchSource::U16(SharedMemorySource::from_image(image)),
        BenchImage::F32(image) => PreloadedBenchSource::F32(SharedMemorySource::from_image(image)),
    }
}

pub fn build_viprs_pipeline(input: &Path, op: &str, op_args: &[String]) -> CompiledPipeline {
    if op == "histogram" {
        return build_viprs_source_only_pipeline(input);
    }

    match load_bench_image(input) {
        BenchImage::U8(image) if matches!(op, "dilate" | "erode" | "open" | "close") => {
            build_viprs_u8_morphology_pipeline_from_source(
                image_into_memory_source(image),
                op,
                op_args,
            )
        }
        BenchImage::U8(image) => {
            build_viprs_pipeline_from_source(image_into_memory_source(image), op, op_args)
        }
        BenchImage::U16(image) => {
            build_viprs_pipeline_from_source(image_into_memory_source(image), op, op_args)
        }
        BenchImage::F32(image) => {
            build_viprs_pipeline_from_source(image_into_memory_source(image), op, op_args)
        }
    }
}

fn ensure_composite_input_has_alpha(bands: u32) {
    if bands < 2 {
        eprintln!("composite expects an input image with an alpha band, got {bands} bands");
        std::process::exit(1);
    }
}

fn build_viprs_composite_pipeline_from_source<S>(source: S, mode: BlendMode) -> CompiledPipeline
where
    S: ImageSource<Format = F32> + 'static,
{
    let bands = source.bands();
    ensure_composite_input_has_alpha(bands);

    let mut arena = PipelineArena::with_source(Box::new(source));
    let base = arena.add_node(Box::new(OperationBridge::new_pixel_local(
        CopyOp::<F32>::default(),
        bands,
    )));
    let overlay = arena.add_node(Box::new(OperationBridge::new_pixel_local(
        CopyOp::<F32>::default(),
        bands,
    )));
    let composite = arena.add_node(Box::new(
        CompositeOp::<F32>::new(mode, false, bands).expect("composite op configuration"),
    ));

    arena
        .connect(base, overlay)
        .expect("composite overlay copy");
    arena
        .connect(base, composite)
        .expect("composite base input");
    arena
        .connect_to_slot(overlay, composite, 1)
        .expect("composite overlay input");

    arena
        .enable_cache(composite, default_viprs_cache_bytes())
        .expect("composite cache_last_op failed");

    arena.compile().expect("composite pipeline build failed")
}

fn build_viprs_u8_composite_pipeline_from_source<S>(source: S, mode: BlendMode) -> CompiledPipeline
where
    S: ImageSource<Format = U8> + 'static,
{
    let bands = source.bands();
    ensure_composite_input_has_alpha(bands);

    let mut arena = PipelineArena::with_source(Box::new(source));
    let base = arena.add_node(Box::new(OperationBridge::new_pixel_local(
        Cast::<U8, F32>::new(bands),
        bands,
    )));
    let overlay = arena.add_node(Box::new(OperationBridge::new_pixel_local(
        CopyOp::<F32>::default(),
        bands,
    )));
    let composite = arena.add_node(Box::new(
        CompositeOp::<F32>::new(mode, false, bands).expect("composite op configuration"),
    ));

    arena
        .connect(base, overlay)
        .expect("composite overlay copy");
    arena
        .connect(base, composite)
        .expect("composite base input");
    arena
        .connect_to_slot(overlay, composite, 1)
        .expect("composite overlay input");

    arena
        .enable_cache(composite, default_viprs_cache_bytes())
        .expect("composite cache_last_op failed");

    arena.compile().expect("composite pipeline build failed")
}

fn build_viprs_u16_composite_pipeline_from_source<S>(source: S, mode: BlendMode) -> CompiledPipeline
where
    S: ImageSource<Format = U16> + 'static,
{
    let bands = source.bands();
    ensure_composite_input_has_alpha(bands);

    let mut arena = PipelineArena::with_source(Box::new(source));
    let base = arena.add_node(Box::new(OperationBridge::new_pixel_local(
        Cast::<U16, F32>::new(bands),
        bands,
    )));
    let overlay = arena.add_node(Box::new(OperationBridge::new_pixel_local(
        CopyOp::<F32>::default(),
        bands,
    )));
    let composite = arena.add_node(Box::new(
        CompositeOp::<F32>::new(mode, false, bands).expect("composite op configuration"),
    ));

    arena
        .connect(base, overlay)
        .expect("composite overlay copy");
    arena
        .connect(base, composite)
        .expect("composite base input");
    arena
        .connect_to_slot(overlay, composite, 1)
        .expect("composite overlay input");

    arena
        .enable_cache(composite, default_viprs_cache_bytes())
        .expect("composite cache_last_op failed");

    arena.compile().expect("composite pipeline build failed")
}

pub fn build_viprs_composite_pipeline(input: &Path, mode: BlendMode) -> CompiledPipeline {
    match load_bench_image(input) {
        BenchImage::U8(image) => {
            build_viprs_u8_composite_pipeline_from_source(image_into_memory_source(image), mode)
        }
        BenchImage::U16(image) => {
            build_viprs_u16_composite_pipeline_from_source(image_into_memory_source(image), mode)
        }
        BenchImage::F32(image) => {
            build_viprs_composite_pipeline_from_source(image_into_memory_source(image), mode)
        }
    }
}

pub fn build_viprs_composite_pipeline_from_preloaded(
    source: &PreloadedBenchSource,
    mode: BlendMode,
) -> CompiledPipeline {
    match source {
        PreloadedBenchSource::U8(source) => {
            build_viprs_u8_composite_pipeline_from_source(source.clone(), mode)
        }
        PreloadedBenchSource::U16(source) => {
            build_viprs_u16_composite_pipeline_from_source(source.clone(), mode)
        }
        PreloadedBenchSource::F32(source) => {
            build_viprs_composite_pipeline_from_source(source.clone(), mode)
        }
    }
}

pub fn build_viprs_source_only_pipeline_from_image(image: BenchImage) -> CompiledPipeline {
    match image {
        BenchImage::U8(image) => PipelineBuilder::from_source(image_into_memory_source(image))
            .build()
            .expect("source-only pipeline build failed"),
        BenchImage::U16(image) => PipelineBuilder::from_source(image_into_memory_source(image))
            .build()
            .expect("source-only pipeline build failed"),
        BenchImage::F32(image) => PipelineBuilder::from_source(image_into_memory_source(image))
            .build()
            .expect("source-only pipeline build failed"),
    }
}

pub fn build_viprs_source_only_pipeline(input: &Path) -> CompiledPipeline {
    build_viprs_source_only_pipeline_from_image(load_bench_image(input))
}

pub fn build_viprs_jpeg_source_only_pipeline(input: &Path) -> CompiledPipeline {
    let source = match JpegSequentialScanlineSource::open(input) {
        Ok(source) => source,
        Err(err) => {
            eprintln!("failed to open JPEG load source: {err}");
            std::process::exit(1);
        }
    };

    match PipelineBuilder::from_source(source)
        .with_sequential_access(true)
        .build()
    {
        Ok(pipeline) => pipeline,
        Err(err) => {
            eprintln!("JPEG source-only pipeline build failed: {err}");
            std::process::exit(1);
        }
    }
}

pub fn build_viprs_source_only_pipeline_from_preloaded(
    source: &PreloadedBenchSource,
) -> CompiledPipeline {
    match source {
        PreloadedBenchSource::U8(source) => PipelineBuilder::from_source(source.clone())
            .build()
            .expect("source-only pipeline build failed"),
        PreloadedBenchSource::U16(source) => PipelineBuilder::from_source(source.clone())
            .build()
            .expect("source-only pipeline build failed"),
        PreloadedBenchSource::F32(source) => PipelineBuilder::from_source(source.clone())
            .build()
            .expect("source-only pipeline build failed"),
    }
}

fn png_path_band_format(input: &Path, opts: &LoadOptions) -> BandFormatId {
    let probe = PngCodec::default()
        .probe_path_with_options(input, opts)
        .expect("failed to probe PNG benchmark input");
    match probe.metadata.interpretation {
        Some(Interpretation::Grey16 | Interpretation::Rgb16) => BandFormatId::U16,
        _ => BandFormatId::U8,
    }
}

fn build_viprs_png_path_backed_pipeline(
    input: &Path,
    op: &str,
    op_args: &[String],
) -> CompiledPipeline {
    let opts = LoadOptions::default();
    let demand_hint = (op == "invert").then_some(DemandHint::FatStrip);
    match png_path_band_format(input, &opts) {
        BandFormatId::U8 => build_viprs_pipeline_from_source_with_access_and_hint(
            if op == "invert" {
                DecoderSource::<_, U8>::streaming_path(PngCodec::default(), input, opts)
                    .expect("failed to build PNG sequential benchmark source")
            } else {
                DecoderSource::<_, U8>::probed_path(PngCodec::default(), input)
                    .expect("failed to probe PNG benchmark input")
            },
            op,
            op_args,
            true,
            demand_hint,
        ),
        BandFormatId::U16 => build_viprs_pipeline_from_source_with_access_and_hint(
            if op == "invert" {
                DecoderSource::<_, U16>::streaming_path(PngCodec::default(), input, opts)
                    .expect("failed to build 16-bit PNG sequential benchmark source")
            } else {
                DecoderSource::<_, U16>::probed_path(PngCodec::default(), input)
                    .expect("failed to probe 16-bit PNG benchmark input")
            },
            op,
            op_args,
            true,
            demand_hint,
        ),
        other => panic!("unsupported PNG benchmark format {other:?}"),
    }
}

fn png_thumbnail_prefers_eager_e2e_decode(op: &str) -> bool {
    matches!(
        op,
        "thumbnail"
            | "thumbnail_sharpen"
            | "thumbnail_gauss_blur"
            | "thumbnail_linear"
            | "thumbnail_colourspace_cast"
            | "three_op_chain"
            | "perceptual_enhance"
    )
}

fn build_viprs_path_backed_pipeline(
    input: &Path,
    op: &str,
    op_args: &[String],
) -> Option<CompiledPipeline> {
    let extension = input
        .extension()
        .and_then(std::ffi::OsStr::to_str)?
        .to_ascii_lowercase();
    let opts = LoadOptions::default();
    match extension.as_str() {
        "jpg" | "jpeg" | "jpe" => Some(build_viprs_pipeline_from_source_with_access(
            JpegSequentialScanlineSource::open(input)
                .expect("failed to open JPEG scanline thumbnail/workflow source"),
            op,
            op_args,
            true,
        )),
        "png" => (!png_thumbnail_prefers_eager_e2e_decode(op))
            .then(|| build_viprs_png_path_backed_pipeline(input, op, op_args)),
        "webp"
            if matches!(
                op,
                "thumbnail"
                    | "thumbnail_sharpen"
                    | "thumbnail_gauss_blur"
                    | "thumbnail_linear"
                    | "thumbnail_colourspace_cast"
            ) =>
        {
            Some(build_viprs_pipeline_from_source(
                DecoderSource::<_, U8>::streaming_path(WebpCodec, input, opts)
                    .expect("failed to build WebP streaming thumbnail source"),
                op,
                op_args,
            ))
        }
        "webp" => Some(build_viprs_pipeline_from_source(
            DecoderSource::<_, U8>::probed_path(WebpCodec, input)
                .expect("failed to probe WebP input"),
            op,
            op_args,
        )),
        "tif" | "tiff" => Some(build_viprs_pipeline_from_source(
            DecoderSource::<_, U8>::streaming_path(TiffDecoder, input, opts)
                .expect("failed to build TIFF streaming thumbnail/workflow source"),
            op,
            op_args,
        )),
        _ => None,
    }
}

pub fn build_viprs_e2e_pipeline(input: &Path, op: &str, op_args: &[String]) -> CompiledPipeline {
    if op == "load-jpeg" {
        return build_viprs_jpeg_source_only_pipeline(input);
    }

    if op == "histogram" {
        return build_viprs_source_only_pipeline(input);
    }

    if matches!(
        op,
        "thumbnail"
            | "workflow"
            | "thumbnail_colourspace_cast"
            | "invert"
            | "invert_invert"
            | "thumbnail_sharpen"
            | "thumbnail_gauss_blur"
            | "thumbnail_linear"
            | "three_op_chain"
            | "perceptual_enhance"
    ) {
        if let Some(pipeline) = build_viprs_path_backed_pipeline(input, op, op_args) {
            return pipeline;
        }
    }

    build_viprs_pipeline(input, op, op_args)
}

pub fn build_viprs_pipeline_from_preloaded(
    source: &PreloadedBenchSource,
    op: &str,
    op_args: &[String],
) -> CompiledPipeline {
    if op == "load-jpeg" {
        return build_viprs_source_only_pipeline_from_preloaded(source);
    }

    if op == "grey" {
        return build_viprs_grey_pipeline(source_width(source), source_height(source));
    }

    if op == "histogram" {
        return build_viprs_source_only_pipeline_from_preloaded(source);
    }

    match source {
        PreloadedBenchSource::U8(source) if matches!(op, "dilate" | "erode" | "open" | "close") => {
            build_viprs_u8_morphology_pipeline_from_source(source.clone(), op, op_args)
        }
        PreloadedBenchSource::U8(source) => {
            build_viprs_pipeline_from_source(source.clone(), op, op_args)
        }
        PreloadedBenchSource::U16(source) => {
            build_viprs_pipeline_from_source(source.clone(), op, op_args)
        }
        PreloadedBenchSource::F32(source) => {
            build_viprs_pipeline_from_source(source.clone(), op, op_args)
        }
    }
}

fn source_width(source: &PreloadedBenchSource) -> u32 {
    match source {
        PreloadedBenchSource::U8(source) => source.width(),
        PreloadedBenchSource::U16(source) => source.width(),
        PreloadedBenchSource::F32(source) => source.width(),
    }
}

fn source_height(source: &PreloadedBenchSource) -> u32 {
    match source {
        PreloadedBenchSource::U8(source) => source.height(),
        PreloadedBenchSource::U16(source) => source.height(),
        PreloadedBenchSource::F32(source) => source.height(),
    }
}

pub fn build_viprs_pipeline_from_source<F, S>(
    source: S,
    op: &str,
    op_args: &[String],
) -> CompiledPipeline
where
    F: NumericBand + 'static,
    F::Sample: BandMeanSample
        + Default
        + PartialOrd
        + Pod
        + ToF32
        + ToF64
        + viprs::domain::ops::conversion::gamma::GammaSample,
    S: ImageSource<Format = F> + 'static,
    ConvOp<F>: Op,
    Flip<F>: Op,
    GaussBlurH<F>: Op,
    GaussBlurV<F32>: Op,
    GammaOp<F>: Op,
    Median<F>: Op,
    Prewitt<F>: Op,
    Sharpen<F>: Op,
    Sobel<F>: Op,
{
    build_viprs_pipeline_from_source_with_access(source, op, op_args, false)
}

pub fn build_viprs_pipeline_from_source_with_access<F, S>(
    source: S,
    op: &str,
    op_args: &[String],
    sequential: bool,
) -> CompiledPipeline
where
    F: NumericBand + 'static,
    F::Sample: BandMeanSample
        + Default
        + PartialOrd
        + Pod
        + ToF32
        + ToF64
        + viprs::domain::ops::conversion::gamma::GammaSample,
    S: ImageSource<Format = F> + 'static,
    ConvOp<F>: Op,
    Flip<F>: Op,
    GaussBlurH<F>: Op,
    GaussBlurV<F32>: Op,
    GammaOp<F>: Op,
    Median<F>: Op,
    Prewitt<F>: Op,
    Sharpen<F>: Op,
    Sobel<F>: Op,
{
    build_viprs_pipeline_from_source_with_access_and_hint(source, op, op_args, sequential, None)
}

fn build_viprs_pipeline_from_source_with_access_and_hint<F, S>(
    source: S,
    op: &str,
    op_args: &[String],
    sequential: bool,
    demand_hint_override: Option<DemandHint>,
) -> CompiledPipeline
where
    F: NumericBand + 'static,
    F::Sample: BandMeanSample
        + Default
        + PartialOrd
        + Pod
        + ToF32
        + ToF64
        + viprs::domain::ops::conversion::gamma::GammaSample,
    S: ImageSource<Format = F> + 'static,
    ConvOp<F>: Op,
    Flip<F>: Op,
    GaussBlurH<F>: Op,
    GaussBlurV<F32>: Op,
    GammaOp<F>: Op,
    Median<F>: Op,
    Prewitt<F>: Op,
    Sharpen<F>: Op,
    Sobel<F>: Op,
{
    if op == "grey" {
        return build_viprs_grey_pipeline(source.width(), source.height());
    }

    let bands = source.bands();
    let source_width = source.width();
    let source_height = source.height();
    let source_colorspace = source
        .metadata()
        .interpretation
        .and_then(interpretation_to_bench_colorspace)
        .unwrap_or(ColorspaceId::SRgb);
    let builder = PipelineBuilder::from_source(source).with_sequential_access(sequential);
    let builder = if let Some(demand_hint) = demand_hint_override {
        builder.with_demand_hint_override(demand_hint)
    } else {
        builder
    };

    let builder = match op {
        "invert" => builder
            .invert()
            .expect("invert failed")
            .flush_into_identity()
            .expect("flush failed"),
        "abs" => builder
            .then(match F::ID {
                BandFormatId::U8 => {
                    Box::new(OperationBridge::new_pixel_local(Abs::<U8>::new(), bands))
                }
                BandFormatId::U16 => {
                    Box::new(OperationBridge::new_pixel_local(Abs::<U16>::new(), bands))
                }
                BandFormatId::I16 => {
                    Box::new(OperationBridge::new_pixel_local(Abs::<I16>::new(), bands))
                }
                BandFormatId::U32 => {
                    Box::new(OperationBridge::new_pixel_local(Abs::<U32>::new(), bands))
                }
                BandFormatId::I32 => {
                    Box::new(OperationBridge::new_pixel_local(Abs::<I32>::new(), bands))
                }
                BandFormatId::F32 => {
                    Box::new(OperationBridge::new_pixel_local(Abs::<F32>::new(), bands))
                }
                BandFormatId::F64 => {
                    Box::new(OperationBridge::new_pixel_local(Abs::<F64>::new(), bands))
                }
            })
            .expect("abs failed"),
        "sign" => builder
            .then(match F::ID {
                BandFormatId::U8 => {
                    Box::new(OperationBridge::new_pixel_local(Sign::<U8>::new(), bands))
                }
                BandFormatId::U16 => {
                    Box::new(OperationBridge::new_pixel_local(Sign::<U16>::new(), bands))
                }
                BandFormatId::I16 => {
                    Box::new(OperationBridge::new_pixel_local(Sign::<I16>::new(), bands))
                }
                BandFormatId::U32 => {
                    Box::new(OperationBridge::new_pixel_local(Sign::<U32>::new(), bands))
                }
                BandFormatId::I32 => {
                    Box::new(OperationBridge::new_pixel_local(Sign::<I32>::new(), bands))
                }
                BandFormatId::F32 => {
                    Box::new(OperationBridge::new_pixel_local(Sign::<F32>::new(), bands))
                }
                BandFormatId::F64 => {
                    Box::new(OperationBridge::new_pixel_local(Sign::<F64>::new(), bands))
                }
            })
            .expect("sign failed"),
        "bandmean" => builder
            .then(Box::new(BandMean::<F>::new(bands as usize).into_bridge()))
            .expect("bandmean failed"),
        "add" => builder
            .then(match F::ID {
                BandFormatId::U8 => Box::new(OperationBridge::new(Add::<U8>::new(vec![5]), bands)),
                BandFormatId::U16 => {
                    Box::new(OperationBridge::new(Add::<U16>::new(vec![5]), bands))
                }
                BandFormatId::F32 => {
                    Box::new(OperationBridge::new(Add::<F32>::new(vec![5.0]), bands))
                }
                other => {
                    eprintln!("Unsupported add format: {other:?}");
                    std::process::exit(1);
                }
            })
            .expect("add failed"),
        "multiply" => builder
            .then(match F::ID {
                BandFormatId::U8 => {
                    Box::new(OperationBridge::new(Multiply::<U8>::new(vec![2]), bands))
                }
                BandFormatId::U16 => {
                    Box::new(OperationBridge::new(Multiply::<U16>::new(vec![2]), bands))
                }
                BandFormatId::F32 => {
                    Box::new(OperationBridge::new(Multiply::<F32>::new(vec![2.0]), bands))
                }
                other => {
                    eprintln!("Unsupported multiply format: {other:?}");
                    std::process::exit(1);
                }
            })
            .expect("multiply failed"),
        "subtract" => builder
            .then(match F::ID {
                BandFormatId::U8 => {
                    Box::new(OperationBridge::new(Subtract::<U8>::new(vec![5]), bands))
                }
                BandFormatId::U16 => {
                    Box::new(OperationBridge::new(Subtract::<U16>::new(vec![5]), bands))
                }
                BandFormatId::F32 => {
                    Box::new(OperationBridge::new(Subtract::<F32>::new(vec![5.0]), bands))
                }
                other => {
                    eprintln!("Unsupported subtract format: {other:?}");
                    std::process::exit(1);
                }
            })
            .expect("subtract failed"),
        "and" => builder
            .then(match F::ID {
                BandFormatId::U8 => Box::new(OperationBridge::new(And::<U8>::new(0xF0u8), bands)),
                BandFormatId::U16 => {
                    Box::new(OperationBridge::new(And::<U16>::new(0x00F0u16), bands))
                }
                BandFormatId::F32 => {
                    Box::new(OperationBridge::new(And::<F32, U16>::new(0x00F0u16), bands))
                }
                other => {
                    eprintln!("Unsupported and format: {other:?}");
                    std::process::exit(1);
                }
            })
            .expect("and failed"),
        "equal" => builder
            .then(match F::ID {
                BandFormatId::U8 => Box::new(OperationBridge::new_pixel_local(
                    Equal::<U8>::new(128u8),
                    bands,
                )),
                BandFormatId::U16 => Box::new(OperationBridge::new_pixel_local(
                    Equal::<U16>::new(32_768u16),
                    bands,
                )),
                BandFormatId::F32 => Box::new(OperationBridge::new_pixel_local(
                    Equal::<F32>::new(0.5f32),
                    bands,
                )),
                other => {
                    eprintln!("Unsupported equal format: {other:?}");
                    std::process::exit(1);
                }
            })
            .expect("equal failed"),
        "linear" => {
            let scale: f64 = op_args.first().and_then(|s| s.parse().ok()).unwrap_or(2.0);
            let offset: f64 = op_args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5.0);
            builder
                .linear(scale, offset)
                .expect("linear failed")
                .flush_into_identity()
                .expect("flush failed")
        }
        "round" => builder
            .then(match F::ID {
                BandFormatId::F32 => {
                    Box::new(OperationBridge::new_pixel_local(Round::<F32>::new(), bands))
                }
                BandFormatId::F64 => {
                    Box::new(OperationBridge::new_pixel_local(Round::<F64>::new(), bands))
                }
                other => {
                    eprintln!("Unsupported round format: {other:?}");
                    std::process::exit(1);
                }
            })
            .expect("round failed"),
        "floor" => builder
            .then(match F::ID {
                BandFormatId::F32 => {
                    Box::new(OperationBridge::new_pixel_local(Floor::<F32>::new(), bands))
                }
                BandFormatId::F64 => {
                    Box::new(OperationBridge::new_pixel_local(Floor::<F64>::new(), bands))
                }
                other => {
                    eprintln!("Unsupported floor format: {other:?}");
                    std::process::exit(1);
                }
            })
            .expect("floor failed"),
        "ceil" => builder
            .then(match F::ID {
                BandFormatId::F32 => {
                    Box::new(OperationBridge::new_pixel_local(Ceil::<F32>::new(), bands))
                }
                BandFormatId::F64 => {
                    Box::new(OperationBridge::new_pixel_local(Ceil::<F64>::new(), bands))
                }
                other => {
                    eprintln!("Unsupported ceil format: {other:?}");
                    std::process::exit(1);
                }
            })
            .expect("ceil failed"),
        "cast" => builder
            .cast(parse_cast_target(F::ID, op_args))
            .expect("cast failed"),
        "flip" => builder
            .then(Box::new(OperationBridge::new(
                match parse_flip_direction(op_args) {
                    FlipDirection::Horizontal => Flip::<F>::horizontal(source_width),
                    FlipDirection::Vertical => Flip::<F>::vertical(source_height),
                },
                bands,
            )))
            .expect("flip failed"),
        "gamma" => builder
            .then(Box::new(OperationBridge::new_pixel_local(
                GammaOp::<F>::new(parse_gamma_exponent(op_args)),
                bands,
            )))
            .expect("gamma failed"),
        "convolve" => builder
            .then(Box::new(OperationBridge::new(
                ConvOp::<F>::new(box_3x3_kernel()).expect("convolve kernel"),
                bands,
            )))
            .expect("convolve failed"),
        "conv_sharpen3" => builder
            .then(Box::new(OperationBridge::new(
                ConvOp::<F>::new(sharpen_3x3_kernel()).expect("conv_sharpen3 kernel"),
                bands,
            )))
            .expect("conv_sharpen3 failed"),
        "conv_sobel3" => builder
            .then(Box::new(OperationBridge::new(
                ConvOp::<F>::new(sobel_x_3x3_kernel()).expect("conv_sobel3 kernel"),
                bands,
            )))
            .expect("conv_sobel3 failed"),
        "sobel" => builder
            .then(Box::new(OperationBridge::new(Sobel::<F>::new(), bands)))
            .expect("sobel failed"),
        "prewitt" => builder
            .then(Box::new(OperationBridge::new(Prewitt::<F>::new(), bands)))
            .expect("prewitt failed"),
        "laplacian" => builder
            .then(Box::new(OperationBridge::new(
                ConvOp::<F>::new(laplacian_3x3_kernel()).expect("laplacian kernel"),
                bands,
            )))
            .expect("laplacian failed"),
        "median_blur" => {
            let kernel_size = parse_median_kernel_size(op_args);
            builder
                .then(Box::new(OperationBridge::new(
                    Median::<F>::new(kernel_size, kernel_size).expect("median_blur kernel"),
                    bands,
                )))
                .expect("median_blur failed")
        }
        "unsharp_mask" => {
            let sigma: f32 = op_args.first().and_then(|s| s.parse().ok()).unwrap_or(0.5);
            let strength: f32 = op_args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3.0);
            builder
                .then(Box::new(OperationBridge::new(
                    Sharpen::<F>::new(sigma, strength),
                    bands,
                )))
                .expect("unsharp_mask failed")
        }
        "recomb" => {
            if bands != 3 {
                eprintln!("recomb benchmark expects a 3-band input image, got {bands}");
                std::process::exit(1);
            }
            builder
                .then(match F::ID {
                    BandFormatId::U8 => Box::new(OperationBridge::with_dynamic_bands_pixel_local(
                        RecombOp::<U8>::new(recomb_matrix()),
                        bands,
                        2,
                    )),
                    BandFormatId::F32 => Box::new(OperationBridge::with_dynamic_bands_pixel_local(
                        RecombOp::<F32>::new(recomb_matrix()),
                        bands,
                        2,
                    )),
                    other => {
                        eprintln!("Unsupported recomb format: {other:?}");
                        std::process::exit(1);
                    }
                })
                .expect("recomb failed")
        }
        "freqfilt" => builder
            .then(Box::new(BandMean::<F>::new(bands as usize).into_bridge()))
            .and_then(|b| {
                b.then(Box::new(OperationBridge::new(
                    FwFftOp::<F>::new(source_width, source_height)
                        .expect("FwFftOp should construct"),
                    1,
                )))
            })
            .and_then(|b| {
                b.then(Box::new(OperationBridge::new(
                    InvFftOp::<F32>::new(source_width, source_height)
                        .expect("InvFftOp should construct"),
                    2,
                )))
            })
            .expect("freqfilt failed"),
        "resize" => {
            let scale: f64 = op_args.first().and_then(|s| s.parse().ok()).unwrap_or(0.5);
            apply_resize(builder, scale).expect("resize failed")
        }
        "zoom" => {
            let xfac: u32 = op_args.first().and_then(|s| s.parse().ok()).unwrap_or(2);
            let yfac: u32 = op_args.get(1).and_then(|s| s.parse().ok()).unwrap_or(xfac);
            builder.zoom(xfac, yfac).expect("zoom failed")
        }
        "affine" => {
            let inverse = invert_affine_forward_matrix(parse_affine_forward_matrix(op_args));
            builder
                .affine(
                    inverse,
                    0.0,
                    0.0,
                    source_width,
                    source_height,
                    InterpolationKernel::Bilinear,
                )
                .expect("affine failed")
        }
        "similarity" => {
            let scale: f64 = op_args
                .first()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_SIMILARITY_SCALE);
            let angle: f64 = op_args
                .get(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_SIMILARITY_ANGLE);
            builder
                .similarity(scale, angle, InterpolationKernel::Bilinear)
                .expect("similarity failed")
        }
        "shrink" => {
            let h_factor: u32 = op_args.first().and_then(|s| s.parse().ok()).unwrap_or(2);
            let v_factor: u32 = op_args
                .get(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(h_factor);
            builder.shrink(h_factor, v_factor).expect("shrink failed")
        }
        "shrinkh" => {
            let factor: u32 = op_args.first().and_then(|s| s.parse().ok()).unwrap_or(2);
            builder.shrink_h(factor).expect("shrinkh failed")
        }
        "shrinkv" => {
            let factor: u32 = op_args.first().and_then(|s| s.parse().ok()).unwrap_or(2);
            builder.shrink_v(factor).expect("shrinkv failed")
        }
        "thumbnail" => {
            let tw: u32 = op_args.first().and_then(|s| s.parse().ok()).unwrap_or(800);
            apply_thumbnail(builder, tw).expect("thumbnail failed")
        }
        "sharpen" => {
            // Keep the xtask CLI stable: op_args are `[sigma, strength]`, where `strength`
            // maps to libvips' `m2` parameter and the remaining parameters use libvips defaults.
            let sigma: f32 = op_args.first().and_then(|s| s.parse().ok()).unwrap_or(0.5);
            let strength: f32 = op_args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3.0);
            apply_sharpen(builder, sigma, strength).expect("sharpen failed")
        }
        "draw_line" => builder
            .then(match F::ID {
                BandFormatId::U8 => Box::new(OperationBridge::new_pixel_local(
                    DrawLinePipelineOp::<U8>::new(
                        source_width,
                        source_height,
                        vec![u8::MAX; bands as usize],
                    ),
                    bands,
                )),
                BandFormatId::U16 => Box::new(OperationBridge::new_pixel_local(
                    DrawLinePipelineOp::<U16>::new(
                        source_width,
                        source_height,
                        vec![u16::MAX; bands as usize],
                    ),
                    bands,
                )),
                BandFormatId::F32 => Box::new(OperationBridge::new_pixel_local(
                    DrawLinePipelineOp::<F32>::new(
                        source_width,
                        source_height,
                        vec![1.0; bands as usize],
                    ),
                    bands,
                )),
                other => {
                    eprintln!("Unsupported draw_line format: {other:?}");
                    std::process::exit(1);
                }
            })
            .expect("draw_line failed"),
        "draw_rect" => builder
            .then(match F::ID {
                BandFormatId::U8 => Box::new(OperationBridge::new_pixel_local(
                    DrawRectPipelineOp::<U8>::new(
                        source_width,
                        source_height,
                        vec![u8::MAX; bands as usize],
                    ),
                    bands,
                )),
                BandFormatId::U16 => Box::new(OperationBridge::new_pixel_local(
                    DrawRectPipelineOp::<U16>::new(
                        source_width,
                        source_height,
                        vec![u16::MAX; bands as usize],
                    ),
                    bands,
                )),
                BandFormatId::F32 => Box::new(OperationBridge::new_pixel_local(
                    DrawRectPipelineOp::<F32>::new(
                        source_width,
                        source_height,
                        vec![1.0; bands as usize],
                    ),
                    bands,
                )),
                other => {
                    eprintln!("Unsupported draw_rect format: {other:?}");
                    std::process::exit(1);
                }
            })
            .expect("draw_rect failed"),
        "draw_circle" => builder
            .then(match F::ID {
                BandFormatId::U8 => Box::new(OperationBridge::new_pixel_local(
                    DrawCirclePipelineOp::<U8>::new(
                        source_width,
                        source_height,
                        vec![u8::MAX; bands as usize],
                    ),
                    bands,
                )),
                BandFormatId::U16 => Box::new(OperationBridge::new_pixel_local(
                    DrawCirclePipelineOp::<U16>::new(
                        source_width,
                        source_height,
                        vec![u16::MAX; bands as usize],
                    ),
                    bands,
                )),
                BandFormatId::F32 => Box::new(OperationBridge::new_pixel_local(
                    DrawCirclePipelineOp::<F32>::new(
                        source_width,
                        source_height,
                        vec![1.0; bands as usize],
                    ),
                    bands,
                )),
                other => {
                    eprintln!("Unsupported draw_circle format: {other:?}");
                    std::process::exit(1);
                }
            })
            .expect("draw_circle failed"),
        "gauss_blur" => {
            let sigma: f32 = op_args.first().and_then(|s| s.parse().ok()).unwrap_or(1.5);
            apply_gauss_blur::<F>(builder, bands, sigma).expect("gauss_blur failed")
        }
        "colourspace" => {
            if op_args.is_empty()
                && source_colorspace == ColorspaceId::SRgb
                && F::ID == BandFormatId::U8
            {
                builder
                    .then(Box::new(OperationBridge::new_pixel_local(
                        SRgbLabRoundtrip,
                        bands,
                    )))
                    .expect("colourspace failed")
            } else {
                let mut builder = builder.with_colorspace(source_colorspace);
                for destination in colourspace_route_from_args(source_colorspace, op_args) {
                    builder = apply_colourspace_destination(builder, destination)
                        .expect("colourspace failed");
                }
                builder
            }
        }
        "srgb_to_lab" => builder
            .with_colorspace(ColorspaceId::SRgb)
            .colourspace::<Lab>()
            .expect("srgb_to_lab failed"),
        // workflow: thumbnail → sharpen → encode (encode step is handled in runner, not the pipeline)
        "workflow" => {
            let tw: u32 = op_args.get(1).and_then(|s| s.parse().ok()).unwrap_or(400);
            apply_thumbnail(builder.with_colorspace(ColorspaceId::SRgb), tw)
                .and_then(|b| apply_sharpen(b, 0.5, 3.0))
                .expect("workflow pipeline failed")
        }
        "invert_invert" => builder
            .invert()
            .and_then(PipelineBuilder::invert)
            .and_then(|b| b.flush_into_identity())
            .expect("invert_invert pipeline failed"),
        "thumbnail_sharpen" => apply_thumbnail(builder.with_colorspace(source_colorspace), 400)
            .and_then(|b| apply_sharpen(b, 0.5, 3.0))
            .expect("thumbnail_sharpen pipeline failed"),
        "thumbnail_colourspace_cast" => {
            apply_thumbnail(builder.with_colorspace(source_colorspace), 400)
                .and_then(|b| b.colourspace::<Lab>())
                .and_then(|b| b.cast(BandFormatId::U8))
                .expect("thumbnail_colourspace_cast pipeline failed")
        }
        "thumbnail_gauss_blur" => apply_thumbnail(builder, 400)
            .and_then(|b| apply_gauss_blur::<F>(b, bands, 2.0))
            .expect("thumbnail_gauss_blur pipeline failed"),
        "thumbnail_linear" => apply_thumbnail(builder, 400)
            .and_then(|b| b.linear(1.2, 0.0))
            .and_then(|b| b.flush_into_identity())
            .expect("thumbnail_linear pipeline failed"),
        "resize_colourspace" => apply_resize(builder.with_colorspace(source_colorspace), 0.5)
            .and_then(|b| b.colourspace::<Lab>())
            .expect("resize_colourspace pipeline failed"),
        "embed" => builder
            .embed(
                source_width + EMBED_PAD_WIDTH,
                source_height + EMBED_PAD_HEIGHT,
                EMBED_OFFSET_X,
                EMBED_OFFSET_Y,
                source_width,
                source_height,
                ExtendMode::Copy,
            )
            .expect("embed pipeline failed"),
        "extract-area" => builder
            .extract_area(
                EXTRACT_OFFSET_X,
                EXTRACT_OFFSET_Y,
                source_width - EXTRACT_TRIM_WIDTH,
                source_height - EXTRACT_TRIM_HEIGHT,
            )
            .expect("extract-area pipeline failed"),
        "embed_extract" => {
            // Keep the benchmark valid on the 8192 fixture by treating 2048×2048 as the
            // minimum canvas size rather than shrinking larger sources before embed.
            let embed_width = source_width.max(2048);
            let embed_height = source_height.max(2048);
            builder
                .embed(
                    embed_width,
                    embed_height,
                    0,
                    0,
                    source_width,
                    source_height,
                    ExtendMode::Copy,
                )
                .and_then(|b| b.extract_area(100, 100, 800, 600))
                .expect("embed_extract pipeline failed")
        }
        "three_op_chain" => apply_thumbnail(builder.with_colorspace(source_colorspace), 400)
            .and_then(|b| apply_sharpen(b, 0.5, 3.0))
            .and_then(|b| apply_gauss_blur::<F>(b, bands, 1.0))
            .expect("three_op_chain pipeline failed"),
        // perceptual_enhance: production e-commerce image pipeline
        //
        // thumbnail(800px, Lanczos3)
        //   → sRGB → Lab → linear(×1.05, −3.5) → sRGB
        //   → sharpen(σ=0.5)         (unsharp mask; internally converts to Lab again)
        //   → GammaOp(0.95)          (compensate slight darkening from sharpening)
        //   → cast(U8)               (ensure 8-bit for WebP output)
        //
        // This is what a stock-photo or marketplace CDN pipeline does: the Lab round-trip
        // around contrast prevents hue shifts, and sharpening in Lab avoids colour fringing.
        "perceptual_enhance" => {
            let tw: u32 = op_args.get(1).and_then(|s| s.parse().ok()).unwrap_or(800);
            apply_thumbnail(builder.with_colorspace(ColorspaceId::SRgb), tw)
                .and_then(|b| {
                    b.then(Box::new(OperationBridge::new_pixel_local(
                        cached_perceptual_lab_adjust()?,
                        3,
                    )))
                })
                .and_then(|b| apply_sharpen(b, 0.5, 3.0))
                .and_then(|b| b.cast(BandFormatId::U8))
                .and_then(|b| {
                    b.then(Box::new(OperationBridge::new_pixel_local(
                        GammaOp::<U8>::new(0.95),
                        3,
                    )))
                })
                .expect("perceptual_enhance pipeline failed")
        }
        other => {
            eprintln!("Unsupported viprs operation: {other}");
            std::process::exit(1);
        }
    };

    let builder = builder
        .cache_last_op(default_viprs_cache_bytes())
        .expect("cache_last_op failed");

    builder.build().expect("pipeline build failed")
}

#[cfg(test)]
mod tests {
    use super::*;
    use viprs::domain::format::U8;
    use viprs::{
        adapters::{scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink},
        ports::scheduler::TileScheduler,
    };

    #[test]
    fn shared_memory_source_clone_reuses_backing_pixels() {
        let image = Image::<U8>::from_buffer(2, 1, 1, vec![3, 7]).expect("image");
        let source = SharedMemorySource::from_image(image);
        let cloned = source.clone();

        assert!(Arc::ptr_eq(&source.data, &cloned.data));
        assert_eq!(source.width(), cloned.width());
        assert_eq!(source.height(), cloned.height());
        assert_eq!(source.bands(), cloned.bands());
    }

    #[test]
    fn shared_memory_source_full_width_read_copies_contiguous_rows() {
        let image = Image::<U8>::from_buffer(2, 2, 1, vec![1, 2, 3, 4]).expect("image");
        let source = SharedMemorySource::from_image(image);
        let mut output = vec![0u8; 4];

        source
            .read_region(Region::new(0, 0, 2, 2), &mut output)
            .expect("full-width read");

        assert_eq!(output, vec![1, 2, 3, 4]);
    }

    #[test]
    fn shared_memory_source_borrows_full_rows_for_zero_origin_region() {
        let image = Image::<U8>::from_buffer(4, 2, 1, vec![1, 2, 3, 4, 5, 6, 7, 8]).expect("image");
        let source = SharedMemorySource::from_image(image);

        let borrowed = source
            .borrow_region(Region::new(0, 1, 2, 1))
            .expect("borrowed full row");

        assert_eq!(borrowed, &[5, 6, 7, 8]);
    }

    #[test]
    fn shared_memory_source_rejects_non_contiguous_borrow() {
        let image = Image::<U8>::from_buffer(4, 2, 1, vec![1, 2, 3, 4, 5, 6, 7, 8]).expect("image");
        let source = SharedMemorySource::from_image(image);

        assert!(source.borrow_region(Region::new(1, 0, 2, 2)).is_none());
    }

    #[test]
    fn shared_memory_source_clamps_out_of_bounds_without_per_pixel_layout_changes() {
        let image = Image::<U8>::from_buffer(4, 2, 1, vec![1, 2, 3, 4, 5, 6, 7, 8]).expect("image");
        let source = SharedMemorySource::from_image(image);
        let mut output = vec![0u8; 12];

        source
            .read_region(Region::new(-1, 0, 6, 2), &mut output)
            .expect("clamped read");

        assert_eq!(output, vec![1, 1, 2, 3, 4, 4, 5, 5, 6, 7, 8, 8]);
    }

    #[test]
    fn shared_memory_source_zero_width_clamped_read_panics() {
        let image = Image::<U8>::from_buffer(0, 1, 1, vec![]).expect("image");
        let source = SharedMemorySource::from_image(image);
        let mut output = vec![9u8; 1];

        source
            .read_region(Region::new(-1, 0, 1, 1), &mut output)
            .expect("zero-width clamped read");

        assert_eq!(output, vec![0]);
    }

    #[test]
    fn shared_memory_source_zero_height_clamped_read_panics() {
        let image = Image::<U8>::from_buffer(1, 0, 1, vec![]).expect("image");
        let source = SharedMemorySource::from_image(image);
        let mut output = vec![9u8; 1];

        source
            .read_region(Region::new(0, -1, 1, 1), &mut output)
            .expect("zero-height clamped read");

        assert_eq!(output, vec![0]);
    }

    #[test]
    fn three_op_chain_accepts_sources_without_interpretation_metadata() {
        let image = Image::<U8>::from_buffer(32, 32, 3, vec![17; 32 * 32 * 3]).expect("image");
        let source = SharedMemorySource::from_image(image);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            build_viprs_pipeline_from_source::<U8, _>(source, "three_op_chain", &[])
        }));

        assert!(result.is_ok());
    }

    #[test]
    fn build_viprs_e2e_pipeline_supports_load_jpeg() {
        let input =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/images/sample.jpg");

        let pipeline = build_viprs_e2e_pipeline(&input, "load-jpeg", &[]);

        assert!(pipeline.width > 0);
        assert!(pipeline.height > 0);
        assert!(pipeline.output_bands > 0);
    }

    #[test]
    fn perceptual_enhance_pipeline_uses_second_arg_as_width() {
        let input = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/images/bench_2048x2048.jpg");

        let pipeline = build_viprs_e2e_pipeline(
            &input,
            "perceptual_enhance",
            &["webp".to_owned(), "640".to_owned()],
        );

        assert_eq!(pipeline.width, 640);
    }

    #[test]
    fn build_viprs_e2e_pipeline_uses_sequential_png_invert_path() {
        let input = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/images/bench_2048x2048.png");

        let pipeline = build_viprs_e2e_pipeline(&input, "invert", &[]);

        assert!(pipeline.sequential);
        assert_eq!(pipeline.demand_hint, DemandHint::FatStrip);
    }

    #[test]
    fn build_viprs_e2e_pipeline_prefers_eager_png_thumbnail_path() {
        let input = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/images/bench_2048x2048.png");

        let pipeline = build_viprs_e2e_pipeline(&input, "thumbnail", &["400".to_owned()]);

        assert!(!pipeline.sequential);
    }

    #[test]
    fn build_viprs_pipeline_supports_histogram_source_only_path() {
        let input = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/images/bench_2048x2048.jpg");

        let pipeline = build_viprs_pipeline(&input, "histogram", &[]);

        assert!(pipeline.nodes.is_empty());
        assert!(pipeline.width > 0);
        assert!(pipeline.height > 0);
    }

    #[test]
    fn build_viprs_pipeline_supports_draw_ops() {
        let input = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/images/bench_2048x2048.jpg");

        for op in ["draw_line", "draw_rect", "draw_circle"] {
            let pipeline = build_viprs_pipeline(&input, op, &[]);
            assert!(
                !pipeline.nodes.is_empty(),
                "{op} should compile to at least one pipeline node"
            );
        }
    }

    #[test]
    fn build_viprs_e2e_pipeline_runs_512px_png_invert_without_panicking() {
        let input = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/images/bench_512x512.png");
        let pipeline = build_viprs_e2e_pipeline(&input, "invert", &[]);
        let sink = MemorySink::for_pipeline(&pipeline).expect("memory sink");

        RayonScheduler::new(4)
            .expect("scheduler")
            .run_concurrent(&pipeline, &sink)
            .expect("512px png invert benchmark pipeline should complete");
    }

    #[test]
    fn build_viprs_e2e_pipeline_runs_16bit_png_invert_without_panicking() {
        let input =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/images/sample.png");
        let pipeline = build_viprs_e2e_pipeline(&input, "invert", &[]);
        let sink = MemorySink::for_pipeline(&pipeline).expect("memory sink");

        RayonScheduler::new(4)
            .expect("scheduler")
            .run_concurrent(&pipeline, &sink)
            .expect("16-bit png invert benchmark pipeline should complete");
    }

    #[test]
    fn build_viprs_e2e_pipeline_runs_jpeg_bandmean_with_single_band_output() {
        let input = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/images/bench_512x512.jpg");
        let pipeline = build_viprs_e2e_pipeline(&input, "bandmean", &[]);
        let sink = MemorySink::for_pipeline(&pipeline).expect("memory sink");

        assert_eq!(pipeline.output_bands, 1);

        RayonScheduler::new(4)
            .expect("scheduler")
            .run_concurrent(&pipeline, &sink)
            .expect("jpeg bandmean benchmark pipeline should complete");
    }

    #[test]
    fn split_jpeg_thumbnail_shrink_preserves_native_8x_and_tracks_residual_2x() {
        assert_eq!(split_jpeg_thumbnail_shrink_factor(8), (8, 1));
        assert_eq!(split_jpeg_thumbnail_shrink_factor(16), (8, 2));
    }

    #[test]
    fn jpeg_row_pair_factor2_box_averages_rgb() {
        let top = [
            10, 20, 30, 30, 40, 50, //
            50, 60, 70, 70, 80, 90,
        ];
        let bottom = [
            20, 30, 40, 40, 50, 60, //
            60, 70, 80, 80, 90, 100,
        ];
        let mut output = [0u8; 6];

        box_shrink_row_pair_factor2(&top, &bottom, 3, &mut output);

        assert_eq!(output, [25, 35, 45, 65, 75, 85]);
    }

    #[test]
    fn jpeg_scanline_thumbnail_hint_16_exposes_residual_shrunk_dimensions() {
        let input = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/images/bench_8192x8192.jpg");
        let mut source = JpegSequentialScanlineSource::open(&input).expect("jpeg source");

        assert!(
            source
                .set_thumbnail_shrink_on_load(NonZeroU8::new(16).expect("non-zero factor"))
                .expect("set thumbnail hint")
        );
        assert_eq!(source.width(), 512);
        assert_eq!(source.height(), 512);

        let state = source.state.lock().expect("state");
        assert_eq!(state.native_shrink_factor, 8);
        assert_eq!(state.residual_shrink_factor, 2);
    }
}

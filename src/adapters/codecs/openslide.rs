//! Openslide adapter module.
//!
//! This module provides concrete codec-related infrastructure used by the
//! adapter layer for loading, saving, or normalizing external image formats.

#![cfg(feature = "openslide")]

//! OpenSlide decoder for whole-slide microscopy formats (SVS, NDPI, SCN, MRXS, ...).
//!
//! The upstream OpenSlide API is path-based, while [`ImageDecoder`] consumes byte
//! slices. For direct decoder usage we persist a content-addressed runtime copy
//! under `target/openslide-runtime/` and cache the opened handle across region
//! reads so streaming tile decodes do not rewrite the slide on every call.

use std::{
    collections::{HashMap, hash_map::DefaultHasher},
    fs,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    sync::{Arc, OnceLock, RwLock},
};

use openslide_rs::{Address, OpenSlide, Region as OpenSlideRegion, Size};

use crate::{
    domain::{
        codec_options::LoadOptions,
        error::{OpenSlideCodecError, ViprsError},
        format::{BandFormat, BandFormatId},
        image::{Image, ImageMetadata, Interpretation, Region},
    },
    ports::codec::{ImageDecoder, ImageMetadataProbe, TileImageDecoder},
};

const OPENSLIDE_RUNTIME_DIR: &str = "target/openslide-runtime";
const OPENSLIDE_BANDS: u32 = 4;
const OPENSLIDE_MAX_RGBA_ALLOCATION_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// File extensions recognized as whole-slide images (decoded via OpenSlide).
pub const OPENSLIDE_EXTENSIONS: &[&str] =
    &["svs", "vms", "vmu", "ndpi", "scn", "mrxs", "svslide", "bif"];

type SlideCache = HashMap<SlideCacheKey, Arc<CachedSlide>>;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct SlideCacheKey {
    hash: u64,
    len: usize,
}

#[derive(Debug)]
struct CachedSlide {
    slide: OpenSlide,
    _path: PathBuf,
}

#[derive(Debug, Clone, Copy)]
struct SelectedLevel {
    index: u32,
    size: Size,
    downsample: f64,
}

#[derive(Debug, Clone, Copy, Default)]
/// The `OpenSlideDecoder` type provides concrete adapter functionality in the `codecs` module.
/// Use this type when you need the runtime behavior implemented by this adapter.
///
/// # Examples
///
/// ```rust
/// let _ = core::mem::size_of::<viprs::adapters::codecs::openslide::OpenSlideDecoder>();
/// ```
pub struct OpenSlideDecoder;

fn require_u8<F: BandFormat>() -> Result<(), ViprsError> {
    if F::ID != BandFormatId::U8 {
        return Err(ViprsError::Codec(format!(
            "openslide: unsupported format {:?}; only U8 is supported",
            F::ID
        )));
    }

    Ok(())
}

fn openslide_cache() -> &'static RwLock<SlideCache> {
    static CACHE: OnceLock<RwLock<SlideCache>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

fn cache_key(src: &[u8]) -> SlideCacheKey {
    let mut hasher = DefaultHasher::new();
    src.hash(&mut hasher);
    SlideCacheKey {
        hash: hasher.finish(),
        len: src.len(),
    }
}

fn runtime_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(OPENSLIDE_RUNTIME_DIR)
}

fn runtime_slide_path(key: SlideCacheKey) -> PathBuf {
    runtime_dir().join(format!(
        "openslide-{hash:016x}-{len}.wsi",
        hash = key.hash,
        len = key.len
    ))
}

fn open_slide(path: &Path) -> Result<OpenSlide, ViprsError> {
    OpenSlide::new(path)
        .map_err(|err| ViprsError::Codec(format!("openslide: open '{}': {err}", path.display())))
}

fn cached_slide_from_bytes(src: &[u8]) -> Result<Arc<CachedSlide>, ViprsError> {
    let key = cache_key(src);
    if let Some(slide) = openslide_cache()
        .read()
        .map_err(|_| ViprsError::Codec("openslide: runtime cache poisoned".into()))?
        .get(&key)
        .cloned()
    {
        return Ok(slide);
    }

    let dir = runtime_dir();
    fs::create_dir_all(&dir).map_err(|err| {
        ViprsError::Codec(format!(
            "openslide: failed to create runtime directory '{}': {err}",
            dir.display()
        ))
    })?;

    let path = runtime_slide_path(key);
    if !path.exists() {
        fs::write(&path, src).map_err(|err| {
            ViprsError::Codec(format!(
                "openslide: failed to persist runtime slide '{}': {err}",
                path.display()
            ))
        })?;
    }

    let cached = Arc::new(CachedSlide {
        slide: open_slide(&path)?,
        _path: path,
    });
    openslide_cache()
        .write()
        .map_err(|_| ViprsError::Codec("openslide: runtime cache poisoned".into()))?
        .insert(key, Arc::clone(&cached));
    Ok(cached)
}

fn requested_downsample(root: Size, opts: &LoadOptions) -> f64 {
    if let Some(factor) = opts.shrink_factor.map(|factor| factor.get()) {
        if factor > 1 {
            return f64::from(factor);
        }
    }

    if let Some(max_dimension) = opts.max_dimension.filter(|value| *value > 0) {
        let longest = f64::from(root.w.max(root.h));
        return (longest / f64::from(max_dimension)).max(1.0);
    }

    1.0
}

fn select_level(slide: &OpenSlide, opts: &LoadOptions) -> Result<SelectedLevel, ViprsError> {
    let root = slide
        .get_level_dimensions(0)
        .map_err(|err| ViprsError::Codec(format!("openslide: level 0 dimensions: {err}")))?;
    let requested = requested_downsample(root, opts);
    let index = if requested > 1.0 {
        slide
            .get_best_level_for_downsample(requested)
            .map_err(|err| {
                ViprsError::Codec(format!(
                    "openslide: best level for downsample {requested}: {err}"
                ))
            })?
    } else {
        0
    };
    let size = slide
        .get_level_dimensions(index)
        .map_err(|err| ViprsError::Codec(format!("openslide: level {index} dimensions: {err}")))?;
    let downsample = slide
        .get_level_downsample(index)
        .map_err(|err| ViprsError::Codec(format!("openslide: level {index} downsample: {err}")))?;

    validate_selected_level(SelectedLevel {
        index,
        size,
        downsample,
    })
}

fn validate_selected_level(selected: SelectedLevel) -> Result<SelectedLevel, ViprsError> {
    if selected.size.w == 0 || selected.size.h == 0 {
        return Err(OpenSlideCodecError::ZeroSizedLevel {
            level: selected.index,
            width: selected.size.w,
            height: selected.size.h,
        }
        .into());
    }

    if !selected.downsample.is_finite() || selected.downsample <= 0.0 {
        return Err(OpenSlideCodecError::InvalidLevelDownsample {
            level: selected.index,
            downsample: selected.downsample,
        }
        .into());
    }

    Ok(selected)
}

fn pixels_per_mm_from_mpp(axis: &'static str, microns_per_pixel: f64) -> Result<f64, ViprsError> {
    if !microns_per_pixel.is_finite() || microns_per_pixel <= 0.0 {
        return Err(OpenSlideCodecError::InvalidMicronsPerPixel {
            axis,
            microns_per_pixel,
        }
        .into());
    }

    Ok(1_000.0 / microns_per_pixel)
}

fn metadata_from_slide(
    slide: &OpenSlide,
    selected: SelectedLevel,
) -> Result<ImageMetadata, ViprsError> {
    let properties = slide.properties();
    let openslide = &properties.openslide_properties;
    let mut metadata = ImageMetadata {
        interpretation: Some(Interpretation::Srgb),
        xres: openslide
            .mpp_x
            .map(|mpp| pixels_per_mm_from_mpp("openslide.mpp-x", f64::from(mpp)))
            .transpose()?,
        yres: openslide
            .mpp_y
            .map(|mpp| pixels_per_mm_from_mpp("openslide.mpp-y", f64::from(mpp)))
            .transpose()?,
        ..ImageMetadata::default()
    };

    if let Some(vendor) = openslide.vendor.as_ref() {
        metadata
            .extra
            .insert("openslide.vendor".into(), vendor.clone());
    }
    if let Some(objective_power) = openslide.objective_power {
        metadata.extra.insert(
            "openslide.objective-power".into(),
            objective_power.to_string(),
        );
    }
    if let Some(quickhash) = openslide.quickhash_1.as_ref() {
        metadata
            .extra
            .insert("openslide.quickhash-1".into(), quickhash.clone());
    }
    if let Some(background) = openslide.background_color.as_ref() {
        metadata
            .extra
            .insert("openslide.background-color".into(), background.clone());
    }
    if let Some(icc_size) = openslide.icc_profile_size {
        metadata
            .extra
            .insert("openslide.icc-size".into(), icc_size.to_string());
    }
    if let Some(level_count) = openslide.level_count {
        metadata
            .extra
            .insert("openslide.level-count".into(), level_count.to_string());
    }
    metadata.extra.insert(
        "openslide.selected-level".into(),
        selected.index.to_string(),
    );
    metadata.extra.insert(
        "openslide.selected-downsample".into(),
        selected.downsample.to_string(),
    );
    metadata.extra.insert(
        "openslide.selected-width".into(),
        selected.size.w.to_string(),
    );
    metadata.extra.insert(
        "openslide.selected-height".into(),
        selected.size.h.to_string(),
    );

    for (index, level) in openslide.levels.iter().enumerate() {
        if let Some(width) = level.width {
            metadata
                .extra
                .insert(format!("openslide.level[{index}].width"), width.to_string());
        }
        if let Some(height) = level.height {
            metadata.extra.insert(
                format!("openslide.level[{index}].height"),
                height.to_string(),
            );
        }
        if let Some(downsample) = level.downsample {
            metadata.extra.insert(
                format!("openslide.level[{index}].downsample"),
                downsample.to_string(),
            );
        }
        if let Some(tile_width) = level.tile_width {
            metadata.extra.insert(
                format!("openslide.level[{index}].tile-width"),
                tile_width.to_string(),
            );
        }
        if let Some(tile_height) = level.tile_height {
            metadata.extra.insert(
                format!("openslide.level[{index}].tile-height"),
                tile_height.to_string(),
            );
        }
    }

    if let Ok(names) = slide.get_associated_image_names() {
        metadata
            .extra
            .insert("openslide.associated-images".into(), names.join(","));
    }

    Ok(metadata)
}

fn validate_output_len<F: BandFormat>(region: Region, output: &[u8]) -> Result<(), ViprsError> {
    let expected = checked_openslide_output_len::<F>(region.width, region.height)?;
    if output.len() != expected {
        return Err(ViprsError::Codec(format!(
            "openslide: output buffer size mismatch (got {}, expected {expected})",
            output.len()
        )));
    }
    Ok(())
}

fn checked_openslide_rgba_allocation_len(width: u32, height: u32) -> Result<usize, ViprsError> {
    let bytes = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixel_count| pixel_count.checked_mul(u64::from(OPENSLIDE_BANDS)))
        .ok_or_else(|| {
            let total_bytes = u128::from(width) * u128::from(height) * u128::from(OPENSLIDE_BANDS);
            ViprsError::ImageTooLarge {
                width,
                height,
                bands: OPENSLIDE_BANDS,
                bytes: total_bytes,
                limit_bytes: u128::from(OPENSLIDE_MAX_RGBA_ALLOCATION_BYTES),
                details: "openslide: full level decode RGBA allocation exceeds safe limit",
            }
        })?;
    if bytes > OPENSLIDE_MAX_RGBA_ALLOCATION_BYTES {
        return Err(ViprsError::ImageTooLarge {
            width,
            height,
            bands: OPENSLIDE_BANDS,
            bytes: u128::from(bytes),
            limit_bytes: u128::from(OPENSLIDE_MAX_RGBA_ALLOCATION_BYTES),
            details: "openslide: full level decode RGBA allocation exceeds safe limit",
        });
    }

    usize::try_from(bytes).map_err(|_| ViprsError::ImageTooLarge {
        width,
        height,
        bands: OPENSLIDE_BANDS,
        bytes: u128::from(bytes),
        limit_bytes: u128::try_from(usize::MAX).unwrap_or(u128::MAX),
        details: "openslide: full level decode RGBA allocation exceeds platform addressable memory",
    })
}

fn checked_openslide_output_len<F: BandFormat>(
    width: u32,
    height: u32,
) -> Result<usize, ViprsError> {
    checked_openslide_rgba_allocation_len(width, height).and_then(|samples| {
        samples
            .checked_mul(std::mem::size_of::<F::Sample>())
            .ok_or_else(|| ViprsError::Codec("openslide: output buffer length overflow".into()))
    })
}

fn clamp_coord(origin: i32, offset: u32, limit: u32) -> i32 {
    let position = i64::from(origin) + i64::from(offset);
    position.clamp(0, i64::from(limit) - 1) as i32
}

fn level0_coordinate(position: i32, downsample: f64) -> u32 {
    ((f64::from(position)) * downsample).round().max(0.0) as u32
}

fn decode_region_from_slide(
    slide: &OpenSlide,
    selected: SelectedLevel,
    region: Region,
    output: &mut [u8],
) -> Result<(), ViprsError> {
    validate_output_len::<crate::domain::format::U8>(region, output)?;
    if region.width == 0 || region.height == 0 {
        return Ok(());
    }

    let level_width = selected.size.w;
    let level_height = selected.size.h;
    if level_width == 0 || level_height == 0 {
        return Err(OpenSlideCodecError::ZeroSizedLevel {
            level: selected.index,
            width: level_width,
            height: level_height,
        }
        .into());
    }

    let min_x = clamp_coord(region.x, 0, level_width);
    let min_y = clamp_coord(region.y, 0, level_height);
    let max_x = clamp_coord(region.x, region.width.saturating_sub(1), level_width);
    let max_y = clamp_coord(region.y, region.height.saturating_sub(1), level_height);
    let read_width = u32::try_from(max_x - min_x + 1)
        .map_err(|_| ViprsError::Codec("openslide: invalid region width".into()))?;
    let read_height = u32::try_from(max_y - min_y + 1)
        .map_err(|_| ViprsError::Codec("openslide: invalid region height".into()))?;

    let read_region = OpenSlideRegion {
        address: Address {
            x: level0_coordinate(min_x, selected.downsample),
            y: level0_coordinate(min_y, selected.downsample),
        },
        level: selected.index,
        size: Size {
            w: read_width,
            h: read_height,
        },
    };
    let pixels = slide
        .read_region(&read_region)
        .map_err(|err| ViprsError::Codec(format!("openslide: read_region: {err}")))?;

    for out_y in 0..region.height {
        let src_y = usize::try_from(clamp_coord(region.y, out_y, level_height) - min_y)
            .map_err(|_| ViprsError::Codec("openslide: invalid clamped y".into()))?;
        for out_x in 0..region.width {
            let src_x = usize::try_from(clamp_coord(region.x, out_x, level_width) - min_x)
                .map_err(|_| ViprsError::Codec("openslide: invalid clamped x".into()))?;
            let src = (src_y * read_width as usize + src_x) * 4;
            let dst = (out_y as usize * region.width as usize + out_x as usize) * 4;
            output[dst] = pixels[src + 2];
            output[dst + 1] = pixels[src + 1];
            output[dst + 2] = pixels[src];
            output[dst + 3] = pixels[src + 3];
        }
    }

    Ok(())
}

fn decode_slide_with_options<F: BandFormat>(
    slide: &OpenSlide,
    opts: &LoadOptions,
) -> Result<Image<F>, ViprsError> {
    require_u8::<F>()?;
    let selected = select_level(slide, opts)?;
    let region = Region::new(0, 0, selected.size.w, selected.size.h);
    let rgba_len = checked_openslide_rgba_allocation_len(selected.size.w, selected.size.h)?;
    let mut rgba = vec![0u8; rgba_len];
    decode_region_from_slide(slide, selected, region, &mut rgba)?;
    let typed =
        bytemuck::allocation::try_cast_vec::<u8, F::Sample>(rgba).map_err(|(_err, _rgba)| {
            ViprsError::Codec(format!(
                "openslide: failed to cast decoded samples into {:?}",
                F::ID
            ))
        })?;
    let metadata = metadata_from_slide(slide, selected)?;
    Image::from_buffer(selected.size.w, selected.size.h, OPENSLIDE_BANDS, typed)
        .map(|image| image.with_metadata(metadata))
        .map_err(|err| ViprsError::Codec(format!("openslide: {err}")))
}

fn probe_slide_with_options(
    slide: &OpenSlide,
    opts: &LoadOptions,
) -> Result<ImageMetadataProbe, ViprsError> {
    let selected = select_level(slide, opts)?;
    let metadata = metadata_from_slide(slide, selected)?;
    Ok(
        ImageMetadataProbe::new(selected.size.w, selected.size.h, OPENSLIDE_BANDS)
            .with_metadata(metadata),
    )
}

impl OpenSlideDecoder {
    #[must_use]
    /// `supports_path` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let _ = viprs::adapters::codecs::openslide::supports_path;
    /// ```
    pub fn supports_path(path: &Path) -> bool {
        path.extension()
            .and_then(std::ffi::OsStr::to_str)
            .is_some_and(|extension| OPENSLIDE_EXTENSIONS.contains(&extension))
    }

    #[must_use]
    /// `can_open_path` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let _ = viprs::adapters::codecs::openslide::can_open_path;
    /// ```
    pub fn can_open_path(path: &Path) -> bool {
        if Self::supports_path(path) {
            return true;
        }

        path.extension()
            .and_then(std::ffi::OsStr::to_str)
            .is_some_and(|extension| matches!(extension, "tif" | "tiff"))
            && OpenSlide::detect_vendor(path).is_ok()
    }
}

impl ImageDecoder for OpenSlideDecoder {
    fn format_name(&self) -> &'static str {
        "openslide"
    }

    fn can_decode_path(&self, path: &Path) -> bool {
        Self::can_open_path(path)
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
        opts: &LoadOptions,
    ) -> Result<Image<F>, ViprsError>
    where
        Self: Sized,
    {
        let cached = cached_slide_from_bytes(src)?;
        decode_slide_with_options::<F>(&cached.slide, opts)
    }

    fn decode_path_with_options<F: BandFormat>(
        &self,
        path: &Path,
        opts: &LoadOptions,
    ) -> Result<Image<F>, ViprsError>
    where
        Self: Sized,
    {
        let slide = open_slide(path)?;
        decode_slide_with_options::<F>(&slide, opts)
    }

    fn probe(&self, src: &[u8]) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized,
    {
        let info = self.probe_with_options(src, &LoadOptions::default())?;
        Ok((info.width, info.height, info.bands))
    }

    fn probe_path(&self, path: &Path) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized,
    {
        let info = self.probe_path_with_options(path, &LoadOptions::default())?;
        Ok((info.width, info.height, info.bands))
    }
}

impl TileImageDecoder for OpenSlideDecoder {
    fn probe_with_options(
        &self,
        src: &[u8],
        opts: &LoadOptions,
    ) -> Result<ImageMetadataProbe, ViprsError>
    where
        Self: Sized,
    {
        let cached = cached_slide_from_bytes(src)?;
        probe_slide_with_options(&cached.slide, opts)
    }

    fn probe_path_with_options(
        &self,
        path: &Path,
        opts: &LoadOptions,
    ) -> Result<ImageMetadataProbe, ViprsError>
    where
        Self: Sized,
    {
        let slide = open_slide(path)?;
        probe_slide_with_options(&slide, opts)
    }

    fn decode_region_into<F: BandFormat>(
        &self,
        src: &[u8],
        opts: &LoadOptions,
        region: Region,
        output: &mut [u8],
    ) -> Result<(), ViprsError>
    where
        Self: Sized,
    {
        require_u8::<F>()?;
        let cached = cached_slide_from_bytes(src)?;
        let selected = select_level(&cached.slide, opts)?;
        decode_region_from_slide(&cached.slide, selected, region, output)
    }

    fn decode_region_from_path<F: BandFormat>(
        &self,
        path: &Path,
        opts: &LoadOptions,
        region: Region,
        output: &mut [u8],
    ) -> Result<(), ViprsError>
    where
        Self: Sized,
    {
        require_u8::<F>()?;
        let slide = open_slide(path)?;
        let selected = select_level(&slide, opts)?;
        decode_region_from_slide(&slide, selected, region, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU8;

    #[test]
    fn requested_downsample_prefers_shrink_factor() {
        let root = Size { w: 4000, h: 2000 };
        let opts = LoadOptions::default()
            .with_max_dimension(500)
            .with_shrink(NonZeroU8::new(4).expect("non-zero"));
        assert_eq!(requested_downsample(root, &opts), 4.0);
    }

    #[test]
    fn requested_downsample_uses_max_dimension_when_present() {
        let root = Size { w: 4000, h: 2000 };
        let opts = LoadOptions::default().with_max_dimension(1000);
        assert_eq!(requested_downsample(root, &opts), 4.0);
    }

    #[test]
    fn supports_known_wsi_extensions() {
        assert!(OpenSlideDecoder::supports_path(Path::new("slide.svs")));
        assert!(OpenSlideDecoder::supports_path(Path::new("slide.mrxs")));
        assert!(!OpenSlideDecoder::supports_path(Path::new("slide.png")));
    }

    #[test]
    fn test_openslide_full_decode_allocation_rejects_huge_dimensions() {
        let err = checked_openslide_rgba_allocation_len(u32::MAX, u32::MAX)
            .expect_err("huge RGBA allocation must be rejected");

        match err {
            ViprsError::ImageTooLarge {
                width,
                height,
                bands,
                bytes,
                limit_bytes,
                details,
            } => {
                assert_eq!(width, u32::MAX);
                assert_eq!(height, u32::MAX);
                assert_eq!(bands, OPENSLIDE_BANDS);
                assert!(bytes > limit_bytes);
                assert!(details.contains("full level decode"));
            }
            other => panic!("expected ImageTooLarge, got {other:?}"),
        }
    }

    #[test]
    fn test_selected_level_with_zero_width_is_rejected() {
        let selected = SelectedLevel {
            index: 2,
            size: Size { w: 0, h: 512 },
            downsample: 4.0,
        };

        let err = validate_selected_level(selected).expect_err("zero-sized level must error");
        assert!(matches!(
            err,
            ViprsError::OpenSlide(crate::domain::error::OpenSlideCodecError::ZeroSizedLevel {
                level: 2,
                width: 0,
                height: 512,
            })
        ));
    }

    #[test]
    fn test_selected_level_with_zero_downsample_is_rejected() {
        let selected = SelectedLevel {
            index: 3,
            size: Size { w: 512, h: 512 },
            downsample: 0.0,
        };

        assert!(matches!(
            validate_selected_level(selected),
            Err(ViprsError::OpenSlide(
                crate::domain::error::OpenSlideCodecError::InvalidLevelDownsample {
                    level: 3,
                    downsample,
                }
            )) if downsample == 0.0
        ));
    }

    #[test]
    fn test_selected_level_with_negative_downsample_is_rejected() {
        let selected = SelectedLevel {
            index: 4,
            size: Size { w: 512, h: 512 },
            downsample: -2.0,
        };

        assert!(matches!(
            validate_selected_level(selected),
            Err(ViprsError::OpenSlide(
                crate::domain::error::OpenSlideCodecError::InvalidLevelDownsample {
                    level: 4,
                    downsample,
                }
            )) if downsample == -2.0
        ));
    }

    #[test]
    fn test_selected_level_with_nan_downsample_is_rejected() {
        let selected = SelectedLevel {
            index: 5,
            size: Size { w: 512, h: 512 },
            downsample: f64::NAN,
        };

        assert!(matches!(
            validate_selected_level(selected),
            Err(ViprsError::OpenSlide(
                crate::domain::error::OpenSlideCodecError::InvalidLevelDownsample {
                    level: 5,
                    downsample,
                }
            )) if downsample.is_nan()
        ));
    }

    #[test]
    fn test_selected_level_with_infinite_downsample_is_rejected() {
        let selected = SelectedLevel {
            index: 6,
            size: Size { w: 512, h: 512 },
            downsample: f64::INFINITY,
        };

        assert!(matches!(
            validate_selected_level(selected),
            Err(ViprsError::OpenSlide(
                crate::domain::error::OpenSlideCodecError::InvalidLevelDownsample {
                    level: 6,
                    downsample,
                }
            )) if downsample.is_infinite()
        ));
    }

    #[test]
    fn zero_mpp_is_rejected() {
        assert!(matches!(
            pixels_per_mm_from_mpp("openslide.mpp-x", 0.0),
            Err(ViprsError::OpenSlide(
                crate::domain::error::OpenSlideCodecError::InvalidMicronsPerPixel {
                    axis,
                    microns_per_pixel,
                }
            )) if axis == "openslide.mpp-x" && microns_per_pixel == 0.0
        ));
    }

    #[test]
    fn non_finite_mpp_is_rejected() {
        assert!(matches!(
            pixels_per_mm_from_mpp("openslide.mpp-y", f64::NAN),
            Err(ViprsError::OpenSlide(
                crate::domain::error::OpenSlideCodecError::InvalidMicronsPerPixel {
                    axis,
                    microns_per_pixel,
                }
            )) if axis == "openslide.mpp-y" && microns_per_pixel.is_nan()
        ));
    }

    #[test]
    fn negative_mpp_is_rejected() {
        assert!(matches!(
            pixels_per_mm_from_mpp("openslide.mpp-x", -0.5),
            Err(ViprsError::OpenSlide(
                crate::domain::error::OpenSlideCodecError::InvalidMicronsPerPixel {
                    axis,
                    microns_per_pixel,
                }
            )) if axis == "openslide.mpp-x" && microns_per_pixel == -0.5
        ));
    }
}

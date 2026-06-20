use crate::format::{BandFormat, U8};

use super::core::Image;

/// Runtime interpretation of an image buffer.
///
/// Interpretation tracks how consumers should read the same band data, such as sRGB, Lab, or
/// histogram output.
///
/// # Examples
/// ```rust
/// # use viprs_core::image::Interpretation;
/// assert_eq!(Interpretation::Rgb.max_alpha(), 255.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpretation {
    /// Uses the `Multiband` variant of `Interpretation`.
    Multiband,
    /// Uses the `BW` variant of `Interpretation`.
    BW,
    /// Uses the `Histogram` variant of `Interpretation`.
    Histogram,
    /// Uses the `Xyz` variant of `Interpretation`.
    Xyz,
    /// Uses the `Lab` variant of `Interpretation`.
    Lab,
    /// Uses the `Cmyk` variant of `Interpretation`.
    Cmyk,
    /// Uses the `Labq` variant of `Interpretation`.
    Labq,
    /// Uses the `Rgb` variant of `Interpretation`.
    Rgb,
    /// Uses the `Cmc` variant of `Interpretation`.
    Cmc,
    /// Uses the `Lch` variant of `Interpretation`.
    Lch,
    /// Uses the `Labs` variant of `Interpretation`.
    Labs,
    /// Uses the `Srgb` variant of `Interpretation`.
    Srgb,
    /// Uses the `Yxy` variant of `Interpretation`.
    Yxy,
    /// Uses the `Fourier` variant of `Interpretation`.
    Fourier,
    /// Uses the `Rgb16` variant of `Interpretation`.
    Rgb16,
    /// Uses the `Grey16` variant of `Interpretation`.
    Grey16,
    /// Uses the `Matrix` variant of `Interpretation`.
    Matrix,
    /// Uses the `Scrgb` variant of `Interpretation`.
    Scrgb,
    /// Uses the `Hsv` variant of `Interpretation`.
    Hsv,
}

impl Interpretation {
    /// Returns the libvips default maximum alpha for this interpretation.
    #[must_use]
    pub const fn max_alpha(self) -> f64 {
        match self {
            Self::Rgb16 | Self::Grey16 => 65535.0,
            Self::Scrgb => 1.0,
            _ => 255.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Metadata describing how an Ultra HDR gain map should be applied.
pub struct UhdrGainMapMetadata {
    /// Stores the `gamma` value for this item.
    pub gamma: [f32; 3],
    /// Stores the `min_content_boost` value for this item.
    pub min_content_boost: [f32; 3],
    /// Stores the `max_content_boost` value for this item.
    pub max_content_boost: [f32; 3],
    /// Stores the `offset_hdr` value for this item.
    pub offset_hdr: [f32; 3],
    /// Stores the `offset_sdr` value for this item.
    pub offset_sdr: [f32; 3],
}

impl Default for UhdrGainMapMetadata {
    fn default() -> Self {
        Self {
            gamma: [1.0; 3],
            min_content_boost: [1.0; 3],
            max_content_boost: [1.0; 3],
            offset_hdr: [0.0; 3],
            offset_sdr: [0.0; 3],
        }
    }
}

/// Sparse metadata edits applied while copying image pixels unchanged.
///
/// Callers use this to update selected metadata fields without rebuilding a full
/// [`ImageMetadata`] value.
///
/// # Examples
/// ```ignore
/// # use viprs_core::image::{Interpretation, MetadataOverrides};
/// let overrides = MetadataOverrides {
///     interpretation: Some(Interpretation::SRgb),
///     ..MetadataOverrides::default()
/// };
/// assert_eq!(overrides.interpretation, Some(Interpretation::SRgb));
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MetadataOverrides {
    /// Stores the `interpretation` value for this item.
    pub interpretation: Option<Interpretation>,
    /// Stores the `orientation` value for this item.
    pub orientation: Option<u8>,
    /// Stores the `icc_profile` value for this item.
    pub icc_profile: Option<Option<Vec<u8>>>,
    /// Stores the `xres` value for this item.
    pub xres: Option<f64>,
    /// Stores the `yres` value for this item.
    pub yres: Option<f64>,
}

/// Decoded Ultra HDR gain map attached to an SDR base image.
///
/// This bundles the auxiliary gain image and metadata needed to reconstruct HDR output later in
/// the pipeline.
///
/// # Examples
/// ```rust
/// # use viprs_core::{format::U8, image::{Image, UhdrGainMap}};
/// let image = Image::<U8>::from_buffer(1, 1, 1, vec![0]).unwrap();
/// let gain_map = UhdrGainMap {
///     image: Box::new(image),
///     metadata: Default::default(),
///     hdr_capacity_min: 1.0,
///     hdr_capacity_max: 1.0,
///     base_rendition_is_hdr: false,
/// };
/// assert_eq!(gain_map.image().width(), 1);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct UhdrGainMap {
    /// Stores the `image` value for this item.
    pub image: Box<Image<U8>>,
    /// Stores the `metadata` value for this item.
    pub metadata: UhdrGainMapMetadata,
    /// Stores the `hdr_capacity_min` value for this item.
    pub hdr_capacity_min: f32,
    /// Stores the `hdr_capacity_max` value for this item.
    pub hdr_capacity_max: f32,
    /// Stores the `base_rendition_is_hdr` value for this item.
    pub base_rendition_is_hdr: bool,
}

impl UhdrGainMap {
    #[must_use]
    /// Borrow the decoded gain-map image.
    ///
    /// This lets HDR-aware code inspect the auxiliary pixels without taking ownership.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs_core::{format::U8, image::{Image, UhdrGainMap}};
    /// let gain_map = UhdrGainMap {
    ///     image: Box::new(Image::<U8>::from_buffer(1, 1, 1, vec![0]).unwrap()),
    ///     metadata: Default::default(),
    ///     hdr_capacity_min: 1.0,
    ///     hdr_capacity_max: 1.0,
    ///     base_rendition_is_hdr: false,
    /// };
    /// assert_eq!(gain_map.image().bands(), 1);
    /// ```
    pub fn image(&self) -> &Image<U8> {
        self.image.as_ref()
    }

    #[must_use]
    /// Return the parsed Ultra HDR metadata associated with the gain map.
    ///
    /// This keeps conversion code synchronized with the embedded reconstruction parameters.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs_core::{format::U8, image::{Image, UhdrGainMap}};
    /// let gain_map = UhdrGainMap {
    ///     image: Box::new(Image::<U8>::from_buffer(1, 1, 1, vec![0]).unwrap()),
    ///     metadata: Default::default(),
    ///     hdr_capacity_min: 1.0,
    ///     hdr_capacity_max: 1.0,
    ///     base_rendition_is_hdr: false,
    /// };
    /// let _metadata = gain_map.metadata();
    /// ```
    pub const fn metadata(&self) -> UhdrGainMapMetadata {
        self.metadata
    }
}

/// GIF-style disposal semantics for an animation frame.
///
/// Disposal controls how the next frame should reuse or clear the current frame's canvas area.
///
/// # Examples
/// ```rust
/// # use viprs_core::image::FrameDisposal;
/// assert_eq!(FrameDisposal::default(), FrameDisposal::Keep);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrameDisposal {
    /// Decoder-specific default disposal.
    Any,
    /// Leave the frame contents on the canvas.
    #[default]
    Keep,
    /// Clear the frame area back to the background colour.
    Background,
    /// Restore the previous canvas contents after this frame.
    Previous,
}

/// Loop behaviour for animated image playback.
///
/// This captures whether an animation repeats forever or a bounded number of times.
///
/// # Examples
/// ```rust
/// # use viprs_core::image::AnimationLoopCount;
/// let loop_count = AnimationLoopCount::Finite(2);
/// assert!(matches!(loop_count, AnimationLoopCount::Finite(2)));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimationLoopCount {
    /// Repeat forever.
    Infinite,
    /// Repeat exactly `n` times after the first presentation.
    Finite(u16),
}

/// A single decoded or user-authored animation frame.
#[derive(Debug)]
pub struct AnimationFrame<F: BandFormat> {
    image: Box<Image<F>>,
    delay_ms: u32,
    disposal: FrameDisposal,
}

impl<F: BandFormat> AnimationFrame<F> {
    /// Build a typed animation frame around a full in-memory image.
    #[must_use]
    pub fn new(image: Image<F>, delay_ms: u32, disposal: FrameDisposal) -> Self {
        Self {
            image: Box::new(image),
            delay_ms,
            disposal,
        }
    }

    /// Borrow the underlying frame image.
    #[must_use]
    pub fn image(&self) -> &Image<F> {
        self.image.as_ref()
    }

    /// Consume the frame and return the underlying image.
    #[must_use]
    pub fn into_image(self) -> Image<F> {
        *self.image
    }

    /// Delay for this frame in milliseconds.
    #[must_use]
    pub const fn delay_ms(&self) -> u32 {
        self.delay_ms
    }

    /// Disposal method for this frame.
    #[must_use]
    pub const fn disposal(&self) -> FrameDisposal {
        self.disposal
    }
}

impl<F: BandFormat> Clone for AnimationFrame<F>
where
    F::Sample: Clone,
{
    fn clone(&self) -> Self {
        Self {
            image: self.image.clone(),
            delay_ms: self.delay_ms,
            disposal: self.disposal,
        }
    }
}

impl<F: BandFormat> PartialEq for AnimationFrame<F>
where
    F::Sample: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.image == other.image
            && self.delay_ms == other.delay_ms
            && self.disposal == other.disposal
    }
}

/// Metadata associated with an image. Stored alongside pixel data but separate
/// from the `BandFormat` parameter so it can be moved or cloned independently.
#[derive(Debug, Clone, PartialEq, Default)]
/// Metadata associated with an image buffer.
///
/// Keeping metadata separate from the pixel format lets pipelines transform pixels while updating
/// interpretation, orientation, and container-specific side data independently.
///
/// # Examples
/// ```ignore
/// # use viprs_core::image::{ImageMetadata, Interpretation};
/// let metadata = ImageMetadata {
///     interpretation: Some(Interpretation::SRgb),
///     ..ImageMetadata::default()
/// };
/// assert_eq!(metadata.interpretation, Some(Interpretation::SRgb));
/// ```
pub struct ImageMetadata {
    /// Colorspace interpretation (e.g. sRGB, Lab, CMYK). `None` means unknown.
    pub interpretation: Option<Interpretation>,
    /// EXIF orientation tag (1–8). `None` means unspecified.
    pub orientation: Option<u8>,
    /// Raw ICC profile bytes.
    pub icc_profile: Option<Vec<u8>>,
    /// Raw EXIF blob bytes.
    pub exif: Option<Vec<u8>>,
    /// Raw XMP packet bytes.
    pub xmp: Option<Vec<u8>>,
    /// Horizontal resolution in pixels per mm.
    pub xres: Option<f64>,
    /// Vertical resolution in pixels per mm.
    pub yres: Option<f64>,
    /// Page height for vertically stacked multi-page images.
    pub page_height: Option<u32>,
    /// Total number of pages/frames in the source image.
    pub n_pages: Option<u32>,
    /// Animation loop behaviour for formats that support repeated playback.
    pub animation_loop_count: Option<AnimationLoopCount>,
    /// Attached Ultra HDR gain map plus the metadata needed by `uhdr2scrgb`.
    pub uhdr_gainmap: Option<UhdrGainMap>,
    /// Codec-specific metadata that does not map to typed fields.
    pub extra: std::collections::HashMap<String, String>,
}

impl ImageMetadata {
    /// Strip all metadata — EXIF, XMP, ICC profile, orientation, UHDR gainmap, extras.
    /// Preserves only structural fields (interpretation, resolution, page info).
    ///
    /// Use for privacy: removes all identifying information from the image.
    #[must_use]
    pub fn strip_all(&self) -> Self {
        Self {
            interpretation: self.interpretation,
            orientation: None,
            icc_profile: None,
            exif: None,
            xmp: None,
            xres: self.xres,
            yres: self.yres,
            page_height: self.page_height,
            n_pages: self.n_pages,
            animation_loop_count: self.animation_loop_count,
            uhdr_gainmap: None,
            extra: std::collections::HashMap::new(),
        }
    }

    /// Strip metadata but preserve the ICC color profile.
    ///
    /// Use when you need color accuracy (e.g., `ProPhoto` RGB or Adobe RGB workflows)
    /// but want to remove `EXIF`/`XMP`/extras for privacy.
    #[must_use]
    pub fn strip_preserving_icc(&self) -> Self {
        Self {
            interpretation: self.interpretation,
            orientation: None,
            icc_profile: self.icc_profile.clone(),
            exif: None,
            xmp: None,
            xres: self.xres,
            yres: self.yres,
            page_height: self.page_height,
            n_pages: self.n_pages,
            animation_loop_count: self.animation_loop_count,
            uhdr_gainmap: None,
            extra: std::collections::HashMap::new(),
        }
    }

    /// Returns true if this metadata has any EXIF data.
    #[must_use]
    pub fn has_exif(&self) -> bool {
        self.exif.as_ref().is_some_and(|exif| !exif.is_empty())
    }

    /// Returns true if this metadata has an ICC profile.
    #[must_use]
    pub fn has_icc_profile(&self) -> bool {
        self.icc_profile
            .as_ref()
            .is_some_and(|profile| !profile.is_empty())
    }

    /// Returns true if this metadata has XMP data.
    #[must_use]
    pub fn has_xmp(&self) -> bool {
        self.xmp.as_ref().is_some_and(|xmp| !xmp.is_empty())
    }

    /// Remove the orientation tag from typed metadata and embedded EXIF bytes.
    ///
    /// This is useful after physically rotating pixels so the image does not get rotated twice.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs_core::image::ImageMetadata;
    /// let mut metadata = ImageMetadata { orientation: Some(6), ..ImageMetadata::default() };
    /// metadata.remove_orientation();
    /// assert_eq!(metadata.orientation, None);
    /// ```
    pub fn remove_orientation(&mut self) {
        self.orientation = None;
        if let Some(exif) = self.exif.as_mut() {
            remove_exif_ifd0_orientation(exif);
        }
    }

    #[must_use]
    /// Merge sparse metadata overrides onto this metadata set.
    ///
    /// This supports copy-style operations that keep pixels unchanged while editing selected tags.
    ///
    /// # Examples
    /// ```ignore
    /// # use viprs_core::image::{ImageMetadata, Interpretation, MetadataOverrides};
    /// let metadata = ImageMetadata::default();
    /// let merged = metadata.merge_overrides(&MetadataOverrides {
    ///     interpretation: Some(Interpretation::SRgb),
    ///     ..MetadataOverrides::default()
    /// });
    /// assert_eq!(merged.interpretation, Some(Interpretation::SRgb));
    /// ```
    pub fn merge_overrides(&self, overrides: &MetadataOverrides) -> Self {
        let mut merged = self.clone();

        if let Some(interpretation) = overrides.interpretation {
            merged.interpretation = Some(interpretation);
        }
        if let Some(orientation) = overrides.orientation {
            merged.orientation = Some(orientation);
        }
        if let Some(icc_profile) = &overrides.icc_profile {
            merged.icc_profile.clone_from(icc_profile);
        }
        if let Some(xres) = overrides.xres {
            merged.xres = Some(xres);
        }
        if let Some(yres) = overrides.yres {
            merged.yres = Some(yres);
        }

        merged
    }

    #[must_use]
    /// Borrow the attached Ultra HDR gain map, if present.
    ///
    /// This exposes HDR reconstruction metadata to later pipeline stages without cloning it.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs_core::image::ImageMetadata;
    /// let metadata = ImageMetadata::default();
    /// assert!(metadata.uhdr_gainmap().is_none());
    /// ```
    pub const fn uhdr_gainmap(&self) -> Option<&UhdrGainMap> {
        self.uhdr_gainmap.as_ref()
    }
}

const EXIF_SIGNATURE: &[u8] = b"Exif\0\0";
const TIFF_MAGIC: u16 = 42;
const EXIF_TAG_ORIENTATION: u16 = 0x0112;
const EXIF_FIELD_TYPE_SHORT: u16 = 3;
const EXIF_ORIENTATION_COMPONENTS: u32 = 1;
const TIFF_ENTRY_BYTES: usize = 12;
const TIFF_NEXT_IFD_BYTES: usize = 4;

#[derive(Clone, Copy)]
enum TiffEndian {
    Little,
    Big,
}

fn remove_exif_ifd0_orientation(exif: &mut [u8]) -> bool {
    let Some(tiff_start) = exif_tiff_start(exif) else {
        return false;
    };
    let tiff = &mut exif[tiff_start..];
    if tiff.len() < 8 {
        return false;
    }

    let endian = match &tiff[..2] {
        b"II" => TiffEndian::Little,
        b"MM" => TiffEndian::Big,
        _ => return false,
    };
    if read_tiff_u16(&tiff[2..4], endian) != Some(TIFF_MAGIC) {
        return false;
    }

    let Some(ifd0_offset) = read_tiff_u32(&tiff[4..8], endian).map(|offset| offset as usize) else {
        return false;
    };
    let Some(entry_count_bytes) = tiff.get(ifd0_offset..ifd0_offset.saturating_add(2)) else {
        return false;
    };
    let Some(entry_count) = read_tiff_u16(entry_count_bytes, endian).map(usize::from) else {
        return false;
    };
    let Some(entries_start) = ifd0_offset.checked_add(2) else {
        return false;
    };
    let Some(entries_bytes) = entry_count.checked_mul(TIFF_ENTRY_BYTES) else {
        return false;
    };
    let Some(next_ifd_offset) = entries_start.checked_add(entries_bytes) else {
        return false;
    };
    if next_ifd_offset
        .checked_add(TIFF_NEXT_IFD_BYTES)
        .is_none_or(|end| end > tiff.len())
    {
        return false;
    }

    for entry_index in 0..entry_count {
        let entry_offset = entries_start + entry_index * TIFF_ENTRY_BYTES;
        let Some(entry) = tiff.get(entry_offset..entry_offset + TIFF_ENTRY_BYTES) else {
            return false;
        };
        if !is_ifd0_orientation_entry(entry, endian) {
            continue;
        }

        remove_tiff_ifd_entry(
            tiff,
            ifd0_offset,
            entries_start,
            entry_count,
            entry_index,
            endian,
        );
        return true;
    }

    false
}

fn exif_tiff_start(exif: &[u8]) -> Option<usize> {
    if exif.starts_with(EXIF_SIGNATURE) {
        Some(EXIF_SIGNATURE.len())
    } else if matches!(exif.get(..2), Some(b"II" | b"MM")) {
        Some(0)
    } else {
        None
    }
}

fn is_ifd0_orientation_entry(entry: &[u8], endian: TiffEndian) -> bool {
    read_tiff_u16(&entry[0..2], endian) == Some(EXIF_TAG_ORIENTATION)
        && read_tiff_u16(&entry[2..4], endian) == Some(EXIF_FIELD_TYPE_SHORT)
        && read_tiff_u32(&entry[4..8], endian) == Some(EXIF_ORIENTATION_COMPONENTS)
}

fn remove_tiff_ifd_entry(
    tiff: &mut [u8],
    ifd_offset: usize,
    entries_start: usize,
    entry_count: usize,
    entry_index: usize,
    endian: TiffEndian,
) {
    write_tiff_u16(
        &mut tiff[ifd_offset..ifd_offset + 2],
        (entry_count - 1) as u16,
        endian,
    );

    let entry_offset = entries_start + entry_index * TIFF_ENTRY_BYTES;
    let after_entry = entry_offset + TIFF_ENTRY_BYTES;
    let old_ifd_end = entries_start + entry_count * TIFF_ENTRY_BYTES + TIFF_NEXT_IFD_BYTES;
    tiff.copy_within(after_entry..old_ifd_end, entry_offset);
    tiff[old_ifd_end - TIFF_ENTRY_BYTES..old_ifd_end].fill(0);
}

fn read_tiff_u16(bytes: &[u8], endian: TiffEndian) -> Option<u16> {
    let pair: [u8; 2] = bytes.get(..2)?.try_into().ok()?;
    Some(match endian {
        TiffEndian::Little => u16::from_le_bytes(pair),
        TiffEndian::Big => u16::from_be_bytes(pair),
    })
}

fn read_tiff_u32(bytes: &[u8], endian: TiffEndian) -> Option<u32> {
    let quad: [u8; 4] = bytes.get(..4)?.try_into().ok()?;
    Some(match endian {
        TiffEndian::Little => u32::from_le_bytes(quad),
        TiffEndian::Big => u32::from_be_bytes(quad),
    })
}

fn write_tiff_u16(bytes: &mut [u8], value: u16, endian: TiffEndian) {
    let encoded = match endian {
        TiffEndian::Little => value.to_le_bytes(),
        TiffEndian::Big => value.to_be_bytes(),
    };
    bytes[..2].copy_from_slice(&encoded);
}

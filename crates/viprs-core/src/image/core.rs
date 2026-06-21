use crate::{error::ViprsError, format::BandFormat};

use super::metadata::{AnimationFrame, AnimationLoopCount, ImageMetadata};

/// A fully-decoded image stored in a contiguous row-major pixel buffer.
///
/// `Clone` and `Debug` are implemented manually: `Debug` prints dimensions and
/// metadata (not the pixel data, which can be gigabytes), and `Clone` performs
/// an explicit buffer copy with no hidden cost.
pub struct Image<F: BandFormat> {
    width: u32,
    height: u32,
    bands: u32,
    data: Vec<F::Sample>,
    metadata: ImageMetadata,
    frames: Option<Vec<Self>>,
    animation_frames: Option<Vec<AnimationFrame<F>>>,
}

impl<F: BandFormat> Image<F> {
    /// Construct an image from an existing pixel buffer.
    ///
    /// Returns [`ViprsError::ImageTooLarge`] if the dimensions exceed
    /// addressable memory, or [`ViprsError::RegionOutOfBounds`] if the buffer
    /// length does not match `width * height * bands`.
    pub fn from_buffer(
        width: u32,
        height: u32,
        bands: u32,
        data: Vec<F::Sample>,
    ) -> Result<Self, ViprsError> {
        let expected = checked_image_buffer_len(width, height, bands)?;
        if data.len() != expected {
            return Err(ViprsError::RegionOutOfBounds {
                requested: format!(
                    "buffer length {} does not match {}x{}x{}={}",
                    data.len(),
                    width,
                    height,
                    bands,
                    expected
                ),
                width,
                height,
            });
        }
        Ok(Self {
            width,
            height,
            bands,
            data,
            metadata: ImageMetadata::default(),
            frames: None,
            animation_frames: None,
        })
    }

    /// Construct an animated image from typed frames.
    ///
    /// All frames must share the same width, height, and band count.
    pub fn from_frames(frames: Vec<AnimationFrame<F>>) -> Result<Self, ViprsError>
    where
        F::Sample: Clone,
    {
        let Some(first) = frames.first() else {
            return Err(ViprsError::Codec(
                "image: animation sequence must contain at least one frame".into(),
            ));
        };

        let width = first.image().width();
        let height = first.image().height();
        let bands = first.image().bands();
        for (frame_index, frame) in frames.iter().enumerate() {
            if frame.image().width() != width
                || frame.image().height() != height
                || frame.image().bands() != bands
            {
                return Err(ViprsError::Codec(format!(
                    "image: animation frame {frame_index} has shape {}x{}x{}, expected {width}x{height}x{bands}",
                    frame.image().width(),
                    frame.image().height(),
                    frame.image().bands(),
                )));
            }
        }

        let mut image = first.image().clone();
        image.frames = Some(frames.iter().map(|frame| frame.image().clone()).collect());
        image.animation_frames = Some(frames);
        image.metadata.n_pages = image.frames.as_ref().map(|frames| frames.len() as u32);
        image.metadata.page_height = (image.metadata.n_pages.unwrap_or(1) > 1).then_some(height);
        Ok(image)
    }

    #[must_use]
    /// Replace the image metadata while leaving pixels unchanged.
    ///
    /// This supports builder-style image construction after decoding or processing.
    ///
    /// # Examples
    /// ```ignore
    /// # use viprs_core::{format::U8, image::{Image, ImageMetadata, Interpretation}};
    /// let image = Image::<U8>::from_buffer(1, 1, 1, vec![0]).unwrap()
    ///     .with_metadata(ImageMetadata { interpretation: Some(Interpretation::SRgb), ..ImageMetadata::default() });
    /// assert_eq!(image.metadata().interpretation, Some(Interpretation::SRgb));
    /// ```
    pub fn with_metadata(mut self, metadata: ImageMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    #[must_use]
    /// Attach legacy frame images to this image.
    ///
    /// This keeps compatibility with multi-page consumers that expect `frames()` rather than full
    /// animation timing metadata.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs_core::{format::U8, image::Image};
    /// let frame = Image::<U8>::from_buffer(1, 1, 1, vec![0]).unwrap();
    /// let image = Image::<U8>::from_buffer(1, 1, 1, vec![0]).unwrap().with_frames(vec![frame]);
    /// assert_eq!(image.frames().unwrap().len(), 1);
    /// ```
    pub fn with_frames(mut self, frames: Vec<Self>) -> Self {
        self.frames = Some(frames);
        self
    }

    /// Attach typed animation frames to this image and mirror their images into
    /// the legacy `frames()` accessor.
    #[must_use]
    pub fn with_animation_frames(mut self, frames: Vec<AnimationFrame<F>>) -> Self
    where
        F::Sample: Clone,
    {
        self.frames = Some(frames.iter().map(|frame| frame.image().clone()).collect());
        self.animation_frames = Some(frames);
        self.metadata.n_pages = self.frames.as_ref().map(|frames| frames.len() as u32);
        self.metadata.page_height = (self.metadata.n_pages.unwrap_or(1) > 1).then_some(self.height);
        self
    }

    /// Attach animation loop metadata to this image.
    #[must_use]
    pub const fn with_animation_loop_count(mut self, loop_count: AnimationLoopCount) -> Self {
        self.metadata.animation_loop_count = Some(loop_count);
        self
    }

    /// Return the image width in pixels.
    ///
    /// This gives schedulers and consumers geometry without exposing internal fields.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs_core::{format::U8, image::Image};
    /// let image = Image::<U8>::from_buffer(2, 1, 1, vec![0, 1]).unwrap();
    /// assert_eq!(image.width(), 2);
    /// ```
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }
    /// Return the image height in pixels.
    ///
    /// This keeps downstream geometry code independent from struct layout.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs_core::{format::U8, image::Image};
    /// let image = Image::<U8>::from_buffer(1, 2, 1, vec![0, 1]).unwrap();
    /// assert_eq!(image.height(), 2);
    /// ```
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }
    /// Return the number of interleaved bands per pixel.
    ///
    /// This helps generic code validate colorspace and codec expectations.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs_core::{format::U8, image::Image};
    /// let image = Image::<U8>::from_buffer(1, 1, 3, vec![0, 0, 0]).unwrap();
    /// assert_eq!(image.bands(), 3);
    /// ```
    #[must_use]
    pub const fn bands(&self) -> u32 {
        self.bands
    }
    /// Borrow the contiguous interleaved pixel buffer.
    ///
    /// This is useful for zero-copy inspection or handing pixels to sinks and tests.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs_core::{format::U8, image::Image};
    /// let image = Image::<U8>::from_buffer(1, 1, 1, vec![7]).unwrap();
    /// assert_eq!(image.pixels(), &[7]);
    /// ```
    #[must_use]
    pub fn pixels(&self) -> &[F::Sample] {
        &self.data
    }
    /// Borrow the image metadata.
    ///
    /// This lets callers inspect interpretation and container state without cloning it.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs_core::{format::U8, image::Image};
    /// let image = Image::<U8>::from_buffer(1, 1, 1, vec![0]).unwrap();
    /// assert!(image.metadata().interpretation.is_none());
    /// ```
    #[must_use]
    pub const fn metadata(&self) -> &ImageMetadata {
        &self.metadata
    }

    /// Borrow legacy frame images for multi-page or animated content.
    ///
    /// This provides compatibility with consumers that only need per-frame pixels.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs_core::{format::U8, image::Image};
    /// let frame = Image::<U8>::from_buffer(1, 1, 1, vec![0]).unwrap();
    /// let image = Image::<U8>::from_buffer(1, 1, 1, vec![0]).unwrap().with_frames(vec![frame]);
    /// assert_eq!(image.frames().unwrap().len(), 1);
    /// ```
    #[must_use]
    pub fn frames(&self) -> Option<&[Self]> {
        self.frames.as_deref()
    }

    /// Borrow the typed animation frame sequence, including delays/disposal.
    #[must_use]
    pub fn animation_frames(&self) -> Option<&[AnimationFrame<F>]> {
        self.animation_frames.as_deref()
    }

    /// Consume the image and return its owned pixel buffer.
    ///
    /// This avoids an extra copy when handing image data to foreign encoders or tests.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs_core::{format::U8, image::Image};
    /// let image = Image::<U8>::from_buffer(1, 1, 1, vec![9]).unwrap();
    /// assert_eq!(image.into_buffer(), vec![9]);
    /// ```
    #[must_use]
    pub fn into_buffer(self) -> Vec<F::Sample> {
        self.data
    }
}

pub(super) fn checked_image_buffer_len(
    width: u32,
    height: u32,
    bands: u32,
) -> Result<usize, ViprsError> {
    let Some(bytes) = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixel_count| pixel_count.checked_mul(u64::from(bands)))
    else {
        let total_bytes = u128::from(width) * u128::from(height) * u128::from(bands);
        return Err(ViprsError::ImageTooLarge {
            width,
            height,
            bands,
            bytes: total_bytes,
            limit_bytes: usize::MAX as u128,
            details: "image buffer dimensions exceed addressable memory",
        });
    };

    usize::try_from(bytes).map_err(|_| ViprsError::ImageTooLarge {
        width,
        height,
        bands,
        bytes: u128::from(bytes),
        limit_bytes: usize::MAX as u128,
        details: "image buffer dimensions exceed addressable memory",
    })
}

impl<F: BandFormat> std::fmt::Debug for Image<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Image")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("bands", &self.bands)
            .field("metadata", &self.metadata)
            .field("frame_count", &self.frames.as_ref().map(std::vec::Vec::len))
            .finish_non_exhaustive()
    }
}

impl<F: BandFormat> Clone for Image<F>
where
    F::Sample: Clone,
{
    fn clone(&self) -> Self {
        Self {
            width: self.width,
            height: self.height,
            bands: self.bands,
            data: self.data.clone(),
            metadata: self.metadata.clone(),
            frames: self.frames.clone(),
            animation_frames: self.animation_frames.clone(),
        }
    }
}

impl<F: BandFormat> PartialEq for Image<F>
where
    F::Sample: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.width == other.width
            && self.height == other.height
            && self.bands == other.bands
            && self.data == other.data
            && self.metadata == other.metadata
            && self.frames == other.frames
            && self.animation_frames == other.animation_frames
    }
}

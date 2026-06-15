use crate::domain::{
    draw::DrawOp,
    error::{DrawError, ViprsError},
    format::BandFormat,
    image::TileMut,
};

use super::clip_overlay;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
/// Enumerates the available draw image mode values.
pub enum DrawImageMode {
    /// Uses the `Set` variant of `DrawImageMode`.
    Set,
    /// Uses the `Add` variant of `DrawImageMode`.
    Add,
}

/// Defines the contract for draw image sample.
pub trait DrawImageSample: Copy {
    /// Returns or performs add clipped.
    fn add_clipped(base: Self, overlay: Self) -> Self;
}

impl DrawImageSample for u8 {
    #[inline]
    fn add_clipped(base: Self, overlay: Self) -> Self {
        base.saturating_add(overlay)
    }
}

impl DrawImageSample for u16 {
    #[inline]
    fn add_clipped(base: Self, overlay: Self) -> Self {
        base.saturating_add(overlay)
    }
}

impl DrawImageSample for i16 {
    #[inline]
    fn add_clipped(base: Self, overlay: Self) -> Self {
        let sum = i32::from(base) + i32::from(overlay);
        sum.clamp(i32::from(i8::MIN), i32::from(i8::MAX)) as Self
    }
}

impl DrawImageSample for u32 {
    #[inline]
    fn add_clipped(base: Self, overlay: Self) -> Self {
        base.saturating_add(overlay)
    }
}

impl DrawImageSample for i32 {
    #[inline]
    fn add_clipped(base: Self, overlay: Self) -> Self {
        base.saturating_add(overlay)
    }
}

impl DrawImageSample for f32 {
    #[inline]
    fn add_clipped(base: Self, overlay: Self) -> Self {
        base + overlay
    }
}

impl DrawImageSample for f64 {
    #[inline]
    fn add_clipped(base: Self, overlay: Self) -> Self {
        base + overlay
    }
}

/// Applies the `draw image` drawing operation to an image. It updates a target region by
/// painting or blending the requested primitive.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::draw::draw_image::DrawImageOp;
///
/// let op = DrawImageOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct DrawImageOp<F: BandFormat> {
    sub_width: u32,
    sub_height: u32,
    sub_bands: u32,
    sub: Vec<F::Sample>,
    x: i32,
    y: i32,
    mode: DrawImageMode,
}

impl<F> DrawImageOp<F>
where
    F: BandFormat,
    F::Sample: DrawImageSample,
{
    /// Creates a new `DrawImageOp`.
    pub fn new(
        sub_width: u32,
        sub_height: u32,
        sub_bands: u32,
        sub: Vec<F::Sample>,
        x: i32,
        y: i32,
        mode: DrawImageMode,
    ) -> Result<Self, ViprsError> {
        validate_sub_image(sub_width, sub_height, sub_bands, sub.len())?;
        Ok(Self {
            sub_width,
            sub_height,
            sub_bands,
            sub,
            x,
            y,
            mode,
        })
    }

    #[inline]
    /// Processes one output region from the supplied input tiles.
    pub fn process_region(&self, tile: &mut TileMut<F>) {
        draw_image_in_region(
            tile.data,
            tile.region,
            tile.bands,
            self.sub_width,
            self.sub_height,
            self.sub_bands,
            &self.sub,
            self.x,
            self.y,
            self.mode,
        );
    }
}

impl<F> DrawOp<F> for DrawImageOp<F>
where
    F: BandFormat,
    F::Sample: DrawImageSample,
{
    fn draw(&self, tile: &mut TileMut<F>) {
        self.process_region(tile);
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "draw_image mirrors libvips and existing draw_* positional APIs"
)]
/// Returns or performs draw image.
pub fn draw_image<T: DrawImageSample>(
    buf: &mut [T],
    width: u32,
    height: u32,
    bands: u32,
    sub: &[T],
    sub_width: u32,
    sub_height: u32,
    sub_bands: u32,
    x: i32,
    y: i32,
    mode: DrawImageMode,
) -> Result<(), ViprsError> {
    validate_image_buffer(buf, width, height, bands)?;
    validate_sub_image(sub_width, sub_height, sub_bands, sub.len())?;
    validate_overlay_bands(bands, sub_bands)?;
    draw_image_in_region(
        buf,
        crate::domain::image::Region::new(0, 0, width, height),
        bands,
        sub_width,
        sub_height,
        sub_bands,
        sub,
        x,
        y,
        mode,
    );
    Ok(())
}

pub(crate) fn validate_overlay_bands(
    image_bands: u32,
    overlay_bands: u32,
) -> Result<(), ViprsError> {
    if overlay_bands == 1 || overlay_bands == image_bands {
        return Ok(());
    }

    Err(DrawError::BandCountMismatch {
        overlay_bands,
        image_bands,
    }
    .into())
}

#[expect(
    clippy::too_many_arguments,
    reason = "hot-path helper keeps draw geometry as scalars like the existing draw helpers"
)]
pub(crate) fn draw_image_in_region<T: DrawImageSample>(
    data: &mut [T],
    region: crate::domain::image::Region,
    bands: u32,
    sub_width: u32,
    sub_height: u32,
    sub_bands: u32,
    sub: &[T],
    x: i32,
    y: i32,
    mode: DrawImageMode,
) {
    let Some((bands, sub_bands, clip)) = validate_and_clip(
        data, region, bands, sub_width, sub_height, sub_bands, sub, x, y,
    ) else {
        return;
    };

    let region_width = region.width as usize;
    let sub_width = sub_width as usize;
    for row in 0..clip.height {
        let dst_y = (clip.top + row) as usize;
        let src_y = (clip.sub_top + row) as usize;
        let dst_start = (dst_y * region_width + clip.left as usize) * bands;
        let src_start = (src_y * sub_width + clip.sub_left as usize) * sub_bands;
        let width = clip.width as usize;

        match (mode, sub_bands == bands) {
            (DrawImageMode::Set, true) => {
                let count = width * bands;
                data[dst_start..dst_start + count]
                    .copy_from_slice(&sub[src_start..src_start + count]);
            }
            (DrawImageMode::Set, false) => {
                for col in 0..width {
                    let value = sub[src_start + col];
                    let dst = dst_start + col * bands;
                    for band in 0..bands {
                        data[dst + band] = value;
                    }
                }
            }
            (DrawImageMode::Add, true) => {
                let count = width * bands;
                for index in 0..count {
                    let dst = dst_start + index;
                    data[dst] = T::add_clipped(data[dst], sub[src_start + index]);
                }
            }
            (DrawImageMode::Add, false) => {
                for col in 0..width {
                    let value = sub[src_start + col];
                    let dst = dst_start + col * bands;
                    for band in 0..bands {
                        data[dst + band] = T::add_clipped(data[dst + band], value);
                    }
                }
            }
        }
    }
}

pub(crate) fn validate_image_buffer<T>(
    buf: &[T],
    width: u32,
    height: u32,
    bands: u32,
) -> Result<(), ViprsError> {
    let bands_usize = usize::try_from(bands).map_err(|_| DrawError::BufferDimensionsOverflow {
        width,
        height,
        bands: usize::MAX,
    })?;
    if bands == 0 {
        return Err(DrawError::InvalidBandCount { bands }.into());
    }

    let expected = usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
        .and_then(|pixels| pixels.checked_mul(bands_usize))
        .ok_or(DrawError::BufferDimensionsOverflow {
            width,
            height,
            bands: bands_usize,
        })?;

    if buf.len() != expected {
        return Err(DrawError::BufferLengthMismatch {
            len: buf.len(),
            expected,
            width,
            height,
            bands: bands_usize,
        }
        .into());
    }

    Ok(())
}

pub(crate) fn validate_sub_image(
    width: u32,
    height: u32,
    bands: u32,
    len: usize,
) -> Result<(), ViprsError> {
    if bands == 0 {
        return Err(DrawError::InvalidBandCount { bands }.into());
    }

    let bands_usize = bands as usize;
    let expected = usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
        .and_then(|pixels| pixels.checked_mul(bands_usize))
        .ok_or(DrawError::BufferDimensionsOverflow {
            width,
            height,
            bands: bands_usize,
        })?;

    if len != expected {
        return Err(DrawError::BufferLengthMismatch {
            len,
            expected,
            width,
            height,
            bands: bands_usize,
        }
        .into());
    }

    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "validation mirrors the draw_image hot-path parameters without a temporary config object"
)]
fn validate_and_clip<T>(
    data: &[T],
    region: crate::domain::image::Region,
    bands: u32,
    sub_width: u32,
    sub_height: u32,
    sub_bands: u32,
    sub: &[T],
    x: i32,
    y: i32,
) -> Option<(usize, usize, super::OverlayClip)> {
    let bands = usize::try_from(bands).ok()?;
    let sub_bands = usize::try_from(sub_bands).ok()?;
    let expected = region.pixel_count().checked_mul(bands)?;
    let sub_expected = usize::try_from(sub_width)
        .ok()?
        .checked_mul(usize::try_from(sub_height).ok()?)?
        .checked_mul(sub_bands)?;
    debug_assert_eq!(data.len(), expected);
    debug_assert_eq!(sub.len(), sub_expected);

    if data.len() != expected
        || sub.len() != sub_expected
        || bands == 0
        || !(sub_bands == 1 || sub_bands == bands)
    {
        return None;
    }

    let clip = clip_overlay(region, x, y, sub_width, sub_height)?;
    Some((bands, sub_bands, clip))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        draw::DrawOp,
        format::U8,
        image::{Region, TileMut},
    };
    use proptest::prelude::*;

    #[test]
    fn set_mode_copies_overlapping_sub_image() {
        let mut buf = vec![0_u8; 4 * 3];
        let sub = vec![1, 2, 3, 4, 5, 6];

        draw_image(&mut buf, 4, 3, 1, &sub, 3, 2, 1, 1, 1, DrawImageMode::Set).unwrap();

        assert_eq!(
            buf,
            vec![
                0, 0, 0, 0, //
                0, 1, 2, 3, //
                0, 4, 5, 6,
            ]
        );
    }

    #[test]
    fn add_mode_matches_libvips_i16_clip_bounds() {
        let mut buf = vec![120_i16, -120, 100, -100];
        let sub = vec![20_i16, -20, 100, -100];

        draw_image(&mut buf, 2, 2, 1, &sub, 2, 2, 1, 0, 0, DrawImageMode::Add).unwrap();

        assert_eq!(buf, vec![127, -128, 127, -128]);
    }

    #[test]
    fn one_band_sub_image_expands_to_target_bands() {
        let op = DrawImageOp::<U8>::new(2, 1, 1, vec![8, 9], 1, 0, DrawImageMode::Set).unwrap();
        let mut pixels = vec![0_u8; 3 * 2 * 3];
        let mut tile = TileMut::new(Region::new(0, 0, 3, 2), 3, &mut pixels);

        op.draw(&mut tile);

        assert_eq!(&tile.data[3..9], &[8, 8, 8, 9, 9, 9]);
    }

    #[test]
    fn set_mode_clips_overlay_against_left_edge() {
        let mut buf = vec![0_u8; 4 * 3];
        let sub = vec![
            1, 2, 3, //
            4, 5, 6,
        ];

        draw_image(&mut buf, 4, 3, 1, &sub, 3, 2, 1, -1, 1, DrawImageMode::Set).unwrap();

        assert_eq!(
            buf,
            vec![
                0, 0, 0, 0, //
                2, 3, 0, 0, //
                5, 6, 0, 0,
            ]
        );
    }

    #[test]
    fn add_mode_expands_single_overlay_band_across_target_bands() {
        let mut buf = vec![
            1_u8, 2, 3, //
            4, 5, 6,
        ];

        draw_image(
            &mut buf,
            2,
            1,
            3,
            &[10, 20],
            2,
            1,
            1,
            0,
            0,
            DrawImageMode::Add,
        )
        .unwrap();

        assert_eq!(
            buf,
            vec![
                11, 12, 13, //
                24, 25, 26,
            ]
        );
    }

    #[test]
    fn draw_image_in_region_uses_region_offsets() {
        let mut pixels = vec![0_u8; 4];

        draw_image_in_region(
            &mut pixels,
            Region::new(2, 1, 2, 2),
            1,
            2,
            2,
            1,
            &[7, 8, 9, 10],
            1,
            1,
            DrawImageMode::Set,
        );

        assert_eq!(
            pixels,
            vec![
                8, 0, //
                10, 0,
            ]
        );
    }

    #[test]
    fn validate_overlay_bands_rejects_mismatch() {
        let err = validate_overlay_bands(3, 2).unwrap_err();

        assert!(matches!(
            err,
            ViprsError::Draw(DrawError::BandCountMismatch {
                image_bands: 3,
                overlay_bands: 2,
            })
        ));
    }

    #[test]
    fn validate_image_buffer_rejects_zero_bands() {
        let err = validate_image_buffer(&[0_u8], 1, 1, 0).unwrap_err();

        assert!(matches!(
            err,
            ViprsError::Draw(DrawError::InvalidBandCount { bands: 0 })
        ));
    }

    #[test]
    fn validate_sub_image_rejects_wrong_length() {
        let err = validate_sub_image(2, 2, 1, 3).unwrap_err();

        assert!(matches!(
            err,
            ViprsError::Draw(DrawError::BufferLengthMismatch {
                len: 3,
                expected: 4,
                width: 2,
                height: 2,
                bands: 1,
            })
        ));
    }

    #[test]
    fn sample_impls_cover_non_u8_paths() {
        assert_eq!(
            <u16 as DrawImageSample>::add_clipped(u16::MAX - 1, 5),
            u16::MAX
        );
        assert_eq!(<i16 as DrawImageSample>::add_clipped(120, 20), 127);
        assert_eq!(
            <u32 as DrawImageSample>::add_clipped(u32::MAX - 1, 5),
            u32::MAX
        );
        assert_eq!(
            <i32 as DrawImageSample>::add_clipped(i32::MAX - 1, 5),
            i32::MAX
        );
        assert_eq!(<f32 as DrawImageSample>::add_clipped(1.5, 2.25), 3.75);
        assert_eq!(<f64 as DrawImageSample>::add_clipped(1.5, 2.25), 3.75);
    }

    #[test]
    fn validate_sub_image_rejects_zero_bands() {
        let err = validate_sub_image(1, 1, 0, 0).unwrap_err();

        assert!(matches!(
            err,
            ViprsError::Draw(DrawError::InvalidBandCount { bands: 0 })
        ));
    }

    #[test]
    fn draw_image_in_region_ignores_invalid_overlay_bands() {
        let mut pixels = vec![1_u8, 2, 3, 4];
        let original = pixels.clone();

        draw_image_in_region(
            &mut pixels,
            Region::new(0, 0, 2, 2),
            1,
            2,
            2,
            2,
            &[9, 9, 9, 9, 9, 9, 9, 9],
            0,
            0,
            DrawImageMode::Set,
        );

        assert_eq!(pixels, original);
    }

    proptest! {
        #[test]
        fn set_identity_when_sub_image_is_outside(
            mut pixels in proptest::collection::vec(any::<u8>(), 1..128),
            sub in proptest::collection::vec(any::<u8>(), 1..32)
        ) {
            let original = pixels.clone();
            let width = pixels.len() as u32;
            draw_image(
                &mut pixels,
                width,
                1,
                1,
                &sub,
                sub.len() as u32,
                1,
                1,
                width as i32 + 1,
                0,
                DrawImageMode::Set,
            ).unwrap();

            prop_assert_eq!(pixels, original);
        }

        #[test]
        fn add_mode_saturates_u8_boundary(
            base in any::<u8>(),
            overlay in any::<u8>()
        ) {
            let mut pixels = vec![base];
            draw_image(
                &mut pixels,
                1,
                1,
                1,
                &[overlay],
                1,
                1,
                1,
                0,
                0,
                DrawImageMode::Add,
            ).unwrap();

            prop_assert_eq!(pixels[0], base.saturating_add(overlay));
        }

        #[test]
        fn draw_image_in_region_is_identity_when_overlay_misses_offset_tile(
            mut pixels in proptest::collection::vec(any::<u8>(), 1..16)
        ) {
            let original = pixels.clone();
            let width = pixels.len() as u32;

            draw_image_in_region(
                &mut pixels,
                Region::new(10, 10, width, 1),
                1,
                2,
                1,
                1,
                &[1, 2],
                0,
                0,
                DrawImageMode::Set,
            );

            prop_assert_eq!(pixels, original);
        }
    }
}

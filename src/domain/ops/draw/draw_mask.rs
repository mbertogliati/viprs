use crate::domain::{
    draw::DrawOp,
    error::{DrawError, ViprsError},
    format::BandFormat,
    image::{Region, TileMut},
};

use super::{
    clip_overlay,
    draw_image::{validate_image_buffer, validate_overlay_bands},
    validate_ink,
};

/// Defines the contract for draw mask sample.
pub trait DrawMaskSample: Copy {
    /// Returns or performs blend.
    fn blend(base: Self, ink: Self, mask: u8) -> Self;
}

impl DrawMaskSample for u8 {
    #[inline]
    fn blend(base: Self, ink: Self, mask: u8) -> Self {
        let mask = u32::from(mask);
        let inverse = 255 - mask;
        ((u32::from(ink) * mask + u32::from(base) * inverse) / 255) as Self
    }
}

impl DrawMaskSample for u16 {
    #[inline]
    fn blend(base: Self, ink: Self, mask: u8) -> Self {
        let mask = u32::from(mask);
        let inverse = 255 - mask;
        ((u32::from(ink) * mask + u32::from(base) * inverse) / 255) as Self
    }
}

impl DrawMaskSample for i16 {
    #[inline]
    fn blend(base: Self, ink: Self, mask: u8) -> Self {
        let mask = i32::from(mask);
        let inverse = 255 - mask;
        ((i32::from(ink) * mask + i32::from(base) * inverse) / 255) as Self
    }
}

impl DrawMaskSample for u32 {
    #[inline]
    fn blend(base: Self, ink: Self, mask: u8) -> Self {
        let mask = f64::from(mask);
        (f64::from(base).mul_add(255.0 - mask, f64::from(ink) * mask) / 255.0) as Self
    }
}

impl DrawMaskSample for i32 {
    #[inline]
    fn blend(base: Self, ink: Self, mask: u8) -> Self {
        let mask = f64::from(mask);
        (f64::from(base).mul_add(255.0 - mask, f64::from(ink) * mask) / 255.0) as Self
    }
}

impl DrawMaskSample for f32 {
    #[inline]
    fn blend(base: Self, ink: Self, mask: u8) -> Self {
        let mask = Self::from(mask);
        base.mul_add(255.0 - mask, ink * mask) / 255.0
    }
}

impl DrawMaskSample for f64 {
    #[inline]
    fn blend(base: Self, ink: Self, mask: u8) -> Self {
        let mask = Self::from(mask);
        base.mul_add(255.0 - mask, ink * mask) / 255.0
    }
}

/// Applies the `draw mask` drawing operation to an image. It updates a target region by
/// painting or blending the requested primitive.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::draw::draw_mask::DrawMaskOp;
///
/// let op = DrawMaskOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct DrawMaskOp<F: BandFormat> {
    mask_width: u32,
    mask_height: u32,
    mask: Vec<u8>,
    ink: Vec<F::Sample>,
    x: i32,
    y: i32,
}

impl<F> DrawMaskOp<F>
where
    F: BandFormat,
    F::Sample: DrawMaskSample,
{
    /// Creates a new `DrawMaskOp`.
    pub fn new(
        mask_width: u32,
        mask_height: u32,
        mask: Vec<u8>,
        ink: Vec<F::Sample>,
        x: i32,
        y: i32,
    ) -> Result<Self, ViprsError> {
        validate_mask_buffer(&mask, mask_width, mask_height)?;
        validate_ink(&ink)?;
        Ok(Self {
            mask_width,
            mask_height,
            mask,
            ink,
            x,
            y,
        })
    }

    #[inline]
    /// Processes one output region from the supplied input tiles.
    pub fn process_region(&self, tile: &mut TileMut<F>) {
        draw_mask_in_region(
            tile.data,
            tile.region,
            tile.bands,
            &self.mask,
            self.mask_width,
            self.mask_height,
            &self.ink,
            self.x,
            self.y,
        );
    }
}

impl<F> DrawOp<F> for DrawMaskOp<F>
where
    F: BandFormat,
    F::Sample: DrawMaskSample,
{
    fn draw(&self, tile: &mut TileMut<F>) {
        self.process_region(tile);
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "draw_mask mirrors libvips and existing draw_* positional APIs"
)]
/// Returns or performs draw mask.
pub fn draw_mask<T: DrawMaskSample>(
    buf: &mut [T],
    width: u32,
    height: u32,
    bands: u32,
    mask: &[u8],
    mask_width: u32,
    mask_height: u32,
    ink: &[T],
    x: i32,
    y: i32,
) -> Result<(), ViprsError> {
    validate_image_buffer(buf, width, height, bands)?;
    validate_mask_buffer(mask, mask_width, mask_height)?;
    validate_ink(ink)?;
    let ink_bands = u32::try_from(ink.len()).map_err(|_| DrawError::BufferDimensionsOverflow {
        width: 1,
        height: 1,
        bands: ink.len(),
    })?;
    validate_overlay_bands(bands, ink_bands)?;
    draw_mask_in_region(
        buf,
        Region::new(0, 0, width, height),
        bands,
        mask,
        mask_width,
        mask_height,
        ink,
        x,
        y,
    );
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "hot-path helper keeps draw geometry as scalars like the existing draw helpers"
)]
pub(crate) fn draw_mask_in_region<T: DrawMaskSample>(
    data: &mut [T],
    region: Region,
    bands: u32,
    mask: &[u8],
    mask_width: u32,
    mask_height: u32,
    ink: &[T],
    x: i32,
    y: i32,
) {
    let Some((bands, ink_bands, clip)) = validate_and_clip(
        data,
        region,
        bands,
        mask,
        mask_width,
        mask_height,
        ink,
        x,
        y,
    ) else {
        return;
    };

    let region_width = region.width as usize;
    let mask_width = mask_width as usize;
    for row in 0..clip.height {
        let dst_y = (clip.top + row) as usize;
        let mask_y = (clip.sub_top + row) as usize;
        let dst_start = (dst_y * region_width + clip.left as usize) * bands;
        let mask_start = mask_y * mask_width + clip.sub_left as usize;
        let width = clip.width as usize;

        for col in 0..width {
            let alpha = mask[mask_start + col];
            let dst = dst_start + col * bands;
            for band in 0..bands {
                let ink = ink[if ink_bands == 1 { 0 } else { band }];
                data[dst + band] = T::blend(data[dst + band], ink, alpha);
            }
        }
    }
}

fn validate_mask_buffer(mask: &[u8], width: u32, height: u32) -> Result<(), ViprsError> {
    validate_image_buffer(mask, width, height, 1)
}

#[expect(
    clippy::too_many_arguments,
    reason = "validation mirrors the draw_mask hot-path parameters without a temporary config object"
)]
fn validate_and_clip<T>(
    data: &[T],
    region: Region,
    bands: u32,
    mask: &[u8],
    mask_width: u32,
    mask_height: u32,
    ink: &[T],
    x: i32,
    y: i32,
) -> Option<(usize, usize, super::OverlayClip)> {
    let bands = usize::try_from(bands).ok()?;
    let expected = region.pixel_count().checked_mul(bands)?;
    let mask_expected = usize::try_from(mask_width)
        .ok()?
        .checked_mul(usize::try_from(mask_height).ok()?)?;
    debug_assert_eq!(data.len(), expected);
    debug_assert_eq!(mask.len(), mask_expected);
    debug_assert!(ink.len() == 1 || ink.len() == bands);

    if data.len() != expected
        || mask.len() != mask_expected
        || !(ink.len() == 1 || ink.len() == bands)
        || bands == 0
    {
        return None;
    }

    let clip = clip_overlay(region, x, y, mask_width, mask_height)?;
    Some((bands, ink.len(), clip))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        draw::DrawOp,
        error::DrawError,
        format::U8,
        image::{Region, TileMut},
    };
    use proptest::prelude::*;

    fn blend_u8(base: u8, ink: u8, mask: u8) -> u8 {
        let mask = u32::from(mask);
        let inverse = 255 - mask;
        ((u32::from(ink) * mask + u32::from(base) * inverse) / 255) as u8
    }

    #[test]
    fn mask_255_paints_ink_and_mask_0_preserves_base() {
        let mut buf = vec![10_u8, 20, 30, 40];
        let mask = vec![0, 255, 128, 255];

        draw_mask(&mut buf, 2, 2, 1, &mask, 2, 2, &[110], 0, 0).unwrap();

        assert_eq!(buf, vec![10, 110, 70, 110]);
    }

    #[test]
    fn clips_mask_against_image_bounds() {
        let op = DrawMaskOp::<U8>::new(3, 2, vec![255; 6], vec![9], -1, 1).unwrap();
        let mut pixels = vec![0_u8; 4 * 3];
        let mut tile = TileMut::new(Region::new(0, 0, 4, 3), 1, &mut pixels);

        op.draw(&mut tile);

        assert_eq!(
            tile.data,
            &[
                0, 0, 0, 0, //
                9, 9, 0, 0, //
                9, 9, 0, 0,
            ]
        );
    }

    #[test]
    fn one_band_ink_expands_to_target_bands() {
        let mut pixels = vec![0_u8; 2 * 2 * 3];

        draw_mask(&mut pixels, 2, 2, 3, &[255; 4], 2, 2, &[7], 0, 0).unwrap();

        assert_eq!(pixels, vec![7; 2 * 2 * 3]);
    }

    #[test]
    fn clipped_mask_uses_source_offset() {
        let mut pixels = vec![10_u8, 20, 30, 40];

        draw_mask(&mut pixels, 4, 1, 1, &[0, 128, 255], 3, 1, &[110], -1, 0).unwrap();

        assert_eq!(pixels, vec![blend_u8(10, 110, 128), 110, 30, 40]);
    }

    #[test]
    fn multi_band_ink_blends_each_band_independently() {
        let mut pixels = vec![10_u8, 20, 30];

        draw_mask(&mut pixels, 1, 1, 3, &[128], 1, 1, &[110, 120, 130], 0, 0).unwrap();

        assert_eq!(
            pixels,
            vec![
                blend_u8(10, 110, 128),
                blend_u8(20, 120, 128),
                blend_u8(30, 130, 128),
            ]
        );
    }

    #[test]
    fn draw_mask_in_region_uses_region_offsets() {
        let mut pixels = vec![10_u8, 20, 30, 40];

        draw_mask_in_region(
            &mut pixels,
            Region::new(2, 1, 2, 2),
            1,
            &[0, 255, 255, 128],
            2,
            2,
            &[90],
            1,
            1,
        );

        assert_eq!(
            pixels,
            vec![
                90,
                20, //
                blend_u8(30, 90, 128),
                40,
            ]
        );
    }

    #[test]
    fn draw_mask_rejects_mismatched_ink_bands() {
        let err = draw_mask(&mut [0_u8; 3], 1, 1, 3, &[255], 1, 1, &[1, 2], 0, 0).unwrap_err();

        assert!(matches!(
            err,
            ViprsError::Draw(DrawError::BandCountMismatch {
                image_bands: 3,
                overlay_bands: 2,
            })
        ));
    }

    #[test]
    fn draw_mask_op_new_rejects_empty_ink() {
        assert!(matches!(
            DrawMaskOp::<U8>::new(1, 1, vec![255], vec![], 0, 0),
            Err(ViprsError::Draw(DrawError::EmptyColor))
        ));
    }

    #[test]
    fn sample_impls_cover_non_u8_paths() {
        assert_eq!(<u16 as DrawMaskSample>::blend(10, 110, 128), 60);
        assert_eq!(<i16 as DrawMaskSample>::blend(-10, 110, 128), 50);
        assert_eq!(<u32 as DrawMaskSample>::blend(10, 110, 128), 60);
        assert_eq!(<i32 as DrawMaskSample>::blend(-10, 110, 128), 50);
        assert!((<f32 as DrawMaskSample>::blend(10.0, 110.0, 128) - 60.196_08).abs() < 1e-5);
        assert!((<f64 as DrawMaskSample>::blend(10.0, 110.0, 128) - 60.196_078).abs() < 1e-6);
    }

    #[test]
    fn draw_mask_in_region_returns_early_for_zero_bands() {
        let mut pixels = Vec::<u8>::new();

        draw_mask_in_region(
            &mut pixels,
            Region::new(0, 0, 0, 0),
            0,
            &[],
            0,
            0,
            &[9],
            0,
            0,
        );

        assert!(pixels.is_empty());
    }

    proptest! {
        #[test]
        fn zero_mask_is_identity(mut pixels in proptest::collection::vec(any::<u8>(), 1..128)) {
            let original = pixels.clone();
            let mask = vec![0_u8; pixels.len()];

            draw_mask(
                &mut pixels,
                original.len() as u32,
                1,
                1,
                &mask,
                original.len() as u32,
                1,
                &[255],
                0,
                0,
            ).unwrap();

            prop_assert_eq!(pixels, original);
        }

        #[test]
        fn full_mask_sets_boundary_values(base in any::<u8>(), ink in any::<u8>()) {
            let mut pixels = vec![base];

            draw_mask(
                &mut pixels,
                1,
                1,
                1,
                &[255],
                1,
                1,
                &[ink],
                0,
                0,
            ).unwrap();

            prop_assert_eq!(pixels[0], ink);
        }

        #[test]
        fn draw_mask_in_region_is_identity_when_mask_misses_offset_tile(
            mut pixels in proptest::collection::vec(any::<u8>(), 1..16)
        ) {
            let original = pixels.clone();
            let width = pixels.len() as u32;

            draw_mask_in_region(
                &mut pixels,
                Region::new(10, 10, width, 1),
                1,
                &[255, 255],
                2,
                1,
                &[42],
                0,
                0,
            );

            prop_assert_eq!(pixels, original);
        }
    }
}

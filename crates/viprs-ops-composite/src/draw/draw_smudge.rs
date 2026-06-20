use viprs_core::{
    draw::DrawOp,
    error::ViprsError,
    format::BandFormat,
    image::{Region, TileMut},
};

use super::draw_image::validate_image_buffer;

/// Defines the contract for smudge sample.
pub trait SmudgeSample: Copy {
    /// Returns or performs smudge.
    fn smudge(current: Self, total: f64) -> Self;
    /// Converts this value to f64.
    fn to_f64(self) -> f64;
}

impl SmudgeSample for u8 {
    #[inline]
    fn smudge(current: Self, total: f64) -> Self {
        (16.0f64.mul_add(f64::from(current), total) / 25.0) as Self
    }

    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl SmudgeSample for u16 {
    #[inline]
    fn smudge(current: Self, total: f64) -> Self {
        (16.0f64.mul_add(f64::from(current), total) / 25.0) as Self
    }

    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl SmudgeSample for i16 {
    #[inline]
    fn smudge(current: Self, total: f64) -> Self {
        (16.0f64.mul_add(f64::from(current), total) / 25.0) as Self
    }

    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl SmudgeSample for u32 {
    #[inline]
    fn smudge(current: Self, total: f64) -> Self {
        (16.0f64.mul_add(f64::from(current), total) / 25.0) as Self
    }

    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl SmudgeSample for i32 {
    #[inline]
    fn smudge(current: Self, total: f64) -> Self {
        (16.0f64.mul_add(f64::from(current), total) / 25.0) as Self
    }

    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl SmudgeSample for f32 {
    #[inline]
    fn smudge(current: Self, total: f64) -> Self {
        (16.0f64.mul_add(f64::from(current), total) / 25.0) as Self
    }

    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl SmudgeSample for f64 {
    #[inline]
    fn smudge(current: Self, total: f64) -> Self {
        16.0f64.mul_add(current, total) / 25.0
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
}

/// Applies the `draw smudge` drawing operation to an image. It updates a target region by
/// painting or blending the requested primitive.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::draw::draw_smudge::DrawSmudgeOp;
///
/// let op = DrawSmudgeOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct DrawSmudgeOp<F: BandFormat> {
    image_width: u32,
    image_height: u32,
    left: i32,
    top: i32,
    width: u32,
    height: u32,
    _format: std::marker::PhantomData<F>,
}

impl<F> DrawSmudgeOp<F>
where
    F: BandFormat,
    F::Sample: SmudgeSample,
{
    #[must_use]
    /// Creates a new `DrawSmudgeOp`.
    pub const fn new(
        image_width: u32,
        image_height: u32,
        left: i32,
        top: i32,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            image_width,
            image_height,
            left,
            top,
            width,
            height,
            _format: std::marker::PhantomData,
        }
    }

    #[inline]
    /// Processes one output region from the supplied input tiles.
    pub fn process_region(&self, tile: &mut TileMut<F>) {
        draw_smudge_in_region(
            tile.data,
            tile.region,
            tile.bands,
            self.image_width,
            self.image_height,
            self.left,
            self.top,
            self.width,
            self.height,
        );
    }
}

impl<F> DrawOp<F> for DrawSmudgeOp<F>
where
    F: BandFormat,
    F::Sample: SmudgeSample,
{
    fn draw(&self, tile: &mut TileMut<F>) {
        self.process_region(tile);
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "draw_smudge mirrors libvips and existing draw_* positional APIs"
)]
/// Returns or performs draw smudge.
pub fn draw_smudge<T: SmudgeSample>(
    buf: &mut [T],
    image_width: u32,
    image_height: u32,
    bands: u32,
    left: i32,
    top: i32,
    width: u32,
    height: u32,
) -> Result<(), ViprsError> {
    validate_image_buffer(buf, image_width, image_height, bands)?;
    draw_smudge_in_region(
        buf,
        Region::new(0, 0, image_width, image_height),
        bands,
        image_width,
        image_height,
        left,
        top,
        width,
        height,
    );
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "hot-path helper keeps draw geometry as scalars like the existing draw helpers"
)]
pub(crate) fn draw_smudge_in_region<T: SmudgeSample>(
    data: &mut [T],
    region: Region,
    bands: u32,
    image_width: u32,
    image_height: u32,
    left: i32,
    top: i32,
    width: u32,
    height: u32,
) {
    let Some((bands, clip)) = validate_and_clip(
        data,
        region,
        bands,
        image_width,
        image_height,
        left,
        top,
        width,
        height,
    ) else {
        return;
    };

    let region_width = region.width as usize;
    let row_stride = region_width * bands;
    for row in 0..clip.height {
        let y = clip.top as usize + row as usize;
        for col in 0..clip.width {
            let x = clip.left as usize + col as usize;
            let offset = (y * region_width + x) * bands;
            for band in 0..bands {
                let mut total = 0.0;
                let above = offset - row_stride - bands + band;
                let current_row = offset - bands + band;
                let below = offset + row_stride - bands + band;
                total += data[above].to_f64();
                total += data[above + bands].to_f64();
                total += data[above + 2 * bands].to_f64();
                total += data[current_row].to_f64();
                total += data[current_row + bands].to_f64();
                total += data[current_row + 2 * bands].to_f64();
                total += data[below].to_f64();
                total += data[below + bands].to_f64();
                total += data[below + 2 * bands].to_f64();
                data[offset + band] = T::smudge(data[offset + band], total);
            }
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "validation mirrors the draw_smudge hot-path parameters without a temporary config object"
)]
fn validate_and_clip<T>(
    data: &[T],
    region: Region,
    bands: u32,
    image_width: u32,
    image_height: u32,
    left: i32,
    top: i32,
    width: u32,
    height: u32,
) -> Option<(usize, super::OverlayClip)> {
    let bands = usize::try_from(bands).ok()?;
    let expected = region.pixel_count().checked_mul(bands)?;
    debug_assert_eq!(data.len(), expected);
    if data.len() != expected || bands == 0 || image_width < 3 || image_height < 3 {
        return None;
    }

    let area = super::clip_overlay(region, left, top, width, height)?;
    let interior = super::clip_overlay(
        region,
        1,
        1,
        image_width.saturating_sub(2),
        image_height.saturating_sub(2),
    )?;
    let clipped = intersect_local(&area, &interior)?;
    let has_halo =
        clipped.left > 0 && clipped.top > 0 && clipped.left + clipped.width < region.width;
    let has_bottom_halo = clipped.top + clipped.height < region.height;
    if !has_halo || !has_bottom_halo {
        return None;
    }

    Some((bands, clipped))
}

fn intersect_local(a: &super::OverlayClip, b: &super::OverlayClip) -> Option<super::OverlayClip> {
    let left = a.left.max(b.left);
    let top = a.top.max(b.top);
    let right = (a.left + a.width).min(b.left + b.width);
    let bottom = (a.top + a.height).min(b.top + b.height);
    if left >= right || top >= bottom {
        return None;
    }

    Some(super::OverlayClip {
        left,
        top,
        width: right - left,
        height: bottom - top,
        sub_left: 0,
        sub_top: 0,
    })
}

#[cfg(all(test, feature = "_integration"))]
mod tests {
    use super::*;
    use crate::draw::OverlayClip;
    use proptest::prelude::*;
    use viprs_core::{
        draw::DrawOp,
        format::U8,
        image::{Region, TileMut},
    };

    fn expected_smudge_u8(current: u8, total: f64) -> u8 {
        ((16.0 * f64::from(current) + total) / 25.0) as u8
    }

    #[test]
    fn smudges_center_pixel_with_libvips_formula() {
        let mut pixels = vec![
            0_u8, 0, 0, //
            0, 100, 0, //
            0, 0, 0,
        ];

        draw_smudge(&mut pixels, 3, 3, 1, 1, 1, 1, 1).unwrap();

        assert_eq!(pixels[4], 68);
    }

    #[test]
    fn skips_outer_image_margin() {
        let op = DrawSmudgeOp::<U8>::new(4, 4, 0, 0, 4, 4);
        let mut pixels = vec![50_u8; 4 * 4];
        pixels[0] = 250;
        let mut tile = TileMut::new(Region::new(0, 0, 4, 4), 1, &mut pixels);

        op.draw(&mut tile);

        assert_eq!(tile.data[0], 250);
    }

    #[test]
    fn draw_smudge_in_region_uses_region_offsets() {
        let mut pixels = vec![
            0_u8, 0, 0, //
            0, 100, 0, //
            0, 0, 0,
        ];

        draw_smudge_in_region(&mut pixels, Region::new(1, 1, 3, 3), 1, 5, 5, 2, 2, 1, 1);

        assert_eq!(pixels[4], 68);
    }

    #[test]
    fn smudge_updates_each_band_independently() {
        let mut pixels = vec![
            0_u8, 0, 0, 0, 0, 0, //
            0, 0, 100, 50, 0, 0, //
            0, 0, 0, 0, 0, 0,
        ];

        draw_smudge(&mut pixels, 3, 3, 2, 1, 1, 1, 1).unwrap();

        assert_eq!(pixels[8], expected_smudge_u8(100, 100.0));
        assert_eq!(pixels[9], expected_smudge_u8(50, 50.0));
    }

    #[test]
    fn edge_touching_smudge_region_is_ignored() {
        let mut pixels = vec![0_u8; 5 * 5];
        pixels[24] = 90;
        let original = pixels.clone();

        draw_smudge(&mut pixels, 5, 5, 1, 4, 4, 1, 1).unwrap();

        assert_eq!(pixels, original);
    }

    #[test]
    fn draw_smudge_in_region_ignores_images_smaller_than_kernel() {
        let mut pixels = vec![5_u8; 4];
        let original = pixels.clone();

        draw_smudge_in_region(&mut pixels, Region::new(0, 0, 2, 2), 1, 2, 2, 0, 0, 2, 2);

        assert_eq!(pixels, original);
    }

    #[test]
    fn sample_impls_cover_non_u8_paths() {
        assert_eq!(<u16 as SmudgeSample>::smudge(100, 50.0), 66);
        assert_eq!(<u16 as SmudgeSample>::to_f64(12), 12.0);
        assert_eq!(<i16 as SmudgeSample>::smudge(-100, -50.0), -66);
        assert_eq!(<i16 as SmudgeSample>::to_f64(-12), -12.0);
        assert_eq!(<u32 as SmudgeSample>::smudge(100, 50.0), 66);
        assert_eq!(<u32 as SmudgeSample>::to_f64(12), 12.0);
        assert_eq!(<i32 as SmudgeSample>::smudge(-100, -50.0), -66);
        assert_eq!(<i32 as SmudgeSample>::to_f64(-12), -12.0);
        assert!((<f32 as SmudgeSample>::smudge(100.0, 50.0) - 66.0).abs() < f32::EPSILON);
        assert_eq!(<f32 as SmudgeSample>::to_f64(12.5), 12.5);
        assert!((<f64 as SmudgeSample>::smudge(100.0, 50.0) - 66.0).abs() < f64::EPSILON);
        assert_eq!(<f64 as SmudgeSample>::to_f64(12.5), 12.5);
    }

    #[test]
    fn draw_smudge_in_region_returns_early_without_full_halo() {
        let mut pixels = vec![
            0_u8, 0, 0, //
            0, 100, 0, //
            0, 0, 0,
        ];
        let original = pixels.clone();

        draw_smudge_in_region(&mut pixels, Region::new(1, 1, 3, 3), 1, 5, 5, 1, 1, 1, 1);

        assert_eq!(pixels, original);
    }

    #[test]
    fn intersect_local_returns_none_for_disjoint_areas() {
        let a = OverlayClip {
            left: 0,
            top: 0,
            width: 1,
            height: 1,
            sub_left: 0,
            sub_top: 0,
        };
        let b = OverlayClip {
            left: 2,
            top: 2,
            width: 1,
            height: 1,
            sub_left: 0,
            sub_top: 0,
        };

        assert!(intersect_local(&a, &b).is_none());
    }

    proptest! {
        #[test]
        fn zero_sized_area_is_identity(mut pixels in proptest::collection::vec(any::<u8>(), 9..128)) {
            let width = pixels.len() as u32;
            let original = pixels.clone();

            draw_smudge(&mut pixels, width, 1, 1, 0, 0, 0, 1).unwrap();

            prop_assert_eq!(pixels, original);
        }

        #[test]
        fn constant_image_is_identity(value in any::<u8>()) {
            let mut pixels = vec![value; 5 * 5];

            draw_smudge(&mut pixels, 5, 5, 1, 0, 0, 5, 5).unwrap();

            prop_assert_eq!(pixels, vec![value; 5 * 5]);
        }

        #[test]
        fn draw_smudge_in_region_is_identity_when_area_misses_offset_tile(
            mut pixels in proptest::collection::vec(any::<u8>(), 9..49)
        ) {
            let side = (pixels.len() as f64).sqrt().floor() as usize;
            let side = side.max(3);
            pixels.truncate(side * side);
            let original = pixels.clone();

            draw_smudge_in_region(
                &mut pixels,
                Region::new(10, 10, side as u32, side as u32),
                1,
                32,
                32,
                0,
                0,
                2,
                2,
            );

            prop_assert_eq!(pixels, original);
        }
    }
}

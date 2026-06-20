use viprs_core::{draw::DrawOp, error::ViprsError, format::BandFormat, image::TileMut};

use super::{DrawMode, draw_rect_in_region, validate_ink};

/// Applies the `draw rect` drawing operation to an image. It updates a target region by
/// painting or blending the requested primitive.
///
/// # Examples
/// ```ignore
/// use viprs_ops_composite::draw::draw_rect::DrawRectOp;
///
/// let op = DrawRectOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct DrawRectOp<F: BandFormat> {
    left: i32,
    top: i32,
    width: u32,
    height: u32,
    ink: Vec<F::Sample>,
    fill: bool,
}

impl<F: BandFormat> DrawRectOp<F> {
    /// Creates a new `DrawRectOp`.
    pub fn new(
        left: i32,
        top: i32,
        width: u32,
        height: u32,
        ink: Vec<F::Sample>,
        fill: bool,
    ) -> Result<Self, ViprsError> {
        validate_ink(&ink)?;
        Ok(Self {
            left,
            top,
            width,
            height,
            ink,
            fill,
        })
    }
}

impl<F: BandFormat> DrawOp<F> for DrawRectOp<F> {
    fn draw(&self, tile: &mut TileMut<F>) {
        draw_rect_in_region(
            tile.data,
            tile.region,
            tile.bands,
            self.left,
            self.top,
            self.width,
            self.height,
            &self.ink,
            if self.fill {
                DrawMode::Fill
            } else {
                DrawMode::Stroke
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use viprs_core::{
        format::U8,
        image::{Region, TileMut},
    };

    #[test]
    fn draws_filled_rectangle() {
        let op = DrawRectOp::<U8>::new(1, 1, 3, 2, vec![5], true).unwrap();
        let mut pixels = vec![0_u8; 5 * 4];
        let mut tile = TileMut::new(Region::new(0, 0, 5, 4), 1, &mut pixels);

        op.draw(&mut tile);

        for y in 1..=2 {
            for x in 1..=3 {
                assert_eq!(tile.data[y * 5 + x], 5);
            }
        }
    }
}

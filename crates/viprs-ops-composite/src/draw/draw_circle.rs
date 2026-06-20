use viprs_core::{draw::DrawOp, error::ViprsError, format::BandFormat, image::TileMut};

use super::{DrawMode, draw_circle_in_region, validate_ink};

/// Applies the `draw circle` drawing operation to an image. It updates a target region by
/// painting or blending the requested primitive.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::draw::draw_circle::DrawCircleOp;
///
/// let op = DrawCircleOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct DrawCircleOp<F: BandFormat> {
    cx: i32,
    cy: i32,
    radius: u32,
    ink: Vec<F::Sample>,
    fill: bool,
}

impl<F: BandFormat> DrawCircleOp<F> {
    /// Creates a new `DrawCircleOp`.
    pub fn new(
        cx: i32,
        cy: i32,
        radius: u32,
        ink: Vec<F::Sample>,
        fill: bool,
    ) -> Result<Self, ViprsError> {
        validate_ink(&ink)?;
        Ok(Self {
            cx,
            cy,
            radius,
            ink,
            fill,
        })
    }
}

impl<F: BandFormat> DrawOp<F> for DrawCircleOp<F> {
    fn draw(&self, tile: &mut TileMut<F>) {
        draw_circle_in_region(
            tile.data,
            tile.region,
            tile.bands,
            self.cx,
            self.cy,
            self.radius,
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
    fn draws_visible_circle_pixels() {
        let op = DrawCircleOp::<U8>::new(3, 3, 2, vec![4], false).unwrap();
        let mut pixels = vec![0_u8; 7 * 7];
        let mut tile = TileMut::new(Region::new(0, 0, 7, 7), 1, &mut pixels);

        op.draw(&mut tile);

        assert!(tile.data.contains(&4));
    }
}

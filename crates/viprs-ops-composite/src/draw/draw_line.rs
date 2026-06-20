use viprs_core::{draw::DrawOp, error::ViprsError, format::BandFormat, image::TileMut};

use super::{draw_line_in_region, validate_ink};

/// Applies the `draw line` drawing operation to an image. It updates a target region by
/// painting or blending the requested primitive.
///
/// # Examples
/// ```ignore
/// use viprs_ops_composite::draw::draw_line::DrawLineOp;
///
/// let op = DrawLineOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct DrawLineOp<F: BandFormat> {
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
    ink: Vec<F::Sample>,
}

impl<F: BandFormat> DrawLineOp<F> {
    /// Creates a new `DrawLineOp`.
    pub fn new(
        x1: i32,
        y1: i32,
        x2: i32,
        y2: i32,
        ink: Vec<F::Sample>,
    ) -> Result<Self, ViprsError> {
        validate_ink(&ink)?;
        Ok(Self {
            x1,
            y1,
            x2,
            y2,
            ink,
        })
    }
}

impl<F: BandFormat> DrawOp<F> for DrawLineOp<F> {
    fn draw(&self, tile: &mut TileMut<F>) {
        draw_line_in_region(
            tile.data,
            tile.region,
            tile.bands,
            self.x1,
            self.y1,
            self.x2,
            self.y2,
            &self.ink,
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
    fn draws_horizontal_line() {
        let op = DrawLineOp::<U8>::new(1, 2, 4, 2, vec![9]).unwrap();
        let mut pixels = vec![0_u8; 6 * 5];
        let mut tile = TileMut::new(Region::new(0, 0, 6, 5), 1, &mut pixels);

        op.draw(&mut tile);

        for x in 1..=4 {
            assert_eq!(tile.data[2 * 6 + x], 9);
        }
    }
}

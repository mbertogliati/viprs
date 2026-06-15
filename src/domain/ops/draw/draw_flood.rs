use crate::domain::{draw::DrawOp, error::ViprsError, format::BandFormat, image::TileMut};

use super::{draw_flood_in_region, validate_ink};

/// Applies the `draw flood` drawing operation to an image. It updates a target region by
/// painting or blending the requested primitive.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::draw::draw_flood::DrawFloodOp;
///
/// let op = DrawFloodOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct DrawFloodOp<F: BandFormat> {
    x: i32,
    y: i32,
    ink: Vec<F::Sample>,
}

impl<F: BandFormat> DrawFloodOp<F> {
    /// Creates a new `DrawFloodOp`.
    pub fn new(x: i32, y: i32, ink: Vec<F::Sample>) -> Result<Self, ViprsError> {
        validate_ink(&ink)?;
        Ok(Self { x, y, ink })
    }
}

impl<F> DrawOp<F> for DrawFloodOp<F>
where
    F: BandFormat,
    F::Sample: PartialEq,
{
    fn draw(&self, tile: &mut TileMut<F>) {
        draw_flood_in_region(
            tile.data,
            tile.region,
            tile.bands,
            self.x,
            self.y,
            &self.ink,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        format::U8,
        image::{Region, TileMut},
    };

    #[test]
    fn fills_connected_region() {
        let op = DrawFloodOp::<U8>::new(1, 1, vec![9]).unwrap();
        let mut pixels = vec![0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 2];
        let mut tile = TileMut::new(Region::new(0, 0, 4, 4), 1, &mut pixels);

        op.draw(&mut tile);

        assert_eq!(tile.data[5], 9);
        assert_eq!(tile.data[6], 9);
        assert_eq!(tile.data[9], 9);
        assert_eq!(tile.data[10], 9);
        assert_eq!(tile.data[15], 2);
    }
}

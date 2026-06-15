//! In-place draw-operation contracts for mutable image tiles.
//!
//! Draw ops skip the normal transform pipeline and instead mutate an already materialized tile
//! directly.

use crate::domain::{format::BandFormat, image::TileMut};

/// In-place image mutation interface for libvips-style draw operations.
///
/// Draw implementations use this trait when they need direct mutable access to a destination tile
/// rather than producing a new tile from an input tile.
///
/// # Examples
/// ```rust
/// # use viprs::domain::{draw::DrawOp, format::U8, image::{Region, TileMut}};
/// struct Fill;
/// impl DrawOp<U8> for Fill {
///     fn draw(&self, tile: &mut TileMut<U8>) {
///         tile.data.fill(1);
///     }
/// }
/// let mut pixels = [0_u8; 4];
/// let mut tile = TileMut::<U8>::new(Region::new(0, 0, 4, 1), 1, &mut pixels);
/// Fill.draw(&mut tile);
/// assert_eq!(tile.data, &[1, 1, 1, 1]);
/// ```
pub trait DrawOp<F: BandFormat>: Send + Sync {
    /// Mutate the target tile in place.
    fn draw(&self, tile: &mut TileMut<F>);
}

#[cfg(test)]
mod tests {
    use super::DrawOp;
    use crate::domain::{
        format::U8,
        image::{Region, TileMut},
        ops::draw::DrawLineOp,
    };

    fn apply(op: &impl DrawOp<U8>, tile: &mut TileMut<U8>) {
        op.draw(tile);
    }

    #[test]
    fn draw_op_trait_accepts_line_ops() {
        let op = DrawLineOp::<U8>::new(0, 0, 3, 0, vec![7]).unwrap();
        let mut pixels = vec![0_u8; 4];
        let mut tile = TileMut::new(Region::new(0, 0, 4, 1), 1, &mut pixels);

        apply(&op, &mut tile);

        assert_eq!(tile.data, &[7, 7, 7, 7]);
    }
}

use crate::domain::{
    error::ViprsError,
    format::F32,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

#[derive(Clone, Copy, Debug, PartialEq)]
/// Enumerates the available sdf shape values.
pub enum SdfShape {
    /// Uses the `Circle` variant of `SdfShape`.
    Circle {
        /// Stores the `center` value for this item.
        center: [f32; 2],
        /// Radius parameter associated with this mask profile.
        radius: f32,
    },
    /// Uses the `Rect` variant of `SdfShape`.
    Rect {
        /// Stores the `top_left` value for this item.
        top_left: [f32; 2],
        /// Stores the `bottom_right` value for this item.
        bottom_right: [f32; 2],
    },
}

/// Generate signed distance fields for basic libvips-compatible shapes.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::create::sdf::SdfOp;
///
/// let op = SdfOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SdfOp {
    width: u32,
    height: u32,
    shape: SdfShape,
}

impl SdfOp {
    /// Creates a new `SdfOp`.
    pub fn new(width: u32, height: u32, shape: SdfShape) -> Result<Self, ViprsError> {
        if width == 0 || height == 0 {
            return Err(ViprsError::Scheduler(format!(
                "SdfOp width and height must be > 0, got {width}x{height}"
            )));
        }

        match shape {
            SdfShape::Circle { center, radius } => {
                if !center.iter().all(|value| value.is_finite())
                    || !radius.is_finite()
                    || radius < 0.0
                {
                    return Err(ViprsError::Scheduler(format!(
                        "SdfOp circle requires finite center and radius >= 0, got center={center:?}, radius={radius}"
                    )));
                }
            }
            SdfShape::Rect {
                top_left,
                bottom_right,
            } => {
                if !top_left.iter().all(|value| value.is_finite())
                    || !bottom_right.iter().all(|value| value.is_finite())
                    || bottom_right[0] < top_left[0]
                    || bottom_right[1] < top_left[1]
                {
                    return Err(ViprsError::Scheduler(format!(
                        "SdfOp rect requires finite corners with bottom_right >= top_left, got top_left={top_left:?}, bottom_right={bottom_right:?}"
                    )));
                }
            }
        }

        Ok(Self {
            width,
            height,
            shape,
        })
    }

    #[must_use]
    /// Returns or performs width.
    pub const fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    /// Returns or performs height.
    pub const fn height(&self) -> u32 {
        self.height
    }
}

#[inline(always)]
fn sdf_circle(center: [f32; 2], radius: f32, x: f32, y: f32) -> f32 {
    (x - center[0]).hypot(y - center[1]) - radius
}

#[inline(always)]
fn sdf_rect(top_left: [f32; 2], bottom_right: [f32; 2], x: f32, y: f32) -> f32 {
    let center_x = (top_left[0] + bottom_right[0]) * 0.5;
    let center_y = (top_left[1] + bottom_right[1]) * 0.5;
    let half_width = (bottom_right[0] - top_left[0]) * 0.5;
    let half_height = (bottom_right[1] - top_left[1]) * 0.5;
    let dx = (x - center_x).abs() - half_width;
    let dy = (y - center_y).abs() - half_height;
    dx.max(0.0).hypot(dy.max(0.0)) + dx.max(dy).min(0.0)
}

impl Op for SdfOp {
    type Input = F32;
    type Output = F32;
    type State = ();

    const OUTPUT_BANDS: Option<usize> = Some(1);

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), _input: &Tile<F32>, output: &mut TileMut<F32>) {
        debug_assert_eq!(output.bands, 1, "SdfOp output must be single-band");
        debug_assert!(output.region.x >= 0 && output.region.y >= 0);
        debug_assert!(output.region.x as u32 + output.region.width <= self.width);
        debug_assert!(output.region.y as u32 + output.region.height <= self.height);

        let region_width = output.region.width as usize;

        for row in 0..output.region.height as usize {
            let y = output.region.y as f32 + row as f32;
            for col in 0..region_width {
                let x = output.region.x as f32 + col as f32;
                output.data[row * region_width + col] = match self.shape {
                    SdfShape::Circle { center, radius } => sdf_circle(center, radius, x, y),
                    SdfShape::Rect {
                        top_left,
                        bottom_right,
                    } => sdf_rect(top_left, bottom_right, x, y),
                };
            }
        }
    }
}

impl PixelLocalOp for SdfOp {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::image::{Region, Tile, TileMut};
    use proptest::prelude::*;

    fn render(op: &SdfOp) -> Vec<f32> {
        let region = Region::new(0, 0, op.width(), op.height());
        let input_data = vec![0.0f32; region.pixel_count()];
        let mut output_data = vec![0.0f32; region.pixel_count()];
        let input = Tile::<F32>::new(region, 1, &input_data);
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);
        output_data
    }

    #[test]
    fn constructor_rejects_invalid_geometry() {
        assert!(
            SdfOp::new(
                0,
                8,
                SdfShape::Circle {
                    center: [4.0, 4.0],
                    radius: 2.0
                }
            )
            .is_err()
        );
        assert!(
            SdfOp::new(
                8,
                0,
                SdfShape::Circle {
                    center: [4.0, 4.0],
                    radius: 2.0
                }
            )
            .is_err()
        );
        assert!(
            SdfOp::new(
                8,
                8,
                SdfShape::Circle {
                    center: [4.0, 4.0],
                    radius: -1.0
                }
            )
            .is_err()
        );
        assert!(
            SdfOp::new(
                8,
                8,
                SdfShape::Rect {
                    top_left: [5.0, 5.0],
                    bottom_right: [4.0, 6.0],
                },
            )
            .is_err()
        );
    }

    #[test]
    fn circle_is_negative_inside_and_positive_outside() {
        let op = SdfOp::new(
            9,
            9,
            SdfShape::Circle {
                center: [4.0, 4.0],
                radius: 3.0,
            },
        )
        .unwrap();
        let samples = render(&op);
        assert!(samples[4 * 9 + 4] < 0.0);
        assert!((samples[4 * 9 + 7]).abs() < 1e-6);
        assert!(samples[0] > 0.0);
    }

    #[test]
    fn rect_is_negative_inside_and_positive_outside() {
        let op = SdfOp::new(
            8,
            8,
            SdfShape::Rect {
                top_left: [2.0, 2.0],
                bottom_right: [5.0, 5.0],
            },
        )
        .unwrap();
        let samples = render(&op);
        assert!(samples[3 * 8 + 3] < 0.0);
        assert_eq!(samples[2 * 8 + 2], 0.0);
        assert!(samples[0] > 0.0);
    }

    proptest! {
        #[test]
        fn prop_output_is_finite_for_circles(
            width in 1u32..=32,
            height in 1u32..=32,
            radius in 0.0f32..=16.0,
            cx in 0.0f32..=32.0,
            cy in 0.0f32..=32.0,
        ) {
            let op = SdfOp::new(
                width,
                height,
                SdfShape::Circle {
                    center: [cx, cy],
                    radius,
                },
            )
            .unwrap();
            let samples = render(&op);
            prop_assert_eq!(samples.len(), width as usize * height as usize);
            prop_assert!(samples.iter().all(|sample| sample.is_finite()));
        }
    }
}

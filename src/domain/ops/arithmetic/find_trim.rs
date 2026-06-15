use crate::domain::{
    format::BandFormat,
    image::{Region, Tile},
    ops::resample::sample_conv::ToF64,
    reducer::TileReducer,
};

/// Bounding box of significant content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrimBox {
    /// Stores the `left` value for this item.
    pub left: u32,
    /// Stores the `top` value for this item.
    pub top: u32,
    /// Width associated with this item.
    pub width: u32,
    /// Height associated with this item.
    pub height: u32,
}

#[derive(Debug, Clone, Copy)]
/// Represents a find trim partial.
pub struct FindTrimPartial {
    found: bool,
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
}

/// Global reduction that finds the bounding box of non-background pixels.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::arithmetic::find_trim::FindTrimOp;
///
/// let op = FindTrimOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct FindTrimOp {
    threshold: f64,
    background: Vec<f64>,
    bands: usize,
}

impl FindTrimOp {
    #[must_use]
    /// Creates a new `FindTrimOp`.
    pub fn new(threshold: f64, background: Vec<f64>, bands: usize) -> Self {
        debug_assert!(bands > 0, "FindTrimOp: bands must be at least 1");
        debug_assert!(
            background.len() == 1 || background.len() == bands,
            "FindTrimOp: background must have length 1 or match bands"
        );
        Self {
            threshold,
            background,
            bands,
        }
    }

    fn sample_is_significant<S>(&self, samples: &[S]) -> bool
    where
        S: ToF64,
    {
        samples.iter().enumerate().any(|(band, sample)| {
            let background = if self.background.len() == 1 {
                self.background[0]
            } else {
                self.background[band]
            };
            (sample.to_f64() - background).abs() > self.threshold
        })
    }
}

impl<F> TileReducer<F> for FindTrimOp
where
    F: BandFormat,
    F::Sample: ToF64,
{
    type Partial = FindTrimPartial;
    type Output = TrimBox;
    type Scratch = ();

    fn reduce_tile(&self, tile: &Tile<F>, region: &Region) -> Self::Partial {
        debug_assert_eq!(tile.bands as usize, self.bands);
        let mut partial = FindTrimPartial {
            found: false,
            min_x: u32::MAX,
            min_y: u32::MAX,
            max_x: 0,
            max_y: 0,
        };

        let width = region.width as usize;
        let height = region.height as usize;
        for row in 0..height {
            for col in 0..width {
                let base = (row * width + col) * self.bands;
                if !self.sample_is_significant(&tile.data[base..base + self.bands]) {
                    continue;
                }

                let x = region.x.saturating_add(col as i32) as u32;
                let y = region.y.saturating_add(row as i32) as u32;
                partial.found = true;
                partial.min_x = partial.min_x.min(x);
                partial.min_y = partial.min_y.min(y);
                partial.max_x = partial.max_x.max(x);
                partial.max_y = partial.max_y.max(y);
            }
        }

        partial
    }

    fn combine(&self, a: Self::Partial, b: Self::Partial) -> Self::Partial {
        match (a.found, b.found) {
            (false | true, false) => a,
            (false, true) => b,
            (true, true) => FindTrimPartial {
                found: true,
                min_x: a.min_x.min(b.min_x),
                min_y: a.min_y.min(b.min_y),
                max_x: a.max_x.max(b.max_x),
                max_y: a.max_y.max(b.max_y),
            },
        }
    }

    fn finalize(&self, combined: Self::Partial) -> Self::Output {
        if !combined.found {
            return TrimBox {
                left: 0,
                top: 0,
                width: 0,
                height: 0,
            };
        }

        TrimBox {
            left: combined.min_x,
            top: combined.min_y,
            width: combined.max_x - combined.min_x + 1,
            height: combined.max_y - combined.min_y + 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{format::U8, reducer::TileReducer};

    #[test]
    fn finds_known_trim_box() {
        let reducer = FindTrimOp::new(0.0, vec![0.0], 1);
        let region = Region::new(0, 0, 5, 4);
        let data = vec![0u8, 0, 0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0];
        let tile = Tile::<U8>::new(region, 1, &data);
        let partial = reducer.reduce_tile(&tile, &region);
        let trim = <FindTrimOp as TileReducer<U8>>::finalize(&reducer, partial);
        assert_eq!(
            trim,
            TrimBox {
                left: 2,
                top: 1,
                width: 2,
                height: 2,
            }
        );
    }

    #[test]
    fn empty_content_returns_zero_box() {
        let reducer = FindTrimOp::new(0.0, vec![0.0], 1);
        let region = Region::new(0, 0, 2, 2);
        let data = vec![0u8; 4];
        let tile = Tile::<U8>::new(region, 1, &data);
        let partial = reducer.reduce_tile(&tile, &region);
        assert_eq!(
            <FindTrimOp as TileReducer<U8>>::finalize(&reducer, partial),
            TrimBox {
                left: 0,
                top: 0,
                width: 0,
                height: 0,
            }
        );
    }

    #[test]
    fn single_non_background_pixel_returns_unit_box_at_offset() {
        let reducer = FindTrimOp::new(0.0, vec![0.0], 1);
        let region = Region::new(10, 20, 4, 3);
        let data = vec![0u8, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0];
        let tile = Tile::<U8>::new(region, 1, &data);
        let partial = reducer.reduce_tile(&tile, &region);

        assert_eq!(
            <FindTrimOp as TileReducer<U8>>::finalize(&reducer, partial),
            TrimBox {
                left: 11,
                top: 22,
                width: 1,
                height: 1,
            }
        );
    }

    #[test]
    fn per_band_background_and_threshold_control_significance() {
        let reducer = FindTrimOp::new(0.5, vec![10.0, 20.0], 2);
        let region = Region::new(0, 0, 2, 1);
        let data = vec![10u8, 20, 10, 21];
        let tile = Tile::<U8>::new(region, 2, &data);
        let partial = reducer.reduce_tile(&tile, &region);

        assert_eq!(
            <FindTrimOp as TileReducer<U8>>::finalize(&reducer, partial),
            TrimBox {
                left: 1,
                top: 0,
                width: 1,
                height: 1,
            }
        );
    }

    #[test]
    fn combine_merges_found_and_missing_partials() {
        let reducer = FindTrimOp::new(0.0, vec![0.0], 1);
        let empty = FindTrimPartial {
            found: false,
            min_x: u32::MAX,
            min_y: u32::MAX,
            max_x: 0,
            max_y: 0,
        };
        let found = FindTrimPartial {
            found: true,
            min_x: 4,
            min_y: 2,
            max_x: 6,
            max_y: 7,
        };

        let combined_left = <FindTrimOp as TileReducer<U8>>::combine(&reducer, empty, found);
        assert!(combined_left.found);
        assert_eq!(combined_left.min_x, found.min_x);
        assert_eq!(combined_left.min_y, found.min_y);
        assert_eq!(combined_left.max_x, found.max_x);
        assert_eq!(combined_left.max_y, found.max_y);

        let combined_right = <FindTrimOp as TileReducer<U8>>::combine(&reducer, found, empty);
        assert!(combined_right.found);
        assert_eq!(combined_right.min_x, found.min_x);
        assert_eq!(combined_right.min_y, found.min_y);
        assert_eq!(combined_right.max_x, found.max_x);
        assert_eq!(combined_right.max_y, found.max_y);
    }
}

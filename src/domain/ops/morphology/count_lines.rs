use crate::domain::{
    format::U8,
    image::{Region, Tile},
    reducer::TileReducer,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Enumerates the available count lines direction values.
pub enum CountLinesDirection {
    /// Uses the `Horizontal` variant of `CountLinesDirection`.
    Horizontal,
    /// Uses the `Vertical` variant of `CountLinesDirection`.
    Vertical,
}

/// Applies the `count lines` morphological operation to the image. Use it for
/// neighbourhood-based shape filtering and mask analysis.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::morphology::count_lines::CountLinesOp;
///
/// let op = CountLinesOp { /* operation parameters */ };
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct CountLinesOp {
    width: u32,
    height: u32,
    direction: CountLinesDirection,
}

/// Enumerates the available count lines partial values.
pub enum CountLinesPartial {
    /// Partial result for a tile that covered the whole input image.
    FullImage {
        /// Image region that was reduced.
        region: Region,
        /// Mean line count measured across the region.
        mean: f64,
    },
    /// Marker used when reduction was attempted over multiple tiles.
    Invalid {
        /// Number of tiles seen by the reducer.
        tile_count: usize,
    },
}

impl CountLinesOp {
    #[must_use]
    /// Creates a new `CountLinesOp`.
    pub const fn new(width: u32, height: u32, direction: CountLinesDirection) -> Self {
        Self {
            width,
            height,
            direction,
        }
    }

    #[must_use]
    /// Returns or performs horizontal.
    pub const fn horizontal(width: u32, height: u32) -> Self {
        Self::new(width, height, CountLinesDirection::Horizontal)
    }

    #[must_use]
    /// Returns or performs vertical.
    pub const fn vertical(width: u32, height: u32) -> Self {
        Self::new(width, height, CountLinesDirection::Vertical)
    }

    const fn full_region(&self) -> Region {
        Region::new(0, 0, self.width, self.height)
    }

    #[inline]
    fn mean_line_count(&self, input: &Tile<U8>) -> f64 {
        let width = input.region.width as usize;
        let height = input.region.height as usize;
        debug_assert_eq!(
            input.bands, 1,
            "CountLinesOp expects a single-band U8 image"
        );
        debug_assert_eq!(width, self.width as usize);
        debug_assert_eq!(height, self.height as usize);

        match self.direction {
            CountLinesDirection::Horizontal => {
                let mut total = 0usize;
                for x in 0..width {
                    let mut previous_white = false;
                    for y in 0..height {
                        let white = input.data[y * width + x] >= 128;
                        if white && !previous_white {
                            total += 1;
                        }
                        previous_white = white;
                    }
                }
                total as f64 / width as f64
            }
            CountLinesDirection::Vertical => {
                let mut total = 0usize;
                for y in 0..height {
                    let mut previous_white = false;
                    for x in 0..width {
                        let white = input.data[y * width + x] >= 128;
                        if white && !previous_white {
                            total += 1;
                        }
                        previous_white = white;
                    }
                }
                total as f64 / height as f64
            }
        }
    }
}

impl TileReducer<U8> for CountLinesOp {
    type Partial = CountLinesPartial;
    type Output = f64;
    type Scratch = ();

    fn reduce_tile(&self, tile: &Tile<U8>, region: &Region) -> Self::Partial {
        let full_region = self.full_region();
        if tile.region == full_region && *region == full_region {
            CountLinesPartial::FullImage {
                region: *region,
                mean: self.mean_line_count(tile),
            }
        } else {
            // Countlines averages whole-image run counts; splitting at tile borders
            // would change the answer because runs can cross tile boundaries.
            CountLinesPartial::Invalid { tile_count: 1 }
        }
    }

    fn combine(&self, a: Self::Partial, b: Self::Partial) -> Self::Partial {
        match (a, b) {
            (
                CountLinesPartial::Invalid {
                    tile_count: left_tiles,
                },
                CountLinesPartial::Invalid {
                    tile_count: right_tiles,
                },
            ) => CountLinesPartial::Invalid {
                tile_count: left_tiles + right_tiles,
            },
            (CountLinesPartial::FullImage { .. }, CountLinesPartial::FullImage { .. }) => {
                CountLinesPartial::Invalid { tile_count: 2 }
            }
            (CountLinesPartial::FullImage { .. }, CountLinesPartial::Invalid { tile_count })
            | (CountLinesPartial::Invalid { tile_count }, CountLinesPartial::FullImage { .. }) => {
                CountLinesPartial::Invalid {
                    tile_count: tile_count + 1,
                }
            }
        }
    }

    fn finalize(&self, combined: Self::Partial) -> Self::Output {
        match combined {
            CountLinesPartial::FullImage { region, mean } => {
                debug_assert_eq!(region, self.full_region());
                mean
            }
            CountLinesPartial::Invalid { tile_count } => {
                debug_assert!(
                    false,
                    "CountLinesOp requires a single full-image tile, got {tile_count} tiles"
                );
                f64::NAN
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn run_reducer(width: u32, height: u32, direction: CountLinesDirection, data: &[u8]) -> f64 {
        let op = CountLinesOp::new(width, height, direction);
        let region = Region::new(0, 0, width, height);
        let input = Tile::<U8>::new(region, 1, data);
        let partial = op.reduce_tile(&input, &region);
        op.finalize(partial)
    }

    fn reference_count_lines(
        width: u32,
        height: u32,
        direction: CountLinesDirection,
        data: &[u8],
    ) -> f64 {
        let width = width as usize;
        let height = height as usize;

        match direction {
            CountLinesDirection::Horizontal => {
                let mut total = 0usize;
                for x in 0..width {
                    let mut previous_white = false;
                    for y in 0..height {
                        let white = data[y * width + x] >= 128;
                        if white && !previous_white {
                            total += 1;
                        }
                        previous_white = white;
                    }
                }
                total as f64 / width as f64
            }
            CountLinesDirection::Vertical => {
                let mut total = 0usize;
                for y in 0..height {
                    let mut previous_white = false;
                    for x in 0..width {
                        let white = data[y * width + x] >= 128;
                        if white && !previous_white {
                            total += 1;
                        }
                        previous_white = white;
                    }
                }
                total as f64 / height as f64
            }
        }
    }

    fn transpose(width: u32, height: u32, data: &[u8]) -> Vec<u8> {
        let width = width as usize;
        let height = height as usize;
        let mut transposed = vec![0u8; data.len()];
        for y in 0..height {
            for x in 0..width {
                transposed[x * height + y] = data[y * width + x];
            }
        }
        transposed
    }

    #[test]
    fn reducer_returns_scalar_average_for_known_input() {
        let data = [
            0, 0, 255, 255, //
            0, 255, 255, 0, //
            0, 0, 0, 0, //
        ];

        let horizontal = run_reducer(4, 3, CountLinesDirection::Horizontal, &data);
        let vertical = run_reducer(4, 3, CountLinesDirection::Vertical, &data);

        assert!((horizontal - 0.75).abs() < 1e-12);
        assert!((vertical - (2.0 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn threshold_is_black_below_128_and_white_at_or_above_128() {
        let vertical = run_reducer(4, 1, CountLinesDirection::Vertical, &[0, 127, 128, 255]);
        assert!((vertical - 1.0).abs() < 1e-12);
    }

    #[test]
    fn horizontal_and_vertical_directions_match_reference_examples() {
        let data = [
            0, 0, 255, 255, //
            0, 255, 255, 0, //
            0, 0, 0, 0, //
        ];

        let horizontal = run_reducer(4, 3, CountLinesDirection::Horizontal, &data);
        let vertical = run_reducer(4, 3, CountLinesDirection::Vertical, &data);
        let horizontal_expected =
            reference_count_lines(4, 3, CountLinesDirection::Horizontal, &data);
        let vertical_expected = reference_count_lines(4, 3, CountLinesDirection::Vertical, &data);

        assert!((horizontal - horizontal_expected).abs() < 1e-12);
        assert!((vertical - vertical_expected).abs() < 1e-12);
    }

    #[test]
    fn transpose_swaps_horizontal_and_vertical_counts() {
        let width = 3;
        let height = 4;
        let data = [
            0, 255, 0, //
            0, 255, 255, //
            0, 0, 255, //
            255, 255, 0, //
        ];
        let transposed = transpose(width, height, &data);

        let horizontal = run_reducer(width, height, CountLinesDirection::Horizontal, &data);
        let vertical_on_transpose =
            run_reducer(height, width, CountLinesDirection::Vertical, &transposed);

        assert!((horizontal - vertical_on_transpose).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "CountLinesOp requires a single full-image tile")]
    fn reducer_rejects_multi_tile_usage() {
        let op = CountLinesOp::horizontal(4, 2);
        let left_region = Region::new(0, 0, 2, 2);
        let right_region = Region::new(2, 0, 2, 2);
        let left = Tile::<U8>::new(left_region, 1, &[0, 255, 0, 255]);
        let right = Tile::<U8>::new(right_region, 1, &[255, 0, 255, 0]);

        let combined = op.combine(
            op.reduce_tile(&left, &left_region),
            op.reduce_tile(&right, &right_region),
        );

        let _ = op.finalize(combined);
    }

    proptest! {
        #[test]
        fn count_lines_random_inputs_match_reference_and_transpose_symmetry(
            width in 1u32..=8,
            height in 1u32..=8,
            data in proptest::collection::vec(any::<u8>(), 1..=64),
        ) {
            let pixel_count = width as usize * height as usize;
            prop_assume!(pixel_count <= data.len());
            let input = &data[..pixel_count];
            let transposed = transpose(width, height, input);

            let horizontal = run_reducer(width, height, CountLinesDirection::Horizontal, input);
            let horizontal_expected =
                reference_count_lines(width, height, CountLinesDirection::Horizontal, input);
            let vertical = run_reducer(width, height, CountLinesDirection::Vertical, input);
            let vertical_expected =
                reference_count_lines(width, height, CountLinesDirection::Vertical, input);

            prop_assert!((horizontal - horizontal_expected).abs() < 1e-12);
            prop_assert!((vertical - vertical_expected).abs() < 1e-12);
            let vertical_on_transpose =
                run_reducer(height, width, CountLinesDirection::Vertical, &transposed);
            prop_assert!((horizontal - vertical_on_transpose).abs() < 1e-12);
        }
    }
}

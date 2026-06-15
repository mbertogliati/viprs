//! `TileReducer` port — ops that produce scalar or aggregate outputs from images.
//!
//! Design rationale for reducers.
//! Scratch-state APIs eliminate per-tile heap allocations.

use crate::domain::{
    format::BandFormat,
    image::{Region, Tile},
};

/// Accumulates per-tile partial results and combines them into a final aggregate.
///
/// # Design
///
/// `min`, `max`, `avg`, `hist_find` and similar operations do not produce pixel
/// buffers — they produce scalar values or histograms. They are NOT pipeline nodes.
/// Instead, a `TileReducer` runs as a separate pass over a materialized pixel buffer
/// (e.g., the output of a `MemorySink`) using rayon `fold` + `reduce`.
///
/// This keeps the pipeline scheduler single-phase and avoids shared mutable state:
/// each thread produces its own `Partial` value; `combine` merges them in a tree.
///
/// # Thread safety contract
///
/// - `reduce_tile` is called once per tile per thread and returns an **owned** `Partial`.
///   No shared state, no locks.
/// - `combine` is called to merge two `Partial` values. It must be **associative** and
///   **commutative** (rayon does not guarantee call order).
/// - `finalize` is called exactly once on the fully merged partial.
///
/// # Zero-allocation hot path
///
/// `reduce_tile` allocates a fresh `Partial` per tile. For reducers with large
/// accumulators (histograms, Hough accumulators, per-band stats), this is a per-tile
/// heap allocation on the hot path — exactly what rule P1 prohibits.
///
/// The scratch-state API eliminates this:
/// - `type Scratch: Default + Send` — a per-thread buffer initialized once via
///   `Default::default()` and reused across tiles.
/// - `accumulate_into(&self, tile, region, scratch)` — accumulates the tile into
///   `scratch` in place, clearing it first. No allocation if the scratch `Vec`
///   already has the required capacity.
/// - The scheduler calls `Default::default()` once per rayon fold partition (once
///   per thread), then `accumulate_into` per tile, then `combine` for cross-thread
///   merging.
///
/// Reducers that do not need scratch storage set `type Scratch = ()` and the
/// default `accumulate_into` body falls through to `reduce_tile` + `combine`.
///
/// # Multi-source reducers
///
/// `TileReducer` remains intentionally single-source: the scheduler reduces the
/// output tile of the compiled pipeline and nothing else. Reducers that need a
/// synchronized side input should implement [`BiSourceReducer`] and own a
/// prevalidated secondary representation (for example a materialized mask or
/// index buffer) inside the reducer itself. `TileReducer` remains the scheduler
/// contract; reducers with side inputs typically delegate `reduce_tile` to
/// `BiSourceReducer::reduce_tile_with_side_input`.
pub trait TileReducer<F: BandFormat>: Send + Sync {
    /// Per-tile partial result. Must be `Send` to cross rayon thread boundaries.
    ///
    /// For scalar stats: a plain struct (e.g., `PartialStats { min, max, sum, count }`).
    /// For histograms: a `Vec<u64>` of bins — ideally reused per worker via
    /// `accumulate_into`.
    type Partial: Send;

    /// Final output type produced after all partials have been combined.
    type Output;

    /// Per-thread scratch buffer. Initialized once per rayon fold partition via
    /// `Default::default()` and passed mutably to `accumulate_into` per tile.
    ///
    /// For reducers with large accumulators (e.g., histogram bins), set this to the
    /// accumulator type (e.g., `Vec<u64>`) and implement `accumulate_into` to
    /// `clear()` + refill without re-allocating. Reducers that do not need a scratch
    /// buffer set this to `()`.
    ///
    type Scratch: Default + Send;

    /// Compute a partial result from one tile.
    ///
    /// `tile.data` contains the interleaved samples for the given `region`.
    /// Implementations iterate over samples but must not allocate per pixel.
    ///
    /// **Hot-path note**: the default implementation allocates per tile. Reducers with
    /// large accumulators should override `accumulate_into` instead and leave this as
    /// a thin wrapper. See `accumulate_into` docs.
    fn reduce_tile(&self, tile: &Tile<F>, region: &Region) -> Self::Partial;

    /// Accumulate one tile into `scratch` in place without allocating.
    ///
    /// `scratch` is a `Self::Scratch` owned by the calling rayon thread, initialized
    /// with `Default::default()` once per fold partition. The implementor must reset
    /// any tile-specific state inside `scratch` at the start of this call (e.g.,
    /// `Vec::clear()` + refill), then merge the tile into `partial`.
    ///
    /// The default implementation creates a fresh partial via `reduce_tile` and
    /// combines it into `partial`. This preserves backwards compatibility but does not
    /// eliminate the per-tile allocation. Override this method to achieve zero-alloc.
    ///
    fn accumulate_into(
        &self,
        tile: &Tile<F>,
        region: &Region,
        scratch: &mut Self::Scratch,
        partial: &mut Option<Self::Partial>,
    ) {
        let _ = scratch; // unused in default path
        let tile_partial = self.reduce_tile(tile, region);
        *partial = Some(match partial.take() {
            Some(existing) => self.combine(existing, tile_partial),
            None => tile_partial,
        });
    }

    /// Accumulate one tile into a reusable partial.
    ///
    /// The default implementation preserves the original contract by materializing
    /// `reduce_tile(tile, region)` and merging it into `partial`. Reducers with
    /// sizeable accumulators should override `accumulate_into` instead.
    fn accumulate_tile(
        &self,
        partial: &mut Option<Self::Partial>,
        tile: &Tile<F>,
        region: &Region,
    ) {
        let tile_partial = self.reduce_tile(tile, region);
        *partial = Some(match partial.take() {
            Some(existing) => self.combine(existing, tile_partial),
            None => tile_partial,
        });
    }

    /// Merge two partial results into one.
    ///
    /// Must be associative: `combine(combine(a, b), c) == combine(a, combine(b, c))`.
    /// Must be commutative: rayon does not guarantee reduction order.
    fn combine(&self, a: Self::Partial, b: Self::Partial) -> Self::Partial;

    /// Produce the final output from the fully combined partial.
    ///
    /// Called exactly once. May allocate freely — this is outside the hot path.
    fn finalize(&self, combined: Self::Partial) -> Self::Output;
}

/// Helper trait for reducers that combine the main tile with one synchronized side input.
///
/// This formalizes the chosen single-source approach: the scheduler stays single-source while the
/// reducer owns any secondary input it needs. That secondary state is prepared
/// once at reducer construction time and reused for each tile reduction; there
/// is no per-tile source lookup or extra scheduler plumbing.
///
/// Implementors should ensure that their internal side input is aligned with the
/// same image-space coordinates as the primary tile stream. Any validation or
/// materialization needed to uphold that invariant must happen before the first
/// call to `reduce_tile_with_side_input`.
pub trait BiSourceReducer<F: BandFormat>: Send + Sync {
    /// Per-tile partial result.
    type Partial: Send;

    /// Final aggregate result after all partials are combined.
    type Output;

    /// Compute a partial result using one primary tile plus the synchronized side input.
    fn reduce_tile_with_side_input(&self, tile: &Tile<F>, region: &Region) -> Self::Partial;

    /// Merge two partial results into one.
    fn combine(&self, a: Self::Partial, b: Self::Partial) -> Self::Partial;

    /// Produce the final output from the fully combined partial.
    fn finalize(&self, combined: Self::Partial) -> Self::Output;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        format::U8,
        image::{Region, Tile},
    };

    /// A minimal reducer that counts all pixels (ergonomic API demo).
    struct PixelCounter;

    impl TileReducer<U8> for PixelCounter {
        type Partial = u64;
        type Output = u64;
        type Scratch = ();

        fn reduce_tile(&self, tile: &Tile<U8>, _region: &Region) -> u64 {
            tile.data.len() as u64
        }

        fn combine(&self, a: u64, b: u64) -> u64 {
            a + b
        }

        fn finalize(&self, combined: u64) -> u64 {
            combined
        }
    }

    #[test]
    fn pixel_counter_counts_correctly() {
        let region = Region::new(0, 0, 4, 2);
        let data = vec![10u8; 8]; // 4×2 single-band
        let tile = Tile::<U8>::new(region, 1, &data);
        let reducer = PixelCounter;
        let partial = reducer.reduce_tile(&tile, &region);
        assert_eq!(partial, 8);
        let combined = reducer.combine(partial, 4);
        assert_eq!(reducer.finalize(combined), 12);
    }

    struct ScratchAwareCounter;

    impl TileReducer<U8> for ScratchAwareCounter {
        type Partial = u64;
        type Output = u64;
        type Scratch = Vec<u8>;

        fn reduce_tile(&self, tile: &Tile<U8>, _region: &Region) -> Self::Partial {
            tile.data.len() as u64
        }

        fn combine(&self, a: Self::Partial, b: Self::Partial) -> Self::Partial {
            a + b
        }

        fn finalize(&self, combined: Self::Partial) -> Self::Output {
            combined
        }
    }

    #[test]
    fn default_accumulate_into_initializes_partial_combines_tiles_and_preserves_scratch() {
        let reducer = ScratchAwareCounter;
        let mut scratch = vec![9u8, 8, 7];
        let mut partial = None;

        let first_region = Region::new(0, 0, 2, 1);
        let first_data = vec![1u8, 2];
        let first_tile = Tile::<U8>::new(first_region, 1, &first_data);

        reducer.accumulate_into(&first_tile, &first_region, &mut scratch, &mut partial);

        assert_eq!(partial, Some(2));
        assert_eq!(scratch, vec![9u8, 8, 7]);

        let second_region = Region::new(0, 0, 1, 3);
        let second_data = vec![3u8, 4, 5];
        let second_tile = Tile::<U8>::new(second_region, 1, &second_data);

        reducer.accumulate_into(&second_tile, &second_region, &mut scratch, &mut partial);

        assert_eq!(partial, Some(5));
        assert_eq!(scratch, vec![9u8, 8, 7]);
    }

    #[test]
    fn default_accumulate_tile_combines_partial_results_across_calls() {
        let reducer = ScratchAwareCounter;
        let mut partial = None;

        let first_region = Region::new(0, 0, 3, 1);
        let first_data = vec![1u8, 2, 3];
        let first_tile = Tile::<U8>::new(first_region, 1, &first_data);
        reducer.accumulate_tile(&mut partial, &first_tile, &first_region);
        assert_eq!(partial, Some(3));

        let second_region = Region::new(0, 0, 2, 2);
        let second_data = vec![4u8, 5, 6, 7];
        let second_tile = Tile::<U8>::new(second_region, 1, &second_data);
        reducer.accumulate_tile(&mut partial, &second_tile, &second_region);

        assert_eq!(partial, Some(7));
        assert_eq!(
            reducer.finalize(
                partial
                    .take()
                    .expect("tests may unwrap the accumulated partial")
            ),
            7
        );
    }

    struct MaskedPixelCounter {
        width: usize,
        mask: Vec<u8>,
    }

    impl BiSourceReducer<U8> for MaskedPixelCounter {
        type Partial = u64;
        type Output = u64;

        fn reduce_tile_with_side_input(&self, tile: &Tile<U8>, region: &Region) -> Self::Partial {
            let row_width = region.width as usize;
            let mut total = 0u64;

            for row in 0..region.height as usize {
                let y = region.y as usize + row;
                let secondary_row = y * self.width;
                let tile_row = row * row_width;
                for col in 0..row_width {
                    if self.mask[secondary_row + col + region.x as usize] != 0 {
                        total += u64::from(tile.data[tile_row + col]);
                    }
                }
            }

            total
        }

        fn combine(&self, a: Self::Partial, b: Self::Partial) -> Self::Partial {
            a + b
        }

        fn finalize(&self, combined: Self::Partial) -> Self::Output {
            combined
        }
    }

    impl TileReducer<U8> for MaskedPixelCounter {
        type Partial = <Self as BiSourceReducer<U8>>::Partial;
        type Output = <Self as BiSourceReducer<U8>>::Output;
        type Scratch = ();

        fn reduce_tile(&self, tile: &Tile<U8>, region: &Region) -> Self::Partial {
            <Self as BiSourceReducer<U8>>::reduce_tile_with_side_input(self, tile, region)
        }

        fn combine(&self, a: Self::Partial, b: Self::Partial) -> Self::Partial {
            <Self as BiSourceReducer<U8>>::combine(self, a, b)
        }

        fn finalize(&self, combined: Self::Partial) -> Self::Output {
            <Self as BiSourceReducer<U8>>::finalize(self, combined)
        }
    }

    #[test]
    fn bisource_reducer_tile_reducer_delegation_uses_synchronized_side_input() {
        let region = Region::new(1, 0, 2, 2);
        let data = vec![3u8, 4, 5, 6];
        let tile = Tile::<U8>::new(region, 1, &data);
        let reducer = MaskedPixelCounter {
            width: 4,
            mask: vec![0, 1, 0, 0, 1, 0, 1, 0],
        };

        let partial =
            <MaskedPixelCounter as TileReducer<U8>>::reduce_tile(&reducer, &tile, &region);

        assert_eq!(
            partial, 9,
            "mask selects samples 3 and 6 from the synchronized tile"
        );
        assert_eq!(
            <MaskedPixelCounter as TileReducer<U8>>::finalize(&reducer, partial),
            9
        );
    }

    #[test]
    fn bisource_reducer_accumulate_tile_uses_tile_reducer_combine_delegation() {
        let reducer = MaskedPixelCounter {
            width: 2,
            mask: vec![1, 1],
        };
        let mut partial = None;

        let left_region = Region::new(0, 0, 1, 1);
        let left_data = vec![7u8];
        let left_tile = Tile::<U8>::new(left_region, 1, &left_data);
        <MaskedPixelCounter as TileReducer<U8>>::accumulate_tile(
            &reducer,
            &mut partial,
            &left_tile,
            &left_region,
        );
        assert_eq!(partial, Some(7));

        let right_region = Region::new(1, 0, 1, 1);
        let right_data = vec![9u8];
        let right_tile = Tile::<U8>::new(right_region, 1, &right_data);
        <MaskedPixelCounter as TileReducer<U8>>::accumulate_tile(
            &reducer,
            &mut partial,
            &right_tile,
            &right_region,
        );

        assert_eq!(partial, Some(16));
        assert_eq!(
            <MaskedPixelCounter as TileReducer<U8>>::finalize(
                &reducer,
                partial
                    .take()
                    .expect("tests may unwrap the accumulated partial"),
            ),
            16
        );
    }
}

//! Reducers for building histogram-equalization lookup tables.

use crate::domain::error::ViprsError;
use crate::{
    domain::reducer::TileReducer,
    domain::{
        format::BandFormat,
        image::{Region, Tile},
    },
};

/// Builds the lookup table used by global histogram equalization.
///
/// This reducer counts the distribution of one input band and converts the cumulative histogram
/// into an output LUT that can later remap intensities across the full `u8` range.
///
/// # Examples
/// ```rust
/// use viprs::domain::{
///     format::U8,
///     image::{Region, Tile},
///     reducer::TileReducer,
///     reducers::HistEqualReducer,
/// };
///
/// let reducer = HistEqualReducer::new(1, 0, 256).unwrap();
/// let region = Region::new(0, 0, 4, 1);
/// let tile = Tile::<U8>::new(region, 1, &[0, 64, 128, 255]);
/// let partial = reducer.reduce_tile(&tile, &region);
/// let lut = reducer.finalize(partial);
///
/// assert_eq!(lut.len(), 256);
/// ```
pub struct HistEqualReducer {
    input_bands: u32,
    /// Stores the `band` value for this item.
    pub band: u32,
    /// Stores the `bin_count` value for this item.
    pub bin_count: usize,
}

impl HistEqualReducer {
    /// Creates a histogram-equalization reducer for one band of an input image.
    ///
    /// This constructor validates the selected band ahead of execution so tile reducers can stay
    /// branch-light while they accumulate histogram bins.
    ///
    /// # Examples
    /// ```rust
    /// use viprs::domain::reducers::HistEqualReducer;
    ///
    /// let reducer = HistEqualReducer::new(3, 1, 256).unwrap();
    /// assert_eq!(reducer.band, 1);
    /// ```
    pub fn new(input_bands: u32, band: u32, bin_count: usize) -> Result<Self, ViprsError> {
        if input_bands == 0 {
            return Err(ViprsError::Scheduler(
                "hist_equal requires at least one band".into(),
            ));
        }
        if band >= input_bands {
            return Err(ViprsError::Scheduler(format!(
                "hist_equal band {band} is out of range for {input_bands}-band input"
            )));
        }

        Ok(Self {
            input_bands,
            band,
            bin_count,
        })
    }
}

impl<F: BandFormat> TileReducer<F> for HistEqualReducer
where
    F::Sample: Into<f64> + Copy,
{
    type Partial = Vec<u64>;
    type Output = Vec<u8>;
    /// Pre-allocated bin buffer reused across tiles on the same rayon thread.
    type Scratch = Vec<u64>;

    fn reduce_tile(&self, tile: &Tile<F>, _region: &Region) -> Vec<u64> {
        let mut bins = vec![0u64; self.bin_count];
        if self.bin_count == 0 {
            return bins;
        }

        if tile.bands != self.input_bands {
            debug_assert_eq!(
                tile.bands, self.input_bands,
                "HistEqualReducer tile band count must match validated constructor input",
            );
            return bins;
        }

        let bands = self.input_bands as usize;
        let band = self.band as usize;

        for chunk in tile.data.chunks(bands) {
            if let Some(&sample) = chunk.get(band) {
                let value = sample.into().round();
                let bin = value.clamp(0.0, (self.bin_count - 1) as f64) as usize;
                bins[bin] += 1;
            }
        }

        bins
    }

    /// Zero-allocation tile accumulation using a pre-allocated bin buffer.
    ///
    /// `scratch` is resized on the first call and zeroed at the start of each tile.
    /// No allocation after the first call per rayon thread.
    fn accumulate_into(
        &self,
        tile: &Tile<F>,
        _region: &Region,
        scratch: &mut Self::Scratch,
        partial: &mut Option<Self::Partial>,
    ) {
        if self.bin_count == 0 {
            // Nothing to accumulate; ensure partial exists with empty bins.
            if partial.is_none() {
                *partial = Some(Vec::new());
            }
            return;
        }

        scratch.resize(self.bin_count, 0u64);
        scratch.fill(0u64);

        if tile.bands != self.input_bands {
            debug_assert_eq!(
                tile.bands, self.input_bands,
                "HistEqualReducer tile band count must match validated constructor input",
            );
            if partial.is_none() {
                *partial = Some(vec![0u64; self.bin_count]);
            }
            return;
        }

        let bands = self.input_bands as usize;
        let band = self.band as usize;

        for chunk in tile.data.chunks(bands) {
            if let Some(&sample) = chunk.get(band) {
                let value = sample.into().round();
                let bin = value.clamp(0.0, (self.bin_count - 1) as f64) as usize;
                scratch[bin] += 1;
            }
        }

        match partial {
            Some(acc) => {
                for (ai, si) in acc.iter_mut().zip(scratch.iter()) {
                    *ai += si;
                }
            }
            None => {
                *partial = Some(scratch.clone());
            }
        }
    }

    fn combine(&self, mut a: Vec<u64>, b: Vec<u64>) -> Vec<u64> {
        for (lhs, rhs) in a.iter_mut().zip(b.iter()) {
            *lhs += rhs;
        }
        a
    }

    fn finalize(&self, bins: Vec<u64>) -> Vec<u8> {
        if bins.is_empty() {
            return Vec::new();
        }

        let total: u64 = bins.iter().sum();
        if total == 0 {
            return identity_lut(bins.len());
        }

        if bins.iter().filter(|&&count| count > 0).take(2).count() <= 1 {
            return identity_lut(bins.len());
        }

        let first_non_zero = bins.iter().position(|&count| count > 0).unwrap_or(0);
        let cdf_min = bins[first_non_zero];

        let denom = (total - cdf_min) as f64;
        let mut lut = Vec::with_capacity(bins.len());
        let mut cdf = 0u64;
        for count in bins {
            cdf += count;
            let value = ((cdf.saturating_sub(cdf_min)) as f64 / denom * f64::from(u8::MAX))
                .round()
                .clamp(0.0, f64::from(u8::MAX)) as u8;
            lut.push(value);
        }

        lut
    }
}

fn identity_lut(bin_count: usize) -> Vec<u8> {
    if bin_count <= 1 {
        return vec![0u8; bin_count];
    }

    let scale = f64::from(u8::MAX) / (bin_count - 1) as f64;
    (0..bin_count)
        .map(|idx| (idx as f64 * scale).round().clamp(0.0, f64::from(u8::MAX)) as u8)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{format::U8, image::Region};
    use proptest::prelude::*;

    #[test]
    fn hist_equal_reducer_uniform_histogram_produces_identity_lut() {
        let reducer = HistEqualReducer::new(1, 0, 256).unwrap();
        let lut = <HistEqualReducer as TileReducer<U8>>::finalize(&reducer, vec![1u64; 256]);
        assert_eq!(lut, (0u8..=255u8).collect::<Vec<_>>());
    }

    #[test]
    fn histequal_1x1_preserves_value() {
        let reducer = HistEqualReducer::new(1, 0, 256).unwrap();
        let mut bins = vec![0u64; 256];
        bins[128] = 1;

        let lut = <HistEqualReducer as TileReducer<U8>>::finalize(&reducer, bins);

        assert_eq!(lut, (0u8..=255u8).collect::<Vec<_>>());
    }

    #[test]
    fn histequal_uniform_image_preserves_value() {
        let reducer = HistEqualReducer::new(1, 0, 256).unwrap();
        let mut bins = vec![0u64; 256];
        bins[128] = 10_000;

        let lut = <HistEqualReducer as TileReducer<U8>>::finalize(&reducer, bins);

        assert_eq!(lut, (0u8..=255u8).collect::<Vec<_>>());
    }

    #[test]
    fn histequal_two_value_image_spreads_to_full_range() {
        let reducer = HistEqualReducer::new(1, 0, 256).unwrap();
        let mut bins = vec![0u64; 256];
        bins[64] = 50;
        bins[192] = 50;

        let lut = <HistEqualReducer as TileReducer<U8>>::finalize(&reducer, bins);

        assert_eq!(lut[64], 0);
        assert_eq!(lut[192], 255);
    }

    #[test]
    fn hist_equal_reducer_non_empty_histogram_reaches_full_range() {
        let reducer = HistEqualReducer::new(1, 0, 256).unwrap();
        let mut bins = vec![0u64; 256];
        bins[64] = 10;
        bins[128] = 20;
        bins[255] = 5;
        let lut = <HistEqualReducer as TileReducer<U8>>::finalize(&reducer, bins);
        assert_eq!(lut[255], 255);
        // All LUT values are u8, so they are always ≤ u8::MAX by definition.
        assert!(!lut.is_empty());
    }

    #[test]
    fn hist_equal_reducer_reduce_tile_counts_band_values() {
        let reducer = HistEqualReducer::new(1, 0, 256).unwrap();
        let region = Region::new(0, 0, 4, 1);
        let data = vec![0u8, 64, 64, 255];
        let tile = Tile::<U8>::new(region, 1, &data);
        let bins = <HistEqualReducer as TileReducer<U8>>::reduce_tile(&reducer, &tile, &region);
        assert_eq!(bins[0], 1);
        assert_eq!(bins[64], 2);
        assert_eq!(bins[255], 1);
    }

    #[test]
    fn hist_equal_reducer_rejects_zero_band_inputs() {
        let err = match HistEqualReducer::new(0, 0, 256) {
            Ok(_) => panic!("HistEqualReducer must reject zero-band inputs at construction"),
            Err(err) => err,
        };

        assert!(
            matches!(
                err,
                crate::domain::error::ViprsError::Scheduler(ref message)
                if message == "hist_equal requires at least one band"
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn hist_equal_reducer_rejects_out_of_range_band_index() {
        let err = match HistEqualReducer::new(2, 2, 256) {
            Ok(_) => panic!("HistEqualReducer must reject band indices outside the input range"),
            Err(err) => err,
        };

        assert!(
            matches!(
                err,
                crate::domain::error::ViprsError::Scheduler(ref message)
                if message == "hist_equal band 2 is out of range for 2-band input"
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn hist_equal_reducer_reduce_tile_returns_empty_bins_when_bin_count_is_zero() {
        let reducer = HistEqualReducer::new(1, 0, 0).unwrap();
        let region = Region::new(0, 0, 2, 1);
        let data = vec![3u8, 7u8];
        let tile = Tile::<U8>::new(region, 1, &data);

        let bins = <HistEqualReducer as TileReducer<U8>>::reduce_tile(&reducer, &tile, &region);

        assert!(bins.is_empty());
    }

    #[test]
    fn hist_equal_reducer_accumulate_into_initializes_empty_partial_for_zero_bins() {
        let reducer = HistEqualReducer::new(1, 0, 0).unwrap();
        let region = Region::new(0, 0, 1, 1);
        let data = vec![42u8];
        let tile = Tile::<U8>::new(region, 1, &data);
        let mut scratch = vec![99u64];
        let mut partial = None;

        <HistEqualReducer as TileReducer<U8>>::accumulate_into(
            &reducer,
            &tile,
            &region,
            &mut scratch,
            &mut partial,
        );

        assert_eq!(partial, Some(Vec::new()));
        assert_eq!(scratch, vec![99u64]);
    }

    #[test]
    fn hist_equal_reducer_accumulate_into_merges_multiple_tiles_for_selected_band() {
        let reducer = HistEqualReducer::new(2, 1, 4).unwrap();
        let region_a = Region::new(0, 0, 3, 1);
        let data_a = vec![9u8, 1, 9, 2, 9, 2];
        let tile_a = Tile::<U8>::new(region_a, 2, &data_a);
        let region_b = Region::new(0, 0, 2, 1);
        let data_b = vec![9u8, 3, 9, 3];
        let tile_b = Tile::<U8>::new(region_b, 2, &data_b);
        let mut scratch = Vec::new();
        let mut partial = None;

        <HistEqualReducer as TileReducer<U8>>::accumulate_into(
            &reducer,
            &tile_a,
            &region_a,
            &mut scratch,
            &mut partial,
        );
        assert_eq!(partial, Some(vec![0u64, 1, 2, 0]));
        assert_eq!(scratch, vec![0u64, 1, 2, 0]);

        <HistEqualReducer as TileReducer<U8>>::accumulate_into(
            &reducer,
            &tile_b,
            &region_b,
            &mut scratch,
            &mut partial,
        );

        assert_eq!(partial, Some(vec![0u64, 1, 2, 2]));
        assert_eq!(scratch, vec![0u64, 0, 0, 2]);
    }

    #[test]
    fn hist_equal_reducer_combine_adds_matching_bins() {
        let reducer = HistEqualReducer::new(1, 0, 3).unwrap();
        let combined = <HistEqualReducer as TileReducer<U8>>::combine(
            &reducer,
            vec![1u64, 2, 3],
            vec![4u64, 5, 6],
        );

        assert_eq!(combined, vec![5u64, 7, 9]);
    }

    #[test]
    fn hist_equal_reducer_finalize_empty_bins_returns_empty_lut() {
        let reducer = HistEqualReducer::new(1, 0, 0).unwrap();
        let lut = <HistEqualReducer as TileReducer<U8>>::finalize(&reducer, Vec::new());

        assert!(lut.is_empty());
    }

    #[test]
    fn hist_equal_reducer_finalize_zero_total_returns_identity_lut() {
        let reducer = HistEqualReducer::new(1, 0, 4).unwrap();
        let lut = <HistEqualReducer as TileReducer<U8>>::finalize(&reducer, vec![0u64; 4]);

        assert_eq!(lut, vec![0u8, 85, 170, 255]);
    }

    #[test]
    fn hist_equal_identity_lut_handles_zero_and_single_bin_ranges() {
        assert_eq!(identity_lut(0), Vec::<u8>::new());
        assert_eq!(identity_lut(1), vec![0u8]);
    }

    proptest! {
        #[test]
        fn hist_equal_reducer_uniform_histogram_identity_prop(count in 1u8..=32u8) {
            let reducer = HistEqualReducer::new(1, 0, 256).unwrap();
            let lut = <HistEqualReducer as TileReducer<U8>>::finalize(&reducer, vec![u64::from(count); 256]);
            prop_assert_eq!(lut, (0u8..=255u8).collect::<Vec<_>>());
        }
    }
}

//! Reducers for one-dimensional and multi-dimensional image histograms.

use crate::{
    domain::reducer::TileReducer,
    domain::{
        format::{BandFormat, BandFormatId},
        image::{Region, Tile},
        ops::resample::sample_conv::ToF64,
        stats::Histogram,
    },
};

/// Computes a frequency histogram for a single image band.
///
/// This reducer solves per-band distribution analysis by accumulating tile-local counts and
/// packaging them into a [`Histogram`] value with the original band metadata attached.
///
/// # Examples
/// ```ignore
/// use viprs::domain::{
///     format::{BandFormatId, U8},
///     image::{Region, Tile},
///     reducer::TileReducer,
///     reducers::HistFindReducer,
/// };
///
/// let reducer = HistFindReducer::new(0, 4, BandFormatId::U8);
/// let region = Region::new(0, 0, 4, 1);
/// let tile = Tile::<U8>::new(region, 1, &[0, 64, 128, 255]);
/// let histogram = reducer.finalize(reducer.reduce_tile(&tile, &region));
///
/// assert_eq!(histogram.total(), 4);
/// ```
pub struct HistFindReducer {
    /// Stores the `band` value for this item.
    pub band: u32,
    /// Stores the `bin_count` value for this item.
    pub bin_count: usize,
    /// Band format associated with this condition.
    pub format: BandFormatId,
}

/// Stores the dense bin volume produced by an N-dimensional histogram reduction.
///
/// This result type keeps the histogram shape alongside its flattened bin buffer so callers can
/// interpret 1D, 2D, and 3D histograms without extra metadata channels.
///
/// # Examples
/// ```rust
/// use viprs::domain::reducers::HistFindNDimResult;
///
/// let result = HistFindNDimResult {
///     width: 2,
///     height: 2,
///     bands: 1,
///     bins: vec![1, 2, 3, 4],
/// };
///
/// assert_eq!(result.total(), 10);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistFindNDimResult {
    /// Width associated with this item.
    pub width: u32,
    /// Height associated with this item.
    pub height: u32,
    /// Number of bands associated with this item.
    pub bands: u32,
    /// Stores the `bins` value for this item.
    pub bins: Vec<u64>,
}

impl HistFindNDimResult {
    /// Returns the total number of samples accumulated across all bins.
    ///
    /// This helper solves the common need to validate histogram completeness without forcing
    /// callers to manually sum the flattened storage buffer.
    ///
    /// # Examples
    /// ```rust
    /// use viprs::domain::reducers::HistFindNDimResult;
    ///
    /// let result = HistFindNDimResult {
    ///     width: 1,
    ///     height: 1,
    ///     bands: 2,
    ///     bins: vec![2, 3],
    /// };
    ///
    /// assert_eq!(result.total(), 5);
    /// ```
    #[must_use]
    pub fn total(&self) -> u64 {
        self.bins.iter().sum()
    }
}

/// Computes 1D, 2D, or 3D histograms over the leading image bands.
///
/// This reducer solves joint-distribution analysis for grayscale, two-band, and RGB-style images
/// by projecting each sample tuple into a dense histogram volume.
///
/// # Examples
/// ```ignore
/// use viprs::domain::{
///     format::U8,
///     image::{Region, Tile},
///     reducer::TileReducer,
///     reducers::HistFindNDimReducer,
/// };
///
/// let reducer = HistFindNDimReducer::new(3, 2, u8::MAX as u32);
/// let region = Region::new(0, 0, 1, 1);
/// let tile = Tile::<U8>::new(region, 3, &[255, 0, 0]);
/// let histogram = reducer.finalize(reducer.reduce_tile(&tile, &region));
///
/// assert_eq!(histogram.total(), 1);
/// ```
pub struct HistFindNDimReducer {
    input_bands: u32,
    bins_per_axis: u32,
    max_sample_value: u32,
}

impl HistFindNDimReducer {
    /// Default number of bins used for each axis when matching libvips defaults.
    ///
    /// This constant lets callers request the conventional joint-histogram resolution without
    /// repeating magic numbers at call sites.
    ///
    /// # Examples
    /// ```rust
    /// use viprs::domain::reducers::HistFindNDimReducer;
    ///
    /// assert_eq!(HistFindNDimReducer::DEFAULT_BINS_PER_AXIS, 10);
    /// ```
    pub const DEFAULT_BINS_PER_AXIS: u32 = 10;

    /// Creates an N-dimensional histogram reducer for the leading `input_bands`.
    ///
    /// This constructor defines the output histogram shape up front so each tile can quantize
    /// samples into the correct dense bin volume with no dynamic shape discovery.
    ///
    /// # Examples
    /// ```rust
    /// use viprs::domain::reducers::HistFindNDimReducer;
    ///
    /// let reducer = HistFindNDimReducer::new(3, 8, u8::MAX as u32);
    /// let _ = reducer;
    /// ```
    #[must_use]
    pub fn new(input_bands: u32, bins_per_axis: u32, max_sample_value: u32) -> Self {
        debug_assert!(
            (1..=3).contains(&input_bands),
            "HistFindNDimReducer: input_bands must be in 1..=3"
        );
        debug_assert!(
            bins_per_axis > 0 && bins_per_axis <= max_sample_value + 1,
            "HistFindNDimReducer: bins_per_axis must fit sample range"
        );
        Self {
            input_bands,
            bins_per_axis,
            max_sample_value,
        }
    }

    /// Creates an N-dimensional histogram reducer using the libvips default axis resolution.
    ///
    /// This helper solves the common case where callers want standard histogram sizing while
    /// still specifying the number of input bands and the sample range explicitly.
    ///
    /// # Examples
    /// ```rust
    /// use viprs::domain::reducers::HistFindNDimReducer;
    ///
    /// let reducer = HistFindNDimReducer::with_default_bins(3, u8::MAX as u32);
    /// let _ = reducer;
    /// ```
    #[must_use]
    pub fn with_default_bins(input_bands: u32, max_sample_value: u32) -> Self {
        Self::new(input_bands, Self::DEFAULT_BINS_PER_AXIS, max_sample_value)
    }

    const fn output_height(&self) -> u32 {
        if self.input_bands > 1 {
            self.bins_per_axis
        } else {
            1
        }
    }

    const fn output_bands(&self) -> u32 {
        if self.input_bands > 2 {
            self.bins_per_axis
        } else {
            1
        }
    }

    pub(crate) fn empty_partial(&self) -> HistFindNDimResult {
        let width = self.bins_per_axis;
        let height = self.output_height();
        let bands = self.output_bands();
        HistFindNDimResult {
            width,
            height,
            bands,
            bins: vec![0u64; width as usize * height as usize * bands as usize],
        }
    }

    fn accumulate_partial<F>(
        &self,
        partial: &mut HistFindNDimResult,
        tile: &Tile<F>,
        region: &Region,
    ) where
        F: BandFormat,
        F::Sample: ToF64,
    {
        let width = partial.width;
        let bands = partial.bands;
        let input_bands = self.input_bands as usize;

        for row in 0..region.height as usize {
            for col in 0..region.width as usize {
                let base = (row * region.width as usize + col) * input_bands;
                let x = ndim_axis(tile.data[base], self.bins_per_axis, self.max_sample_value);
                let y = if input_bands > 1 {
                    ndim_axis(
                        tile.data[base + 1],
                        self.bins_per_axis,
                        self.max_sample_value,
                    )
                } else {
                    0
                };
                let z = if input_bands > 2 {
                    ndim_axis(
                        tile.data[base + 2],
                        self.bins_per_axis,
                        self.max_sample_value,
                    )
                } else {
                    0
                };
                let idx = (y * width * bands + x * bands + z) as usize;
                partial.bins[idx] += 1;
            }
        }
    }
}

impl HistFindReducer {
    /// Construct a histogram reducer for the given band.
    ///
    /// This constructor solves single-band histogram setup by storing the selected band,
    /// quantization size, and output metadata together in one reducer value.
    ///
    /// # Examples
    /// ```rust
    /// use viprs::domain::{format::BandFormatId, reducers::HistFindReducer};
    ///
    /// let reducer = HistFindReducer::new(1, 16, BandFormatId::U8);
    /// assert_eq!(reducer.bin_count, 16);
    /// ```
    #[must_use]
    pub const fn new(band: u32, bin_count: usize, format: BandFormatId) -> Self {
        Self {
            band,
            bin_count,
            format,
        }
    }
}

impl<F: BandFormat> TileReducer<F> for HistFindReducer
where
    F::Sample: Into<f64> + Copy,
{
    /// Per-tile bin counts. Allocated once per tile (not per pixel) by the default path.
    /// When using the scratch-state API, the same `Vec<u64>` is reused across tiles.
    type Partial = Vec<u64>;
    type Output = Histogram;
    /// Pre-allocated bin buffer reused across tiles by `accumulate_into`.
    type Scratch = Vec<u64>;

    fn reduce_tile(&self, tile: &Tile<F>, _region: &Region) -> Vec<u64> {
        let bands = tile.bands as usize;
        let band = self.band as usize;
        let bin_count = self.bin_count;
        let mut bins = vec![0u64; bin_count];

        // Iterate only over the samples belonging to `self.band`.
        for chunk in tile.data.chunks(bands) {
            if let Some(&sample) = chunk.get(band) {
                let v: f64 = sample.into();
                // Quantize into [0, bin_count). Clamp to guard against float rounding.
                let bin = ((v / f64::from(u8::MAX)) * (bin_count as f64)) as usize;
                let bin = bin.min(bin_count - 1);
                bins[bin] += 1;
            }
        }

        bins
    }

    /// Zero-allocation tile accumulation. `scratch` holds a `Vec<u64>` that is
    /// re-used across tiles on the same rayon thread. On the first call the `Vec`
    /// is empty (from `Default`); it is resized to `bin_count` via `resize`, which
    /// only allocates on that first call. Subsequent calls `fill` it with zeros
    /// and accumulate in place — no allocator involvement.
    fn accumulate_into(
        &self,
        tile: &Tile<F>,
        _region: &Region,
        scratch: &mut Self::Scratch,
        partial: &mut Option<Self::Partial>,
    ) {
        let bands = tile.bands as usize;
        let band = self.band as usize;
        let bin_count = self.bin_count;

        // Resize only on first call for this thread; noop if already correct size.
        scratch.resize(bin_count, 0u64);
        // Reset counts without deallocating.
        scratch.fill(0u64);

        for chunk in tile.data.chunks(bands) {
            if let Some(&sample) = chunk.get(band) {
                let v: f64 = sample.into();
                let bin = ((v / f64::from(u8::MAX)) * (bin_count as f64)) as usize;
                let bin = bin.min(bin_count - 1);
                scratch[bin] += 1;
            }
        }

        // Merge scratch counts into the thread-local partial accumulator.
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
        for (ai, bi) in a.iter_mut().zip(b.iter()) {
            *ai += bi;
        }
        a
    }

    fn finalize(&self, bins: Vec<u64>) -> Histogram {
        Histogram {
            format: self.format,
            band: self.band,
            bins,
        }
    }
}

impl<F> TileReducer<F> for HistFindNDimReducer
where
    F: BandFormat,
    F::Sample: ToF64,
{
    type Partial = HistFindNDimResult;
    type Output = HistFindNDimResult;
    type Scratch = ();

    fn reduce_tile(&self, tile: &Tile<F>, region: &Region) -> Self::Partial {
        let mut partial = self.empty_partial();
        self.accumulate_partial(&mut partial, tile, region);
        partial
    }

    fn accumulate_tile(
        &self,
        partial: &mut Option<Self::Partial>,
        tile: &Tile<F>,
        region: &Region,
    ) {
        let partial = partial.get_or_insert_with(|| self.empty_partial());
        self.accumulate_partial(partial, tile, region);
    }

    fn combine(&self, mut a: Self::Partial, b: Self::Partial) -> Self::Partial {
        for (left, right) in a.bins.iter_mut().zip(b.bins) {
            *left += right;
        }
        a
    }

    fn finalize(&self, combined: Self::Partial) -> Self::Output {
        combined
    }
}

fn ndim_axis<S: ToF64>(sample: S, bins_per_axis: u32, max_sample_value: u32) -> u32 {
    let value = sample.to_f64();
    if !value.is_finite() {
        return 0;
    }
    let scale = (f64::from(max_sample_value) + 1.0) / f64::from(bins_per_axis);
    ((value.clamp(0.0, f64::from(max_sample_value)) / scale) as u32).min(bins_per_axis - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        format::{F32, U8},
        image::Region,
        reducer::TileReducer,
    };
    use proptest::prelude::*;

    #[test]
    fn histogram_bins_sum_to_pixel_count() {
        let region = Region::new(0, 0, 4, 2);
        let data = vec![0u8, 128, 255, 64, 0, 128, 255, 64];
        let tile = Tile::<U8>::new(region, 1, &data);
        let reducer = HistFindReducer::new(0, 256, BandFormatId::U8);
        let partial = <HistFindReducer as crate::domain::reducer::TileReducer<U8>>::reduce_tile(
            &reducer, &tile, &region,
        );
        let hist = <HistFindReducer as crate::domain::reducer::TileReducer<U8>>::finalize(
            &reducer, partial,
        );
        assert_eq!(hist.total(), 8);
    }

    #[test]
    fn ndim_histogram_maps_rgb_samples_to_xyz_histogram() {
        let region = Region::new(0, 0, 2, 1);
        let tile = Tile::<U8>::new(region, 3, &[0, 0, 0, 255, 255, 255]);
        let reducer = HistFindNDimReducer::new(3, 2, u8::MAX as u32);
        let partial = reducer.reduce_tile(&tile, &region);
        let hist = <HistFindNDimReducer as TileReducer<U8>>::finalize(&reducer, partial);

        assert_eq!(hist.width, 2);
        assert_eq!(hist.height, 2);
        assert_eq!(hist.bands, 2);
        assert_eq!(hist.bins[0], 1);
        assert_eq!(hist.bins[7], 1);
        assert_eq!(hist.total(), 2);
    }

    #[test]
    fn ndim_histogram_default_matches_libvips_ten_bin_default() {
        let reducer = HistFindNDimReducer::with_default_bins(3, u8::MAX as u32);
        let partial = reducer.empty_partial();

        assert_eq!(HistFindNDimReducer::DEFAULT_BINS_PER_AXIS, 10);
        assert_eq!(partial.width, 10);
        assert_eq!(partial.height, 10);
        assert_eq!(partial.bands, 10);
    }

    #[test]
    fn hist_find_reducer_counts_selected_band_and_ignores_missing_band() {
        let region = Region::new(0, 0, 2, 1);
        let tile = Tile::<U8>::new(region, 2, &[5, 0, 7, 255]);

        let selected_band = HistFindReducer::new(1, 4, BandFormatId::U8);
        let selected_bins = selected_band.reduce_tile(&tile, &region);
        assert_eq!(selected_bins, vec![1, 0, 0, 1]);

        let missing_band = HistFindReducer::new(3, 4, BandFormatId::U8);
        let missing_bins = missing_band.reduce_tile(&tile, &region);
        assert_eq!(missing_bins, vec![0, 0, 0, 0]);
    }

    #[test]
    fn hist_find_reducer_accumulate_into_reuses_scratch_and_merges_partials() {
        let region = Region::new(0, 0, 2, 1);
        let reducer = HistFindReducer::new(0, 4, BandFormatId::U8);
        let first_tile = Tile::<U8>::new(region, 1, &[0, 255]);
        let second_tile = Tile::<U8>::new(region, 1, &[64, 64]);

        let mut scratch = vec![99_u64];
        let mut partial = None;

        reducer.accumulate_into(&first_tile, &region, &mut scratch, &mut partial);
        assert_eq!(scratch, vec![1, 0, 0, 1]);
        assert_eq!(partial, Some(vec![1, 0, 0, 1]));

        reducer.accumulate_into(&second_tile, &region, &mut scratch, &mut partial);
        assert_eq!(scratch, vec![0, 2, 0, 0]);
        assert_eq!(partial, Some(vec![1, 2, 0, 1]));
    }

    #[test]
    fn hist_find_reducer_combine_and_finalize_preserve_histogram_metadata() {
        let reducer = HistFindReducer::new(2, 4, BandFormatId::U8);
        let combined = <HistFindReducer as TileReducer<U8>>::combine(
            &reducer,
            vec![1, 0, 2, 0],
            vec![0, 3, 1, 4],
        );
        let histogram = <HistFindReducer as TileReducer<U8>>::finalize(&reducer, combined);

        assert_eq!(histogram.format, BandFormatId::U8);
        assert_eq!(histogram.band, 2);
        assert_eq!(histogram.bins, vec![1, 3, 3, 4]);
        assert_eq!(histogram.total(), 11);
    }

    #[test]
    fn ndim_histogram_shape_depends_on_input_band_count() {
        let one_band = HistFindNDimReducer::new(1, 4, u8::MAX as u32).empty_partial();
        assert_eq!(one_band.width, 4);
        assert_eq!(one_band.height, 1);
        assert_eq!(one_band.bands, 1);

        let two_band = HistFindNDimReducer::new(2, 4, u8::MAX as u32).empty_partial();
        assert_eq!(two_band.width, 4);
        assert_eq!(two_band.height, 4);
        assert_eq!(two_band.bands, 1);
    }

    #[test]
    fn ndim_histogram_clamps_non_finite_negative_and_high_samples() {
        let region = Region::new(0, 0, 2, 1);
        let tile = Tile::<F32>::new(
            region,
            3,
            &[f32::NAN, -10.0, f32::INFINITY, 128.0, 255.0, 300.0],
        );
        let reducer = HistFindNDimReducer::new(3, 4, u8::MAX as u32);
        let partial = reducer.reduce_tile(&tile, &region);

        assert_eq!(partial.bins[0], 1);
        assert_eq!(partial.bins[59], 1);
        assert_eq!(partial.total(), 2);
    }

    #[test]
    fn ndim_histogram_accumulate_tile_combine_and_finalize_sum_counts() {
        let region = Region::new(0, 0, 2, 1);
        let tile = Tile::<F32>::new(region, 2, &[0.0, 255.0, 255.0, 0.0]);
        let reducer = HistFindNDimReducer::new(2, 2, u8::MAX as u32);
        let mut partial = None;

        reducer.accumulate_tile(&mut partial, &tile, &region);
        reducer.accumulate_tile(&mut partial, &tile, &region);

        let accumulated = partial.expect("accumulate_tile should initialize the partial histogram");
        assert_eq!(accumulated.bins, vec![0, 2, 2, 0]);

        let combined = <HistFindNDimReducer as TileReducer<F32>>::combine(
            &reducer,
            accumulated.clone(),
            reducer.reduce_tile(&tile, &region),
        );
        let finalized = <HistFindNDimReducer as TileReducer<F32>>::finalize(&reducer, combined);

        assert_eq!(finalized.width, 2);
        assert_eq!(finalized.height, 2);
        assert_eq!(finalized.bands, 1);
        assert_eq!(finalized.bins, vec![0, 3, 3, 0]);
        assert_eq!(finalized.total(), 6);
    }

    proptest! {
        #[test]
        fn ndim_histogram_preserves_total_count(samples in proptest::collection::vec(any::<u8>(), 3..=96)) {
            let pixel_count = samples.len() / 3;
            let pixels = samples[..pixel_count * 3].to_vec();
            let region = Region::new(0, 0, pixel_count as u32, 1);
            let tile = Tile::<U8>::new(region, 3, &pixels);
            let reducer = HistFindNDimReducer::new(3, 4, u8::MAX as u32);
            let partial = reducer.reduce_tile(&tile, &region);
            let hist = <HistFindNDimReducer as TileReducer<U8>>::finalize(&reducer, partial);

            prop_assert_eq!(hist.total(), pixel_count as u64);
        }
    }
}

use viprs_core::{
    format::BandFormat,
    image::{Region, Tile},
    reducer::TileReducer,
    shared_ops::sample_conv::ToF64,
};

#[derive(Debug, Clone, PartialEq, Eq)]
/// Represents a hist find result.
pub struct HistFindResult {
    /// Width associated with this item.
    pub width: u32,
    /// Number of bands associated with this item.
    pub bands: u32,
    /// Stores the `bins` value for this item.
    pub bins: Vec<u64>,
}

impl HistFindResult {
    #[must_use]
    /// Returns or performs height.
    pub const fn height(&self) -> u32 {
        1
    }

    #[must_use]
    /// Returns or performs total.
    pub fn total(&self) -> u64 {
        self.bins.iter().sum()
    }
}

#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistFindPartial {
    bands: u32,
    bins: Vec<u64>,
    max_bin: usize,
}

/// Applies the `histogram search` histogram operation to the image. It derives histogram-based
/// measurements or adjustments from the input samples.
///
/// # Examples
/// ```ignore
/// use viprs_ops_colour::histogram::hist_find_facades::HistFindOp;
///
/// let op = HistFindOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct HistFindOp {
    input_bands: u32,
    band: Option<u32>,
    bins_per_band: usize,
    fixed_width: bool,
}

impl HistFindOp {
    #[must_use]
    /// Creates a new `HistFindOp`.
    pub fn new(input_bands: u32, band: Option<u32>, bins_per_band: usize) -> Self {
        debug_assert!(
            input_bands > 0,
            "HistFindOp: input_bands must be at least 1"
        );
        debug_assert!(
            band.is_none_or(|band| band < input_bands),
            "HistFindOp: band must be within input band count"
        );
        debug_assert!(
            bins_per_band > 0,
            "HistFindOp: bins_per_band must be at least 1"
        );
        Self {
            input_bands,
            band,
            bins_per_band,
            fixed_width: false,
        }
    }

    #[must_use]
    /// Returns or performs for format.
    pub fn for_format(input_bands: u32, band: Option<u32>, max_sample_value: u32) -> Self {
        let fixed_width = band.is_none() && max_sample_value == u32::from(u8::MAX);
        Self {
            fixed_width,
            ..Self::new(input_bands, band, max_sample_value as usize + 1)
        }
    }

    const fn output_bands(&self) -> usize {
        if self.band.is_some() {
            1
        } else {
            self.input_bands as usize
        }
    }

    fn empty_partial(&self) -> HistFindPartial {
        HistFindPartial {
            bands: self.output_bands() as u32,
            bins: vec![0u64; self.bins_per_band * self.output_bands()],
            max_bin: if self.fixed_width {
                self.bins_per_band.saturating_sub(1)
            } else {
                0
            },
        }
    }

    fn accumulate_partial<F>(&self, partial: &mut HistFindPartial, tile: &Tile<F>, region: &Region)
    where
        F: BandFormat,
        F::Sample: ToF64,
    {
        let input_bands = self.input_bands as usize;
        let output_bands = partial.bands as usize;

        for row in 0..region.height as usize {
            for col in 0..region.width as usize {
                let base = (row * region.width as usize + col) * input_bands;
                match self.band {
                    Some(band) => {
                        let bin = sample_bin(tile.data[base + band as usize], self.bins_per_band);
                        partial.bins[bin] += 1;
                        partial.max_bin = partial.max_bin.max(bin);
                    }
                    None => {
                        for band in 0..input_bands {
                            let bin = sample_bin(tile.data[base + band], self.bins_per_band);
                            partial.bins[bin * output_bands + band] += 1;
                            partial.max_bin = partial.max_bin.max(bin);
                        }
                    }
                }
            }
        }
    }
}

impl<F> TileReducer<F> for HistFindOp
where
    F: BandFormat,
    F::Sample: ToF64,
{
    type Partial = HistFindPartial;
    type Output = HistFindResult;
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

    fn accumulate_into(
        &self,
        tile: &Tile<F>,
        region: &Region,
        _scratch: &mut Self::Scratch,
        partial: &mut Option<Self::Partial>,
    ) {
        let partial = partial.get_or_insert_with(|| self.empty_partial());
        self.accumulate_partial(partial, tile, region);
    }

    fn combine(&self, mut a: Self::Partial, b: Self::Partial) -> Self::Partial {
        for (left, right) in a.bins.iter_mut().zip(b.bins) {
            *left += right;
        }
        a.max_bin = a.max_bin.max(b.max_bin);
        a
    }

    fn finalize(&self, combined: Self::Partial) -> Self::Output {
        let width = combined.max_bin + 1;
        let bins = combined.bins[..width * combined.bands as usize].to_vec();
        HistFindResult {
            width: width as u32,
            bands: combined.bands,
            bins,
        }
    }
}

fn sample_bin<S: ToF64>(sample: S, bins_per_band: usize) -> usize {
    let max_bin = bins_per_band - 1;
    let value = sample.to_f64();
    if !value.is_finite() {
        return 0;
    }
    value.clamp(0.0, max_bin as f64).round() as usize
}

#[cfg(all(test, feature = "_integration"))]
mod tests {
    use super::*;
    use crate::histogram::HistFindNDimOp;
    use viprs_core::{format::U8, reducer::TileReducer};

    #[test]
    fn hist_find_all_bands_interleaves_bins_like_image_output() {
        let region = Region::new(0, 0, 2, 1);
        let tile = Tile::<U8>::new(region, 2, &[0, 1, 0, 2]);
        let reducer = HistFindOp::for_format(2, None, u8::MAX as u32);
        let partial = reducer.reduce_tile(&tile, &region);
        let hist = <HistFindOp as TileReducer<U8>>::finalize(&reducer, partial);

        assert_eq!(hist.width, 256);
        assert_eq!(hist.height(), 1);
        assert_eq!(hist.bands, 2);
        assert_eq!(hist.total(), 4);
        assert_eq!(hist.bins[0], 2);
        assert_eq!(hist.bins[1], 0);
        assert_eq!(hist.bins[3], 1);
        assert_eq!(hist.bins[5], 1);
    }

    #[test]
    fn hist_find_selected_band_outputs_one_band_histogram() {
        let region = Region::new(0, 0, 2, 1);
        let tile = Tile::<U8>::new(region, 2, &[0, 1, 0, 2]);
        let reducer = HistFindOp::for_format(2, Some(1), u8::MAX as u32);
        let partial = reducer.reduce_tile(&tile, &region);
        let hist = <HistFindOp as TileReducer<U8>>::finalize(&reducer, partial);

        assert_eq!(hist.width, 3);
        assert_eq!(hist.height(), 1);
        assert_eq!(hist.bands, 1);
        assert_eq!(hist.bins[1], 1);
        assert_eq!(hist.bins[2], 1);
        assert_eq!(hist.total(), 2);
    }

    #[test]
    fn hist_find_u16_trims_output_width_to_highest_seen_bin() {
        let region = Region::new(0, 0, 3, 1);
        let tile = Tile::<viprs_core::format::U16>::new(region, 1, &[0, 4, 4]);
        let reducer = HistFindOp::for_format(1, None, u16::MAX as u32);
        let partial = reducer.reduce_tile(&tile, &region);
        let hist =
            <HistFindOp as TileReducer<viprs_core::format::U16>>::finalize(&reducer, partial);

        assert_eq!(hist.width, 5);
        assert_eq!(hist.height(), 1);
        assert_eq!(hist.bands, 1);
        assert_eq!(hist.bins, vec![1, 0, 0, 0, 2]);
    }

    #[test]
    fn hist_find_ndim_maps_rgb_samples_to_xyz_histogram() {
        let region = Region::new(0, 0, 2, 1);
        let tile = Tile::<U8>::new(region, 3, &[0, 0, 0, 255, 255, 255]);
        let reducer = HistFindNDimOp::new(3, 2, u8::MAX as u32);
        let partial = reducer.reduce_tile(&tile, &region);
        let hist = <HistFindNDimOp as TileReducer<U8>>::finalize(&reducer, partial);

        assert_eq!(hist.width, 2);
        assert_eq!(hist.height, 2);
        assert_eq!(hist.bands, 2);
        assert_eq!(hist.bins[0], 1);
        assert_eq!(hist.bins[7], 1);
        assert_eq!(hist.total(), 2);
    }

    #[test]
    fn hist_find_ndim_default_matches_libvips_ten_bin_default() {
        let reducer = HistFindNDimOp::with_default_bins(3, u8::MAX as u32);
        let partial = reducer.empty_partial();

        assert_eq!(HistFindNDimOp::DEFAULT_BINS_PER_AXIS, 10);
        assert_eq!(partial.width, 10);
        assert_eq!(partial.height, 10);
        assert_eq!(partial.bands, 10);
    }

    #[test]
    fn hist_find_accumulate_tile_reuses_same_bin_buffer() {
        let region = Region::new(0, 0, 1, 1);
        let first = Tile::<U8>::new(region, 1, &[0]);
        let second = Tile::<U8>::new(region, 1, &[1]);
        let reducer = HistFindOp::for_format(1, None, u8::MAX as u32);
        let mut partial = None;

        reducer.accumulate_tile(&mut partial, &first, &region);
        let first_ptr = partial.as_ref().unwrap().bins.as_ptr();

        reducer.accumulate_tile(&mut partial, &second, &region);
        let partial = partial.unwrap();

        assert_eq!(first_ptr, partial.bins.as_ptr());
        assert_eq!(partial.bins[0], 1);
        assert_eq!(partial.bins[1], 1);
        assert_eq!(partial.bins.iter().sum::<u64>(), 2);
    }

    #[test]
    fn hist_find_accumulate_into_reuses_same_bin_buffer() {
        let region = Region::new(0, 0, 1, 1);
        let first = Tile::<U8>::new(region, 1, &[0]);
        let second = Tile::<U8>::new(region, 1, &[1]);
        let reducer = HistFindOp::for_format(1, None, u8::MAX as u32);
        let mut scratch = ();
        let mut partial = None;

        reducer.accumulate_into(&first, &region, &mut scratch, &mut partial);
        let first_ptr = partial.as_ref().unwrap().bins.as_ptr();

        reducer.accumulate_into(&second, &region, &mut scratch, &mut partial);
        let partial = partial.unwrap();

        assert_eq!(first_ptr, partial.bins.as_ptr());
        assert_eq!(partial.bins[0], 1);
        assert_eq!(partial.bins[1], 1);
        assert_eq!(partial.bins.iter().sum::<u64>(), 2);
    }
}

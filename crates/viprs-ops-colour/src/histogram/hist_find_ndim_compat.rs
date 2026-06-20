use viprs_core::{
    format::BandFormat,
    image::{Region, Tile},
    reducer::TileReducer,
    shared_ops::sample_conv::ToF64,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HistFindNDimResult {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) bands: u32,
    pub(crate) bins: Vec<u64>,
}

impl HistFindNDimResult {
    pub(crate) fn total(&self) -> u64 {
        self.bins.iter().sum()
    }
}

pub(crate) struct HistFindNDimOp {
    input_bands: u32,
    bins_per_axis: u32,
    max_sample_value: u32,
}

impl HistFindNDimOp {
    pub(crate) const DEFAULT_BINS_PER_AXIS: u32 = 10;

    pub(crate) fn new(input_bands: u32, bins_per_axis: u32, max_sample_value: u32) -> Self {
        debug_assert!((1..=3).contains(&input_bands));
        debug_assert!(bins_per_axis > 0 && bins_per_axis <= max_sample_value + 1);
        Self {
            input_bands,
            bins_per_axis,
            max_sample_value,
        }
    }

    pub(crate) fn with_default_bins(input_bands: u32, max_sample_value: u32) -> Self {
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

impl<F> TileReducer<F> for HistFindNDimOp
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

use crate::{
    domain::{
        error::{BuildError, ViprsError},
        format::{BandFormat, BandFormatId},
        image::{Region, Tile},
        reducer::{BiSourceReducer, TileReducer},
    },
    ports::source::DynImageSource,
};

use std::sync::Arc;

#[derive(PartialEq, Eq)]
/// Represents a hist find indexed result.
pub struct HistFindIndexedResult {
    /// Width associated with this item.
    pub width: u32,
    /// Number of bands associated with this item.
    pub bands: u32,
    /// Stores the `bins` value for this item.
    pub bins: Vec<u64>,
}

impl HistFindIndexedResult {
    #[must_use]
    /// Returns or performs total.
    pub fn total(&self) -> u64 {
        self.bins.iter().sum()
    }
}

/// Represents a hist find indexed partial.
pub struct HistFindIndexedPartial {
    bins: Vec<u64>,
}

enum IndexBuffer {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

impl IndexBuffer {
    fn load<I>(index: &I, width: u32, height: u32) -> Result<Self, ViprsError>
    where
        I: DynImageSource + ?Sized,
    {
        let full_region = Region::new(0, 0, width, height);
        match index.format() {
            BandFormatId::U8 => {
                let mut data = vec![0u8; width as usize * height as usize];
                index.read_region(full_region, &mut data)?;
                Ok(Self::U8(data))
            }
            BandFormatId::U16 => {
                let mut data = vec![0u16; width as usize * height as usize];
                index.read_region(full_region, bytemuck::cast_slice_mut(&mut data))?;
                Ok(Self::U16(data))
            }
            format => Err(BuildError::UnsupportedFormat {
                op: "hist_find_indexed",
                format,
            }
            .into()),
        }
    }

    const fn type_max(&self) -> u32 {
        match self {
            Self::U8(_) => u8::MAX as u32,
            Self::U16(_) => u16::MAX as u32,
        }
    }

    fn index_at(&self, x: usize, y: usize, width: usize) -> u32 {
        let offset = y * width + x;
        match self {
            Self::U8(data) => u32::from(data[offset]),
            Self::U16(data) => u32::from(data[offset]),
        }
    }
}

/// Applies the `indexed histogram search` histogram operation to the image. It derives
/// histogram-based measurements or adjustments from the input samples.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::histogram::hist_find_indexed::HistFindIndexedOp;
///
/// let op = HistFindIndexedOp { /* operation parameters */ };
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct HistFindIndexedOp {
    input_width: u32,
    input_height: u32,
    input_bands: u32,
    max_index: u32,
    index: IndexBuffer,
}

impl HistFindIndexedOp {
    /// Creates a new `HistFindIndexedOp`.
    #[allow(clippy::needless_pass_by_value)]
    // REASON: public API stability for callers that already own the shared index source.
    pub fn new<I>(
        input_width: u32,
        input_height: u32,
        input_bands: u32,
        max_index: u32,
        index: Arc<I>,
    ) -> Result<Self, ViprsError>
    where
        I: DynImageSource + ?Sized,
    {
        if input_bands == 0 {
            return Err(ViprsError::Scheduler(
                "hist_find_indexed: input band count must be at least 1".into(),
            ));
        }

        if index.width() != input_width || index.height() != input_height {
            return Err(ViprsError::Scheduler(format!(
                "hist_find_indexed: index size {}x{} does not match input {}x{}",
                index.width(),
                index.height(),
                input_width,
                input_height,
            )));
        }

        if index.bands() != 1 {
            return Err(ViprsError::Scheduler(format!(
                "hist_find_indexed: index image must be single-band, got {} bands",
                index.bands(),
            )));
        }

        let index = IndexBuffer::load(index.as_ref(), input_width, input_height)?;
        if max_index > index.type_max() {
            return Err(ViprsError::Scheduler(format!(
                "hist_find_indexed: max_index {max_index} exceeds index format range {}",
                index.type_max(),
            )));
        }

        Ok(Self {
            input_width,
            input_height,
            input_bands,
            max_index,
            index,
        })
    }

    fn empty_partial(&self) -> HistFindIndexedPartial {
        HistFindIndexedPartial {
            bins: vec![0u64; (self.max_index as usize + 1) * self.input_bands as usize],
        }
    }

    fn accumulate_partial<F>(
        &self,
        partial: &mut HistFindIndexedPartial,
        tile: &Tile<F>,
        region: &Region,
    ) where
        F: BandFormat,
    {
        debug_assert_eq!(
            tile.bands, self.input_bands,
            "hist_find_indexed: tile band count must match reducer configuration"
        );

        let bands = self.input_bands as usize;
        let width = self.input_width as usize;
        let height = self.input_height as i32;

        for row in 0..region.height as usize {
            let y = (region.y + row as i32).clamp(0, height - 1) as usize;
            for col in 0..region.width as usize {
                let x = (region.x + col as i32).clamp(0, self.input_width as i32 - 1) as usize;
                let bin = self.index.index_at(x, y, width).min(self.max_index) as usize;
                let base = bin * bands;
                for band in 0..bands {
                    partial.bins[base + band] += 1;
                }
            }
        }
    }
}

impl<F> BiSourceReducer<F> for HistFindIndexedOp
where
    F: BandFormat,
{
    type Partial = HistFindIndexedPartial;
    type Output = HistFindIndexedResult;

    fn reduce_tile_with_side_input(&self, tile: &Tile<F>, region: &Region) -> Self::Partial {
        let mut partial = self.empty_partial();
        self.accumulate_partial(&mut partial, tile, region);
        partial
    }

    fn combine(&self, mut a: Self::Partial, b: Self::Partial) -> Self::Partial {
        for (left, right) in a.bins.iter_mut().zip(b.bins) {
            *left += right;
        }
        a
    }

    fn finalize(&self, combined: Self::Partial) -> Self::Output {
        HistFindIndexedResult {
            width: self.max_index + 1,
            bands: self.input_bands,
            bins: combined.bins,
        }
    }
}

impl<F> TileReducer<F> for HistFindIndexedOp
where
    F: BandFormat,
{
    type Partial = <Self as BiSourceReducer<F>>::Partial;
    type Output = <Self as BiSourceReducer<F>>::Output;
    type Scratch = ();

    fn reduce_tile(&self, tile: &Tile<F>, region: &Region) -> Self::Partial {
        <Self as BiSourceReducer<F>>::reduce_tile_with_side_input(self, tile, region)
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

    fn combine(&self, a: Self::Partial, b: Self::Partial) -> Self::Partial {
        <Self as BiSourceReducer<F>>::combine(self, a, b)
    }

    fn finalize(&self, combined: Self::Partial) -> Self::Output {
        <Self as BiSourceReducer<F>>::finalize(self, combined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        adapters::{
            pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
            sinks::memory::MemorySink, sources::memory::MemorySource,
        },
        domain::{
            format::{U8, U16, U32},
            op::OperationBridge,
            ops::arithmetic::Linear,
            ops::histogram::{HistFindNDimOp, HistFindOp},
            reducer::TileReducer,
        },
        ports::scheduler::ReducingScheduler,
    };

    fn run_indexed_histogram(
        width: u32,
        height: u32,
        bands: u32,
        pixels: Vec<u8>,
        index_pixels: Vec<u8>,
        max_index: u32,
    ) -> HistFindIndexedResult {
        let source = MemorySource::<U8>::new(width, height, bands, pixels).unwrap();
        let pipeline = PipelineBuilder::from_source(source)
            .then(Box::new(OperationBridge::new_pixel_local(
                Linear::<U8>::new(1, 0).unwrap(),
                bands,
            )))
            .unwrap()
            .build()
            .unwrap();
        let sink = MemorySink::for_pipeline(&pipeline).unwrap();
        let scheduler = RayonScheduler::new(2).unwrap();
        let index = Arc::new(MemorySource::<U8>::new(width, height, 1, index_pixels).unwrap());
        let reducer = HistFindIndexedOp::new(width, height, bands, max_index, index).unwrap();

        scheduler
            .run_with_reducer::<U8, HistFindIndexedOp>(&pipeline, &sink, &reducer)
            .unwrap()
    }

    #[test]
    fn u8_values_and_u8_index_count_bins_per_band() {
        let hist = run_indexed_histogram(
            4,
            1,
            2,
            vec![10, 20, 11, 21, 12, 22, 13, 23],
            vec![0, 2, 2, 1],
            3,
        );

        assert_eq!(hist.width, 4);
        assert_eq!(hist.bands, 2);
        assert_eq!(hist.total(), 8);
        assert_eq!(hist.bins[0..8], [1, 1, 1, 1, 2, 2, 0, 0]);
    }

    #[test]
    fn indexed_histogram_matches_hist_find_when_index_equals_value() {
        let region = Region::new(0, 0, 4, 1);
        let values = vec![0u8, 1, 2, 255];
        let tile = Tile::<U8>::new(region, 1, &values);
        let plain_op = HistFindOp::for_format(1, None, u8::MAX as u32);
        let plain = <HistFindOp as TileReducer<U8>>::finalize(
            &plain_op,
            <HistFindOp as TileReducer<U8>>::reduce_tile(&plain_op, &tile, &region),
        );

        let indexed = run_indexed_histogram(4, 1, 1, values.clone(), values, u8::MAX as u32);

        assert_eq!(indexed.width, plain.width);
        assert_eq!(indexed.bands, plain.bands);
        assert_eq!(indexed.bins, plain.bins);
    }

    #[test]
    fn out_of_range_indices_clamp_to_max_bin() {
        let hist = run_indexed_histogram(3, 1, 1, vec![7, 8, 9], vec![0, 1, 7], 3);

        assert_eq!(hist.width, 4);
        assert_eq!(hist.bands, 1);
        assert_eq!(hist.bins, vec![1, 1, 0, 1]);
    }

    #[test]
    fn u16_index_is_supported_and_counts_into_requested_range() {
        let source = MemorySource::<U8>::new(3, 1, 1, vec![1, 2, 3]).unwrap();
        let pipeline = PipelineBuilder::from_source(source)
            .then(Box::new(OperationBridge::new_pixel_local(
                Linear::<U8>::new(1, 0).unwrap(),
                1,
            )))
            .unwrap()
            .build()
            .unwrap();
        let sink = MemorySink::for_pipeline(&pipeline).unwrap();
        let scheduler = RayonScheduler::new(2).unwrap();
        let index = Arc::new(
            MemorySource::<crate::domain::format::U16>::new(3, 1, 1, vec![0, 2, 2]).unwrap(),
        );
        let reducer = HistFindIndexedOp::new(3, 1, 1, 2, index).unwrap();

        let hist = scheduler
            .run_with_reducer::<U8, HistFindIndexedOp>(&pipeline, &sink, &reducer)
            .unwrap();

        assert_eq!(hist.bins, vec![1, 0, 2]);
    }

    #[test]
    fn accumulate_tile_reuses_partial_storage_and_accumulates_expected_bins() {
        let region = Region::new(0, 0, 2, 1);
        let first = Tile::<U8>::new(region, 2, &[10, 20, 11, 21]);
        let second = Tile::<U8>::new(region, 2, &[12, 22, 13, 23]);
        let index = Arc::new(MemorySource::<U8>::new(2, 1, 1, vec![0, 2]).unwrap());
        let reducer = HistFindIndexedOp::new(2, 1, 2, 2, index).unwrap();
        let mut partial = None;

        reducer.accumulate_tile(&mut partial, &first, &region);
        let bins_ptr = partial.as_ref().unwrap().bins.as_ptr();

        reducer.accumulate_tile(&mut partial, &second, &region);
        let hist =
            <HistFindIndexedOp as TileReducer<U8>>::finalize(&reducer, partial.take().unwrap());

        assert_eq!(bins_ptr, hist.bins.as_ptr());
        assert_eq!(hist.width, 3);
        assert_eq!(hist.bands, 2);
        assert_eq!(hist.bins, vec![2, 2, 0, 0, 2, 2]);
    }

    #[test]
    fn histogram_ops_with_reducers_override_accumulate_tile() {
        let region = Region::new(0, 0, 2, 1);
        let first = Tile::<U8>::new(region, 3, &[0, 0, 0, 255, 255, 255]);
        let second = Tile::<U8>::new(region, 3, &[255, 0, 0, 0, 255, 255]);

        let find = HistFindOp::for_format(3, None, u8::MAX as u32);
        let mut find_partial = None;
        find.accumulate_tile(&mut find_partial, &first, &region);
        find.accumulate_tile(&mut find_partial, &second, &region);
        let find_hist =
            <HistFindOp as TileReducer<U8>>::finalize(&find, find_partial.take().unwrap());
        assert_eq!(find_hist.total(), 12);

        let find_ndim = HistFindNDimOp::new(3, 2, u8::MAX as u32);
        let mut ndim_partial = None;
        find_ndim.accumulate_tile(&mut ndim_partial, &first, &region);
        find_ndim.accumulate_tile(&mut ndim_partial, &second, &region);
        let ndim_hist =
            <HistFindNDimOp as TileReducer<U8>>::finalize(&find_ndim, ndim_partial.take().unwrap());
        assert_eq!(ndim_hist.total(), 4);
        assert_eq!(ndim_hist.bins[0], 1);
        assert_eq!(ndim_hist.bins[2], 1);
        assert_eq!(ndim_hist.bins[5], 1);
        assert_eq!(ndim_hist.bins[7], 1);
    }

    #[test]
    fn new_rejects_invalid_configuration_and_unsupported_index_format() {
        let index = Arc::new(MemorySource::<U8>::new(2, 2, 1, vec![0, 1, 2, 3]).unwrap());
        assert!(HistFindIndexedOp::new(2, 2, 0, 3, index.clone()).is_err());

        let wrong_size = Arc::new(MemorySource::<U8>::new(1, 2, 1, vec![0, 1]).unwrap());
        assert!(HistFindIndexedOp::new(2, 2, 1, 3, wrong_size).is_err());

        let multi_band = Arc::new(MemorySource::<U8>::new(2, 2, 2, vec![0; 8]).unwrap());
        assert!(HistFindIndexedOp::new(2, 2, 1, 3, multi_band).is_err());

        let u8_index = Arc::new(MemorySource::<U8>::new(2, 2, 1, vec![0, 1, 2, 3]).unwrap());
        assert!(HistFindIndexedOp::new(2, 2, 1, u16::from(u8::MAX) as u32 + 1, u8_index).is_err());

        let unsupported = Arc::new(MemorySource::<U32>::new(2, 2, 1, vec![0, 1, 2, 3]).unwrap());
        assert!(HistFindIndexedOp::new(2, 2, 1, 3, unsupported).is_err());
    }

    #[test]
    fn reduce_tile_clamps_region_coordinates_to_input_bounds() {
        let region = Region::new(0, 0, 2, 2);
        let tile = Tile::<U8>::new(region, 1, &[0, 1, 2, 3]);
        let index = Arc::new(MemorySource::<U16>::new(2, 2, 1, vec![0, 1, 2, 3]).unwrap());
        let reducer = HistFindIndexedOp::new(2, 2, 1, 2, index).unwrap();
        let wide_region = Region::new(-1, -1, 4, 4);

        let partial =
            <HistFindIndexedOp as TileReducer<U8>>::reduce_tile(&reducer, &tile, &wide_region);
        let combined = <HistFindIndexedOp as TileReducer<U8>>::combine(
            &reducer,
            partial,
            HistFindIndexedPartial {
                bins: vec![0, 0, 0],
            },
        );
        let hist = <HistFindIndexedOp as TileReducer<U8>>::finalize(&reducer, combined);

        assert_eq!(hist.total(), 16);
        assert_eq!(hist.bins, vec![4, 4, 8]);
    }
}

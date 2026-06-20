use super::*;
use crate::{
    format::{BandFormatId, U8},
    image::{Region, Tile, TileMut},
};

/// A minimal no-op operation that copies input to output unchanged.
struct PassThrough;

impl Op for PassThrough {
    type Input = U8;
    type Output = U8;
    type State = ();

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }
    fn start(&self) {}
    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<U8>, output: &mut TileMut<U8>) {
        output.data.copy_from_slice(input.data);
    }
}

#[test]
fn operation_bridge_dyn_operation_format() {
    let bridge = OperationBridge::new(PassThrough, 1u32);
    assert_eq!(bridge.input_format(), BandFormatId::U8);
    assert_eq!(bridge.output_format(), BandFormatId::U8);
    assert_eq!(bridge.bands(), 1);
}

#[test]
fn op_default_preferred_tile_geometry_is_small_tile() {
    let bridge = OperationBridge::new(PassThrough, 1u32);
    assert_eq!(bridge.demand_hint(), DemandHint::SmallTile);
}

#[test]
fn operation_bridge_required_input_region_delegates() {
    let bridge = OperationBridge::new(PassThrough, 1u32);
    let region = Region::new(0, 0, 10, 10);
    assert_eq!(bridge.required_input_region(&region), region);
}

#[test]
fn operation_bridge_dyn_process_region_delegates() {
    let bridge = OperationBridge::new(PassThrough, 1u32);
    let region = Region::new(0, 0, 2, 2);
    let input = vec![1u8, 2u8, 3u8, 4u8];
    let mut output = vec![0u8; 4];
    let mut state = bridge.dyn_start();
    // PassThrough is pixel-local: input_region == output_region.
    bridge.dyn_process_region(state.as_mut(), &input, &mut output, region, region);
    assert_eq!(output, vec![1u8, 2u8, 3u8, 4u8]);
}

#[test]
fn operation_bridge_dyn_start_with_tile_delegates() {
    struct TileAwareStart;

    impl Op for TileAwareStart {
        type Input = U8;
        type Output = U8;
        type State = (u32, u32, u32);

        fn demand_hint(&self) -> DemandHint {
            DemandHint::Any
        }

        fn required_input_region(&self, output: &Region) -> Region {
            *output
        }

        fn start(&self) -> Self::State {
            (0, 0, 0)
        }

        fn start_with_tile(&self, tile_w: u32, tile_h: u32) -> Self::State {
            (tile_w, tile_h, 0)
        }

        fn start_with_tile_and_bands(&self, tile_w: u32, tile_h: u32, bands: u32) -> Self::State {
            (tile_w, tile_h, bands)
        }

        #[inline]
        fn process_region(
            &self,
            _state: &mut Self::State,
            input: &Tile<U8>,
            output: &mut TileMut<U8>,
        ) {
            output.data.copy_from_slice(input.data);
        }
    }

    let bridge = OperationBridge::new(TileAwareStart, 1u32);
    let state = bridge.dyn_start_with_tile_and_bands(7, 9, 3);
    let state = state
        .downcast::<(u32, u32, u32)>()
        .expect("bridge state must preserve tile-aware start");

    assert_eq!(*state, (7, 9, 1));
}

/// `OperationBridge::new` must honour `OUTPUT_BANDS = Some(n)` and override
/// the caller-supplied band count.
#[test]
fn operation_bridge_new_honours_output_bands_const() {
    struct FixedOneBand;
    impl Op for FixedOneBand {
        type Input = U8;
        type Output = U8;
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
        fn process_region(&self, _state: &mut (), input: &Tile<U8>, output: &mut TileMut<U8>) {
            output.data.copy_from_slice(input.data);
        }
    }
    // Caller passes bands=4, but OUTPUT_BANDS=Some(1) must win.
    let bridge = OperationBridge::new(FixedOneBand, 4u32);
    assert_eq!(bridge.bands(), 1);
}

/// `OperationBridge::with_dynamic_bands` must always use the supplied output band count,
/// even when `OUTPUT_BANDS` is `Some(n)`.
#[test]
fn operation_bridge_with_dynamic_bands_ignores_output_bands_const() {
    struct FixedOneBand;
    impl Op for FixedOneBand {
        type Input = U8;
        type Output = U8;
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
        fn process_region(&self, _state: &mut (), input: &Tile<U8>, output: &mut TileMut<U8>) {
            output.data.copy_from_slice(input.data);
        }
    }
    // with_dynamic_bands always trusts the caller.
    let bridge = OperationBridge::with_dynamic_bands(FixedOneBand, 4u32, 3u32);
    assert_eq!(bridge.bands(), 3);
}

#[test]
fn operation_bridge_preserves_distinct_input_and_output_band_counts() {
    struct FirstBand;
    impl Op for FirstBand {
        type Input = U8;
        type Output = U8;
        type State = ();

        fn demand_hint(&self) -> DemandHint {
            DemandHint::Any
        }

        fn required_input_region(&self, output: &Region) -> Region {
            *output
        }

        fn start(&self) {}

        #[inline]
        fn process_region(&self, _state: &mut (), input: &Tile<U8>, output: &mut TileMut<U8>) {
            assert_eq!(input.bands, 3);
            assert_eq!(output.bands, 1);
            for (src, dst) in input
                .data
                .chunks_exact(input.bands as usize)
                .zip(output.data.chunks_exact_mut(output.bands as usize))
            {
                dst[0] = src[0];
            }
        }
    }
    impl PixelLocalOp for FirstBand {}

    let bridge = OperationBridge::with_dynamic_bands_pixel_local(FirstBand, 3u32, 1u32);
    let region = Region::new(0, 0, 2, 1);
    let input = vec![10u8, 20, 30, 40, 50, 60];
    let mut output = vec![0u8; 2];
    let mut state = bridge.dyn_start();

    bridge.dyn_process_region(state.as_mut(), &input, &mut output, region, region);

    assert_eq!(output, vec![10u8, 40]);
}

#[test]
fn demand_hint_geometry_matches_libvips_ordering() {
    assert!(DemandHint::OneLine > DemandHint::FullImage);
    assert!(DemandHint::FullImage > DemandHint::SmallTile);
    assert!(DemandHint::SmallTile > DemandHint::FatStrip);
    assert!(DemandHint::FatStrip > DemandHint::ThinStrip);
    assert!(DemandHint::ThinStrip > DemandHint::Any);
    assert_eq!(DemandHint::OneLine.tile_width(512), 512);
    assert_eq!(DemandHint::OneLine.tile_height(512, 1024), 1);
}

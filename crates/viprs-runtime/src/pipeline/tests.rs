//! Tests support for compiled image pipelines.
//!
//! These helpers participate in turning fluent pipeline descriptions into
//! scheduler-ready execution plans.

use super::*;
use crate::{
    domain::format::{F32, U8, U16},
    domain::image::{DemandHint, Image, ImageMetadata, Interpretation, Tile, TileMut},
    domain::op::{
        CoordinateDrivenSourceSpec, DynOperation, NodeSpec, Op, OperationBridge, SourceReadPlan,
    },
    domain::ops::resample::thumbnail::ThumbnailTarget,
    pipeline::arena::source_region_for_scheduler_tile,
    scheduler::rayon_scheduler::RayonScheduler,
    sources::memory::MemorySource,
    sources::zero::ZeroSource,
};
use proptest::prelude::*;
use std::any::Any;

#[allow(dead_code)]
struct PassThrough {
    bands: u32,
}

impl Op for PassThrough {
    type Input = U8;
    type Output = U8;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }
    fn required_input_region(&self, r: &Region) -> Region {
        *r
    }
    fn start(&self) {}
    fn process_region(&self, (): &mut (), input: &Tile<U8>, output: &mut TileMut<U8>) {
        output.data.copy_from_slice(input.data);
    }
}

struct F32PassThrough;
impl Op for F32PassThrough {
    type Input = F32;
    type Output = F32;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }
    fn required_input_region(&self, r: &Region) -> Region {
        *r
    }
    fn start(&self) {}
    fn process_region(&self, (): &mut (), input: &Tile<F32>, output: &mut TileMut<F32>) {
        output.data.copy_from_slice(input.data);
    }
}

fn pass_op(bands: u32) -> Box<dyn DynOperation> {
    Box::new(OperationBridge::new(PassThrough { bands }, bands))
}

struct NonPixelLocalPass {
    bands: u32,
}

impl DynOperation for NonPixelLocalPass {
    fn input_format(&self) -> BandFormatId {
        BandFormatId::U8
    }

    fn output_format(&self) -> BandFormatId {
        BandFormatId::U8
    }

    fn bands(&self) -> u32 {
        self.bands
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec::identity(tile_w, tile_h)
    }

    fn dyn_start(&self) -> Box<dyn Any + Send> {
        Box::new(())
    }

    fn dyn_process_region(
        &self,
        _state: &mut dyn Any,
        input: &[u8],
        output: &mut [u8],
        _input_region: Region,
        _output_region: Region,
    ) {
        output.copy_from_slice(input);
    }
}

fn non_pixel_local_pass_op(bands: u32) -> Box<dyn DynOperation> {
    Box::new(NonPixelLocalPass { bands })
}

fn zero_band_source() -> MemorySource<U8> {
    MemorySource::<U8>::new(1, 1, 0, vec![]).unwrap()
}

struct CoordinateDrivenSourceStub {
    bands: u32,
    source_halo: u32,
    full_source: Region,
}

impl DynOperation for CoordinateDrivenSourceStub {
    fn input_format(&self) -> BandFormatId {
        BandFormatId::U8
    }

    fn output_format(&self) -> BandFormatId {
        BandFormatId::U8
    }

    fn bands(&self) -> u32 {
        self.bands
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::SmallTile
    }

    fn input_slot_count(&self) -> usize {
        2
    }

    fn input_format_slot(&self, _slot: usize) -> BandFormatId {
        BandFormatId::U8
    }

    fn input_bands_slot(&self, _slot: usize) -> u32 {
        self.bands
    }

    fn required_input_region(&self, _output: &Region) -> Region {
        self.full_source
    }

    fn required_input_region_slot(&self, output: &Region, slot: usize) -> Region {
        if slot == 0 { self.full_source } else { *output }
    }

    fn source_read_plan_slot(&self, output: &Region, slot: usize) -> SourceReadPlan {
        if slot == 0 {
            SourceReadPlan::rect(Region::new(
                output.x,
                output.y,
                output.width.saturating_add(self.source_halo),
                output.height.saturating_add(self.source_halo),
            ))
        } else {
            SourceReadPlan::rect(*output)
        }
    }

    fn coordinate_driven_source_spec(&self) -> Option<CoordinateDrivenSourceSpec> {
        Some(CoordinateDrivenSourceSpec {
            source_slot: 0,
            dependency_slot: 1,
        })
    }

    fn output_width(&self, input_w: u32) -> u32 {
        input_w
    }

    fn output_height(&self, input_h: u32) -> u32 {
        input_h
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec::identity(tile_w, tile_h).with_coordinate_driven_source(0, 1)
    }

    fn dyn_start(&self) -> Box<dyn Any + Send> {
        Box::new(())
    }

    fn dyn_process_region(
        &self,
        _state: &mut dyn Any,
        _input: &[u8],
        _output: &mut [u8],
        _input_region: Region,
        _output_region: Region,
    ) {
    }

    fn dyn_process_region_multi(
        &self,
        _state: &mut dyn Any,
        _inputs: &[&[u8]],
        _output: &mut [u8],
        _input_regions: &[Region],
        _output_region: Region,
    ) {
    }
}

mod apply;
mod builder_core;
mod colourspace;
mod geometry;
mod spatial_integration;
mod thumbnail_execution;

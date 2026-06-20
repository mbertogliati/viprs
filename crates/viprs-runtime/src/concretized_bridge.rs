//! Static fusion bridge: `ConcretizedBridge<C, F>`.
//!
//! Converts a `Concretize` chain into a `DynOperation` that the pipeline can execute.
//! This is the monomorphization point: `C` is the statically-typed chain of ops,
//! `F` is the pixel format resolved at flush time.
#![allow(clippy::items_after_statements, clippy::unnecessary_wraps)]
// REASON: bridge builders mirror the dynamic-operation API shape while keeping monomorphized code local.
//!
//! `ConcretizedBridge` executes the entire chain in a SINGLE loop with zero
//! intermediate allocations. LLVM sees the full chain and auto-vectorizes.

use std::any::Any;
use std::marker::PhantomData;

use crate::domain::{
    concretize::Concretize,
    error::BuildError,
    format::{BandFormat, BandFormatId, PointSample},
    image::{DemandHint, ImageMetadata, Region},
    op::{DynOperation, NodeSpec},
};

/// A `Concretize` chain materialized as a `DynOperation` for pipeline execution.
///
/// # Type parameters
///
/// - `C`: The composed chain (e.g., `(Invert, Linear)` or deeper tuples)
/// - `F`: The pixel format (e.g., `U8`, `F32`), resolved at flush time
///
/// # Performance
///
/// `dyn_process_region` runs a single tight loop over the pixel buffer,
/// calling `chain.apply_sample::<F>()` per sample. LLVM monomorphizes
/// and vectorizes this into optimal SIMD code.
pub struct ConcretizedBridge<C, F> {
    chain: C,
    bands: u32,
    _format: PhantomData<F>,
}

impl<C: Concretize, F: BandFormat> ConcretizedBridge<C, F>
where
    F::Sample: PointSample,
{
    /// `new` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::concretized_bridge::new;
    /// ```
    pub const fn new(chain: C, bands: u32) -> Self {
        Self {
            chain,
            bands,
            _format: PhantomData,
        }
    }
}

impl<C: Concretize, F: BandFormat> DynOperation for ConcretizedBridge<C, F>
where
    F::Sample: PointSample,
{
    fn input_format(&self) -> BandFormatId {
        F::ID
    }

    fn output_format(&self) -> BandFormatId {
        F::ID
    }

    fn bands(&self) -> u32 {
        self.bands
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn is_pixel_local(&self) -> bool {
        true
    }

    fn transform_metadata(&self, source: &ImageMetadata) -> ImageMetadata {
        source.clone()
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec::identity(tile_w, tile_h)
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
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
        // For U8 format, try the SIMD bulk path first.
        if F::ID == BandFormatId::U8 {
            if self.chain.try_apply_bulk_u8(input, output) {
                return;
            }
            use crate::domain::concretize::{Width, apply_chain_wide_u8};
            match self.chain.min_width() {
                Width::Native | Width::I16 => {
                    apply_chain_wide_u8::<i16, C>(&self.chain, input, output);
                }
                Width::F32 => {
                    apply_chain_wide_u8::<f32, C>(&self.chain, input, output);
                }
            }
            return;
        }

        // For other formats (f32, i16, etc.), apply_sample works without
        // overhead since those types don't have the u8↔f32 round-trip problem.
        let src: &[F::Sample] = bytemuck::cast_slice(input);
        let dst: &mut [F::Sample] = bytemuck::cast_slice_mut(output);

        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d = self.chain.apply_sample::<F>(*s);
        }
    }
}

/// Flush a `Concretize` chain into a boxed `DynOperation`, dispatching on runtime format.
///
/// This is the single monomorphization point for the entire pipeline. Each `match` arm
/// instantiates a fully-typed `ConcretizedBridge<C, F>` that LLVM can optimize.
pub fn flush_concretize_chain<C: Concretize + Clone>(
    chain: &C,
    format: BandFormatId,
    bands: u32,
) -> Result<Box<dyn DynOperation>, BuildError> {
    use crate::domain::format::{F32, F64, I16, I32, U8, U16, U32};

    let op: Box<dyn DynOperation> = match format {
        BandFormatId::U8 => Box::new(ConcretizedBridge::<C, U8>::new(chain.clone(), bands)),
        BandFormatId::U16 => Box::new(ConcretizedBridge::<C, U16>::new(chain.clone(), bands)),
        BandFormatId::I16 => Box::new(ConcretizedBridge::<C, I16>::new(chain.clone(), bands)),
        BandFormatId::U32 => Box::new(ConcretizedBridge::<C, U32>::new(chain.clone(), bands)),
        BandFormatId::I32 => Box::new(ConcretizedBridge::<C, I32>::new(chain.clone(), bands)),
        BandFormatId::F32 => Box::new(ConcretizedBridge::<C, F32>::new(chain.clone(), bands)),
        BandFormatId::F64 => Box::new(ConcretizedBridge::<C, F64>::new(chain.clone(), bands)),
    };
    Ok(op)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::format::{F32, U8};
    use crate::domain::image::Region;
    use crate::domain::ops::point::{Invert, Linear};

    #[test]
    fn concretized_bridge_invert_u8() {
        let bridge = ConcretizedBridge::<Invert, U8>::new(Invert, 1);

        assert_eq!(bridge.input_format(), BandFormatId::U8);
        assert_eq!(bridge.output_format(), BandFormatId::U8);
        assert!(bridge.is_pixel_local());

        let input: Vec<u8> = vec![0, 128, 255, 100];
        let mut output: Vec<u8> = vec![0; 4];
        let region = Region {
            x: 0,
            y: 0,
            width: 4,
            height: 1,
        };
        let mut state = bridge.dyn_start();

        bridge.dyn_process_region(state.as_mut(), &input, &mut output, region, region);
        assert_eq!(output, vec![255, 127, 0, 155]);
    }

    #[test]
    fn concretized_bridge_fused_chain() {
        let chain = (Invert, Linear::new(2.0, -0.5));
        let bridge = ConcretizedBridge::<_, U8>::new(chain, 1);

        let input: Vec<u8> = vec![128]; // invert→127, linear→253.5→253
        let mut output: Vec<u8> = vec![0];
        let region = Region {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        };
        let mut state = bridge.dyn_start();

        bridge.dyn_process_region(state.as_mut(), &input, &mut output, region, region);
        assert_eq!(output, vec![253]);
    }

    #[test]
    fn concretized_bridge_f32() {
        let chain = (Invert, Linear::new(2.0, -0.5));
        let bridge = ConcretizedBridge::<_, F32>::new(chain, 1);

        assert_eq!(bridge.input_format(), BandFormatId::F32);

        let input: Vec<u8> = bytemuck::cast_slice(&[0.5f32, 1.0f32]).to_vec();
        let mut output: Vec<u8> = vec![0; 8];
        let region = Region {
            x: 0,
            y: 0,
            width: 2,
            height: 1,
        };
        let mut state = bridge.dyn_start();

        bridge.dyn_process_region(state.as_mut(), &input, &mut output, region, region);
        let result: &[f32] = bytemuck::cast_slice(&output);
        // invert(0.5)=0.5, linear(0.5,2,-0.5)=0.5
        assert!((result[0] - 0.5).abs() < 1e-6);
        // invert(1.0)=0.0, linear(0.0,2,-0.5)=-0.5
        assert!((result[1] + 0.5).abs() < 1e-6);
    }

    #[test]
    fn flush_dispatches_all_formats() {
        let chain = Invert;
        // Just verify it compiles and returns Ok for all formats
        for fmt in [
            BandFormatId::U8,
            BandFormatId::U16,
            BandFormatId::I16,
            BandFormatId::U32,
            BandFormatId::I32,
            BandFormatId::F32,
            BandFormatId::F64,
        ] {
            let op = flush_concretize_chain(&chain, fmt, 3);
            assert!(op.is_ok());
            let op = op.unwrap();
            assert_eq!(op.input_format(), fmt);
            assert_eq!(op.output_format(), fmt);
            assert_eq!(op.bands(), 3);
            assert!(op.is_pixel_local());
        }
    }
}

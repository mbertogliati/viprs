use super::*;

#[test]
fn cache_last_op_skips_linear_output_nodes() {
    let pipeline = PipelineBuilder::new(16, 16)
        .then(non_pixel_local_pass_op(1))
        .unwrap()
        .cache_last_op(NonZeroUsize::new(32).unwrap())
        .unwrap()
        .build()
        .unwrap();

    assert!(pipeline.nodes[0].cache_op_id().is_none());
    assert!(pipeline.tile_cache.is_none());
}

/// Fused pipeline produces correct output: two consecutive invert ops cancel out.
///
/// This exercises builder-level Concretize fusion, which should collapse the
/// two point ops into one `ConcretizedBridge` node while preserving semantics.
#[test]
fn coalescing_invert_invert_is_identity_end_to_end() {
    use crate::adapters::{
        scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
        sources::memory::MemorySource,
    };

    let w = 8u32;
    let h = 4u32;
    let bands = 1u32;
    let input_data: Vec<u8> = (0..w * h).map(|i| i as u8).collect();
    let source = MemorySource::<U8>::new(w, h, bands, input_data.clone()).unwrap();

    let pipeline = PipelineBuilder::from_source(source)
        .invert()
        .unwrap()
        .invert()
        .unwrap()
        .build()
        .unwrap();

    // The Concretize chain should collapse the double invert into one compiled node.
    assert_eq!(
        pipeline.nodes.len(),
        1,
        "two inverts should compile into one concretized node"
    );

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    let scheduler = RayonScheduler::new(RayonScheduler::default_threads()).unwrap();
    scheduler.run(&pipeline, &mut sink).unwrap();
    let result_data = sink.into_buffer();

    assert_eq!(result_data, input_data, "invert ∘ invert must be identity");
}

#[test]
fn fused_resize_upscale_uses_output_height_for_intermediate_buffers() {
    use crate::{
        adapters::{
            scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
            sources::memory::MemorySource,
        },
        domain::kernel::InterpolationKernel,
        domain::ops::resample::resize::Resize,
    };

    let width = 79u32;
    let height = 101u32;
    let bands = 4u32;
    let pixels = (0..width * height * bands)
        .map(|value| value as u8)
        .collect();
    let source = MemorySource::<U8>::new(width, height, bands, pixels).unwrap();

    let pipeline = PipelineBuilder::from_source(source)
        .resize(Resize::new(1.25, 1.25, InterpolationKernel::Lanczos3))
        .unwrap()
        .build()
        .unwrap();

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(2)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();

    let expected_len = pipeline.width as usize * pipeline.height as usize * bands as usize;
    assert_eq!(sink.into_buffer().len(), expected_len);
}

#[test]
fn run_to_image_propagates_source_metadata() {
    use crate::{
        adapters::{scheduler::rayon_scheduler::RayonScheduler, sources::memory::MemorySource},
        domain::{
            format::U8,
            image::{ImageMetadata, Interpretation, Tile, TileMut},
        },
    };

    struct PassThrough;

    impl Op for PassThrough {
        type Input = U8;
        type Output = U8;
        type State = ();

        fn demand_hint(&self) -> DemandHint {
            DemandHint::Any
        }

        fn required_input_region(&self, region: &Region) -> Region {
            *region
        }

        fn start(&self) {}

        #[inline]
        fn process_region(&self, _: &mut (), input: &Tile<U8>, output: &mut TileMut<U8>) {
            output.data.copy_from_slice(input.data);
        }
    }

    let mut metadata = ImageMetadata::default();
    metadata.interpretation = Some(Interpretation::Srgb);
    let source = MemorySource::<U8>::new(4, 4, 3, vec![0u8; 48])
        .unwrap()
        .with_metadata(metadata.clone());

    let pipeline = PipelineBuilder::from_source(source)
        .then(Box::new(OperationBridge::new(PassThrough, 3)))
        .unwrap()
        .build()
        .unwrap();

    let scheduler = RayonScheduler::new(1).unwrap();
    let image = pipeline.run_to_image::<U8, _>(&scheduler).unwrap();

    assert_eq!(image.metadata(), &metadata);
}

#[test]
fn run_to_image_updates_colourspace_metadata_and_invalidates_source_icc() {
    use crate::{
        adapters::{scheduler::rayon_scheduler::RayonScheduler, sources::memory::MemorySource},
        domain::{
            colorspace::{ColorspaceId, Lab},
            format::{F32, U8},
            image::{ImageMetadata, Interpretation},
        },
    };

    let mut metadata = ImageMetadata::default();
    metadata.interpretation = Some(Interpretation::Srgb);
    metadata.icc_profile = Some((0u8..32).collect());

    let source = MemorySource::<U8>::new(
        2,
        2,
        3,
        vec![32, 64, 96, 128, 96, 64, 16, 32, 48, 255, 128, 0],
    )
    .unwrap()
    .with_metadata(metadata);

    let pipeline = PipelineBuilder::from_source(source)
        .with_colorspace(ColorspaceId::SRgb)
        .colourspace::<Lab>()
        .unwrap()
        .build()
        .unwrap();

    let scheduler = RayonScheduler::new(1).unwrap();
    let image = pipeline.run_to_image::<F32, _>(&scheduler).unwrap();

    assert_eq!(image.metadata().interpretation, Some(Interpretation::Lab));
    assert!(image.metadata().icc_profile.is_none());
}

#[cfg(feature = "icc")]
#[test]
fn normalize_to_srgb_matches_web_encode_normalization_for_gray_alpha_sources() {
    use crate::{
        adapters::{scheduler::rayon_scheduler::RayonScheduler, sources::memory::MemorySource},
        domain::{
            format::U8,
            image::{Image, ImageMetadata, Interpretation},
            ops::colour::{icc::srgb_profile_bytes, profile_load},
        },
    };

    let mut metadata = ImageMetadata::default();
    metadata.interpretation = Some(Interpretation::BW);
    metadata.icc_profile = Some(profile_load("gray").expect("load gray ICC profile"));
    let input = Image::<U8>::from_buffer(2, 1, 2, vec![32, 7, 160, 9])
        .unwrap()
        .with_metadata(metadata);
    let pipeline = PipelineBuilder::from_source(
        MemorySource::<U8>::new(
            input.width(),
            input.height(),
            input.bands(),
            input.pixels().to_vec(),
        )
        .unwrap()
        .with_metadata(input.metadata().clone()),
    )
    .normalize_to_srgb()
    .unwrap()
    .build()
    .unwrap();

    let actual = pipeline
        .run_to_image::<U8, _>(&RayonScheduler::new(1).unwrap())
        .unwrap();
    let srgb = srgb_profile_bytes().expect("load srgb ICC profile");

    assert_eq!(actual.bands(), 4);
    assert_eq!(actual.metadata().interpretation, Some(Interpretation::Srgb));
    assert_eq!(
        actual.metadata().icc_profile.as_deref(),
        Some(srgb.as_slice())
    );
    assert_eq!(actual.pixels()[3], 7);
    assert_eq!(actual.pixels()[7], 9);
}

#[cfg(feature = "icc")]
#[test]
fn normalize_to_srgb_is_noop_for_existing_srgb_profile() {
    use crate::{
        adapters::sources::memory::MemorySource,
        domain::{
            format::U8,
            image::{ImageMetadata, Interpretation},
            ops::colour::profile_load,
        },
    };

    let mut metadata = ImageMetadata::default();
    metadata.interpretation = Some(Interpretation::Srgb);
    metadata.icc_profile = Some(profile_load("srgb").expect("load srgb ICC profile"));

    let builder = PipelineBuilder::from_source(
        MemorySource::<U8>::new(2, 1, 3, vec![10, 20, 30, 40, 50, 60])
            .unwrap()
            .with_metadata(metadata),
    );
    let baseline_nodes = builder.node_count();

    let normalized = builder.normalize_to_srgb().unwrap();

    assert_eq!(normalized.node_count(), baseline_nodes);
}

#[test]
fn branch_point_cache_reuses_upstream_tile_within_and_across_runs() {
    use crate::{
        adapters::{scheduler::rayon_scheduler::RayonScheduler, sources::memory::MemorySource},
        domain::{
            format::BandFormatId,
            format::U8,
            image::{Tile, TileMut},
            op::{Op, OperationBridge},
            ops::conversion::BandJoin,
        },
    };
    use std::{
        num::NonZeroUsize,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };

    struct CountingPass {
        calls: Arc<AtomicUsize>,
    }

    impl Op for CountingPass {
        type Input = U8;
        type Output = U8;
        type State = ();

        fn demand_hint(&self) -> DemandHint {
            DemandHint::ThinStrip
        }

        fn required_input_region(&self, output: &Region) -> Region {
            *output
        }

        fn start(&self) -> Self::State {}

        #[inline]
        fn process_region(&self, _: &mut Self::State, input: &Tile<U8>, output: &mut TileMut<U8>) {
            self.calls.fetch_add(1, Ordering::SeqCst);
            output.data.copy_from_slice(input.data);
        }
    }

    let source = MemorySource::<U8>::new(4, 4, 1, (0..16u8).collect()).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let mut arena = PipelineArena::with_source(Box::new(source));
    let root = arena.add_node(Box::new(OperationBridge::new(
        CountingPass {
            calls: Arc::clone(&calls),
        },
        1,
    )));
    let branch = arena.add_node(pass_op(1));
    let merge = arena.add_node(Box::new(BandJoin::new(1, 1, BandFormatId::U8)));
    arena.connect(root, branch).unwrap();
    arena.connect(root, merge).unwrap();
    arena.connect_to_slot(branch, merge, 1).unwrap();
    arena
        .enable_cache(root, NonZeroUsize::new(64).unwrap())
        .unwrap();
    let pipeline = arena.compile().unwrap();

    let scheduler = RayonScheduler::new(1).unwrap();
    let first = pipeline.run_to_image::<U8, _>(&scheduler).unwrap();
    let second = pipeline.run_to_image::<U8, _>(&scheduler).unwrap();

    assert_eq!(first.pixels(), second.pixels());
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "branch-point cache should reuse the upstream tile within a run and on the second run"
    );
}

#[test]
fn clearing_branch_point_cache_forces_recompute_on_next_run() {
    use crate::{
        adapters::{scheduler::rayon_scheduler::RayonScheduler, sources::memory::MemorySource},
        domain::{
            format::BandFormatId,
            format::U8,
            image::{Tile, TileMut},
            op::{Op, OperationBridge},
            ops::conversion::BandJoin,
        },
    };
    use std::{
        num::NonZeroUsize,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };

    struct CountingPass {
        calls: Arc<AtomicUsize>,
    }

    impl Op for CountingPass {
        type Input = U8;
        type Output = U8;
        type State = ();

        fn demand_hint(&self) -> DemandHint {
            DemandHint::ThinStrip
        }

        fn required_input_region(&self, output: &Region) -> Region {
            *output
        }

        fn start(&self) -> Self::State {}

        #[inline]
        fn process_region(&self, _: &mut Self::State, input: &Tile<U8>, output: &mut TileMut<U8>) {
            self.calls.fetch_add(1, Ordering::SeqCst);
            output.data.copy_from_slice(input.data);
        }
    }

    let source = MemorySource::<U8>::new(4, 4, 1, (0..16u8).collect()).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let mut arena = PipelineArena::with_source(Box::new(source));
    let root = arena.add_node(Box::new(OperationBridge::new(
        CountingPass {
            calls: Arc::clone(&calls),
        },
        1,
    )));
    let branch = arena.add_node(pass_op(1));
    let merge = arena.add_node(Box::new(BandJoin::new(1, 1, BandFormatId::U8)));
    arena.connect(root, branch).unwrap();
    arena.connect(root, merge).unwrap();
    arena.connect_to_slot(branch, merge, 1).unwrap();
    arena
        .enable_cache(root, NonZeroUsize::new(64).unwrap())
        .unwrap();
    let pipeline = arena.compile().unwrap();

    let scheduler = RayonScheduler::new(1).unwrap();
    let first = pipeline.run_to_image::<U8, _>(&scheduler).unwrap();
    pipeline.clear_tile_cache().unwrap();
    let second = pipeline.run_to_image::<U8, _>(&scheduler).unwrap();

    assert_eq!(first.pixels(), second.pixels());
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "clearing the branch-point cache should force the upstream operation to run again"
    );
}

#[test]
fn from_source_maps_interpretation_for_colourspace_builder() {
    use crate::adapters::sources::memory::MemorySource;
    use crate::domain::{colorspace::Lab, format::U8, image::ImageMetadata};

    let mut metadata = ImageMetadata::default();
    metadata.interpretation = Some(Interpretation::Srgb);
    let source = MemorySource::<U8>::new(1, 1, 3, vec![128, 64, 32])
        .unwrap()
        .with_metadata(metadata);

    let pipeline = PipelineBuilder::from_source(source)
        .colourspace::<Lab>()
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(pipeline.output_bands, 3);
}

#[cfg(feature = "jpeg")]
#[test]
fn jpeg_decoder_source_preserves_interpretation_through_pipeline() {
    use crate::adapters::{
        codecs::JpegCodec, scheduler::rayon_scheduler::RayonScheduler,
        sources::decoder_source::DecoderSource,
    };
    use crate::domain::{codec_options::SaveOptions, format::U8, image::Image};
    use crate::ports::codec::ImageEncoder;

    let codec = JpegCodec;
    let original = Image::<U8>::from_buffer(2, 2, 3, vec![64u8; 12]).unwrap();
    let encoded = codec
        .encode_with_options(&original, &SaveOptions::default().with_quality(100))
        .unwrap();
    let source = DecoderSource::<_, U8>::new(codec, &encoded).unwrap();

    let pipeline = PipelineBuilder::from_source(source)
        .invert()
        .unwrap()
        .invert()
        .unwrap()
        .build()
        .unwrap();
    let image = pipeline
        .run_to_image::<U8, _>(&RayonScheduler::new(1).unwrap())
        .unwrap();

    assert_eq!(image.metadata().interpretation, Some(Interpretation::Srgb));
}

#[test]
fn thumbnail_passes_loader_specific_hint_to_jpeg_decoder_source() {
    use crate::adapters::sources::decoder_source::DecoderSource;
    use crate::domain::ops::resample::thumbnail::ThumbnailTarget;
    use crate::domain::{codec_options::LoadOptions, format::U8, image::Image};
    use crate::ports::codec::ImageDecoder;
    use std::num::NonZeroU8;
    use std::sync::{Arc, Mutex};

    struct TrackingDecoder {
        seen_factors: Arc<Mutex<Vec<u8>>>,
    }

    impl ImageDecoder for TrackingDecoder {
        fn format_name(&self) -> &'static str {
            "jpeg"
        }

        fn sniff(&self, _: &[u8]) -> bool {
            true
        }

        fn decode<F: BandFormat>(&self, src: &[u8]) -> Result<Image<F>, ViprsError> {
            self.decode_with_options(src, &LoadOptions::default())
        }

        fn decode_with_options<F: BandFormat>(
            &self,
            _: &[u8],
            opts: &LoadOptions,
        ) -> Result<Image<F>, ViprsError> {
            if F::ID != U8::ID {
                return Err(ViprsError::Codec(
                    "tracking decoder only supports U8".into(),
                ));
            }

            let factor = opts.shrink_factor.map_or(1, NonZeroU8::get);
            self.seen_factors.lock().unwrap().push(factor);
            let width = (800 / u32::from(factor)).max(1);
            let height = (600 / u32::from(factor)).max(1);
            let image =
                Image::from_buffer(width, height, 3, vec![0u8; (width * height * 3) as usize])
                    .map_err(|e| ViprsError::Codec(e.to_string()))?;

            // SAFETY: F::ID == U8 implies F::Sample == u8 because BandFormat is sealed.
            let cast = unsafe { std::mem::transmute::<Image<U8>, Image<F>>(image) };
            Ok(cast)
        }

        fn probe(&self, _: &[u8]) -> Result<(u32, u32, u32), ViprsError> {
            Ok((800, 600, 3))
        }
    }

    let seen_factors = Arc::new(Mutex::new(Vec::new()));
    let source = DecoderSource::<_, U8>::new(
        TrackingDecoder {
            seen_factors: Arc::clone(&seen_factors),
        },
        b"encoded",
    )
    .unwrap();

    let pipeline = PipelineBuilder::from_source(source)
        .thumbnail(Thumbnail::new(
            ThumbnailTarget::Width(19),
            InterpolationKernel::Lanczos3,
        ))
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(pipeline.width, 19);
    assert_eq!(pipeline.height, 14);
    assert_eq!(&*seen_factors.lock().unwrap(), &[1, 16]);
}

#[test]
fn thumbnail_passes_loader_hint_before_first_path_decode() {
    use crate::adapters::sources::decoder_source::DecoderSource;
    use crate::domain::ops::resample::thumbnail::ThumbnailTarget;
    use crate::domain::{codec_options::LoadOptions, format::U8, image::Image};
    use crate::ports::codec::ImageDecoder;
    use std::fs;
    use std::num::NonZeroU8;
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    struct PathTrackingDecoder {
        seen_factors: Arc<Mutex<Vec<u8>>>,
    }

    impl ImageDecoder for PathTrackingDecoder {
        fn format_name(&self) -> &'static str {
            "jpeg"
        }

        fn sniff(&self, _: &[u8]) -> bool {
            true
        }

        fn decode<F: BandFormat>(&self, src: &[u8]) -> Result<Image<F>, ViprsError> {
            self.decode_with_options(src, &LoadOptions::default())
        }

        fn decode_with_options<F: BandFormat>(
            &self,
            _: &[u8],
            opts: &LoadOptions,
        ) -> Result<Image<F>, ViprsError> {
            if F::ID != U8::ID {
                return Err(ViprsError::Codec(
                    "path tracking decoder only supports U8".into(),
                ));
            }

            let factor = opts.shrink_factor.map_or(1, NonZeroU8::get);
            self.seen_factors.lock().unwrap().push(factor);
            let width = (800 / u32::from(factor)).max(1);
            let height = (600 / u32::from(factor)).max(1);
            let image =
                Image::from_buffer(width, height, 3, vec![0u8; (width * height * 3) as usize])
                    .map_err(|e| ViprsError::Codec(e.to_string()))?;

            // SAFETY: F::ID == U8 implies F::Sample == u8 because BandFormat is sealed.
            let cast = unsafe { std::mem::transmute::<Image<U8>, Image<F>>(image) };
            Ok(cast)
        }

        fn decode_path_with_options<F: BandFormat>(
            &self,
            _: &Path,
            opts: &LoadOptions,
        ) -> Result<Image<F>, ViprsError>
        where
            Self: Sized,
        {
            self.decode_with_options(&[], opts)
        }

        fn probe(&self, _: &[u8]) -> Result<(u32, u32, u32), ViprsError> {
            Ok((800, 600, 3))
        }

        fn probe_path(&self, _: &Path) -> Result<(u32, u32, u32), ViprsError>
        where
            Self: Sized,
        {
            Ok((800, 600, 3))
        }
    }

    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("pipeline-thumbnail-probed.jpg");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(&path, b"jpeg").unwrap();

    let seen_factors = Arc::new(Mutex::new(Vec::new()));
    let source = DecoderSource::<_, U8>::probed_path(
        PathTrackingDecoder {
            seen_factors: Arc::clone(&seen_factors),
        },
        &path,
    )
    .unwrap();

    let pipeline = PipelineBuilder::from_source(source)
        .thumbnail(Thumbnail::new(
            ThumbnailTarget::Width(19),
            InterpolationKernel::Lanczos3,
        ))
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(pipeline.width, 19);
    assert_eq!(pipeline.height, 14);
    assert_eq!(&*seen_factors.lock().unwrap(), &[16]);
}

#[test]
fn large_thumbnail_with_native_source_hint_keeps_thin_strip_demand() {
    use crate::adapters::sources::decoder_source::DecoderSource;
    use crate::domain::ops::resample::thumbnail::ThumbnailTarget;
    use crate::domain::{codec_options::LoadOptions, format::U8, image::Image};
    use crate::ports::codec::ImageDecoder;
    use std::num::NonZeroU8;

    struct NativeHintDecoder;

    impl ImageDecoder for NativeHintDecoder {
        fn format_name(&self) -> &'static str {
            "jpeg"
        }

        fn sniff(&self, _: &[u8]) -> bool {
            true
        }

        fn decode<F: BandFormat>(&self, src: &[u8]) -> Result<Image<F>, ViprsError> {
            self.decode_with_options(src, &LoadOptions::default())
        }

        fn decode_with_options<F: BandFormat>(
            &self,
            _: &[u8],
            opts: &LoadOptions,
        ) -> Result<Image<F>, ViprsError> {
            if F::ID != U8::ID {
                return Err(ViprsError::Codec(
                    "native hint decoder only supports U8".into(),
                ));
            }

            let factor = opts.shrink_factor.map_or(1, NonZeroU8::get);
            let width = (2048 / u32::from(factor)).max(1);
            let height = (1536 / u32::from(factor)).max(1);
            let image =
                Image::from_buffer(width, height, 3, vec![0u8; (width * height * 3) as usize])
                    .map_err(|err| ViprsError::Codec(err.to_string()))?;

            // SAFETY: F::ID == U8 implies F::Sample == u8 because BandFormat is sealed.
            let cast = unsafe { std::mem::transmute::<Image<U8>, Image<F>>(image) };
            Ok(cast)
        }

        fn probe(&self, _: &[u8]) -> Result<(u32, u32, u32), ViprsError> {
            Ok((2048, 1536, 3))
        }
    }

    let source = DecoderSource::<_, U8>::new(NativeHintDecoder, b"jpeg").unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .with_colorspace(crate::domain::colorspace::ColorspaceId::SRgb)
        .thumbnail(Thumbnail::new(
            ThumbnailTarget::Width(400),
            InterpolationKernel::Lanczos3,
        ))
        .unwrap()
        .sharpen(0.5, 2.0, 10.0, 20.0, 0.0, 3.0)
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(pipeline.demand_hint, DemandHint::ThinStrip);
}

#[test]
fn thumbnail_replans_after_native_shrink_changes_actual_dimensions() {
    use crate::adapters::{
        scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
        sources::decoder_source::DecoderSource,
    };
    use crate::domain::ops::resample::thumbnail::ThumbnailTarget;
    use crate::domain::{
        codec_options::LoadOptions, colorspace::ColorspaceId, format::U8, image::Image,
    };
    use crate::ports::{codec::ImageDecoder, scheduler::TileScheduler};
    use std::num::NonZeroU8;

    struct NativeShrinkRoundingDecoder;

    impl ImageDecoder for NativeShrinkRoundingDecoder {
        fn format_name(&self) -> &'static str {
            "jpeg"
        }

        fn sniff(&self, _: &[u8]) -> bool {
            true
        }

        fn decode<F: BandFormat>(&self, src: &[u8]) -> Result<Image<F>, ViprsError> {
            self.decode_with_options(src, &LoadOptions::default())
        }

        fn decode_with_options<F: BandFormat>(
            &self,
            _: &[u8],
            opts: &LoadOptions,
        ) -> Result<Image<F>, ViprsError> {
            if F::ID != U8::ID {
                return Err(ViprsError::Codec(
                    "native shrink rounding decoder only supports U8".into(),
                ));
            }

            let factor = opts.shrink_factor.map_or(1, NonZeroU8::get);
            let (width, height) = if factor == 2 {
                (1023, 1024)
            } else {
                (2048, 2048)
            };
            let image =
                Image::from_buffer(width, height, 3, vec![64u8; (width * height * 3) as usize])
                    .map_err(|err| ViprsError::Codec(err.to_string()))?;

            // SAFETY: F::ID == U8 implies F::Sample == u8 because BandFormat is sealed.
            let cast = unsafe { std::mem::transmute::<Image<U8>, Image<F>>(image) };
            Ok(cast)
        }

        fn probe(&self, _: &[u8]) -> Result<(u32, u32, u32), ViprsError> {
            Ok((2048, 2048, 3))
        }
    }

    let source = DecoderSource::<_, U8>::new(NativeShrinkRoundingDecoder, b"jpeg").unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .with_colorspace(ColorspaceId::SRgb)
        .thumbnail(Thumbnail::new(
            ThumbnailTarget::Width(400),
            InterpolationKernel::Lanczos3,
        ))
        .unwrap()
        .sharpen(0.5, 2.0, 10.0, 20.0, 0.0, 3.0)
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(pipeline.width, 400);
    assert_eq!(pipeline.height, 400);

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();

    assert_eq!(sink.into_buffer().len(), 400 * 400 * 3);
}

#[test]
fn large_thumbnail_avoids_full_image_hint() {
    use crate::{
        adapters::sources::memory::MemorySource,
        domain::{kernel::InterpolationKernel, ops::resample::thumbnail::ThumbnailTarget},
    };

    let source = MemorySource::<U8>::new(2048, 2048, 3, vec![0u8; 2048 * 2048 * 3]).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .thumbnail(Thumbnail::new(
            ThumbnailTarget::Width(400),
            InterpolationKernel::Lanczos3,
        ))
        .unwrap()
        .build()
        .unwrap();

    assert_ne!(pipeline.demand_hint, DemandHint::FullImage);
}

#[test]
fn chained_thumbnail_uses_intermediate_dimensions() {
    use crate::{
        adapters::sources::memory::MemorySource,
        domain::{
            format::U8,
            image::{Image, ImageMetadata},
            kernel::InterpolationKernel,
            ops::resample::thumbnail::{Thumbnail, ThumbnailTarget},
        },
    };

    fn patterned_rgb_u8(width: u32, height: u32) -> Image<U8> {
        let pixels = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    [
                        ((x * 17 + y * 13) % 256) as u8,
                        ((x * 11 + y * 19 + 37) % 256) as u8,
                        ((x * 5 + y * 23 + 91) % 256) as u8,
                    ]
                })
            })
            .collect();
        Image::from_buffer(width, height, 3, pixels).unwrap()
    }

    fn run_thumbnail(image: &Image<U8>, width: u32) -> (CompiledPipeline, Image<U8>) {
        let source = MemorySource::<U8>::new(
            image.width(),
            image.height(),
            image.bands(),
            image.pixels().to_vec(),
        )
        .unwrap()
        .with_metadata(ImageMetadata::default());
        let pipeline = PipelineBuilder::from_source(source)
            .thumbnail(Thumbnail::new(
                ThumbnailTarget::Width(width),
                InterpolationKernel::Lanczos3,
            ))
            .unwrap()
            .build()
            .unwrap();
        let image = pipeline
            .run_to_image::<U8, _>(&RayonScheduler::new(1).unwrap())
            .unwrap();
        (pipeline, image)
    }

    let image = patterned_rgb_u8(777, 333);
    let (_, first) = run_thumbnail(&image, 100);
    let (_, sequential) = run_thumbnail(&first, 50);

    let source = MemorySource::<U8>::new(
        image.width(),
        image.height(),
        image.bands(),
        image.pixels().to_vec(),
    )
    .unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .thumbnail(Thumbnail::new(
            ThumbnailTarget::Width(100),
            InterpolationKernel::Lanczos3,
        ))
        .unwrap()
        .thumbnail(Thumbnail::new(
            ThumbnailTarget::Width(50),
            InterpolationKernel::Lanczos3,
        ))
        .unwrap()
        .build()
        .unwrap();
    let chained = pipeline
        .run_to_image::<U8, _>(&RayonScheduler::new(1).unwrap())
        .unwrap();

    assert_eq!((pipeline.width, pipeline.height), (50, 22));
    assert_eq!(
        (pipeline.width, pipeline.height),
        (sequential.width(), sequential.height())
    );
    assert_eq!((chained.width(), chained.height()), (50, 22));
}

#[test]
fn chained_thumbnail_single_row_matches_sequential_execution() {
    use crate::{
        adapters::sources::memory::MemorySource,
        domain::{
            format::U8,
            image::{Image, ImageMetadata},
            kernel::InterpolationKernel,
            ops::resample::thumbnail::{Thumbnail, ThumbnailTarget},
        },
    };

    fn patterned_rgb_u8(width: u32, height: u32) -> Image<U8> {
        let pixels = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    [
                        ((x * 17 + y * 13) % 256) as u8,
                        ((x * 11 + y * 19 + 37) % 256) as u8,
                        ((x * 5 + y * 23 + 91) % 256) as u8,
                    ]
                })
            })
            .collect();
        Image::from_buffer(width, height, 3, pixels).unwrap()
    }

    fn run_thumbnail(image: &Image<U8>, width: u32) -> (CompiledPipeline, Image<U8>) {
        let source = MemorySource::<U8>::new(
            image.width(),
            image.height(),
            image.bands(),
            image.pixels().to_vec(),
        )
        .unwrap()
        .with_metadata(ImageMetadata::default());
        let pipeline = PipelineBuilder::from_source(source)
            .thumbnail(Thumbnail::new(
                ThumbnailTarget::Width(width),
                InterpolationKernel::Lanczos3,
            ))
            .unwrap()
            .build()
            .unwrap();
        let image = pipeline
            .run_to_image::<U8, _>(&RayonScheduler::new(1).unwrap())
            .unwrap();
        (pipeline, image)
    }

    let image = patterned_rgb_u8(8192, 1);
    let (_, first) = run_thumbnail(&image, 400);
    let (_, second) = run_thumbnail(&first, 64);
    let (_, sequential) = run_thumbnail(&second, 7);

    let source = MemorySource::<U8>::new(
        image.width(),
        image.height(),
        image.bands(),
        image.pixels().to_vec(),
    )
    .unwrap()
    .with_metadata(ImageMetadata::default());
    let pipeline = PipelineBuilder::from_source(source)
        .thumbnail(Thumbnail::new(
            ThumbnailTarget::Width(400),
            InterpolationKernel::Lanczos3,
        ))
        .unwrap()
        .thumbnail(Thumbnail::new(
            ThumbnailTarget::Width(64),
            InterpolationKernel::Lanczos3,
        ))
        .unwrap()
        .thumbnail(Thumbnail::new(
            ThumbnailTarget::Width(7),
            InterpolationKernel::Lanczos3,
        ))
        .unwrap()
        .build()
        .unwrap();
    let chained = pipeline
        .run_to_image::<U8, _>(&RayonScheduler::new(2).unwrap())
        .unwrap();

    assert_eq!(
        (pipeline.width, pipeline.height),
        (sequential.width(), sequential.height())
    );
    assert_eq!((chained.width(), chained.height()), (7, 1));
    assert_eq!(chained.pixels(), sequential.pixels());
}

#[test]
fn thumbnail_after_colourspace_uses_intermediate_dimensions() {
    use crate::{
        adapters::sources::memory::MemorySource,
        domain::{
            colorspace::{ColorspaceId, Lab, SRgb},
            format::U8,
            image::Image,
            kernel::InterpolationKernel,
            ops::resample::thumbnail::{Thumbnail, ThumbnailTarget},
        },
    };

    fn patterned_rgb_u8(width: u32, height: u32) -> Image<U8> {
        let pixels = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    [
                        ((x * 17 + y * 13) % 256) as u8,
                        ((x * 11 + y * 19 + 37) % 256) as u8,
                        ((x * 5 + y * 23 + 91) % 256) as u8,
                    ]
                })
            })
            .collect();
        Image::from_buffer(width, height, 3, pixels).unwrap()
    }

    let image = patterned_rgb_u8(777, 333);
    let source = MemorySource::<U8>::new(
        image.width(),
        image.height(),
        image.bands(),
        image.pixels().to_vec(),
    )
    .unwrap();
    let first_pipeline = PipelineBuilder::from_source(source)
        .with_colorspace(ColorspaceId::SRgb)
        .thumbnail(Thumbnail::new(
            ThumbnailTarget::Width(400),
            InterpolationKernel::Lanczos3,
        ))
        .unwrap()
        .colourspace::<Lab>()
        .unwrap()
        .colourspace::<SRgb>()
        .unwrap()
        .build()
        .unwrap();
    let first = first_pipeline
        .run_to_image::<U8, _>(&RayonScheduler::new(1).unwrap())
        .unwrap();

    let sequential_source = MemorySource::<U8>::new(
        first.width(),
        first.height(),
        first.bands(),
        first.pixels().to_vec(),
    )
    .unwrap()
    .with_metadata(first.metadata().clone());
    let sequential_pipeline = PipelineBuilder::from_source(sequential_source)
        .thumbnail(Thumbnail::new(
            ThumbnailTarget::Width(200),
            InterpolationKernel::Lanczos3,
        ))
        .unwrap()
        .build()
        .unwrap();
    let sequential = sequential_pipeline
        .run_to_image::<U8, _>(&RayonScheduler::new(1).unwrap())
        .unwrap();

    let chained_source = MemorySource::<U8>::new(
        image.width(),
        image.height(),
        image.bands(),
        image.pixels().to_vec(),
    )
    .unwrap();
    let chained_pipeline = PipelineBuilder::from_source(chained_source)
        .with_colorspace(ColorspaceId::SRgb)
        .thumbnail(Thumbnail::new(
            ThumbnailTarget::Width(400),
            InterpolationKernel::Lanczos3,
        ))
        .unwrap()
        .colourspace::<Lab>()
        .unwrap()
        .colourspace::<SRgb>()
        .unwrap()
        .thumbnail(Thumbnail::new(
            ThumbnailTarget::Width(200),
            InterpolationKernel::Lanczos3,
        ))
        .unwrap()
        .build()
        .unwrap();
    let chained = chained_pipeline
        .run_to_image::<U8, _>(&RayonScheduler::new(1).unwrap())
        .unwrap();

    assert_eq!((chained_pipeline.width, chained_pipeline.height), (200, 86));
    assert_eq!(
        (chained_pipeline.width, chained_pipeline.height),
        (sequential.width(), sequential.height())
    );
    assert_eq!((chained.width(), chained.height()), (200, 86));
}

#[test]
fn replicate_runs_end_to_end_without_tile_propagation_panic() {
    use crate::adapters::{
        scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
        sources::memory::MemorySource,
    };

    let width = 8u32;
    let height = 5u32;
    let input_data: Vec<u8> = (0..width * height).map(|value| value as u8).collect();
    let source = MemorySource::<U8>::new(width, height, 1, input_data.clone()).unwrap();

    let pipeline = PipelineBuilder::from_source(source)
        .replicate(2, 3)
        .unwrap()
        .build()
        .unwrap();

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();
    let output = sink.into_buffer();

    assert_eq!(pipeline.width, width * 2);
    assert_eq!(pipeline.height, height * 3);

    for y in 0..pipeline.height as usize {
        for x in 0..pipeline.width as usize {
            let expected =
                input_data[(y % height as usize) * width as usize + (x % width as usize)];
            assert_eq!(output[y * pipeline.width as usize + x], expected);
        }
    }
}

#[test]
fn thread_buffer_pool_allocates_per_slot_scratch_storage_for_multi_input_nodes() {
    use crate::ports::source::DynImageSource;
    use std::any::Any;

    struct MergeTwoSlots;

    impl DynOperation for MergeTwoSlots {
        fn input_format(&self) -> BandFormatId {
            BandFormatId::U8
        }

        fn output_format(&self) -> BandFormatId {
            BandFormatId::U8
        }

        fn bands(&self) -> u32 {
            1
        }

        fn demand_hint(&self) -> DemandHint {
            DemandHint::ThinStrip
        }

        fn input_slot_count(&self) -> usize {
            2
        }

        fn required_input_region_slot(&self, output: &Region, slot: usize) -> Region {
            match slot {
                0 => Region::new(output.x, output.y, 4, 4),
                1 => Region::new(output.x + 1, output.y + 1, 2, 2),
                _ => Region::new(0, 0, 0, 0),
            }
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
            _input: &[u8],
            _output: &mut [u8],
            _input_region: Region,
            _output_region: Region,
        ) {
        }
    }

    let source = ZeroSource::<U8>::new(8, 8, 1);
    let mut arena = PipelineArena::with_source(Box::new(source) as Box<dyn DynImageSource>);
    let upstream = arena.add_node(pass_op(1));
    let branch = arena.add_node(pass_op(1));
    let merge = arena.add_node(Box::new(MergeTwoSlots));

    arena.connect(upstream, branch).unwrap();
    arena.connect(upstream, merge).unwrap();
    arena.connect_to_slot(branch, merge, 1).unwrap();

    let pipeline = arena.compile().unwrap();
    let pool = ThreadBufferPool::new(&pipeline);
    let merge_idx = pipeline.nodes.len() - 1;

    assert_eq!(pool.scratch_regions[merge_idx].len(), 2);
    assert_eq!(pool.input_scratch_buffers[merge_idx].len(), 2);
}

#[test]
fn linear_pipeline_keeps_monotone_buffer_count() {
    let source = ZeroSource::<U8>::new(8, 8, 1);
    let mut arena = PipelineArena::with_source(Box::new(source));
    let first = arena.add_node(non_pixel_local_pass_op(1));
    let second = arena.add_node(non_pixel_local_pass_op(1));
    let third = arena.add_node(non_pixel_local_pass_op(1));

    arena.connect(first, second).unwrap();
    arena.connect(second, third).unwrap();

    let pipeline = arena.compile().unwrap();
    assert_eq!(
        pipeline.buffer_count, 4,
        "linear pipelines should keep the same source+N buffer layout"
    );
}

#[test]
fn compile_skips_cache_on_single_consumer_linear_nodes() {
    let source = ZeroSource::<U8>::new(8, 8, 1);
    let mut arena = PipelineArena::with_source(Box::new(source));
    let first = arena.add_node(non_pixel_local_pass_op(1));
    let second = arena.add_node(non_pixel_local_pass_op(1));
    let third = arena.add_node(non_pixel_local_pass_op(1));

    arena.connect(first, second).unwrap();
    arena.connect(second, third).unwrap();
    arena
        .enable_cache(second, NonZeroUsize::new(64).unwrap())
        .unwrap();

    let pipeline = arena.compile().unwrap();

    assert!(
        pipeline
            .nodes
            .iter()
            .all(|node| node.cache_op_id().is_none())
    );
    assert!(pipeline.tile_cache.is_none());
}

#[test]
fn compile_keeps_cache_on_branch_points() {
    use crate::domain::ops::conversion::BandJoin;

    let source = ZeroSource::<U8>::new(8, 8, 1);
    let mut arena = PipelineArena::with_source(Box::new(source));
    let root = arena.add_node(non_pixel_local_pass_op(1));
    let branch_a = arena.add_node(non_pixel_local_pass_op(1));
    let branch_b = arena.add_node(non_pixel_local_pass_op(1));
    let merge = arena.add_node(Box::new(BandJoin::new(1, 1, BandFormatId::U8)));

    arena.connect(root, branch_a).unwrap();
    arena.connect(root, branch_b).unwrap();
    arena.connect_to_slot(branch_a, merge, 0).unwrap();
    arena.connect_to_slot(branch_b, merge, 1).unwrap();
    arena
        .enable_cache(root, NonZeroUsize::new(64).unwrap())
        .unwrap();

    let pipeline = arena.compile().unwrap();

    assert_eq!(
        pipeline
            .nodes
            .iter()
            .filter(|node| node.cache_op_id().is_some())
            .count(),
        1
    );
    assert!(pipeline.tile_cache.is_some());
}

#[test]
fn sequential_builder_enables_thin_strip_streaming_defaults() {
    use crate::adapters::pipeline::LineCacheConfig;

    let source = ZeroSource::<U8>::new(64, 64, 1);
    let pipeline = PipelineBuilder::from_source(source)
        .sequential(0)
        .then(pass_op(1))
        .unwrap()
        .build()
        .unwrap();

    let expected_lines =
        DemandHint::ThinStrip.tile_height(pipeline.width, pipeline.height) as usize * 2;
    assert!(pipeline.sequential);
    assert_eq!(pipeline.demand_hint, DemandHint::ThinStrip);
    assert_eq!(
        pipeline.sequential_line_cache,
        Some(LineCacheConfig {
            lines_ahead: expected_lines,
        })
    );
}

#[test]
fn linecache_op_exposes_explicit_line_budget() {
    use crate::adapters::pipeline::LineCacheConfig;
    use crate::domain::ops::conversion::LineCacheOp;

    let source = ZeroSource::<U8>::new(64, 64, 1);
    let pipeline = PipelineBuilder::from_source(source)
        .apply(LineCacheOp::new(7))
        .unwrap()
        .then(pass_op(1))
        .unwrap()
        .build()
        .unwrap();

    assert!(!pipeline.sequential);
    assert_eq!(pipeline.demand_hint, DemandHint::ThinStrip);
    assert_eq!(
        pipeline.sequential_line_cache,
        Some(LineCacheConfig { lines_ahead: 7 })
    );
}

#[test]
fn compile_keeps_cache_on_merge_points() {
    use crate::domain::ops::conversion::BandJoin;

    let source = ZeroSource::<U8>::new(8, 8, 1);
    let mut arena = PipelineArena::with_source(Box::new(source));
    let left = arena.add_node(non_pixel_local_pass_op(1));
    let right = arena.add_node(non_pixel_local_pass_op(1));
    let merge = arena.add_node(Box::new(BandJoin::new(1, 1, BandFormatId::U8)));
    let sink = arena.add_node(non_pixel_local_pass_op(2));

    arena.connect_to_slot(left, merge, 0).unwrap();
    arena.connect_to_slot(right, merge, 1).unwrap();
    arena.connect(merge, sink).unwrap();
    arena
        .enable_cache(merge, NonZeroUsize::new(64).unwrap())
        .unwrap();

    let pipeline = arena.compile().unwrap();

    assert_eq!(
        pipeline
            .nodes
            .iter()
            .filter(|node| node.cache_op_id().is_some())
            .count(),
        1
    );
    assert!(pipeline.tile_cache.is_some());
}

#[test]
fn reorder_completes_one_branch_before_the_other_and_reduces_buffer_bytes() {
    use crate::domain::ops::conversion::BandJoin;

    let source = ZeroSource::<U8>::new(64, 64, 1);
    let mut arena = PipelineArena::with_source(Box::new(source));
    let root = arena.add_node(non_pixel_local_pass_op(1));
    let branch_a_1 = arena.add_node(non_pixel_local_pass_op(1));
    let branch_a_2 = arena.add_node(non_pixel_local_pass_op(1));
    let branch_b_1 = arena.add_node(non_pixel_local_pass_op(1));
    let branch_b_2 = arena.add_node(non_pixel_local_pass_op(1));
    let merge = arena.add_node(Box::new(BandJoin::new(1, 1, BandFormatId::U8)));

    arena.connect(root, branch_a_1).unwrap();
    arena.connect(branch_a_1, branch_a_2).unwrap();
    arena.connect(root, branch_b_1).unwrap();
    arena.connect(branch_b_1, branch_b_2).unwrap();
    arena.connect_to_slot(branch_a_2, merge, 0).unwrap();
    arena.connect_to_slot(branch_b_2, merge, 1).unwrap();

    let pipeline = arena.compile().unwrap();

    assert_eq!(pipeline.nodes.len(), 6);
    assert_eq!(pipeline.nodes[1].input_upstreams()[0], Some(0));
    assert_eq!(pipeline.nodes[2].input_upstreams()[0], Some(1));
    assert_eq!(pipeline.nodes[3].input_upstreams()[0], Some(0));
    assert_eq!(pipeline.nodes[4].input_upstreams()[0], Some(3));
    assert_eq!(pipeline.nodes[5].input_upstreams(), &[Some(2), Some(4)]);
    assert!(
        pipeline.buffer_count < 7,
        "branching reorder should recycle at least one intermediate buffer"
    );
}

#[test]
fn ifthenelse_pipeline_accepts_u8_condition_and_f32_branches() {
    use crate::adapters::{
        scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
        sources::memory::MemorySource,
    };
    use crate::domain::ops::{
        arithmetic::linear::Linear,
        conversion::{IfThenElseOp, cast::Cast},
    };
    use crate::ports::scheduler::TileScheduler;

    let source = MemorySource::<U8>::new(3, 1, 1, vec![0, 64, 255]).unwrap();
    let mut arena = PipelineArena::with_source(Box::new(source));

    let cond = arena.add_node(pass_op(1));
    let cast = arena.add_node(Box::new(OperationBridge::new_pixel_local(
        Cast::<U8, F32>::new(1),
        1,
    )));
    let then_node = arena.add_node(Box::new(OperationBridge::new_pixel_local(
        Linear::<F32>::new(1.0, 10.0).unwrap(),
        1,
    )));
    let else_node = arena.add_node(Box::new(OperationBridge::new_pixel_local(
        Linear::<F32>::new(-1.0, -10.0).unwrap(),
        1,
    )));
    let merge = arena.add_node(Box::new(IfThenElseOp::<F32>::new(1)));

    arena.connect(cond, cast).unwrap();
    arena.connect(cast, then_node).unwrap();
    arena.connect(cast, else_node).unwrap();
    arena.connect_to_slot(cond, merge, 0).unwrap();
    arena.connect_to_slot(then_node, merge, 1).unwrap();
    arena.connect_to_slot(else_node, merge, 2).unwrap();

    let pipeline = arena.compile().unwrap();
    assert_eq!(pipeline.output_format, BandFormatId::F32);
    assert_eq!(pipeline.output_bands, 1);

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();
    let output = bytemuck::cast_slice::<u8, f32>(&sink.into_buffer()).to_vec();

    assert_eq!(output[0], -10.0);
    assert!((output[1] - (10.0 + 64.0 / 255.0)).abs() < f32::EPSILON);
    assert_eq!(output[2], 11.0);
}

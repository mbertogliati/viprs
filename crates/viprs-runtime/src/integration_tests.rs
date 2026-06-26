use std::{
    fs,
    num::NonZeroU8,
    sync::{Arc, Mutex},
};

use viprs_codecs::{
    png::PngCodec,
    tiff::{TiffDecoder, TiffEncoder},
    webp::WebpCodec,
};
use viprs_core::{
    codec_options::{LoadOptions, SaveOptions},
    error::ViprsError,
    format::{F32, U8, U16},
    image::{DemandHint, Image, Region},
    op::{DynOperation, OperationBridge},
};
use viprs_ops_colour::colour::DE00;
use viprs_ops_composite::conversion::{
    autorot::AutorotBridge,
    bandfold::BandfoldBridge,
    bandunfold::BandunfoldBridge,
    falsecolour::{FALSECOLOUR_PET_LUT, FalsecolourOp},
    grid::GridBridge,
    subsample::SubsampleBridge,
};
use viprs_ops_pixel::arithmetic::invert::Invert;
use viprs_ports::{
    codec::{ImageDecoder, ImageEncoder},
    scheduler::TileScheduler,
    source::ImageSource,
};

use crate::{
    pipeline::internal::PipelinePlan, scheduler::rayon_scheduler::RayonScheduler,
    sinks::memory::MemorySink, sources::decoder_source::DecoderSource,
    sources::memory::MemorySource,
};

fn run_u8_pipeline(
    width: u32,
    height: u32,
    bands: u32,
    pixels: Vec<u8>,
    op: Box<dyn DynOperation>,
) -> (u32, u32, u32, Vec<u8>) {
    let source = MemorySource::<U8>::new(width, height, bands, pixels).unwrap();
    let pipeline = PipelinePlan::from_source(source)
        .append_dyn_op(op)
        .unwrap()
        .compile()
        .unwrap();
    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();
    (
        pipeline.width,
        pipeline.height,
        pipeline.output_bands,
        sink.into_buffer(),
    )
}

fn expected_subsample(
    input: &[u8],
    input_width: usize,
    bands: usize,
    output_width: usize,
    output_height: usize,
    xfac: usize,
    yfac: usize,
) -> Vec<u8> {
    let mut output = vec![0u8; output_width * output_height * bands];
    for out_y in 0..output_height {
        let src_y = out_y * yfac;
        for out_x in 0..output_width {
            let src_x = out_x * xfac;
            let src = (src_y * input_width + src_x) * bands;
            let dst = (out_y * output_width + out_x) * bands;
            output[dst..dst + bands].copy_from_slice(&input[src..src + bands]);
        }
    }
    output
}

struct RecordingSource {
    reads: Arc<Mutex<Vec<Region>>>,
}

impl ImageSource for RecordingSource {
    type Format = U8;

    fn width(&self) -> u32 {
        64
    }

    fn height(&self) -> u32 {
        32
    }

    fn bands(&self) -> u32 {
        1
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        self.reads.lock().unwrap().push(region);
        for row in 0..region.height {
            for col in 0..region.width {
                let x = (region.x + col as i32).clamp(0, 63) as u8;
                let y = (region.y + row as i32).clamp(0, 31) as u8;
                output[row as usize * region.width as usize + col as usize] = x + y;
            }
        }
        Ok(())
    }
}

fn clamped_region_pixels_u8(image: &Image<U8>, region: Region) -> Vec<u8> {
    let bands = image.bands() as usize;
    let mut output = vec![0u8; region.pixel_count() * bands];
    for out_y in 0..region.height {
        let src_y = (region.y + out_y as i32).clamp(0, image.height() as i32 - 1) as usize;
        for out_x in 0..region.width {
            let src_x = (region.x + out_x as i32).clamp(0, image.width() as i32 - 1) as usize;
            let src = (src_y * image.width() as usize + src_x) * bands;
            let dst = (out_y as usize * region.width as usize + out_x as usize) * bands;
            output[dst..dst + bands].copy_from_slice(&image.pixels()[src..src + bands]);
        }
    }
    output
}

fn patterned_rgb(width: u32, height: u32) -> Image<U8> {
    let mut data = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            data.push(((x * 17 + y * 3) % 256) as u8);
            data.push(((x * 7 + y * 29) % 256) as u8);
            data.push((((x ^ y) * 11) % 256) as u8);
        }
    }
    Image::from_buffer(width, height, 3, data).unwrap()
}

#[test]
fn de00_pipeline_rejects_three_band_input() {
    let source = MemorySource::<F32>::new(1, 1, 3, vec![0.0, 0.0, 0.0]).unwrap();
    assert!(
        PipelinePlan::from_source(source)
            .append_dyn_op(Box::new(OperationBridge::new_pixel_local(DE00, 3)))
            .is_err()
    );
}

#[test]
fn de00_pipeline_emits_single_distance_band() {
    let source =
        MemorySource::<F32>::new(1, 1, 6, vec![50.0, 10.0, 20.0, 40.0, -20.0, 10.0]).unwrap();
    let pipeline = PipelinePlan::from_source(source)
        .append_dyn_op(Box::new(OperationBridge::new_pixel_local(DE00, 6)))
        .unwrap()
        .compile()
        .unwrap();
    let image = pipeline
        .run_to_image::<F32, _>(&RayonScheduler::new(1).unwrap())
        .unwrap();

    assert_eq!(image.bands(), 1);
    assert!(image.pixels()[0] > 0.0);
}

#[test]
fn falsecolour_pipeline_maps_grayscale_to_rgb() {
    let (_, _, bands, output) = run_u8_pipeline(
        2,
        1,
        1,
        vec![0, 255],
        Box::new(FalsecolourOp::<U8>::new().into_bridge(1)),
    );

    assert_eq!(bands, 3);
    assert_eq!(
        output,
        [FALSECOLOUR_PET_LUT[0], FALSECOLOUR_PET_LUT[255]].concat()
    );
}

#[test]
fn grid_pipeline_arranges_strips() {
    let (width, height, bands, output) = run_u8_pipeline(
        2,
        8,
        1,
        vec![1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
        Box::new(GridBridge::<U8>::new(2, 8, 2, 2, 1)),
    );

    assert_eq!((width, height, bands), (4, 4, 1));
    assert_eq!(output, vec![1, 1, 3, 3, 2, 2, 4, 4, 5, 5, 7, 7, 6, 6, 8, 8]);
}

#[test]
fn bandfold_and_bandunfold_pipelines_report_geometry() {
    let (width, height, bands, folded) = run_u8_pipeline(
        4,
        1,
        2,
        vec![1, 10, 2, 20, 3, 30, 4, 40],
        Box::new(BandfoldBridge::<U8>::new(2, 4, 2)),
    );
    assert_eq!((width, height, bands), (2, 1, 4));

    let (width, height, bands, unfolded) =
        run_u8_pipeline(2, 1, 4, folded, Box::new(BandunfoldBridge::<U8>::new(2, 4)));
    assert_eq!((width, height, bands), (4, 1, 2));
    assert_eq!(unfolded, vec![1, 10, 2, 20, 3, 30, 4, 40]);
}

#[test]
fn autorot_pipeline_applies_orientation_six() {
    let (width, height, bands, output) = run_u8_pipeline(
        2,
        3,
        1,
        vec![1, 2, 3, 4, 5, 6],
        Box::new(AutorotBridge::<U8>::new(2, 3, 1, 6)),
    );

    assert_eq!((width, height, bands), (3, 2, 1));
    assert_eq!(output, vec![5, 3, 1, 6, 4, 2]);
}

#[test]
fn subsample_pipeline_uses_subsampled_dimensions() {
    let (width, height, bands, output) = run_u8_pipeline(
        4,
        4,
        1,
        (0u8..16).collect(),
        Box::new(SubsampleBridge::<U8>::new(2, 2, 1).unwrap()),
    );

    assert_eq!((width, height, bands), (2, 2, 1));
    assert_eq!(output, vec![0, 2, 8, 10]);
}

#[test]
fn subsample_point_mode_pipeline_reads_single_source_pixels() {
    let reads = Arc::new(Mutex::new(Vec::new()));
    let source = RecordingSource {
        reads: Arc::clone(&reads),
    };
    let pipeline = PipelinePlan::from_source(source)
        .append_dyn_op(Box::new(
            SubsampleBridge::<U8>::with_point(12, 5, 1, true).unwrap(),
        ))
        .unwrap()
        .compile()
        .unwrap();
    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();

    RayonScheduler::new(1)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();

    let reads = reads.lock().unwrap();
    assert_eq!(reads.len(), 30);
    assert!(
        reads
            .iter()
            .all(|region| region.width == 1 && region.height == 1)
    );
    assert_eq!(reads[0], Region::new(0, 0, 1, 1));
    assert_eq!(reads[1], Region::new(12, 0, 1, 1));
    assert_eq!(reads[5], Region::new(0, 5, 1, 1));
    assert!(!reads.contains(&Region::new(0, 0, 49, 26)));

    let expected = expected_subsample(
        &(0..32)
            .flat_map(|y| (0..64).map(move |x| (x + y) as u8))
            .collect::<Vec<_>>(),
        64,
        1,
        5,
        6,
        12,
        5,
    );
    assert_eq!(sink.into_buffer(), expected);
}

#[test]
fn subsample_point_mode_pipeline_after_invert_reads_single_source_pixels() {
    let reads = Arc::new(Mutex::new(Vec::new()));
    let source = RecordingSource {
        reads: Arc::clone(&reads),
    };
    let pipeline = PipelinePlan::from_source(source)
        .append_dyn_op(Box::new(OperationBridge::new_pixel_local(
            Invert::<U8>::new(),
            1,
        )))
        .unwrap()
        .append_dyn_op(Box::new(
            SubsampleBridge::<U8>::with_point(12, 5, 1, true).unwrap(),
        ))
        .unwrap()
        .compile()
        .unwrap();
    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();

    RayonScheduler::new(1)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();

    let reads = reads.lock().unwrap();
    assert_eq!(reads.len(), 30);
    assert!(
        reads
            .iter()
            .all(|region| region.width == 1 && region.height == 1)
    );
    assert_eq!(reads[0], Region::new(0, 0, 1, 1));
    assert_eq!(reads[1], Region::new(12, 0, 1, 1));
    assert_eq!(reads[5], Region::new(0, 5, 1, 1));
    assert!(!reads.contains(&Region::new(0, 0, 49, 26)));

    let expected = expected_subsample(
        &(0..32)
            .flat_map(|y| (0..64).map(move |x| 255 - (x + y) as u8))
            .collect::<Vec<_>>(),
        64,
        1,
        5,
        6,
        12,
        5,
    );
    assert_eq!(sink.into_buffer(), expected);
}

#[test]
fn png_decoder_source_streams_regions_without_resident_frame() {
    let image = Image::<U8>::from_buffer(5, 4, 3, (0u8..60).collect()).unwrap();
    let encoded = PngCodec::default().encode(&image).unwrap();
    let source =
        DecoderSource::<_, U8>::streaming(PngCodec::default(), &encoded, LoadOptions::default())
            .unwrap();

    assert!(source.is_streaming());
    assert_eq!(source.resident_decoded_bytes(), 0);

    let region = Region::new(2, 2, 2, 2);
    let mut output = vec![0u8; region.pixel_count() * 3];
    source.read_region(region, &mut output).unwrap();
    assert_eq!(output, vec![36, 37, 38, 39, 40, 41, 51, 52, 53, 54, 55, 56]);
}

#[test]
fn png_decoder_source_streams_u16_region() {
    let image = Image::<U16>::from_buffer(3, 3, 3, (0u16..27).map(|sample| sample * 257).collect())
        .unwrap();
    let encoded = PngCodec::default().encode(&image).unwrap();
    let source =
        DecoderSource::<_, U16>::streaming(PngCodec::default(), &encoded, LoadOptions::default())
            .unwrap();
    let region = Region::new(1, 1, 2, 1);
    let mut output = vec![0u8; region.pixel_count() * 3 * std::mem::size_of::<u16>()];

    source.read_region(region, &mut output).unwrap();
    let samples: &[u16] = bytemuck::try_cast_slice(&output).unwrap();
    assert_eq!(samples, &[3084, 3341, 3598, 3855, 4112, 4369]);
    assert_eq!(source.resident_decoded_bytes(), 0);
}

#[test]
fn png_decoder_source_streams_interlaced_regions() {
    let image =
        Image::<U8>::from_buffer(8, 8, 3, (0u8..192).cycle().take(8 * 8 * 3).collect()).unwrap();
    let encoded = PngCodec::default().encode(&image).unwrap();
    let eager = PngCodec::default().decode::<U8>(&encoded).unwrap();
    let source =
        DecoderSource::<_, U8>::streaming(PngCodec::default(), &encoded, LoadOptions::default())
            .unwrap();
    let region = Region::new(-1, 6, 4, 3);
    let mut output = vec![0u8; region.pixel_count() * 3];

    source.read_region(region, &mut output).unwrap();
    assert_eq!(output, clamped_region_pixels_u8(&eager, region));
}

#[test]
fn tiff_decoder_source_streams_path_regions() {
    let image =
        Image::<U8>::from_buffer(8, 6, 3, (0..8 * 6 * 3).map(|v| (v % 251) as u8).collect())
            .unwrap();
    let encoded = TiffEncoder::default()
        .encode_with_options(
            &image,
            &SaveOptions::default()
                .with_tile_width(4)
                .with_tile_height(3),
        )
        .unwrap();
    let eager = TiffDecoder
        .decode_with_options::<U8>(&encoded, &LoadOptions::default())
        .unwrap();
    let path = std::env::temp_dir().join("viprs-runtime-tiff-streaming-region.tiff");
    fs::write(&path, &encoded).unwrap();

    let source =
        DecoderSource::<_, U8>::streaming_path(TiffDecoder, &path, LoadOptions::default()).unwrap();
    let region = Region::new(-1, 4, 4, 3);
    let mut output = vec![0u8; region.pixel_count() * eager.bands() as usize];
    source.read_region(region, &mut output).unwrap();
    let _ = fs::remove_file(path);

    assert!(source.is_streaming());
    assert_eq!(source.resident_decoded_bytes(), 0);
    assert_eq!(output, clamped_region_pixels_u8(&eager, region));
}

#[test]
fn webp_decoder_source_streams_shrunk_regions() {
    let image = patterned_rgb(19, 17);
    let encoded = WebpCodec.encode(&image).unwrap();
    let opts = LoadOptions::default().with_shrink(NonZeroU8::new(2).unwrap());
    let eager = WebpCodec
        .decode_with_options::<U8>(&encoded, &opts)
        .unwrap();
    let source = DecoderSource::<_, U8>::streaming(WebpCodec, &encoded, opts).unwrap();
    let region = Region::new(1, 1, 2, 2);
    let mut output = vec![0u8; region.pixel_count() * eager.bands() as usize];

    source.read_region(region, &mut output).unwrap();
    assert!(source.is_streaming());
    assert_eq!(source.resident_decoded_bytes(), 0);
    assert_eq!(output, clamped_region_pixels_u8(&eager, region));
}

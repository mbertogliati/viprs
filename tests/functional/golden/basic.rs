use super::support as golden;

#[allow(dead_code)]
use std::{any::Any, fs};

use viprs::{
    BuildError, DynOperation, Multiply, OperationBridge, TileScheduler,
    adapters::{
        scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
        sources::memory::MemorySource,
    },
    domain::{
        colorspace::{ColorspaceId, Lab},
        format::{BandFormat, BandFormatId, F32, U8},
        image::{DemandHint, Region},
        op::NodeSpec,
        ops::{
            arithmetic::{Abs, Divide},
            conversion::ExtendMode,
            convolution::GaussBlur,
            resample::{ShrinkH, ShrinkV},
        },
        resample::ResampleOp,
    },
};

const ARITHMETIC_WIDTH: u32 = 16;
const ARITHMETIC_HEIGHT: u32 = 16;
const COLOUR_WIDTH: u32 = 16;
const COLOUR_HEIGHT: u32 = 16;
const GAUSS_WIDTH: u32 = 64;
const GAUSS_HEIGHT: u32 = 64;
const STRUCTURAL_WIDTH: u32 = 10;
const STRUCTURAL_HEIGHT: u32 = 16;

struct TestResampleBridge<T: ResampleOp> {
    inner: OperationBridge<T>,
}

impl<T> TestResampleBridge<T>
where
    T: ResampleOp,
    T::Input: BandFormat,
    T::Output: BandFormat,
    <T::Input as BandFormat>::Sample: bytemuck::Pod,
    <T::Output as BandFormat>::Sample: bytemuck::Pod,
{
    fn new(op: T, bands: u32) -> Self {
        Self {
            inner: OperationBridge::new(op, bands),
        }
    }
}

impl<T> DynOperation for TestResampleBridge<T>
where
    T: ResampleOp + Send + Sync,
    T::Input: BandFormat,
    T::Output: BandFormat,
    <T::Input as BandFormat>::Sample: bytemuck::Pod,
    <T::Output as BandFormat>::Sample: bytemuck::Pod,
{
    fn input_format(&self) -> BandFormatId {
        self.inner.input_format()
    }

    fn output_format(&self) -> BandFormatId {
        self.inner.output_format()
    }

    fn bands(&self) -> u32 {
        self.inner.bands()
    }

    fn demand_hint(&self) -> DemandHint {
        self.inner.demand_hint()
    }

    fn required_input_region(&self, output: &Region) -> Region {
        self.inner.required_input_region(output)
    }

    fn output_width(&self, input_w: u32) -> u32 {
        self.inner.op.output_width(input_w)
    }

    fn output_height(&self, input_h: u32) -> u32 {
        self.inner.op.output_height(input_h)
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        self.inner.node_spec(tile_w, tile_h)
    }

    fn dyn_start(&self) -> Box<dyn Any + Send> {
        self.inner.dyn_start()
    }

    fn dyn_process_region(
        &self,
        state: &mut dyn Any,
        input: &[u8],
        output: &mut [u8],
        input_region: Region,
        output_region: Region,
    ) {
        self.inner
            .dyn_process_region(state, input, output, input_region, output_region);
    }
}

fn arithmetic_source() -> Vec<f32> {
    let mut pixels = Vec::with_capacity((ARITHMETIC_WIDTH * ARITHMETIC_HEIGHT) as usize);
    for y in 0..ARITHMETIC_HEIGHT {
        for x in 0..ARITHMETIC_WIDTH {
            pixels.push(x as f32 * 1.25 - y as f32 * 0.5 - 6.0);
        }
    }
    pixels
}

fn arithmetic_rhs() -> Vec<f32> {
    let mut pixels = Vec::with_capacity((ARITHMETIC_WIDTH * ARITHMETIC_HEIGHT) as usize);
    for y in 0..ARITHMETIC_HEIGHT {
        for x in 0..ARITHMETIC_WIDTH {
            pixels.push(((x % 5) as f32 + 1.0) * 0.5 + (y % 4) as f32 * 0.25);
        }
    }
    pixels
}

fn colour_source() -> Vec<u8> {
    let mut pixels = Vec::with_capacity((COLOUR_WIDTH * COLOUR_HEIGHT * 3) as usize);
    for y in 0..COLOUR_HEIGHT {
        for x in 0..COLOUR_WIDTH {
            pixels.push(((x * 17 + y * 3) % 256) as u8);
            pixels.push(((x * 11 + y * 19) % 256) as u8);
            pixels.push(((x * 7 + y * 5) % 256) as u8);
        }
    }
    pixels
}

fn gauss_source() -> Vec<f32> {
    vec![37.5; (GAUSS_WIDTH * GAUSS_HEIGHT) as usize]
}

fn structural_source() -> Vec<u8> {
    let mut pixels = Vec::with_capacity((STRUCTURAL_WIDTH * STRUCTURAL_HEIGHT) as usize);
    for y in 0..STRUCTURAL_HEIGHT {
        for x in 0..STRUCTURAL_WIDTH {
            pixels.push(((x * 17 + y * 13) % 256) as u8);
        }
    }
    pixels
}

fn run_pipeline_u8<S: viprs_runtime::pipeline::internal::CommitPlan>(
    source_pixels: Vec<u8>,
    width: u32,
    height: u32,
    bands: u32,
    configure: impl FnOnce(
        viprs_runtime::pipeline::internal::PipelinePlan,
    )
        -> Result<viprs_runtime::pipeline::internal::PipelinePlan<S>, BuildError>,
) -> Vec<u8> {
    let source = MemorySource::<U8>::new(width, height, bands, source_pixels).unwrap();
    let pipeline = configure(viprs_runtime::pipeline::internal::PipelinePlan::from_source(source))
        .unwrap()
        .compile()
        .unwrap();

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();
    sink.into_buffer()
}

fn run_pipeline_f32<S: viprs_runtime::pipeline::internal::CommitPlan>(
    source_pixels: Vec<f32>,
    width: u32,
    height: u32,
    bands: u32,
    configure: impl FnOnce(
        viprs_runtime::pipeline::internal::PipelinePlan,
    )
        -> Result<viprs_runtime::pipeline::internal::PipelinePlan<S>, BuildError>,
) -> Vec<u8> {
    let source = MemorySource::<F32>::new(width, height, bands, source_pixels).unwrap();
    let pipeline = configure(viprs_runtime::pipeline::internal::PipelinePlan::from_source(source))
        .unwrap()
        .compile()
        .unwrap();

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();
    sink.into_buffer()
}

fn read_fixture(op: &str, case: &str) -> Vec<u8> {
    let path = golden::fixture_path(op, case);
    fs::read(&path).unwrap_or_else(|e| panic!("failed to read fixture {path:?}: {e}"))
}

fn assert_exact_fixture(op: &str, case: &str, actual: &[u8]) {
    let expected = read_fixture(op, case);
    assert_eq!(
        actual,
        expected.as_slice(),
        "golden mismatch for op={op} case={case}"
    );
}

fn decode_f32_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn assert_f32_fixture(op: &str, case: &str, actual: &[u8], epsilon: f32) {
    let expected = decode_f32_le(&read_fixture(op, case));
    let actual = decode_f32_le(actual);
    assert_eq!(
        actual.len(),
        expected.len(),
        "f32 golden length mismatch for op={op} case={case}"
    );

    for (idx, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= epsilon,
            "f32 golden mismatch for op={op} case={case} at sample {idx}: got {got}, want {want}, epsilon {epsilon}"
        );
    }
}

#[test]
fn multiply_matches_libvips_fixture() {
    let source = arithmetic_source();
    let rhs = arithmetic_rhs();

    let output = run_pipeline_f32(
        source,
        ARITHMETIC_WIDTH,
        ARITHMETIC_HEIGHT,
        1,
        move |builder| {
            builder.append_dyn_op(Box::new(OperationBridge::new_pixel_local(
                Multiply::<F32>::new(rhs),
                1,
            )))
        },
    );

    assert_f32_fixture("multiply", "gradient_times_modulated", &output, 1e-6);
}

#[test]
fn divide_matches_libvips_fixture() {
    let source = arithmetic_source();
    let rhs = arithmetic_rhs();

    let output = run_pipeline_f32(
        source,
        ARITHMETIC_WIDTH,
        ARITHMETIC_HEIGHT,
        1,
        move |builder| {
            builder.append_dyn_op(Box::new(OperationBridge::new_pixel_local(
                Divide::<F32>::new(rhs, ARITHMETIC_WIDTH, 1),
                1,
            )))
        },
    );

    assert_f32_fixture("divide", "gradient_divided_by_modulated", &output, 1e-6);
}

#[test]
fn abs_matches_libvips_fixture() {
    let source = arithmetic_source();

    let output = run_pipeline_f32(source, ARITHMETIC_WIDTH, ARITHMETIC_HEIGHT, 1, |builder| {
        builder.append_dyn_op(Box::new(OperationBridge::new_pixel_local(
            Abs::<F32>::new(),
            1,
        )))
    });

    assert_f32_fixture("abs", "signed_gradient", &output, 1e-6);
}

#[test]
fn srgb_to_lab_matches_libvips_fixture() {
    let source = colour_source();

    let output = run_pipeline_u8(source, COLOUR_WIDTH, COLOUR_HEIGHT, 3, |builder| {
        builder
            .with_colorspace(ColorspaceId::SRgb)
            .plan_colourspace::<Lab>()
    });

    assert_f32_fixture("srgb_to_lab", "rgb_gradient", &output, 3e-2);
}

#[test]
fn gauss_blur_matches_libvips_fixture() {
    let source = gauss_source();
    let blur = GaussBlur::new(1.5);

    let output = run_pipeline_f32(source, GAUSS_WIDTH, GAUSS_HEIGHT, 1, |builder| {
        builder
            .append_dyn_op(Box::new(OperationBridge::new(blur.h, 1)))?
            .append_dyn_op(Box::new(OperationBridge::new(blur.v, 1)))
    });

    assert_f32_fixture("gauss_blur", "uniform_sigma_1_5", &output, 1e-3);
}

#[test]
fn flip_horizontal_matches_libvips_fixture() {
    let source = structural_source();

    let output = run_pipeline_u8(source, STRUCTURAL_WIDTH, STRUCTURAL_HEIGHT, 1, |builder| {
        builder.plan_flip_horizontal()
    });

    assert_exact_fixture("flip_horizontal", "grayscale_gradient", &output);
}

#[test]
fn rotate90_matches_libvips_fixture() {
    let source = structural_source();

    let output = run_pipeline_u8(source, STRUCTURAL_WIDTH, STRUCTURAL_HEIGHT, 1, |builder| {
        builder.plan_rotate90()
    });

    assert_exact_fixture("rotate90", "grayscale_gradient", &output);
}

#[test]
fn embed_matches_libvips_fixture() {
    let source = structural_source();

    let output = run_pipeline_u8(source, STRUCTURAL_WIDTH, STRUCTURAL_HEIGHT, 1, |builder| {
        builder.plan_embed(
            14,
            18,
            2,
            1,
            STRUCTURAL_WIDTH,
            STRUCTURAL_HEIGHT,
            ExtendMode::Black,
        )
    });

    assert_exact_fixture("embed", "grayscale_offset_black", &output);
}

#[test]
fn shrinkh_matches_libvips_fixture() {
    let source = structural_source();

    let output = run_pipeline_u8(source, STRUCTURAL_WIDTH, STRUCTURAL_HEIGHT, 1, |builder| {
        builder.append_dyn_op(Box::new(TestResampleBridge::new(
            ShrinkH::<U8>::new(2).unwrap(),
            1,
        )))
    });

    assert_exact_fixture("shrinkh", "grayscale_factor_2", &output);
}

#[test]
fn shrinkv_matches_libvips_fixture() {
    let source = structural_source();

    let output = run_pipeline_u8(source, STRUCTURAL_WIDTH, STRUCTURAL_HEIGHT, 1, |builder| {
        builder.append_dyn_op(Box::new(TestResampleBridge::new(
            ShrinkV::<U8>::new(2).unwrap(),
            1,
        )))
    });

    assert_exact_fixture("shrinkv", "grayscale_factor_2", &output);
}

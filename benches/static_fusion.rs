//! Benchmark: PointFn fusion vs manual kernels vs dyn dispatch.
//!
//! This benchmark keeps the manual kernel and dyn-dispatch baselines while the
//! new zero-buffer `FusedOp` skeleton is wired in.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::any::Any;
use viprs::domain::{
    format::U8,
    image::{DemandHint, Region, Tile, TileMut},
    op::{DynOperation, Op, OperationBridge},
    ops::{
        arithmetic::{clamp::ClampOp, invert::Invert, linear::LinearKernelU8},
        fused::FusedOp,
    },
};

const IMAGE_WIDTHS: [u32; 3] = [512, 2048, 8192];
const BANDS: u32 = 4;

trait PixelKernel: Send + Sync {
    type Input: Copy;
    type Output: Copy;

    fn apply(&self, sample: Self::Input) -> Self::Output;
}

struct FusedKernel<A, B> {
    a: A,
    b: B,
}

impl<A, B> FusedKernel<A, B> {
    fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A, B> PixelKernel for FusedKernel<A, B>
where
    A: PixelKernel,
    B: PixelKernel<Input = A::Output>,
{
    type Input = A::Input;
    type Output = B::Output;

    #[inline(always)]
    fn apply(&self, sample: Self::Input) -> Self::Output {
        self.b.apply(self.a.apply(sample))
    }
}

#[inline(always)]
fn run_kernel<K: PixelKernel<Input = u8, Output = u8>>(
    kernel: &K,
    input: &[u8],
    output: &mut [u8],
) {
    for (src, dst) in input.iter().zip(output.iter_mut()) {
        *dst = kernel.apply(*src);
    }
}

struct InvertKernel;

impl PixelKernel for InvertKernel {
    type Input = u8;
    type Output = u8;

    #[inline(always)]
    fn apply(&self, sample: u8) -> u8 {
        255 - sample
    }
}

struct LinearKernel {
    scale: i16,
    offset: i16,
}

impl PixelKernel for LinearKernel {
    type Input = u8;
    type Output = u8;

    #[inline(always)]
    fn apply(&self, sample: u8) -> u8 {
        let v = (sample as i16) * self.scale + self.offset;
        v.clamp(0, 255) as u8
    }
}

struct ClampKernel {
    min: i16,
    max: i16,
}

impl PixelKernel for ClampKernel {
    type Input = u8;
    type Output = u8;

    #[inline(always)]
    fn apply(&self, sample: u8) -> u8 {
        (sample as i16).clamp(self.min, self.max) as u8
    }
}

fn make_fused_kernel() -> FusedKernel<FusedKernel<InvertKernel, LinearKernel>, ClampKernel> {
    FusedKernel::new(
        FusedKernel::new(
            InvertKernel,
            LinearKernel {
                scale: 2,
                offset: 10,
            },
        ),
        ClampKernel { min: 0, max: 255 },
    )
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_fused_invert_linear_clamp(src: &[u8], dst: &mut [u8], scale: i16, offset: i16) {
    use std::arch::aarch64::{
        vaddq_s16, vcombine_u8, vdupq_n_s16, vget_high_u8, vget_low_u8, vld1q_u8, vmovl_u8,
        vmulq_s16, vmvnq_u8, vqmovun_s16, vreinterpretq_s16_u16, vst1q_u8,
    };

    let len = src.len().min(dst.len());
    let chunks = len / 16;
    let remainder = len % 16;
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    unsafe {
        let v_scale = vdupq_n_s16(scale);
        let v_offset = vdupq_n_s16(offset);

        for i in 0..chunks {
            let offset_bytes = i * 16;
            let data = vld1q_u8(src_ptr.add(offset_bytes));
            let inverted = vmvnq_u8(data);
            let lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(inverted)));
            let hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(inverted)));
            let lo_scaled = vaddq_s16(vmulq_s16(lo, v_scale), v_offset);
            let hi_scaled = vaddq_s16(vmulq_s16(hi, v_scale), v_offset);
            let result = vcombine_u8(vqmovun_s16(lo_scaled), vqmovun_s16(hi_scaled));
            vst1q_u8(dst_ptr.add(offset_bytes), result);
        }

        let base = chunks * 16;
        for i in 0..remainder {
            let x = *src_ptr.add(base + i);
            let inv = 255u8.wrapping_sub(x);
            let v = (inv as i16) * scale + offset;
            *dst_ptr.add(base + i) = v.clamp(0, 255) as u8;
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn run_neon_fused(src: &[u8], dst: &mut [u8], scale: i16, offset: i16) {
    unsafe { neon_fused_invert_linear_clamp(src, dst, scale, offset) };
}

#[cfg(not(target_arch = "aarch64"))]
fn run_neon_fused(src: &[u8], dst: &mut [u8], scale: i16, offset: i16) {
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        let inv = 255u8.wrapping_sub(*s);
        let v = (inv as i16) * scale + offset;
        *d = v.clamp(0, 255) as u8;
    }
}

type PointFnFused3 = FusedOp<FusedOp<Invert<U8>, LinearKernelU8>, ClampOp<U8>>;

fn make_point_fn_fused() -> PointFnFused3 {
    let invert = Invert::<U8>::new();
    let linear = LinearKernelU8::new(2, 10);
    let clamp = ClampOp::<U8>::new(0.0, 255.0);
    let inner = FusedOp::new(invert, linear);
    FusedOp::new(inner, clamp)
}

struct DynSequential3 {
    ops: [Box<dyn DynOperation>; 3],
    tile_w: u32,
    tile_h: u32,
    bands: u32,
}

struct DynSequential3State {
    state0: Box<dyn Any + Send>,
    state1: Box<dyn Any + Send>,
    state2: Box<dyn Any + Send>,
    mid_buf0: Vec<u8>,
    mid_buf1: Vec<u8>,
}

impl DynSequential3 {
    fn new(tile_w: u32, tile_h: u32, bands: u32) -> Self {
        let invert = Box::new(OperationBridge::new_pixel_local(Invert::<U8>::new(), bands))
            as Box<dyn DynOperation>;
        let linear = Box::new(OperationBridge::new_pixel_local(
            LinearKernelU8::new(2, 10),
            bands,
        )) as Box<dyn DynOperation>;
        let clamp = Box::new(OperationBridge::new_pixel_local(
            ClampOp::<U8>::new(0.0, 255.0),
            bands,
        )) as Box<dyn DynOperation>;

        Self {
            ops: [invert, linear, clamp],
            tile_w,
            tile_h,
            bands,
        }
    }

    fn start(&self) -> DynSequential3State {
        let tile_samples = (self.tile_w * self.tile_h * self.bands) as usize;
        DynSequential3State {
            state0: self.ops[0].dyn_start_with_tile_and_bands(self.tile_w, self.tile_h, self.bands),
            state1: self.ops[1].dyn_start_with_tile_and_bands(self.tile_w, self.tile_h, self.bands),
            state2: self.ops[2].dyn_start_with_tile_and_bands(self.tile_w, self.tile_h, self.bands),
            mid_buf0: vec![0u8; tile_samples],
            mid_buf1: vec![0u8; tile_samples],
        }
    }

    fn process_region(&self, state: &mut DynSequential3State, input: &[u8], output: &mut [u8]) {
        let region = Region::new(0, 0, self.tile_w, self.tile_h);

        self.ops[0].dyn_process_region(
            &mut *state.state0,
            input,
            &mut state.mid_buf0,
            region,
            region,
        );
        self.ops[1].dyn_process_region(
            &mut *state.state1,
            &state.mid_buf0,
            &mut state.mid_buf1,
            region,
            region,
        );
        self.ops[2].dyn_process_region(&mut *state.state2, &state.mid_buf1, output, region, region);
    }
}

struct TileFixture {
    label: String,
    region: Region,
    tile_w: u32,
    tile_h: u32,
    tile_samples: usize,
    input: Vec<u8>,
}

fn make_fixture(width: u32) -> TileFixture {
    let height = DemandHint::ThinStrip.tile_height(width, width);
    let region = Region::new(0, 0, width, height);
    let tile_samples = region.pixel_count() * BANDS as usize;
    let input: Vec<u8> = (0..tile_samples).map(|i| (i % 96) as u8).collect();

    TileFixture {
        label: format!("{width}x{height}"),
        region,
        tile_w: width,
        tile_h: height,
        tile_samples,
        input,
    }
}

fn assert_equivalent(fixture: &TileFixture) {
    let fused = make_point_fn_fused();
    let mut fused_state = fused.start();
    let mut fused_output = vec![0u8; fixture.tile_samples];
    {
        let input = Tile::<U8>::new(fixture.region, BANDS, &fixture.input);
        let mut out_tile = TileMut::<U8>::new(fixture.region, BANDS, &mut fused_output);
        fused.process_region(&mut fused_state, &input, &mut out_tile);
    }

    let dyn_seq = DynSequential3::new(fixture.tile_w, fixture.tile_h, BANDS);
    let mut dyn_state = dyn_seq.start();
    let mut dyn_output = vec![0u8; fixture.tile_samples];
    dyn_seq.process_region(&mut dyn_state, &fixture.input, &mut dyn_output);

    let kernel = make_fused_kernel();
    let mut kernel_output = vec![0u8; fixture.tile_samples];
    run_kernel(&kernel, &fixture.input, &mut kernel_output);

    let mut neon_output = vec![0u8; fixture.tile_samples];
    run_neon_fused(&fixture.input, &mut neon_output, 2, 10);

    assert_eq!(
        fused_output, dyn_output,
        "PointFn FusedOp and dyn dispatch must agree"
    );
    assert_eq!(
        fused_output, kernel_output,
        "manual kernel and PointFn FusedOp must agree"
    );
    assert_eq!(
        fused_output, neon_output,
        "NEON kernel and PointFn FusedOp must agree"
    );
}

fn bench_static_fusion_3ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("static_fusion_3ops_u8_rgba");

    for &width in &IMAGE_WIDTHS {
        let fixture = make_fixture(width);
        assert_equivalent(&fixture);
        group.throughput(Throughput::Bytes(fixture.tile_samples as u64));

        group.bench_with_input(
            BenchmarkId::new("neon_fused", &fixture.label),
            &fixture,
            |b, fixture| {
                let mut output = vec![0u8; fixture.tile_samples];
                b.iter(|| {
                    run_neon_fused(&fixture.input, &mut output, 2, 10);
                    black_box(&output);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("kernel_fused", &fixture.label),
            &fixture,
            |b, fixture| {
                let kernel = make_fused_kernel();
                let mut output = vec![0u8; fixture.tile_samples];
                b.iter(|| {
                    run_kernel(&kernel, &fixture.input, &mut output);
                    black_box(&output);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("point_fn_fused_op", &fixture.label),
            &fixture,
            |b, fixture| {
                let fused = make_point_fn_fused();
                let input = Tile::<U8>::new(fixture.region, BANDS, &fixture.input);
                let mut state = fused.start();
                let mut output = vec![0u8; fixture.tile_samples];
                b.iter(|| {
                    let mut out_tile = TileMut::<U8>::new(fixture.region, BANDS, &mut output);
                    fused.process_region(&mut state, &input, &mut out_tile);
                    black_box(&output);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("dyn_dispatched", &fixture.label),
            &fixture,
            |b, fixture| {
                let dyn_seq = DynSequential3::new(fixture.tile_w, fixture.tile_h, BANDS);
                let mut state = dyn_seq.start();
                let mut output = vec![0u8; fixture.tile_samples];
                b.iter(|| {
                    dyn_seq.process_region(&mut state, &fixture.input, &mut output);
                    black_box(&output);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_static_fusion_3ops);
criterion_main!(benches);

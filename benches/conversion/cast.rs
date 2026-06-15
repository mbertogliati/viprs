#[path = "../common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use viprs::domain::{
    format::{F32, U8},
    ops::conversion::cast::Cast,
};

fn bench_cast(c: &mut Criterion) {
    common::bench_direct_op_with_regions::<Cast<U8, F32>, U8, F32, _, _, _>(
        c,
        "cast_u8_to_f32",
        3,
        3,
        || Cast::<U8, F32>::new(3),
        common::direct_tile_regions::<Cast<U8, F32>>,
        |samples| vec![128u8; samples],
    );

    common::bench_direct_op_with_regions::<Cast<F32, U8>, F32, U8, _, _, _>(
        c,
        "cast_f32_to_u8",
        3,
        3,
        || Cast::<F32, U8>::new(3),
        common::direct_tile_regions::<Cast<F32, U8>>,
        |samples| vec![0.5f32; samples],
    );
}

criterion_group!(benches, bench_cast);
criterion_main!(benches);

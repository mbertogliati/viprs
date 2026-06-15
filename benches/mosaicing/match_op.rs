#[path = "../common/mod.rs"]
mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::{
    format::U8,
    image::Region,
    ops::mosaicing::{MatchOp, TiePointPair},
};

fn synthetic_pairs(size: u32) -> Vec<TiePointPair> {
    let transform = (1.0025, -0.018, 0.011, 0.9975, 4.0, -3.5);
    let stride = (size / 8).max(32) as f64;
    let max = f64::from(size.saturating_sub(64));
    let mut pairs = Vec::with_capacity(16);
    let mut y = 32.0;
    while y <= max && pairs.len() < 16 {
        let mut x = 32.0;
        while x <= max && pairs.len() < 16 {
            let xs = transform.0 * x + transform.1 * y + transform.4;
            let ys = transform.2 * x + transform.3 * y + transform.5;
            pairs.push(TiePointPair::from_xy(x, y, xs, ys));
            x += stride;
        }
        y += stride;
    }
    pairs
}

fn bench_match_op(c: &mut Criterion) {
    let mut group = c.benchmark_group("match_op_u8");

    for &size in &common::STANDARD_SIZES {
        let op = MatchOp::<U8>::new(Region::new(0, 0, size, size), Region::new(0, 0, size, size));
        let pairs = synthetic_pairs(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| black_box(op.fit(&pairs).unwrap()));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_match_op);
criterion_main!(benches);

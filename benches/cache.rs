use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use std::sync::Arc;
use viprs::{CacheKey, CachedResult, OperationCache};

fn make_key(seed: u64) -> CacheKey {
    CacheKey::from_hash(seed.wrapping_mul(11_400_714_819_323_198_485))
}

fn make_result(size: usize, fill: u8) -> CachedResult {
    CachedResult {
        data: Arc::new(vec![fill; size]),
        width: size as u32,
        height: 1,
        bands: 1,
    }
}

fn bench_cache_hit(c: &mut Criterion) {
    let cache = OperationCache::with_limits(100, 100 * 256);
    let keys: Vec<_> = (0..100u64).map(make_key).collect();

    for (idx, key) in keys.iter().enumerate() {
        cache.insert(key.clone(), make_result(256, idx as u8));
    }

    let mut next = 0usize;
    c.bench_function("operation_cache_hit", |b| {
        b.iter(|| {
            let key = &keys[next % keys.len()];
            next = next.wrapping_add(1);
            black_box(cache.get(black_box(key)))
        });
    });
}

fn bench_cache_miss(c: &mut Criterion) {
    let cache = OperationCache::with_limits(100, 100 * 256);
    let mut next = 0u64;

    c.bench_function("operation_cache_miss", |b| {
        b.iter(|| {
            let key = make_key(next);
            next = next.wrapping_add(1);
            black_box(cache.get(black_box(&key)))
        });
    });
}

fn bench_cache_insert(c: &mut Criterion) {
    c.bench_function("operation_cache_insert_with_eviction", |b| {
        b.iter_batched(
            || OperationCache::with_limits(100, 100 * 256),
            |cache| {
                for idx in 0..1000u64 {
                    cache.insert(make_key(idx), make_result(256, idx as u8));
                }
                black_box(cache.bytes_used());
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_cache(c: &mut Criterion) {
    bench_cache_hit(c);
    bench_cache_miss(c);
    bench_cache_insert(c);
}

criterion_group!(benches, bench_cache);
criterion_main!(benches);

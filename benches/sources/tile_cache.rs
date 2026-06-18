#![allow(missing_docs)]
/// Benchmark: TileCache<MemorySource<U8>> — LRU tile cache latency and throughput.
///
/// Two scenarios are measured:
///
/// - **hit**: the same 64×64 tile is requested 1 000 times. After the first miss
///   every call is a cache hit; measures mutex + VecDeque promote overhead.
///
/// - **full_scan**: all non-overlapping tiles of the image are requested once each.
///   Every call is a miss; measures allocation + inner-source read + LRU insert cost.
///   The cache is sized to hold all tiles so no eviction occurs during the scan.
///
/// Image sizes: 512, 2 048, 8 192 px (square, single-band U8).
/// Tile size: 64×64 (matches libvips sinkscreen default, see ADR-022 / sinkscreen.c).
use std::num::NonZeroUsize;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::sources::{memory::MemorySource, tile_cache::TileCache},
    domain::{format::U8, image::Region},
    ports::source::ImageSource,
};

const TILE: u32 = 64;

fn make_cache(size: u32, max_bytes: usize) -> TileCache<MemorySource<U8>> {
    let pixels = vec![128u8; (size as usize) * (size as usize)];
    let inner = MemorySource::<U8>::new(size, size, 1, pixels).unwrap();
    TileCache::new(inner, NonZeroUsize::new(max_bytes).unwrap())
}

/// Bytes occupied by one TILE×TILE single-band U8 tile.
fn tile_bytes() -> usize {
    (TILE as usize) * (TILE as usize)
}

fn bench_cache_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("tile_cache_hit");

    for &size in &[512u32, 2048, 8192] {
        // One tile fits easily — no eviction during hit loop.
        let cache = make_cache(size, tile_bytes() * 4);
        let region = Region::new(0, 0, TILE, TILE);
        let mut out = vec![0u8; tile_bytes()];

        // Warm the cache: load the tile once before the benchmark loop.
        cache.read_region(region, &mut out).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                cache.read_region(region, &mut out).unwrap();
                black_box(&out);
            });
        });
    }

    group.finish();
}

fn bench_cache_full_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("tile_cache_full_scan");

    for &size in &[512u32, 2048, 8192] {
        let tiles_x = size / TILE;
        let tiles_y = size / TILE;
        let n_tiles = (tiles_x * tiles_y) as usize;

        // Size the cache to hold all tiles — measures pure miss + insert cost,
        // not eviction cost (eviction is a separate concern).
        let max_bytes = n_tiles * tile_bytes();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                // Construct a fresh cache each iteration so every tile is a miss.
                let cache = make_cache(size, max_bytes);
                let mut out = vec![0u8; tile_bytes()];

                for ty in 0..tiles_y {
                    for tx in 0..tiles_x {
                        let region = Region::new(
                            (tx * TILE) as i32,
                            (ty * TILE) as i32,
                            TILE,
                            TILE,
                        );
                        cache.read_region(region, &mut out).unwrap();
                        black_box(&out);
                    }
                }
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_cache_hit, bench_cache_full_scan);
criterion_main!(benches);

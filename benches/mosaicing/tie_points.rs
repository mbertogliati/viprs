use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{Region, TiePointSearchOp, Tile, U8};

fn make_pixels(size: u32) -> Vec<u8> {
    let pixel_count = size as usize * size as usize;
    (0..pixel_count)
        .map(|idx| (((idx * 37) + (idx / size as usize) * 11) % 251) as u8 + 1)
        .collect()
}

fn bench_tie_points(c: &mut Criterion) {
    let mut group = c.benchmark_group("tie_points_u8");

    for &size in &[512u32, 2048, 8192] {
        let reference_pixels = make_pixels(size);
        let secondary_pixels = reference_pixels.clone();
        let overlap_extent = size.min(128);
        let reference_region = Region::new(0, 0, size, size);
        let secondary_region = Region::new(1, 0, size, size);
        let overlap = Region::new(0, 0, overlap_extent, overlap_extent);
        let search = TiePointSearchOp::new(2).with_minimum_overlap(64);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let reference = Tile::<U8>::new(reference_region, 1, &reference_pixels);
                let secondary = Tile::<U8>::new(secondary_region, 1, &secondary_pixels);
                let result = search.search(&reference, &secondary, overlap).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_tie_points);
criterion_main!(benches);

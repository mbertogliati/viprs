use bytemuck::cast_slice;
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::{format::U8, image::Region, op::DynOperation, ops::histogram::HistPlotOp};

fn histogram_bins(len: usize) -> Vec<u8> {
    (0..len)
        .map(|idx| ((idx * 17 + idx / 9 * 23) % 256) as u8)
        .collect()
}

// Criterion-only baseline: `hist_plot` consumes histogram tensors rather than a source image,
// so there is no direct xtask/libvips fixture entrypoint to compare through the current runner.
fn bench_hist_plot(c: &mut Criterion) {
    {
        let mut horizontal = c.benchmark_group("hist_plot_horizontal_u8");
        for &size in &[512_u32, 2048, 8192] {
            let horizontal_hist = histogram_bins(size as usize);
            let horizontal_op =
                HistPlotOp::<U8>::from_histogram(size, 1, 1, &horizontal_hist).unwrap();
            let horizontal_input = Region::new(0, 0, size, 1);
            let horizontal_output_region = Region::new(
                0,
                0,
                horizontal_op.plot_width(),
                horizontal_op.plot_height(),
            );
            let mut horizontal_output = vec![0u8; horizontal_output_region.pixel_count()];

            horizontal.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
                b.iter(|| {
                    let mut state = ();
                    horizontal_op.dyn_process_region(
                        &mut state,
                        cast_slice(&horizontal_hist),
                        &mut horizontal_output,
                        horizontal_input,
                        horizontal_output_region,
                    );
                    black_box(&horizontal_output);
                });
            });
        }
        horizontal.finish();
    }

    let mut vertical = c.benchmark_group("hist_plot_vertical_u8");
    for &size in &[512_u32, 2048, 8192] {
        let vertical_hist = histogram_bins(size as usize);
        let vertical_op = HistPlotOp::<U8>::from_histogram(1, size, 1, &vertical_hist).unwrap();
        let vertical_input = Region::new(0, 0, 1, size);
        let vertical_output_region =
            Region::new(0, 0, vertical_op.plot_width(), vertical_op.plot_height());
        let mut vertical_output = vec![0u8; vertical_output_region.pixel_count()];

        vertical.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let mut state = ();
                vertical_op.dyn_process_region(
                    &mut state,
                    cast_slice(&vertical_hist),
                    &mut vertical_output,
                    vertical_input,
                    vertical_output_region,
                );
                black_box(&vertical_output);
            });
        });
    }
    vertical.finish();
}

criterion_group!(benches, bench_hist_plot);
criterion_main!(benches);

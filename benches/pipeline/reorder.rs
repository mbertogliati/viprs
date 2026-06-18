#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::{
    op::NodeSpec,
    reorder::{ReorderNode, peak_buffer_slots_for_order, reorder_dag, stable_topological_order},
};

fn branch_graph(depth: usize) -> (Vec<ReorderNode>, Vec<(usize, usize)>) {
    let mut nodes = Vec::with_capacity(depth * 2 + 2);
    let mut edges = Vec::with_capacity(depth * 2 + 2);

    nodes.push(ReorderNode::transform(NodeSpec::identity(128, 128))); // root
    let mut previous_left = 0usize;
    let mut previous_right = 0usize;

    for _ in 0..depth {
        let left = nodes.len();
        nodes.push(ReorderNode::transform(NodeSpec::identity(128, 128)));
        edges.push((previous_left, left));
        previous_left = left;

        let right = nodes.len();
        nodes.push(ReorderNode::transform(NodeSpec::identity(128, 128)));
        edges.push((previous_right, right));
        previous_right = right;
    }

    let merge = nodes.len();
    nodes.push(ReorderNode::transform(NodeSpec::identity(128, 128)));
    edges.push((previous_left, merge));
    edges.push((previous_right, merge));

    (nodes, edges)
}

fn bench_reorder(c: &mut Criterion) {
    let mut group = c.benchmark_group("reorder_graph");

    for depth in [4usize, 16, 64] {
        let (nodes, edges) = branch_graph(depth);
        let stable = stable_topological_order(&nodes, &edges).unwrap();
        let stable_peak = peak_buffer_slots_for_order(&nodes, &edges, &stable).unwrap();
        let reordered = reorder_dag(&nodes, &edges).unwrap();

        assert!(
            peak_buffer_slots_for_order(&nodes, &edges, &reordered.order).unwrap() < stable_peak,
            "reordered schedule should lower monotone buffer pressure for a branching graph"
        );

        group.bench_function(BenchmarkId::new("stable_buffer_slots", depth), |b| {
            b.iter(|| {
                peak_buffer_slots_for_order(
                    black_box(&nodes),
                    black_box(&edges),
                    black_box(&stable),
                )
            })
        });
        group.bench_function(BenchmarkId::new("reordered_schedule", depth), |b| {
            b.iter(|| reorder_dag(black_box(&nodes), black_box(&edges)))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_reorder);
criterion_main!(benches);

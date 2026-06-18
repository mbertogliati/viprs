#![allow(missing_docs)]
#[path = "create/frequency_mask.rs"]
mod frequency_mask;
#[path = "create/invertlut.rs"]
mod invertlut;
#[path = "create/logmat.rs"]
mod logmat;
#[path = "create/tonelut.rs"]
mod tonelut;

use criterion::{criterion_group, criterion_main};

criterion_group!(
    benches,
    frequency_mask::bench_frequency_mask,
    logmat::bench_logmat,
    invertlut::bench_invertlut,
    tonelut::bench_tonelut
);
criterion_main!(benches);

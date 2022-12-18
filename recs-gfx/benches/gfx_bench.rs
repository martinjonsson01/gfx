use criterion::{criterion_group, criterion_main, Bencher, BenchmarkId, Criterion};
use std::time::Duration;

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group(group_name);
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(1000);
    group.bench_with_input(
        BenchmarkId::from_parameter(format!("10^{magnitude:?}")),
        &search_up_to,
        || println!("No benchmark implemented yet"),
    );
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = criterion_benchmark
}
criterion_main!(benches);

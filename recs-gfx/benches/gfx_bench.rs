use criterion::{criterion_group, criterion_main, Criterion};
use std::thread;
use std::time::Duration;

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("example group");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(1000);
    group.bench_function("example", |bencher| {
        bencher.iter(|| thread::sleep(Duration::from_micros(100)))
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = criterion_benchmark
}
criterion_main!(benches);

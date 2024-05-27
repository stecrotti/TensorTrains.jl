using PkgBenchmark, Dates

# adjust these two with the desired commits/branchs
target = "HEAD"
baseline = "main"

bench_target = benchmarkpkg("TensorTrains", target)
bench_base = benchmarkpkg("TensorTrains", baseline)
comparison = judge(bench_target, bench_base)

directory = dirname(@__FILE__)
datetime = string(now())
fn = joinpath(directory, "benchmark_comparison_" * datetime * ".md")
export_markdown(fn, comparison; export_invariants=true)
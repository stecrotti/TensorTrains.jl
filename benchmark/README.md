## Run the benchmarks

```julia
using PkgBenchmark
res = benchmarkpkg("TensorTrains")
```

Pretty-print on a markdown file
```julia
using Dates
day = string(today())
directory = "./"
fn = directory * "benchmark_" * day * ".md"
export_markdown(fn, res)
```

## Compare across commits
Run `benchmark/run_benchmarks.jl` after modifying the baseline and target variables therein.
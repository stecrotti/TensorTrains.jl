## Run the benchmarks

```julia
using PkgBenchmark
res = benchmarkpkg("TensorTrains")
```

Pretty-print on a markdown file
```julia
using Dates
datetime = string(now())
directory = "./"
fn = directory * "benchmark_" * datetime * ".md"
export_markdown(fn, res)
```

## Compare across commits
Run `benchmark/run_benchmarks.jl` after modifying the baseline and target variables therein.
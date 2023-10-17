To run the benchmarks, do

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
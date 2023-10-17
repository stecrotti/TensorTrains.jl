To run the benchmarks, do

```julia
using PkgBenchmark
res = benchmarkpkg("TensorTrains")
```

Optionally pretty-print on a file
```julia
using Dates
day = string(today())
directory = "./"
fn = directory * "benchmark_" * day * ".md"
export_markdown(fn, res)
```

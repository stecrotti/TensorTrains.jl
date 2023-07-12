using Documenter
using TensorTrains

makedocs(
    sitename = "TensorTrains.jl",
    format = Documenter.HTML(),
    modules = [TensorTrains]
)

deploydocs(
    repo = "https://github.com/stecrotti/TensorTrains.jl.git"
)

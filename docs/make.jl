using Documenter
using TensorTrains

makedocs(
    sitename = "TensorTrains",
    format = Documenter.HTML(),
    modules = [TensorTrains]
)

deploydocs(
    repo = "https://github.com/stecrotti/TensorTrains.jl.git"
)

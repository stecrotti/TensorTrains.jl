using Documenter
using TensorTrains
using TensorTrains.UniformTensorTrains

makedocs(
    sitename = "TensorTrains.jl",
    format = Documenter.HTML(),
    modules = [
        TensorTrains,
        TensorTrains.UniformTensorTrains
        ],
    pages = [
        "Home" => "index.md",
        "Guide" => "guide.md"
    ]
)

deploydocs(
    repo = "https://github.com/stecrotti/TensorTrains.jl.git",
    push_preview = true
)

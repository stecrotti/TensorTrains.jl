using Documenter
using TensorTrains
using TensorTrains.UniformTensorTrains

makedocs(
    sitename = "TensorTrains.jl",
    format = Documenter.HTML(),
    modules = [
        TensorTrains,
        ],
    pages = [
        "Home" => "index.md",
        "Guide" => "guide.md"
    ],
    checkdocs=:exports  # doesn't complain if a documented method is not included in the docs, if it's not exported
)

deploydocs(
    repo = "https://github.com/stecrotti/TensorTrains.jl.git",
    push_preview = true
)

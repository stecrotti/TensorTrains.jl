using Documenter
using TensorTrains

makedocs(
    sitename = "TensorTrains.jl",
    format = Documenter.HTML(),
    modules = [TensorTrains],
    pages = [
        "Home" => "index.md",
        "Guide" => "guide.md"
    ]
)

deploydocs(
    repo = "https://github.com/stecrotti/TensorTrains.jl.git"
)

using TensorTrains
using Random, Suppressor, InvertedIndices
using Test
using Aqua

@testset "Aqua" begin
    Aqua.test_all(TensorTrains, ambiguities=false)
    Aqua.test_ambiguities(TensorTrains)
end

include("exact.jl")
include("tensor_train.jl")
include("periodic_tensor_train.jl")
include("svd_trunc.jl")
include("MatrixProductStates.jl")

nothing
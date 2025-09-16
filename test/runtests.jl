using TensorTrains
using TensorTrains: transfer_operator, leading_eig, _merge_tensors
using Random, Suppressor, InvertedIndices, LinearAlgebra, Statistics
using Test
using Aqua

@testset "Aqua" begin
    Aqua.test_all(TensorTrains, ambiguities=false)
    Aqua.test_ambiguities(TensorTrains)
end

include("svd_trunc.jl")
include("exact.jl")
include("tensor_train.jl")
include("periodic_tensor_train.jl")
include("uniform_tensor_train.jl")
include("mps.jl")


nothing

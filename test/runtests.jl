using TensorTrains
using TensorTrains.UniformTensorTrains
using TensorTrains.UniformTensorTrains: transfer_operator, leading_eig
using Random, Suppressor, InvertedIndices, LinearAlgebra
using Test
using Random, Suppressor, InvertedIndices
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


nothing
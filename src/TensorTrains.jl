module TensorTrains

using KrylovKit: eigsolve
using Lazy: @forward
using LinearAlgebra: LinearAlgebra, svd, norm, tr, I, dot, normalize!
using LogarithmicNumbers: Logarithmic
using MKL
using Random: AbstractRNG, default_rng
using StatsBase: StatsBase, sample!, sample
using TensorCast: @cast, TensorCast
using Tullio: @tullio


export 
    getindex, iterate, firstindex, lastindex, setindex!, eachindex, length, show,
    SVDTrunc, TruncBond, TruncThresh, TruncBondMax, TruncBondThresh, summary_compact,
    AbstractTensorTrain, TensorTrain, normalize_eachmatrix!, +, -, ==, isapprox, evaluate, 
    bond_dims, flat_tt, rand_tt, orthogonalize_right!, orthogonalize_left!, compress!,
    marginals, twovar_marginals, lognormalization, normalization,  
    dot, norm, norm2m,
    sample!, sample,
    AbstractPeriodicTensorTrain, PeriodicTensorTrain, flat_periodic_tt, rand_periodic_tt,

    # Uniform Tensor Trains
    AbstractUniformTensorTrain, UniformTensorTrain, InfiniteUniformTensorTrain,
    symmetrized_uniform_tensor_train, periodic_tensor_train,
    flat_infinite_uniform_tt, rand_infinite_uniform_tt


include("utils.jl")
include("svd_trunc.jl")
include("abstract_tensor_train.jl")
include("tensor_train.jl")
include("periodic_tensor_train.jl")

# Uniform Tensor Trains
include("UniformTensorTrains/uniform_tensor_train.jl")
include("UniformTensorTrains/transfer_operator.jl")

end # end module

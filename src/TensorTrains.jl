module TensorTrains

using Reexport
@reexport import Base:
    eltype, getindex, iterate, firstindex, lastindex, setindex!, eachindex, 
    length, isapprox, ==, +, -, show
using Lazy: @forward
using TensorCast: @cast, @reduce, TensorCast
using LinearAlgebra: svd, norm, tr, I, Hermitian
using LinearAlgebra
using Tullio: @tullio
using Random: AbstractRNG, GLOBAL_RNG
using StatsBase: sample!, sample
using StatsBase

export 
    SVDTrunc, TruncBond, TruncThresh, TruncBondMax, TruncBondThresh, summary_compact,
    AbstractTensorTrain, TensorTrain, normalize_eachmatrix!, +, -, ==, isapprox, evaluate, 
    bond_dims, uniform_tt, rand_tt, orthogonalize_right!, orthogonalize_left!, compress!,
    marginals, twovar_marginals, normalization, normalize!, dot, norm, norm2m,
    sample!, sample,
    AbstractPeriodicTensorTrain, PeriodicTensorTrain, uniform_periodic_tt, rand_periodic_tt,
    UniformTensorTrain, periodic_tensor_train, symmetrized_uniform_tensor_train

include("utils.jl")
include("svd_trunc.jl")
include("abstract_tensor_train.jl")
include("tensor_train.jl")
include("periodic_tensor_train.jl")
include("uniform_tensor_train.jl")


end # end module

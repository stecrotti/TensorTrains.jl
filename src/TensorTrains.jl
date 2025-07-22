module TensorTrains

using KrylovKit: eigsolve
using Lazy: @forward
using LinearAlgebra: LinearAlgebra, svd, norm, tr, I, dot, normalize!
using LogarithmicNumbers: Logarithmic
using MKL
using MPSKit: InfiniteMPS, DenseMPO, VUMPS, approximate, dot, add_util_leg, site_type, physicalspace
using Random: AbstractRNG, default_rng
using StatsBase: StatsBase, sample!, sample
using TensorCast: @cast, TensorCast
using TensorKit: TensorMap, ⊗, ℝ, id, storagetype
using Tullio: @tullio
using OffsetArrays

export 
    getindex, iterate, firstindex, lastindex, setindex!, eachindex, length, show,
    SVDTrunc, TruncBond, TruncThresh, TruncBondMax, TruncBondThresh, summary_compact,
    AbstractTensorTrain, TensorTrain, normalize_eachmatrix!, +, -, ==, isapprox, evaluate, 
    bond_dims, flat_tt, rand_tt, orthogonalize_right!, orthogonalize_left!, orthogonalize_center!,
    compress!,
    marginals, twovar_marginals, lognormalization, normalization,  
    dot, norm, norm2m,
    sample!, sample,
    AbstractPeriodicTensorTrain, PeriodicTensorTrain, flat_periodic_tt, rand_periodic_tt,
    is_left_canonical, is_right_canonical, is_canonical,
    grad,

    # Uniform Tensor Trains
    AbstractUniformTensorTrain, UniformTensorTrain, InfiniteUniformTensorTrain,
    symmetrized_uniform_tensor_train, periodic_tensor_train,
    flat_infinite_uniform_tt, rand_infinite_uniform_tt,
    TruncVUMPS


include("utils.jl")
include("svd_trunc.jl")
include("abstract_tensor_train.jl")
include("tensor_train.jl")
include("periodic_tensor_train.jl")

# Uniform Tensor Trains
include("UniformTensorTrains/uniform_tensor_train.jl")
include("UniformTensorTrains/transfer_operator.jl")
include("UniformTensorTrains/trunc_vumps.jl")

# Matrix Product States
include("MatrixProductStates/MatrixProductStates.jl")

end # end module

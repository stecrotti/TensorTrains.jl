module TensorTrains

using Reexport
@reexport import Base:
    eltype, getindex, iterate, firstindex, lastindex, setindex!, eachindex, 
    length, isapprox, ==, +, -, show
import Lazy: @forward
import TensorCast: @cast, @reduce, TensorCast
import LinearAlgebra: svd, normalize!, norm, tr, I, Hermitian
import Tullio: @tullio
import Random: AbstractRNG, GLOBAL_RNG
import StatsBase: sample!, sample

export 
    SVDTrunc, TruncBond, TruncThresh, TruncBondMax, TruncBondThresh, summary_compact,
    AbstractTensorTrain, TensorTrain, normalize_eachmatrix!, +, -, ==, isapprox, evaluate, 
    bond_dims, uniform_tt, rand_tt, orthogonalize_right!, orthogonalize_left!, compress!,
    marginals, twovar_marginals, normalization, normalize!, norm, trABt, norm2m,
    sample!, sample,
    PeriodicTensorTrain, uniform_periodic_tt, rand_periodic_tt

include("utils.jl")
include("svd_trunc.jl")
include("abstract_tensor_train.jl")
include("tensor_train.jl")
include("periodic_tensor_train.jl")


end # end module

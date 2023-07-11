module TensorTrains

using Reexport
@reexport import Base:
    eltype, getindex, iterate, firstindex, lastindex, setindex!, eachindex, 
    length, isapprox, +, -, show
import Lazy: @forward
import TensorCast: @cast, @reduce, TensorCast
import LinearAlgebra: svd, normalize!, norm
import Tullio: @tullio
import Random: AbstractRNG, GLOBAL_RNG
import StatsBase: sample!, sample

export 
    SVDTrunc, TruncBond, TruncThresh, TruncBondMax, TruncBondThresh, summary_compact,
    AbstractTensorTrain, TensorTrain, normalize_eachmatrix!, +, -, isapprox, evaluate, 
    bond_dims, uniform_tt, rand_tt, sweep_RtoL!, sweep_LtoR!, compress!,
    marginalize, marginals, marginals_tu, normalization, normalize!,
    sample!, sample

include("utils.jl")
include("svd_trunc.jl")
include("tensor_train.jl")


end # end module

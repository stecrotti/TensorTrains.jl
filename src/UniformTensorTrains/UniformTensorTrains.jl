module UniformTensorTrains

using ..TensorTrains
using ..TensorTrains: TensorTrains, AbstractPeriodicTensorTrain,
                    AbstractTensorTrain, PeriodicTensorTrain, TruncThresh,
                    flat_periodic_tt, normalization, rand_periodic_tt,
                    _reshape1

using LinearAlgebra: LinearAlgebra, dot, tr, I
using KrylovKit: eigsolve
using TensorCast: TensorCast, @cast
using Tullio: @tullio
using LogarithmicNumbers: Logarithmic

export AbstractPeriodicTensorTrain, PeriodicTensorTrain, flat_periodic_tt, rand_periodic_tt,
       AbstractUniformTensorTrain, UniformTensorTrain, periodic_tensor_train,
       symmetrized_uniform_tensor_train, InfiniteUniformTensorTrain,
       flat_infinite_uniform_tt, rand_infinite_uniform_tt,
       dot


include("uniform_tensor_train.jl")
include("transfer_operator.jl")


end # module
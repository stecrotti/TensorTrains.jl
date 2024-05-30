module UniformTensorTrains

using ..TensorTrains
using ..TensorTrains: _reshape1

using LinearAlgebra: LinearAlgebra, dot, tr
using KrylovKit: eigsolve
using TensorCast: TensorCast, @cast
using TensorTrains: TensorTrains, AbstractPeriodicTensorTrain,
                    AbstractTensorTrain, PeriodicTensorTrain, TruncThresh,
                    flat_periodic_tt, normalization, rand_periodic_tt
using Tullio: @tullio
using LogarithmicNumbers: Logarithmic

export AbstractPeriodicTensorTrain, PeriodicTensorTrain, flat_periodic_tt, rand_periodic_tt,
       AbstractUniformTensorTrain, UniformTensorTrain, periodic_tensor_train,
       symmetrized_uniform_tensor_train, InfiniteUniformTensorTrain, dot


include("uniform_tensor_train.jl")
include("transfer_operator.jl")


end # module